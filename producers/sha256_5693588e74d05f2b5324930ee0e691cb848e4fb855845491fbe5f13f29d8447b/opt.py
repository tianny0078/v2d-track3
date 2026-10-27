"""Joint hand-object refinement of the source hands (hands_contact_v1).

Per hand, per frame: depth scale along the camera ray (lambda), small world translation (delta), rotation about the
wrist (rotvec r), per-finger curl about the MCP/CMC (theta, 5); per hand per episode: size factor kappa.
    w'  = c + lambda (w - c) + delta
    X'_j = w' + kappa R(r) curl_theta(X_j - w)
Energy (all robust, see notes/track3-hand-contact-v1.md §3):
    2D     : reprojection of X' vs the source 2D (projection of the source joints = WiLoR/DBT evidence), sigma 6 px
    depth  : X' camera z vs FoundationStereo depth at the joint pixel (+ skin offset), sigma 1.5 cm, gross > 15 cm dropped
    ground : hand capsule samples above the table plane (z_table >= radius)
    pen    : hand capsule samples outside every object (sd >= radius), visual-mesh SDF
    contact: frames where object b is moving (contact_plan core, dilated) and hand h is assigned to b from 2D mask
             proximity: thumb tip and soft-min(other tips) on the surface (sd - r_tip = 0)
    rigid  : assigned+moving frames: hand joints fixed in the object frame frame-to-frame (grasp rigidity)
    smooth : joint acceleration of X', first differences of the correction parameters
    prior  : |delta| 3 cm, |r| 0.3 rad, |theta| 0.5 rad, lambda/kappa bounded
MediaPipe 2D is NOT used here (held out for validation).
Writes outputs/track3/hands_contact_v1/<E>/hands_world.npz (same keys, joints_world/joints_camera_m replaced, + hc_*).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa
import evidence as EV

torch.set_default_dtype(torch.float64)
torch.set_num_threads(12)
FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
SKIN_OFF = np.array([.012, .010, .009, .008, .004, .010, .008, .007, .004, .010, .008, .007, .004,
                     .010, .008, .007, .004, .010, .008, .007, .004])
CFG = dict(sig_px=6., sig_depth=.015, w_depth=1., depth_gross=.15, w_2d=1., w_ground=1., sig_pen=.002, w_pen=5.,
           sig_contact=.004, w_contact=5., huber_contact=3., sig_rigid=.004, w_rigid=.5, sig_acc=.004, w_acc=1., w_dparam=1.,
           sig_delta=.03, sig_r=.3, sig_theta=.5, lam_lo=.7, lam_hi=1.15, kap_lo=.8, kap_hi=1.1, theta_lo=-.4, theta_hi=1.2,
           pre=4, post=2, assign_px=30., assign_margin_px=20., iters=1500, lr=.05, contact_on='human', robot_scale=1.2, robot_radius_scale=1.25, robot_palm_r=.024)


def rodrigues(rv):
    th = torch.linalg.norm(rv, dim=-1, keepdim=True).clamp_min(1e-9); k = rv/th
    K = torch.zeros(*rv.shape[:-1], 3, 3, dtype=rv.dtype)
    K[..., 0, 1] = -k[..., 2]; K[..., 0, 2] = k[..., 1]; K[..., 1, 0] = k[..., 2]
    K[..., 1, 2] = -k[..., 0]; K[..., 2, 0] = -k[..., 1]; K[..., 2, 1] = k[..., 0]
    s = torch.sin(th)[..., None]; c = torch.cos(th)[..., None]
    return torch.eye(3)+s*K+(1-c)*(K@K)


def huber(x, d=1.):
    a = x.abs(); return torch.where(a < d, .5*x*x, d*(a-.5*d))


class SDF:
    def __init__(self, grid):
        sd = torch.tensor(grid["sdf"], dtype=torch.float64); self.n = torch.tensor(sd.shape, dtype=torch.float64)
        self.t = sd.permute(2, 1, 0)[None, None].contiguous()  # (1,1,D=z,H=y,W=x)
        self.o = torch.tensor(grid['origin']); self.res = float(grid['res'])

    def __call__(self, X):  # X (...,3) mesh frame
        sh = X.shape; P = X.reshape(-1, 3)
        idx = (P-self.o)/self.res; cl = torch.minimum(torch.maximum(idx, torch.zeros(3)), self.n-1)
        g = 2*cl/(self.n-1)-1
        v = Fn.grid_sample(self.t, g[None, None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(-1)
        v = v+torch.linalg.norm((idx-cl)*self.res, dim=-1)
        return v.reshape(sh[:-1])


def segments(x):
    e = np.flatnonzero(np.diff(np.r_[0, x.astype(int), 0])); return list(zip(e[::2], e[1::2]))


def assignment(core, ev, cfg, T, B):
    """held weight (T,2,B): 1 on core frames, .5 on dilation, for hands whose fingertips/palm sit on the object mask in 2D."""
    held = np.zeros((T, 2, B)); info = []
    for b in range(B):
        for s, e in segments(core[:, b]):
            s0, e0 = max(0, s-cfg['pre']), min(T, e+cfg['post'])
            med = np.array([np.nanmedian(ev['mask_px'][s:e, h, b]) if np.isfinite(ev['mask_px'][s:e, h, b]).any() else np.inf for h in range(2)])
            ok = (med < cfg['assign_px']) & (med <= med.min()+cfg['assign_margin_px'])
            for h in np.flatnonzero(ok):
                held[s0:e0, h, b] = np.maximum(held[s0:e0, h, b], .5); held[s:e, h, b] = 1.
            info.append(dict(object=str(ev['names'][b]), start=int(s), end=int(e), mask_px_median=[round(float(m), 1) for m in med],
                             hands=[['left', 'right'][h] for h in np.flatnonzero(ok)]))
    return held, info


def run(ep, out_root='hands_contact_v1', cfg_over=None, src='learning_v5/hands', verbose=True):
    cfg = dict(CFG); cfg.update(cfg_over or {})
    t0 = time.time(); d = load_episode(ep, hands_root=src); ev = EV.build(ep); T = d['T']; B = len(d['names'])
    J0n = d['J']; valid = d['present'] & np.isfinite(J0n).all((-1, -2))
    J0 = torch.tensor(np.nan_to_num(J0n)); V = torch.tensor(valid, dtype=torch.float64)
    cam = torch.tensor(d['world_T_rgb'][:, :3, 3]); Rc = torch.tensor(d['world_T_rgb'][:, :3, :3]); K = d['K']
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    A = torch.tensor(d['A']); Pw = torch.tensor(d['P']); Pinv_R = Pw[..., :3, :3].transpose(-1, -2); Pt = Pw[..., :3, 3]
    sdfs = [SDF(sdf_grid(m)) for m in d['mesh']]
    core = moving_mask(d['A'] @ d['P']); held, info = assignment(core, ev, cfg, T, B)
    Hd = torch.tensor(held); coreT = torch.tensor(core, dtype=torch.float64)
    uv0 = torch.tensor(np.nan_to_num(ev['uv_src'])); uvw = torch.tensor(np.isfinite(ev['uv_src']).all(-1) & valid[..., None], dtype=torch.float64)
    dob = ev['d_obs']; dok = np.isfinite(dob) & (np.abs(dob-ev['z_src']) < cfg['depth_gross']) & valid[..., None]
    Dobs = torch.tensor(np.nan_to_num(dob)+SKIN_OFF); Dw = torch.tensor(dok, dtype=torch.float64)
    # palm normal (palm side) and curl axes from the source (fixed)
    sgn = torch.tensor([1., -1.])[None, :, None]
    n = torch.cross(J0[:, :, 9]-J0[:, :, 0], J0[:, :, 5]-J0[:, :, 17], dim=-1)*sgn
    n = n/torch.linalg.norm(n, dim=-1, keepdim=True).clamp_min(1e-9)
    axes = []
    for f in FINGERS:
        a = torch.cross(J0[:, :, f[1]]-J0[:, :, f[0]], n, dim=-1); axes.append(a/torch.linalg.norm(a, dim=-1, keepdim=True).clamp_min(1e-9))
    axes = torch.stack(axes, 2)  # (T,2,5,3)
    rel0 = J0-J0[:, :, :1]
    # parameters
    p_lam = torch.zeros(T, 2, requires_grad=True); p_del = torch.zeros(T, 2, 3, requires_grad=True)
    p_r = torch.zeros(T, 2, 3, requires_grad=True); p_th = torch.zeros(T, 2, 5, requires_grad=True); p_kap = torch.zeros(2, requires_grad=True)

    def bounded(p, lo, hi, init=1.):  # p=0 -> init
        x0 = np.log((init-lo)/(hi-init)); return lo+(hi-lo)*torch.sigmoid(p+x0)

    def forward():
        # parameter units: lambda .02/unit (bounded), delta 1 cm/unit, r .05 rad/unit, theta .1 rad/unit (bounded)
        lam = bounded(.4*p_lam, cfg['lam_lo'], cfg['lam_hi']); kap = bounded(.4*p_kap, cfg['kap_lo'], cfg['kap_hi'])
        th = bounded(.1*p_th/.4, cfg['theta_lo'], cfg['theta_hi'], 0.)
        rel = rel0.clone()
        for i, f in enumerate(FINGERS):
            Rf = rodrigues(axes[:, :, i]*th[:, :, i:i+1])  # (T,2,3,3)
            piv = rel0[:, :, f[0]:f[0]+1]
            rel[:, :, f[1:]] = piv+torch.einsum('thij,thkj->thki', Rf, rel0[:, :, f[1:]]-piv)
        w = J0[:, :, 0]; wp = cam[:, None]+lam[..., None]*(w-cam[:, None])+.01*p_del
        X = wp[:, :, None]+kap[None, :, None, None]*torch.einsum('thij,thkj->thki', rodrigues(.05*p_r), rel)
        return X, lam, kap, th

    def samples(X):
        pts = [X]; rad = [torch.tensor(RADIUS)]
        for j in range(1, 21):
            p = PARENT[j]
            for a in (.25, .5, .75):
                pts.append(((1-a)*X[:, :, p]+a*X[:, :, j])[:, :, None]); rad.append(torch.tensor([(1-a)*RADIUS[p]+a*RADIUS[j]]))
        for m in (5, 9, 13, 17):
            for a in (.33, .66):
                pts.append(((1-a)*X[:, :, 0]+a*X[:, :, m])[:, :, None]); rad.append(torch.tensor([.015]))
        return torch.cat(pts, 2), torch.cat(rad)

    def to_obj(X, b):  # world (T,2,N,3) -> object b frame
        return torch.einsum('tij,thnj->thni', Pinv_R[:, b], X-Pt[:, b, None, None])

    def energy(detail=False):
        X, lam, kap, th = forward(); E = {}
        Xc = torch.einsum('tji,thnj->thni', Rc, X-cam[:, None, None])
        z = Xc[..., 2].clamp_min(.05); uv = torch.stack([Xc[..., 0]/z*fx+cx, Xc[..., 1]/z*fy+cy], -1)
        E['2d'] = cfg['w_2d']*(huber(torch.linalg.norm(uv-uv0, dim=-1)/cfg['sig_px'])*uvw).sum()
        E['depth'] = cfg['w_depth']*(huber((Xc[..., 2]-Dobs)/cfg['sig_depth'])*Dw).sum()
        robot = cfg['contact_on'] == 'robot'
        # 'robot': geometry terms on the IK targets the retarget will track (wrist + 1.2 (X - wrist), official
        # morphology scale of initialize_sharpa_retarget) with radii scaled by robot_radius_scale; 2D/depth stay human.
        Xg = X[:, :, :1]+cfg['robot_scale']*(X-X[:, :, :1]) if robot else X
        S, rad = samples(Xg); Vs = V[:, :, None]
        if robot: rad = torch.where(rad >= .015, torch.full_like(rad, cfg['robot_palm_r']), rad*cfg['robot_radius_scale'])  # Sharpa palm half-thickness 24 mm
        tip_r = TIP_R*cfg['robot_radius_scale'] if robot else TIP_R
        zt = torch.einsum('ij,thnj->thni', A[:3, :3], S)[..., 2]+A[2, 3]
        E['ground'] = cfg['w_ground']*((Fn.relu(rad-zt)/cfg['sig_pen'])**2*Vs).sum()
        Ep = 0.; Ec = 0.; Er = 0.
        for b, sdf in enumerate(sdfs):
            sd = sdf(to_obj(S, b)); Ep = Ep+((Fn.relu(rad-sd)/cfg['sig_pen'])**2*Vs).sum()
            hb = Hd[:, :, b]*V
            if hb.sum() > 0:
                # finger contact = soft-min skin distance over the distal phalanx (tip + 3 samples towards the DIP/IP),
                # so a finger lying along the surface counts as touching (no tip-vs-bone conflict)
                dist = torch.stack([Xg[:, :, TIPS], *[(1-a)*Xg[:, :, TIPS]+a*Xg[:, :, [t-1 for t in TIPS]] for a in (.25, .5, .75)]], 3)
                rd = torch.tensor([TIP_R, *[(1-a)*TIP_R+a*RADIUS[3] for a in (.25, .5, .75)]])*(cfg['robot_radius_scale'] if robot else 1.)
                sdt = -.002*torch.logsumexp(-(sdf(to_obj(dist.reshape(T, 2, -1, 3), b)).reshape(T, 2, 5, 4)-rd)/.002, -1)  # (T,2,5)
                fing = -.004*torch.logsumexp(-sdt[..., 1:]/.004, -1)
                Ec = Ec+((huber(sdt[..., 0]/cfg['sig_contact'], cfg['huber_contact'])+huber(fing/cfg['sig_contact'], cfg['huber_contact']))*hb).sum()
                Xo = to_obj(X, b); hr = (coreT[1:, b]*coreT[:-1, b])[:, None]*(Hd[1:, :, b] > .99)*(Hd[:-1, :, b] > .99)*V[1:]*V[:-1]
                Er = Er+((torch.linalg.norm(Xo[1:]-Xo[:-1], dim=-1)/cfg['sig_rigid'])**2).sum(-1).mul(hr).sum()
        E['pen'] = cfg['w_pen']*Ep; E['contact'] = cfg['w_contact']*Ec; E['rigid'] = cfg['w_rigid']*Er
        v3 = V[2:]*V[1:-1]*V[:-2]
        # acceleration: in the world frame, but while hand h holds object b measured in b's frame (the hand may follow
        # the reference object's frame-to-frame pose jitter, which keeps contact without penetration)
        accn = lambda Y: ((torch.linalg.norm(Y[2:]-2*Y[1:-1]+Y[:-2], dim=-1)/cfg['sig_acc'])**2).sum(-1)  # noqa: E731
        Hm = Hd[1:-1]*V[1:-1, :, None]; hsum = Hm.sum(-1).clamp(max=1.)
        Eacc = (1-hsum)*accn(X)
        for b in range(len(sdfs)):
            if Hm[..., b].sum() > 0: Eacc = Eacc+Hm[..., b]/Hm.sum(-1).clamp(min=1.)*accn(to_obj(X, b))
        E['acc'] = cfg['w_acc']*(Eacc*v3).sum()
        v2 = (V[1:]*V[:-1])
        dl = (lam[1:]-lam[:-1])/.01; dd = torch.linalg.norm(p_del[1:]-p_del[:-1], dim=-1)/1.
        dr = torch.linalg.norm(p_r[1:]-p_r[:-1], dim=-1)/1.; dt = torch.linalg.norm(th[1:]-th[:-1], dim=-1)/.1
        E['dparam'] = cfg['w_dparam']*((dl**2+dd**2+dr**2+dt**2)*v2).sum()
        E['prior'] = ((torch.linalg.norm(.01*p_del, dim=-1)/cfg['sig_delta'])**2*V).sum()+((torch.linalg.norm(.05*p_r, dim=-1)/cfg['sig_r'])**2*V).sum() \
            + ((th/cfg['sig_theta'])**2).sum(-1).mul(V).sum()
        tot = sum(E.values())
        return (tot, {k: float(v) for k, v in E.items()}, X.detach()) if detail else tot

    params = [p_lam, p_del, p_r, p_th, p_kap]
    e0 = energy(True)[1]
    opt = torch.optim.Adam(params, lr=cfg['lr'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['iters'])
    for it in range(cfg['iters']):
        opt.zero_grad(); L = energy(); L.backward()
        opt.step(); sched.step()
        if verbose and it % 250 == 0: print(f'  it {it} E={float(L):.1f}', flush=True)
    tot, e1, X = energy(True)
    with torch.no_grad():
        Xf, lam, kap, th = forward()
    Xn = Xf.detach().numpy().copy(); Xn[~valid] = np.nan
    # outputs
    h = dict(d['h']); h['joints_world'] = Xn
    Rcn = d['world_T_rgb'][:, :3, :3]; cn = d['world_T_rgb'][:, :3, 3]
    h['joints_camera_m'] = np.einsum('fji,fhnj->fhni', Rcn, Xn-cn[:, None, None])
    h.update(hc_source_joints_world=J0n, hc_lambda=lam.detach().numpy(), hc_delta=.01*p_del.detach().numpy(), hc_rotvec=.05*p_r.detach().numpy(),
             hc_curl=th.detach().numpy(), hc_kappa=kap.detach().numpy(), hc_held=held, hc_object_names=np.array(d['names']),
             hc_config=json.dumps(cfg))
    o = T3/out_root/d['E']; o.mkdir(parents=True, exist_ok=True); np.savez_compressed(o/'hands_world.npz', **h)
    rep = dict(episode=ep, source=str(T3/src/d['E']/'hands_world.npz'), objects=d['names'], config=cfg, energy_before=e0, energy_after=e1,
               held_segments=info, kappa=kap.detach().numpy().round(3).tolist(),
               lambda_p10_50_90=[np.round(np.quantile(lam.detach().numpy()[valid[:, i], i], [.1, .5, .9]), 3).tolist() for i in range(2)],
               delta_cm_p50_p95=np.round(np.quantile(np.linalg.norm(.01*p_del.detach().numpy()[valid], axis=-1), [.5, .95])*100, 2).tolist(),
               seconds=round(time.time()-t0, 1))
    save_json(o/'opt_report.json', rep)
    if verbose: print(json.dumps({k: rep[k] for k in ['energy_before', 'energy_after', 'kappa', 'lambda_p10_50_90', 'delta_cm_p50_p95', 'seconds']}), flush=True)
    return rep


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('episodes', type=int, nargs='+'); ap.add_argument('--out', default='hands_contact_v1')
    ap.add_argument('--set', nargs='*', default=[], help='cfg overrides key=value'); a = ap.parse_args()
    over = {}
    for kv in a.set:
        k, v = kv.split('='); over[k] = type(CFG[k])(v) if not isinstance(CFG[k], str) else v
    for e in a.episodes:
        print('episode', e, flush=True); run(e, a.out, over)
