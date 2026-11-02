"""Scored-only re-initialisation of the water_pitcher procedural parameters (water_pitcher.py bas-relief model: side silhouette
S(y, z) + half-thickness field w(y, z) about the symmetry plane x = x0), analogue of sweeper_scored_init.py.

Deformation (canonical frame, differentiable on the procedural visual_4096 vertices):
  in-plane warp  (y, z) -> (y, z) + d(y, z), d bilinear on a G x G control grid over the silhouette bbox (+10 %)
  thickness      x -> x0 + (x - x0) * exp(t(y, z)), t bilinear on the same grid
Regularisers: first differences of d (/ 1 cm) and t (/ 0.15), magnitude |d| / 3 cm, |t| / 0.4; |d| <= 6 cm, |t| <= 0.7.
Joint fit with per-frame pose increments (scale fixed = retrack_v8 mesh_scale) on ep20 retrack_v8 frames (IoU >= 0.6),
loss = known-occluder soft IoU + w_dep x depth Huber + 0.5 x back-projected points (common.terms).
Held-out: alternating time blocks (12 frames; 1 of every 4 blocks held out); before/after = pose-only refit (60 it) with the
current vs the new shape, metrics = full-image known-occluder IoU + depth MAE (common.eval_full).
Mapped back to generator params by inverse-warping the S / w grids (fixed-point inverse of d), rebuilt with
water_pitcher.solid + decimate (geom venv) -> outputs/track3/retrack_v8/pitcher_scored_init/{visual,visual_4096}.obj + params.json.
Scored-split inputs only. Usage: python3 scripts/retrack_v8/pitcher_scored_init.py [--gpu 0] [--grid 6] [--iters 500]
"""
import argparse, json, os, sys, math, subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--gpu', default='0'); ap.add_argument('--iters', type=int, default=500)
    ap.add_argument('--grid', type=int, default=6); ap.add_argument('--w-dep', type=float, default=1.0); ap.add_argument('--out', default='pitcher_scored_init')
    ap.add_argument('--reg', type=float, default=0.002); ap.add_argument('--lr-d', type=float, default=0.002)
    a = ap.parse_args(); os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import numpy as np, torch
    import common as C
    obj = 'water_pitcher'; ep = 20; out = C.OUTR/a.out; out.mkdir(parents=True, exist_ok=True)
    C.LOGF = open(C.ROOT/f'logs/retrack_v8/{a.out}.log', 'a'); C.say('args', json.dumps(vars(a)))
    pg = C.T3/'procedural_geometry_v1'/obj; pp = json.loads((pg/'params.json').read_text())
    cx = C.Cx(); mesh = C.Mesh(cx, pg/'visual_4096.obj'); V0 = mesh.Vn.copy(); x0 = pp['x0']
    lo = V0[:, 1:].min(0); hi = V0[:, 1:].max(0); pad = 0.1*(hi - lo); glo = lo - pad; ghi = hi + pad; G = a.grid
    Vt = cx.T(V0)
    d = torch.zeros(G, G, 2, device=cx.dev, requires_grad=True); t = torch.zeros(G, G, device=cx.dev, requires_grad=True)

    def interp(field, yz):   # bilinear on the control grid; yz [N,2] (torch)
        u = ((yz - cx.T(glo))/cx.T(ghi - glo)).clamp(0, 1)*(G - 1)
        i0 = u.floor().clamp(0, G - 2).long(); f = u - i0
        a00 = field[i0[:, 0], i0[:, 1]]; a10 = field[i0[:, 0] + 1, i0[:, 1]]; a01 = field[i0[:, 0], i0[:, 1] + 1]; a11 = field[i0[:, 0] + 1, i0[:, 1] + 1]
        fy, fz = f[:, 0], f[:, 1]
        if field.dim() == 3: fy, fz = fy[:, None], fz[:, None]
        return a00*(1 - fy)*(1 - fz) + a10*fy*(1 - fz) + a01*(1 - fy)*fz + a11*fy*fz

    def deform(d, t, V=None):
        V = Vt if V is None else V
        yz = V[:, 1:]; dd = interp(d, yz); tt = interp(t, yz)
        return torch.stack([x0 + (V[:, 0] - x0)*torch.exp(tt), yz[:, 0] + dd[:, 0], yz[:, 1] + dd[:, 1]], 1)

    def reg(d, t):
        r = ((d[1:] - d[:-1])**2).sum()/0.01**2 + ((d[:, 1:] - d[:, :-1])**2).sum()/0.01**2
        r = r + ((t[1:] - t[:-1])**2).sum()/0.15**2 + ((t[:, 1:] - t[:, :-1])**2).sum()/0.15**2
        r = r + (d**2).sum()/0.03**2 + (t**2).sum()/0.4**2
        return a.reg*r

    z = np.load(C.OUTR/f'episode_{ep:06d}/{obj}_poses.npz', allow_pickle=True); s0 = float(z['mesh_scale'])
    e = C.Episode(cx, ep, obj)
    ok = [f for f in range(e.n) if z['iou'][f] >= 0.6 and e.area[f] >= 1500]
    pick = [ok[i] for i in np.linspace(0, len(ok) - 1, min(64, len(ok))).round().astype(int)]
    hold = [f for f in pick if (f//12) % 4 == 2]; fit = [f for f in pick if f not in hold]
    C.say(f'{len(fit)} fit frames, {len(hold)} held-out frames (blocks of 12, every 4th block); scale {s0:.4f}')
    P0 = {f: z['camera_T_object'][f] for f in pick}
    obs = {f: e.sample(f, e.crop_box(f, pad=1.35), 256) for f in pick}
    rng = np.random.default_rng(0); Fn = mesh.Fn
    A_ = np.linalg.norm(np.cross(V0[Fn[:, 1]] - V0[Fn[:, 0]], V0[Fn[:, 2]] - V0[Fn[:, 0]]), axis=1)
    fi = cx.T(Fn[rng.choice(len(Fn), 3000, p=A_/A_.sum())], torch.long); bw = cx.T(rng.dirichlet([1, 1, 1], 3000))
    cc = cx.T(mesh.cc)

    class Mm: pass

    def mk(V):
        m = Mm(); m.V = V - cc; m.F = mesh.F; m.S = (V[fi]*bw[..., None]).sum(1) - cc; return m

    def run(fr, shape_opt, iters, dd=None, tt=None):
        R0 = cx.T(np.stack([P0[f][:3, :3] for f in fr])); p0 = cx.T(np.stack([P0[f][:3, 3] + s0*P0[f][:3, :3] @ mesh.cc for f in fr]))
        n = len(fr); w = torch.zeros(n, 3, device=cx.dev, requires_grad=True); dp = torch.zeros(n, 3, device=cx.dev, requires_grad=True)
        groups = [dict(params=[w], lr=0.006), dict(params=[dp], lr=0.002)]
        if shape_opt: groups += [dict(params=[d], lr=a.lr_d), dict(params=[t], lr=0.01)]
        opt = torch.optim.Adam(groups); base = [g['lr'] for g in groups]
        dv, tv = (d, t) if shape_opt else (dd, tt)
        for it in range(iters):
            sc = 0.1 + 0.9*0.5*(1 + math.cos(math.pi*it/iters))
            for g, b_ in zip(opt.param_groups, base): g['lr'] = b_*sc
            idx = np.random.default_rng(it).choice(n, min(8, n), replace=False)
            m = mk(deform(dv, tv)); R = C.so3(torch, w[idx]) @ R0[idx]; p = p0[idx] + dp[idx]; loss = 0
            for k, j in enumerate(idx):
                tm = C.terms(cx, m, obs[fr[j]], (s0*R[k])[None], p[k][None], e.pts[fr[j]], soft=True)
                loss = loss + (tm['sil'] + a.w_dep*tm['dep'] + 0.5*tm['pts']).sum()
            loss = loss/len(idx)
            if shape_opt: loss = loss + reg(d, t)
            opt.zero_grad(); loss.backward(); opt.step()
            if shape_opt:
                with torch.no_grad(): d.clamp_(-0.06, 0.06); t.clamp_(-0.7, 0.7)
                if it % 100 == 0 or it == iters - 1:
                    C.say(f'it {it} loss {loss.item():.4f} |d| max {d.abs().max().item()*1000:.1f} mm t [{t.min().item():.2f}, {t.max().item():.2f}]')
        rows = []
        with torch.no_grad():
            V = deform(dv, tv); R = C.so3(torch, w) @ R0; p = p0 + dp
            class Mf: pass
            mf = Mf(); mf.V = V - cc; mf.F = mesh.F; mf.cc = mesh.cc; mf.S = None
            for k, f in enumerate(fr):
                T = C.camT_from(R[k].cpu().numpy(), p[k].cpu().numpy(), s0, mesh.cc)
                ev = C.eval_full(cx, mf, e, f, T, s0); rows.append((float(ev['iou_known'][0]), float(ev['dmae'][0]*1000), float(ev['iou'][0])))
        return np.array(rows)

    d0 = torch.zeros_like(d); t0 = torch.zeros_like(t)
    hb = run(hold, False, 60, d0, t0); fb = run(fit, False, 60, d0, t0)
    C.say(f'before: fit IoU {np.median(fb[:,0]):.3f} dep {np.median(fb[:,1]):.1f} mm | held-out IoU {np.median(hb[:,0]):.3f} dep {np.median(hb[:,1]):.1f} mm (recon-protocol {np.median(hb[:,2]):.3f})')
    fa = run(fit, True, a.iters)
    dv, tv = d.detach().clone(), t.detach().clone()
    ha = run(hold, False, 60, dv, tv)
    C.say(f'after: fit IoU {np.median(fa[:,0]):.3f} dep {np.median(fa[:,1]):.1f} mm | held-out IoU {np.median(ha[:,0]):.3f} dep {np.median(ha[:,1]):.1f} mm (recon-protocol {np.median(ha[:,2]):.3f})')
    res = dict(fit_frames=fit, holdout_frames=hold, before=dict(fit_iou=float(np.median(fb[:, 0])), fit_dep=float(np.median(fb[:, 1])),
               hold_iou=float(np.median(hb[:, 0])), hold_dep=float(np.median(hb[:, 1])), hold_iou_recon=float(np.median(hb[:, 2]))),
               after=dict(fit_iou=float(np.median(fa[:, 0])), fit_dep=float(np.median(fa[:, 1])), hold_iou=float(np.median(ha[:, 0])),
                          hold_dep=float(np.median(ha[:, 1])), hold_iou_recon=float(np.median(ha[:, 2]))),
               hold_rows_before=hb.tolist(), hold_rows_after=ha.tolist(), args=vars(a), mesh_scale=s0)
    # ---- map back to the generator grid (inverse warp of S and w)
    H, W = pp['grid_shape']; res_ = pp['res']; olo = np.array(pp['origin_yz'])
    sys.path.insert(0, str(C.ROOT/'scripts/procedural_geometry'))
    import importlib.util
    spec = importlib.util.spec_from_file_location('wp', C.ROOT/'scripts/procedural_geometry/water_pitcher.py')
    S = None
    src = (C.ROOT/'scripts/procedural_geometry/water_pitcher.py').read_text()
    # _unrle without importing the module (it imports trimesh/cv2 helpers only; safe)
    def unrle(r, shape):
        f = np.zeros(shape[0]*shape[1], bool)
        for a_, b_ in zip(r[::2], r[1::2]): f[a_:b_] = True
        return f.reshape(shape)
    S = unrle(pp['silhouette_rle'], (H, W)).astype(np.float32); wv = np.array(pp['w_mm'], np.float32)/1000
    dmax = float(dv.abs().max())
    nlo = olo - dmax - 0.01; nW = W + int(2*(dmax + 0.01)/res_) + 1; nH = H + int(2*(dmax + 0.01)/res_) + 1
    jj, ii = np.meshgrid(np.arange(nW), np.arange(nH))
    Yq = torch.as_tensor(np.stack([nlo[0] + jj.ravel()*res_, nlo[1] + ii.ravel()*res_], 1), dtype=torch.float32, device=cx.dev)
    with torch.no_grad():
        src_ = Yq.clone()
        for _ in range(30): src_ = Yq - interp(dv, src_)
        tq = interp(tv, src_).cpu().numpy(); src_ = src_.cpu().numpy()
    import cv2
    mx = ((src_[:, 0] - olo[0])/res_).astype(np.float32).reshape(nH, nW); my = ((src_[:, 1] - olo[1])/res_).astype(np.float32).reshape(nH, nW)
    S2 = cv2.remap(S, mx, my, cv2.INTER_LINEAR, borderValue=0) > 0.5
    w2 = cv2.remap(wv, mx, my, cv2.INTER_LINEAR, borderValue=0)*np.exp(tq.reshape(nH, nW))
    w2 = np.where(S2, np.maximum(w2, 0.002), 0)
    q = dict(pp); q.update(origin_yz=nlo.tolist(), grid_shape=[int(nH), int(nW)], w_mm=np.round(w2*1000, 2).tolist())
    f = S2.ravel().astype(np.int8); q['silhouette_rle'] = np.flatnonzero(np.diff(np.r_[0, f, 0])).tolist()
    q['provenance_scored_init'] = dict(method=__doc__.strip().splitlines()[0], grid=G, control_bbox_yz=[glo.tolist(), ghi.tolist()],
                                       d_m=dv.cpu().numpy().tolist(), t_log=tv.cpu().numpy().tolist(), base_params=str(pg/'params.json'),
                                       inputs='scored ep20 masks, FoundationStereo depth, retrack_v8 poses', **{k: v for k, v in res.items() if k not in ('hold_rows_before', 'hold_rows_after')})
    (out/'params.json').write_text(json.dumps(q)); (out/'fit_report.json').write_text(json.dumps(res, indent=1, default=float))
    np.save(out/'deformed_visual_4096_vertices.npy', deform(dv, tv).detach().cpu().numpy())
    code = f"""
import sys, json; sys.path.insert(0, '{C.ROOT}/scripts/procedural_geometry')
import water_pitcher as WP
from common import decimate
p = json.load(open('{out}/params.json'))
hi = decimate(WP.solid(p, 0.0035), 20000); lo = decimate(WP.solid(p, 0.0045), 3700)
hi.export('{out}/visual.obj'); lo.export('{out}/visual_4096.obj')
print('built', len(hi.faces), len(lo.faces), hi.is_watertight, lo.is_watertight, hi.extents, lo.extents)
"""
    r_ = subprocess.run([str(C.ROOT/'.deps/geom-venv/bin/python'), '-c', code], capture_output=True, text=True)
    C.say(r_.stdout.strip(), r_.stderr.strip()[-800:])


if __name__ == '__main__':
    main()
