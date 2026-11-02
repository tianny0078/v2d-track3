"""Shared GPU render-and-compare machinery for retrack_v8 (global pose+scale init and full-sequence re-tracking).

Conventions (copied from scripts/render_pipeline_steps.py / scripts/multiep_recon/recon.py):
  camera_T_object = inv(world_T_rgb[f]) @ world_T_canonical[f]; world_T_rgb = stereo_sequence_orb_objmask/camera_poses.npz;
  pixel = K @ X_cam (calibrated_sequence/rgb_K.npy, undistorted 1024x767 frames); masks object_masks_v3 (SAM2.1);
  depth foundation_stereo_v1/depth_rgb (m, rgb camera), invalid / <0.1 / >3 m -> 0.
Mesh model: X_cam = s * R @ (V_canonical - cc) + p  (s = isotropic scale applied to the procedural canonical mesh about the
canonical ORIGIN; in camera_T_object form: camT[:3,:3] = R, camT[:3,3] = p - s R cc, mesh vertices = s * V_canonical).
Rendering is done in square-ish crops around the object (intrinsics shifted/scaled), observations are sampled into the same
crop with nearest-neighbour lookups; pixels outside the image are don't-care.
Scored-split data only.
"""
import math, os, sys, time, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, cv2

ROOT = Path(__file__).resolve().parents[2]
T3 = ROOT/'outputs/track3'
OUTR = T3/'retrack_v8'
sys.path.insert(0, str(ROOT/'scripts/multiep_recon'))
import recon as RC  # noqa: E402  (data loaders only: episode_meta, load_frame, hand_masks, backproject)

H, W = 767, 1024
LOGF = None


def say(*x):
    s = time.strftime('%H:%M:%S', time.gmtime()) + 'Z ' + ' '.join(map(str, x)); print(s, flush=True)
    if LOGF: LOGF.write(s + '\n'); LOGF.flush()


def so3(torch, w):
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-8)   # clamped as in recon.so3 (keeps the gradient at w = 0)
    k = w/th
    Kx = torch.zeros(*w.shape[:-1], 3, 3, device=w.device, dtype=w.dtype)
    Kx[..., 0, 1], Kx[..., 0, 2], Kx[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    Kx[..., 1, 2], Kx[..., 2, 0], Kx[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    th = th[..., None]
    return torch.eye(3, device=w.device, dtype=w.dtype) + torch.sin(th)*Kx + (1 - torch.cos(th))*Kx@Kx


def rot_angle_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1)/2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def random_rotations(n, seed=0):
    from scipy.spatial.transform import Rotation
    return Rotation.random(n, random_state=seed).as_matrix()


class Mesh:
    def __init__(self, cx, path, n_surf=2500):
        import trimesh
        m = trimesh.load(path, process=False, force='mesh')
        self.path = str(path); self.Vn = np.asarray(m.vertices, np.float64); self.Fn = np.asarray(m.faces, np.int64)
        self.cc = (self.Vn.min(0) + self.Vn.max(0))/2
        self.radius = float(np.linalg.norm(self.Vn - self.cc, axis=1).max())
        S = trimesh.sample.sample_surface(m, n_surf, seed=0)[0]
        self.V = cx.T(self.Vn - self.cc); self.F = cx.T(self.Fn, cx.torch.int32); self.S = cx.T(S - self.cc)


class Cx:
    def __init__(self):
        import torch, nvdiffrast.torch as dr
        self.torch, self.dr = torch, dr; self.dev = 'cuda'; self.glctx = dr.RasterizeCudaContext()

    def T(self, x, dtype=None):
        return self.torch.as_tensor(np.asarray(x), dtype=dtype or self.torch.float32, device=self.dev)

    def render(self, mesh, A, t, Kc, res, soft=False):
        """A [B,3,3] (= s R), t [B,3] (camera-frame position of the mesh-centre-shifted origin), Kc [B,4] crop intrinsics,
        res (rw, rh) multiples of 8. Returns hard bool [B,rh,rw], zr [B,rh,rw] (0 where empty), soft or None."""
        torch, dr = self.torch, self.dr; rw, rh = res
        Xc = torch.einsum('bij,vj->bvi', A, mesh.V) + t[:, None]
        z = Xc[..., 2:3].clamp_min(1e-4); near, far = 0.02, 5.0
        fx, fy, cx, cy = [Kc[:, i, None, None] for i in range(4)]
        u = fx*Xc[..., 0:1]/z + cx; v = fy*Xc[..., 1:2]/z + cy
        clip = torch.cat([(2*u/rw - 1)*z, (2*v/rh - 1)*z, ((far + near)/(far - near))*z - 2*far*near/(far - near), z], -1).contiguous()
        rast, _ = dr.rasterize(self.glctx, clip, mesh.F, (rh, rw))
        hard = rast[..., 3] > 0
        zr = dr.interpolate(Xc[..., 2:3].contiguous(), rast, mesh.F)[0][..., 0]
        sf = dr.antialias((rast[..., 3:4] > 0).float(), rast, clip, mesh.F)[..., 0] if soft else None
        return hard, zr, sf


class Episode:
    """GPU-resident observations of one scored episode for one object."""
    def __init__(self, cx, ep, obj):
        torch = cx.torch; self.cx = cx; self.ep = ep; self.obj = obj
        m = RC.episode_meta(ep, obj); self.meta = m; n = m['n']; self.n = n; self.E = m['E']
        self.K = m['K']; self.Kv = np.array([m['K'][0, 0], m['K'][1, 1], m['K'][0, 2], m['K'][1, 2]], np.float32)
        self.wTc = m['wTc']; self.camT_v7 = m['camT0']; self.scale_v7 = m['scale']
        hands = RC.hand_masks(m['E'], n)
        self.obs = torch.zeros(n, 5, H, W, dtype=torch.float16, device=cx.dev)  # M, O, Hand, Merode, D
        self.area = np.zeros(n, int); self.bbox = np.zeros((n, 4), int); self.cen = np.zeros((n, 2)); self.dmed = np.zeros(n)
        self.sharp = np.full(n, np.nan); self.hand_near = np.zeros(n, int); self.other_near = np.zeros(n, int)
        self.border = np.zeros(n, bool); self.pts = [None]*n
        def ld(f):
            M, O, D, rgb = RC.load_frame(m['E'], f, obj)
            Mer = cv2.erode(M.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3).astype(bool)
            info = {}
            if M.sum() > 0:
                ys, xs = np.nonzero(M); info['bbox'] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1); info['cen'] = (xs.mean(), ys.mean())
                dd = D[Mer & (D > 0)]; info['dmed'] = float(np.median(dd)) if len(dd) > 20 else 0.0
                g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY); x0, y0, x1, y1 = info['bbox']
                info['sharp'] = float(cv2.Laplacian(g[y0:y1, x0:x1], cv2.CV_64F)[M[y0:y1, x0:x1]].var())
                Md = cv2.dilate(M.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)   # ~15 px ring
                info['hand_near'] = int((Md & hands[f]).sum()) if hands is not None else 0
                info['other_near'] = int((Md & O & ~M).sum())
                info['border'] = bool(x0 < 6 or y0 < 6 or x1 > W - 6 or y1 > H - 6)
            pts = RC.backproject(M, D, m['K'], max_pts=800, seed=f)
            Hm = hands[f] if hands is not None else np.zeros((H, W), bool)
            return f, np.stack([M, O, Hm, Mer, D]).astype(np.float16), info, pts
        with ThreadPoolExecutor(24) as ex:
            for f, arr, info, pts in ex.map(ld, range(n)):
                self.obs[f] = torch.as_tensor(arr, device=cx.dev); self.area[f] = int((arr[0] > 0.5).sum())
                self.pts[f] = cx.T(pts)
                if info:
                    self.bbox[f] = info['bbox']; self.cen[f] = info['cen']; self.dmed[f] = info['dmed']; self.sharp[f] = info['sharp']
                    self.hand_near[f] = info['hand_near']; self.other_near[f] = info['other_near']; self.border[f] = info['border']
        say(f'ep{ep} {obj}: {n} frames loaded, mask area median {np.median(self.area[self.area > 0]):.0f}, '
            f'absent {(self.area == 0).sum()}, GPU {torch.cuda.memory_allocated()/1e9:.2f} GB')

    def crop_box(self, f, extra_box=None, pad=1.45, min_side=96):
        """Square box (x0, y0, side) around the mask bbox (union extra_box), side multiple of anything; may exceed the image."""
        boxes = []
        if self.area[f] > 0: boxes.append(self.bbox[f])
        if extra_box is not None: boxes.append(extra_box)
        if not boxes: return np.array([0., 0., float(W)])
        b = np.array(boxes, float); x0, y0 = b[:, 0].min(), b[:, 1].min(); x1, y1 = b[:, 2].max(), b[:, 3].max()
        side = max(x1 - x0, y1 - y0)*pad + 24; side = max(side, min_side)
        return np.array([(x0 + x1)/2 - side/2, (y0 + y1)/2 - side/2, side])

    def sample(self, f, box, res):
        """Observations of frame f in crop box (x0, y0, side) at res (r, r): dict of [1, r, r] tensors + crop intrinsics [1,4]."""
        torch = self.cx.torch; r = res
        x0, y0, side = box; sc = r/side
        i = torch.arange(r, device=self.cx.dev, dtype=torch.float32)
        u = x0 + (i + 0.5)/sc - 0.5; v = y0 + (i + 0.5)/sc - 0.5
        gx = (u + 0.5)/W*2 - 1; gy = (v + 0.5)/H*2 - 1
        grid = torch.stack(torch.meshgrid(gy, gx, indexing='ij')[::-1], -1)[None]
        o = torch.nn.functional.grid_sample(self.obs[f:f+1].float(), grid, mode='nearest', padding_mode='zeros', align_corners=False)[0]
        inside = ((u >= -0.5) & (u < W - 0.5))[None, :] & ((v >= -0.5) & (v < H - 0.5))[:, None]
        Kc = self.cx.T([[self.Kv[0]*sc, self.Kv[1]*sc, (self.Kv[2] - x0)*sc, (self.Kv[3] - y0)*sc]])
        return dict(M=o[0] > 0.5, O=o[1] > 0.5, Hd=o[2] > 0.5, Mer=o[3] > 0.5, D=o[4], inside=inside, Kc=Kc, res=(r, r), box=box)


def pose_to_At(torch, R, p, s, cc_unused=None):
    return s[:, None, None]*R, p


def terms(cx, mesh, ob, A, t, pts=None, soft=True, tau=0.015, hand_dc=True, unknown_dc=True, sigma_pts=0.01, dc_mode='known'):
    """Per-hypothesis loss terms for B poses on ONE frame's observation dict ob (expanded over B).
    dc_mode 'known' (default for fitting): don't care = known occluders only (hand-landmark hulls outside the object mask,
    other objects' masks) + outside the image. 'recon': multiep_recon eval protocol (anything in front by depth, other object
    in front, unknown depth). 'recon' lets a too-large/too-far object hide behind its own support table at zero cost, so it
    is only used for reporting."""
    torch = cx.torch; B = len(A)
    hard, zr, sf = cx.render(mesh, A, t, ob['Kc'].expand(B, 4), ob['res'], soft=soft)
    M = ob['M'][None]; O = ob['O'][None]; Hd = ob['Hd'][None]; D = ob['D'][None]; valid = D > 0; ins = ob['inside'][None]
    zrd = zr.detach()
    if dc_mode == 'recon':
        dc = (hard & ~M & valid & (D < zrd - tau)) | (O & valid & hard & (D < zrd + tau)) | ~ins
        if unknown_dc: dc = dc | (hard & ~M & ~valid)
    else:
        dc = (O & ~M) | ~ins
    if hand_dc: dc = dc | (Hd & ~M)
    care = (~dc).float(); Mf = M.float()
    s_ = sf if soft else hard.float()
    inter = (s_*Mf*care).flatten(1).sum(1); union = ((s_ + Mf - s_*Mf)*care).flatten(1).sum(1)
    sil = 1 - inter/union.clamp_min(1)
    P = ob['Mer'][None] & hard & valid
    r = (zr - D).abs(); hub = torch.where(r < 0.01, 0.5*r**2/0.01, r - 0.005).clamp(max=0.045)
    npx = P.flatten(1).sum(1)
    dep = (hub*P).flatten(1).sum(1)/npx.clamp_min(1)/0.01
    dmae = ((r.clamp(max=0.05))*P).flatten(1).sum(1)/npx.clamp_min(1)
    hi = (hard & M & care.bool()).flatten(1).sum(1).float(); hu = ((hard | M) & care.bool()).flatten(1).sum(1).float()
    out = dict(sil=sil, dep=dep, iou=hi/hu.clamp_min(1), dmae=torch.where(npx > 20, dmae, torch.full_like(dmae, 0.05)),
               rpx=hard.flatten(1).sum(1))
    if pts is not None and len(pts) >= 20:
        # canonical-frame (centre-shifted) coordinates of the observed points: A^-1 (x - t)
        q = torch.einsum('bij,bpj->bpi', torch.linalg.inv(A), pts[None] - t[:, None])
        s = torch.linalg.det(A).abs()**(1/3)
        d = torch.cdist(q, mesh.S[None].expand(B, -1, -1)).min(-1).values*s[:, None]
        out['pts'] = (d**2/(d**2 + sigma_pts**2)).mean(1)
    else:
        out['pts'] = torch.zeros(B, device=cx.dev)
    return out


def score_of(tm):
    return tm['iou'] - 4*tm['dmae'].clamp(max=0.05)


def refine(cx, mesh, ob, R0, p0, s0, pts=None, iters=80, opt_scale=True, lr=(0.03, 0.006, 0.01), tether=None,
           w=(1.0, 1.0, 0.5), sigma_pts=0.01, hand_dc=True, unknown_dc=True):
    """Batched Adam refinement of B hypotheses on one frame. R0 [B,3,3], p0 [B,3], s0 [B] torch.
    tether: (R_ref [B,3,3], p_ref [B,3], rot_sigma_rad, trans_sigma_m) or None. Returns R, p, s, terms (no grad)."""
    torch = cx.torch; B = len(R0)
    wv = torch.zeros(B, 3, device=cx.dev, requires_grad=True); dp = torch.zeros(B, 3, device=cx.dev, requires_grad=True)
    ls = torch.zeros(B, device=cx.dev, requires_grad=True)
    groups = [dict(params=[wv], lr=lr[0]), dict(params=[dp], lr=lr[1])]
    if opt_scale: groups.append(dict(params=[ls], lr=lr[2]))
    opt = torch.optim.Adam(groups); base = [g['lr'] for g in opt.param_groups]
    def cur():
        R = so3(torch, wv) @ R0; p = p0 + dp; s = s0*torch.exp(ls); return R, p, s
    for it in range(iters):
        sc = 0.15 + 0.85*0.5*(1 + math.cos(math.pi*it/max(iters, 1)))
        for g, b in zip(opt.param_groups, base): g['lr'] = b*sc
        R, p, s = cur(); A = s[:, None, None]*R
        tm = terms(cx, mesh, ob, A, p, pts, soft=True, sigma_pts=sigma_pts, hand_dc=hand_dc, unknown_dc=unknown_dc)
        loss = w[0]*tm['sil'] + w[1]*tm['dep'] + w[2]*tm['pts']
        if tether is not None:
            Rr, pr, rs, ts = tether
            c = ((torch.einsum('bij,bij->b', R, Rr) - 1)/2).clamp(-1, 1)
            loss = loss + (1 - c)/(1 - math.cos(rs)) + ((p - pr)**2).sum(-1)/ts**2
        opt.zero_grad(); loss.sum().backward(); opt.step()
    with torch.no_grad():
        R, p, s = cur(); A = s[:, None, None]*R
        tm = terms(cx, mesh, ob, A, p, pts, soft=False, sigma_pts=sigma_pts)
    return R.detach(), p.detach(), s.detach(), {k: v.detach() for k, v in tm.items()}


def camT_from(R, p, s, cc):
    """numpy: camera_T_object for mesh vertices = s * V_canonical (canonical origin), from X = s R (V - cc) + p."""
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p - s*R @ cc; return T


def Rp_from_camT(T, s, cc):
    R = T[:3, :3]; return R, T[:3, 3] + s*R @ cc


def eval_full(cx, mesh, epi, f, camT, s, res=None):
    """Full-image hard metrics (as multiep_recon eval_frames): occlusion-aware IoU, plain IoU, depth MAE (m), render px."""
    torch = cx.torch
    ob = epi.sample(f, np.array([0., -128., 1024.]), 1024)   # 1024x1024 crop = whole image (rows -128..895), outside = don't care
    T = np.asarray(camT)[None] if np.ndim(camT) == 2 else np.asarray(camT)
    R = cx.T(T[:, :3, :3]); tt = cx.T(T[:, :3, 3]); cc = cx.T(mesh.cc)
    A = float(s)*R; p = tt + torch.einsum('bij,j->bi', A, cc)
    with torch.no_grad():
        tm = terms(cx, mesh, ob, A, p, None, soft=False, hand_dc=False, unknown_dc=True, dc_mode='recon')
        tk = terms(cx, mesh, ob, A, p, None, soft=False, hand_dc=True, dc_mode='known')
        hard, zr, _ = cx.render(mesh, A, p, ob['Kc'].expand(len(A), 4), ob['res'])
        M = ob['M'][None]; ins = ob['inside'][None]
        plain = ((hard & M & ins).flatten(1).sum(1).float()/((hard | M) & ins).flatten(1).sum(1).float().clamp_min(1))
        zs = torch.where(ob['Mer'][None] & hard & (ob['D'][None] > 0), zr - ob['D'][None], torch.full_like(zr, float('nan')))
        sres = torch.nanmedian(zs.flatten(1), dim=1).values
    return dict(iou=tm['iou'].cpu().numpy(), iou_known=tk['iou'].cpu().numpy(), iou_plain=plain.cpu().numpy(), dmae=tm['dmae'].cpu().numpy(),
                rpx=tm['rpx'].cpu().numpy(), depth_signed_median=sres.cpu().numpy())


def overlay_tile(cx, mesh, epi, f, camT, s, size=256, label='', box=None, col=(255, 255, 0)):
    """Crop of RGB frame f around the mask: mask contour red, mesh render (camera_T_object camT, mesh = s x canonical)
    translucent + outline cyan. Returns uint8 BGR [size,size,3]."""
    torch = cx.torch
    rgb = cv2.imread(str(epi.E/f'calibrated_sequence/frames/{f:06d}.jpg'))
    M = epi.obs[f, 0].cpu().numpy() > 0.5
    if box is None: box = epi.crop_box(f, pad=1.3, min_side=160)
    x0, y0, side = box
    R = cx.T(np.asarray(camT)[None, :3, :3]); tt = cx.T(np.asarray(camT)[None, :3, 3]); cc = cx.T(mesh.cc)
    A = float(s)*R; p = tt + torch.einsum('bij,j->bi', A, cc)
    with torch.no_grad():
        hard, _, _ = cx.render(mesh, A, p, cx.T(epi.Kv[None]), (W, 768))
    Rh = hard[0, :H].cpu().numpy()
    im = rgb.copy(); im[Rh] = (0.6*im[Rh] + 0.4*np.array(col)).astype(np.uint8)
    cs, _ = cv2.findContours(M.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(im, cs, -1, (0, 0, 255), 2)
    cr, _ = cv2.findContours(Rh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(im, cr, -1, col, 2)
    pad = int(side) + 2
    big = cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
    xi, yi, si = int(round(x0)) + pad, int(round(y0)) + pad, int(round(side))
    t = cv2.resize(big[yi:yi+si, xi:xi+si], (size, size), interpolation=cv2.INTER_AREA)
    if label:
        cv2.putText(t, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3); cv2.putText(t, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return t


def global_search(cx, mesh, e, f, Rs, s_fixed=None, s_lo=0.6, s_hi=1.5, topk=24, iters=120, res=256, n_keep=6, score_fn=None):
    """Multi-hypothesis render-and-compare on frame f (see global_init.py docstring). Rs [N,3,3] rotation samples.
    Returns up to n_keep distinct hypotheses (>10 deg apart) sorted by full-image score: dicts R, camT, s, iou, iou_plain, dmae_mm, score."""
    torch = cx.torch
    box = e.crop_box(f)
    ob = e.sample(f, box, res); pts = e.pts[f]
    K = e.K; um, vm = e.cen[f]; Dm = float(e.dmed[f]); Am = float(e.area[f])
    if Dm <= 0 or Am < 50: return []
    s0 = s_fixed if s_fixed is not None else 1.0
    ray = np.linalg.inv(K) @ np.array([um, vm, 1.0]); p0 = ray*Dm
    cc = mesh.cc; sc_px = res/box[2]; Kinv = cx.T(np.linalg.inv(K))
    hyps = []
    with torch.no_grad():
        ii = torch.arange(res, device=cx.dev, dtype=torch.float32)
        for b0 in range(0, len(Rs), 512):
            R = cx.T(Rs[b0:b0+512]); B = len(R)
            hard, zr, _ = cx.render(mesh, s0*R, cx.T(p0)[None].expand(B, 3), ob['Kc'].expand(B, 4), ob['res'])
            npx = hard.flatten(1).sum(1).float(); hf = hard.float()
            cu = (hf*ii[None, None, :]).flatten(1).sum(1)/npx.clamp_min(1); cv = (hf*ii[None, :, None]).flatten(1).sum(1)/npx.clamp_min(1)
            zm = (zr*hard).flatten(1).sum(1)/npx.clamp_min(1)
            Ar = npx/sc_px**2
            cu = box[0] + (cu + 0.5)/sc_px - 0.5; cv = box[1] + (cv + 0.5)/sc_px - 0.5
            k = torch.sqrt(Am/Ar.clamp_min(1)); delta = zm - Dm
            if s_fixed is None:
                z = Dm/(1 + delta*k/Dm); s = (s0*k*z/Dm).clamp(s_lo, s_hi); z = Dm - delta*s/s0
            else:
                s = torch.full_like(k, s0); z = Dm - delta
            okb = (npx > 30) & (z > 0.1)
            fac = (s/s0)*(Dm/z.clamp_min(0.05))
            tu = um - (cu - um)*fac; tv = vm - (cv - vm)*fac
            p = torch.stack([tu, tv, torch.ones_like(tu)], 1) @ Kinv.T*z[:, None]
            tm = terms(cx, mesh, ob, s[:, None, None]*R, p, None, soft=False, hand_dc=True, unknown_dc=False)
            scv = score_of(tm); scv[~okb] = -9
            scv = scv.cpu().numpy(); sn = s.cpu().numpy(); pn = p.cpu().numpy()
            for j in range(B): hyps.append((float(scv[j]), b0 + j, float(sn[j]), pn[j]))
    hyps.sort(key=lambda h: -h[0])
    keep = []
    for h in hyps:
        if h[0] < -1: break
        if all(rot_angle_deg(Rs[h[1]], Rs[k[1]]) > 15 for k in keep): keep.append(h)
        if len(keep) >= topk: break
    if not keep: return []
    R0 = cx.T(np.stack([Rs[h[1]] for h in keep])); p0t = cx.T(np.stack([h[3] for h in keep])); s0t = cx.T([h[2] for h in keep])
    R1, p1, s1, _ = refine(cx, mesh, ob, R0, p0t, s0t, pts, iters=iters//2, opt_scale=s_fixed is None, sigma_pts=0.03, unknown_dc=False)
    ob2 = e.sample(f, box, 320)
    R2, p2, s2, tm = refine(cx, mesh, ob2, R1, p1, s1, pts, iters=iters//2, opt_scale=s_fixed is None, sigma_pts=0.01, lr=(0.01, 0.002, 0.004))
    sc = score_of(tm).cpu().numpy(); order = np.argsort(-sc)
    R2n, p2n, s2n = R2.cpu().numpy(), p2.cpu().numpy(), s2.cpu().numpy()
    out = []
    for j in order:
        if any(rot_angle_deg(R2n[j], o['R']) < 10 for o in out): continue
        T = camT_from(R2n[j], p2n[j], s2n[j], cc)
        ev = eval_full(cx, mesh, e, f, T, s2n[j])
        out.append(dict(R=R2n[j], camT=T, s=float(s2n[j]), score=float(ev['iou_known'][0] - 4*min(ev['dmae'][0], 0.05)),
                        iou=float(ev['iou_known'][0]), iou_recon=float(ev['iou'][0]), iou_plain=float(ev['iou_plain'][0]),
                        dmae_mm=float(ev['dmae'][0]*1000), depth_signed_mm=float(ev['depth_signed_median'][0]*1000), pts=float(tm['pts'][j])))
        if len(out) >= n_keep: break
    out.sort(key=lambda o: -o['score'])
    return out
