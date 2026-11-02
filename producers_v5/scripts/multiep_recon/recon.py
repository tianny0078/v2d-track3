"""Multi-episode joint object reconstruction from keyframes (Track3, scored split only).

One object, ALL scored episodes containing it (learning_v7_geom objects.json). Shared canonical shape + per-vertex
albedo; per-episode exposure (gain+bias RGB); per-keyframe camera_T_object refinement deltas (weak tether).
Stages (each cached on disk -> resumable/idempotent):
  init      : shared init mesh = procedural_geometry_v1 mesh (canonical frame of every v7 pose) x geometric mean of
              the per-episode iso scales, isotropically remeshed (project remesher, remesh.py); albedo = nearest-
              neighbour transfer of SAM3D raw GLB vertex colours (raw x scale_m_per_unit = metric canonical frame).
  keyframes : per-episode keyframe selector (visible fraction, truncation, hand / other-object proximity and
              depth-in-front occlusion, mask/render agreement, mask temporal jump, sharpness, greedy viewpoint
              diversity in the object frame). Hold-out = clean frames not selected and >2 frames from any keyframe.
  param     : structural-prior refit = surface-of-revolution profile deformation around the procedural axis
              (radial scale at K heights, bottom/rim height, inner-wall offset) + poses/exposure/albedo.
  free      : free-form vertex optimisation (gradient diffusion preconditioning, per-vertex isotropic Adam,
              Laplacian-detail + normal-consistency regularisers) with periodic isotropic remeshing (split/collapse/
              flip/relax; genus/watertightness checked, previous mesh kept if a remesh fails).
  eval      : held-out frames (and all frames for reference) before vs after; before = submitted per-episode
              mesh (procedural x s_e) with the v7 poses. Protocols: (a) initial poses, (b) pose-only refit
              (shape/albedo/exposure frozen), identical for before and after.
  export    : meshes, poses, renders, overlays, report.json.
Losses: occlusion-aware soft IoU (SAM2.1 masks v3), robust colour (Charbonnier) in eroded visible mask,
FoundationStereo depth Huber (truncated), back-projected depth points to surface (Geman-McClure).
Inputs: outputs/track3/episode_*/{calibrated_sequence,foundation_stereo_v1,object_masks_v3,stereo_sequence_orb_objmask,
hand_detector_mediapipe_sequence}, learning_v7_geom objects/poses. No public-split data is read.
"""
import argparse, json, math, os, sys, time, hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, cv2, trimesh

ROOT = Path(__file__).resolve().parents[2]
T3 = ROOT/'outputs/track3'
H, W, HP = 767, 1024, 768
sys.path.insert(0, str(Path(__file__).resolve().parent))
import remesh as RM  # noqa: E402

LOG = None


def say(*x):
    s = time.strftime('%H:%M:%S ') + ' '.join(map(str, x)); print(s, flush=True)
    if LOG: LOG.write(s+'\n'); LOG.flush()


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def dir_digest(paths):
    """sha256 over the sorted list of (name, sha256) of files -> one digest for a directory of frames."""
    h = hashlib.sha256()
    for p in sorted(paths): h.update(Path(p).name.encode()); h.update(sha256(p).encode())
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------- data
def find_episodes(obj):
    eps = []
    for p in sorted((T3/'learning_v7_geom').glob('episode_*/objects/objects.json')):
        d = json.loads(p.read_text())
        if obj in d['object_names']: eps.append(int(d['episode_id']))
    return eps


def alt_init(ep, obj, pattern, proc_mesh):
    """Alternative initial-pose source (e.g. retrack_v8): npz with world_T_canonical [n,4,4], mesh_scale and mesh_path whose
    canonical frame is the procedural one. Writes (once) the procedural visual(.obj/_4096.obj) scaled by mesh_scale next to the
    npz so that 'submitted per-episode mesh = scale x procedural' holds for the init/eval code. Returns (wTo, scale, mesh_path,
    mesh_path_4096, poses_path, observed)."""
    pz = Path(pattern.format(ep=ep, obj=obj)); z = np.load(pz, allow_pickle=True)
    s = float(z['mesh_scale']); wTo = z['world_T_canonical']; assert (z['frame_index'] == np.arange(len(wTo))).all()
    src = Path(str(z['mesh_path'])).resolve(); P = Path(proc_mesh).resolve()
    assert src.parent == P.parent, f'init poses are for {src}, not the procedural canonical frame of {P}'
    paths = []
    for nm in ('visual.obj', 'visual_4096.obj'):
        q = pz.parent/f'{obj}_{nm[:-4]}_x{s:.4f}.obj'
        if not q.exists():
            m = trimesh.load(P.parent/nm, process=False, force='mesh'); m.vertices = m.vertices*s; m.export(q)
        paths.append(str(q))
    obs = z['measured'] if 'measured' in z.files else np.ones(len(wTo), bool)
    return wTo, s, paths[0], paths[1], str(pz), obs


INIT_POSES = None   # set by main() from --init-poses (format string with {ep} and {obj}); None = learning_v7_geom poses
INIT_PROC = None    # procedural mesh (visual.obj) whose canonical frame the alternative poses use (= --proc-mesh)


def episode_meta(ep, obj):
    E = T3/f'episode_{ep:06d}'
    oj = T3/f'learning_v7_geom/episode_{ep:06d}/objects/objects.json'
    d = json.loads(oj.read_text()); o = [x for x in d['objects'] if x['name'] == obj][0]
    z = np.load(d['poses_path']); names = [str(x) for x in z['object_names']]; slot = names.index(obj)
    wTo = z['world_T_canonical'][:, slot]; assert (z['frame_index'] == np.arange(len(wTo))).all()
    if INIT_POSES is not None:
        wTo, s_alt, mp, mp4, pp, obs = alt_init(ep, obj, INIT_POSES, INIT_PROC or T3/f'procedural_geometry_v1/{obj}/visual.obj')
        o = dict(o, uniform_scale_applied=s_alt, mesh_path=mp, mesh_path_4096=mp4); d = dict(d, poses_path=pp)
        z = dict(observed=obs[:, None]); slot = 0
    cp = E/'stereo_sequence_orb_objmask/camera_poses.npz'; P = np.load(cp)
    assert (P['frame_index'] == np.arange(len(P['frame_index']))).all()
    wTc = P['world_T_rgb']
    frames = sorted((E/'calibrated_sequence/frames').glob('*.jpg')); n = len(frames)
    assert len(wTo) == n == len(wTc), (len(wTo), n, len(wTc))
    K = np.load(E/'calibrated_sequence/rgb_K.npy')
    camT0 = np.linalg.inv(wTc) @ wTo
    return dict(ep=ep, E=E, n=n, K=K, wTc=wTc, camT0=camT0, scale=float(o.get('uniform_scale_applied', 1.0)),
                mesh_path=o['mesh_path'], mesh_path_4096=o['mesh_path_4096'], objects_json=str(oj), poses_path=d['poses_path'],
                camera_poses=str(cp), observed=z['observed'][:, slot])


def hand_masks(E, n):
    p = E/'hand_detector_mediapipe_sequence/visible_candidates.npz'
    if not p.exists(): return None
    z = np.load(p); out = []
    for f in range(n):
        m = np.zeros((H, W), np.uint8)
        for h in range(z['xy_pixels'].shape[1]):
            if not z['candidate_present'][f, h]: continue
            xy = z['xy_pixels'][f, h][z['valid'][f, h]]
            if len(xy) < 3: continue
            cv2.fillConvexPoly(m, cv2.convexHull(np.round(xy).astype(np.int32)), 1)
        if m.any(): m = cv2.dilate(m, np.ones((17, 17), np.uint8))  # hull of landmarks + ~8 px finger margin
        out.append(m.astype(bool))
    return out


def load_frame(E, f, obj):
    z = np.load(E/f'object_masks_v3/masks/{f:06d}.npz'); nm = [str(x) for x in z['object_names']]; mk = z['masks']
    M = mk[nm.index(obj)].astype(bool) if obj in nm else np.zeros((H, W), bool)
    O = np.zeros((H, W), bool)
    for j in range(len(nm)):
        if nm[j] != obj: O |= mk[j].astype(bool)
    d = np.load(E/f'foundation_stereo_v1/depth_rgb/{f:06d}.npz')['depth'].astype(np.float32)
    d[~np.isfinite(d) | (d < 0.1) | (d > 3.0)] = 0
    rgb = cv2.imread(str(E/f'calibrated_sequence/frames/{f:06d}.jpg'))[:, :, ::-1].copy()
    return M, O, d, rgb


def backproject(M, D, K, max_pts=1500, seed=0):
    er = cv2.erode(M.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3).astype(bool) & (D > 0)
    v, u = np.nonzero(er)
    if len(v) > max_pts:
        s = np.random.default_rng(seed).choice(len(v), max_pts, replace=False); v, u = v[s], u[s]
    z = D[v, u]; fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.stack([(u-cx)*z/fx, (v-cy)*z/fy, z], 1).astype(np.float32)


# ----------------------------------------------------------------------------------------------- torch helpers
class Ctx:
    def __init__(self):
        import torch, nvdiffrast.torch as dr
        self.torch, self.dr = torch, dr
        self.dev = 'cuda'; self.glctx = dr.RasterizeCudaContext()

    def T(self, x, dtype=None):
        torch = self.torch
        return torch.as_tensor(np.asarray(x), dtype=dtype or torch.float32, device=self.dev)

    def render(self, V, F, Rn, tn, Kb, attr=None, want_soft=True, attr_geom_grad=True):
        """V [Nv,3] canonical, F int32 [Nf,3], Rn [B,3,3], tn [B,3], Kb [B,4] fx fy cx cy. Returns soft, zr, hard, attr."""
        torch, dr = self.torch, self.dr
        Xc = torch.einsum('bij,vj->bvi', Rn, V) + tn[:, None]
        z = Xc[..., 2:3].clamp_min(1e-4); near, far = 0.05, 5.0
        fx, fy, cx, cy = [Kb[:, i, None, None] for i in range(4)]
        u = fx*Xc[..., 0:1]/z + cx; v = fy*Xc[..., 1:2]/z + cy
        clip = torch.cat([(2*u/W-1)*z, (2*v/HP-1)*z, ((far+near)/(far-near))*z-2*far*near/(far-near), z], -1).contiguous()
        rast, _ = dr.rasterize(self.glctx, clip, F, (HP, W))
        hard = (rast[..., 3] > 0)[:, :H]
        soft = dr.antialias((rast[..., 3:4] > 0).float(), rast, clip, F)[..., 0][:, :H] if want_soft else hard.float()
        zr = dr.interpolate(Xc[..., 2:3].contiguous(), rast, F)[0][..., 0][:, :H]
        col = dr.interpolate(attr[None].contiguous(), rast if attr_geom_grad else rast.detach(), F)[0][:, :H] if attr is not None else None
        return soft, zr, hard, col


def so3(torch, w):
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-8); k = w/th
    Kx = torch.zeros(*w.shape[:-1], 3, 3, device=w.device)
    Kx[..., 0, 1], Kx[..., 0, 2], Kx[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    Kx[..., 1, 2], Kx[..., 2, 0], Kx[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    th = th[..., None]
    return torch.eye(3, device=w.device) + torch.sin(th)*Kx + (1-torch.cos(th))*Kx@Kx


class Poses:
    """Left increments in the camera frame, rotation about the initial object centre."""
    def __init__(self, cx, camT0, centre):
        torch = cx.torch; self.cx = cx
        self.Ti = cx.T(camT0); c = cx.T(centre)
        self.cen0 = (self.Ti[:, :3, :3] @ c + self.Ti[:, :3, 3])
        n = len(camT0)
        self.rot = torch.zeros(n, 3, device=cx.dev, requires_grad=True)
        self.tra = torch.zeros(n, 3, device=cx.dev, requires_grad=True)

    def get(self, idx):
        torch = self.cx.torch
        Rd = so3(torch, self.rot[idx]); Ti = self.Ti[idx]; c = self.cen0[idx]
        Rn = Rd @ Ti[:, :3, :3]; tn = (Rd @ (Ti[:, :3, 3]-c)[..., None])[..., 0] + c + self.tra[idx]
        return Rn, tn

    def gauge(self):
        """Mean object-frame pose increment over all frames -> 0 (shape must not drift in the canonical frame)."""
        torch = self.cx.torch
        Rn, tn = self.get(torch.arange(len(self.Ti), device=self.cx.dev))
        RiT = self.Ti[:, :3, :3].transpose(1, 2)
        Rd = RiT @ Rn; td = (RiT @ (tn - self.Ti[:, :3, 3])[..., None])[..., 0]
        w = torch.stack([Rd[:, 2, 1]-Rd[:, 1, 2], Rd[:, 0, 2]-Rd[:, 2, 0], Rd[:, 1, 0]-Rd[:, 0, 1]], -1)/2
        return (w.mean(0)**2).sum()/math.radians(0.5)**2 + (td.mean(0)**2).sum()/0.0005**2

    def tether(self):
        return ((self.rot**2).sum(-1)/math.radians(5)**2 + (self.tra**2).sum(-1)/0.02**2).mean()

    def numpy(self):
        torch = self.cx.torch
        with torch.no_grad():
            Rn, tn = self.get(torch.arange(len(self.Ti), device=self.cx.dev))
        out = np.tile(np.eye(4), (len(Rn), 1, 1)); out[:, :3, :3] = Rn.cpu().numpy(); out[:, :3, 3] = tn.cpu().numpy()
        return out


class FrameSet:
    """GPU-resident per-frame data for a list of (episode index, frame) items."""
    def __init__(self, cx, items, metas, obj, hands):
        torch = cx.torch; self.items = items; n = len(items)
        self.M = torch.zeros(n, H, W, dtype=torch.bool, device=cx.dev); self.O = torch.zeros_like(self.M)
        self.Mer = torch.zeros_like(self.M); self.Hm = torch.zeros_like(self.M)
        self.D = torch.zeros(n, H, W, dtype=torch.float16, device=cx.dev)
        self.RGB = torch.zeros(n, H, W, 3, dtype=torch.uint8, device=cx.dev)
        self.K = torch.zeros(n, 4, device=cx.dev); self.epi = torch.zeros(n, dtype=torch.long, device=cx.dev)
        self.pts = []
        def ld(i):
            ei, f = items[i]; m = metas[ei]; M, O, D, rgb = load_frame(m['E'], f, obj)
            return i, M, O, D, rgb, backproject(M, D, m['K'], seed=f)
        with ThreadPoolExecutor(16) as ex:
            for i, M, O, D, rgb, p in ex.map(ld, range(n)):
                ei, f = items[i]; K = metas[ei]['K']
                self.M[i] = cx.T(M, torch.bool); self.O[i] = cx.T(O, torch.bool); self.D[i] = cx.T(D, torch.float16)
                self.Mer[i] = cx.T(cv2.erode(M.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3).astype(bool), torch.bool)
                if hands[ei] is not None: self.Hm[i] = cx.T(hands[ei][f], torch.bool)
                self.RGB[i] = cx.T(rgb, torch.uint8); self.K[i] = cx.T([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
                self.epi[i] = ei; self.pts.append(cx.T(p))


def frame_terms(cx, fs, idx, V, F, albedo, gain, bias, Rn, tn, surf, a, want_pts=True):
    """Per-frame loss terms for frames idx of FrameSet fs. Returns dict of [B] tensors."""
    torch = cx.torch
    soft, zr, hard, col = cx.render(V, F, Rn, tn, fs.K[idx], albedo, attr_geom_grad=bool(a.col_geom))
    Mb = fs.M[idx]; Ob = fs.O[idx]; Db = fs.D[idx].float(); valid = Db > 0; zrd = zr.detach(); Mf = Mb.float()
    occl = hard & ~Mb & valid & (Db < zrd - a.tau_occ)
    unknown = hard & ~Mb & ~valid
    other_front = Ob & valid & hard & (Db < zrd + a.tau_occ)
    w = (~(occl | unknown | other_front)).float()
    inter = (soft*Mf*w).flatten(1).sum(1); union = ((soft + Mf - soft*Mf)*w).flatten(1).sum(1)
    l_sil = 1 - inter/union.clamp_min(1)
    Mer = fs.Mer[idx]
    P = Mer & hard & valid
    r = (zr - Db).abs(); hub = torch.where(r < 0.01, 0.5*r**2/0.01, r - 0.005).clamp(max=0.045)
    l_dep = (hub*P).flatten(1).sum(1)/P.flatten(1).sum(1).clamp_min(1)/0.01
    Pc = Mer & hard & (~valid | ((zrd - Db).abs() < 0.02))
    g = gain[fs.epi[idx]][:, None, None, :]; bb = bias[fs.epi[idx]][:, None, None, :]
    pred = col*g + bb; obs = fs.RGB[idx].float()/255
    rr = pred - obs; ch = (torch.sqrt(rr**2 + 0.01**2) - 0.01).sum(-1).clamp(max=0.6)
    l_col = (ch*Pc).flatten(1).sum(1)/Pc.flatten(1).sum(1).clamp_min(1)/0.05
    out = dict(sil=l_sil, dep=l_dep, col=l_col)
    if want_pts:
        lp = []
        for k, i in enumerate(idx.tolist()):
            p = fs.pts[i]
            if len(p) < 20: lp.append(torch.zeros((), device=cx.dev)); continue
            pl = (p - tn[k]) @ Rn[k]
            d2 = torch.cdist(pl, surf).min(1).values**2
            lp.append((d2/(d2 + 0.01**2)).mean())
        out['pts'] = torch.stack(lp)
    return out


def sample_surface(torch, V, F, n, gen=None):
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    ar = torch.cross(v1-v0, v2-v0, dim=-1).norm(dim=-1)
    fi = torch.multinomial(ar.detach() + 1e-12, n, replacement=True, generator=gen)
    r1 = torch.rand(n, 1, device=V.device, generator=gen); r2 = torch.rand(n, 1, device=V.device, generator=gen); s = r1.sqrt()
    return (1-s)*v0[fi] + s*(1-r2)*v1[fi] + s*r2*v2[fi]


def eval_frames(cx, fs, V, F, albedo, gain, bias, Rn_all, tn_all, a, idx_list=None):
    """Hard-render metrics per frame: occlusion-aware IoU, colour MAE (0-255), depth MAE (mm, |r| truncated 5 cm)."""
    torch = cx.torch; rows = []
    idx_all = list(range(len(fs.items))) if idx_list is None else idx_list
    with torch.no_grad():
        for s in range(0, len(idx_all), 12):
            idx = torch.as_tensor(idx_all[s:s+12], device=cx.dev)
            _, zr, hard, col = cx.render(V, F, Rn_all[idx], tn_all[idx], fs.K[idx], albedo, want_soft=False)
            Mb = fs.M[idx]; Ob = fs.O[idx]; Db = fs.D[idx].float(); valid = Db > 0
            dc = (hard & ~Mb & valid & (Db < zr - a.tau_occ)) | (hard & ~Mb & ~valid) | (Ob & valid & hard & (Db < zr + a.tau_occ))
            care = ~dc
            inter = (hard & Mb & care).flatten(1).sum(1).float(); union = ((hard | Mb) & care).flatten(1).sum(1).float()
            P = fs.Mer[idx] & hard & valid
            dmae = ((zr - Db).abs().clamp(max=0.05)*P).flatten(1).sum(1)/P.flatten(1).sum(1).clamp_min(1)
            Pc = fs.Mer[idx] & hard & (~valid | ((zr - Db).abs() < 0.02))
            pred = (col*gain[fs.epi[idx]][:, None, None, :] + bias[fs.epi[idx]][:, None, None, :]).clamp(0, 1)
            cm = ((pred - fs.RGB[idx].float()/255).abs().mean(-1)*Pc).flatten(1).sum(1)/Pc.flatten(1).sum(1).clamp_min(1)*255
            for k, i in enumerate(idx.tolist()):
                rows.append(dict(item=list(map(int, fs.items[i])), iou=float(inter[k]/union[k].clamp_min(1)),
                                 color_mae=float(cm[k]) if Pc[k].any() else None,
                                 depth_mae_mm=float(dmae[k]*1000) if P[k].any() else None))
    return rows


def summarise(rows):
    def st(key):
        x = np.array([r[key] for r in rows if r[key] is not None], float)
        return dict(median=float(np.median(x)), p10=float(np.percentile(x, 10)), mean=float(x.mean()), n=int(len(x))) if len(x) else None
    return dict(iou=st('iou'), color_mae=st('color_mae'), depth_mae_mm=st('depth_mae_mm'))


# ----------------------------------------------------------------------------------------------- mesh helpers
def mesh_ok(V, F):
    m = trimesh.Trimesh(V, F, process=False)
    return bool(m.is_watertight and m.is_winding_consistent), m


def build_adjacency(torch, F, nv, dev):
    E, _, _ = RM.edge_maps(F)
    i = np.concatenate([E[:, 0], E[:, 1]]); j = np.concatenate([E[:, 1], E[:, 0]])
    deg = np.bincount(i, minlength=nv).astype(np.float32)
    vals = 1.0/deg[i]
    A = torch.sparse_coo_tensor(torch.as_tensor(np.stack([i, j])), torch.as_tensor(vals), (nv, nv)).coalesce().to(dev)
    return A, torch.as_tensor(E, device=dev)


def edge_face_pairs(F):
    E, fe, ef = RM.edge_maps(F)
    return ef


def vertex_normals_t(torch, V, F):
    fn = torch.cross(V[F[:, 1]]-V[F[:, 0]], V[F[:, 2]]-V[F[:, 0]], dim=-1)
    vn = torch.zeros_like(V).index_add_(0, F[:, 0], fn).index_add_(0, F[:, 1], fn).index_add_(0, F[:, 2], fn)
    return vn/vn.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def rms_radius(V, F):
    """Area-weighted surface RMS radius about the area centroid (as the CD-O normalisation), via 200k area-uniform samples."""
    pts = trimesh.sample.sample_surface(trimesh.Trimesh(V, F, process=False), 200000, seed=0)[0]
    cen = pts.mean(0)
    return float(np.sqrt((np.linalg.norm(pts-cen, axis=1)**2).mean())), cen


# ----------------------------------------------------------------------------------------------- stages
def stage_init(a, out, metas):
    f = out/'init.npz'
    if f.exists(): return dict(np.load(f))
    proc = Path(a.proc_mesh); P = trimesh.load(proc, process=False, force='mesh')
    scales = np.array([m['scale'] for m in metas]); sbar = float(np.exp(np.log(scales).mean()))
    # verify the per-episode mesh == scale x procedural (about the canonical origin)
    for m in metas:
        q = trimesh.load(m['mesh_path'], process=False, force='mesh')
        err = np.abs(q.vertices - m['scale']*P.vertices).max() if len(q.vertices) == len(P.vertices) else np.inf
        say(f'ep{m["ep"]} mesh = {m["scale"]:.4f} x procedural : max vertex err {err*1000:.4f} mm')
        if a.raw_glb is not None: assert err < 1e-5
    if a.edge_mm <= 0:
        a.edge_mm = float(np.sqrt(P.area*sbar**2/(a.target_faces*0.433))*1000); say(f'auto edge length {a.edge_mm:.2f} mm')
    t0 = time.time()
    V, F, _ = RM.remesh(P.vertices, P.faces, None, a.edge_mm/1000/sbar, rounds=3, relax_lam=0.0)
    ok, mm = mesh_ok(V, F); assert ok and mm.euler_number == P.euler_number, 'init remesh broke topology'
    from scipy.spatial import cKDTree
    d = cKDTree(trimesh.sample.sample_surface(P, 400000, seed=1)[0]).query(V)[0]
    say(f'init remesh {len(P.faces)}F -> {len(F)}F in {time.time()-t0:.0f}s; dev from procedural max {d.max()*1000:.3f} mm, '
        f'volume ratio {mm.volume/P.volume:.4f}')
    # colours: SAM3D raw GLB vertices x scale_m_per_unit are the metric canonical frame (fallback: constant grey)
    alb = np.full((len(V), 3), 0.5); align = np.array([np.nan]*4)
    if a.raw_glb is not None:
        from scipy.spatial import cKDTree
        raw = trimesh.load(a.raw_glb, force='mesh', process=False)
        reg = np.load(a.metric_registration) if a.metric_registration and Path(a.metric_registration).exists() else None
        s_raw = float(reg['scale_m_per_unit']) if reg is not None else 1.0
        Xm = raw.vertices*s_raw
        if getattr(raw.visual, 'kind', None) == 'vertex':
            colors = raw.visual.vertex_colors[:, :3].astype(np.float64)/255
            tree = cKDTree(Xm); S = trimesh.sample.sample_surface(P, 20000, seed=0)[0]; dd, _ = tree.query(S)
            align = np.array([np.median(dd), np.percentile(dd, 90), np.nan, np.nan])
            say(f'colour transfer alignment: procedural->SAM3D NN residual median {np.median(dd)*1000:.2f} mm p90 {np.percentile(dd,90)*1000:.2f} mm')
            if np.median(dd) < 0.003:
                alb = colors[tree.query(V)[1]]
            else: say('colour transfer rejected (residual > 3 mm): constant grey albedo')
        else: say('raw GLB has no vertex colours: constant grey albedo')
    else: say('no SAM3D colour source allowed: constant grey albedo')
    res = dict(V=V*sbar, F=F, albedo=alb, sbar=sbar, scales=scales, Vp=V, align=align, edge_mm=a.edge_mm)
    np.savez(f, **res); return res


def render_init_stats(cx, metas, obj, hands, P, a):
    """Per-frame selector statistics with the submitted per-episode mesh at the initial pose."""
    torch = cx.torch; stats = []
    Vt = cx.T(P.vertices); Ft = cx.T(P.faces, torch.int32)
    for ei, m in enumerate(metas):
        n = m['n']; K = m['K']; Kb = cx.T([[K[0, 0], K[1, 1], K[0, 2], K[1, 2]]])
        def ld(f): return f, load_frame(m['E'], f, obj)
        with ThreadPoolExecutor(16) as ex:
            for f, (M, O, D, rgb) in ex.map(ld, range(n)):
                T0 = m['camT0'][f]
                with torch.no_grad():
                    _, zr, hard, _ = cx.render(Vt*m['scale'], Ft, cx.T(T0[None, :3, :3]), cx.T(T0[None, :3, 3]), Kb, want_soft=False)
                Rh = hard[0].cpu().numpy(); z = zr[0].cpu().numpy()
                aM, aR = int(M.sum()), int(Rh.sum())
                st = dict(ep=m['ep'], frame=f, mask_px=aM, render_px=aR)
                if aM > 0 and aR > 0:
                    valid = D > 0
                    st['iou_init'] = float((M & Rh).sum()/(M | Rh).sum())
                    st['vis_frac'] = float(aM/aR)
                    ys, xs = np.nonzero(M | Rh); bd = 4
                    st['truncated'] = bool(ys.min() < bd or xs.min() < bd or ys.max() >= H-bd or xs.max() >= W-bd)
                    st['occl_frac'] = float((Rh & ~M & valid & (D < z - a.tau_occ)).sum()/aR)
                    # other object IN FRONT of the rendered object (depth test), and hand-landmark hull over the render
                    st['other_frac'] = float((Rh & O & valid & (D < z - a.tau_occ)).sum()/aR)
                    st['hand_frac'] = float((Rh & hands[ei][f]).sum()/aR) if hands[ei] is not None else 0.0
                    ys, xs = np.nonzero(M); y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
                    g = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
                    st['sharp'] = float(cv2.Laplacian(g, cv2.CV_64F)[M[y0:y1, x0:x1]].var())
                    st['view_dir'] = (-(T0[:3, :3].T @ T0[:3, 3])/np.linalg.norm(T0[:3, 3])).tolist()
                stats.append(st)
        say(f'selector stats ep{m["ep"]}: {n} frames')
    return stats


def select_keyframes(stats, a):
    by_ep = {}
    for s in stats: by_ep.setdefault(s['ep'], []).append(s)
    sel = {}
    for ep, ss in by_ep.items():
        ss = sorted(ss, key=lambda s: s['frame'])
        areas = np.array([s['mask_px'] for s in ss], float)
        sharp = np.array([s.get('sharp', np.nan) for s in ss]); smed = np.nanmedian(sharp)
        for k, s in enumerate(ss):
            rs = []
            if s['mask_px'] < a.min_mask_px or 'iou_init' not in s: rs.append('small/absent')
            else:
                if s['vis_frac'] < a.min_vis: rs.append('occluded(vis_frac)')
                if s['vis_frac'] > 1.3: rs.append('mask>render')
                if s['truncated']: rs.append('truncated')
                if s['occl_frac'] > 0.05: rs.append('depth_in_front')
                if s['hand_frac'] > a.max_hand_frac: rs.append('hand_over_object')
                if s['other_frac'] > a.max_other_frac: rs.append('other_obj_in_front')
                if s['iou_init'] < a.min_iou_init: rs.append('pose_mask_disagree')
                nb = areas[max(0, k-2):k+3]; nb = np.delete(nb, min(k, 2)); nb = nb[nb > 0]
                if len(nb) and abs(s['mask_px']/np.median(nb) - 1) > 0.25: rs.append('mask_jump')
                if s['sharp'] < a.min_sharp_rel*smed: rs.append('blur')
            s['reject'] = rs
        clean = [s for s in ss if not s['reject']]
        # greedy farthest-point viewpoint selection in the object frame
        keys = []
        if clean:
            dirs = np.array([s['view_dir'] for s in clean]); q = np.array([s['iou_init'] for s in clean])
            first = int(np.argmax(q)); keys = [first]
            mind = np.degrees(np.arccos(np.clip(dirs @ dirs[first], -1, 1)))
            while len(keys) < a.max_keys:
                fr = np.array([s['frame'] for s in clean]); kf = fr[keys]
                tsep = np.abs(fr[:, None] - kf[None]).min(1) >= 3
                score = np.where(tsep, mind, -1)
                j = int(np.argmax(score))
                if score[j] < a.min_angle:
                    # fill up to min_keys with temporally spread clean frames (>= 10 frames apart), still farthest-view first
                    if len(keys) >= a.min_keys: break
                    score = np.where(np.abs(fr[:, None] - kf[None]).min(1) >= 10, mind, -1); j = int(np.argmax(score))
                    if score[j] < 0: break
                keys.append(j); mind = np.minimum(mind, np.degrees(np.arccos(np.clip(dirs @ dirs[j], -1, 1))))
        kf = sorted(clean[j]['frame'] for j in keys)
        hold = [s['frame'] for s in clean if s['frame'] not in kf and min(abs(s['frame']-k) for k in kf) > 2] if kf else []
        if len(hold) > a.max_hold: hold = [hold[i] for i in np.linspace(0, len(hold)-1, a.max_hold).round().astype(int)]
        for s in ss: s['role'] = 'key' if s['frame'] in kf else ('holdout' if s['frame'] in hold else ('clean' if not s['reject'] else 'rejected'))
        sel[ep] = dict(keyframes=kf, holdout=hold, n_clean=len(clean), n_frames=len(ss),
                       reject_counts={r: sum(r in s['reject'] for s in ss) for r in sorted({r for s in ss for r in s['reject']})})
        say(f'ep{ep}: {len(ss)} frames, clean {len(clean)}, keyframes {len(kf)}, holdout {len(hold)}; rejects {sel[ep]["reject_counts"]}')
    return sel


def contact_sheet(metas, obj, stats, sel, path):
    tiles = []
    def tile(m, f, label, col):
        M, O, D, rgb = load_frame(m['E'], f, obj); im = rgb[:, :, ::-1].copy()
        cs, _ = cv2.findContours(M.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(im, cs, -1, (0, 0, 255), 2)
        if M.any():
            ys, xs = np.nonzero(M); cy, cx_ = (ys.min()+ys.max())//2, (xs.min()+xs.max())//2
            s = int(max(ys.max()-ys.min(), xs.max()-xs.min())*0.8) + 40
        else: cy, cx_, s = H//2, W//2, 300
        y0, x0 = max(0, cy-s), max(0, cx_-s); crop = im[y0:cy+s, x0:cx_+s]
        t = cv2.resize(crop, (180, 180)); cv2.rectangle(t, (0, 0), (179, 179), col, 3)
        cv2.putText(t, label[:28], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        cv2.putText(t, label[:28], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        return t
    rows = []
    for m in metas:
        ep = m['ep']; ks = sel[ep]['keyframes']
        ts = [tile(m, f, f'ep{ep} f{f} KEY', (0, 200, 0)) for f in ks]
        rej = [s for s in stats if s['ep'] == ep and s['reject'] and s['reject'] != ['small/absent']]
        seen = set(); ex = []
        for s in rej:
            r = s['reject'][0]
            if r in seen: continue
            seen.add(r); ex.append(s)
        ts += [tile(m, s['frame'], f'f{s["frame"]} X {",".join(s["reject"])}', (0, 0, 220)) for s in ex[:8]]
        for i in range(0, len(ts), 12):
            r = ts[i:i+12]; r += [np.zeros_like(ts[0])]*(12-len(r)); rows.append(np.hstack(r))
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def yaw_search(cx, fs, idx_list, V, F, camT, A, C, step_deg=10, min_gain=0.03):
    """Symmetry-aware pose re-initialisation for revolve objects: for each frame try rotations of the object about its
    axis (through C) and keep the one with the best silhouette IoU vs the mask (only if it beats the input by min_gain).
    Fixes tracking flips of thin asymmetric parts (handles) that a local refinement cannot undo."""
    torch = cx.torch
    yaws = np.radians(np.arange(0, 360, step_deg)); A = np.asarray(A, float)/np.linalg.norm(A); C = np.asarray(C, float)
    Kx = np.array([[0, -A[2], A[1]], [A[2], 0, -A[0]], [-A[1], A[0], 0]])
    Rs = [np.eye(3) + np.sin(y)*Kx + (1-np.cos(y))*Kx@Kx for y in yaws]
    D = np.stack([np.block([[R, (C - R@C)[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]) for R in Rs])
    Vt = cx.T(V); Ft = cx.T(F, torch.int32); out = camT.copy(); log = []
    with torch.no_grad():
        for k, i in enumerate(idx_list):
            Tc = camT[k] @ D
            _, _, hard, _ = cx.render(Vt, Ft, cx.T(Tc[:, :3, :3]), cx.T(Tc[:, :3, 3]), fs.K[i].expand(len(D), 4), want_soft=False)
            M = fs.M[i][None]
            iou = ((hard & M).flatten(1).sum(1).float()/(hard | M).flatten(1).sum(1).float().clamp_min(1)).cpu().numpy()
            j = int(np.argmax(iou))
            if iou[j] > iou[0] + min_gain: out[k] = Tc[j]
            log.append(dict(item=list(map(int, fs.items[i])), yaw_deg=float(np.degrees(yaws[j])) if iou[j] > iou[0] + min_gain else 0.0,
                            iou0=float(iou[0]), iou_best=float(iou[j])))
    return out, log


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--object', default='blue_cup'); ap.add_argument('--episodes', default='auto')
    ap.add_argument('--gpu', default='0'); ap.add_argument('--out-root', default=str(T3/'multiep_recon_v1'))
    ap.add_argument('--proc-mesh', default=None); ap.add_argument('--proc-params', default=None)
    ap.add_argument('--raw-glb', default=None); ap.add_argument('--metric-registration', default=None)
    ap.add_argument('--edge-mm', type=float, default=2.2, help='<=0: auto from --target-faces')
    ap.add_argument('--target-faces', type=int, default=40000)
    ap.add_argument('--deform', default='auto', choices=['auto', 'revolve', 'box'])
    ap.add_argument('--cad', type=int, default=0, help='1: param stage only, export final_4096.obj by deforming procedural visual_4096 (cad path)')
    ap.add_argument('--tau-occ', type=float, default=0.015)
    # selector
    ap.add_argument('--min-mask-px', type=int, default=1500); ap.add_argument('--min-vis', type=float, default=0.80)
    ap.add_argument('--max-hand-frac', type=float, default=0.03); ap.add_argument('--max-other-frac', type=float, default=0.03)
    ap.add_argument('--min-iou-init', type=float, default=0.65)
    ap.add_argument('--min-sharp-rel', type=float, default=0.4); ap.add_argument('--min-angle', type=float, default=10.0)
    ap.add_argument('--max-keys', type=int, default=40); ap.add_argument('--min-keys', type=int, default=6); ap.add_argument('--max-hold', type=int, default=40)
    # optimisation
    ap.add_argument('--iters-param', type=int, default=800); ap.add_argument('--iters-free', type=int, default=1000)
    ap.add_argument('--batch', type=int, default=8); ap.add_argument('--remesh-every', type=int, default=250)
    ap.add_argument('--w-sil', type=float, default=1.0); ap.add_argument('--w-col', type=float, default=0.5)
    ap.add_argument('--w-dep', type=float, default=1.0); ap.add_argument('--w-pts', type=float, default=0.5)
    ap.add_argument('--w-tether', type=float, default=1e-2); ap.add_argument('--w-lap', type=float, default=50.0)
    ap.add_argument('--w-nc', type=float, default=0.02); ap.add_argument('--lr-v', type=float, default=3e-5)
    ap.add_argument('--diffuse', type=int, default=20); ap.add_argument('--n-prof', type=int, default=5)
    ap.add_argument('--refit-iters', type=int, default=100); ap.add_argument('--w-gauge', type=float, default=1.0)
    ap.add_argument('--col-geom', type=int, default=1, help='param stage: 1 = colour loss also moves geometry'); ap.add_argument('--col-geom-free', type=int, default=0); ap.add_argument('--vnorm', default='vertex', choices=['global', 'vertex'])
    ap.add_argument('--yaw-search', type=int, default=1, help='revolve objects: discrete rotation-about-axis re-init of keyframe/held-out poses')
    ap.add_argument('--remesh', type=int, default=0, help='1: periodic isotropic remeshing in the free stage (carried Laplacian reference; produced wrinkles in tests)')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--stop-after', default='')
    ap.add_argument('--init-poses', default=None, help='alternative initial poses: npz path format with {ep} {obj} (world_T_canonical, '
                    'mesh_scale, mesh_path in the procedural canonical frame, e.g. retrack_v8); default = learning_v7_geom poses')
    a = ap.parse_args()
    obj = a.object
    global INIT_POSES, INIT_PROC; INIT_POSES = a.init_poses
    pg = T3/'procedural_geometry_v1'/obj
    a.proc_mesh = a.proc_mesh or str(pg/'visual.obj'); a.proc_params = a.proc_params or str(pg/'params.json')
    INIT_PROC = a.proc_mesh
    lin = {x['object']: x for x in json.loads((ROOT/'notes/track3-sam3d-lineage.json').read_text())}.get(obj)
    public_lineage = lin is None or 'public' in lin['source']['dataset']
    if public_lineage:
        assert Path(a.proc_mesh).resolve() != (pg/'visual.obj').resolve(), f'{obj}: procedural mesh descends from public data; pass a scored-only --proc-mesh'
        a.raw_glb = None; a.metric_registration = None      # no SAM3D colours either
    else:
        a.raw_glb = a.raw_glb or lin['mesh_raw']
        a.metric_registration = a.metric_registration or str(Path(lin['mesh_metric']).parent/'registration.npz')
    if a.cad: a.iters_free = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    out = Path(a.out_root)/obj; out.mkdir(parents=True, exist_ok=True)
    global LOG; LOG = open(out/'run.log', 'a')
    if (out/'report.json').exists(): say('already done', out); return
    say('args', json.dumps(vars(a)))
    eps = find_episodes(obj) if a.episodes == 'auto' else [int(x) for x in a.episodes.split(',')]
    say('episodes', eps)
    metas = [episode_meta(e, obj) for e in eps]
    import torch
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    cx = Ctx(); dev = cx.dev
    init = stage_init(a, out, metas)
    P = trimesh.load(a.proc_mesh, process=False, force='mesh')
    hands = [hand_masks(m['E'], m['n']) for m in metas]

    # ---------------- keyframes
    kfile = out/'keyframes.json'
    if kfile.exists():
        kj = json.loads(kfile.read_text()); stats, sel = kj['stats'], {int(k): v for k, v in kj['selection'].items()}
    else:
        stats = render_init_stats(cx, metas, obj, hands, P, a)
        sel = select_keyframes(stats, a)
        kfile.write_text(json.dumps(dict(selector=dict(min_mask_px=a.min_mask_px, min_vis=a.min_vis, max_hand_frac=a.max_hand_frac, max_other_frac=a.max_other_frac,
                                                       min_iou_init=a.min_iou_init, min_sharp_rel=a.min_sharp_rel, min_angle=a.min_angle, min_keys=a.min_keys,
                                                       max_keys=a.max_keys, max_hold=a.max_hold, tau_occ=a.tau_occ),
                                         selection={str(k): v for k, v in sel.items()}, stats=stats)))
        contact_sheet(metas, obj, stats, sel, out/'keyframes_contact_sheet.jpg')
    if a.stop_after == 'keyframes': say('stop after keyframes'); return
    ep_index = {m['ep']: i for i, m in enumerate(metas)}
    key_items = [(ep_index[ep], f) for ep in eps for f in sel[ep]['keyframes']]
    hold_items = [(ep_index[ep], f) for ep in eps for f in sel[ep]['holdout']]
    t0 = time.time()
    KF = FrameSet(cx, key_items, metas, obj, hands); HO = FrameSet(cx, hold_items, metas, obj, hands)
    say(f'loaded {len(key_items)} keyframes + {len(hold_items)} holdout frames in {time.time()-t0:.0f}s; '
        f'GPU mem {torch.cuda.memory_allocated()/1e9:.2f} GB')
    camT_key0 = np.stack([metas[ei]['camT0'][f] for ei, f in key_items])
    camT_hold0 = np.stack([metas[ei]['camT0'][f] for ei, f in hold_items])
    ne = len(metas)

    # ---------------- param stage: profile deformation of the shared init mesh
    pp = json.loads(Path(a.proc_params).read_text())
    V0 = init['V']; F0 = init['F'].astype(np.int64); sbar = float(init['sbar'])
    if 'edge_mm' in init: a.edge_mm = float(init['edge_mm'])
    centre = V0.mean(0)
    from deform import Deformer
    kind = a.deform if a.deform != 'auto' else ('revolve' if 'axis' in pp and 'centre' in pp else 'box')
    dfm = Deformer(cx, kind, pp, V0, sbar, a.n_prof, F0)
    p4 = Path(a.proc_mesh).parent/'visual_4096.obj'
    pfile = out/'stage_param.pt'
    if pfile.exists():
        st = torch.load(pfile, weights_only=False)
    else:
        Ft = cx.T(F0, torch.int32); Fl = Ft.long()
        S0 = dfm.prepare(V0, F0)
        albedo = cx.T(init['albedo']).requires_grad_(True)
        gain = torch.ones(ne, 3, device=dev, requires_grad=True); bias = torch.zeros(ne, 3, device=dev, requires_grad=True)
        camT_key_init = camT_key0.copy(); ylog = []
        if kind == 'revolve' and a.yaw_search:
            camT_key_init, ylog = yaw_search(cx, KF, list(range(len(key_items))), V0, F0, camT_key0, pp['axis'], np.array(pp['centre'])*sbar)
            say(f'yaw search on keyframes: {sum(l["yaw_deg"] != 0 for l in ylog)}/{len(ylog)} re-initialised: '
                + ', '.join(f'ep{metas[l["item"][0]]["ep"]}f{l["item"][1]} {l["yaw_deg"]:.0f}deg {l["iou0"]:.2f}->{l["iou_best"]:.2f}' for l in ylog if l['yaw_deg'] != 0))
        poses = Poses(cx, camT_key_init, centre)
        opt = torch.optim.Adam([dict(params=[poses.rot, poses.tra], lr=1e-3), dict(params=dfm.params, lr=2e-3),
                                dict(params=[albedo], lr=1e-2), dict(params=[gain, bias], lr=5e-3)])
        base_lr = [g['lr'] for g in opt.param_groups]
        gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
        for it in range(a.iters_param):
            sc = 0.1 + 0.9*0.5*(1+math.cos(math.pi*it/a.iters_param))
            for g, l in zip(opt.param_groups, base_lr): g['lr'] = l*sc
            if it < 100: opt.param_groups[1]['lr'] = 0.0
            idx = torch.as_tensor(np.random.choice(len(key_items), min(a.batch, len(key_items)), replace=False), device=dev)
            V = dfm.apply(S0); surf = sample_surface(torch, V, Fl, 6000, gen)
            Rn, tn = poses.get(idx)
            tm = frame_terms(cx, KF, idx, V, Ft, albedo, gain, bias, Rn, tn, surf, a)
            loss = a.w_sil*tm['sil'].mean() + a.w_col*tm['col'].mean() + a.w_dep*tm['dep'].mean() + a.w_pts*tm['pts'].mean() \
                + a.w_tether*poses.tether() + a.w_gauge*poses.gauge() + dfm.prior()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad(): albedo.clamp_(0, 1); dfm.clamp()
            if it % 100 == 0 or it == a.iters_param-1:
                say(f'param it {it} loss {loss.item():.4f} sil {tm["sil"].mean().item():.4f} col {tm["col"].mean().item():.3f} '
                    f'dep {tm["dep"].mean().item():.3f} pts {tm["pts"].mean().item():.4f} {json.dumps(dfm.describe())} gain {np.round(gain.mean(1).tolist(),3).tolist()}')
        with torch.no_grad():
            Vp = dfm.apply(S0)
            c4 = trimesh.load(p4, process=False, force='mesh')
            V4 = dfm.apply(dfm.prepare(c4.vertices*sbar, c4.faces)).cpu().numpy().astype(np.float64)
        st = dict(V=Vp.cpu().numpy().astype(np.float64), F=F0, albedo=albedo.detach().cpu().numpy(), gain=gain.detach().cpu().numpy(),
                  bias=bias.detach().cpu().numpy(), rot=poses.rot.detach().cpu().numpy(), tra=poses.tra.detach().cpu().numpy(),
                  deform=dfm.describe(), deform_state=dfm.state(), V4096=V4, camT_key_init=camT_key_init, yaw_log=ylog, F4096=np.asarray(c4.faces, np.int64), cad_source=str(p4))
        if kind == 'revolve':
            st.update(prof=np.exp(dfm.state()[0]).tolist(), dh=dfm.state()[1].tolist(), dt=float(dfm.dt.item()))
        torch.save(st, pfile)
    say('param stage:', json.dumps(st.get('deform', {})))
    camT_key_used = st.get('camT_key_init', camT_key0)
    a.yaw_axis = pp['axis'] if (kind == 'revolve' and a.yaw_search) else None
    a.yaw_centre = (np.array(pp['centre'])*sbar).tolist() if a.yaw_axis is not None else None

    # ---------------- free-form stage with periodic remeshing
    ffile = out/'stage_free.pt'; ck = out/'stage_free_ckpt.pt'
    if ffile.exists():
        fr = torch.load(ffile, weights_only=False)
    else:
        if ck.exists():
            c = torch.load(ck, weights_only=False); say('resume free stage from it', c['it'])
        else:
            c = dict(it=0, V=st['V'], F=st['F'], albedo=st['albedo'], gain=st['gain'], bias=st['bias'], rot=st['rot'], tra=st['tra'],
                     m1=np.zeros_like(st['V']), m2=np.zeros(len(st['V'])), nremesh=0, remesh_fail=0, lref=None)
        a.col_geom = a.col_geom_free
        poses = Poses(cx, camT_key_used, centre)
        with torch.no_grad(): poses.rot.copy_(cx.T(c['rot'])); poses.tra.copy_(cx.T(c['tra']))
        gain = cx.T(c['gain']).requires_grad_(True); bias = cx.T(c['bias']).requires_grad_(True)
        opt_p = torch.optim.Adam([dict(params=[poses.rot, poses.tra], lr=1e-3), dict(params=[gain, bias], lr=3e-3)])
        gen = torch.Generator(device=dev); gen.manual_seed(a.seed + c['it'])
        def setup(Vn, Fn, An, m1, m2, lr0=None):
            Vt = cx.T(Vn).requires_grad_(True); Ft = cx.T(Fn, torch.int32); Fl = Ft.long()
            Adj, Et = build_adjacency(torch, Fn, len(Vn), dev)
            ef = torch.as_tensor(edge_face_pairs(Fn), device=dev)
            # reference Laplacian (detail) of the param-stage shape, carried through remeshing as a vertex attribute
            with torch.no_grad(): lref = Vt - torch.sparse.mm(Adj, Vt) if lr0 is None else cx.T(lr0)
            alb = cx.T(An).requires_grad_(True); oa = torch.optim.Adam([alb], lr=5e-3)
            return Vt, Ft, Fl, Adj, ef, lref, alb, oa, cx.T(m1), cx.T(m2)
        poses_gauge = poses
        Vt, Ft, Fl, Adj, ef, lref, alb, oa, m1, m2 = setup(c['V'], c['F'], c['albedo'], c['m1'], c['m2'], c['lref'])
        b1, b2 = 0.9, 0.99
        it = c['it']; nremesh, remesh_fail = c['nremesh'], c['remesh_fail']
        while it < a.iters_free:
            sc = 0.1 + 0.9*0.5*(1+math.cos(math.pi*it/a.iters_free))
            for g, l in zip(opt_p.param_groups, [1e-3, 3e-3]): g['lr'] = l*sc
            idx = torch.as_tensor(np.random.choice(len(key_items), min(a.batch, len(key_items)), replace=False), device=dev)
            surf = sample_surface(torch, Vt, Fl, 6000, gen)
            Rn, tn = poses.get(idx)
            tm = frame_terms(cx, KF, idx, Vt, Ft, alb, gain, bias, Rn, tn, surf, a)
            lap = Vt - torch.sparse.mm(Adj, Vt)
            l_lap = ((lap - lref)**2).sum(-1).mean()/(a.edge_mm/1000)**2
            fn = torch.cross(Vt[Fl[:, 1]]-Vt[Fl[:, 0]], Vt[Fl[:, 2]]-Vt[Fl[:, 0]], dim=-1); fn = fn/fn.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            l_nc = (1 - (fn[ef[:, 0]]*fn[ef[:, 1]]).sum(-1)).mean()
            loss = a.w_sil*tm['sil'].mean() + a.w_col*tm['col'].mean() + a.w_dep*tm['dep'].mean() + a.w_pts*tm['pts'].mean() \
                + a.w_tether*poses.tether() + a.w_gauge*poses_gauge.gauge() + a.w_lap*l_lap + a.w_nc*l_nc
            opt_p.zero_grad(); oa.zero_grad(); Vt.grad = None
            loss.backward(); opt_p.step(); oa.step()
            with torch.no_grad():
                alb.clamp_(0, 1)
                g = Vt.grad
                for _ in range(a.diffuse): g = 0.5*g + 0.5*torch.sparse.mm(Adj, g)
                t = it + 1
                m1.mul_(b1).add_(g, alpha=1-b1)
                if a.vnorm == 'global': m2.mul_(b2).add_((g**2).sum(-1).mean().expand_as(m2), alpha=1-b2)
                else: m2.mul_(b2).add_((g**2).sum(-1), alpha=1-b2)
                step = (m1/(1-b1**min(t, 1000)))/((m2/(1-b2**min(t, 1000))).sqrt()[:, None] + 1e-12)
                ramp = min(1.0, (it+1)/100)
                Vt.sub_(a.lr_v*sc*ramp*step)
            it += 1
            if it % 100 == 0 or it == a.iters_free:
                ext = (Vt.max(0).values - Vt.min(0).values).tolist()
                say(f'free it {it} loss {loss.item():.4f} sil {tm["sil"].mean().item():.4f} col {tm["col"].mean().item():.3f} '
                    f'dep {tm["dep"].mean().item():.3f} pts {tm["pts"].mean().item():.4f} lap {l_lap.item():.4f} nc {l_nc.item():.4f} '
                    f'V {len(Vt)} ext_mm {np.round(np.array(ext)*1000,1).tolist()}')
            if (it % a.remesh_every == 0 and it < a.iters_free) or it == a.iters_free:
                do_remesh = bool(a.remesh)
                Vn = Vt.detach().cpu().numpy().astype(np.float64); Fn = Ft.cpu().numpy().astype(np.int64)
                if it < a.iters_free and do_remesh:
                    An = np.concatenate([alb.detach().cpu().numpy(), m1.cpu().numpy(), m2.cpu().numpy()[:, None], lref.cpu().numpy()], 1)
                    try:
                        V2, F2, A2 = RM.remesh(Vn, Fn, An, a.edge_mm/1000, rounds=1, relax_lam=0.2)
                        ok, mm = mesh_ok(V2, F2); ok = ok and mm.euler_number == trimesh.Trimesh(Vn, Fn, process=False).euler_number
                    except Exception as e:
                        ok = False; say('remesh error', e)
                    if ok:
                        nremesh += 1
                        Vt, Ft, Fl, Adj, ef, lref, alb, oa, m1, m2 = setup(V2, F2, A2[:, :3], A2[:, 3:6], A2[:, 6], A2[:, 7:10])
                        say(f'remesh #{nremesh} at it {it}: {len(Fn)}F -> {len(F2)}F')
                    else:
                        remesh_fail += 1; say(f'remesh at it {it} rejected (topology check), keeping mesh')
                c = dict(it=it, V=Vt.detach().cpu().numpy().astype(np.float64), F=Ft.cpu().numpy().astype(np.int64),
                         albedo=alb.detach().cpu().numpy(), gain=gain.detach().cpu().numpy(), bias=bias.detach().cpu().numpy(),
                         rot=poses.rot.detach().cpu().numpy(), tra=poses.tra.detach().cpu().numpy(), m1=m1.cpu().numpy(), m2=m2.cpu().numpy(),
                         nremesh=nremesh, remesh_fail=remesh_fail, lref=lref.cpu().numpy())
                torch.save(c, ck)
        fr = c; fr['camT_key'] = poses.numpy(); torch.save(fr, ffile)

    # ---------------- evaluation
    import export as EX
    EX.finish(a, out, cx, metas, obj, eps, init, st, fr, sel, stats, KF, HO, key_items, hold_items, camT_key0, camT_hold0, P, centre,
              say, summarise, eval_frames, frame_terms, Poses, sample_surface, rms_radius, sha256, dir_digest, FrameSet, hands)


if __name__ == '__main__':
    main()
