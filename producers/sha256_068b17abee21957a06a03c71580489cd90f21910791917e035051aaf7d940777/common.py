"""hand_contact_v1 shared loaders: hands (MANO21, world), objects (v7_geom), cameras, 2D/depth evidence, SDF grids.

Conventions (verified, see notes/track3-hand-contact-v1.md §2):
- hands_world.npz hand axis 0 = left, 1 = right (build_track3_scored_reference.SIDES); joints MANO21 output order
  (0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky; tips 4/8/12/16/20).
- world = ORB-SLAM3 world; joints_world = world_T_rgb @ joints_camera_m (build_track3_v5_hands.py).
- table frame = table_T_world @ world, support plane z = 0.
- object poses: world_T_canonical (link origin = mesh frame), meshes = procedural_geometry_v1 visual.obj (watertight).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/tianyuan/projects/v2d')
T3 = ROOT/'outputs/track3'
WORK = ROOT/'work/hand_contact_v1'
NAS = Path('/mnt/nas3/tianyuan/v2d-track3/hand_contact_v1')
EPS = [1, 3, 5, 6, 8, 10, 14, 15, 16, 19, 20, 25, 28, 30, 32, 33, 35, 36, 37, 38]
TIPS = [4, 8, 12, 16, 20]
PALM = [0, 1, 5, 9, 13, 17]
# MANO21 kinematic tree (parent of each joint)
PARENT = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
# skin radius proxy per joint (m): wrist/palm thicker, fingers ~8 mm, tip keypoints are skin vertices (MANO tip ids)
RADIUS = np.array([.020, .012, .009, .008, .005, .010, .008, .007, .005, .010, .008, .007, .005,
                   .010, .008, .007, .005, .010, .007, .006, .005])
TIP_R = .005  # tip keypoint (MANO tip vertex) to finger-pad contact: skin distance = sd - TIP_R


def ep_name(ep):
    return f'episode_{ep:06d}'


def load_episode(ep, hands_root='learning_v5/hands', objects_tree='learning_v7_geom'):
    E = ep_name(ep)
    h = dict(np.load(T3/hands_root/E/'hands_world.npz', allow_pickle=True))
    man = json.loads((T3/objects_tree/E/'objects/objects.json').read_text())
    z = np.load(man['poses_path'], allow_pickle=True)
    names = [str(n) for n in z['object_names']]
    assert names == list(man['object_names'])
    items = {o['name']: o for o in man['objects']}
    P = np.asarray(z['world_T_canonical'], np.float64)
    A = np.asarray(man['table_T_world'], np.float64)
    cam = np.load(T3/E/'stereo_sequence_orb_objmask/camera_poses.npz')
    assert np.array_equal(cam['frame_index'], h['frame_index'])
    K = np.load(T3/E/'calibrated_sequence/rgb_K.npy')
    assert np.array_equal(np.asarray(man['frame_index']), h['frame_index']), 'hand/object frame axes differ'
    return dict(ep=ep, E=E, h=h, J=np.asarray(h['joints_world'], np.float64), present=np.asarray(h['present'], bool),
                names=names, mesh=[items[n]['mesh_path'] for n in names], urdf=[items[n]['urdf_path'] for n in names],
                mass=[items[n].get('mass_kg') for n in names], P=P, A=A, man=man,
                world_T_rgb=np.asarray(cam['world_T_rgb'], np.float64), K=K, T=len(h['frame_index']))


def to_table(X, A):
    return X @ A[:3, :3].T + A[:3, 3]


def project(Xw, world_T_rgb, K):
    """Xw (...,T?,N,3) world -> pixels; world_T_rgb broadcast over leading axis 0 (frames)."""
    Rc = world_T_rgb[:, :3, :3]; tc = world_T_rgb[:, :3, 3]
    sh = Xw.shape
    X = Xw.reshape(sh[0], -1, 3)
    Xc = np.einsum('fji,fnj->fni', Rc, X - tc[:, None])
    uv = Xc[..., :2] / Xc[..., 2:3] * np.diag(K)[:2] + K[:2, 2]
    return uv.reshape(*sh[:-1], 2), Xc[..., 2].reshape(sh[:-1])


def moving_mask(P_table, fps=20.0):
    """contact_plan.plan core definition (lin > 5 cm/s or ang > 0.8 rad/s 3-frame smoothed, or lifted > 3 cm,
    binary opening 3) evaluated on table-frame object poses (T,B,4,4)."""
    from scipy.ndimage import binary_opening, uniform_filter1d
    from scipy.spatial.transform import Rotation
    p = P_table[..., :3, 3]; R = P_table[..., :3, :3]; T, B = p.shape[:2]
    lin = uniform_filter1d(np.linalg.norm(np.gradient(p, axis=0)*fps, axis=-1), 3, axis=0)
    dR = np.einsum('fbij,fbkj->fbik', R[1:], R[:-1])
    ang = np.linalg.norm(Rotation.from_matrix(dR.reshape(-1, 3, 3)).as_rotvec().reshape(T-1, B, 3), axis=-1)*fps
    ang = uniform_filter1d(np.r_[ang[:1], ang], 3, axis=0)
    lifted = (p[..., 2]-p[..., 2].min(0)) > .03
    core = (lin > .05) | (ang > .8) | lifted
    for b in range(B):
        core[:, b] = binary_opening(core[:, b], np.ones(3, bool))
    return core


def ref_moved(P_table):
    p = P_table[..., :3, 3]
    return np.linalg.norm(p-p[:1], axis=-1)  # (T,B) displacement from frame 0


# ------------------------------------------------------------------------------------------------ SDF grids
FILL_NAMES = ('cup', 'white_pot', 'pitcher')  # containers whose open cavity is treated as solid for the hand (see fill_cavity)


def sdf_grid(mesh_path, res=0.0025, pad=0.08, fill=None):
    """Signed distance grid (negative inside) of a watertight mesh in its own (canonical) frame, cached under WORK.
    fill=None -> auto (container names in FILL_NAMES); fill=True -> cavity-filled SDF (fill_cavity)."""
    mp = Path(mesh_path)
    if fill is None: fill = any(n in mp.parent.name for n in FILL_NAMES)
    if fill: return fill_cavity(mesh_path, res, pad)
    key = hashlib.sha256(mp.read_bytes()+f'{res}{pad}'.encode()).hexdigest()[:16]
    cache = WORK/'sdf'/f'{mp.parent.name}_{key}.npz'
    if cache.exists():
        z = np.load(cache); return dict(origin=z['origin'], res=float(z['res']), sdf=z['sdf'])
    import open3d as o3d
    import trimesh
    m = trimesh.load(mp, force='mesh', process=False)
    lo = m.bounds[0]-pad; hi = m.bounds[1]+pad; n = np.ceil((hi-lo)/res).astype(int)+1
    g = np.stack(np.meshgrid(*[lo[i]+res*np.arange(n[i]) for i in range(3)], indexing='ij'), -1).reshape(-1, 3).astype(np.float32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.ascontiguousarray(m.vertices, np.float64)),
                                  o3d.utility.Vector3iVector(np.ascontiguousarray(m.faces, np.int32)))))
    sd = scene.compute_signed_distance(o3d.core.Tensor(g), nsamples=3).numpy().reshape(*n).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, origin=lo, res=res, sdf=sd, mesh=str(mp), watertight=m.is_watertight)
    return dict(origin=lo, res=res, sdf=sd)


def fill_cavity(mesh_path, res=0.0025, pad=0.08, frac=.75):
    """SDF of the mesh with enclosed open cavities filled (cup/pot interior): an outside grid point inside the convex hull
    is 'enclosed' if >= frac of 26 rays (3x3x3 directions) hit the mesh. Handle loops (2 open directions along the loop
    axis + more) and the gap under a pot handle stay open. Enclosed cells get -(voxel distance to the nearest open cell);
    outside cells get min(original sd, distance to the filled volume)."""
    from scipy.ndimage import distance_transform_edt
    mp = Path(mesh_path); key = hashlib.sha256(mp.read_bytes()+f'{res}{pad}{frac}fill'.encode()).hexdigest()[:16]
    cache = WORK/'sdf'/f'{mp.parent.name}_fill_{key}.npz'
    if cache.exists():
        z = np.load(cache); return dict(origin=z['origin'], res=float(z['res']), sdf=z['sdf'])
    import open3d as o3d
    import trimesh
    g = sdf_grid(mesh_path, res, pad, fill=False); sd = g['sdf']; o = g['origin']
    m = trimesh.load(mp, force='mesh', process=False)
    mk = lambda mm: o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh(  # noqa: E731
        o3d.utility.Vector3dVector(np.ascontiguousarray(mm.vertices, np.float64)), o3d.utility.Vector3iVector(np.ascontiguousarray(mm.faces, np.int32))))
    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(mk(m))
    hs = o3d.t.geometry.RaycastingScene(); hs.add_triangles(mk(m.convex_hull))
    n = np.array(sd.shape); P = np.stack(np.meshgrid(*[o[i]+res*np.arange(n[i]) for i in range(3)], indexing='ij'), -1).reshape(-1, 3).astype(np.float32)
    cand = np.flatnonzero((sd.reshape(-1) > 0) & (hs.compute_signed_distance(o3d.core.Tensor(P)).numpy() < 0))
    dirs = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)], np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True); hits = np.zeros(len(cand))
    for dvec in dirs:
        rays = np.concatenate([P[cand], np.broadcast_to(dvec, (len(cand), 3))], 1)
        hits += np.isfinite(sc.cast_rays(o3d.core.Tensor(np.ascontiguousarray(rays)))['t_hit'].numpy())
    occ = (sd < 0).reshape(-1); enc = np.zeros_like(occ); enc[cand[hits/len(dirs) >= frac]] = True
    occ = (occ | enc).reshape(sd.shape); enc = enc.reshape(sd.shape)
    din = distance_transform_edt(occ)*res; dout = distance_transform_edt(~occ)*res
    out = np.where(enc, -din, np.where(sd < 0, sd, np.minimum(sd, np.maximum(dout-res/2, 0)))).astype(np.float32)
    np.savez_compressed(cache, origin=o, res=res, sdf=out, enclosed_cells=int(enc.sum()), mesh=str(mp))
    return dict(origin=o, res=res, sdf=out)


def sdf_query_np(grid, X):
    """Trilinear SDF lookup (numpy), X (...,3) in mesh frame; outside the grid: distance to grid box + border value."""
    from scipy.ndimage import map_coordinates
    sd = grid['sdf']; o = grid['origin']; r = grid['res']
    idx = (X.reshape(-1, 3)-o)/r
    n = np.array(sd.shape)-1
    cl = np.clip(idx, 0, n)
    v = map_coordinates(sd, cl.T, order=1, mode='nearest')
    v = v + np.linalg.norm((idx-cl)*r, axis=-1)
    return v.reshape(X.shape[:-1])


def hand_samples(J, per_bone=3):
    """Points along every MANO bone (joint + per_bone interior samples) and their radii. J (...,21,3)."""
    pts = [J]; rad = [RADIUS]
    for j in range(1, 21):
        p = PARENT[j]
        for k in range(1, per_bone+1):
            a = k/(per_bone+1)
            pts.append(((1-a)*J[..., p, :]+a*J[..., j, :])[..., None, :]); rad.append(np.array([(1-a)*RADIUS[p]+a*RADIUS[j]]))
    # palm interior (between wrist and the 4 finger MCPs)
    for m in (5, 9, 13, 17):
        for a in (.33, .66):
            pts.append(((1-a)*J[..., 0, :]+a*J[..., m, :])[..., None, :]); rad.append(np.array([.015]))
    return np.concatenate(pts, -2), np.concatenate(rad)


def object_sd(data, grids, X_world, frames=None):
    """Signed distance of world points X (T,...,3) to every object at each frame -> (T,...,B)."""
    P = data['P'] if frames is None else data['P'][frames]
    out = []
    for b, g in enumerate(grids):
        R = P[:, b, :3, :3]; t = P[:, b, :3, 3]
        sh = X_world.shape
        Xl = np.einsum('fji,fnj->fni', R, X_world.reshape(sh[0], -1, 3)-t[:, None])
        out.append(sdf_query_np(g, Xl).reshape(sh[:-1]))
    return np.stack(out, -1)


def save_json(p, d):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); q = p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(d, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))+'\n'); q.replace(p)
