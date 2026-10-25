"""COPY of scripts/fix_track3_reference_support.py (unchanged logic) re-run on the procedural geometry (procedural_geometry_v1).

Differences to the original (only these):
  * poses: the same pre-fix source (learning_v5 poses.npz + hands), assets: the learning_v7_geom manifest (new URDFs);
    output written into the learning_v7_geom tree (poses.npz + support_fix.json; objects.json keeps the v7 assets)
  * the frame-0 object-object lift cap LIFT_MAX is a parameter (original: 0.02 m); with clean concave collision parts
    the depth is monotonic in the lift, and ep1's lid starts ~45 mm inside the pot (FoundationPose height error)
  * imports point to scripts/ (the original helpers are reused, not modified)
Usage: .deps/flash-chord-venv/bin/python scripts/procedural_geometry/fix_support_v7.py [--tree learning_v7_geom] [--lift-max 0.06] [eps...]

Original description:
Support-consistent object references (learning_v6): remove table penetration / floating from the object trajectories.

Training scene = infinite ground plane z=0 in the table frame; objects = URDF collision hulls. The v5 references start
3.7-42.6 mm inside the table (or 23-73 mm above it), so the reset ejects / drops them (penetration_report.json).

Per object (table frame, height = lowest collision-hull point):
  resting = still (COM speed < STILL_MPS over a 5-frame window), not touched (no hand joint within HELD_M of its hull,
            dilated +-HELD_PAD frames; hands = the learning tree's hands_world.npz) and not sitting on another object
            (another object's hull footprint under its COM and that object's top above 3 cm). An untouched still object
            must be on a support, whatever height the reference puts it at.
  stacked frames follow the supporting object's dz.
  dz[f]   = CLEAR - minz[f] on resting frames, linearly interpolated in between (held constant before the first / after
            the last resting frame), then dz[f] >= CLEAR - minz[f] on every frame (never below the table).
  Objects with no resting frame get only the never-below-table clamp. Then object-object penetration at frame 0 is
  removed by lifting the upper object (higher COM) by depth + CLEAR over the whole clip (iterated).
Only z (table normal) changes; x/y/rotation are untouched. Writes outputs/track3/<out>/episode_X/objects/ (assets copied
by reference, poses.npz replaced) + support_fix.json.
"""
import json, shutil, sys
from pathlib import Path
import numpy as np, trimesh
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_argv = sys.argv; sys.argv = sys.argv[:1]
from diag_track3_initial_penetration import parts, xf, depth_into  # module reads argv at import
sys.argv = _argv

R = Path(__file__).resolve().parents[2]; T = R/'outputs/track3'
SRC, OUT = 'learning_v5', 'learning_v7_geom'; LIFT_MAX = 0.06
STILL_MPS, HELD_M, HELD_PAD, CLEAR, FPS = 0.03, 0.04, 5, 0.001, 30.0
SETTLE, TOPPLE_DEG = 1.5, 25.0  # stage-2 quasi-static settle seconds (0 = off) / max accepted settle rotation
EPS = [1, 3, 5, 6, 8, 10, 14, 15, 16, 19, 20, 25, 28, 30, 32, 33, 35, 36, 37, 38]

def minz_all(P, geo): return np.array([min(xf(p, v)[:, 2].min() for v, _, _ in geo) for p in P])

def footprint_under(Pa, geoa, Pb, geob):
    """True if b's COM xy lies inside a's hull xy bounding box and a's top is > 3 cm (b may sit on a)."""
    va = np.concatenate([xf(Pa, v) for v, _, _ in geoa]); c = Pb[:3, 3]
    lo, hi = va[:, :2].min(0), va[:, :2].max(0)
    return bool((c[:2] > lo).all() and (c[:2] < hi).all() and va[:, 2].max() > 0.03)

def stable_ups(verts, com):
    """object-frame up vectors of the hull's stable resting faces: the COM projects inside the (coplanar) face polygon."""
    from scipy.spatial import ConvexHull, Delaunay
    h = ConvexHull(verts); out = []; seen = []
    for eq in h.equations:
        n, d = eq[:3], -eq[3]  # outward normal, plane n.x = d
        if any(np.dot(n, m) > 0.999 for m in seen): continue
        seen.append(n); on = verts[np.abs(verts @ n - d) < 1e-3]
        if len(on) < 3: continue
        e1 = np.cross(n, [1, 0, 0] if abs(n[0]) < .9 else [0, 1, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
        uv = np.c_[on @ e1, on @ e2]; c = np.array([com @ e1, com @ e2])
        try:
            if Delaunay(uv).find_simplex(c) >= 0: out.append(-n)
        except Exception: pass
    return out

def settle(P2, names, items, geo, rests, rep):
    """Stage 2: quasi-static settle of each rest segment (first frame; other objects static, damped free joint, SETTLE s).
    The object-frame correction L = P^-1 S is applied on the segment, interpolated (translation linear, rotation slerp)
    between segments, held outside; then the never-below-table clamp again. A settle that rotates > TOPPLE_DEG is a
    topple (the reference pose is not a stable pose near the video's), kept as identity and reported."""
    import mujoco
    from scipy.spatial.transform import Rotation, Slerp
    from rest_test_track3_reference import mjcf
    import xml.etree.ElementTree as ET
    n, B = P2.shape[:2]; P3 = P2.copy()
    coms = [np.array(list(map(float, ET.parse(items[m]['urdf_path']).getroot().find('link/inertial/origin').attrib['xyz'].split()))) for m in names]
    for b in sorted(range(B), key=lambda k: P2[0, k, 2, 3]):  # lower objects first; others static at their corrected poses
        nm = names[b]; r = rests[b]; segs = np.split(np.flatnonzero(r), np.flatnonzero(np.diff(np.flatnonzero(r)) > 1) + 1) if r.any() else []
        def settle_one(Q, f):
            objs = [(m, items[m]['urdf_path'], P3[f, k]) for k, m in enumerate(names)]
            for it in range(4):  # damped settle, then an undamped 0.5 s check; repeat from the settled pose until it holds
                objs[b] = (nm, items[nm]['urdf_path'], Q)
                M = mujoco.MjModel.from_xml_string(mjcf(objs, free={b}, damping=0.02)); d = mujoco.MjData(M)
                for _ in range(int(SETTLE/M.opt.timestep)): mujoco.mj_step(M, d)
                Q = np.eye(4); Q[:3, :3] = d.body(nm).xmat.reshape(3, 3); Q[:3, 3] = d.body(nm).xpos; objs[b] = (nm, items[nm]['urdf_path'], Q)
                M = mujoco.MjModel.from_xml_string(mjcf(objs, free={b})); d = mujoco.MjData(M); mujoco.mj_step(M, d); x0 = d.body(nm).xipos.copy()
                for _ in range(249): mujoco.mj_step(M, d)
                if np.linalg.norm(d.body(nm).xipos - x0) < 0.002: break
            return Q, it + 1
        res = {}; info = []; up = None
        for attempt in range(2):  # attempt 1: re-seed toppling segments upright (object up-axis from a stable segment)
            for sg in segs:
                f = int(sg[0])
                if f in res: continue
                Q = P2[f, b].copy()
                if attempt == 1:
                    u = up
                    if u is None:  # no stable segment: nearest stable pose of the hull on a plane (trimesh), probability > 5%
                        cands = stable_ups(np.concatenate([v for v, _, _ in geo[b]]), coms[b])
                        if not cands: break
                        u = max(cands, key=lambda c: (Q[:3, :3] @ c)[2])
                    Rfix = Rotation.align_vectors([[0, 0, 1.]], [Q[:3, :3] @ u])[0].as_matrix()
                    Q[:3, :3] = Rfix @ Q[:3, :3]; Q[2, 3] += CLEAR - min(xf(Q, v)[:, 2].min() for v, _, _ in geo[b])
                S, iters = settle_one(Q, f)
                L = np.linalg.inv(P2[f, b]) @ S; rot = float(np.degrees(Rotation.from_matrix(L[:3, :3]).magnitude()))
                rs = float(np.degrees(Rotation.from_matrix(Q[:3, :3].T @ S[:3, :3]).magnitude()))  # settle rotation from its seed
                ok = bool(np.isfinite(L).all() and (rot <= TOPPLE_DEG if attempt == 0 else (rs <= TOPPLE_DEG and rot <= 45.)))
                info.append(dict(frame=f, len=len(sg), attempt=attempt, iters=iters, settle_rot_deg=round(rot, 1),
                                 settle_move_mm=round(float(np.linalg.norm(S[:3, 3]-P2[f, b, :3, 3])*1000), 1), used=ok))
                if ok:
                    res[f] = (sg, L)
                    if up is None: up = S[:3, :3].T @ np.array([0, 0, 1.])
        keys = [int(g) for f, (sg, L) in res.items() for g in sg]; Ls = [L for f, (sg, L) in res.items() for g in sg]
        rep[nm]['settle'] = info
        if not keys: continue
        keys, Ls = zip(*sorted(zip(keys, Ls), key=lambda x: x[0]))
        keys = np.array(keys); Ls = np.array(Ls); fr = np.clip(np.arange(n), keys[0], keys[-1])
        t = np.stack([np.interp(fr, keys, Ls[:, i, 3]) for i in range(3)], 1)
        Rm = Slerp(keys, Rotation.from_matrix(Ls[:, :3, :3]))(fr).as_matrix()
        Lf = np.tile(np.eye(4), (n, 1, 1)); Lf[:, :3, :3] = Rm; Lf[:, :3, 3] = t
        P3[:, b] = P2[:, b] @ Lf
        mz = minz_all(P3[:, b], geo[b]); P3[:, b, 2, 3] += np.maximum(0, CLEAR - mz)
    return P3

def main(ep):
    S = T/SRC/f'episode_{ep:06d}/objects'; D = T/OUT/f'episode_{ep:06d}/objects'
    man = json.loads((D/'objects.json').read_text()); z = dict(np.load(S/'poses.npz'))  # v7 assets, v5 (pre-fix) poses
    A = np.asarray(man['table_T_world'], float); Ainv = np.linalg.inv(A)
    W = z['world_T_canonical']; n, B = W.shape[:2]; names = [str(x) for x in z['object_names']]
    items = {o['name']: o for o in man['objects']}; geo = [parts(items[nm]['urdf_path']) for nm in names]
    P = np.einsum('ij,fbjk->fbik', A, W)  # table_T_link
    hz = np.load(T/SRC/f'hands/episode_{ep:06d}/hands_world.npz'); assert (hz['frame_index'] == z['frame_index']).all()
    J = np.where(hz['present'][..., None, None], hz['joints_world'], np.nan).reshape(n, -1, 3); H = J @ A[:3, :3].T + A[:3, 3]
    rep = {}; dzs = np.zeros((n, B)); sups = {}; mzs = {}; rests = {}
    for b, nm in enumerate(names):
        mz = minz_all(P[:, b], geo[b]); mzs[b] = mz; c = P[:, b, :3, 3]
        v = np.r_[0, np.linalg.norm(np.diff(c, axis=0), axis=1)]*FPS; vs = np.convolve(v, np.ones(5)/5, 'same')
        sup = np.full(n, -1)
        for f in range(n):
            for k in range(B):
                if k != b and P[f, k, 2, 3] < P[f, b, 2, 3] and footprint_under(P[f, k], geo[k], P[f, b], geo[b]): sup[f] = k
        stacked = sup >= 0; sups[b] = sup
        vb = np.concatenate([v for v, _, _ in geo[b]])[::7]; held = np.zeros(n, bool)
        for f in range(n):
            j = H[f][np.isfinite(H[f]).all(1)]
            if len(j): held[f] = np.linalg.norm(((j - P[f, b, :3, 3]) @ P[f, b, :3, :3])[:, None] - vb[None], axis=-1).min() < HELD_M
        held = np.convolve(held, np.ones(2*HELD_PAD+1), 'same') > 0
        rest = (vs < STILL_MPS) & ~held & ~stacked; rest[0] = not stacked[0]  # the clip starts with objects set down
        rests[b] = ((vs < STILL_MPS) & ~held) | rest  # settle segments: still and untouched, stacked included
        if rest.any():
            idx = np.flatnonzero(rest); dz = np.interp(np.arange(n), idx, CLEAR - mz[idx])
        else: dz = np.zeros(n)
        dz = np.maximum(dz, CLEAR - mz); dzs[:, b] = dz
        rep[nm] = dict(rest_frames=int(rest.sum()), held_frames=int(held.sum()), stacked_frames=int(stacked.sum()), f0_minz_mm=round(mz[0]*1000, 1),
                       dz_f0_mm=round(dz[0]*1000, 1), dz_median_mm=round(float(np.median(dz))*1000, 1), dz_max_abs_mm=round(float(np.abs(dz).max())*1000, 1))
    for b in range(B):  # stacked frames ride on their support's correction
        for f in np.flatnonzero(sups[b] >= 0): dzs[f, b] = max(dzs[f, sups[b][f]], CLEAR - mzs[b][f])
    P2 = P.copy(); P2[..., 2, 3] += dzs
    for _ in range(4):  # object-object penetration at frame 0
        moved = False
        for i in range(B):
            for j in range(i+1, B):
                a = np.concatenate([xf(P2[0, i], v) for v, _, _ in geo[i]]); c = np.concatenate([xf(P2[0, j], v) for v, _, _ in geo[j]])
                d = max(depth_into(a, geo[j], P2[0, j]), depth_into(c, geo[i], P2[0, i]))
                if d > CLEAR:  # lift the upper object in 1 mm steps until frame-0 overlap <= CLEAR (concave parts: depth is not monotonic)
                    up, lo = (i, j) if P2[0, i, 2, 3] > P2[0, j, 2, 3] else (j, i); lift = 0.
                    while d > CLEAR and lift < LIFT_MAX:
                        lift += 0.001; Q = P2[0, up].copy(); Q[2, 3] += lift
                        q = np.concatenate([xf(Q, v) for v, _, _ in geo[up]]); r = np.concatenate([xf(P2[0, lo], v) for v, _, _ in geo[lo]])
                        d = max(depth_into(q, geo[lo], P2[0, lo]), depth_into(r, geo[up], Q))
                    if d > CLEAR:  # not separable by a small lift (concave decomposition): leave it, report it
                        rep[names[up]]['pair_unresolved_f0_mm'] = round(d*1000, 1); continue
                    P2[:, up, 2, 3] += lift; dzs[:, up] += lift; moved = True
                    rep[names[up]]['pair_lift_mm'] = round(rep[names[up]].get('pair_lift_mm', 0) + lift*1000, 1)
        if not moved: break
    if SETTLE: P2 = settle(P2, names, items, geo, rests, rep)
    for b, nm in enumerate(names): rep[nm]['f0_minz_after_mm'] = round(minz_all(P2[:1, b], geo[b])[0]*1000, 1)
    W2 = np.einsum('ij,fbjk->fbik', Ainv, P2)
    z2 = dict(z, world_T_canonical=W2); z2['object_pose_wxyz'] = np.concatenate([W2[..., :3, 3], z['object_pose_wxyz'][..., 3:]], -1)
    np.savez_compressed(D/'poses.npz', **z2)
    (D/'objects.json').write_text(json.dumps(dict(man, poses_path=str(D/'poses.npz'), poses_source='learning_v5 poses re-fixed by scripts/procedural_geometry/fix_support_v7.py on the procedural geometry',
        reconstruction_source=man.get('reconstruction_source', '') + ' | support fix: procedural_geometry/fix_support_v7.py (copy of fix_track3_reference_support.py)'), indent=1))
    (D/'support_fix.json').write_text(json.dumps(dict(episode=ep, source=str(S), still_mps=STILL_MPS, held_m=HELD_M, held_pad=HELD_PAD, clearance_m=CLEAR, lift_max_m=LIFT_MAX, objects=rep), indent=1))
    print(ep, json.dumps(rep), flush=True)

if __name__ == '__main__':
    a = sys.argv[1:]
    while a and a[0].startswith('--'):
        k = a.pop(0)[2:]; v = a.pop(0)
        if k == 'tree': OUT = v
        if k == 'lift-max': LIFT_MAX = float(v)
    assert OUT != 'learning_v6'
    for e in (map(int, a) if a else EPS): main(e)
