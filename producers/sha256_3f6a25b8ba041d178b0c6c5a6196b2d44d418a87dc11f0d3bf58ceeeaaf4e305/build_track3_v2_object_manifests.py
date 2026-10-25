"""Object manifests for learning v3: same assets as the v2 cohort, object trajectories replaced by FoundationPose v2.

world_T_link = world_T_rgb (VO) @ camera_T_link (FP v2). Per object: frames with a jump >JUMP_M or >JUMP_DEG from the
running pose, or missing, are dropped and re-filled (linear translation, slerp rotation); then a 5-frame centred
smoothing (translation mean, quaternion chordal mean). observed=kept raw frames; planned_reference=filled frames.
Writes outputs/track3/learning_v3/episode_X/objects/{objects.json,poses.npz,report.json}.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

R = Path(__file__).resolve().parents[1]; T = R/'outputs/track3'; OUT = T/'learning_v3'
SRC_TAG = 'foundationpose_v2'  # overridden by --src-tag / --out (e.g. foundationpose_v4 -> learning_v4)
CAMS = 'stereo_sequence'  # --cams=stereo_sequence_orb_objmask for v5 (ORB-SLAM3 VO)
JUMP_M, JUMP_DEG, WIN = 0.08, 30.0, 5
EPS = [1, 3, 5, 6, 8, 10, 14, 15, 16, 19, 20, 25, 28, 30, 32, 33, 35, 36, 37, 38]

def clean(P):
    n = len(P); ok = np.isfinite(P).all((1, 2)); keep = ok.copy(); last = None
    for f in range(n):
        if not ok[f]: continue
        if last is not None:
            dt = np.linalg.norm(P[f, :3, 3]-P[last, :3, 3]); dr = np.degrees(Rotation.from_matrix(P[last, :3, :3].T @ P[f, :3, :3]).magnitude())
            if dt > JUMP_M*(f-last) or dr > JUMP_DEG*(f-last): keep[f] = False; continue
        last = f
    idx = np.flatnonzero(keep)
    if len(idx) < 2: raise RuntimeError('fewer than 2 usable frames')
    t = np.stack([np.interp(np.arange(n), idx, P[idx, :3, 3][:, k]) for k in range(3)], 1)
    rots = Rotation.from_matrix(P[idx, :3, :3]); fr = np.clip(np.arange(n), idx[0], idx[-1])
    r = Slerp(idx, rots)(fr).as_quat()
    for k in range(1, n): r[k] *= np.sign(np.dot(r[k], r[k-1])) or 1
    ts = np.array([t[max(0, f-WIN//2):f+WIN//2+1].mean(0) for f in range(n)])
    qs = np.array([r[max(0, f-WIN//2):f+WIN//2+1].mean(0) for f in range(n)]); qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    S = np.tile(np.eye(4), (n, 1, 1)); S[:, :3, :3] = Rotation.from_quat(qs).as_matrix(); S[:, :3, 3] = ts
    return S, keep

def scaled_assets(item, scale, dst):
    """copy an object's URDF asset dir with every .obj uniformly scaled about the link origin; COM*s, inertia*s^2, mass kept."""
    import re, shutil, trimesh
    src = Path(item['urdf_path']).parent; dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.suffix == '.obj':
            m = trimesh.load(f, force='mesh', process=False); m.vertices = m.vertices*scale; m.export(dst/f.name)
        elif f.suffix != '.urdf' and f.is_file(): shutil.copy2(f, dst/f.name)
    u = (src/'object.urdf').read_text()
    u = re.sub(r'(<origin xyz=")([^"]+)(")', lambda g: g.group(1)+' '.join(repr(float(x)*scale) for x in g.group(2).split())+g.group(3), u, count=1)
    u = re.sub(r'(i(?:xx|xy|xz|yy|yz|zz)=")([^"]+)(")', lambda g: g.group(1)+repr(float(g.group(2))*scale**2)+g.group(3), u)
    (dst/'object.urdf').write_text(u)
    it = dict(item, mesh_path=str(dst/Path(item['mesh_path']).name), urdf_path=str(dst/'object.urdf'), canonical_COM=[c*scale for c in item['canonical_COM']],
              bounding_radius_m=item['bounding_radius_m']*scale, uniform_scale_applied=scale)
    return it

def main(ep):
    c = {e['episode']: e for e in json.loads((T/'full20-comparison/comparison.json').read_text())['episodes']}[ep]
    src = Path(c['scene']['object_manifest']); man = json.loads(src.read_text())
    E = T/f'episode_{ep:06d}'; fp = np.load(E/SRC_TAG/'poses_table.npz')
    fpn = list(map(str, fp['object_names'])); names = man['object_names']; camT = fp['camera_T_link'][:, [fpn.index(n) for n in names]]
    P = np.load(E/CAMS/'camera_poses.npz'); cams = dict(zip(P['frame_index'].tolist(), P['world_T_rgb']))
    frames = np.asarray(man['frame_index']); n = len(frames)
    W = np.full((n, len(names), 4, 4), np.nan)
    for i, f in enumerate(frames):
        if f in cams: W[i] = cams[f] @ camT[f]
    out = OUT/f'episode_{ep:06d}/objects'; out.mkdir(parents=True, exist_ok=True)
    WS = np.zeros_like(W); obs = np.zeros((n, len(names)), bool); stats = {}
    for i, nm in enumerate(names):
        WS[:, i], obs[:, i] = clean(W[:, i])
        old = np.load(man['poses_path'])['world_T_canonical'][:, i]
        stats[nm] = dict(kept=int(obs[:, i].sum()), frames=n, filled=int(n-obs[:, i].sum()),
                         mean_shift_vs_old_cm=float(np.linalg.norm(WS[:, i, :3, 3]-old[:, :3, 3], axis=1).mean()*100),
                         max_motion_cm=float(np.linalg.norm(WS[:, i, :3, 3]-WS[:1, i, :3, 3], axis=1).max()*100))
    q = Rotation.from_matrix(WS[..., :3, :3].reshape(-1, 3, 3)).as_quat(scalar_first=True).reshape(n, len(names), 4)
    np.savez_compressed(out/'poses.npz', frame_index=frames, object_names=np.array(names), world_T_canonical=WS,
                        object_pose_wxyz=np.concatenate([WS[..., :3, 3], q], -1), observed=obs, planned_reference=~obs)
    srcj = E/SRC_TAG/'SOURCE.json'
    if srcj.exists():
        sj = json.loads(srcj.read_text()); objs = []
        for item in man['objects']:
            sc = float(sj.get(item['name'], {}).get('scale', 1.0))
            objs.append(scaled_assets(item, sc, out.parent/'assets'/item['name']) if abs(sc-1) > 1e-6 else item)
        man = dict(man, objects=objs)
    new = dict(man); new.update(poses_path=str(out/'poses.npz'), object_pose_wxyz=None, observed=True, experimental=True,
                                reconstruction_source=man.get('reconstruction_source', '')+f' | poses: {SRC_TAG}',
                                source_manifest=str(src))
    new.pop('object_pose_wxyz')
    (out/'objects.json').write_text(json.dumps(new, indent=1))
    (out/'report.json').write_text(json.dumps(dict(episode=ep, source_manifest=str(src), jump_m=JUMP_M, jump_deg=JUMP_DEG, smooth_window=WIN, objects=stats), indent=1))
    print(ep, json.dumps(stats), flush=True)

if __name__ == '__main__':
    args = sys.argv[1:]
    while args and args[0].startswith('--'):
        k, v = args[0][2:].split('=', 1); args = args[1:]
        if k == 'src-tag': SRC_TAG = v
        if k == 'out': OUT = T/v
        if k == 'cams': CAMS = v
    for e in (map(int, args) if args else EPS): main(e)
