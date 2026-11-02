"""Rigidly align the adopted (multiep_recon_v1) white_pot into the OLD procedural_geometry_v1 white_pot canonical frame
(main-session decision 2026-10-31: learning_v5 FoundationPose source poses live in the old mesh frame).
Rigid only (rotation + translation, no scale): trimmed point-to-point ICP of the v8 high-res visual (new frame) onto the
v1 visual, multi-start over yaw about the pot axis (every 15 deg, both handle orientations) so the handles fix the yaw;
the start with the lowest trimmed RMS wins. Writes outputs/track3/procedural_geometry_v8/white_pot/align_to_v1.json
(T_old_from_new 4x4, angle, translation, residuals, all starts). Run with .deps/geom-venv/bin/python."""
import json, sys
from pathlib import Path
import numpy as np, trimesh
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).parent))
from common import T3, icp, sample, chamfer

D = T3/'procedural_geometry_v8/white_pot'
new = trimesh.load(D/'visual.obj', force='mesh'); old = trimesh.load(T3/'procedural_geometry_v1/white_pot/visual.obj', force='mesh')
sp = json.loads((D/'deform_spec.json').read_text())
assert sp['spec'].get('post_rigid') is None, 'deform_spec already carries a post_rigid transform: align from the un-aligned build'
ax = np.array(json.loads((T3/'procedural_geometry_v1/white_pot/params.json').read_text())['axis'], float); ax /= np.linalg.norm(ax)
src = sample(new, 30000, 7); c_new, c_old = src.mean(0), sample(old, 30000, 8).mean(0)
starts = []
for yaw in range(0, 360, 15):
    R0 = Rotation.from_rotvec(np.radians(yaw)*ax).as_matrix(); t0 = c_old - R0 @ c_new
    M, rms, med = icp(src, old, iters=60, trim=0.9, R0=R0, t0=t0)
    starts.append(dict(yaw0=yaw, rms_mm=rms, median_mm=med, M=M.tolist(),
                       angle_deg=float(np.degrees(Rotation.from_matrix(M[:3, :3]).magnitude()))))
best = min(starts, key=lambda s: s['rms_mm']); M = np.array(best['M'])
alt = min((s for s in starts if np.degrees(Rotation.from_matrix(np.array(s['M'])[:3, :3] @ M[:3, :3].T).magnitude()) > 20), key=lambda s: s['rms_mm'], default=None)
moved = new.copy(); moved.apply_transform(M)
rv = Rotation.from_matrix(M[:3, :3]).as_rotvec()
rep = dict(T_old_from_new=M.tolist(), rotation_angle_deg=float(np.degrees(np.linalg.norm(rv))), rotation_axis=(rv/np.linalg.norm(rv)).tolist(),
           translation_mm=(M[:3, 3]*1000).tolist(), translation_norm_mm=float(np.linalg.norm(M[:3, 3])*1000),
           axis_angle_new_vs_old_before_deg=float(np.degrees(np.arccos(abs(np.dot(np.array(json.loads(Path(sp['proc_params']).read_text())['axis'])/np.linalg.norm(json.loads(Path(sp['proc_params']).read_text())['axis']), ax))))),
           icp=dict(rms_mm=best['rms_mm'], median_mm=best['median_mm'], best_yaw_start=best['yaw0'],
                    second_branch=None if alt is None else dict(yaw0=alt['yaw0'], rms_mm=alt['rms_mm'])),
           chamfer_new_vs_old_before=chamfer(new, old), chamfer_new_vs_old_after=chamfer(moved, old),
           starts=[{k: v for k, v in s.items() if k != 'M'} for s in starts],
           note='new (multiep_recon_v1 white_pot, init_scored frame) -> old (procedural_geometry_v1 white_pot frame); rigid, no scale')
(D/'align_to_v1.json').write_text(json.dumps(rep, indent=1))
print(json.dumps({k: rep[k] for k in ['rotation_angle_deg', 'rotation_axis', 'translation_mm', 'translation_norm_mm', 'axis_angle_new_vs_old_before_deg', 'icp']}, indent=1))
print('chamfer before/after', round(rep['chamfer_new_vs_old_before']['chamfer_mean_mm'], 2), round(rep['chamfer_new_vs_old_after']['chamfer_mean_mm'], 2))
