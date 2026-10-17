"""Backproject detector candidates using measured stereo only; no identity inference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('episode', type=Path)
a = p.parse_args()
root = a.episode
src = root / 'hand_detector_mediapipe_sequence/visible_candidates.npz'
z = np.load(src)
xy, visible = z['xy_pixels'], z['valid']
T = len(xy)
K = np.load(root / 'stereo_sequence/rgb_K.npy')
c = np.load(root / 'stereo_sequence/camera_poses.npz')
world = np.full((T, 2, 21, 3), np.nan)
valid = np.zeros((T, 2, 21), bool)
depth = np.full((T, 2, 21), np.nan)
hashes = {}
for f in range(T):
    dp = root / f'stereo_sequence/depth_rgb/{f:06d}.npz'
    hashes[str(dp)] = hashlib.sha256(dp.read_bytes()).hexdigest()
    d = np.load(dp)['depth']
    k = K if K.ndim == 2 else K[f]
    C = c['world_T_rgb'][f]
    for h, j in np.argwhere(visible[f]):
        u, v = xy[f, h, j]
        x, y = int(round(float(u))), int(round(float(v)))
        if not (0 <= x < d.shape[1] and 0 <= y < d.shape[0]):
            continue
        # Exact landmark pixel: do not borrow depth from nearby object/background.
        zz = float(d[y, x])
        if not np.isfinite(zz) or not .1 < zz < 3.:
            continue
        ray = np.linalg.solve(k, [u, v, 1.])
        world[f, h, j] = C[:3, :3] @ (ray * zz) + C[:3, 3]
        depth[f, h, j] = zz
        valid[f, h, j] = True
out = root / 'hands_world_candidates.npz'
np.savez_compressed(out, frame_index=np.arange(T), joints_world=world,
                   joint_valid=valid, depth_m=depth, candidate_present=z['candidate_present'],
                   side_identity_known=np.zeros((T, 2), bool))
# The reference interface requires anatomical left/right. Unidentified candidates
# must not silently become anatomical observations; preserve them separately.
np.savez_compressed(root / 'hands_world.npz', frame_index=np.arange(T),
                   present=np.zeros((T, 2), bool), joints_world=np.full_like(world, np.nan))
checks = dict(finite_iff_valid=bool(np.array_equal(np.isfinite(world).all(-1), valid)),
              missing_depth_nan=bool(np.isnan(world[~valid]).all()), all_frames=T == len(c['world_T_rgb']))
report = dict(frames=T, valid_joints=int(valid.sum()), total_image_valid=int(visible.sum()),
              complete_candidate_hands=int(valid.all(-1).sum()), checks=checks,
              all_passed=all(checks.values()), side_identity_known=False, ground_truth_known=False,
              reference_human_present=0, quality_pass=False,
              limitation='Stereo surface depth candidates; anatomical identity and occluded joint centers unverified. Reference uses explicit robot retreat, never invented human observations.',
              depth_sha256=hashes)
for path in [src, root/'stereo_sequence/rgb_K.npy', root/'stereo_sequence/camera_poses.npz', Path(__file__)]:
    hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
(root/'hands-world-report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='depth_sha256'}))
assert report['all_passed']
