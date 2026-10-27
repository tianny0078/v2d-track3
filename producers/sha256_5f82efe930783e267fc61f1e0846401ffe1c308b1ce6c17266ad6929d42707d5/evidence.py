"""Per-frame image evidence for the source hands (cached in work/hand_contact_v1/evidence/<E>.npz).

- uv_src (T,2,21,2): projection of the source joints (= the WiLoR/DBT 2D the source was built from).
- uv_mp  (T,2,21,2): MediaPipe 2D landmarks matched to our hands (no identity in the raw slots -> Hungarian on 2D
                     distance, accept < 80 px median). Held out: used only for validation, never in the objective.
- d_obs  (T,2,21):   FoundationStereo depth (m, rgb camera) at each projected joint: median of the 7x7 patch with
                     object-mask pixels removed; NaN if < 10 valid px.
- mask_px (T,2,B):   2D distance (px) from the hand's fingertips+palm keypoints to object b's mask (0 = on the mask).
- mask_area (T,B):   object mask pixel count.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa


def mediapipe_match(E, uv, T):
    z = np.load(T3/E/'hand_detector_mediapipe_sequence/visible_candidates.npz')
    xy = z['xy_pixels'].astype(float); valid = z['valid']; xy[~valid] = np.nan
    out = np.full((T, 2, 21, 2), np.nan); dist = np.full((T, 2), np.nan)
    for f in range(T):
        C = np.full((2, xy.shape[1]), 1e9)
        for h in range(2):
            for c in range(xy.shape[1]):
                dd = np.linalg.norm(uv[f, h]-xy[f, c], axis=-1)
                if np.isfinite(dd).sum() >= 8: C[h, c] = np.nanmedian(dd)
        r, c = linear_sum_assignment(C)
        for h, cc in zip(r, c):
            if C[h, cc] < 80: out[f, h] = xy[f, cc]; dist[f, h] = C[h, cc]
    return out, dist


def build(ep, force=False):
    E = ep_name(ep); out = WORK/'evidence'/f'{E}.npz'
    if out.exists() and not force: return dict(np.load(out, allow_pickle=True))
    d = load_episode(ep); T = d['T']; B = len(d['names'])
    uv, z = project(d['J'], d['world_T_rgb'], d['K'])
    uv_mp, mp_dist = mediapipe_match(E, uv, T)
    d_obs = np.full((T, 2, 21), np.nan); mask_px = np.full((T, 2, B), np.nan); area = np.zeros((T, B))
    KP = [0, 4, 8, 12, 16, 20, 5, 9, 13, 17]
    for f in range(T):
        D = np.load(T3/E/f'foundation_stereo_v1/depth_rgb/{f:06d}.npz')['depth'].astype(np.float32)
        mz = np.load(T3/E/f'object_masks_v3/masks/{f:06d}.npz'); M = mz['masks']; mn = [str(n) for n in mz['object_names']]
        Many = M.any(0); H, W = D.shape
        for h in range(2):
            if not np.isfinite(uv[f, h]).all(): continue
            for j in range(21):
                x, y = np.round(uv[f, h, j]).astype(int)
                if not (0 <= x < W and 0 <= y < H): continue
                p = D[max(0, y-3):y+4, max(0, x-3):x+4]; m = Many[max(0, y-3):y+4, max(0, x-3):x+4]
                v = p[~m & np.isfinite(p)]
                if len(v) >= 10: d_obs[f, h, j] = np.median(v)
        for b, n in enumerate(d['names']):
            if n not in mn: continue
            mb = M[mn.index(n)]; area[f, b] = mb.sum()
            if not mb.any(): continue
            dt = distance_transform_edt(~mb)
            for h in range(2):
                if not np.isfinite(uv[f, h]).all(): continue
                q = np.round(uv[f, h, KP]).astype(int); ok = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)
                if ok.any(): mask_px[f, h, b] = dt[q[ok, 1], q[ok, 0]].min()
    res = dict(uv_src=uv, z_src=z, uv_mp=uv_mp, mp_dist=mp_dist, d_obs=d_obs, mask_px=mask_px, mask_area=area,
               names=np.array(d['names']))
    out.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(out, **res)
    return res


if __name__ == '__main__':
    for e in map(int, sys.argv[1:]):
        r = build(e, force=True); print(e, 'MediaPipe matched', np.isfinite(r['mp_dist']).sum(0), 'depth valid frac', np.isfinite(r['d_obs']).mean().round(2), flush=True)
