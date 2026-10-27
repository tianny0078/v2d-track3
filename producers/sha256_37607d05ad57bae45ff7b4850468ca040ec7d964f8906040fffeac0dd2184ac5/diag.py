"""Read-only diagnostics of the source hands vs objects / 2D / depth evidence (before any correction).
Usage: python scripts/hand_contact_v1/diag.py 8 14 28 37
"""
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from common import *  # noqa


def mediapipe_match(data, uv):
    """Match MediaPipe candidates (no identity) to our 2 hands by mean 2D distance of wrist+MCP+tips. -> (T,2,21,2) with NaN."""
    z = np.load(T3/data['E']/'hand_detector_mediapipe_sequence/visible_candidates.npz')
    xy = z['xy_pixels'].astype(float); valid = z['valid']; xy[~valid] = np.nan
    T = data['T']; out = np.full((T, 2, 21, 2), np.nan); dist = np.full((T, 2), np.nan)
    for f in range(T):
        C = np.full((2, xy.shape[1]), 1e9)
        for h in range(2):
            for c in range(xy.shape[1]):
                d = np.linalg.norm(uv[f, h]-xy[f, c], axis=-1)
                if np.isfinite(d).sum() >= 8: C[h, c] = np.nanmedian(d)
        r, c = linear_sum_assignment(C)
        for h, cc in zip(r, c):
            if C[h, cc] < 80: out[f, h] = xy[f, cc]; dist[f, h] = C[h, cc]
    return out, dist


def depth_check(data, f, Xc_z, uv):
    D = np.load(T3/data['E']/f'foundation_stereo_v1/depth_rgb/{f:06d}.npz')['depth'].astype(np.float32)
    M = np.load(T3/data['E']/f'object_masks_v3/masks/{f:06d}.npz')['masks'].any(0)
    H, W = D.shape; res = []
    for h in range(2):
        u = np.round(uv[h]).astype(int)
        ok = (u[:, 0] >= 0) & (u[:, 0] < W) & (u[:, 1] >= 0) & (u[:, 1] < H)
        vals = []
        for j in np.flatnonzero(ok):
            x, y = u[j]; patch = D[max(0, y-2):y+3, max(0, x-2):x+3]; m = M[max(0, y-2):y+3, max(0, x-2):x+3]
            v = patch[~m & np.isfinite(patch)]
            vals.append(np.median(v)-Xc_z[h, j] if len(v) else np.nan)
        res.append(np.nanmedian(vals) if len(vals) and np.isfinite(vals).any() else np.nan)
    return res


def main(ep):
    d = load_episode(ep); T = d['T']; J = d['J']
    grids = [sdf_grid(m) for m in d['mesh']]
    # SDF sanity
    import trimesh
    for b, m in enumerate(d['mesh']):
        mm = trimesh.load(m, force='mesh', process=False); s, _ = trimesh.sample.sample_surface(mm, 2000, seed=0)
        print(f'  sdf check {d["names"][b]}: |sd| on surface p50/p99 = {np.median(abs(sdf_query_np(grids[b], s)))*1000:.2f}/'
              f'{np.quantile(abs(sdf_query_np(grids[b], s)), .99)*1000:.2f} mm, watertight={mm.is_watertight}, extent cm={np.round(mm.extents*100,1)}')
    Pt = d['A'] @ d['P']; core = moving_mask(Pt)
    S, rad = hand_samples(J)
    sd = object_sd(d, grids, S)                              # (T,2,N,B)
    tip = sd[:, :, TIPS]                                     # (T,2,5,B)
    pen = np.nanmax(rad[None, None, :, None]-sd, axis=2)    # (T,2,B)
    print(f'ep{ep} T={T} objects={d["names"]} moving frames={core.sum(0).tolist()}')
    for b, n in enumerate(d['names']):
        fs = np.flatnonzero(core[:, b])
        if not len(fs): continue
        dmin = np.nanmin(np.abs(tip[fs, :, :, b]).reshape(len(fs), -1), -1)
        print(f'  {n}: nearest tip |sd| over both hands p10/50/90 = {np.round(np.nanquantile(dmin,[.1,.5,.9])*100,1)} cm; '
              f'hand pen (r-sd) max over hands p50/max = {np.round([np.nanmedian(np.nanmax(pen[fs,:,b],1))*1000, np.nanmax(pen[fs,:,b])*1000],1)} mm')
        # best global time shift between hand and object (hand frame f+s vs object frame f)
        for s in range(-4, 5):
            ff = fs[(fs+s >= 0) & (fs+s < T)]
            Ss, _ = hand_samples(J[ff+s]); sds = object_sd(d, grids, Ss, frames=ff)[:, :, TIPS, b]
            print(f'    shift {s:+d}: median nearest tip {np.nanmedian(np.nanmin(np.abs(sds).reshape(len(ff), -1), -1))*100:.2f} cm', end='')
        print()
    uv, z = project(J, d['world_T_rgb'], d['K'])
    mp, md = mediapipe_match(d, uv)
    e = np.linalg.norm(uv-mp, axis=-1)
    print(f'  2D: matched MediaPipe frames L/R = {np.isfinite(md).sum(0).tolist()}/{T}; source-vs-MediaPipe px p50/p90 = '
          f'{np.round(np.nanquantile(e,[.5,.9]),1)}; wrist {np.round(np.nanquantile(e[...,0],[.5,.9]),1)} tips {np.round(np.nanquantile(e[...,TIPS],[.5,.9]),1)}')
    dz = np.array([depth_check(d, f, z[f], uv[f]) for f in range(0, T, 3)])
    print(f'  depth: stereo(hand pixels, non-object) - joint z, median per hand p10/50/90 L={np.round(np.nanquantile(dz[:,0],[.1,.5,.9])*100,1)} '
          f'R={np.round(np.nanquantile(dz[:,1],[.1,.5,.9])*100,1)} cm')
    # ground
    Jt = to_table(J, d['A'])
    print(f'  ground: min joint z (table) p1/p50 = {np.round(np.nanquantile(np.nanmin(Jt[...,2],-1),[.01,.5])*100,1)} cm, frames below -1cm = {(np.nanmin(Jt[...,2],-1)< -.01).sum(0).tolist()}')


if __name__ == '__main__':
    for e in map(int, sys.argv[1:]): main(e)
