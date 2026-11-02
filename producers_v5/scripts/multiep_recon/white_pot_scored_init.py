"""Scored-only initialisation of the white_pot parametric model (no public data, no SAM3D mesh, no procedural_geometry_v1/white_pot).

Inputs (scored split only): object_masks_v3 masks (eroded), foundation_stereo_v1 depth, calibrated rgb_K, learning_v7_geom
initial poses (world_T_canonical, used ONLY as the coordinate frame / initial poses) + ORB world_T_rgb.
Method: back-project eroded-mask stereo points of every frame with a usable mask into the canonical frame, then
  1. robust cylinder fit (axis direction, axis point, R) to the fused points (soft-L1, handle/floor points are outliers),
  2. axis sign = side where near-axis (floor) points lie -> bottom; h0/h1 = 1st / 99.5th percentile of body-point heights,
  3. handle = points beyond R+8 mm: mean radial direction, reach (99th pct), width / lateral offset (P3..P97), centre
     height + slope (as in white_pot.fit), 4) generic constants from white_pot.py: t_wall 2.5 mm, handle 4 mm thick,
     hole r 5 mm, bottom fillet r 15 mm (no scored measurement of these), NO markers (marker detection is not allowed).
Writes <out>/params_scored.json (white_pot.build() format) + fit diagnostics.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, cv2
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recon as RC  # noqa: E402  (data loaders only)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--obj', default='white_pot')
    ap.add_argument('--pts-per-frame', type=int, default=200); ap.add_argument('--step', type=int, default=2)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    eps = RC.find_episodes(a.obj); rng = np.random.default_rng(0)
    P = []; per_ep = {}
    for e in eps:
        m = RC.episode_meta(e, a.obj); n0 = len(P)
        for f in range(0, m['n'], a.step):
            z = np.load(m['E']/f'object_masks_v3/masks/{f:06d}.npz'); nm = [str(x) for x in z['object_names']]
            if a.obj not in nm: continue
            M = z['masks'][nm.index(a.obj)].astype(np.uint8)
            if M.sum() < 3000: continue
            M = cv2.erode(M, np.ones((3, 3), np.uint8), iterations=4).astype(bool)
            D = np.load(m['E']/f'foundation_stereo_v1/depth_rgb/{f:06d}.npz')['depth'].astype(np.float32)
            ok = M & np.isfinite(D) & (D > 0.1) & (D < 2.0); v, u = np.nonzero(ok)
            if len(v) < 200: continue
            s = rng.choice(len(v), min(a.pts_per_frame, len(v)), replace=False); v, u = v[s], u[s]; zz = D[v, u]; K = m['K']
            Xc = np.stack([(u-K[0, 2])*zz/K[0, 0], (v-K[1, 2])*zz/K[1, 1], zz], 1)
            T = np.linalg.inv(m['camT0'][f]); P.append(Xc @ T[:3, :3].T + T[:3, 3])
        per_ep[e] = len(P) - n0
    X = np.concatenate(P); print('fused points', X.shape, 'frames per episode', per_ep)
    # robust cylinder fit: params (theta, phi) of axis, centre (2 in-plane offsets), R
    c0 = np.median(X, 0); ev, E = np.linalg.eigh(np.cov((X - c0).T))
    def axis_of(q): th, ph = q[0], q[1]; return np.array([np.sin(th)*np.cos(ph), np.cos(th), np.sin(th)*np.sin(ph)])
    def frame(ax):
        t = np.array([1., 0, 0]) if abs(ax[0]) < 0.9 else np.array([0, 0, 1.]); e1 = np.cross(ax, t); e1 /= np.linalg.norm(e1); return e1, np.cross(ax, e1)
    best = None
    # start: each principal direction and canonical y as candidate axis (the pot is a squat cylinder)
    # fused cloud (3-view density plot, see init_scored/fused_points_*.png) shows a squat cylinder around canonical y with the
    # handle along one radial direction and some table-plane bleed -> start from canonical y with a 2-D circle start in x-z
    xz = X[:, [0, 2]]; c0 = np.array([np.median(X[:, 0]), np.median(X[:, 1]), np.median(X[:, 2])])
    for ax0 in [np.array([0, 1., 0])]:
        ax0 = ax0/np.linalg.norm(ax0)*np.sign(ax0[1] if abs(ax0[1]) > 1e-3 else 1)
        th0 = np.arccos(np.clip(ax0[1], -1, 1)); ph0 = np.arctan2(ax0[2], ax0[0])
        def res(q):
            ax = axis_of(q); e1, e2 = frame(ax0); c = c0 + q[2]*e1 + q[3]*e2
            d = X - c; r = np.linalg.norm(d - np.outer(d @ ax, ax), axis=1); return r - q[4]
        q = least_squares(res, [th0, ph0, 0, 0, 0.085], loss='cauchy', f_scale=0.002).x
        cost = np.median(np.abs(res(q)))
        if best is None or cost < best[0]: best = (cost, q, ax0)
    cost, q, ax0 = best; ax = axis_of(q); e1, e2 = frame(ax0); c = c0 + q[2]*e1 + q[3]*e2; R = float(q[4])
    d = X - c; h = d @ ax; radial = d - np.outer(h, ax); r = np.linalg.norm(radial, axis=1)
    body = np.abs(r - R) < 0.006
    near = r < 0.5*R
    print(f'cylinder: axis {np.round(ax,4)} R {R*1000:.1f} mm median |res| {cost*1000:.2f} mm, body pts {body.mean():.2f}, near-axis pts {near.sum()}')
    # axis sign: the pot rests on the table, whose plane bleeds into the masks as a wide flat band of points far outside the
    # body (r > R + 30 mm); the side of the body where that band lies is the bottom (near-axis points are ambiguous: floor
    # seen from above vs the lid in ep1)
    # axis sign (bottom -> rim): the head camera looks down at the pot while it rests on the table at the start of every
    # episode (ASSUMPTION: pot upright at rest in the first 15 frames) -> the camera centre lies on the rim side.
    votes = []
    for e in eps:
        m = RC.episode_meta(e, a.obj)
        for f in range(min(15, m['n'])):
            T = m['camT0'][f]; cc = -T[:3, :3].T @ T[:3, 3]; votes.append(float((cc - c) @ ax))
    votes = np.array(votes)
    flip_reason = f'camera-above vote over first 15 frames/episode: {int((votes > 0).sum())} rim-side vs {int((votes < 0).sum())} bottom-side'
    if np.median(votes) < 0: ax = -ax; h = -h; radial = d - np.outer(h, ax)
    h0 = float(np.percentile(h[body], 1.0)); h1 = float(np.percentile(h[body], 99.5))
    # handle: points beyond the wall, above the table band; direction = peak of the azimuth histogram
    out_pts = (r > R + 0.008) & (h > h0 + 0.02) & (h < h1 + 0.02)
    hp = dict()
    if out_pts.sum() > 200:
        e1_, e2_ = frame(ax); az = np.arctan2(radial @ e2_, radial @ e1_)
        hist, edges = np.histogram(az[out_pts], bins=36, range=(-np.pi, np.pi)); k = int(np.argmax(hist)); a0 = (edges[k] + edges[k+1])/2
        sel_ = out_pts & (np.abs(np.angle(np.exp(1j*(az - a0)))) < np.radians(25))
        dirv = radial[sel_]/r[sel_, None]; dh = dirv.mean(0); dh -= np.dot(dh, ax)*ax; dh /= np.linalg.norm(dh)
        out_pts = sel_
        side = np.cross(ax, dh); s = d @ dh; l = d @ side
        hk = out_pts & (np.abs(l) < 0.05) & (s > R)
        smax = float(np.percentile(s[hk], 97))
        mid = hk & (s > R + 0.015) & (s < smax - 0.03)
        if mid.sum() < 50: mid = hk
        w = float(np.percentile(l[mid], 97) - np.percentile(l[mid], 3)); lc = float((np.percentile(l[mid], 97) + np.percentile(l[mid], 3))/2)
        hc = float(np.median(h[mid]))
        sb = np.linspace(R + 0.015, smax - 0.03, 7); mids = []
        for lo_, hi_ in zip(sb[:-1], sb[1:]):
            qq = mid & (s >= lo_) & (s < hi_)
            if qq.sum() > 30: mids.append(((lo_+hi_)/2, np.median(h[qq])))
        mids = np.array(mids); slope = float(np.polyfit(mids[:, 0], mids[:, 1], 1)[0]) if len(mids) > 2 else 0.0
        hp = dict(handle_dir=dh.tolist(), handle_side=side.tolist(), handle_width=float(np.clip(w, 0.015, 0.06)), handle_lateral_offset=lc,
                  handle_h_centre=hc, handle_slope=slope, handle_reach=smax, handle_pts=int(out_pts.sum()))
        print(f'handle: pts {out_pts.sum()} reach {smax*1000:.1f} mm width {w*1000:.1f} mm h_centre {hc*1000:.1f} mm slope {slope:.3f}')
    else:
        raise SystemExit('no handle points found')
    p = dict(axis=ax.tolist(), centre=c.tolist(), R=R, h0=h0, h1=h1, height=h1 - h0, r_fillet=0.015, t_wall=0.0025,
             handle_thickness=0.004, handle_hole_r=0.005, markers=[], **{k: v for k, v in hp.items() if k != 'handle_pts'},
             provenance=dict(method=__doc__.strip().splitlines()[0], scored_episodes=eps, frames_used_per_episode=per_ep, fused_points=int(len(X)),
                             cylinder_median_abs_residual_mm=float(cost*1000), axis_sign=flip_reason, handle_points=hp.get('handle_pts'),
                             generic_constants='t_wall 2.5 mm, handle_thickness 4 mm, hole 5 mm (white_pot.py assumptions), r_fillet 15 mm (generic), no markers'))
    (out/'params_scored.json').write_text(json.dumps(p, indent=1))
    np.save(out/'fused_points_canonical.npy', X[::4].astype(np.float32))
    print(json.dumps({k: v for k, v in p.items() if k != 'provenance'}, indent=None)[:800])


if __name__ == '__main__':
    main()
