"""Flat bamboo board with a halving-joint slot (wooden_piece_1, wooden_piece_2) + Vicon markers.

Parameters and how they are obtained:
  plane / envelope   board normal = canonical y (SVD plane fit of the up-facing samples: normal within 0.2 deg of y);
                     the SAM3D board is a hollow box y in [y_lo, y_hi] (outer faces) with an internal void
  thickness          8 mm (video: edge-on frames ep25 f118/f236/f300 -> 6-9 mm against the 14 mm markers / board length;
                     SAM3D's 12.8-13.5 mm is depth-inflated); centred on the SAM3D envelope centre
  outline            1 mm raster of the slab samples projected on x-z, morphological closing (fills the SAM3D slot gap),
                     largest contour, Douglas-Peucker 0.7 mm
  slot               direction/position from the SAM3D gap pixels (closing minus raster, PCA line); length = half the
                     board width across the slot (halving joint, ep25 f300 shows both boards flush); width = thickness +
                     0.5 mm; the open edge: the edge the SAM3D gap reaches (piece_2) or, when the SAM3D gap crosses the
                     whole board (piece_1), the half whose projected slot line is dark in ep25 frames 0 and 30
                     (mean intensity 148/140 vs 219/223 on a parallel reference line)
  markers            markers_tip() against the SAM3D-envelope model, re-attached to the thin board
  collision          CoACD (threshold 0.02) of the slotted board + one hull per marker
Mass: bamboo 700 kg/m^3 (ASSUMPTION).
"""
import numpy as np, trimesh, cv2
import manifold3d as m3d
from common import prism_xz, to_trimesh
from fitting import surf, markers_tip, repost, douglas_peucker
from parts import Part, sphere_part, hull

RGBA = '0.85 0.70 0.45 1'
BAMBOO = 700.0; MARKER = 400.0; T_BOARD = 0.008


def fit_board(base, slot_side=None):
    P, N = surf(base, 600000); x, y, z = P.T
    up = N[:, 1] > 0.9; dn = N[:, 1] < -0.9
    y_hi = float(np.percentile(y[up], 90)); y_lo = float(np.percentile(y[dn], 10))
    slab = (y > y_lo - 0.0005) & (y < y_hi + 0.0005)
    X = P[slab][:, [0, 2]]; lo = X.min(0) - 0.006; res = 0.001
    img = np.zeros((int((X[:, 1].max() - lo[1])/res) + 14, int((X[:, 0].max() - lo[0])/res) + 14), np.uint8)
    ij = ((X - lo)/res).astype(int); img[ij[:, 1], ij[:, 0]] = 255
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))   # sample sparsity
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))  # also fills the slot gap
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); c = max(cnts, key=cv2.contourArea)[:, 0, :].astype(float)
    poly = douglas_peucker(np.r_[c, c[:1]], 0.7)[:-1]*res + lo
    if 0.5*np.sum(poly[:, 0]*np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1)*poly[:, 1]) < 0: poly = poly[::-1]  # CCW
    gap = (closed > 0) & (img == 0)
    gap = cv2.morphologyEx(gap.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    gy, gx = np.nonzero(gap); G = np.c_[gx, gy]*res + lo
    mu = G.mean(0); d = np.linalg.svd(G - mu)[2][0]  # slot direction
    tcoord = (G - mu) @ d
    # board extent along the slot line
    s = np.linspace(-0.4, 0.4, 801); line = mu + np.outer(s, d)
    from matplotlib.path import Path as MP
    inside = MP(poly).contains_points(line); s_in = s[inside]; s0, s1 = float(s_in.min()), float(s_in.max())
    g0, g1 = float(tcoord.min()), float(tcoord.max())
    smid = (s0 + s1)/2
    if slot_side is None:  # SAM3D gap reaches one edge only
        slot_side = 'min' if abs(g0 - s0) < abs(g1 - s1) else 'max'
    p = dict(y_env=[y_lo, y_hi], thickness=T_BOARD, y_centre=(y_lo + y_hi)/2, outline=poly.tolist(), slot_point=mu.tolist(),
             slot_dir=d.tolist(), slot_edge_s=[s0, s1], slot_side=slot_side, slot_len=(s1 - s0)/2, slot_width=T_BOARD + 0.0005,
             sam3d_gap_s=[g0, g1])
    model = trimesh.util.concatenate([q.mesh for q in build_board(p, 'high', markers=False)])
    fat = dict(p, thickness=y_hi - y_lo)
    fm = trimesh.util.concatenate([q.mesh for q in build_board(fat, 'high', markers=False)])
    mk = markers_tip(P[::2], fm, thresh=0.004)
    # posted markers protrude 11-24 mm; the 7-9 mm bumps under the corners (mirror of the top markers in the SAM3D
    # hollow box, not visible in any frame) are dropped so the board lies flat, as in ep25 frame 0
    # a blob > 20 mm off the board (piece_2, detached in the SAM3D mesh) is an artifact, not a marker on a base
    p['markers'] = [m for m in mk if 10.0 <= m['protrusion_mm'] <= 20.0 and m['cluster_points'] > 100]
    p['dropped_bumps'] = [m for m in mk if m not in p['markers']]
    repost(p['markers'], model)
    return p


def build_board(p, res='high', markers=True):
    poly = np.array(p['outline']); t = p['thickness']; yc = p['y_centre']
    mu = np.array(p['slot_point']); d = np.array(p['slot_dir']); nrm = np.array([-d[1], d[0]])
    s0, s1 = p['slot_edge_s']; L = p['slot_len']; w = p['slot_width']
    a, b = (s0 - 0.01, s0 + L) if p['slot_side'] == 'min' else (s1 - L, s1 + 0.01)
    slot = np.array([mu + a*d - w/2*nrm, mu + b*d - w/2*nrm, mu + b*d + w/2*nrm, mu + a*d + w/2*nrm])
    cs = m3d.CrossSection([poly]) - m3d.CrossSection([slot])
    board = prism_xz(cs, yc - t/2, yc + t/2)
    import coacd
    coacd.set_log_level('error')
    parts_ = coacd.run_coacd(coacd.Mesh(board.vertices, board.faces), threshold=0.02, max_convex_hull=24, preprocess_mode='off')
    col = [trimesh.Trimesh(v, f).convex_hull for v, f in parts_]
    parts = [Part(board, col, BAMBOO, 'board')]
    if markers:
        for mk in p.get('markers', []):
            parts.append(sphere_part(np.array(mk['centre']), mk['radius'], MARKER, 'marker', 2 if res == 'high' else 1, np.array(mk['post_to'])))
    return parts
