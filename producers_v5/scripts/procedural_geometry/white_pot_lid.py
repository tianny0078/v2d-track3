"""white_pot_lid: recessed enamel lid = revolved thin plate (flat centre, sloped transition, flat outer flange) + bent
strap handle + Vicon markers on posts.

Parameters and how they are obtained (canonical frame of the current SAM3D mesh):
  axis             normal of an SVD plane fit to the flat centre plate (samples with |n.y| > 0.9, 45 < r < 68 mm),
                   oriented towards the handle side
  centre           iterated circle fit (5 rounds) to the outermost rim samples in the plate plane
  mid-surface      per 3 mm radial bin: mean of the median heights of up- and down-facing samples (handle band and
                   bins with < 30 samples excluded), then Douglas-Peucker (0.4 mm) -> revolve control polyline
  R_lid            99.9 percentile of r
  t_plate          2.5 mm (ASSUMPTION; SAM3D plate is ~4.1 mm, enamel steel ~0.6 mm)
  handle           band of samples > 8 mm above the plate within r < 60 mm: direction = PCA, width = P3-P97 lateral,
                   top surface = P95 of h per 4 mm bin of s; strap centre line = top - t_strap/2; t_strap = 3 mm
                   (ASSUMPTION, SAM3D inflates it to ~7.7 mm); feet extended to the plate
  markers          markers_tip() on protrusions of the current mesh outside the marker-free model
Mass: enamel steel 7850 kg/m^3, physical thickness 0.6 mm (plate) / 1.5 mm (strap) -> effective density = 7850*t_phys/t_geo.
"""
import numpy as np, trimesh
from scipy.optimize import least_squares
from common import frame
from fitting import surf, basis, cyl_coords, markers_tip, douglas_peucker, offset_polyline
from parts import revolve_part, sphere_part, sweep_rect_part, merge_hulls

RGBA = '0.93 0.93 0.95 1'
STEEL = 7850.0; T_PLATE = 0.0025; T_PHYS = 0.0006; T_STRAP = 0.003; T_PHYS_STRAP = 0.0015; MARKER = 400.0


def fit(base):
    P, N = surf(base, 400000)
    c0 = P.mean(0); rr = np.linalg.norm((P - c0)[:, [0, 2]], axis=1)
    k = (np.abs(N[:, 1]) > 0.9) & (rr > 0.045) & (rr < 0.068)
    X = P[k]; mu = X.mean(0); a = np.linalg.svd(X - mu)[2][2]
    _, uu, vv = basis(a); c = mu.copy()
    for _ in range(5):
        d = P - c; x = d @ uu; y = d @ vv; r = np.hypot(x, y); R99 = np.percentile(r, 99.5); ko = r > R99 - 0.004
        q = least_squares(lambda q: np.hypot(x[ko]-q[0], y[ko]-q[1]) - q[2], [0, 0, R99], loss='soft_l1', f_scale=0.001).x
        c = c + q[0]*uu + q[1]*vv
    h, r, _ = cyl_coords(P, a, c)
    hi = (np.abs(h) > 0.012) & (r < 0.06)
    if np.median(h[hi]) < 0: a = -a; h = -h  # axis towards the handle side
    hi = (h > 0.012) & (r < 0.06)
    na = N @ a; R_lid = float(np.percentile(r, 99.9))
    Y = P[hi] - c; Yp = Y - np.outer(Y @ a, a); dh = np.linalg.eigh(np.cov(Yp.T))[1][:, -1]
    side = np.cross(a, dh); l = (P - c) @ side; s = (P - c) @ dh
    lc = float(np.median(l[hi])); band = (np.abs(l - lc) < 0.022) & (r < 0.065)
    prof = []
    for r0 in np.arange(0, R_lid, 0.003):
        kk = (r >= r0) & (r < r0 + 0.003) & ~band
        up = kk & (na > 0.3); dn = kk & (na < -0.3)
        if up.sum() > 30 and dn.sum() > 30: prof.append((r0 + 0.0015, (np.median(h[up]) + np.median(h[dn]))/2))
    prof = np.array(prof)
    # extend: plate to the axis at the plate height, flange to R_lid - t/2
    hp = float(np.median(prof[prof[:, 0] < 0.045, 1]))
    prof = prof[prof[:, 0] >= 0.045]
    prof = np.r_[[[0.0, hp]], prof, [[R_lid - T_PLATE/2, prof[-1, 1]]]]
    poly = douglas_peucker(prof, 0.0004)
    poly[0, 1] = hp
    # handle strap: top surface per s bin within the handle band
    hb = (np.abs(l - lc) < 0.012) & (h > hp + 0.004) & (r < 0.065)
    sv = s[hb]; hv = h[hb]; lv = l[hb]
    w = float(np.percentile(lv, 97) - np.percentile(lv, 3)); lc2 = float((np.percentile(lv, 97) + np.percentile(lv, 3))/2)
    path = []
    for s0 in np.arange(np.percentile(sv, 0.5), np.percentile(sv, 99.5), 0.004):
        q = (sv >= s0) & (sv < s0 + 0.004)
        if q.sum() > 20: path.append((s0 + 0.002, np.percentile(hv[q], 95) - T_STRAP/2))
    path = np.array(path)
    plate_top = hp + T_PLATE/2
    path = np.r_[[[path[0, 0], plate_top - 0.001]], path, [[path[-1, 0], plate_top - 0.001]]]
    path = douglas_peucker(path, 0.0008)
    p = dict(axis=a.tolist(), centre=c.tolist(), R_lid=R_lid, mid_profile=poly.tolist(), t_plate=T_PLATE, plate_h=hp,
             flange_h=float(poly[-1, 1]), recess_depth=float(poly[-1, 1] - hp),
             handle_dir=dh.tolist(), handle_side=side.tolist(), handle_width=w, handle_lateral=lc2,
             handle_path_sh=path.tolist(), handle_t=T_STRAP, handle_height_above_plate=float(path[:, 1].max() + T_STRAP/2 - plate_top))
    model = trimesh.util.concatenate([q.mesh for q in build(p, 'high', markers=False)])
    mk = markers_tip(P[::2], model, thresh=0.004)
    # posted markers protrude 17-20 mm; 4 underside bumps of ~4.7 mm (screw ends / SAM3D noise) are dropped
    p['markers'] = [m for m in mk if m['protrusion_mm'] >= 8.0]; p['dropped_bumps'] = [m for m in mk if m['protrusion_mm'] < 8.0]
    return p


def build(p, res='high', markers=True):
    n = 160 if res == 'high' else 48
    a = np.array(p['axis']); c = np.array(p['centre']); dh = np.array(p['handle_dir']); side = np.array(p['handle_side'])
    xf = frame(a, dh, c)
    vis, _ = offset_polyline(np.array(p['mid_profile']), p['t_plate'])
    vis = np.r_[[[0, vis[0, 1]]], vis[1:-1], [[0, vis[-1, 1]]]]  # close on the axis
    coarse = douglas_peucker(np.array(p['mid_profile']), 0.0015)  # collision: coarser profile, 16 wedges per ring
    _, quads = offset_polyline(coarse, p['t_plate'])
    pieces = []
    for i, q in enumerate(quads):
        if i == 0: q = np.array([[0, q[0, 1]], q[1], q[2], [0, q[3, 1]]])
        pieces.append((q, 16))
    body = revolve_part(pieces, xf, n, 24, STEEL*T_PHYS/p['t_plate'], 'lid_plate', visual_profile=vis)
    path3 = np.array([c + dh*s + a*hh + side*p['handle_lateral'] for s, hh in p['handle_path_sh']])
    strap = sweep_rect_part(path3, side, p['handle_width'], p['handle_t'], STEEL*T_PHYS_STRAP/p['handle_t'], 'handle')
    strap.collision = merge_hulls(strap.collision, 3)
    parts = [body, strap]
    if markers:
        for mk in p.get('markers', []):
            parts.append(sphere_part(np.array(mk['centre']), mk['radius'], MARKER, 'marker', 2 if res == 'high' else 1, np.array(mk['post_to'])))
    return parts
