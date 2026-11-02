"""mini_dust_pan: flat floor plate (trapezoid pan + round handle disc with a hole), perimeter flange walls on both sides
of the floor (the SAM3D section is an I-profile: walls span y -15..+15 mm around the floor at y~0), no wall at the
mouth, a lip strip under the mouth edge, a flange around the handle hole, Vicon markers.
Canonical frame: floor normal y, pan mouth at +z, handle ring at -z.

Parameters and how they are obtained (current SAM3D mesh samples):
  cx, floor y / t      cx = median x; floor = median of up- and down-facing samples with |y| < 5 mm (centre, thickness)
  half-width w(z)      per 4 mm z-slice: P97 of |x - cx| (markers excluded by the percentile)
  handle disc          circle fit to the (+-w(z), z) outline points with z < -20 mm; hole radius = P2 of the in-plane
                       distance of disc-region samples to the disc centre
  pan trapezoid        robust line fit w(z) = a + b z on z in [0, z_front - 3 mm]; z_front = P99.5 of z; it starts at the
                       disc centre (the union gives the neck)
  wall y-range         P3 / P97 of y over samples within 4 mm of the outline (excluding the mouth)
  lip                  samples within 5 mm of the mouth with |x - cx| < 0.6 w: P5..P95 of y, joined to the floor
  t_wall               3 mm (ASSUMPTION; moulded PP flange)
  markers              markers_tip() protrusions outside the marker-free model
Mass: PP 905 kg/m^3 with physical thickness 2 mm on the floor (geometric t_floor) -> effective density; walls 905.
"""
import numpy as np, trimesh
import manifold3d as m3d
from scipy.optimize import least_squares
from common import prism_xz, circle_pts, to_trimesh, union
from fitting import surf, markers_tip, repost
from parts import Part, sphere_part, hull

RGBA = '0.80 0.85 0.15 1'
PP = 905.0; T_WALL = 0.003; T_PHYS = 0.002; MARKER = 400.0


def fit(base):
    P, N = surf(base, 400000); x, y, z = P.T
    cx = float(np.median(x))
    fl = np.abs(y) < 0.005
    yu = np.median(y[fl & (N[:, 1] > 0.8)]); yd = np.median(y[fl & (N[:, 1] < -0.8)])
    yf, tf = float((yu + yd)/2), float(max(yu - yd, 0.0025))  # >= 2.5 mm for contact robustness (SAM3D: 1.9 mm)
    zs = np.arange(z.min() + 0.002, z.max() - 0.001, 0.004); ws = []
    for q in zs:
        k = np.abs(z - q) < 0.0015; ws.append(np.percentile(np.abs(x[k] - cx), 97) if k.sum() > 30 else np.nan)
    ws = np.array(ws); ok = np.isfinite(ws)
    zf = float(np.percentile(z, 99.5)); zb = float(np.percentile(z, 0.5))
    rd = ok & (zs < -0.02); Y = np.r_[np.c_[cx + ws[rd], zs[rd]], np.c_[cx - ws[rd], zs[rd]]]
    q = least_squares(lambda q: np.hypot(Y[:, 0]-q[0], Y[:, 1]-q[1]) - q[2], [cx, -0.04, 0.039], loss='soft_l1', f_scale=0.001).x
    dc = np.array([q[0], q[1]]); Rd = float(q[2])
    reg = np.hypot(x - dc[0], z - dc[1]) < Rd
    Rh = float(np.percentile(np.hypot(x[reg] - dc[0], z[reg] - dc[1]), 2))
    tz = ok & (zs > 0) & (zs < zf - 0.003)
    a, b = least_squares(lambda c: c[0] + c[1]*zs[tz] - ws[tz], [0.035, 0.5], loss='soft_l1', f_scale=0.001).x
    p = dict(cx=cx, floor_y=yf, t_floor=tf, disc_centre=dc.tolist(), disc_R=Rd, hole_R=Rh, trap_a=float(a), trap_b=float(b),
             z_front=zf, z_back=zb, t_wall=T_WALL)
    outline = _outline(p, 64)
    from matplotlib.path import Path as MP
    inside = MP(outline).contains_points(np.c_[x, z])
    # distance to the outline boundary (in x-z)
    from scipy.spatial import cKDTree
    dense = _densify(outline, 0.001); dist = cKDTree(dense).query(np.c_[x, z])[0]
    edge = (dist < 0.004) & (z < zf - 0.006)
    p['wall_y'] = [float(np.percentile(y[edge], 3)), float(np.percentile(y[edge], 97))]
    holeb = np.abs(np.hypot(x - dc[0], z - dc[1]) - Rh) < 0.004
    p['hole_wall_y'] = [float(np.percentile(y[holeb], 3)), float(np.percentile(y[holeb], 97))]
    lip = (z > zf - 0.005) & (np.abs(x - cx) < 0.6*(a + b*zf))
    p['lip_y'] = [float(np.percentile(y[lip], 5)), float(np.percentile(y[lip], 95))]
    model = trimesh.util.concatenate([q.mesh for q in build(p, 'high', markers=False)])
    mk = markers_tip(P[::2], model, thresh=0.004)
    p['markers'] = [m for m in mk if m['protrusion_mm'] >= 6.0 and m['cluster_points'] > 100]
    p['dropped_bumps'] = [m for m in mk if m not in p['markers']]
    repost(p['markers'], model)
    return p


def _densify(poly, step):
    out = []
    for a, b in zip(poly, np.roll(poly, -1, 0)):
        n = max(1, int(np.linalg.norm(b - a)/step)); out.append(a + np.outer(np.arange(n)/n, b - a))
    return np.concatenate(out)


def _outline(p, n):
    """(x, z) outline polygon: disc union trapezoid, CCW."""
    cs = _outline_cs(p, n); pts = np.asarray(cs.to_polygons()[0]); return pts


def _outline_cs(p, n):
    dc = p['disc_centre']; a, b = p['trap_a'], p['trap_b']; cx = p['cx']; z0 = dc[1]; z1 = p['z_front']
    disc = m3d.CrossSection([circle_pts(dc, p['disc_R'], n)])
    trap = m3d.CrossSection([np.array([[cx - (a + b*z0), z0], [cx + (a + b*z0), z0], [cx + (a + b*z1), z1], [cx - (a + b*z1), z1]])])
    return disc + trap


def build(p, res='high', markers=True):
    n = 64 if res == 'high' else 24
    out_cs = _outline_cs(p, n); hole = m3d.CrossSection([circle_pts(p['disc_centre'], p['hole_R'], n)])
    yf, tf, tw = p['floor_y'], p['t_floor'], p['t_wall']
    floor_cs = out_cs - hole
    floor_m = prism_xz(floor_cs, yf - tf/2, yf + tf/2)
    # walls: band inside the outline (minus the mouth strip) + band around the hole
    zf = p['z_front']
    band = out_cs - out_cs.offset(-tw, m3d.JoinType.Miter)
    mouth = m3d.CrossSection([np.array([[-1, zf - tw - 0.0005], [1, zf - tw - 0.0005], [1, 1], [-1, 1.]])])
    band = band - mouth
    hband = hole.offset(tw, m3d.JoinType.Round) - hole
    wy0, wy1 = p['wall_y']; hy0, hy1 = p['hole_wall_y']
    walls = prism_xz(band, wy0, wy1); hwall = prism_xz(hband, hy0, hy1)
    a, b, cx = p['trap_a'], p['trap_b'], p['cx']; w1 = a + b*zf
    ly0 = min(p['lip_y'][0], yf - tf/2); ly1 = yf
    lip = trimesh.creation.box([2*w1 - 2*tw, ly1 - ly0, 0.004]); lip.apply_translation([cx, (ly0 + ly1)/2, zf - 0.002])
    rho_f = PP*T_PHYS/max(tf, 1e-3)
    # collision: trapezoid floor (convex), disc annulus floor as 12 wedges, outline walls as segment hulls, hole flange wedges
    dc = np.array(p['disc_centre']); z0 = dc[1]
    trap = np.array([[cx - (a + b*z0), z0], [cx + (a + b*z0), z0], [cx + w1, zf], [cx - w1, zf]])
    col_floor = [hull(np.r_[np.c_[trap[:, 0], np.full(4, yf - tf/2), trap[:, 1]], np.c_[trap[:, 0], np.full(4, yf + tf/2), trap[:, 1]]])]
    ang = np.linspace(0, 2*np.pi, 13)
    for a0, a1 in zip(ang[:-1], ang[1:]):
        o = circle_pts(dc, p['disc_R'], 3, a0, a1, True); i = circle_pts(dc, p['hole_R'], 3, a0, a1, True); q = np.r_[o, i]
        col_floor.append(hull(np.r_[np.c_[q[:, 0], np.full(6, yf - tf/2), q[:, 1]], np.c_[q[:, 0], np.full(6, yf + tf/2), q[:, 1]]]))
        i2 = circle_pts(dc, p['hole_R'] + tw, 3, a0, a1, True); q = np.r_[i, i2]
        col_floor.append(hull(np.r_[np.c_[q[:, 0], np.full(6, hy0), q[:, 1]], np.c_[q[:, 0], np.full(6, hy1), q[:, 1]]]))
    col_walls = []
    ol = np.asarray(out_cs.to_polygons()[0]); inner = np.asarray(out_cs.offset(-tw, m3d.JoinType.Miter).to_polygons()[0])
    from scipy.spatial import cKDTree
    tr = cKDTree(inner)
    segs = [(ol[i], ol[(i+1) % len(ol)]) for i in range(len(ol))]
    k = 3 if res == 'high' else 1
    for s0 in range(0, len(segs), k):
        zc = zf - tw; pts = []
        for j in range(s0, s0 + k):  # clip each segment to z <= zf - tw (no wall at the mouth)
            a0, a1 = segs[j % len(segs)]
            if a0[1] > zc and a1[1] > zc: continue
            if a0[1] > zc: a0 = a1 + (a0 - a1)*(zc - a1[1])/(a0[1] - a1[1])
            if a1[1] > zc: a1 = a0 + (a1 - a0)*(zc - a0[1])/(a1[1] - a0[1])
            pts += [a0, a1]
        if len(pts) < 2: continue
        pts = np.array(pts)
        inn = inner[tr.query(pts)[1]]; q = np.r_[pts, inn]
        col_walls.append(hull(np.r_[np.c_[q[:, 0], np.full(len(q), wy0), q[:, 1]], np.c_[q[:, 0], np.full(len(q), wy1), q[:, 1]]]))
    parts = [Part(floor_m, col_floor, rho_f, 'floor'), Part(union([walls, hwall]), col_walls, PP, 'walls'), Part(lip, [lip], PP, 'lip')]
    if markers:
        for mk in p.get('markers', []):
            parts.append(sphere_part(np.array(mk['centre']), mk['radius'], MARKER, 'marker', 2 if res == 'high' else 1, np.array(mk['post_to'])))
    return parts
