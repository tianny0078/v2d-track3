"""mini_sweeper: hand brush = wooden ring handle (torus) + yellow plastic body (rounded box) + bristle fan (tapered
trapezoid prism) + Vicon markers. Canonical frame: long axis z (ring at -z, bristles at +z), ring/fan plane = y-z.

Parameters and how they are obtained:
  ring centre/outer radius   circle fit (y-z plane) to the outermost ring samples (z < z_neck); x centre = median x
  ring hole radius           P2 of the y-z distance of ring samples to the centre (hole is present in the SAM3D mesh)
  ring minor radius          (outer - hole)/2, clamped to <= 7 mm: SAM3D inflates the ring's x-thickness to ~26 mm; the
                             edge-on video frames (ep19/ep30 frame 0) show a ring cross-section ~0.27 x the outer diameter
  body (neck) box            P3/P97 extents of the neck slab (between the ring top and the fan start) in x and y
  fan z-range                fan start = end of the neck slab (minimum of the per-z sample count), tip = P99.5 of z
  fan base width             = body y-width (mesh); fan tip width = 1.0 x ring outer diameter: ratio measured on the
                             face-on frames ep30 f87 / ep19 f156 (fan tip and ring are coplanar, ~same depth); the SAM3D
                             fan (41 mm wide, 36 mm thick, a box) was reconstructed from an edge-on view
  fan thickness              14 mm at the base -> 8 mm at the tip (ASSUMPTION from the edge-on frames)
  fan centre (x, y)          centre of the SAM3D fan block's P3/P97 extents
  markers                    markers_tip() on protrusions outside the marker-free model
Mass: beech ring 700 kg/m^3; PP body 905 (solid); bristle fan 350 (nylon tufts ~40 % packing, ASSUMPTION).
"""
import numpy as np, trimesh
from scipy.optimize import least_squares
from common import union
from fitting import surf, markers_tip, repost
from parts import Part, torus_part, sphere_part, hull, loft_part
from common import frame

RGBA = '0.85 0.75 0.35 1'
WOOD = 700.0; PP = 905.0; BRISTLE = 350.0; MARKER = 400.0


def fit(base):
    P, N = surf(base, 300000); x, y, z = P.T
    zs = np.arange(z.min() + 0.002, z.max(), 0.002); cnt = np.array([(np.abs(z - q) < 0.001).sum() for q in zs])
    # neck = local minimum of the slice count between the ring and the fan (search z in [-35, 0] mm)
    win = (zs > -0.035) & (zs < 0.0); zn = zs[win][np.argmin(cnt[win])]
    neck = (z > zn - 0.008) & (z < zn + 0.008)
    # neck slab extents along z: where the slice width in y stays below the ring/fan widths
    ring = z < zn - 0.008
    yr, zr = y[ring], z[ring]
    def circ(q): return np.hypot(yr - q[0], zr - q[1]) - q[2]
    far = np.hypot(yr - np.median(yr), zr - np.median(zr)); sel = far > np.percentile(far, 70)
    q = least_squares(lambda q: (np.hypot(yr[sel]-q[0], zr[sel]-q[1]) - q[2]), [np.median(yr), np.median(zr), 0.026], loss='soft_l1', f_scale=0.001).x
    yc, zc, Ro = float(q[0]), float(q[1]), float(q[2])
    # outer radius = circle fit to the outer 30 % of ring samples (the SAM3D ring is elongated along z towards the neck;
    # P99.5 of the distance would pick that), hole = P2 of the distance
    d = np.hypot(yr - yc, zr - zc); Rin = float(np.percentile(d, 2))
    xc = float(np.median(x[ring]))
    rminor = min((Ro - Rin)/2, 0.007); Rmaj = Ro - rminor
    # body: the slab from the ring top (zc + Ro - small) to the fan start
    fan = z > zn + 0.008
    z_fan0 = float(zn + 0.006); z_tip = float(np.percentile(z, 99.5))
    bx = np.percentile(x[neck], [3, 97]); by = np.percentile(y[neck], [3, 97])
    fx = np.percentile(x[fan & (z > z_fan0 + 0.02)], [3, 97]); fy = np.percentile(y[fan & (z > z_fan0 + 0.02)], [3, 97])
    p = dict(ring_centre=[xc, yc, zc], ring_R_outer=Ro, ring_R_hole=Rin, ring_minor=rminor, ring_major=Rmaj,
             z_neck=float(zn), body_x=[float(bx[0]), float(bx[1])], body_y=[float(by[0]), float(by[1])], body_z=[float(zc + Rmaj), z_fan0 + 0.01],
             fan_z=[z_fan0, z_tip], fan_centre_xy=[float(fx.mean()), float(fy.mean())], fan_base_width=float(by[1] - by[0]),
             fan_tip_width=float(2*Ro*1.0), fan_t_base=0.014, fan_t_tip=0.008,
             sam3d_fan_extent=dict(x=[float(fx[0]), float(fx[1])], y=[float(fy[0]), float(fy[1])]))
    # markers are detected against a 'fat' model whose fan has the SAM3D block extents (otherwise the whole SAM3D fan
    # block is one big protrusion cluster and swallows the markers next to it)
    fat = dict(p, fan_t_base=float(fx[1]-fx[0]), fan_t_tip=float(fx[1]-fx[0]), fan_base_width=float(fy[1]-fy[0]),
               fan_tip_width=float(fy[1]-fy[0]), fan_centre_xy=[float(fx.mean()), float(fy.mean())])
    model = trimesh.util.concatenate([q.mesh for q in build(fat, 'high', markers=False)])
    mk = markers_tip(P[::2], model, thresh=0.004)
    p['markers'] = [m for m in mk if m['protrusion_mm'] >= 6.0 and m['cluster_points'] > 150]
    p['dropped_bumps'] = [m for m in mk if not (m['protrusion_mm'] >= 6.0 and m['cluster_points'] > 150)]
    repost(p['markers'], trimesh.util.concatenate([q.mesh for q in build(p, 'high', markers=False)]))
    return p


def build(p, res='high', markers=True):
    xc, yc, zc = p['ring_centre']
    # torus in the y-z plane: local frame z_local = x axis
    xf = frame([1, 0, 0], [0, 1, 0], np.array([xc, yc, zc]))
    ring = torus_part(p['ring_major'], p['ring_minor'], xf, WOOD, 'ring', 48 if res == 'high' else 20, 16 if res == 'high' else 8, 12)
    bx, by, bz = p['body_x'], p['body_y'], p['body_z']
    body_m = trimesh.creation.box([bx[1]-bx[0], by[1]-by[0], bz[1]-bz[0]]); body_m.apply_translation([np.mean(bx), np.mean(by), np.mean(bz)])
    body = Part(body_m, [body_m], PP, 'body')
    fx, fy = p['fan_centre_xy']; z0, z1 = p['fan_z']; w0, w1 = p['fan_base_width'], p['fan_tip_width']; t0, t1 = p['fan_t_base'], p['fan_t_tip']
    def sec(zz, w, t):
        return np.array([[fx - t/2, fy - w/2, zz], [fx + t/2, fy - w/2, zz], [fx + t/2, fy + w/2, zz], [fx - t/2, fy + w/2, zz]])
    fan_m = hull(np.r_[sec(z0, w0, t0), sec(z1, w1, t1)])
    zm = (z0 + z1)/2
    fan_col = [hull(np.r_[sec(z0, w0, t0), sec(zm, (w0+w1)/2, (t0+t1)/2)]), hull(np.r_[sec(zm, (w0+w1)/2, (t0+t1)/2), sec(z1, w1, t1)])]
    fan = Part(fan_m, fan_col, BRISTLE, 'bristles')
    parts = [ring, body, fan]
    if markers:
        for mk in p.get('markers', []):
            parts.append(sphere_part(np.array(mk['centre']), mk['radius'], MARKER, 'marker', 2 if res == 'high' else 1, np.array(mk['post_to'])))
    return parts
