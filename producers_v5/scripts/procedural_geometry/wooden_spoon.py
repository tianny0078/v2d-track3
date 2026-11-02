"""wooden_spoon: loft of super-elliptic cross-sections along the spoon axis (canonical z): flat paddle bowl + handle,
plus Vicon markers.

Parameters and how they are obtained (current SAM3D mesh samples, per 4 mm z-slice):
  outline             x-centre and half-width from P3/P97 of x (markers removed by the percentiles), smoothed (5-slice median)
  y-centre / thickness from P3/P97 of y; thickness clamped to <= 12 mm (ASSUMPTION: SAM3D inflates the handle to ~23 mm
                      in the unobserved depth direction; wooden spoon handles are 8-12 mm); the bowl keeps its measured
                      7-8 mm (it is a flat paddle in the SAM3D mesh, no concavity is modelled because its side is not
                      determinable from the mesh)
  ends                z range = P0.3 / P99.7 of z over non-marker samples; the first/last section shrink to 30 %
  cross-section       superellipse exponent 2.5 (rounded rectangle), 24 vertices (high) / 12 (low)
  markers             markers_tip() protrusions outside the SAM3D-thickness model: none qualifies -> no markers
Mass: beech 700 kg/m^3 (solid wood, ASSUMPTION).
"""
import numpy as np, trimesh
from scipy.ndimage import median_filter
from fitting import surf, markers_tip, repost
from parts import Part, sphere_part, loft_part, merge_hulls

RGBA = '0.80 0.62 0.42 1'
WOOD = 700.0; MARKER = 400.0; T_MAX = 0.012


def fit(base):
    P, N = surf(base, 300000); x, y, z = P.T
    z0, z1 = np.percentile(z, [0.3, 99.7]); zs = np.arange(z0 + 0.002, z1 - 0.001, 0.004)
    rows = []
    for q in zs:
        k = np.abs(z - q) < 0.002
        if k.sum() < 30: continue
        xl, xh = np.percentile(x[k], [3, 97]); yl, yh = np.percentile(y[k], [3, 97])
        rows.append((q, (xl + xh)/2, (xh - xl)/2, (yl + yh)/2, (yh - yl)/2))
    S = np.array(rows)
    for c in range(1, 5): S[:, c] = median_filter(S[:, c], 5, mode='nearest')
    t_meas = S[:, 4].copy(); S[:, 4] = np.minimum(S[:, 4], T_MAX/2)
    p = dict(z_range=[float(z0), float(z1)], sections=S.tolist(), thickness_clamp=T_MAX, superellipse_n=2.5,
             measured_half_thickness_max=float(t_meas.max()), columns='z, x_centre, half_width, y_centre, half_thickness')
    model = trimesh.util.concatenate([q.mesh for q in build(p, 'high', markers=False)])
    # markers: against a model with the SAM3D thickness (else the inflated handle is one big protrusion)
    fat = dict(p, sections=np.c_[S[:, :4], t_meas].tolist())
    fm = trimesh.util.concatenate([q.mesh for q in build(fat, 'high', markers=False)])
    mk = markers_tip(P[::2], fm, thresh=0.004)
    # no protrusion passes the marker rule (>= 6 mm, > 100 samples): the SAM3D spoon markers are merged into the
    # inflated handle (P97 thickness); the only candidates are 4.5-5.9 mm bumps on the underside -> markers omitted
    p['markers'] = [m for m in mk if m['protrusion_mm'] >= 6.0 and m['cluster_points'] > 100]
    p['dropped_bumps'] = [m for m in mk if m not in p['markers']]
    repost(p['markers'], model)
    return p


def section(zc, xc, a, yc, b, k, n=2.5):
    th = np.linspace(0, 2*np.pi, k, endpoint=False); c, s = np.cos(th), np.sin(th)
    return np.c_[xc + a*np.sign(c)*np.abs(c)**(2/n), yc + b*np.sign(s)*np.abs(s)**(2/n), np.full(k, zc)]


def build(p, res='high', markers=True):
    S = np.array(p['sections']); k = 24 if res == 'high' else 12
    if res != 'high': S = S[::2] if (len(S) - 1) % 2 == 0 else np.r_[S[::2], S[-1:]]
    secs = []
    for i, (zc, xc, a, yc, b) in enumerate(S):
        f = 0.3 if i in (0, len(S) - 1) else 1.0
        secs.append(section(zc, xc, max(a*f, 0.001), yc, max(b*(0.6 if f < 1 else 1), 0.001), k, p['superellipse_n']))
    body = loft_part(secs, WOOD, 'spoon')
    body.collision = merge_hulls(body.collision, 4 if res == 'high' else 2)
    parts = [body]
    if markers:
        for mk in p.get('markers', []):
            parts.append(sphere_part(np.array(mk['centre']), mk['radius'], MARKER, 'marker', 2 if res == 'high' else 1, np.array(mk['post_to'])))
    return parts
