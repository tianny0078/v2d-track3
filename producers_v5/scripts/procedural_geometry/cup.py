"""Tapered open cup (beige_cup, blue_cup): truncated-cone shell + flat bottom + small bottom-edge chamfer + Vicon markers.

Parameters (all fitted on the current canonical mesh, see fit()):
  axis, centre        surface-of-revolution axis: fit_revolution() on outward-facing wall samples (middle section only)
  h0, h1              bottom plane / rim height along the axis: 0.5 / 99.5 percentile of h over non-marker samples
  r = a + b*h         outer wall line: robust linear fit to outward-facing wall samples between h0+15 mm and h1-10 mm
  t_wall              median (outer line - inward-facing wall sample radius) in the middle section
  t_bottom            median h of upward-facing floor samples (r < 0.6 R) minus h0
  chamfer_bottom      3 mm (assumption: moulded cup edge; not resolvable in the SAM3D mesh)
  markers             protrusion clusters of the current mesh outside the marker-free model (> 4 mm), sphere-fitted
"""
import numpy as np
from common import frame
from fitting import surf, fit_revolution, cyl_coords, marker_spheres
from parts import revolve_part, sphere_part, Part
from scipy.optimize import least_squares

PP = 905.0       # polypropylene kg/m^3 (assumption: hard plastic tumbler)
MARKER = 400.0   # Vicon marker effective density (assumption: ~0.5-1 g plastic/foam sphere + base)


def _line(h, r):
    q = least_squares(lambda q: q[0] + q[1]*h - r, [np.median(r), 0], loss='soft_l1', f_scale=0.0005).x
    return float(q[0]), float(q[1])


def fit_cup(base, chamfer=0.003):
    P, N = surf(base, 300000)
    c0 = base.bounds.mean(0); c0[1] = 0
    pc = np.linalg.eigh(np.cov((P - P.mean(0)).T))[1]
    axis0 = pc[:, np.argmax([abs(v[1]) for v in pc.T])]
    axis, c, rms = fit_revolution(P, N, axis0, c0, hrange=(-0.06, 0.06))
    h, r, _ = cyl_coords(P, axis, c)
    # closed end = the end where samples near the axis exist (the floor)
    Rm = np.median(r); near = r < 0.5*Rm
    if np.median(h[near]) > 0: axis = -axis; h = -h
    # outer line, using outward-facing samples in the middle
    d = P - c; radial = d - np.outer(d @ axis, axis); rn = np.einsum('ij,ij->i', radial, N)/np.maximum(r, 1e-9)
    hlo, hhi = np.percentile(h, [1, 99])
    mid = (h > hlo + 0.015) & (h < hhi - 0.010)
    a, b = _line(h[mid & (rn > 0.6)], r[mid & (rn > 0.6)])
    nonmarker = np.abs(r - (a + b*h)) < 0.004
    nonmarker |= near
    h0 = float(np.percentile(h[nonmarker], 0.5)); h1 = float(np.percentile(h[nonmarker], 99.5))
    inner = mid & (rn < -0.6)
    t_wall = float(np.median((a + b*h[inner]) - r[inner]))
    na = N @ axis; floor = (na > 0.8) & (r < 0.6*(a + b*h0)) & (h < h0 + 0.02)
    t_bottom = float(np.median(h[floor]) - h0) if floor.sum() > 50 else t_wall
    x_hint = np.cross(axis, [0, 0, 1.]) if abs(axis[2]) < 0.9 else np.cross(axis, [1., 0, 0])
    params = dict(axis=axis.tolist(), centre=c.tolist(), x_hint=x_hint.tolist(), h0=h0, h1=h1, r_a=a, r_b=b,
                  t_wall=t_wall, t_bottom=t_bottom, chamfer_bottom=chamfer, axis_fit_rms_mm=rms,
                  r_bottom_out=a + b*h0, r_top_out=a + b*h1, height=h1 - h0)
    # markers: protrusions of the current mesh outside the marker-free model
    model = build_cup(params, 'high', markers=False)[0].mesh
    ms = marker_spheres(P[::2], model)
    params['markers'] = []
    for ctr, rad, cl in ms:
        hh, rr, _ = cyl_coords(ctr[None], axis, c)
        Rw = a + b*np.clip(hh[0], h0, h1)
        foot = c + axis*np.clip(hh[0], h0 + 0.002, h1 - 0.002) + (ctr - c - axis*hh[0])/max(rr[0], 1e-9)*(Rw - 0.001)
        params['markers'].append(dict(centre=ctr.tolist(), radius=rad, post_to=foot.tolist(), cluster_points=int(cl['n'])))
    return params


def build_cup(p, res='high', markers=True):
    n = 128 if res == 'high' else 40; nseg = 20
    axis = np.array(p['axis']); xf = frame(axis, p['x_hint'], np.array(p['centre']))
    R = lambda hh: p['r_a'] + p['r_b']*hh
    h0, h1, t, tb, f = p['h0'], p['h1'], p['t_wall'], max(p['t_bottom'], p['chamfer_bottom'] + 0.0005), p['chamfer_bottom']
    bottom = [(0, h0), (R(h0) - f, h0), (R(h0 + f), h0 + f), (R(h0 + tb), h0 + tb), (0, h0 + tb)]
    wall = [(R(h0 + f) - t, h0 + f), (R(h0 + f), h0 + f), (R(h1), h1), (R(h1) - t, h1)]
    parts = [revolve_part([bottom, wall], xf, n, nseg, PP, 'cup_shell')]
    if markers:
        for mk in p.get('markers', []):
            parts.append(sphere_part(np.array(mk['centre']), mk['radius'], MARKER, 'marker', 2 if res == 'high' else 1, np.array(mk['post_to'])))
    return parts
