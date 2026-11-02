"""white_pot: enamel milk pan = cylindrical shell with rounded bottom edge + flat stamped handle with an end hole + Vicon markers.

Parameters and how they are obtained (canonical frame of the current SAM3D mesh):
  axis, centre     fit_revolution() on outward-facing wall samples, straight section only (pot axis ~ canonical y)
  h0 / h1          bottom plane / rim: 0.5 / 99.5 percentile of h over body samples (closed end = side with floor samples)
  R                outer radius: median outward-facing r over the straight section
  r_fillet         bottom-edge radius: 1-D least squares of a quarter-circle fillet to the outward-facing profile near h0
  t_wall           2.5 mm (ASSUMPTION; SAM3D wall is ~6.5 mm thick, enamel steel is ~0.6-1 mm; 2.5 mm keeps contacts robust)
  handle           direction = mean radial direction of samples beyond R+4 mm; width = P95-P5 of lateral coordinate
                   on the middle of the handle; centre height = middle of the SAM3D handle tube (P5/P95 of h); end = max
                   radial reach; thickness 4 mm (ASSUMPTION from video: flat stamped strap; SAM3D inflates it to a 30 mm tube);
                   rounded end radius = width/2, hole r = 5 mm concentric with the end round (video, ep5 frame 0)
  markers          protrusions of the current mesh outside the marker-free model, sphere-fitted
Mass: enamel steel 7850 kg/m^3 with physical thickness 0.8 mm (body) / 1.5 mm (handle) -> effective density on the
thicker geometric shell = 7850 * t_phys / t_geo (ASSUMPTION, +-50 %).
"""
import numpy as np
from scipy.optimize import least_squares
from common import frame
from fitting import surf, fit_revolution, cyl_coords, marker_spheres, protrusions
from parts import revolve_part, sphere_part, Part, extrude_part, hull
import trimesh

RGBA = '0.93 0.93 0.95 1'
STEEL = 7850.0; T_WALL = 0.0025; T_PHYS_BODY = 0.0008; T_HANDLE = 0.004; T_PHYS_HANDLE = 0.0015; MARKER = 400.0


def fit(base):
    P, N = surf(base, 400000)
    # coarse: body circle centre from the x/z extents of the samples on the body side (handle points far along +-z)
    c0 = np.array([0., 0., 0.]); h, r, _ = cyl_coords(P, [0, 1, 0], c0)
    # first guess of the centre: median of samples whose |z| span is body-like
    zc = np.median(P[:, 2]); c0 = np.array([np.median(P[:, 0]), 0, zc])
    body0 = np.linalg.norm((P - c0)[:, [0, 2]], axis=1) < 0.095  # drop the handle (it would dominate the residual)
    axis, c, rms = fit_revolution(P[body0], N[body0], np.array([0, 1., 0]), c0, hrange=(-0.04, 0.035))
    for _ in range(2):  # refit once the centre is right (radial normal test depends on it)
        h_, r_, _ = cyl_coords(P, axis, c); b_ = r_ < 0.095
        axis, c, rms = fit_revolution(P[b_], N[b_], axis, c, hrange=(-0.04, 0.035))
    h, r, th = cyl_coords(P, axis, c)
    d = P - c; radial = d - np.outer(d @ axis, axis); rn = np.einsum('ij,ij->i', radial, N)/np.maximum(r, 1e-9)
    # the SAM3D section is slightly elliptical (85.0 .. 89.0 mm by sector); the real pan is round -> R = mean of the
    # 12 sector medians of the outward-facing wall radius (sector averaging removes sampling-density bias)
    th_ = cyl_coords(P, axis, c)[2]; kk = (rn > 0.6) & (np.abs(h) < 0.035) & (r < 0.1)
    sect = [np.median(r[kk & (th_ >= a) & (th_ < a + np.pi/6)]) for a in np.linspace(-np.pi, np.pi, 13)[:-1]]
    R = float(np.mean(sect)); R_sector_minmax = [float(min(sect)), float(max(sect))]
    body = r < R + 0.004
    near = body & (r < 0.5*R)
    if np.median(h[near]) > 0: axis = -axis; h = -h  # axis: bottom -> rim
    h0 = float(np.percentile(h[body], 0.5)); h1 = float(np.percentile(h[body], 99.5))
    # bottom fillet: outer profile r(h) near h0, model R - rf + sqrt(rf^2 - (rf - (h-h0))^2)
    k = body & (rn > 0.5) & (h < h0 + 0.03)
    def f(q):
        rf = q[0]; z = np.clip(h[k] - h0, 0, None); inside = z < rf
        model = np.where(inside, R - rf + np.sqrt(np.clip(rf**2 - (rf - z)**2, 0, None)), R)
        return model - r[k]
    rf = float(least_squares(f, [0.015], bounds=([0.003], [0.04]), loss='soft_l1', f_scale=0.001).x[0])
    # handle: samples beyond the body radius
    out = (r > R + 0.004) & (h > h0 + 0.01) & (h < h1 - 0.005)
    dirv = radial[out]/r[out, None]; dh = dirv.mean(0); dh -= np.dot(dh, axis)*axis; dh /= np.linalg.norm(dh)
    side = np.cross(axis, dh)
    s = d @ dh; l = d @ side
    hk = out & (np.abs(l) < 0.05) & (s > R)
    smax = float(np.percentile(s[hk], 99.9))
    mid = hk & (s > R + 0.02) & (s < smax - 0.045)
    w = float(np.percentile(l[mid], 97) - np.percentile(l[mid], 3))
    lc = float((np.percentile(l[mid], 97) + np.percentile(l[mid], 3))/2)
    hc_lo, hc_hi = np.percentile(h[mid], [3, 97]); hc = float((hc_lo + hc_hi)/2)
    # slope of the handle centre line along s (top/bottom midline per s-bin)
    sb = np.linspace(R + 0.02, smax - 0.045, 8); mids = []
    for a, b in zip(sb[:-1], sb[1:]):
        q = mid & (s >= a) & (s < b)
        if q.sum() > 30: mids.append(((a+b)/2, (np.percentile(h[q], 97) + np.percentile(h[q], 3))/2))
    mids = np.array(mids); slope = float(np.polyfit(mids[:, 0], mids[:, 1], 1)[0]) if len(mids) > 2 else 0.0
    p = dict(axis=axis.tolist(), centre=c.tolist(), axis_fit_rms_mm=rms, R=R, R_sector_minmax=R_sector_minmax, h0=h0, h1=h1, height=h1 - h0, r_fillet=rf,
             t_wall=T_WALL, handle_dir=dh.tolist(), handle_side=side.tolist(), handle_width=w, handle_lateral_offset=lc,
             handle_h_centre=hc, handle_h_tube_range=[float(hc_lo), float(hc_hi)], handle_slope=slope,
             handle_reach=smax, handle_thickness=T_HANDLE, handle_hole_r=0.005)
    model = trimesh.util.concatenate([q.mesh for q in build(p, 'high', markers=False)])
    p['markers'] = []
    for ctr, rad, cl in marker_spheres(P[::2], model, thresh=0.004):
        # post: towards the closest point of the marker-free model
        cp, dist, _ = trimesh.proximity.closest_point(model, ctr[None])
        foot = cp[0] + (cp[0] - ctr)/max(np.linalg.norm(cp[0] - ctr), 1e-9)*0.001
        p['markers'].append(dict(centre=ctr.tolist(), radius=rad, post_to=foot.tolist(), cluster_points=int(cl['n'])))
    return p


def profile(p, nf):
    """outer/inner revolve profile (r, h) of the shell with a bottom fillet discretised by nf segments.
    Returns visual polygon and convex collision pieces."""
    R, h0, h1, rf, t = p['R'], p['h0'], p['h1'], p['r_fillet'], p['t_wall']
    ri = max(rf - t, 0.001)
    ang = np.linspace(-np.pi/2, 0, nf + 1)
    outer = [(R - rf + rf*np.cos(a), h0 + rf + rf*np.sin(a)) for a in ang]      # from bottom to the side
    inner = [(R - rf + ri*np.cos(a), h0 + rf + ri*np.sin(a)) for a in ang]
    vis = [(0, h0)] + outer + [(R, h1), (R - t, h1)] + inner[::-1] + [(0, h0 + t)]
    pieces = [[(0, h0), (R - rf, h0), (R - rf, h0 + t), (0, h0 + t)]]
    for i in range(nf):
        pieces.append(([inner[i], outer[i], outer[i+1], inner[i+1]], 16))
    pieces.append(([(R - t, h0 + rf), (R, h0 + rf), (R, h1), (R - t, h1)], 24))
    return vis, pieces


def build(p, res='high', markers=True):
    n = 160 if res == 'high' else 44; nf = 8 if res == 'high' else 4
    axis = np.array(p['axis']); c = np.array(p['centre']); dh = np.array(p['handle_dir']); side = np.array(p['handle_side'])
    xf = frame(axis, dh, c)
    vis, pieces = profile(p, nf)
    # collision: 4 fillet bands are enough (merge visual bands pairwise when nf=8)
    _, cpieces = profile(p, 2)
    body = revolve_part(cpieces, xf, n, 24, STEEL*T_PHYS_BODY/p['t_wall'], 'body', visual_profile=vis)
    # handle: flat strap in the plane spanned by (dh, side), centre line h = hc + slope*(s - s_mid)
    R = p['R']; w = p['handle_width']; t = p['handle_thickness']; reach = p['handle_reach']; rend = w/2
    s0 = R - p['t_wall']/2; s_end = reach - rend
    hc = p['handle_h_centre']; sl = p['handle_slope']; smid = (R + reach)/2
    # local 2D frame of the strap: u along handle (dh tilted by slope), v = side; extrude along normal
    u = dh + sl*axis; u /= np.linalg.norm(u); nrm = np.cross(u, side); nrm /= np.linalg.norm(nrm)
    org = c + axis*(hc + sl*(s0 - smid)) + dh*s0 + side*p['handle_lateral_offset']
    L = (s_end - s0)/np.dot(u, dh)
    th = np.linspace(-np.pi/2, np.pi/2, 17 if res == 'high' else 9)
    poly = [(0, -w/2), (L, -w/2)] + [(L + rend*np.cos(a), rend*np.sin(a)) for a in th[1:-1]] + [(L, w/2), (0, w/2)]
    M = np.eye(4); M[:3, 0] = u; M[:3, 1] = side; M[:3, 2] = nrm; M[:3, 3] = org - nrm*t/2
    hole_n = 24 if res == 'high' else 10
    hole = [(L + 0.005*np.cos(a), 0.005*np.sin(a)) for a in np.linspace(0, 2*np.pi, hole_n, endpoint=False)]
    import manifold3d as m3d
    from common import to_trimesh
    cs = m3d.CrossSection([np.array(poly)]) - m3d.CrossSection([np.array(hole)])
    hm = to_trimesh(m3d.Manifold.extrude(cs, t)); hm.apply_transform(M)
    from common import extrude
    col = [extrude([(0, -w/2), (L, -w/2), (L, w/2), (0, w/2)], t, 0.0, M).convex_hull,
           extrude([(L + rend*np.cos(a), rend*np.sin(a)) for a in th], t, 0.0, M).convex_hull]
    handle = Part(hm, col, STEEL*T_PHYS_HANDLE/t, 'handle')
    parts = [body, handle]
    if markers:
        for mk in p.get('markers', []):
            parts.append(sphere_part(np.array(mk['centre']), mk['radius'], MARKER, 'marker', 2 if res == 'high' else 1, np.array(mk['post_to'])))
    return parts
