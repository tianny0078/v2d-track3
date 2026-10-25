"""plastic_dish_rack: moulded drainer = rounded-rectangle tray shell (thick rim walls, raised floor, open underside) +
tall rounded-rectangle utensil holder at the -z end (closed base) + 2 rows of tapered round pegs with rounded tips +
Vicon markers.  Canonical frame of the current mesh: y up, z along the length, holder at -z.

Parameters and how they are obtained (current SAM3D mesh samples, outward normals):
  tray footprint   x: P50 of outward-facing (|n_x|>0.8) wall samples in the drain area (z 130..200 mm, y -30..-15);
                   z: P50 of the outward-facing end walls; corner radius: least squares of the rounded-rectangle
                   half-width to the per-z extreme |x| near both ends
  wall thickness   median(outer - inner) of outward / inward facing samples in the same region (both sides + end wall)
  y_bottom, y_rim  y_rim = P90 of up-facing rim samples; y_bottom: start at the SAM3D 0.5 percentile of y, then a 1-D
                   search (2.5 mm steps up to floor top - 2.5 mm) maximising the median multi-view silhouette IoU against
                   the SAM2 masks at the FoundationPose v4 poses (ep14/15/16/28/30, every 6th frame) -- the SAM3D bottom
                   renders below the real bottom edge in every episode
  floor            top = median y of up-facing samples in the drain area (|x| < 60, z 130..200); the tray is modelled
                   as a closed base block from y_bottom to the floor top (SAM3D has a bottom plate at y_bottom and a raised
                   floor; the reservoir between them is not observable -> filled, light effective density)
  holder           outer z-extent from outward-facing end faces; inner opening from inward-facing faces at y 20..50 mm;
                   top = median y of up-facing samples above y=50; inner floor = median y of up-facing samples inside
  pegs             connected clusters of samples above y=-5 mm in the tray (z > -160): centre = centroid of the cluster's
                   slice at y in [-12,-2]; radius at that slice = median distance to the centre; top = max y;
                   top radius = median distance at y_top-6 mm; base on the floor; cone + hemispherical cap
  markers          markers_tip() protrusions outside the marker-free model
Mass: polypropylene 905 kg/m^3; physical wall ~2.5 mm -> effective density on the thick SAM3D-like walls/holder =
905 * 2.5 / t_geo (ASSUMPTION); pegs and floor solid PP at 905 * 0.5 (hollow moulded pegs, ASSUMPTION).
"""
import numpy as np, trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
import scipy.sparse as sp, scipy.sparse.csgraph as cg
import manifold3d as m3d
from common import to_trimesh, extrude
from fitting import surf, markers_tip, repost
from parts import Part, sphere_part, hull

RGBA = '0.95 0.94 0.90 1'
PP = 905.0; T_PHYS = 0.0025; T_FLOOR = 0.005; MARKER = 400.0


def rrect(cx, cz, hw, hl, rc, n=8):
    """rounded rectangle polygon (x, z), CCW, half sizes hw (x) hl (z), corner radius rc."""
    pts = []
    for (sx, sz, a0) in [(1, 1, 0), (-1, 1, np.pi/2), (-1, -1, np.pi), (1, -1, 1.5*np.pi)]:
        ccx, ccz = cx + sx*(hw - rc), cz + sz*(hl - rc)
        for a in np.linspace(a0, a0 + np.pi/2, n):
            pts.append((ccx + rc*np.cos(a), ccz + rc*np.sin(a)))
    return np.array(pts)


def fit(base):
    P, N = surf(base, 600000); x, y, z = P.T
    drain = (z > 0.13) & (z < 0.2) & (y > -0.03) & (y < -0.015)
    xo_p = np.median(x[drain & (N[:, 0] > 0.8) & (x > 0.07)]); xo_m = np.median(x[drain & (N[:, 0] < -0.8) & (x < -0.07)])
    xi_p = np.median(x[drain & (N[:, 0] < -0.8) & (x > 0.06)]); xi_m = np.median(x[drain & (N[:, 0] > 0.8) & (x < -0.06)])
    endw = (np.abs(x - (xo_p + xo_m)/2) < 0.03) & (y > -0.03) & (y < -0.015)
    zo_p = np.median(z[endw & (N[:, 2] > 0.8) & (z > 0.2)]); zi_p = np.median(z[endw & (N[:, 2] < -0.8) & (z > 0.2)])
    hold = (y > 0.02) & (y < 0.05) & (z < -0.15)
    zo_m = np.median(z[hold & (np.abs(x) < 0.03) & (N[:, 2] < -0.8) & (z < -0.22)])
    cx = float((xo_p + xo_m)/2); hw = float((xo_p - xo_m)/2); cz = float((zo_p + zo_m)/2); hl = float((zo_p - zo_m)/2)
    t_wall = float(np.median([xo_p - xi_p, xi_m - xo_m, zo_p - zi_p]))
    y_bot = float(np.percentile(y, 0.5)); rim = (N[:, 1] > 0.8) & (np.abs(x - cx) > hw - t_wall - 0.004) & (z > 0) & (z < 0.2)
    y_rim = float(np.percentile(y[rim], 90))
    # corner radius from the per-z extreme |x| near both ends (mid height band)
    band = (y > -0.04) & (y < -0.02); zs, hws = [], []
    for zz in np.r_[np.arange(zo_m + 0.002, zo_m + 0.06, 0.004), np.arange(zo_p - 0.06, zo_p - 0.002, 0.004)]:
        q = band & (np.abs(z - zz) < 0.0015)
        if q.sum() > 20: zs.append(zz); hws.append(np.percentile(np.abs(x[q] - cx), 99))
    zs, hws = np.array(zs), np.array(hws)
    def res(q):
        rc = q[0]; dz = np.minimum(zs - (cz - hl), (cz + hl) - zs); inside = dz < rc
        return np.where(inside, hw - rc + np.sqrt(np.clip(rc**2 - (rc - dz)**2, 0, None)), hw) - hws
    rc = float(least_squares(res, [0.04], bounds=([0.005], [hw - 0.001]), loss='soft_l1', f_scale=0.001).x[0])
    fl = (np.abs(x - cx) < 0.06) & (z > 0.13) & (z < 0.2) & (N[:, 1] > 0.8)
    y_floor = float(np.median(y[fl]))
    # holder
    zh_out = float(np.median(z[hold & (np.abs(x) < 0.03) & (N[:, 2] > 0.8) & (z > -0.2)]))
    hx_p = np.median(x[hold & (N[:, 0] < -0.8) & (x > cx) & (x < cx + hw - 0.01)]); hx_m = np.median(x[hold & (N[:, 0] > 0.8) & (x < cx) & (x > cx - hw + 0.01)])
    hz_m = np.median(z[hold & (np.abs(x - cx) < 0.03) & (N[:, 2] > 0.8) & (z < -0.2)]); hz_p = np.median(z[hold & (np.abs(x - cx) < 0.03) & (N[:, 2] < -0.8) & (z > -0.2)])
    y_htop = float(np.median(y[(N[:, 1] > 0.8) & (z < -0.15) & (y > 0.05)]))
    hin = (np.abs(x - (hx_p + hx_m)/2) < 0.03) & (z > hz_m + 0.005) & (z < hz_p - 0.005) & (N[:, 1] > 0.8) & (y < 0.0)
    y_hfloor = float(np.median(y[hin]))
    # pegs
    k = (y > -0.005) & (z > zh_out + 0.005) & (np.abs(x - cx) < hw - t_wall)
    X = P[k]; pr = cKDTree(X).query_pairs(0.003, output_type='ndarray')
    G = sp.coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(X), len(X))); n, lab = cg.connected_components(G, directed=False)
    pegs = []
    for i in range(n):
        Y = X[lab == i]
        if len(Y) < 200: continue
        top = Y[:, 1].max(); s0 = Y[(Y[:, 1] < 0.0) & (Y[:, 1] > -0.005)]
        if len(s0) < 20 or top < 0.015: continue
        c0 = s0[:, [0, 2]].mean(0); r0 = float(np.median(np.linalg.norm(s0[:, [0, 2]] - c0, axis=1)))
        s1 = Y[np.abs(Y[:, 1] - (top - 0.006)) < 0.0015]
        r1 = float(np.median(np.linalg.norm(s1[:, [0, 2]] - c0, axis=1))) if len(s1) > 10 else r0*0.6
        pegs.append(dict(x=float(c0[0]), z=float(c0[1]), r_at_y0=r0, y0=-0.0025, top=float(top), r_top=r1))
    pegs.sort(key=lambda d: (round(d['x'] > cx), d['z']))
    p = dict(cx=cx, cz=cz, half_width=hw, half_length=hl, corner_r=rc, t_wall=t_wall, y_bottom=y_bot, y_rim=y_rim,
             y_floor_top=y_floor, t_floor=T_FLOOR, holder_z_outer=zh_out, holder_in_x=[float(hx_m), float(hx_p)],
             holder_in_z=[float(hz_m), float(hz_p)], holder_top=y_htop, holder_floor=y_hfloor, pegs=pegs,
             raw=dict(x_out=[float(xo_m), float(xo_p)], x_in=[float(xi_m), float(xi_p)], z_out=[float(zo_m), float(zo_p)], z_in_end=float(zi_p)))
    model = trimesh.util.concatenate([q.mesh for q in build(p, 'high', markers=False)])
    mk = markers_tip(P[::3], model, thresh=0.004)
    p['markers'] = [m for m in mk if m['protrusion_mm'] >= 8.0]; p['dropped_bumps'] = [m for m in mk if m['protrusion_mm'] < 8.0]
    # bottom height from the multi-view SAM2 masks (FoundationPose v4 poses of ep14/15/16/28/30): the SAM3D rack's lower
    # part renders below the real bottom edge in every episode (the stereo table plane agrees: 27-43 mm below the table);
    # pick the bottom plane (2.5 mm steps, up to the floor top - 2.5 mm) with the best median silhouette IoU
    p['y_bottom_sam3d'] = y_bot; p['bottom_search'] = _fit_bottom(p)
    p['y_bottom'] = p['bottom_search']['best']
    # the two side markers sit at the foot of the side walls in the video; SAM3D puts their centres at y ~ -45 mm,
    # below the re-estimated bottom -> raise them so that they clear the bottom plane by 1 mm (x, z kept), re-attach
    for m in p['markers']:
        c = np.array(m['centre']); lo_ = p['y_bottom'] + m['radius'] + 0.001
        if c[1] < lo_: m['centre_sam3d'] = c.tolist(); c[1] = lo_; m['centre'] = c.tolist(); m['raised_to_bottom'] = True
    repost(p['markers'], trimesh.util.concatenate([q.mesh for q in build(p, 'high', markers=False)]))
    return p


def _prism(poly_xz, y0, y1):
    """vertical prism of a polygon in (x, z) between heights y0..y1 (canonical y up)."""
    cs = m3d.CrossSection([np.asarray(poly_xz, float)])
    man = m3d.Manifold.extrude(cs, y1 - y0)  # along +z_local
    m = to_trimesh(man)
    v = m.vertices.copy(); m.vertices = np.c_[v[:, 0], y0 + v[:, 2], v[:, 1]]
    if m.volume < 0: m.invert()
    return m


def _ring(outer, inner, y0, y1):
    cs = m3d.CrossSection([np.asarray(outer, float)]) - m3d.CrossSection([np.asarray(inner, float)])
    m = to_trimesh(m3d.Manifold.extrude(cs, y1 - y0)); v = m.vertices.copy(); m.vertices = np.c_[v[:, 0], y0 + v[:, 2], v[:, 1]]
    if m.volume < 0: m.invert()
    return m


def _wall_pieces(outer, inner, y0, y1, k=1):
    """convex collision pieces of a rounded-rectangle ring: quad between consecutive outer/inner vertices (merged k at a time)."""
    out = []; n = len(outer)
    for i in range(0, n, k):
        idx = [(i + j) % n for j in range(k + 1)]
        pts = np.r_[outer[idx], inner[idx]]
        out.append(hull(np.r_[np.c_[pts[:, 0], np.full(len(pts), y0), pts[:, 1]], np.c_[pts[:, 0], np.full(len(pts), y1), pts[:, 1]]]))
    return out


def build(p, res='high', markers=True):
    nc = 8 if res == 'high' else 4
    cx, cz, hw, hl, rc, t = p['cx'], p['cz'], p['half_width'], p['half_length'], p['corner_r'], p['t_wall']
    yb, yr, yf, tf = p['y_bottom'], p['y_rim'], p['y_floor_top'], p['t_floor']
    zmin = cz - hl; zh = p['holder_z_outer']
    outer = rrect(cx, cz, hw, hl, rc, nc); inner = rrect(cx, cz, hw - t, hl - t, max(rc - t, 0.002), nc)
    rho_w = PP*T_PHYS/t
    tray = Part(_ring(outer, inner, yb, yr), _wall_pieces(outer, inner, yb, yr, 2 if res == 'high' else 1), rho_w, 'tray_walls')
    # holder: block from the tray's -z end to zh_out, full width, up to holder_top, minus the inner opening
    hcz = (zmin + zh)/2; hhl = (zh - zmin)/2
    hout = rrect(cx, hcz, hw, hhl, min(rc, hhl - 0.001), nc)
    ix0, ix1 = p['holder_in_x']; iz0, iz1 = p['holder_in_z']
    icx, icz, ihw, ihl = (ix0 + ix1)/2, (iz0 + iz1)/2, (ix1 - ix0)/2, (iz1 - iz0)/2
    hin = rrect(icx, icz, ihw, ihl, min(ihw, ihl)*0.9, nc)
    # base block (closed tray bottom up to the floor top; SAM3D has a bottom plate at y_bottom and a raised floor):
    # solid, minus the holder's inner opening below the floor top (the holder pocket goes down to holder_floor)
    hfl = max(p['holder_floor'], yb + 0.003)  # holder pocket floor cannot go below the (re-estimated) bottom
    blk = _prism(outer, yb, yf); pocket = _prism(hin, hfl, yf + 0.002)
    base_vis = _diff(blk, [pocket])
    ftp = outer[outer[:, 1] >= iz1]; fbt = outer[outer[:, 1] <= iz1]
    ftp = np.r_[ftp, [[cx - hw, iz1], [cx + hw, iz1]]]; fbt = np.r_[fbt, [[cx - hw, iz1], [cx + hw, iz1]]]
    base_col = [hull(np.r_[np.c_[ftp[:, 0], np.full(len(ftp), yb), ftp[:, 1]], np.c_[ftp[:, 0], np.full(len(ftp), yf), ftp[:, 1]]]),
                hull(np.r_[np.c_[fbt[:, 0], np.full(len(fbt), yb), fbt[:, 1]], np.c_[fbt[:, 0], np.full(len(fbt), hfl), fbt[:, 1]]])]
    # effective density: two ~2.5 mm PP plates (bottom + floor) spread over the block thickness (ASSUMPTION)
    base = Part(base_vis, base_col, PP*2*T_PHYS/max(yf - yb, 0.005), 'base_block')
    hw_vis = _ring(hout, hin, yb, p['holder_top'])
    hthick = float(np.mean([hw - ihw, hhl - ihl]))
    # inner partner of each outer vertex: clipped into the inner opening's bounding box (sharp inner corners in collision)
    hcol = _wall_pieces(hout, np.array([(icx + np.clip(q[0]-icx, -ihw, ihw), icz + np.clip(q[1]-icz, -ihl, ihl)) for q in hout]),
                        hfl - 0.001, p['holder_top'], 2 if res == 'high' else 1)
    holder = Part(hw_vis, hcol, PP*T_PHYS/max(hthick, 0.005), 'holder')
    parts = [tray, base, holder]
    # pegs: cone frustum from the floor top to top - r_top, plus a hemispherical cap
    nseg = 24 if res == 'high' else 10
    for pg in p['pegs']:
        y0 = yf - 0.001; ytop = pg['top'] - pg['r_top']
        # radius at the floor by linear extrapolation of the taper between (y0_meas, r) and (top-6mm, r_top)
        slope = (pg['r_at_y0'] - pg['r_top'])/max(pg['top'] - 0.006 - pg['y0'], 1e-3)
        rb = pg['r_at_y0'] + slope*(pg['y0'] - y0)
        th = np.linspace(0, 2*np.pi, nseg, endpoint=False)
        ring_b = np.c_[pg['x'] + rb*np.cos(th), np.full(nseg, y0), pg['z'] + rb*np.sin(th)]
        ring_t = np.c_[pg['x'] + pg['r_top']*np.cos(th), np.full(nseg, ytop), pg['z'] + pg['r_top']*np.sin(th)]
        cap = trimesh.creation.icosphere(2 if res == 'high' else 1, pg['r_top']); cap.apply_translation([pg['x'], ytop, pg['z']])
        body = hull(np.r_[ring_b, ring_t, cap.vertices[cap.vertices[:, 1] >= ytop - 1e-9]])
        parts.append(Part(body, [body], PP*0.5, 'peg'))
    if markers:
        for mk in p.get('markers', []):
            parts.append(sphere_part(np.array(mk['centre']), mk['radius'], MARKER, 'marker', 2 if res == 'high' else 1, np.array(mk['post_to'])))
    return parts


def _union(ms):
    from common import union
    return union(ms)


def _diff(a, bs):
    from common import difference
    return difference(a, bs)


def _fit_bottom(p, eps=(14, 15, 16, 28, 30), step=6):
    from silhouette import compare
    cands = list(np.round(np.arange(p['y_bottom'], p['y_floor_top'] - 0.0025 + 1e-9, 0.0025), 5))
    meshes = {c: _union([q.mesh for q in build(dict(p, y_bottom=float(c)), 'low', markers=False)]) for c in cands}
    scores = {c: [] for c in cands}
    for ep in eps:
        _, out = compare(ep, 'plastic_dish_rack', meshes, step=step)
        for c in cands: scores[c] += [x['iou'] for x in out[c]]
    med = {float(c): float(np.median(v)) for c, v in scores.items()}
    best = max(med, key=med.get)
    return dict(candidates_median_iou=med, best=best, frames=len(scores[cands[0]]), episodes=list(eps))
