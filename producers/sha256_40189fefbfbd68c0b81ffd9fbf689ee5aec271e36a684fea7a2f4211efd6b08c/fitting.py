"""Parameter estimation from the current (SAM3D) mesh point cloud, in its canonical frame."""
import numpy as np, trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


def surf(m, n=200000, seed=0):
    """surface samples + outward face normals of mesh m."""
    p, fi = trimesh.sample.sample_surface(m, n, seed=seed)
    return np.asarray(p), np.asarray(m.face_normals[fi])


def basis(axis):
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    t = np.array([1., 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1., 0])
    u = np.cross(a, t); u /= np.linalg.norm(u); v = np.cross(a, u); return a, u, v


def cyl_coords(P, axis, c):
    a, u, v = basis(axis); d = P - c
    h = d @ a; x, y = d @ u, d @ v
    return h, np.hypot(x, y), np.arctan2(y, x)


def tilt(axis0, ab):
    a, u, v = basis(axis0); w = a + ab[0]*u + ab[1]*v; return w/np.linalg.norm(w)


def fit_revolution(P, N, axis0, c0, hrange=None, outer=True, bin_mm=2.0):
    """refine axis direction + centre of a surface of revolution: minimise the spread of r within height bins
    (outer=True: only points whose normal points away from the axis). Returns axis, c (a point on the axis)."""
    a0, u0, v0 = basis(axis0)

    def sel(axis, c):
        h, r, _ = cyl_coords(P, axis, c); d = P - c; radial = d - np.outer(d @ axis, axis)
        rn = np.einsum('ij,ij->i', radial, N)/np.maximum(r, 1e-9)
        k = rn > 0.6 if outer else rn < -0.6
        if hrange is not None: k &= (h > hrange[0]) & (h < hrange[1])
        return k

    def res(q):
        axis = tilt(a0, q[:2]); c = c0 + q[2]*u0 + q[3]*v0
        h, r, _ = cyl_coords(P[K], axis, c); b = np.floor(h/(bin_mm/1000)).astype(int)
        order = np.argsort(b); bs = b[order]; rs = r[order]
        cut = np.flatnonzero(np.diff(bs)) + 1; med = np.zeros_like(rs)
        for g in np.split(np.arange(len(rs)), cut): med[g] = np.median(rs[g])
        out = np.empty_like(r); out[order] = rs - med; return out
    K = sel(a0, c0)
    q = np.zeros(4)
    for _ in range(2):
        q = least_squares(res, q, loss='soft_l1', f_scale=0.001, x_scale=[0.01, 0.01, 0.002, 0.002]).x
        K = sel(tilt(a0, q[:2]), c0 + q[2]*u0 + q[3]*v0)
    axis = tilt(a0, q[:2]); c = c0 + q[2]*u0 + q[3]*v0
    r = res(q); return axis, c, float(np.sqrt(np.mean(r**2))*1000)


def profile(P, N, axis, c, edges, outer=True, q=50):
    """per height bin: percentile q of r for outer (normal away from axis) or inner points; nan where empty."""
    h, r, _ = cyl_coords(P, axis, c); d = P - c; radial = d - np.outer(d @ axis, axis)
    rn = np.einsum('ij,ij->i', radial, N)/np.maximum(r, 1e-9); k = rn > 0.6 if outer else rn < -0.6
    out = np.full(len(edges)-1, np.nan); cnt = np.zeros(len(edges)-1, int)
    for i in range(len(edges)-1):
        s = k & (h >= edges[i]) & (h < edges[i+1]); cnt[i] = s.sum()
        if s.sum() > 20: out[i] = np.percentile(r[s], q)
    return out, cnt


def fit_sphere(X):
    def f(q): return np.linalg.norm(X - q[:3], axis=1) - q[3]
    c0 = X.mean(0); q = least_squares(f, np.r_[c0, np.linalg.norm(X-c0, axis=1).mean()], loss='soft_l1', f_scale=0.001).x
    return q[:3], abs(q[3]), float(np.sqrt(np.mean(f(q)**2)))


def protrusions(P, model, thresh=0.004, link=0.004, min_pts=40, max_extent=0.04):
    """clusters of base-mesh points lying OUTSIDE the procedural model by > thresh (candidate Vicon markers etc.).
    Returns list of dict(points, centre, extent)."""
    from trimesh.proximity import signed_distance
    sd = signed_distance(model, P)  # >0 inside
    k = sd < -thresh; X = P[k]
    if len(X) == 0: return []
    tree = cKDTree(X); pairs = tree.query_pairs(link, output_type='ndarray')
    import scipy.sparse as sp, scipy.sparse.csgraph as cg
    G = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(X), len(X)))
    n, lab = cg.connected_components(G, directed=False)
    out = []
    for i in range(n):
        Y = X[lab == i]
        if len(Y) < min_pts: continue
        ext = np.ptp(Y, 0).max()
        out.append(dict(points=Y, centre=Y.mean(0), extent=float(ext), n=len(Y), small=bool(ext <= max_extent)))
    return out


def marker_spheres(P, model, r_default=0.0065, r_range=(0.004, 0.009), **kw):
    """Vicon markers = small protrusion clusters; sphere fitted to the cluster's outermost points. Returns [(centre, radius, cluster)]."""
    out = []
    for c in protrusions(P, model, **kw):
        if not c['small']: continue
        ctr, r, rms = fit_sphere(c['points'])
        if not (r_range[0] <= r <= r_range[1]) or rms > 0.002:
            # fall back: default radius, centre pushed so the sphere covers the cluster's far side
            r = r_default; ctr = c['centre']
        out.append((ctr, float(r), c))
    return out


def markers_tip(P, model, thresh=0.004, r_default=0.0065, r_range=(0.005, 0.008), **kw):
    """Vicon markers (sphere on a short post) from protrusion clusters: the tip = cluster point farthest outside the
    model; sphere radius = sphere fit to the cluster points within 1.6*r_default of the tip (clamped to r_range, default
    when the fit is poor); centre = tip - r * outward direction (closest model point -> tip); post foot = closest model
    point (1 mm inside). Returns list of dict(centre, radius, post_to, cluster_points, protrusion_mm)."""
    from trimesh.proximity import closest_point, signed_distance
    out = []
    for c in protrusions(P, model, thresh=thresh, **kw):
        if not c['small']: continue
        X = c['points']; sd = signed_distance(model, X); tip = X[np.argmin(sd)]
        cp = closest_point(model, tip[None])[0][0]; n = tip - cp; prot = np.linalg.norm(n); n = n/max(prot, 1e-9)
        Y = X[np.linalg.norm(X - tip, axis=1) < 1.6*r_default]
        r = r_default
        if len(Y) > 30:
            ctr, rf, rms = fit_sphere(Y)
            if r_range[0] <= rf <= r_range[1] and rms < 0.0015: r = rf
        r = min(r, max(prot/2, 0.003))
        ctr = tip - r*n
        cp2 = closest_point(model, ctr[None])[0][0]; m2 = cp2 - ctr; foot = cp2 + m2/max(np.linalg.norm(m2), 1e-9)*0.001
        out.append(dict(centre=ctr.tolist(), radius=float(r), post_to=foot.tolist(), cluster_points=int(c['n']), protrusion_mm=float(prot*1000)))
    return out


def douglas_peucker(pts, tol):
    pts = np.asarray(pts, float)
    if len(pts) < 3: return pts
    a, b = pts[0], pts[-1]; ab = b - a; L = np.linalg.norm(ab)
    d = np.abs(np.cross(ab, pts - a))/L if L > 0 else np.linalg.norm(pts - a, axis=1)
    i = int(np.argmax(d))
    if d[i] > tol:
        return np.r_[douglas_peucker(pts[:i+1], tol)[:-1], douglas_peucker(pts[i:], tol)]
    return np.array([a, b])


def offset_polyline(pts, t):
    """offset a 2D polyline by +-t/2 along its (averaged) normals -> closed CCW-ish polygon and per-segment quads."""
    pts = np.asarray(pts, float); n = len(pts)
    seg = np.diff(pts, axis=0); seg /= np.linalg.norm(seg, axis=1, keepdims=True)
    nrm = np.c_[-seg[:, 1], seg[:, 0]]
    vn = np.zeros_like(pts); vn[0] = nrm[0]; vn[-1] = nrm[-1]
    for i in range(1, n-1):
        m = nrm[i-1] + nrm[i]; m /= np.linalg.norm(m); vn[i] = m/max(np.dot(m, nrm[i]), 0.5)
    up, dn = pts + vn*t/2, pts - vn*t/2
    quads = [np.array([dn[i], dn[i+1], up[i+1], up[i]]) for i in range(n-1)]
    return np.r_[dn, up[::-1]], quads


def repost(markers, model):
    """re-attach marker posts to the final (marker-free) model: post foot = closest model point, 1 mm inside."""
    from trimesh.proximity import closest_point
    for m in markers:
        c = np.array(m['centre']); cp = closest_point(model, c[None])[0][0]; v = cp - c
        m['post_to'] = (cp + v/max(np.linalg.norm(v), 1e-9)*0.001).tolist(); m['post_len_mm'] = float(np.linalg.norm(v)*1000)
    return markers
