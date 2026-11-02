"""water_pitcher: mirror-symmetric organic jug (symmetry plane x = x0 in the canonical frame), modelled as a symmetric
'bas-relief' solid: a side silhouette S in the y-z plane and a half-thickness field w(y, z) >= 0:
      solid = { (x, y, z) : (y, z) in S  and  |x - x0| <= w(y, z) }
meshed as a manifold3d level set (3.5 mm edges; 4.5 mm for the <=4096-face variant; manifold by construction, largest
component kept), decimated with pymeshlab (topology preserving) to 20000 / 3700 faces, and convex-decomposed with CoACD (threshold 0.02, <= 48 hulls).
The jug is solid here: its opening and thin wall are not modelled (nothing goes into it in ep20).

Parameters and how they are obtained (current SAM3D mesh samples):
  x0          median x (per-z |x| extents are symmetric within 3 mm)
  S           2 mm raster of all samples projected on y-z, closing (5 px) + hole filling
  w(y, z)     per 2 mm cell: P99 of |x - x0| of the samples in the cell; empty cells inside S filled by nearest
              neighbour; Gaussian smoothing sigma 2 cells; w >= 2 mm
Markers: none separable -- the SAM3D marker bumps are absorbed into w(y, z) (P99 per cell) as smooth bumps.
Mass: thin-walled PP jug: outer area x 2 mm x 905 kg/m^3 spread over the solid volume (effective density, ASSUMPTION).
"""
import numpy as np, trimesh, cv2
from scipy.ndimage import binary_fill_holes, gaussian_filter, distance_transform_edt
from fitting import surf, markers_tip, repost
from parts import Part, sphere_part

RGBA = '0.95 0.95 0.95 1'
PP = 905.0; T_PHYS = 0.002; MARKER = 400.0; RES = 0.002


def fit(base):
    P, N = surf(base, 600000); x, y, z = P.T
    x0 = float(np.median(x)); lo = np.array([y.min(), z.min()]) - 0.01
    H = int((z.max() - lo[1])/RES) + 10; W = int((y.max() - lo[0])/RES) + 10
    ij = ((np.c_[y, z] - lo)/RES).astype(int)
    img = np.zeros((H, W), np.uint8); img[ij[:, 1], ij[:, 0]] = 1
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)); S = binary_fill_holes(img)
    w = np.zeros((H, W)); ax = np.abs(x - x0)
    order = np.lexsort((ax, ij[:, 0], ij[:, 1]))
    cell = ij[order, 1]*W + ij[order, 0]; brk = np.r_[np.flatnonzero(np.diff(cell)) + 1, len(cell)]; st = np.r_[0, brk[:-1]]
    for a, b in zip(st, brk):
        v = ax[order[a:b]]; c = cell[a]; w[c//W, c % W] = v[min(len(v) - 1, int(0.99*len(v)))]
    have = w > 0
    _, (ii, jj) = distance_transform_edt(~have, return_indices=True); w = w[ii, jj]
    w = gaussian_filter(w, 2.0); w = np.where(S, np.maximum(w, 0.002), 0)
    p = dict(x0=x0, res=RES, origin_yz=lo.tolist(), silhouette_rle=_rle(S), w_mm=np.round(w*1000, 2).tolist(),
             grid_shape=[H, W], note='w_mm: half-thickness field (mm) on the y-z grid, row = z index, col = y index')
    model = trimesh.util.concatenate([q.mesh for q in build(p, 'high', markers=False)])
    mk = markers_tip(P[::2], model, thresh=0.004)
    p['markers'] = [m for m in mk if 6.0 <= m['protrusion_mm'] <= 20.0 and m['cluster_points'] > 100]
    p['dropped_bumps'] = [m for m in mk if m not in p['markers']]
    repost(p['markers'], model)
    return p


def _rle(B):
    f = B.ravel().astype(np.int8); d = np.flatnonzero(np.diff(np.r_[0, f, 0])); return d.tolist()


def _unrle(r, shape):
    f = np.zeros(shape[0]*shape[1], bool)
    for a, b in zip(r[::2], r[1::2]): f[a:b] = True
    return f.reshape(shape)


def solid(p, edge=0.002):
    """manifold3d level set of F(x, y, z) = w(y, z) - |x - x0| (bilinear w, 0 outside S): manifold by construction."""
    import manifold3d as m3d
    from common import to_trimesh
    H, W = p['grid_shape']; w = np.array(p['w_mm'])/1000; res = p['res']; lo = np.array(p['origin_yz']); x0 = p['x0']
    wm = float(w.max())
    def F(x, y, z):
        u = (y - lo[0])/res; v = (z - lo[1])/res
        if u < 0 or v < 0 or u >= W - 1 or v >= H - 1: return -1.0
        i, j = int(v), int(u); fu, fv = u - j, v - i
        ww = (w[i, j]*(1-fu)*(1-fv) + w[i, j+1]*fu*(1-fv) + w[i+1, j]*(1-fu)*fv + w[i+1, j+1]*fu*fv)
        return ww - abs(x - x0) if ww > 0.0005 else -1.0
    b = [x0 - wm - 0.004, lo[0], lo[1], x0 + wm + 0.004, lo[0] + W*res, lo[1] + H*res]
    man = m3d.Manifold.level_set(F, b, edge, 0.0)
    man = m3d.Manifold.batch_boolean(sorted(man.decompose(), key=lambda q: -q.volume())[:1], m3d.OpType.Add)  # largest piece
    return to_trimesh(man)


def build(p, res='high', markers=True):
    from common import decimate
    m = solid(p, 0.0035 if res == 'high' else 0.0045)
    m = decimate(m, 20000 if res == 'high' else 3700)  # topology-preserving quadric collapse (pymeshlab)
    import coacd
    coacd.set_log_level('error')
    hs = coacd.run_coacd(coacd.Mesh(m.vertices, m.faces), threshold=0.02, max_convex_hull=48, preprocess_mode='off')
    col = [trimesh.Trimesh(v, f).convex_hull for v, f in hs]
    body = Part(m, col, PP*T_PHYS*m.area/max(m.volume, 1e-9), 'jug')  # thin-walled jug spread over the solid
    parts = [body]
    if markers:
        for mk in p.get('markers', []):
            parts.append(sphere_part(np.array(mk['centre']), mk['radius'], MARKER, 'marker', 2 if res == 'high' else 1, np.array(mk['post_to'])))
    return parts
