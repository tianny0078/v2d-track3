"""Isotropic remeshing for closed manifold triangle meshes (project code, numpy).

Operations in the spirit of Botsch & Kobbelt 2004 / Palfinger 2022 ("Continuous Remeshing for Inverse Rendering"):
edge split (> hi*L), edge collapse (< lo*L, link condition + no normal flip + no new long edge), valence-driven edge
flip (near-planar quads only), tangential relaxation. Per-vertex attributes (albedo, optimizer moments) are carried
along: split -> mean of endpoints, collapse -> mean. Topology (genus, closedness) is preserved by construction; the
caller additionally verifies watertightness/Euler number and falls back to the previous mesh if anything is off.
"""
import numpy as np


def edge_maps(F):
    """Unique edges E [ne,2] (sorted), face->edge [nf,3] (local edge k = (F[k],F[k+1])), edge->2 faces [ne,2]."""
    he = np.stack([F, np.roll(F, -1, 1)], -1).reshape(-1, 2)          # half edges, face-major
    key = np.sort(he, 1)
    E, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    fe = inv.reshape(-1, 3)
    order = np.argsort(inv, kind='stable')
    fidx = order // 3
    if not (cnt == 2).all():
        raise ValueError(f'non-manifold/open mesh: edge face counts {np.unique(cnt)}')
    ef = fidx.reshape(-1, 2)
    return E, fe, ef


def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-20)


def split_long(V, F, A, hi, max_rounds=400):
    for _ in range(max_rounds):
        E, fe, ef = edge_maps(F)
        L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
        cand = L > hi
        if not cand.any():
            break
        # rank candidates by length; an edge is selected if it is the best candidate of BOTH its faces
        rank = np.full(len(E), -1, np.int64)
        ci = np.nonzero(cand)[0]
        rank[ci[np.argsort(L[ci])]] = np.arange(len(ci))
        best_f = rank[fe].max(1)                                     # best candidate rank per face
        sel = cand & (rank == best_f[ef[:, 0]]) & (rank == best_f[ef[:, 1]])
        si = np.nonzero(sel)[0]
        nv = len(V)
        mid = np.full(len(E), -1, np.int64); mid[si] = nv + np.arange(len(si))
        V = np.concatenate([V, 0.5 * (V[E[si, 0]] + V[E[si, 1]])])
        A = np.concatenate([A, 0.5 * (A[E[si, 0]] + A[E[si, 1]])]) if A is not None else None
        fsel = np.nonzero((mid[fe] >= 0).any(1))[0]
        k = np.argmax(mid[fe[fsel]] >= 0, axis=1)                    # local edge index being split
        p = F[fsel, k]; q = F[fsel, (k + 1) % 3]; r = F[fsel, (k + 2) % 3]; m = mid[fe[fsel, k]]
        F = F.copy()
        F[fsel] = np.stack([p, m, r], 1)
        F = np.concatenate([F, np.stack([m, q, r], 1)])
        F = flip_edges(V, F, 'delaunay', max_rounds=3, planar_cos=0.998)
    return V, F, A


class _Adj:
    def __init__(self, V, F):
        self.F = F.copy(); self.alive_f = np.ones(len(F), bool); self.alive_v = np.ones(len(V), bool)
        self.vf = [set() for _ in range(len(V))]
        for fi, f in enumerate(self.F):
            for v in f:
                self.vf[v].add(fi)

    def nbrs(self, v):
        s = set()
        for fi in self.vf[v]:
            s.update(self.F[fi])
        s.discard(v)
        return s


def _fnormal(P0, P1, P2):
    n = np.cross(P1 - P0, P2 - P0); l = np.linalg.norm(n)
    return n / l if l > 1e-20 else n, l


def collapse_short(V, F, A, lo, hi, max_rounds=5):
    V = V.copy(); A = A.copy() if A is not None else None
    for _ in range(max_rounds):
        E, _, _ = edge_maps(F)
        L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
        ci = np.nonzero(L < lo)[0]
        if not len(ci):
            break
        ci = ci[np.argsort(L[ci])]
        adj = _Adj(V, F); n_done = 0
        for e in ci:
            a, b = E[e]
            if not (adj.alive_v[a] and adj.alive_v[b]):
                continue
            shared = adj.vf[a] & adj.vf[b]
            if len(shared) != 2:
                continue
            na, nb = adj.nbrs(a), adj.nbrs(b)
            if len(na & nb) != 2 or len(na) <= 3 or len(nb) <= 3:   # link condition, no tetra/valence-3 collapse
                continue
            p = 0.5 * (V[a] + V[b])
            if max(np.linalg.norm(V[list((na | nb) - {a, b})] - p, axis=1)) > hi:
                continue
            ok = True
            for fi in (adj.vf[a] | adj.vf[b]) - shared:
                f = adj.F[fi]
                P = [V[x] for x in f]
                n0, _ = _fnormal(*P)
                P = [p if x in (a, b) else V[x] for x in f]
                n1, ar = _fnormal(*P)
                if ar < 1e-14 or np.dot(n0, n1) < 0.5:
                    ok = False; break
            if not ok:
                continue
            V[a] = p
            if A is not None:
                A[a] = 0.5 * (A[a] + A[b])
            for fi in shared:
                adj.alive_f[fi] = False
                for x in adj.F[fi]:
                    adj.vf[x].discard(fi)
            for fi in list(adj.vf[b]):
                f = adj.F[fi]; f[f == b] = a; adj.vf[a].add(fi)
            adj.vf[b] = set(); adj.alive_v[b] = False; n_done += 1
        F = adj.F[adj.alive_f]
        V, F, A = compact(V, F, A)
        if n_done == 0:
            break
    return V, F, A


def compact(V, F, A):
    used = np.zeros(len(V), bool); used[F.reshape(-1)] = True
    remap = -np.ones(len(V), np.int64); remap[used] = np.arange(used.sum())
    return V[used], remap[F], (A[used] if A is not None else None)


def flip_edges(V, F, mode='delaunay', max_rounds=50, planar_cos=0.95):
    """Vectorised edge flips. mode 'delaunay': flip when the opposite diagonal is shorter (skinny-triangle cleanup);
    mode 'valence': flip when it reduces sum |valence-6| over the 4 quad vertices. Only near-planar convex quads,
    never creating an existing edge; each face takes part in at most one flip per round."""
    F = F.copy()
    for _ in range(max_rounds):
        E, fe, ef = edge_maps(F)
        f0, f1 = ef[:, 0], ef[:, 1]
        a, b = E[:, 0], E[:, 1]
        # opposite vertices
        c = F[f0].sum(1) - a - b; d = F[f1].sum(1) - a - b
        # orientation: does f0 contain a->b ?
        loc = np.argmax(F[f0] == a[:, None], axis=1)
        ab_in_f0 = F[f0, (loc + 1) % 3] == b
        A_ = np.where(ab_in_f0, a, b); B_ = np.where(ab_in_f0, b, a)      # f0 = (A_,B_,c), f1 = (B_,A_,d)
        def fn(i, j, k):
            n = np.cross(V[j] - V[i], V[k] - V[i]); l = np.linalg.norm(n, axis=1)
            return n / np.maximum(l, 1e-30)[:, None], l
        n0, _ = fn(A_, B_, c); n1, _ = fn(B_, A_, d)
        m0, ar0 = fn(A_, d, c); m1, ar1 = fn(B_, c, d)
        nn = n0 + n1
        geo_ok = ((n0 * n1).sum(1) >= planar_cos) & ((m0 * nn).sum(1) > 0) & ((m1 * nn).sum(1) > 0) & \
                 ((m0 * m1).sum(1) >= planar_cos) & (np.minimum(ar0, ar1) > 1e-14) & (c != d)
        if mode == 'delaunay':
            lab = np.linalg.norm(V[a] - V[b], axis=1); lcd = np.linalg.norm(V[c] - V[d], axis=1)
            gain = lab - lcd
            want = gain > 0.05 * lab
        else:
            val = np.bincount(E.reshape(-1), minlength=len(V))
            dev0 = np.abs(val[a] - 6) + np.abs(val[b] - 6) + np.abs(val[c] - 6) + np.abs(val[d] - 6)
            dev1 = np.abs(val[a] - 7) + np.abs(val[b] - 7) + np.abs(val[c] - 5) + np.abs(val[d] - 5)
            gain = (dev0 - dev1).astype(np.float64)
            want = (gain > 0) & (val[a] > 3) & (val[b] > 3)
        cand = want & geo_ok
        if cand.any():
            key = np.sort(np.stack([c, d], 1), 1)
            ek = E[:, 0].astype(np.int64) * len(V) + E[:, 1]
            cand &= ~np.isin(key[:, 0].astype(np.int64) * len(V) + key[:, 1], ek)
        if not cand.any():
            break
        rank = np.full(len(E), -1, np.int64)
        ci = np.nonzero(cand)[0]
        rank[ci[np.lexsort((ci, gain[ci]))]] = np.arange(len(ci))
        best_f = rank[fe].max(1)
        sel = cand & (rank == best_f[f0]) & (rank == best_f[f1])
        # two selected flips could create the same new edge (c,d) -> keep one per key
        si = np.nonzero(sel)[0]
        k2 = key[si, 0].astype(np.int64) * len(V) + key[si, 1]
        _, first = np.unique(k2, return_index=True); si = si[first]
        F[f0[si]] = np.stack([A_[si], d[si], c[si]], 1)
        F[f1[si]] = np.stack([B_[si], c[si], d[si]], 1)
    return F


def flip_valence(V, F, max_rounds=5, planar_cos=0.995):
    return flip_edges(V, F, 'valence', max_rounds, planar_cos)


def relax(V, F, lam=0.5, iters=1):
    E, _, _ = edge_maps(F)
    for _ in range(iters):
        n = vertex_normals(V, F)
        s = np.zeros_like(V); c = np.zeros(len(V))
        np.add.at(s, E[:, 0], V[E[:, 1]]); np.add.at(s, E[:, 1], V[E[:, 0]])
        np.add.at(c, E[:, 0], 1); np.add.at(c, E[:, 1], 1)
        d = s / c[:, None] - V
        d -= (d * n).sum(1, keepdims=True) * n
        V = V + lam * d
    return V


def remesh(V, F, A, L, rounds=3, lo=0.75, hi=1.4, relax_lam=0.3):
    """Returns (V, F, A). A may be None. V/A float64 numpy, F int64."""
    V = np.asarray(V, np.float64); F = np.asarray(F, np.int64)
    A = None if A is None else np.asarray(A, np.float64)
    for _ in range(rounds):
        V, F, A = split_long(V, F, A, hi * L)
        V, F, A = collapse_short(V, F, A, lo * L, hi * L)
        F = flip_valence(V, F)
        if relax_lam > 0:
            V = relax(V, F, relax_lam)
    return V, F, A
