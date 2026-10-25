"""Procedural primitives. Each primitive returns a Part: a watertight mesh for the visual union plus a list of convex
collision pieces that approximate exactly that primitive (so concave assemblies keep their cavities)."""
from dataclasses import dataclass, field
import numpy as np, trimesh
from common import revolve, extrude, frame, union, to_manifold, to_trimesh
import manifold3d as m3d


@dataclass
class Part:
    mesh: trimesh.Trimesh
    collision: list = field(default_factory=list)
    density: float = 1000.0      # effective density used for mass (kg/m^3), see notes for the material assumption
    label: str = ''


def hull(points):
    return trimesh.Trimesh(np.asarray(points, float)).convex_hull if not isinstance(points, trimesh.Trimesh) else points.convex_hull


def _place(m, xf):
    m = m.copy(); m.apply_transform(xf); return m


def revolve_part(profile_pieces, xf, n=96, n_seg=20, density=1000., label='', visual_profile=None):
    """profile_pieces: list of CONVEX (r, z) polygons (CCW) that together form the revolved section. Visual = revolve of
    their union (or of visual_profile, a finer non-convex polygon, when given); collision = each piece revolved (pieces
    touching the axis: one hull; others: n_seg angular wedges)."""
    cs = None
    for pc in ([visual_profile] if visual_profile is not None else profile_pieces):
        if isinstance(pc, tuple) and len(pc) == 2 and np.isscalar(pc[1]): pc = pc[0]
        c = m3d.CrossSection([np.asarray(pc, float)]); cs = c if cs is None else cs + c
    vis = to_trimesh(m3d.Manifold.revolve(cs.simplify(1e-6), n)); vis.apply_transform(xf)
    col = []
    for pc in profile_pieces:
        nsg = n_seg
        if isinstance(pc, tuple) and len(pc) == 2 and np.isscalar(pc[1]): pc, nsg = pc
        pc = np.asarray(pc, float)
        if pc[:, 0].min() < 1e-6:
            th = np.linspace(0, 2*np.pi, 48, endpoint=False)
            pts = np.stack([np.outer(pc[:, 0], np.cos(th)), np.outer(pc[:, 0], np.sin(th)), np.repeat(pc[:, 1, None], len(th), 1)], -1).reshape(-1, 3)
            col.append(_place(hull(pts), xf))
        else:
            edges = np.linspace(0, 2*np.pi, nsg+1)
            for a0, a1 in zip(edges[:-1], edges[1:]):
                th = np.linspace(a0, a1, 4)
                pts = np.stack([np.outer(pc[:, 0], np.cos(th)), np.outer(pc[:, 0], np.sin(th)), np.repeat(pc[:, 1, None], len(th), 1)], -1).reshape(-1, 3)
                col.append(_place(hull(pts), xf))
    return Part(vis, col, density, label)


def sphere_part(c, r, density, label='marker', subdiv=2, post_to=None, post_r=0.0025):
    """sphere (Vicon marker); optional cylindrical post from the centre to point post_to (on the host surface, slightly inside)."""
    s = trimesh.creation.icosphere(subdiv, r); s.apply_translation(c); ms = [s]
    if post_to is not None and np.linalg.norm(np.asarray(post_to)-c) > 1e-4:
        cyl = trimesh.creation.cylinder(post_r, segment=[c, post_to], sections=12); ms.append(cyl)
    vis = union(ms) if len(ms) > 1 else s
    return Part(vis, [vis.convex_hull], density, label)


def box_part(center, R, size, density, label=''):
    b = trimesh.creation.box(size); M = np.eye(4); M[:3, :3] = R; M[:3, 3] = center; b.apply_transform(M)
    return Part(b, [b], density, label)


def extrude_part(poly_xy, h, xf, density, label='', convex_pieces=None):
    """planar polygon (local xy) extruded along local z from 0..h then placed by xf. convex_pieces: optional list of
    convex polygons covering poly_xy for collision (default: hull of the whole extrusion)."""
    vis = extrude(poly_xy, h, 0.0, xf)
    if convex_pieces is None: col = [vis.convex_hull]
    else: col = [extrude(pc, h, 0.0, xf).convex_hull for pc in convex_pieces]
    return Part(vis, col, density, label)


def skin(sections, closed=False):
    """watertight tube through K-gon sections (same K, consistent ordering): quads between consecutive sections,
    centroid fans at the two ends (or wrap-around when closed)."""
    S = np.asarray(sections, float); n, K = S.shape[:2]; V = S.reshape(-1, 3); F = []
    rings = n if closed else n - 1
    for i in range(rings):
        a, b = i*K, ((i+1) % n)*K
        for j in range(K):
            j1 = (j+1) % K; F += [[a+j, a+j1, b+j1], [a+j, b+j1, b+j]]
    if not closed:
        c0 = len(V); c1 = c0 + 1; V = np.r_[V, [S[0].mean(0)], [S[-1].mean(0)]]
        for j in range(K):
            j1 = (j+1) % K; F += [[c0, j1, j], [c1, (n-1)*K+j, (n-1)*K+j1]]
    m = trimesh.Trimesh(V, np.array(F), process=False)
    if m.volume < 0: m.invert()
    return m


def loft_part(sections, density, label='', collision_every=1):
    """sections: list of (K,3) closed CONVEX loops (same K) along a path. Visual = skinned tube through them (watertight);
    collision = hulls of consecutive section pairs (merged every collision_every pairs)."""
    vis = skin(sections)
    col = [hull(np.concatenate([np.r_[sections[i], sections[min(i+collision_every, len(sections)-1)]]]))
           for i in range(0, len(sections)-1, collision_every)]
    return Part(vis, col, density, label)


def torus_part(R, r, xf, density, label='', n_major=48, n_minor=16, n_seg=12, arc=(0, 2*np.pi)):
    """torus (or arc of one) of major radius R, minor radius r in local xy plane; collision = n_seg hulls along the arc."""
    a0, a1 = arc; full = abs((a1 - a0) - 2*np.pi) < 1e-9
    th = np.linspace(a0, a1, n_major+1 if not full else n_major+1)
    ph = np.linspace(0, 2*np.pi, n_minor, endpoint=False)
    def ring(t): return np.stack([(R + r*np.cos(ph))*np.cos(t), (R + r*np.cos(ph))*np.sin(t), r*np.sin(ph)], -1)
    secs = [ring(t) for t in th]
    vis = skin(secs[:-1], closed=True) if full else skin(secs); vis.apply_transform(xf)
    step = max(1, n_major//n_seg)
    col = [_place(hull(np.concatenate(secs[i:i+step+1])), xf) for i in range(0, n_major, step)]
    return Part(vis, col, density, label)


def rect_section(p, d, side, w, t):
    """rectangle centred at p, perpendicular to direction d: width w along `side` (made perpendicular to d), thickness t
    along side x d."""
    d = np.asarray(d, float)/np.linalg.norm(d); a = np.asarray(side, float) - np.dot(side, d)*d; a /= np.linalg.norm(a)
    nn = np.cross(a, d)
    return np.array([p + sa*w/2*a + sb*t/2*nn for sa, sb in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])


def sweep_rect_part(path, side, w, t, density, label=''):
    """flat strap of width w (along `side`) and thickness t swept along polyline path; joint sections use the
    bisector direction, scaled by 1/cos(half bend) in thickness. Visual = union of per-segment hulls; collision = same."""
    path = np.asarray(path, float); secs = []
    for i in range(len(path)):
        d0 = path[i] - path[max(i-1, 0)]; d1 = path[min(i+1, len(path)-1)] - path[i]
        d0 = d0/np.linalg.norm(d0) if np.linalg.norm(d0) > 0 else d1/np.linalg.norm(d1)
        d1 = d1/np.linalg.norm(d1) if np.linalg.norm(d1) > 0 else d0
        d = d0 + d1; c = max(0.3, np.dot(d0, d)/np.linalg.norm(d))
        secs.append(rect_section(path[i], d, side, w, t/c))
    return loft_part(secs, density, label)


def merge_hulls(hulls, k):
    """merge consecutive convex pieces k at a time (hull of their union) -> fewer collision geoms."""
    return [hull(np.concatenate([h.vertices for h in hulls[i:i+k]])) for i in range(0, len(hulls), k)]
