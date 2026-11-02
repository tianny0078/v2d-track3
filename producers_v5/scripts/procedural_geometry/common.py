"""Shared helpers for the procedural (parametric CAD) object geometry of Track 3 (procedural_geometry_v1).

Run with .deps/geom-venv/bin/python (trimesh 5.2 + manifold3d + coacd + embreex + mujoco).
All geometry is in metres, in the CURRENT asset's canonical (link) frame: poses.npz world_T_canonical stays valid.
"""
import json, re, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, trimesh
import manifold3d as m3d

R = Path(__file__).resolve().parents[2]
T3 = R/'outputs/track3'
OUT = T3/'procedural_geometry_v1'
V6 = T3/'learning_v6'
EPS = [1, 3, 5, 6, 8, 10, 14, 15, 16, 19, 20, 25, 28, 30, 32, 33, 35, 36, 37, 38]
A = T3/'advice_restart_20261017'
BASE = {  # unscaled source asset dir per object (traced from learning_v6 objects.json + source_manifest; scaled copies = base*s)
    'white_pot': A/'canonical_all20_scenes_v1/assets/white_pot',
    'white_pot_lid': A/'canonical_collision_qem_extra_v1/white_pot_lid/assets',
    'wooden_spoon': A/'canonical_all20_scenes_v1/assets/wooden_spoon',
    'beige_cup': A/'canonical_all20_scenes_v1/assets/beige_cup',
    'blue_cup': A/'canonical_all20_scenes_v1/assets/blue_cup',
    'plastic_dish_rack': A/'canonical_collision_postqem_v1/plastic_dish_rack/assets',
    'mini_sweeper': A/'canonical_all20_scenes_v1/assets/mini_sweeper',
    'mini_dust_pan': A/'canonical_collision_postqem_v1/mini_dust_pan/assets',
    'water_pitcher': A/'canonical_all20_scenes_v1/assets/water_pitcher',
    'wooden_piece_1': A/'canonical_all20_scenes_v1/assets/wooden_piece_1',
    'wooden_piece_2': A/'canonical_all20_scenes_v1/assets/wooden_piece_2',
}


def load_base(name):
    return trimesh.load(BASE[name]/'visual.obj', force='mesh', process=False)


def base_collision(name):
    u = BASE[name]/'object.urdf'
    return [trimesh.load(u.parent/c.find('geometry/mesh').attrib['filename'], force='mesh', process=False)
            for c in ET.parse(u).getroot().iter('collision')]


def occurrences():
    """{name: [(episode, uniform_scale or 1.0, item)]} from learning_v6 (read only)."""
    occ = {}
    for e in EPS:
        man = json.loads((V6/f'episode_{e:06d}/objects/objects.json').read_text())
        for o in man['objects']:
            occ.setdefault(o['name'], []).append((e, float(o.get('uniform_scale_applied') or 1.0), o))
    return occ


# ---------------------------------------------------------------- manifold helpers
def to_manifold(m):
    m = trimesh.Trimesh(m.vertices, m.faces, process=False)
    man = m3d.Manifold(m3d.Mesh64(vert_properties=np.asarray(m.vertices, np.float64), tri_verts=np.asarray(m.faces, np.uint64)))
    if man.status() != m3d.Error.NoError: raise ValueError(f'not a manifold input: {man.status()}')
    return man


def to_trimesh(man):
    g = man.simplify(1e-7).to_mesh64()  # process=False: manifold output is already 2-manifold; trimesh merging can glue near-duplicate verts
    return trimesh.Trimesh(np.asarray(g.vert_properties)[:, :3].astype(float), np.asarray(g.tri_verts), process=False)


def revolve(profile_rz, n=96, axis_xf=None):
    """Closed (r, z) polygon (CCW, r >= 0) revolved about +z -> trimesh. axis_xf: 4x4 placing the local frame."""
    cs = m3d.CrossSection([np.asarray(profile_rz, float)]).simplify(1e-6)
    man = m3d.Manifold.revolve(cs, n)
    m = to_trimesh(man)
    if axis_xf is not None: m.apply_transform(axis_xf)
    return m


def extrude(poly_xy, h, z0=0.0, xf=None):
    cs = m3d.CrossSection([np.asarray(poly_xy, float)])
    m = to_trimesh(m3d.Manifold.extrude(cs, h).translate([0, 0, z0]))
    if xf is not None: m.apply_transform(xf)
    return m


def union(ms, min_volume=1e-8):
    """manifold union; components smaller than min_volume (m^3, default 0.01 cm^3) are dropped (level-set slivers)."""
    ms = list(ms)
    man = to_manifold(ms[0]) if len(ms) == 1 else m3d.Manifold.batch_boolean([to_manifold(m) for m in ms], m3d.OpType.Add)
    comps = man.decompose()
    if len(comps) > 1:
        keep = [c for c in comps if c.volume() >= min_volume] or [max(comps, key=lambda c: c.volume())]
        man = keep[0] if len(keep) == 1 else m3d.Manifold.batch_boolean(keep, m3d.OpType.Add)
    return to_trimesh(man)


def difference(a, bs):
    man = to_manifold(a)
    for b in bs: man = man - to_manifold(b)
    return to_trimesh(man)


def frame(z_axis, x_hint, origin):
    """4x4 with columns x, y, z (right-handed), z along z_axis, x close to x_hint."""
    z = np.asarray(z_axis, float); z /= np.linalg.norm(z)
    x = np.asarray(x_hint, float) - np.dot(x_hint, z)*z; x /= np.linalg.norm(x); y = np.cross(z, x)
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, origin; return M


# ---------------------------------------------------------------- metrics
def sample(m, n=40000, seed=0):
    return trimesh.sample.sample_surface(m, n, seed=seed)[0]


def chamfer(a, b, n=40000):
    """symmetric mean / 95th-percentile / max surface distance (m) between meshes a and b."""
    from scipy.spatial import cKDTree
    pa, pb = sample(a, n), sample(b, n, 1)
    da = cKDTree(pb).query(pa)[0]; db = cKDTree(pa).query(pb)[0]
    return dict(chamfer_mean_mm=float((da.mean()+db.mean())/2*1000), a2b_mean_mm=float(da.mean()*1000), b2a_mean_mm=float(db.mean()*1000),
                p95_mm=float(np.percentile(np.r_[da, db], 95)*1000), hausdorff_mm=float(max(da.max(), db.max())*1000))


def iou(a, b, pitch=0.002, n=200000, seed=0):
    """volumetric IoU by Monte-Carlo in the joint bounding box (both meshes must be watertight)."""
    lo = np.minimum(a.bounds[0], b.bounds[0]); hi = np.maximum(a.bounds[1], b.bounds[1])
    p = np.random.default_rng(seed).uniform(lo, hi, (n, 3))
    ia, ib = a.contains(p), b.contains(p)
    return float((ia & ib).sum()/max(1, (ia | ib).sum()))


def icp(src_pts, dst_mesh, iters=40, trim=0.9, R0=None, t0=None, allow_scale=False):
    """rigid (optionally similarity) trimmed point-to-point ICP of src points onto dst mesh surface. Returns 4x4, rms."""
    from scipy.spatial import cKDTree
    dst = sample(dst_mesh, 60000, 3); tree = cKDTree(dst)
    Rm = np.eye(3) if R0 is None else R0; t = np.zeros(3) if t0 is None else t0; s = 1.0
    for _ in range(iters):
        p = s*src_pts @ Rm.T + t; d, j = tree.query(p); k = d <= np.quantile(d, trim)
        P, Q = src_pts[k], dst[j[k]]; mp, mq = P.mean(0), Q.mean(0); H = (P-mp).T @ (Q-mq)
        U, S, Vt = np.linalg.svd(H); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T)); Rm = Vt.T @ D @ U.T
        if allow_scale: s = float(np.trace(np.diag(S) @ D)/((P-mp)**2).sum())
        t = mq - s*Rm @ mp
    M = np.eye(4); M[:3, :3] = s*Rm; M[:3, 3] = t
    p = src_pts @ M[:3, :3].T + M[:3, 3]; d = tree.query(p)[0]
    return M, float(np.sqrt((d**2).mean())*1000), float(np.median(d)*1000)


# ---------------------------------------------------------------- rendering (CPU ray casting, embree)
def look(dir_, up=(0, 0, 1)):
    d = np.asarray(dir_, float); d /= np.linalg.norm(d); u = np.asarray(up, float)
    if abs(np.dot(u, d)) > 0.95: u = np.array([0, 1., 0]) if abs(d[1]) < 0.95 else np.array([1., 0, 0])
    x = np.cross(d, u); x /= np.linalg.norm(x); y = np.cross(x, d); return x, y, d


def render(meshes, colors, view_dir, up=(0, 0, 1), res=240, span=None, center=None):
    """orthographic Lambertian render of a list of meshes; returns HxWx3 uint8."""
    scene = trimesh.util.concatenate(meshes); ids = np.concatenate([np.full(len(m.faces), i) for i, m in enumerate(meshes)])
    x, y, d = look(view_dir, up)
    c = scene.bounds.mean(0) if center is None else center
    sp = (np.linalg.norm(scene.extents)*0.55) if span is None else span
    u = np.linspace(-sp, sp, res); U, V = np.meshgrid(u, -u)
    org = c + U[..., None]*x + V[..., None]*y - d*sp*4
    o = org.reshape(-1, 3); dirs = np.tile(d, (len(o), 1))
    loc, ri, ti = scene.ray.intersects_location(o, dirs, multiple_hits=False)
    img = np.full((res*res, 3), 255.)
    if len(ri):
        n = scene.face_normals[ti]; sh = np.abs(n @ -d)*0.75 + 0.25
        col = np.asarray(colors, float)[ids[ti]]
        img[ri] = col*sh[:, None]
    return img.reshape(res, res, 3).astype(np.uint8)


VIEWS = [((0, 0, -1), (0, 1, 0), '+z top'), ((0, 0, 1), (0, 1, 0), '-z'), ((-1, 0, 0), (0, 0, 1), '+x'),
         ((0, -1, 0), (0, 0, 1), '+y'), ((-1, -1, -1), (0, 0, 1), 'iso'), ((1, 1, 0.6), (0, 0, 1), 'iso2')]


def contact_sheet(rows, labels, path, res=240):
    """rows: list of lists of (meshes, colors); each row rendered in the 6 canonical VIEWS."""
    import cv2
    allm = trimesh.util.concatenate([m for r in rows for m in r[0]])
    c = allm.bounds.mean(0); sp = np.linalg.norm(allm.extents)*0.55
    out = []
    for (ms, cols), lab in zip(rows, labels):
        tiles = [render(ms, cols, v, up, res, sp, c) for v, up, _ in VIEWS]
        row = np.hstack(tiles); cv2.putText(row, lab, (5, 18), 0, 0.55, (0, 0, 200), 1); out.append(row)
    hdr = np.full((22, out[0].shape[1], 3), 255, np.uint8)
    for i, (_, _, nm) in enumerate(VIEWS): cv2.putText(hdr, nm, (i*res+5, 16), 0, 0.5, (0, 0, 0), 1)
    cv2.imwrite(str(path), cv2.cvtColor(np.vstack([hdr]+out), cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------- physics export
def mass_props(parts_with_density):
    """[(watertight mesh, density kg/m3)] -> mass, COM, inertia about COM (3x3)."""
    M = 0.; C = np.zeros(3); I = np.zeros((3, 3)); items = []
    for m, rho in parts_with_density:
        mm = m.copy(); mm.density = rho; mp = mm.mass_properties; items.append((mp.mass, np.asarray(mp.center_mass), np.asarray(mp.inertia)))
        M += mp.mass; C += mp.mass*np.asarray(mp.center_mass)
    C /= M
    for mass, c, Ic in items:
        r = c - C; I += Ic + mass*(np.dot(r, r)*np.eye(3) - np.outer(r, r))
    return M, C, I


def write_urdf(path, name, mass, com, I, visual='visual.obj', collisions=(), rgba='0.8 0.8 0.8 1'):
    mass = float(mass); com = [float(x) for x in com]; I = np.asarray(I, float).tolist(); I = np.array(I, dtype=object)
    col = ''.join(f'    <collision>\n      <geometry>\n        <mesh filename="{c}" />\n      </geometry>\n    </collision>\n' for c in collisions)
    Path(path).write_text(
        f'<robot name="{name}">\n  <link name="{name}">\n    <inertial>\n'
        f'      <origin xyz="{com[0]!r} {com[1]!r} {com[2]!r}" rpy="0 0 0" />\n      <mass value="{mass!r}" />\n'
        f'      <inertia ixx="{I[0,0]!r}" ixy="{I[0,1]!r}" ixz="{I[0,2]!r}" iyy="{I[1,1]!r}" iyz="{I[1,2]!r}" izz="{I[2,2]!r}" />\n'
        f'    </inertial>\n    <visual>\n      <geometry>\n        <mesh filename="{visual}" />\n      </geometry>\n'
        f'    <material name="procedural"><color rgba="{rgba}" /></material></visual>\n{col}  </link>\n</robot>\n')


def decimate(m, faces=4000):
    """topology-preserving quadric edge collapse (pymeshlab) to <= faces; returns None unless watertight + volume."""
    if len(m.faces) <= faces: return m
    import pymeshlab
    for tgt in [faces - 4, int(faces*0.9), int(faces*0.75)]:
        ms = pymeshlab.MeshSet(); ms.add_mesh(pymeshlab.Mesh(np.asarray(m.vertices, float), np.asarray(m.faces, np.int32)))
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=tgt, preservetopology=True, preservenormal=True, qualitythr=0.5, planarquadric=True)
        o = ms.current_mesh(); d = trimesh.Trimesh(o.vertex_matrix(), o.face_matrix(), process=True)
        if len(d.faces) <= faces and d.is_watertight and d.is_volume: return d
    return None


def prism_xz(cs, y0, y1):
    """extrude a CrossSection (or polygon) given in canonical (x, z) coordinates along canonical +y from y0 to y1."""
    if not isinstance(cs, m3d.CrossSection): cs = m3d.CrossSection([np.asarray(cs, float)])
    m = to_trimesh(m3d.Manifold.extrude(cs, y1 - y0)); v = m.vertices.copy(); m.vertices = np.c_[v[:, 0], y0 + v[:, 2], v[:, 1]]
    if m.volume < 0: m.invert()
    return m


def circle_pts(c, r, n, a0=0.0, a1=2*np.pi, endpoint=False):
    a = np.linspace(a0, a1, n, endpoint=endpoint); return np.c_[c[0] + r*np.cos(a), c[1] + r*np.sin(a)]
