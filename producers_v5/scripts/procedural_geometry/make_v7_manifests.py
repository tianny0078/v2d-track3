"""Object-manifest variant tree outputs/track3/learning_v7_geom/episode_X/objects/ (manifests only, no training).

Per scored episode: objects.json = learning_v6's with every object's assets swapped for procedural_geometry_v1
(read-only use of learning_v6). Episode scaling rule (same as build_track3_v2_object_manifests.scaled_assets): an object
with uniform_scale_applied = s gets a per-episode copy whose meshes are scaled by s about the link origin; mass and
inertia are recomputed for the scaled solid (mass * s^3, inertia * s^5, COM * s: same densities).
Poses: --poses v6 copies learning_v6 poses.npz unchanged; --poses refit leaves poses to fix_support_v7.py.

Usage: .deps/geom-venv/bin/python scripts/procedural_geometry/make_v7_manifests.py [--tree learning_v7_geom] [--poses v6] [eps...]
"""
import json, re, shutil, sys, hashlib
from pathlib import Path
import numpy as np, trimesh
sys.path.insert(0, str(Path(__file__).parent))
from common import T3, OUT as GEO, V6, EPS


def scaled_copy(src, dst, s):
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.suffix == '.obj':
            m = trimesh.load(f, force='mesh', process=False); m.vertices = m.vertices*s; m.export(dst/f.name)
    u = (src/'object.urdf').read_text()
    u = re.sub(r'(<origin xyz=")([^"]+)(")', lambda g: g.group(1)+' '.join(repr(float(x)*s) for x in g.group(2).split())+g.group(3), u, count=1)
    u = re.sub(r'(<mass value=")([^"]+)(")', lambda g: g.group(1)+repr(float(g.group(2))*s**3)+g.group(3), u)
    u = re.sub(r'(i(?:xx|xy|xz|yy|yz|zz)=")([^"]+)(")', lambda g: g.group(1)+repr(float(g.group(2))*s**5)+g.group(3), u)
    (dst/'object.urdf').write_text(u)
    return dst


def item_for(old, d, s):
    rep = json.loads((GEO/old['name']/'report.json').read_text())
    ph = rep['physics']; ck = rep['checks']
    m = trimesh.load(d/'visual.obj', force='mesh')
    return dict(old, mesh_path=str(d/'visual.obj'), urdf_path=str(d/'object.urdf'), mesh_path_4096=str(d/'visual_4096.obj'),
                canonical_COM=[c*s for c in ph['com']], bounding_radius_m=float(np.linalg.norm(m.vertices, axis=1).max()),
                mass_kg=ph['mass_kg']*s**3,
                physics=dict(mass_from='procedural volume x assumed density (see notes/track3-procedural-geometry.md)',
                             inertia_eigenvalues=(np.array(ph['inertia_eig'])*s**5).tolist(), collision_parts=ck['collision']['parts'],
                             collision_all_convex=ck['collision']['all_convex'], COM_is_solid_volume_weighted=True),
                geometry_quality=dict(faces=len(m.faces), watertight=bool(m.is_watertight), is_volume=bool(m.is_volume),
                                      volume_cm3=float(m.volume*1e6), faces_4096_variant=ck['visual_4096']['faces']),
                reconstruction_source='procedural_geometry_v1 (parametric CAD fitted to the frozen SAM3D canonical mesh, same canonical frame)',
                previous_mesh_path=old['mesh_path'], previous_urdf_path=old['urdf_path'])


def main(ep, tree, poses):
    S = V6/f'episode_{ep:06d}/objects'; D = T3/tree/f'episode_{ep:06d}/objects'; D.mkdir(parents=True, exist_ok=True)
    man = json.loads((S/'objects.json').read_text()); objs = []; missing = []
    for o in man['objects']:
        s = float(o.get('uniform_scale_applied') or 1.0); src = GEO/o['name']
        if not (src/'object.urdf').exists(): objs.append(o); missing.append(o['name']); continue
        d = src if abs(s - 1) < 1e-9 else scaled_copy(src, D.parent/'assets'/o['name'], s)
        objs.append(item_for(o, d, s))
    if poses == 'v6': shutil.copy2(S/'poses.npz', D/'poses.npz')
    new = dict(man, objects=objs, poses_path=str(D/'poses.npz'), source_manifest=str(S/'objects.json'), poses_source=f'learning_v6 ({poses})',
               reconstruction_source=man.get('reconstruction_source', '') + ' | geometry: procedural_geometry_v1',
               procedural_missing=missing)
    (D/'objects.json').write_text(json.dumps(new, indent=1))
    print(ep, [o['name'] for o in objs], 'missing', missing, flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]; tree = 'learning_v7_geom'; poses = 'v6'
    while a and a[0].startswith('--'):
        k = a.pop(0)[2:]; v = a.pop(0)
        if k == 'tree': tree = v
        if k == 'poses': poses = v
    for e in (map(int, a) if a else EPS): main(e, tree, poses)
