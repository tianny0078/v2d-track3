"""Object-manifest tree outputs/track3/learning_v8_geom/episode_X/objects/ (manifests only; poses by fix_support_v8.py).

Per scored episode: objects.json = learning_v7_geom's (read only) with every adopted object's assets swapped for
procedural_geometry_v8 (multiep_recon_v1 meshes + v8 collision/URDF). The v8 meshes are SHARED across episodes with the
multi-episode scale already baked in (blue_cup: sbar = 0.9293 = geometric mean of the v7 per-episode scales
0.911-0.949), so uniform_scale_applied = 1 and no per-episode scaled copy is made; the v7 value is kept as
uniform_scale_applied_v7. Objects not in V8 keep their v7 assets (mini_dust_pan: procedural_geometry_v1, unchanged).
mini_sweeper / water_pitcher: taken from outputs/track3/retrack_v8/READY.json when present (see --retrack).

Usage: .deps/geom-venv/bin/python scripts/procedural_geometry/make_v8_manifests.py [--tree learning_v8_geom] [eps...]
"""
import json, sys
from pathlib import Path
import numpy as np, trimesh
sys.path.insert(0, str(Path(__file__).parent))
from common import T3, EPS

GEO8 = T3/'procedural_geometry_v8'
V8 = ['blue_cup', 'plastic_dish_rack', 'white_pot', 'beige_cup', 'white_pot_lid', 'wooden_piece_1', 'wooden_piece_2', 'wooden_spoon']
PENDING = {'mini_sweeper', 'water_pitcher'}   # sibling re-tracking (retrack_v8); episodes 19, 20, 30


def item_v8(old, d):
    rep = json.loads((d/'report.json').read_text()); ph = rep['physics']; ck = rep['checks']
    m = trimesh.load(d/'visual.obj', force='mesh')
    keep = {k: v for k, v in old.items() if k not in ('uniform_scale_applied',)}
    return dict(keep, mesh_path=str(d/'visual.obj'), urdf_path=str(d/'object.urdf'), mesh_path_4096=str(d/'visual_4096.obj'),
                canonical_COM=list(ph['com']), bounding_radius_m=float(np.linalg.norm(m.vertices, axis=1).max()),
                mass_kg=ph['mass_kg'],
                physics=dict(mass_from='procedural_geometry_v8: generator density groups deformed by the adopted multiep_recon map',
                             inertia_eigenvalues=list(ph['inertia_eig']), collision_parts=ck['collision']['parts'],
                             collision_all_convex=ck['collision']['all_convex'], COM_is_solid_volume_weighted=True),
                geometry_quality=dict(faces=len(m.faces), watertight=bool(m.is_watertight), is_volume=bool(m.is_volume),
                                      volume_cm3=float(m.volume*1e6), faces_4096_variant=ck['visual_4096']['faces']),
                uniform_scale_applied=1.0, uniform_scale_applied_v7=old.get('uniform_scale_applied'),
                reconstruction_source='procedural_geometry_v8 = multiep_recon_v1 final_4096 (shared across scored episodes, scale baked) '
                                      '+ same deformation applied to the procedural_geometry_v1 high-res visual / collision / density groups',
                previous_mesh_path_v7=old['mesh_path'], previous_urdf_path_v7=old['urdf_path'])


def main(ep, tree, retrack):
    S = T3/f'learning_v7_geom/episode_{ep:06d}/objects'; D = T3/tree/f'episode_{ep:06d}/objects'
    man = json.loads((S/'objects.json').read_text()); objs = []; kept = []; swapped = []; repl = {}
    for o in man['objects']:
        n = o['name']
        if n in V8:
            objs.append(item_v8(o, GEO8/n)); swapped.append(n)
        elif n in PENDING:
            r = retrack.get(n, {}); pp = r.get('poses', {}).get(f'ep{ep}')
            if pp is None or not (GEO8/n/'report.json').exists():
                print(ep, f'SKIP: {n} waits for retrack_v8 / procedural_geometry_v8 assets', flush=True); return False
            objs.append(dict(item_v8(o, GEO8/n), retrack_v8_poses=pp)); swapped.append(n + '(retrack_v8)'); repl[n] = pp
        else:
            objs.append(o); kept.append(n)
    D.mkdir(parents=True, exist_ok=True)
    if repl:   # fix_support_v8 source poses: learning_v5 pre-fix poses with the re-tracked slots replaced (same ORB world)
        z = dict(np.load(T3/f'learning_v5/episode_{ep:06d}/objects/poses.npz')); names = [str(x) for x in z['object_names']]
        for n, pp in repl.items():
            q = dict(np.load(pp)); b = names.index(n)
            # retrack poses are stored in float32 (|R^T R - I| up to ~1.4e-5, beyond prepare's 1e-5 check): project to SO(3)
            # by SVD and recompute the wxyz quaternion; translation unchanged
            from scipy.spatial.transform import Rotation
            Wq = np.asarray(q['world_T_canonical'], np.float64).copy(); U, _, Vt = np.linalg.svd(Wq[:, :3, :3]); Wq[:, :3, :3] = U @ Vt
            assert np.abs(Wq[:, :3, :3] - q['world_T_canonical'][:, :3, :3]).max() < 1e-4 and (np.linalg.det(Wq[:, :3, :3]) > 0).all()
            q['world_T_canonical'] = Wq
            q['object_pose_wxyz'] = np.concatenate([Wq[:, :3, 3], Rotation.from_matrix(Wq[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]], 1)
            assert (q['frame_index'] == z['frame_index']).all() and len(q['world_T_canonical']) == len(z['world_T_canonical'])
            z['world_T_canonical'] = z['world_T_canonical'].copy(); z['object_pose_wxyz'] = z['object_pose_wxyz'].copy()
            z['world_T_canonical'][:, b] = q['world_T_canonical']; z['object_pose_wxyz'][:, b] = q['object_pose_wxyz']
            if 'observed' in q and 'observed' in z: z['observed'] = z['observed'].copy(); z['observed'][:, b] = q['observed']
        np.savez_compressed(D/'source_poses.npz', **z)
    new = dict(man, objects=objs, poses_path=str(D/'poses.npz'), source_manifest=str(S/'objects.json'),
               poses_source='set by fix_support_v8.py (learning_v5 pre-fix poses, as v7)',
               reconstruction_source=man.get('reconstruction_source', '').replace(' | geometry: procedural_geometry_v1', '') + ' | geometry: procedural_geometry_v8',
               v8_swapped=swapped, v8_kept_v7_assets=kept, v8_retrack_poses=repl or None,
               v8_source_poses=str(D/'source_poses.npz') if repl else None)
    (D/'objects.json').write_text(json.dumps(new, indent=1))
    print(ep, 'swapped', swapped, 'kept', kept, flush=True)
    return True


if __name__ == '__main__':
    a = sys.argv[1:]; tree = 'learning_v8_geom'
    while a and a[0].startswith('--'):
        k = a.pop(0)[2:]; v = a.pop(0)
        if k == 'tree': tree = v
    assert tree not in ('learning_v7_geom', 'learning_v6')
    rj = T3/'retrack_v8/READY.json'
    retrack = json.loads(rj.read_text())['objects'] if rj.exists() else {}
    for e in (map(int, a) if a else EPS): main(e, tree, retrack)
