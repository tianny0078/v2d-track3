"""VALIDATION ONLY (run once, after all parameters were frozen): CD-O of the current canonical meshes vs the procedural
meshes against the public Vicon scans, with the official metric function and the host's mesh budget:
    CD-O = MPPE._object_chamfer_cm(scan, budget_mesh(ours, 4096, 4096), seed=_object_seed(ep, slot))  [cm, scale-normalised]
Every public (episode, slot) occurrence of the object is scored (only the seed differs). wooden_spoon has no scan.
The scans are never read by any fitting code (scripts/procedural_geometry/*.py other than this file).
Writes outputs/track3/procedural_geometry_v1/eval/public_scans.json.
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd, trimesh
sys.path.insert(0, str(Path(__file__).parent))
from common import R, OUT, BASE

KIT = R/'toolkit/8d2bce64/v2d_submission_kit'
spec = importlib.util.spec_from_file_location('mppe', KIT/'metric_code/track_3/MPPE.py')
MP = importlib.util.module_from_spec(spec); spec.loader.exec_module(MP)
sys.path.insert(0, str(KIT)); from v2dlb.mesh_budget import budget_mesh  # noqa: E402
PUB = R/'data/track3_public/track_3/public'


def scan(name):
    d = PUB/f'mesh/{name}'; p = d/f'{name}.glb' if (d/f'{name}.glb').is_file() else d/f'{name}_visual.glb'
    m = trimesh.load(p, force='mesh'); m.merge_vertices(merge_tex=True, merge_norm=True)
    return np.asarray(m.vertices), np.asarray(m.faces)


def main():
    occ = {}
    for f in sorted((PUB/'data/chunk-000').glob('episode_*.parquet')):
        ep = int(f.stem.split('_')[1]); d = pd.read_parquet(f, columns=['observation.objects'])
        for slot, o in enumerate(d.iloc[0]['observation.objects']): occ.setdefault(o['name'], []).append((ep, slot))
    rep = {}
    for name in BASE:
        if name not in occ: rep[name] = 'no public scan'; continue
        ref = scan(name); arms = {'current': BASE[name]/'visual.obj', 'procedural': OUT/name/'visual_4096.obj'}
        rows = []
        for ep, slot in occ[name]:
            r = dict(episode=ep, slot=slot)
            for k, p in arms.items(): r[k] = float(MP._object_chamfer_cm(ref, budget_mesh(Path(p), 4096, 4096), MP._object_seed(ep, slot)))
            rows.append(r)
        rep[name] = dict(rows=rows, current_mean=float(np.mean([r['current'] for r in rows])), procedural_mean=float(np.mean([r['procedural'] for r in rows])),
                         scan_obb_m=sorted(trimesh.Trimesh(*ref).bounding_box_oriented.extents.round(4).tolist()))
        print(name, json.dumps({k: v for k, v in rep[name].items() if k != 'rows'}), flush=True)
    (OUT/'eval').mkdir(exist_ok=True); (OUT/'eval/public_scans.json').write_text(json.dumps(rep, indent=1))


if __name__ == '__main__': main()
