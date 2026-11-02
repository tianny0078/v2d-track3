"""VALIDATION ONLY: CD-O of multi-episode reconstructions (outputs/track3/multiep_recon_v1/<obj>/mesh_4096.obj) vs the currently
submitted procedural meshes, against the public Vicon scans, with the official metric function and the host's mesh budget:
    CD-O = MPPE._object_chamfer_cm(scan, budget_mesh(mesh, 4096, 4096), seed=_object_seed(ep, slot))  [cm, scale-normalised]
Also reports oriented-bounding-box extents of each arm vs the scan (metric scale, which CD-O ignores).
The scans are never read by the reconstruction code. Writes outputs/track3/multiep_recon_v1/eval/public_scans_<obj>.json.
Usage: python3 scripts/multiep_recon_eval/eval_public_scans.py blue_cup [more objects]
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd, trimesh

R = Path(__file__).resolve().parents[2]; KIT = R/'toolkit/8d2bce64/v2d_submission_kit'
spec = importlib.util.spec_from_file_location('mppe', KIT/'metric_code/track_3/MPPE.py')
MP = importlib.util.module_from_spec(spec); spec.loader.exec_module(MP)
sys.path.insert(0, str(KIT)); from v2dlb.mesh_budget import budget_mesh  # noqa: E402
PUB = R/'data/track3_public/track_3/public'; NEW = R/'outputs/track3/multiep_recon_v1'; CUR = R/'outputs/track3/procedural_geometry_v1'


def scan(name):
    d = PUB/f'mesh/{name}'; p = d/f'{name}.glb' if (d/f'{name}.glb').is_file() else d/f'{name}_visual.glb'
    m = trimesh.load(p, force='mesh'); m.merge_vertices(merge_tex=True, merge_norm=True)
    return np.asarray(m.vertices), np.asarray(m.faces)


def obb(p):
    return sorted(trimesh.load(p, force='mesh').bounding_box_oriented.extents.round(4).tolist())


def main(names):
    occ = {}
    for f in sorted((PUB/'data/chunk-000').glob('episode_*.parquet')):
        ep = int(f.stem.split('_')[1]); d = pd.read_parquet(f, columns=['observation.objects'])
        for slot, o in enumerate(d.iloc[0]['observation.objects']): occ.setdefault(o['name'], []).append((ep, slot))
    (NEW/'eval').mkdir(parents=True, exist_ok=True)
    for name in names:
        arms = {'current': CUR/name/'visual_4096.obj', 'new': NEW/name/'mesh_4096.obj'}
        if name not in occ: rep = dict(object=name, note='no public scan', obb_m={k: obb(p) for k, p in arms.items()})
        else:
            ref = scan(name); rows = []
            for ep, slot in occ[name]:
                r = dict(episode=ep, slot=slot)
                for k, p in arms.items(): r[k] = float(MP._object_chamfer_cm(ref, budget_mesh(Path(p), 4096, 4096), MP._object_seed(ep, slot)))
                rows.append(r)
            rep = dict(object=name, rows=rows, mean={k: float(np.mean([r[k] for r in rows])) for k in arms},
                       worst_delta=float(max(r['new'] - r['current'] for r in rows)),
                       obb_m=dict(scan=sorted(trimesh.Trimesh(*ref).bounding_box_oriented.extents.round(4).tolist()), **{k: obb(p) for k, p in arms.items()}))
        (NEW/f'eval/public_scans_{name}.json').write_text(json.dumps(rep, indent=1))
        print(json.dumps({k: v for k, v in rep.items() if k != 'rows'}), flush=True)


if __name__ == '__main__': main(sys.argv[1:])
