"""VALIDATION ONLY (one-shot reporting, never fed back into reconstruction): public-scan CD-O of the retrack_v8-initialised
multi-episode refits (mini_sweeper, water_pitcher) vs the currently submitted procedural visual_4096, same official metric /
mesh budget / seeds as eval_public_scans.py. Arms: current, final_4096.obj, and every variant final mesh present.
Writes outputs/track3/multiep_recon_v1/eval/public_scans_retrack_v8.json (sibling of public_scans_final_v1.json).
Usage: python3 scripts/multiep_recon_eval/eval_public_scans_retrack.py mini_sweeper water_pitcher
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_public_scans as EP  # noqa: E402  (scan loader, official MPPE metric, budget_mesh)


def main(names):
    occ = {}
    import pandas as pd
    for f in sorted((EP.PUB/'data/chunk-000').glob('episode_*.parquet')):
        ep = int(f.stem.split('_')[1]); d = pd.read_parquet(f, columns=['observation.objects'])
        for slot, o in enumerate(d.iloc[0]['observation.objects']): occ.setdefault(o['name'], []).append((ep, slot))
    outp = EP.NEW/'eval/public_scans_retrack_v8.json'
    allr = json.loads(outp.read_text()) if outp.exists() else {}
    for name in names:
        od = EP.NEW/name
        arms = {'current': EP.CUR/name/'visual_4096.obj', 'final_4096.obj': od/'final_4096.obj'}
        for p in sorted(od.glob('variants/*/*/final_4096.obj')): arms[f'variant{p.parts[-3]}'] = p
        for p in sorted(od.glob('final_4096.variant*.obj')): arms[p.name] = p
        arms = {k: v for k, v in arms.items() if v.exists()}
        ref = EP.scan(name); rows = []
        for ep, slot in occ[name]:
            r = dict(episode=ep, slot=slot)
            for k, p in arms.items(): r[k] = float(EP.MP._object_chamfer_cm(ref, EP.budget_mesh(Path(p), 4096, 4096), EP.MP._object_seed(ep, slot)))
            rows.append(r)
        allr[name] = dict(rows=rows, mean={k: float(np.mean([r[k] for r in rows])) for k in arms}, arms={k: str(v) for k, v in arms.items()},
                          obb={k: EP.obb(p) for k, p in arms.items()}, note='validation only; scans never read by reconstruction/tracking')
        print(name, json.dumps(allr[name]['mean']), flush=True)
    outp.write_text(json.dumps(allr, indent=1))


if __name__ == '__main__': main(sys.argv[1:])
