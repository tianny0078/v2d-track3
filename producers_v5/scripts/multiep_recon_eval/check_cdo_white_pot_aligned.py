"""VALIDATION ONLY: official CD-O on the public white_pot scan for the adopted final_4096 vs its rigidly re-framed copy
(procedural_geometry_v8/white_pot/visual_4096.obj). CD-O is rigid-invariant up to its own alignment, so it should agree."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import eval_public_scans as E
import pandas as pd
occ = []
for f in sorted((E.PUB/'data/chunk-000').glob('episode_*.parquet')):
    ep = int(f.stem.split('_')[1]); d = pd.read_parquet(f, columns=['observation.objects'])
    occ += [(ep, s) for s, o in enumerate(d.iloc[0]['observation.objects']) if o['name'] == 'white_pot']
ref = E.scan('white_pot'); rows = []
arms = {'final_4096': E.R/'outputs/track3/multiep_recon_v1/white_pot/final_4096.obj', 'v8_aligned': E.R/'outputs/track3/procedural_geometry_v8/white_pot/visual_4096.obj'}
for ep, slot in occ:
    r = dict(episode=ep, slot=slot, **{k: float(E.MP._object_chamfer_cm(ref, E.budget_mesh(Path(p), 4096, 4096), E.MP._object_seed(ep, slot))) for k, p in arms.items()})
    rows.append(r); print(r, flush=True)
out = dict(rows=rows, mean={k: float(np.mean([r[k] for r in rows])) for k in arms}, max_abs_diff=float(max(abs(r['final_4096'] - r['v8_aligned']) for r in rows)))
(E.R/'outputs/track3/procedural_geometry_v8/eval/cdo_white_pot_aligned.json').write_text(json.dumps(out, indent=1)); print(out['mean'], out['max_abs_diff'])
