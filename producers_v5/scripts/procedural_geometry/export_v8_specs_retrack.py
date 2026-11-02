"""deform_spec.json for the retrack_v8 objects (outputs/track3/retrack_v8/READY.json): final_4096 = source visual_4096 x
mesh_scale_vs_source (uniform scale about the link origin, no deformation) -> kind 'identity', sbar = that scale.
Verifies the scaled source reproduces final_4096.obj. Usage: python3 export_v8_specs_retrack.py mini_sweeper [water_pitcher]"""
import json, sys
from pathlib import Path
import numpy as np, trimesh
R = Path(__file__).resolve().parents[2]; OUT = R/'outputs/track3/procedural_geometry_v8'
rd = json.loads((R/'outputs/track3/retrack_v8/READY.json').read_text())
for o in sys.argv[1:]:
    x = rd['objects'][o]; s = float(x['mesh_scale_vs_source'])
    src4 = Path(x['mesh_source']); src4 = src4 if src4.is_absolute() else R/src4
    hi = Path(x['mesh_source_highres']); hi = hi if hi.is_absolute() else R/hi
    c4 = trimesh.load(src4, process=False, force='mesh'); fin = trimesh.load(x['final_mesh_4096'], process=False, force='mesh')
    err = float(np.abs(c4.vertices*s - fin.vertices).max()); same_f = bool(np.array_equal(c4.faces, fin.faces))
    assert err < 1e-6 and same_f, (o, err, same_f)
    spec = dict(object=o, adopted_run=f'retrack_v8 READY.json (version {rd.get("version", 1)}, created {rd.get("created_utc")})', proc_mesh=str(hi),
                proc_params=str(hi.parent/'params.json'), final_4096=x['final_mesh_4096'], spec=dict(kind='identity', K=5, sbar=s, normals='area'),
                reproduce_final4096_max_err_mm=err*1000, faces_identical=same_f, ready_json_entry=dict(verdict=x.get('verdict'), poses=x.get('poses')))
    (OUT/o).mkdir(parents=True, exist_ok=True); (OUT/o/'deform_spec.json').write_text(json.dumps(spec, indent=1))
    print(o, 'scale', s, 'err mm', err*1000, 'params', spec['proc_params'], Path(spec['proc_params']).exists())
