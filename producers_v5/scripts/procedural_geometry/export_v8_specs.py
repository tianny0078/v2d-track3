"""Export the adopted multiep_recon_v1 param-stage deformation of each object as a torch-free JSON spec
(outputs/track3/procedural_geometry_v8/<obj>/deform_spec.json) and verify it reproduces final_4096.obj from the
procedural visual_4096 (x sbar). Run with system python3 (torch for stage_param.pt)."""
import json, sys
from pathlib import Path
import numpy as np, trimesh
sys.path.insert(0, str(Path(__file__).parent))
from v8_deform import load_run, ADOPTED, MR, R, DeformNP

OUT = R/'outputs/track3/procedural_geometry_v8'
for o in ADOPTED:
    D, sbar, pm, pj, pp, st = load_run(o)
    c4 = trimesh.load(pm.parent/'visual_4096.obj', process=False, force='mesh')
    fin = trimesh.load(MR/o/'final_4096.obj', process=False, force='mesh')
    V = D(c4.vertices*sbar, c4.faces); err = float(np.abs(V - fin.vertices).max())
    s = dict(object=o, adopted_run=str(ADOPTED[o]), proc_mesh=str(pm), proc_params=str(pj), final_4096=str(MR/o/'final_4096.obj'),
             spec=D.spec(), reproduce_final4096_max_err_mm=err*1000, faces_identical=bool(np.array_equal(fin.faces, c4.faces)))
    V2 = DeformNP.from_spec(json.loads(json.dumps(s['spec'])))(c4.vertices*sbar, c4.faces)
    s['spec_roundtrip_max_err_mm'] = float(np.abs(V2 - fin.vertices).max()*1000)
    assert err < 1e-6 and s['spec_roundtrip_max_err_mm'] < 1e-3, (o, err)
    (OUT/o).mkdir(parents=True, exist_ok=True); (OUT/o/'deform_spec.json').write_text(json.dumps(s, indent=1))
    print(o, D.kind, f'sbar {sbar:.4f} err {err*1e3:.2e} mm faces_identical {s["faces_identical"]}')
