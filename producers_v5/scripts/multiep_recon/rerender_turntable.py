"""Re-render turntable_before_after.png of a finished multiep_recon output dir (after the auto-distance fix).
Usage: rerender_turntable.py <out_dir/obj> [--gpu 0]"""
import json, sys, os
from pathlib import Path
import numpy as np, trimesh, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
d = Path(sys.argv[1]); os.environ.setdefault('CUDA_VISIBLE_DEVICES', sys.argv[3] if len(sys.argv) > 3 else '0')
import recon as RC, export as EX
r = json.loads((d/'report.json').read_text()); a = r['args']
init = dict(np.load(d/'init.npz')); st = torch.load(d/'stage_param.pt', weights_only=False)
fr = torch.load(d/'stage_free.pt', weights_only=False) if (d/'stage_free.pt').exists() else st
q = trimesh.load(json.loads(open(Path(a['proc_mesh'])).read()) if False else a['proc_mesh'], process=False, force='mesh')
pp = json.loads(Path(a['proc_params']).read_text())
if 'axis' in pp: ax = np.array(pp['axis'], float)
else: ax = np.linalg.eigh(np.cov((q.vertices - q.vertices.mean(0)).T))[1][:, -1]
cx = RC.Ctx()
sub = trimesh.load(json.loads((d/'report.json').read_text())['provenance']['episodes'][str(r['episodes'][0])]['submitted_mesh'][0], process=False, force='mesh')
p = d/'turntable_before_after.png'
if p.exists(): p.rename(d/'turntable_before_after.v0-fixeddist.png')
EX.turntable(cx, [(f'before: init mesh x {float(init["sbar"]):.3f} + init colours', init['V'], init['F'], init['albedo'], False),
                  ('after: multi-episode recon (albedo)', fr['V'], fr['F'], fr['albedo'], False),
                  ('after: geometry', fr['V'], fr['F'], fr['albedo'], True),
                  ('before: submitted geometry', sub.vertices, sub.faces, np.full((len(sub.vertices), 3), 0.5), True)], ax, p)
print('wrote', p)
