"""Replay-independent check of identical checkpoint and complete finite repaired rollout."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

b=Path('outputs/track3/episode_000016')
o=b/'substep_stability_v1'
reports=[json.loads((o/f'substeps_{n}/report.json').read_text()) for n in [5,20]]
policy_hash=hashlib.sha256((b/'training/checkpoints/policy_60000.safetensors').read_bytes()).hexdigest()
checks={}
for n,r in zip([5,20],reports):
    t=pq.read_table(o/f'substeps_{n}/episode_000016.parquet')
    checks[f'checkpoint_{n}']=t.schema.metadata[b'flash_chord.checkpoint_sha256'].decode()==policy_hash
    checks[f'rows_{n}']=len(t)==346
    checks[f'no_retraining_{n}']=not r['training_repeated']
    checks[f'valid_mask_{n}']=all(v==(n==20) for v in t['environment_valid'].to_pylist())
    c=np.load(o/f'substeps_{n}/rollout_checkpoint.npz')
    checks[f'invalid_mask_{n}']=c['invalid_worlds'].tolist()==([0] if n==5 else [])
checks['original_failed']=reports[0]['state']=='failed'
checks['candidate_completed']=reports[1]['state']=='completed'
checks['candidate_trajectory']=json.loads((o/'substeps_20/episode_000016.audit.json').read_text())['all_checks_passed']
checks['frozen_sources']=all(hashlib.sha256((o/'producer-source'/n).read_bytes()).hexdigest()==h for n,h in json.loads((o/'producer-source/manifest.json').read_text()).items())
report=dict(all_passed=all(checks.values()),checks=checks,quality_pass=False,reference_promoted=False,simulation_override=True)
(o/'independent-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
assert report['all_passed']
