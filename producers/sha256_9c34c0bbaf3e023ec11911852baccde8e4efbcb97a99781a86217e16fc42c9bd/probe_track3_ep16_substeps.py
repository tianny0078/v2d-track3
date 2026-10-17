"""Same learned policy, explicit simulation-substep diagnostic; no checkpoint edits."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--substeps', type=int, choices=[5,20], required=True)
a = p.parse_args()
b = Path('outputs/track3/episode_000016/training').resolve()
out = b.parent/'substep_stability_v1'/f'substeps_{a.substeps}'
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0,str(b/'producer-source'))
import track3_evaluation_harness as h
original_config = h.checkpoint_config
def config(metadata):
    c = original_config(metadata)
    assert c['env']['sim']['substeps'] == 5
    c['env']['sim']['substeps'] = a.substeps
    return c
h.checkpoint_config = config
original_record = h.record_object_pose_rollout
def record(*args, **kwargs):
    r = original_record(*args, **kwargs)
    # Official recorder zero-fills invalid worlds. Preserve its invalid mask;
    # these values must never be mistaken for finite valid policy output.
    np.savez_compressed(out/'rollout_checkpoint.npz', achieved_pose_w=r.achieved_pose_w,
                        reference_pose_w=r.reference_pose_w, invalid_worlds=r.non_finite_worlds,
                        tracking_error=r.tracking_error)
    h.write_object_trajectories_parquet(out/'episode_000016.parquet',r,
        provenance={'diagnostic_substeps':a.substeps,'checkpoint_sha256':hashlib.sha256((b/'checkpoints/policy_60000.safetensors').read_bytes()).hexdigest()})
    return r
h.record_object_pose_rollout = record
report = dict(substeps=a.substeps, original_substeps=5, policy_checkpoint=str(b/'checkpoints/policy_60000.safetensors'),
              training_repeated=False, object_assistance=False, quality_pass=False, reference_promoted=False,
              simulation_override=a.substeps!=5, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
try:
    r=h.evaluate_checkpoint(b/'checkpoints/policy_60000.safetensors',world_count=1,start_frame=0,reset_mode='explicit',eval_script=Path(__file__))
    h.write_evaluation_report(out/'evaluation.json',r)
    report.update(state='completed',metrics=h.evaluation_scalars(r),non_finite_worlds=list(r.non_finite_worlds))
except Exception as e:
    report.update(state='failed',error=repr(e))
    if (out/'rollout_checkpoint.npz').exists():
        report['non_finite_worlds']=np.load(out/'rollout_checkpoint.npz')['invalid_worlds'].tolist()
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
