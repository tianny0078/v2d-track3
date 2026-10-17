"""Audit completed scored outputs; never substitute a reference for policy output."""
import argparse
import datetime
import json
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--episode', type=int, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
b = root / f'outputs/track3/episode_{a.episode:06d}'
contract = json.loads((root/'notes/track3-submission-contract.json').read_text())['episodes'][str(a.episode)]
frames = contract['scored_frames']
python = root/'.deps/flash-chord-venv/bin/python'
def run(script, *args, native=False):
    subprocess.run([str(python) if native else 'python3', str(root/'scripts'/script), *map(str,args)], check=True, cwd=root)
def save(name, report):
    (b/name).write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
for stage in ['stereo', 'masks', 'hands']:
    assert (b/f'{stage}-job.exit').read_text().strip() == '0', stage
run('audit_track3_episode_rgb.py', b, '--expected-frames', frames)
run('audit_track3_episode_stereo.py', b, '--expected-frames', frames)
run('audit_track3_episode_hands.py', b, '--expected-frames', frames)
run('audit_track3_episode_masks.py', b)
perception_path = b/'perception-completion-report.json'
perception = json.loads(perception_path.read_text()) if perception_path.exists() else {}
perception.update(episode=a.episode, frames=frames, checked_utc=stamp, hard_checks_passed=True, quality_pass=False, reference_promoted=False, scope='Full sequence input integrity; accuracy unverified')
save('perception-completion-report.json', perception)
assert (b/'training/job.exit').read_text().strip() == '0'
run('audit_track3_downstream_training.py', '--output-dir', b/'training', '--reference-mode', 'ground_truth_unknown', '--reference-parquet-suffix', f'episode_{a.episode:06d}/reference/episode_reference.parquet', native=True)
policy = b/f'training/episode_{a.episode:06d}.parquet'
run('audit_track3_policy_trajectory.py', policy, '--frames', frames, '--objects', len(contract['slots']), native=True)
evaluation = json.loads((b/'training/evaluation.json').read_text())
save('training-completion-report.json', dict(episode=a.episode, frames=frames, rows=contract['rows'], completed_utc=stamp, hard_checks_passed=True, evaluation=evaluation, quality_pass=False, reference_promoted=False, ground_truth_force_known=False, object_assistance=False, checkpoint=str(b/'training/checkpoints'), resume_command=f'bash scripts/run_track3_ep{a.episode}_full_pipeline.sh', limits='Coarse reconstruction and retreat reference have no verified grasp. Full rollout is not a task-success claim.'))
