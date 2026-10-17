"""Bounded sequential replay with immutable sources and validated resume guards."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/home/tianyuan/projects/v2d')
BASE = ROOT / 'outputs/track3/scored_baseline_20261017'
OUT = BASE / 'unified-replay-v1'
JOB = ROOT / 'notes/track3-unified-replay-job.json'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p, v):
    t = p.with_suffix('.tmp'); t.write_text(json.dumps(v, indent=2)+'\n'); t.replace(p)

def main():
    import fcntl
    OUT.mkdir(exist_ok=True)
    with (OUT/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        entry = OUT/'producer-source/evaluate_track3_unified.py'
        assert entry.is_file()
        checkpoints = json.loads((BASE/'checkpoints.json').read_text())
        job = json.loads(JOB.read_text())
        assert job['evaluator_sha256'] == sha(entry)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='2', XLA_PYTHON_CLIENT_PREALLOCATE='false',
            OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', WANDB_MODE='disabled')
        contract = json.loads((ROOT/'notes/track3-submission-contract.json').read_text())['episodes']
        # First replay the actual pack offender, then all other policies.
        for episode in ['16'] + [e for e in checkpoints if e != '16']:
            target = OUT/f'episode_{int(episode):06d}'
            target.mkdir(exist_ok=True)
            report = target/'acceptance.json'
            if report.exists():
                accepted = json.loads(report.read_text())
                assert accepted['evaluator_sha256'] == sha(entry)
                assert accepted['checkpoint_sha256'] == sha(Path(checkpoints[episode]))
                assert accepted['trajectory_sha256'] == sha(target/f'episode_{int(episode):06d}.parquet')
                continue
            job.update(state='running', current_episode=int(episode), updated_utc=now())
            save(JOB, job)
            command = [str(ROOT/'.deps/flash-chord-venv/bin/python'), '-u', str(entry),
                '--checkpoint', checkpoints[episode], '--episode', episode, '--output', str(target)]
            if episode == '16': command += ['--substeps', '20']
            with (target/'run.log').open('w') as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
            (target/'run.exit').write_text(str(result.returncode)+'\n')
            if result.returncode:
                job.update(state='failed', updated_utc=now(), failure=f'episode{episode} exit{result.returncode}')
                save(JOB, job)
                raise RuntimeError(job['failure'])
            import pyarrow.parquet as pq
            import numpy as np
            trajectory = target/f'episode_{int(episode):06d}.parquet'
            table = pq.read_table(trajectory)
            meta = table.schema.metadata
            assert meta[b'flash_chord.eval_script_sha256'].decode() == sha(entry)
            assert table.num_rows == contract[episode]['rows']
            for name in table.column_names:
                if name not in ('world_index','frame_index','object_index'):
                    values = table[name].to_numpy()
                    if np.issubdtype(values.dtype, np.number): assert np.isfinite(values).all(), name
            prior = pq.read_table(BASE/'results'/trajectory.name)
            comparison = {}
            for name in table.column_names:
                a, b = table[name].to_numpy(), prior[name].to_numpy()
                comparison[name] = {'exact_equal': bool(np.array_equal(a,b))}
                if np.issubdtype(a.dtype, np.number):
                    comparison[name]['max_abs_difference'] = float(np.max(np.abs(a-b)))
            accepted = json.loads((target/'replay-report.json').read_text())
            accepted.update(rows=table.num_rows, trajectory_sha256=sha(trajectory),
                previous_trajectory_sha256=sha(BASE/'results'/trajectory.name),
                comparison=comparison, completed_utc=now(), hard_checks_passed=True)
            save(report, accepted)
            job.setdefault('completed_episodes', {})[episode] = str(report.relative_to(ROOT))
            save(JOB, job)
        job.update(state='completed', completed_utc=now(), current_episode=None)
        save(JOB, job)

if __name__ == '__main__': main()
