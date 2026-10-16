"""Bounded, resumable official Sharpa FlashSAC diagnostic on public episode 11.

This never changes vendor sources, claims training readiness, or uploads anything.
All physics/learning/inference is performed by the unmodified official CLI.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / 'vendor/video_to_data/robotic_grounding/flash_chord'
OUT = ROOT / 'outputs/track3/episode_000011/baseline_probe'
JOB = ROOT / 'notes/track3-baseline-probe-job.json'
PYTHON = ROOT / '.deps/flash-chord-venv/bin/python'
PARQUET = ROOT / 'outputs/track3/episode_000011/flashchord_reference_preflight/episode11_interface_only.parquet'
BUDGET = 1140


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overrides():
    return [
        'experiment=sharpa_flash_sac', f'task.parquet={PARQUET}',
        'scene.world_count=1', 'scene.support=false', 'scene.decompose_objects=false',
        'scene.contact_friction=0.5', 'reset.seed=42', 'reset.voc_decay_steps=0',
        'reset.always_reset_to_first_frame=true',
        'env.voc_scale=0.0', 'env.voc.max_force=0.0', 'env.voc.max_torque=0.0',
        'voc_scale_override=0.0', 'logging.mode=disabled', 'logging.log_interval=8',
        'training.total_environment_steps=64', 'training.updates_per_collection=1.0',
        'training.replay.capacity=128', 'training.replay.minimum_size=8',
        'training.replay.batch_size=8', 'training.checkpoint.save_interval_environment_steps=16',
        'training.checkpoint.save_replay=true',
        f'output_dir={OUT / "checkpoints"}', f'hydra.run.dir={OUT / "hydra_train"}',
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    old = json.loads((OUT / 'report.json').read_text()) if (OUT / 'report.json').exists() else {}
    first_started = old.get('first_started_utc', old.get('started_utc', now()))
    already_elapsed = max(0.0, (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(first_started)).total_seconds())
    deadline = started + max(0.0, BUDGET - already_elapsed)
    report = {
        'state': 'running', 'started_utc': now(), 'first_started_utc': first_started,
        'scope': 'diagnostic baseline only',
        'training_ready': False, 'task_success_claimed': False, 'submission_ready': False,
        'episode': 11, 'episode_is_scored': False, 'budget_seconds': BUDGET,
        'physical_gpu': 3, 'world_count': 1, 'official_recipe': 'sharpa_flash_sac',
        'training_step_budget': 64, 'resume_terminal_step_budget': 80,
        'training_initialization': 'Always reset to reference frame zero; reset finger openness retains the official 0.7 default.',
        'wrench_semantics': 'Unmodified upstream body/link-origin wrench; this is not a COM conclusion.',
        'object_assistance': 'Zero env VOC scale, zero max force/torque, zero reset decay, zero curriculum override.',
        'quality_gates': {'geometry_material': False, 'object_trajectory': False,
                          'unknown_contacts': False, 'COM_wrench': False},
        'contact_label_limitation': 'Official loader ignores unknown masks; this diagnostic retains its placeholder behavior and is not valid contact-ground-truth training.',
        'stages': [], 'previous_runs': old.get('previous_runs', []) + ([old] if old else []),
    }
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='3', XLA_PYTHON_CLIENT_PREALLOCATE='false',
               OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', HYDRA_FULL_ERROR='1',
               WANDB_MODE='disabled', JAX_COMPILATION_CACHE_DIR=str(OUT / 'jax_cache'))
    job = json.loads(JOB.read_text()) if JOB.exists() else {}

    def snapshot():
        report['updated_utc'] = now()
        report['elapsed_seconds'] = time.monotonic() - started
        report['total_initial_budget_elapsed_seconds'] = already_elapsed + report['elapsed_seconds']
        report['checkpoint_files'] = [str(p) for p in sorted((OUT / 'checkpoints').glob('*.safetensors'))]
        save(OUT / 'report.json', report)
        save(OUT / 'probe_state.json', {k: report[k] for k in ['state', 'updated_utc', 'elapsed_seconds', 'checkpoint_files', 'stages']})
        job.update(state=report['state'], updated_utc=now(), stage=report.get('stage'),
                   report=str(OUT / 'report.json'), checkpoints=report['checkpoint_files'],
                   stages=report['stages'], elapsed_seconds=report['elapsed_seconds'])
        save(JOB, job)

    def run_stage(name, command, seconds=None):
        remaining = deadline - time.monotonic()
        if remaining < 5:
            raise TimeoutError('Overall wall budget exhausted before ' + name)
        seconds = min(remaining, seconds or remaining)
        logfile = OUT / (name + '.log')
        stage = {'name': name, 'command': command, 'command_shell': shlex.join(command),
                 'cwd': str(OFFICIAL), 'log': str(logfile), 'started_utc': now(), 'state': 'running'}
        report['stage'] = name
        report['stages'].append(stage)
        print('STAGE', name, shlex.join(command), flush=True)
        begin = time.monotonic()
        with logfile.open('w') as output:
            child = subprocess.Popen(command, cwd=OFFICIAL, env=env, stdout=output,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            stage['pid'] = child.pid
            snapshot()
            try:
                while child.poll() is None:
                    if time.monotonic() - begin >= seconds:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=8)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                        stage['state'] = 'timed_out'
                        break
                    snapshot()
                    time.sleep(3)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=10)
        stage.update(exit_code=child.returncode, completed_utc=now(), elapsed_seconds=time.monotonic() - begin)
        if stage['state'] == 'running':
            stage['state'] = 'completed' if child.returncode == 0 else 'failed'
        (OUT / (name + '.exit')).write_text(str(child.returncode) + '\n')
        snapshot()
        if child.returncode:
            raise RuntimeError(f'{name} exit {child.returncode}; see {logfile}')

    try:
        sources = [Path(__file__), PARQUET, OFFICIAL / 'uv.lock',
                   OFFICIAL / 'scripts/train_flash_sac.py', OFFICIAL / 'scripts/evaluate_policy.py']
        sources += sorted((OFFICIAL / 'src/flash_chord').rglob('*.py'))
        sources += sorted((OFFICIAL / 'src/flash_chord/configs').rglob('*.yaml'))
        report['source_sha256'] = {str(p.relative_to(ROOT)): digest(p) for p in sources}
        report['source_fingerprint'] = hashlib.sha256(json.dumps(report['source_sha256'], sort_keys=True).encode()).hexdigest()
        report['vendor_revision'] = '193382a184308f6800331f3db829f76dc4294cfe'
        kit = ROOT / 'data/track3_sample/toolkit/v2d_submission_kit.zip'
        compatibility = {'kit_pinned_runtime': '8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef',
                         'installed_vendor_revision': report['vendor_revision'], 'files': []}
        with zipfile.ZipFile(kit) as archive:
            for name in archive.namelist():
                if '/flash_chord/' not in name or name.endswith('/'):
                    continue
                relative = name.split('/flash_chord/', 1)[1]
                data = archive.read(name)
                destination = OUT / 'kit_evaluator' / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                original = OFFICIAL / relative
                compatibility['files'].append({'path': relative, 'kit_sha256': hashlib.sha256(data).hexdigest(),
                                               'identical_to_vendor': original.exists() and original.read_bytes() == data})
        report['evaluator_compatibility'] = compatibility
        snapshot()
        base = [str(PYTHON), '-u', str(OFFICIAL / 'scripts/train_flash_sac.py')] + overrides()
        run_stage('compose_training', base + ['--cfg', 'job', '--resolve'], seconds=90)
        (OUT / 'composed_training.yaml').write_bytes((OUT / 'compose_training.log').read_bytes())
        checkpoints = OUT / 'checkpoints'
        candidates = sorted(checkpoints.glob('state_*.safetensors'), key=lambda p: int(p.stem.split('_')[-1]))
        latest = candidates[-1] if candidates else None
        last_steps = int(latest.stem.split('_')[-1]) if latest else 0
        if last_steps < 64:
            command = list(base)
            if latest:
                command += ['training.resume=true', f'training.checkpoint_path={latest}',
                            f'training.checkpoint.load_replay_path={checkpoints / ("replay_" + str(last_steps) + ".safetensors")}']
            run_stage('train64', command, seconds=840)
            last_steps = 64
        report['policy_training_pass'] = (checkpoints / f'policy_{last_steps}.safetensors').exists()
        if last_steps < 80:
            state = checkpoints / f'state_{last_steps}.safetensors'
            command = base + ['training.total_environment_steps=80', 'training.resume=true',
                              f'training.checkpoint_path={state}',
                              f'training.checkpoint.load_replay_path={checkpoints / ("replay_" + str(last_steps) + ".safetensors")}',
                              f'hydra.run.dir={OUT / "hydra_resume"}']
            report['actual_resume_command'] = shlex.join(command)
            job['actual_resume_command'] = report['actual_resume_command']
            snapshot()
            run_stage('resume80', command, seconds=240)
        report['resume_pass'] = (checkpoints / 'state_80.safetensors').exists()
        actor = checkpoints / 'policy_80.safetensors'
        evaluate = [str(PYTHON), '-u', str(OUT / 'kit_evaluator/scripts/evaluate_policy.py'),
                    f'evaluation.checkpoint={actor}', 'evaluation.world_count=1',
                    'evaluation.reset_mode=explicit', f'evaluation.metrics_output={OUT / "evaluation.json"}',
                    f'evaluation.object_trajectories_output={OUT / "episode_000011.parquet"}',
                    f'hydra.run.dir={OUT / "hydra_evaluate"}', 'logging.mode=disabled']
        run_stage('compose_kit_evaluator', evaluate + ['--cfg', 'job', '--resolve'], seconds=60)
        report['evaluator_compatibility']['actual_cli_compose_pass'] = True
        run_stage('policy_rollout', evaluate, seconds=240)
        report['policy_rollout_pass'] = (OUT / 'evaluation.json').exists() and (OUT / 'episode_000011.parquet').exists()
        report['state'] = 'completed'
    except Exception:
        report['state'] = 'failed'
        report['error'] = traceback.format_exc()
        print(report['error'], flush=True)
    report['completed_utc'] = now()
    snapshot()
    print('RESULT', json.dumps({k: v for k, v in report.items() if k not in ['source_sha256', 'previous_runs']}), flush=True)
    return 0 if report['state'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
