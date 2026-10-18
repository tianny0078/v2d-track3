"""Fixed 2026-10-18 evening Track3 delivery. Root launches --run in host tmux.

Default describes the plan only. Detached fixed stage workers persist log/exit
files. No arbitrary job, GPU selection, training, data download or GT interface.
"""
from datetime import datetime, timezone
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import run_track3_submission_watch as watch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/track3/advice_restart_20261017'
SOURCE_SCENES = BASE/'canonical_all20_scenes_v1'
SCENES = BASE/'canonical_evening_scene_selection_v1'
RECOVERIES = (BASE/'canonical_collision_postqem_v1/plastic_dish_rack',
              BASE/'canonical_collision_postqem_v1/mini_dust_pan',
              BASE/'canonical_collision_qem_extra_v1/white_pot_lid')
COHORT = ROOT/'outputs/track3/scored_baseline_20261017/canonical-checkpoint-v2'
AREA = COHORT/'evening-pipeline-v3'
PREVIOUS_AREA = None
STATUS = ROOT/'notes/track3-evening-delivery.json'
SECOND_STATUS = ROOT/'notes/track3-second-submission-status.json'
SECOND_JOB = ROOT/'notes/track3-second-submission-job.json'
FLASH = ROOT/'.deps/flash-chord-venv/bin/python'
SUBMIT = ROOT/'.deps/v2d-submit-venv/bin/python'
REPORT_TIME = datetime(2026, 10, 19, 3, tzinfo=timezone.utc).timestamp()
WALL_CAP_SECONDS = 86400
STAGES = ('selector', 'replay', 'assemble', 'proxy', 'submission')
CAPS = dict(selector=7200+20*360+300, replay=20*900+600, assemble=1800, proxy=1860, submission=7200)
SOURCES = ('run_track3_evening_pipeline.py', 'run_track3_canonical_checkpoint_batch.py',
           'evaluate_track3_canonical_checkpoint.py', 'evaluate_track3_proxy_gate.py',
           'run_track3_second_submission.py', 'run_track3_submission_watch.py',
           'build_track3_evening_scene_selection.py', 'recover_track3_canonical_collision.py',
           'recover_track3_canonical_collision_qem.py', 'recover_track3_canonical_collision_qem_extra.py',
           'recover_track3_canonical_collision_postqem.py')


def read(path, default=None):
    return watch.read(path, {} if default is None else default)


def remaining():
    protocol = read(AREA/'protocol.json')
    deadline = protocol.get('wall_deadline_utc')
    value = int(datetime.fromisoformat(deadline).timestamp()-time.time()) if deadline else WALL_CAP_SECONDS
    if value <= 0:
        raise TimeoutError('Fixed task 24-hour wall budget exhausted; preserve logs and receipts')
    return value


def area_arguments():
    args = ['--area', AREA.name]
    if PREVIOUS_AREA:
        args += ['--previous-area', PREVIOUS_AREA.name]
    return args


def update(phase, publish=False, **fields):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.with_suffix('.json.pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = read(STATUS)
        changed = value.get('phase') != phase
        overdue = time.time() >= REPORT_TIME
        first_overdue_snapshot = overdue and not (AREA/'report-2000-snapshot.json').exists()
        value.update(state='executing_evening_delivery', phase=phase, updated_utc=watch.now(),
                     pipeline_pid=os.getpid(), pipeline_directory=str(AREA), cohort=str(COHORT),
                     deadline_pdt='2026-10-18 20:00 PDT', deadline_utc='2026-10-19T03:00:00+00:00',
                     deadline_kind='reporting milestone; execution continues', report_overdue=overdue,
                     task_wall_deadline_utc=read(AREA/'protocol.json').get('wall_deadline_utc'),
                     pipeline_resume='python3 -u scripts/run_track3_evening_pipeline.py --run '+' '.join(area_arguments()), **fields)
        if phase in ('completed_second_submission', 'retained_candidate_not_submitted', 'pipeline_failed', 'wall_budget_exhausted'):
            value['state'] = phase
        watch.save(STATUS, value)
        if first_overdue_snapshot:
            watch.save(AREA/'report-2000-snapshot.json', dict(captured_utc=watch.now(),
                       milestone='2026-10-18 20:00 PDT', execution_continues=True, status=value))
    watch.save(AREA/'checkpoint.json', dict(phase=phase, updated_utc=watch.now(), **fields))
    if publish or changed or first_overdue_snapshot:
        path = ROOT/'notes/track3-publication-request.json'
        request = read(path)
        request.update(requested_utc=watch.now(), reason='Track3 evening pipeline: '+phase,
                       result_file=str(STATUS), respect_recent_backend_failure_backoff=True)
        watch.save(path, request)


def scene_ids(manifest):
    rows = manifest.get('episodes', [])
    if isinstance(rows, dict):
        return {int(str(key).replace('episode_', '')) for key in rows}, len(rows)
    return {int(str(row.get('episode', row.get('episode_id'))).replace('episode_', '')) for row in rows}, len(rows)


def wait_scenes():
    expected = {int(key) for key in read(ROOT/'notes/track3-submission-contract.json')['episodes']}
    # The selector owns bounded waiting for v1 and both recovery jobs. Its own
    # durable exit is checked by run_stage, and it never modifies those jobs.
    manifest = read(SCENES/'manifest.json')
    ids, count = scene_ids(manifest)
    if manifest.get('state') != 'completed_bounded_scene_selection' or count != 20 or ids != expected:
        raise ValueError('Completed selector did not provide the exact 20 scored scene entries')
    return manifest


def alive(pid):
    if not isinstance(pid, int):
        return False
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0] != 'Z'
    except OSError:
        return False


def wait_gpu_one():
    started = time.monotonic()
    while True:
        remaining()
        uuid = subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'], text=True, timeout=15).strip()
        rows = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'], text=True, timeout=15).splitlines()
        pids = [row.split(',')[1].strip() for row in rows if row.split(',')[0].strip() == uuid]
        if not pids:
            return
        update('waiting_for_physical_gpu1', gpu1_process_ids=pids)
        if time.monotonic()-started >= 7200:
            raise TimeoutError('GPU1 wait exceeded two hours; no process was killed')
        time.sleep(min(30, remaining()))


def proof_url():
    proof = read(COHORT/'production-code.json')
    value = proof.get('code_commit_url')
    # The previous public commit may exist while root publishes the selector
    # sources. Wait for its real source map, rather than freezing that old URL.
    manifest = read(COHORT/'artifact-manifest.json')
    episodes = manifest.get('episodes',{})
    if len(episodes) != 20:
        return None
    required = {}
    for entry in episodes.values():
        for name, digest in entry.get('producer_source_sha256',{}).items():
            if name in required and required[name] != digest:
                return None
            required[name] = digest
    if not required or any(proof.get('source_sha256',{}).get(name) != digest for name,digest in required.items()):
        return None
    if isinstance(value, str) and re.fullmatch(r'https://github\.com/tianny0078/v2d-track3/commit/[0-9a-f]{40}', value):
        return value
    return None


def command(stage, seconds):
    if stage == 'selector':
        return [str(FLASH), '-u', str(ROOT/'scripts/build_track3_evening_scene_selection.py'),
                '--source', str(SOURCE_SCENES), '--output', str(SCENES),
                *[arg for folder in RECOVERIES for arg in ('--recovery', str(folder))],
                '--wait-seconds', '7200', '--episode-timeout', '360']
    batch = [str(FLASH), '-u', str(ROOT/'scripts/run_track3_canonical_checkpoint_batch.py'),
             '--scene-manifest', str(SCENES/'manifest.json'), '--output', str(COHORT),
             '--substeps', '20', '--device', 'cuda:0', '--episode-timeout', '900']
    if stage == 'replay':
        return batch
    if stage == 'assemble':
        url = read(AREA/'protocol.json').get('code_commit_url')
        if not url:
            raise ValueError('Actual production-code commit URL has not been frozen')
        return batch+['--assemble-only', '--pack', '--code-commit', url, '--online-checks']
    if stage == 'proxy':
        return [str(SUBMIT), '-u', str(ROOT/'scripts/evaluate_track3_proxy_gate.py'), '--cohort', str(COHORT),
                '--packed', str(COHORT/'official-package/canonical_checkpoint.parquet'), '--output', str(COHORT/'proxy'),
                '--surface-samples', '512', '--wall-budget-seconds', '1800', '--resume']
    if stage == 'submission':
        return [str(SUBMIT), '-u', str(ROOT/'scripts/run_track3_second_submission.py'), '--watch', '--submit-on-ready',
                '--cohort', str(COHORT), '--status', str(SECOND_STATUS), '--job', str(SECOND_JOB), '--profile', 'second',
                '--max-seconds', str(seconds), '--poll-seconds', '60']
    raise ValueError('Unknown fixed stage')


def verify_sources():
    protocol = read(AREA/'protocol.json')
    bindings = protocol['source_sha256']
    for name, digest in bindings.items():
        if watch.digest(ROOT/'scripts'/name) != digest:
            raise ValueError('Pinned pipeline/producer changed: '+name+'; retain original candidate')
    if protocol.get('scene_manifest_sha256'):
        if watch.digest(SCENES/'manifest.json') != protocol['scene_manifest_sha256']:
            raise ValueError('Frozen selected scene manifest changed; preserve candidate')


def stage_worker(stage):
    descriptor = AREA/'stages'/stage/'command.json'
    desc = read(descriptor)
    try:
        verify_sources()
        assert desc['argv'] == command(stage, desc['timeout_seconds']) and desc['cwd'] == str(ROOT)
        # Wait until the parent durably records this worker before creating a
        # child; this avoids a launch/process-record race during recovery.
        started = time.monotonic()
        while read(Path(desc['process_file'])).get('worker_pid') != os.getpid():
            if time.monotonic()-started > 10:
                raise RuntimeError('Parent did not persist worker ownership')
            time.sleep(.05)
        # Selector is CPU-only. Every later stage sees physical GPU1 only.
        threads = '2' if stage == 'selector' else '4'
        os.environ.update(CUDA_VISIBLE_DEVICES='' if stage == 'selector' else '1', XLA_PYTHON_CLIENT_PREALLOCATE='false',
                          OPENBLAS_NUM_THREADS=threads, OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, WANDB_MODE='disabled')
        return watch.stage_worker(descriptor)
    except Exception as error:
        watch.save(Path(desc['exit_file']), dict(name=stage, exit_code=125, error_type=type(error).__name__, completed_utc=watch.now()))
        return 125


def run_stage(stage):
    folder = AREA/'stages'/stage
    folder.mkdir(parents=True, exist_ok=True)
    exitfile = folder/'exit.json'
    processfile = folder/'process.json'
    done = read(exitfile)
    if done:
        if done['exit_code']:
            raise RuntimeError(stage+' already failed; original log/exit retained, no automatic rerun')
        return
    process = read(processfile)
    if not alive(process.get('worker_pid')):
        if alive(process.get('child_pid')):
            raise RuntimeError(stage+' child survived without worker; inspect it rather than duplicate it')
        if stage == 'replay':
            wait_gpu_one()
        verify_sources()
        seconds = min(CAPS[stage], remaining())
        desc = dict(name=stage, argv=command(stage, seconds), cwd=str(ROOT), log=str(folder/'run.log'),
                    process_file=str(processfile), exit_file=str(exitfile), timeout_seconds=seconds,
                    created_utc=watch.now(), CUDA_VISIBLE_DEVICES='' if stage == 'selector' else '1',
                    physical_gpu=None if stage == 'selector' else 1)
        old = read(folder/'command.json')
        if old and stage == 'submission':
            # Interrupted reservation/receipt recovery is exclusively the same coordinator.
            # Keep its original wall cap/command descriptor instead of extending it.
            desc = old
        watch.save(folder/'command.json', desc)
        child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--worker-stage', stage, *area_arguments()],
                                 cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, start_new_session=True)
        watch.save(processfile, dict(worker_pid=child.pid, started_utc=watch.now()))
    update(stage+'_running', publish=True, active_stage=stage, stage_directory=str(folder))
    while True:
        remaining()
        done = read(exitfile)
        if done:
            if done['exit_code']:
                raise RuntimeError(stage+' exit '+str(done['exit_code'])+'; inspect persistent stage log')
            verify_sources()
            update(stage+'_completed', publish=True, active_stage=stage, last_stage_exit=done)
            return
        process = read(processfile)
        if not alive(process.get('worker_pid')):
            raise RuntimeError(stage+' worker disappeared without durable exit; preserve logs/receipt')
        fields = dict(active_stage=stage, stage_process=process)
        if stage == 'selector':
            fields.update(scene_selection_job=read(SCENES/'job.json'),
                          canonical_episodes=len(read(SOURCE_SCENES/'manifest.json').get('episodes',[])),
                          scene_selection_checkpoint=read(SCENES/'checkpoint.json'))
        if stage == 'replay':
            fields['replay_checkpoint'] = read(COHORT/'batch-checkpoint.json')
        if stage == 'submission':
            status = read(SECOND_STATUS)
            fields.update(second_submission_state=status.get('state'), second_submissions=status.get('submissions'),
                          new_submission_uploaded=any(item.get('ref') for item in status.get('submissions',{}).values()))
        update(stage+'_running', **fields)
        time.sleep(min(15, remaining()))


def delivery_decision(scene, assembly, official, proxy):
    means = proxy.get('means_equal_weight_per_episode',{}).get('metrics_proxy',{})
    finite = lambda value: isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
    checks = dict(all20_engineering_pass=scene.get('engineering_pass_count') == 20,
                  all36_size_checked=scene.get('oversize_checked_instances') == 36,
                  oversized_instances_zero=scene.get('oversized_instances') == 0,
                  real20_36_9120_assembled=assembly.get('state') == 'completed' and assembly.get('ready_episodes') == 20 and assembly.get('ready_meshes') == 36 and assembly.get('rows') == 9120,
                  all20_proxy_completed=proxy.get('state') == 'completed' and proxy.get('all_20_episodes') is True and len(proxy.get('episodes',[])) == 20,
                  proxy_mean_mppe_at_most15cm=finite(means.get('mppe_cm')) and means['mppe_cm'] <= 15,
                  proxy_mean_rpe_at_most15cm=finite(means.get('rpe_cm')) and means['rpe_cm'] <= 15,
                  official_checks_zero=all(official.get('stages',{}).get(name) == 0 for name in ('pack','preflight_offline','preflight_online','dry_run')))
    return dict(passed=all(checks.values()), checks=checks, proxy_means=means,
                failed_conditions=[key for key,value in checks.items() if not value],
                scope='Root-authorized global #2 conditions. Proxy is self-reference, not GT/Kaggle. No stationary/initial-burst extra veto.',
                official_public_source_verification='The second coordinator re-verifies actual source commit before any upload.')


def run():
    AREA.mkdir(parents=True, exist_ok=True)
    with (AREA/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            remaining()
            protocol_path = AREA/'protocol.json'
            protocol = read(protocol_path)
            sources = {name:watch.digest(ROOT/'scripts'/name) for name in SOURCES}
            if protocol:
                assert protocol['source_sha256'] == sources, 'Pinned task sources changed; preserve previous output'
            else:
                previous = read(PREVIOUS_AREA/'protocol.json') if PREVIOUS_AREA else {}
                if PREVIOUS_AREA:
                    assert previous.get('started_utc') and previous.get('wall_deadline_utc'), 'Previous fixed-task budget evidence missing'
                elif (COHORT/'evening-pipeline/protocol.json').exists():
                    raise ValueError('Pass --previous-area evening-pipeline to preserve original 24-hour task cap')
                protocol = dict(task='20261018_evening_fixed_canonical_checkpoint_delivery', source_sha256=sources,
                                started_utc=previous.get('started_utc',watch.now()), report_due_utc='2026-10-19T03:00:00+00:00', gpu=1,
                                wall_cap_seconds=WALL_CAP_SECONDS,
                                wall_deadline_utc=previous.get('wall_deadline_utc') or datetime.fromtimestamp(time.time()+WALL_CAP_SECONDS, timezone.utc).isoformat(),
                                migrated_from=str(PREVIOUS_AREA) if PREVIOUS_AREA else None,
                                previous_protocol_sha256=watch.digest(PREVIOUS_AREA/'protocol.json') if PREVIOUS_AREA else None,
                                source_v1_manifest=str(SOURCE_SCENES/'manifest.json'),
                                scene_manifest=str(SCENES/'manifest.json'), selector_exit=str(AREA/'stages/selector/exit.json'))
                watch.save(protocol_path, protocol)
            remaining()
            update('waiting_for_scene_selection', publish=True)
            run_stage('selector')
            wait_scenes()
            digest = watch.digest(SCENES/'manifest.json')
            assert not protocol.get('scene_manifest_sha256') or protocol['scene_manifest_sha256'] == digest
            protocol['scene_manifest_sha256'] = digest
            watch.save(protocol_path, protocol)
            run_stage('replay')
            start = time.monotonic()
            while not proof_url():
                remaining()
                update('waiting_for_public_producing_proof')
                if time.monotonic()-start >= 7200:
                    raise TimeoutError('Public producer proof wait exceeded two hours')
                time.sleep(min(30, remaining()))
            url = proof_url()
            assert not protocol.get('code_commit_url') or protocol['code_commit_url'] == url, 'Producing commit changed after task freeze'
            protocol.update(code_commit_url=url, production_code_sha256=watch.digest(COHORT/'production-code.json'))
            watch.save(protocol_path, protocol)
            run_stage('assemble')
            run_stage('proxy')
            decision = delivery_decision(read(SCENES/'report.json'), read(COHORT/'assembly-report.json'),
                                         read(COHORT/'official-package/report.json'), read(COHORT/'proxy/report.json'))
            watch.save(AREA/'submission-decision.json', decision)
            update('global_candidate_review', publish=True, submission_decision=decision)
            if not decision['passed']:
                update('retained_candidate_not_submitted', publish=True, submission_decision=decision, new_submission_uploaded=False)
                return 2
            run_stage('submission')
            second = read(SECOND_STATUS)
            complete = second.get('state') == 'completed'
            update('completed_second_submission' if complete else 'retained_candidate_not_submitted', publish=True,
                   second_submission_state=second.get('state'), second_receipt=second.get('receipt'),
                   second_submissions=second.get('submissions'),
                   new_submission_uploaded=any(item.get('ref') for item in second.get('submissions',{}).values()),
                   new_official_score={name:item.get('public_score') for name,item in second.get('submissions',{}).items()} if complete else None)
            return 0 if complete else 2
        except Exception as error:
            phase = 'wall_budget_exhausted' if isinstance(error,TimeoutError) else 'pipeline_failed'
            update(phase, publish=True, error_type=type(error).__name__, error=watch.redact(str(error)),
                   recovery_note='Retain stage logs, episode caches and same second receipt. No automatic retraining, upload retry or first-receipt changes.')
            return 3 if isinstance(error,TimeoutError) else 2


def main():
    global AREA, PREVIOUS_AREA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Root-reviewed fixed task; launch this in host tmux')
    parser.add_argument('--area', default='evening-pipeline-v3', help='Fixed-cohort sibling state directory name')
    parser.add_argument('--previous-area', help='Previous idle controller area; inherit its first start and 24-hour cap')
    parser.add_argument('--worker-stage', choices=STAGES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in [args.area,args.previous_area]:
        if name is not None and not re.fullmatch(r'evening-pipeline(?:-v[1-9][0-9]*)?',name):
            parser.error('Area must be a named evening-pipeline sibling inside the fixed cohort')
    AREA = COHORT/args.area
    PREVIOUS_AREA = COHORT/args.previous_area if args.previous_area else None
    if PREVIOUS_AREA == AREA:
        parser.error('Previous and active area must differ')
    if args.worker_stage:
        return stage_worker(args.worker_stage)
    if not args.run:
        print(json.dumps(dict(state='prepared_only_not_started', scene_manifest=str(SCENES/'manifest.json'),
              source_v1_manifest=str(SOURCE_SCENES/'manifest.json'), selector_command=command('selector',CAPS['selector']),
              state_area=str(AREA), previous_area=str(PREVIOUS_AREA) if PREVIOUS_AREA else None,
              cohort=str(COHORT), physical_gpu=1,
              report_due_pdt='2026-10-18 20:00 PDT', continue_after_reporting_milestone=True,
              wall_cap_seconds=WALL_CAP_SECONDS, stages=STAGES, timeout_seconds=CAPS,
              status=str(STATUS), run_command='python3 -u scripts/run_track3_evening_pipeline.py --run '+' '.join(area_arguments())), indent=2))
        return 0
    code = run()
    watch.save(AREA/'run.exit.json', dict(exit_code=code, completed_utc=watch.now()))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
