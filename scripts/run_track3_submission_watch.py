"""Persistently package one complete real Track 3 cohort and submit it once.

--inspect only checks local readiness. --launch --submit-on-ready starts a bounded
tmux watcher. No upload is attempted without all artifacts, public producer code,
official preflight and dry-run. Once upload intent exists, restart only observes
that attempt and polls its receipt; ambiguous uploads are never retried here.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import datetime as dt
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import selectors
import shlex
import signal
import struct
import subprocess
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / 'toolkit/8d2bce64/v2d_submission_kit'
PYTHON = ROOT / '.deps/v2d-submit-venv/bin/python'
DEFAULT_COHORT = ROOT / 'outputs/track3/scored_baseline_20261017'
STATUS = ROOT / 'notes/track3-submission-status.json'
METRICS = ('add_auc', 'spider_sr', 'maniptrans_sr', 'rpe_cm', 'mppe_cm', 'cd_o_cm')
SHA = re.compile(r'^[0-9a-f]{64}$')


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def resolve(cohort, path):
    value = Path(path)
    return value.resolve() if value.is_absolute() else (cohort / value).resolve()


def redact(line):
    line = re.sub(r'(?i)(bearer\s+)[A-Za-z0-9._~+/-]+', r'\1[REDACTED]', line)
    return re.sub(r'(?i)((?:access_token|api_key|authorization|kaggle_api_token)[\"\s:=]+)[^\s,\"}]+', r'\1[REDACTED]', line)


def receipt_action(receipt, upload_active=False):
    """Decide recovery without ever interpreting an absent ref as permission to upload."""
    if upload_active:
        return 'upload_in_progress'
    if not isinstance(receipt, dict) or not receipt.get('submissions'):
        return 'needs_upload_review'
    entries = receipt['submissions']
    if set(entries) == set(METRICS) and all(v.get('ref') and v.get('state') == 'COMPLETE' for v in entries.values()):
        return 'complete'
    if any(v.get('ref') and v.get('state') not in ('COMPLETE', 'ERROR', 'UNREADABLE') for v in entries.values()):
        return 'poll_existing_ids'
    if any(v.get('state') == 'ERROR' for v in entries.values()):
        return 'scoring_failed'
    return 'needs_upload_review'


def receipt_summary(receipt):
    keys=('slug','ref','state','public_score','error','poll_error','message')
    return {metric:{key:redact(value) if isinstance(value,str) else value for key,value in entry.items() if key in keys}
            for metric,entry in receipt.get('submissions',{}).items()}


def alive(pid):
    if not isinstance(pid, int):
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def official_module():
    spec = importlib.util.spec_from_file_location('track3_official_submit', KIT / 'submit/v2d_submit.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_entry(with_limits=False):
    result = dict(checked_utc=now(), read_only=True, uploads_performed=0, competitions=[])
    hidden = io.StringIO()
    try:
        with redirect_stdout(hidden), redirect_stderr(hidden):
            module = official_module()
            config = read(KIT / 'config/tracks.json')['tracks']
            competitions = dict(config['track_3']['competitions'])
            competitions.update(config['track_3_mesh']['competitions'])
            api = module.build_api()
            if with_limits:
                # Official preflight already checks each entry as well as quota
                # and identity. Avoid a redundant six-request pass immediately
                # before it, which can turn transient API errors into blockers.
                checked = module.preflight(api, competitions)
                for metric, entry in checked.items():
                    entered = entry.get('entered')
                    result['competitions'].append(dict(metric=metric, slug=competitions[metric], entered=entered is True,
                        status='entered' if entered is True else 'not_entered' if entered is False else 'unverified'))
                result['limits_and_identity'] = {metric:{key:entry.get(key) for key in ['slug','remaining','used_today','user','team_name','submissions_readable']}
                                                 for metric,entry in checked.items()}
            else:
                for metric, slug in competitions.items():
                    entered = module.entry_status(api, slug)
                    result['competitions'].append(dict(metric=metric, slug=slug, entered=entered is True,
                        status='entered' if entered is True else 'not_entered' if entered is False else 'unverified'))
        result['all_six_entered'] = len(result['competitions']) == 6 and all(e['entered'] for e in result['competitions'])
    except (Exception, SystemExit) as error:
        result.update(all_six_entered=False, error_type=type(error).__name__)
    finally:
        hidden.close()
    return result


def checkpoint_info(path):
    """Validate the actual safetensors policy without importing a GPU framework."""
    import numpy as np
    with path.open('rb') as stream:
        length_data = stream.read(8)
        if len(length_data) != 8:
            raise ValueError('Checkpoint has no safetensors header')
        length = struct.unpack('<Q', length_data)[0]
        if length > 32 * 1024 * 1024:
            raise ValueError('Checkpoint header exceeds bounded metadata size')
        header = json.loads(stream.read(length))
        offset = 8 + length
        for key, entry in header.items():
            if key == '__metadata__':
                continue
            start, end = entry['data_offsets']
            if not 0 <= start <= end <= path.stat().st_size - offset:
                raise ValueError('Checkpoint contains invalid tensor offsets')
            dtype = {'F16':'<f2', 'F32':'<f4', 'F64':'<f8'}.get(entry['dtype'])
            if dtype:
                array = np.memmap(path, mode='r', dtype=dtype, offset=offset + start,
                                  shape=((end-start) // np.dtype(dtype).itemsize,))
                if not np.isfinite(array).all():
                    raise ValueError('Policy checkpoint has non-finite tensors')
                del array
            elif entry['dtype'] == 'BF16':
                array = np.memmap(path, mode='r', dtype='<u2', offset=offset+start, shape=((end-start)//2,))
                if np.any((array & 0x7f80) == 0x7f80):
                    raise ValueError('Policy checkpoint has non-finite BF16 tensors')
                del array
    metadata = header.get('__metadata__', {})
    if metadata.get('user.checkpoint_kind') != 'policy':
        raise ValueError('Expected actor policy checkpoint, not state/replay')
    steps = int(metadata.get('user.environment_steps', '0'))
    updates = int(metadata.get('user.global_update_step', '0'))
    if steps <= 0 or updates <= 0:
        raise ValueError('Policy has no recorded actual training steps/updates')
    return dict(sha256=digest(path), environment_steps=steps, global_update_step=updates)


def inspect_artifacts(cohort):
    """Do format/provenance checks, leaving research-quality thresholds to reported metrics."""
    import numpy as np
    import pyarrow.parquet as pq
    import trimesh
    contract = read(ROOT / 'notes/track3-submission-contract.json')
    roster = read(KIT / 'data/track_3_evaluation_objects.json')['episodes']
    artifact_path = cohort / 'artifact-manifest.json'
    manifest = read(artifact_path, {})
    checkpoint_map = read(cohort / 'checkpoints.json', {})
    result = dict(checked_utc=now(), expected_episodes=20, expected_meshes=36, expected_rows=9120,
                  ready_episodes=0, ready_meshes=0, rows=0, episodes=[], blockers=[], artifact_sha256={})
    if manifest.get('schema') != 'track3_cohort_artifacts_v1' or manifest.get('state') != 'completed':
        result['blockers'].append('artifact-manifest.json must declare schema track3_cohort_artifacts_v1 and state completed')
    if set(manifest.get('episodes', {})) != set(roster):
        result['blockers'].append('Artifact manifest must contain exactly the 20 scored IDs')
    if set(checkpoint_map) != set(roster):
        result['blockers'].append('checkpoints.json must contain exactly the 20 scored IDs')
    source_hashes = {}
    for episode, names in roster.items():
        info = manifest.get('episodes', {}).get(episode, {})
        row = dict(episode=int(episode), state='waiting', ready_meshes=0, blockers=[])
        expected_frames = contract['episodes'][episode]['scored_frames']
        try:
            if info.get('state') != 'completed' or info.get('policy_rollout_performed') is not True or info.get('object_assistance') is not False:
                raise ValueError('Need completed actual policy rollout with object_assistance=false')
            if info.get('object_names') != names:
                raise ValueError('Object order differs from official scored roster')
            trajectory = resolve(cohort, info['trajectory'])
            standard = cohort / f'results/episode_{int(episode):06d}.parquet'
            if trajectory != standard.resolve():
                raise ValueError('Trajectory must use the standard cohort/results filename')
            checkpoint = resolve(cohort, info['checkpoint'])
            if resolve(cohort, checkpoint_map[episode]) != checkpoint:
                raise ValueError('checkpoints.json disagrees with episode artifact checkpoint')
            cp = checkpoint_info(checkpoint)
            if info.get('training_environment_steps') != cp['environment_steps']:
                raise ValueError('Declared training steps differ from actual checkpoint')
            table = pq.read_table(trajectory)
            cols = table.to_pydict()
            required = ['environment_id','reference_frame','object_body_id','object_body_name','environment_valid']
            required += ['achieved_position_' + c for c in 'xyz'] + ['achieved_quaternion_' + c for c in 'xyzw']
            if any(key not in cols for key in required):
                raise ValueError('Trajectory lacks official evaluator columns')
            if table.num_rows != expected_frames * len(names):
                raise ValueError('Trajectory row count differs from complete frame/object grid')
            grid = set(zip(cols['environment_id'], cols['reference_frame'], cols['object_body_id']))
            wanted = {(0, f, o) for f in range(expected_frames) for o in range(len(names))}
            if grid != wanted or len(grid) != table.num_rows:
                raise ValueError('Trajectory grid has missing, duplicate or unexpected rows')
            if not all(cols['environment_valid']):
                raise ValueError('A simulated world is invalid; cannot submit sanitised zeros')
            poses = np.stack([cols[c] for c in required if c.startswith('achieved_')], axis=-1)
            if not np.isfinite(poses).all() or np.any(np.abs(np.linalg.norm(poses[:,3:], axis=1)-1) > 1e-3):
                raise ValueError('Achieved poses are non-finite or quaternions are not unit xyzw')
            if any(name != names[slot] for name, slot in zip(cols['object_body_name'], cols['object_body_id'])):
                raise ValueError('Trajectory body-name/slot ordering is inconsistent')
            metadata = {k.decode():v.decode() for k,v in (table.schema.metadata or {}).items()}
            if metadata.get('flash_chord.position_unit') != 'meter' or metadata.get('flash_chord.quaternion_order') != 'xyzw':
                raise ValueError('Trajectory metadata must declare metre position and xyzw quaternion order')
            if metadata.get('flash_chord.checkpoint_sha256') != cp['sha256']:
                raise ValueError('Trajectory provenance does not match actual policy checkpoint')
            for key in ['eval_code_sha256','eval_script_sha256']:
                if not SHA.fullmatch(metadata.get('flash_chord.' + key, '')):
                    raise ValueError('Trajectory lacks official evaluator SHA256 provenance')
            if 'flash_chord.episode_index' in metadata and int(metadata['flash_chord.episode_index']) != int(episode):
                raise ValueError('Trajectory metadata identifies a different episode')
            for name in names:
                mesh_info = info.get('meshes', {}).get(name, {})
                mesh_path = resolve(cohort, mesh_info['path'])
                if mesh_path.parent != (cohort / f'meshes/episode_{int(episode):06d}').resolve() or mesh_path.stem != name:
                    raise ValueError('Mesh must use standard episode/object path')
                reconstruction = resolve(cohort, mesh_info['reconstruction_source'])
                if not reconstruction.is_file():
                    raise ValueError('Actual mesh reconstruction report/source artifact is missing')
                inputs = [resolve(cohort, p) for p in mesh_info['source_inputs']]
                if not inputs or not any(p.is_file() and p.name == f'episode_{int(episode):06d}.mp4'
                                         and p.is_relative_to(ROOT / 'data/track3_scored/track_3/evaluation/videos') for p in inputs):
                    raise ValueError('Mesh reconstruction provenance must identify actual scored input video')
                mesh = trimesh.load(mesh_path, force='mesh', process=False)
                v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
                if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3 or not len(v) or not len(f):
                    raise ValueError('Mesh has no actual triangle surface')
                if not np.isfinite(v).all() or f.min() < 0 or f.max() >= len(v):
                    raise ValueError('Mesh has non-finite vertices or invalid face indices')
                area = np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]], v[f[:,2]]-v[f[:,0]]), axis=1)
                if not np.any(area > 1e-12):
                    raise ValueError('Mesh has no positive-area triangles')
                result['artifact_sha256'][str(mesh_path)] = digest(mesh_path)
                result['artifact_sha256'][str(reconstruction)] = digest(reconstruction)
                row['ready_meshes'] += 1
            producer = info.get('producer_source_sha256', {})
            if not producer or any(not SHA.fullmatch(value) for value in producer.values()):
                raise ValueError('Actual executed producer source hash mapping is required')
            for name, value in producer.items():
                if name in source_hashes and source_hashes[name] != value:
                    raise ValueError('A single producing commit cannot represent conflicting source hashes')
                source_hashes[name] = value
            result['artifact_sha256'][str(checkpoint)] = cp['sha256']
            result['artifact_sha256'][str(trajectory)] = digest(trajectory)
            row.update(state='ready', rows=table.num_rows, checkpoint_sha256=cp['sha256'], training_environment_steps=cp['environment_steps'])
            result['ready_episodes'] += 1
            result['rows'] += table.num_rows
        except (Exception, SystemExit) as error:
            row['blockers'].append(str(error) if isinstance(error, (ValueError, KeyError, FileNotFoundError)) else type(error).__name__)
        result['ready_meshes'] += row['ready_meshes']
        result['episodes'].append(row)
    result['producer_source_sha256'] = source_hashes
    result['ready'] = not result['blockers'] and result['ready_episodes'] == 20 and result['ready_meshes'] == 36 and result['rows'] == 9120
    if artifact_path.exists():
        result['artifact_sha256'][str(artifact_path)] = digest(artifact_path)
    return result


def verify_code(cohort, artifacts):
    path = cohort / 'production-code.json'
    proof = read(path, {})
    url = proof.get('code_commit_url', '')
    match = re.fullmatch(r'https://github\.com/([\w.-]+)/([\w.-]+)/commit/([0-9a-f]{40})', url)
    if not match:
        raise ValueError('production-code.json needs an actual full GitHub commit URL')
    owner, repository, commit = match.groups()
    checkout = Path(proof.get('repo_checkout', str(ROOT / 'repos/v2d-track3')))
    hashes = proof.get('source_sha256', {})
    for name, expected in artifacts['producer_source_sha256'].items():
        if hashes.get(name) != expected or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Production commit source mapping does not cover executed producer hashes')
        source = subprocess.run(['git','-C',str(checkout),'show',f'{commit}:{name}'], capture_output=True)
        if source.returncode or hashlib.sha256(source.stdout).hexdigest() != expected:
            raise ValueError('Public producing commit does not contain the executed source: ' + name)
    for endpoint in [f'https://api.github.com/repos/{owner}/{repository}', f'https://api.github.com/repos/{owner}/{repository}/commits/{commit}']:
        request = Request(endpoint, headers={'User-Agent':'v2d-track3-submission-coordinator','Accept':'application/vnd.github+json'})
        with urlopen(request, timeout=30) as response:
            body = json.load(response)
        if endpoint.endswith(commit):
            if body.get('sha') != commit:
                raise ValueError('Anonymous GitHub commit verification failed')
        elif body.get('private') is not False:
            raise ValueError('Producing GitHub repository is not publicly readable')
    return dict(code_commit_url=url, commit=commit, verified_public_utc=now(), source_files=len(artifacts['producer_source_sha256']),
                production_code_sha256=digest(path))


def stage_worker(descriptor):
    """Independent bounded worker preserves log/exit even if watcher disconnects."""
    desc = read(descriptor)
    process_path = Path(desc['process_file'])
    exit_path = Path(desc['exit_file'])
    start = time.monotonic()
    with Path(desc['log']).open('a') as output:
        output.write(now() + ' stage=' + desc['name'] + '\n'); output.flush()
        child = subprocess.Popen(desc['argv'], cwd=desc['cwd'], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        save(process_path, dict(worker_pid=os.getpid(), child_pid=child.pid, started_utc=now()))
        selector = selectors.DefaultSelector()
        selector.register(child.stdout, selectors.EVENT_READ)
        pending = ''
        timed_out = False
        while selector.get_map():
            if child.poll() is None and time.monotonic() - start >= desc['timeout_seconds']:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
                timed_out = True
            for event, _ in selector.select(timeout=1):
                data = os.read(event.fileobj.fileno(), 65536)
                if not data:
                    selector.unregister(event.fileobj)
                    continue
                pending += data.decode(errors='replace')
                while '\n' in pending:
                    line, pending = pending.split('\n', 1)
                    output.write(redact(line) + '\n'); output.flush()
        if pending:
            output.write(redact(pending) + '\n')
        child.wait()
    result = dict(name=desc['name'], exit_code=child.returncode, timed_out=timed_out,
                  completed_utc=now(), elapsed_seconds=time.monotonic()-start)
    save(exit_path, result)
    return child.returncode if child.returncode is not None else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, default=DEFAULT_COHORT)
    parser.add_argument('--profile', default='baseline')
    parser.add_argument('--inspect', action='store_true', help='One local readiness snapshot, no wait/network/upload')
    parser.add_argument('--check-entry-now', action='store_true', help='Read-only six-competition check')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--launch', action='store_true', help='Launch --watch in persistent server tmux')
    parser.add_argument('--submit-on-ready', action='store_true', help='Use the user-authorized one-time upload after all gates')
    parser.add_argument('--max-seconds', type=int, default=86400)
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--run-stage', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.run_stage:
        return stage_worker(args.run_stage)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.profile) or not 1 <= args.max_seconds <= 86400 or args.poll_seconds < 10:
        parser.error('Profile must be a simple name; wall cap 1..86400 seconds, poll interval >=10 seconds')
    cohort = args.cohort.resolve()
    status = read(STATUS, {})
    status.update(cohort=str(cohort), profile=args.profile, track='track_3', authorized_scope='One complete baseline upload to six metrics',
                  expected_episodes=20, expected_meshes=36, expected_rows=9120, updated_utc=now())
    if args.launch:
        session = 'v2d-t3-first-submission-' + args.profile
        if subprocess.run(['tmux','has-session','-t',session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            raise SystemExit('Matching submission watcher already running')
        log = ROOT / 'logs' / f'track3-first-submission-{args.profile}.log'
        exitfile = log.with_suffix('.exit')
        command = [str(PYTHON),'-u',str(Path(__file__).resolve()),'--watch','--cohort',str(cohort),
                   '--profile',args.profile,'--max-seconds',str(args.max_seconds),'--poll-seconds',str(args.poll_seconds)]
        if args.submit_on_ready:
            command.append('--submit-on-ready')
        shell = shlex.join(command) + ' >> ' + shlex.quote(str(log)) + ' 2>&1; code=$?; printf "%s\\n" "$code" > ' + shlex.quote(str(exitfile))
        status.update(state='launching',session=session,log=str(log),exit_file=str(exitfile),launch_command=command,
                      attach='tmux attach -t '+session,submit_on_ready=args.submit_on_ready)
        save(STATUS, status)
        subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),'bash','-c',shell],check=True)
        print(json.dumps({k:status[k] for k in ['state','session','log','exit_file','attach']},indent=2))
        return 0
    if args.check_entry_now:
        status['entry_check'] = check_entry()
    if not args.watch:
        status.update(state='prepared_not_started', readiness=inspect_artifacts(cohort), updated_utc=now(),
                      upload_attempted=(cohort/'submission/upload-intent.json').exists())
        save(STATUS, status)
        print(json.dumps({k:status[k] for k in ['state','cohort','readiness']},indent=2))
        return 0
    area = cohort / 'submission'
    area.mkdir(parents=True,exist_ok=True)
    lock = (area / '.watch.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Submission watcher already owns this cohort')
    first = status.get('watch_started_utc') or now()
    deadline = dt.datetime.fromisoformat(first).timestamp() + args.max_seconds
    status.update(watch_started_utc=first,expires_utc=dt.datetime.fromtimestamp(deadline,dt.timezone.utc).isoformat(),
                  watcher_pid=os.getpid(),submit_on_ready=args.submit_on_ready,stages=status.get('stages',[]))

    def update(state=None):
        if state:
            status['state'] = state
        status['updated_utc'] = now()
        save(STATUS,status)

    def run_stage(name, argv, folder, timeout):
        descriptor = folder / (name + '.command.json')
        exitfile = folder / (name + '.exit.json')
        processfile = folder / (name + '.process.json')
        done = read(exitfile)
        if done:
            return done['exit_code']
        process = read(processfile,{})
        if not alive(process.get('worker_pid')):
            if name == 'upload' and descriptor.exists():
                raise RuntimeError('Upload worker ended without durable exit; inspect receipt instead of reuploading')
            seconds = max(1,min(timeout,int(deadline-time.time())))
            save(descriptor,dict(name=name,argv=argv,cwd=str(KIT),log=str(folder/(name+'.log')),
                                 exit_file=str(exitfile),process_file=str(processfile),timeout_seconds=seconds))
            child = subprocess.Popen([str(PYTHON),'-u',str(Path(__file__).resolve()),'--run-stage',str(descriptor)],
                                     stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
            save(processfile,dict(worker_pid=child.pid,started_utc=now()))
        status['stage'] = name
        if not any(s.get('command_file') == str(descriptor) for s in status['stages']):
            status['stages'].append(dict(name=name,command_file=str(descriptor),exit_file=str(exitfile),log=str(folder/(name+'.log'))))
        while time.time()<deadline:
            done = read(exitfile)
            if done:
                update(); return done['exit_code']
            current = read(processfile,{})
            if not alive(current.get('worker_pid')):
                raise RuntimeError(name+' worker disappeared; durable logs/receipt retained')
            update(name+'_running'); time.sleep(3)
        raise TimeoutError('Watcher wall budget exhausted; independent stage remains bounded')

    intent_path = area / 'upload-intent.json'
    while time.time()<deadline:
        intent = read(intent_path)
        if intent:
            receipt_path = Path(intent['receipt'])
            folder = Path(intent['attempt_dir'])
            process = read(folder/'upload.process.json',{})
            upload_active = alive(process.get('worker_pid')) and not (folder/'upload.exit.json').exists()
            receipt = read(receipt_path,{})
            action = receipt_action(receipt,upload_active)
            status.update(upload_attempted=True,receipt=str(receipt_path),submissions=receipt_summary(receipt),
                          packed_file=intent['packed_file'],code_commit_url=intent['code_commit_url'])
            update(action)
            if action=='complete':
                status['completed_utc']=now();update('completed');return 0
            if action in ('scoring_failed','needs_upload_review'):
                status['attention_reason']='Inspect saved receipt and exact Kaggle submission IDs. No automatic upload retry is permitted.'
                update(action);return 2
            if action=='poll_existing_ids':
                # A new poll command file is read-only and always uses the same receipt.
                poll_name='poll_'+str(len([p for p in folder.glob('poll_*.command.json')]))
                run_stage(poll_name,[str(PYTHON),str(KIT/'submit/v2d_submit.py'),'--track','track_3','--poll-receipt',str(receipt_path)],folder,1860)
                continue
            time.sleep(min(args.poll_seconds,max(0,deadline-time.time())));continue
        artifacts=inspect_artifacts(cohort)
        status['readiness']=artifacts
        status['upload_attempted']=False
        if not artifacts['ready']:
            update('waiting_for_artifacts')
            time.sleep(min(args.poll_seconds,max(0,deadline-time.time())));continue
        try:
            code=verify_code(cohort,artifacts)
        except (Exception,SystemExit) as error:
            status['production_code_blocker']=str(error) if isinstance(error,ValueError) else type(error).__name__
            update('waiting_for_public_producing_commit')
            time.sleep(min(args.poll_seconds,max(0,deadline-time.time())));continue
        status.pop('production_code_blocker',None)
        status['production_code']=code
        fingerprint=hashlib.sha256(json.dumps(dict(artifacts=artifacts['artifact_sha256'],code=code['code_commit_url']),sort_keys=True).encode()).hexdigest()
        folder=area/('attempt_'+fingerprint[:16]);folder.mkdir(exist_ok=True)
        packed=folder/'track3_baseline.parquet'
        save(folder/'validated-artifacts.json',artifacts)
        stages=[('pack',[str(PYTHON),str(KIT/'tools/pack_submission.py'),'--track','track_3','--results',str(cohort/'results'),
                         '--meshes',str(cohort/'meshes'),'--code-commit',code['code_commit_url'],'-o',str(packed)],900),
                ('preflight',[str(PYTHON),str(KIT/'tools/preflight.py'),'--track','track_3','--results',str(cohort/'results')],180),
                ('dry_run',[str(PYTHON),str(KIT/'submit/v2d_submit.py'),str(packed),'--track','track_3','-m','Track 3 first real scored baseline','--dry-run'],240)]
        failed=False
        for name,command,timeout in stages:
            if run_stage(name,command,folder,timeout):
                status['attention_reason']=name+' failed; see persistent stage log. No upload occurred.'
                update('preparation_failed');failed=True;break
        if failed:
            return 2
        # Detect asset changes during reversible preparation before reserving an upload.
        fresh=inspect_artifacts(cohort)
        if not fresh['ready'] or fresh['artifact_sha256'] != artifacts['artifact_sha256']:
            update('artifacts_changed_during_preparation');continue
        status['entry_check']=check_entry(with_limits=True)
        if not status['entry_check'].get('all_six_entered'):
            update('entry_not_verified');return 2
        if any(row.get('remaining') == 0 for row in status['entry_check'].get('limits_and_identity',{}).values()):
            status['attention_reason']='A selected competition reports zero remaining submission quota. Nothing uploaded.'
            update('quota_exhausted');return 2
        status.update(packed_file=str(packed),packed_sha256=digest(packed),official_pack_pass=True,
                      official_preflight_pass=True,official_dry_run_pass=True)
        if not args.submit_on_ready:
            update('ready_without_upload');return 0
        receipt=folder/'kaggle-receipt.json'
        intent=dict(created_utc=now(),packed_file=str(packed),packed_sha256=digest(packed),receipt=str(receipt),
                    attempt_dir=str(folder),fingerprint=fingerprint,code_commit_url=code['code_commit_url'],metrics=list(METRICS))
        with intent_path.open('x') as handle:
            json.dump(intent,handle,indent=2);handle.flush();os.fsync(handle.fileno())
        status['upload_attempted']=True;update('upload_reserved')
        # No --wait here: upload IDs are persisted first, then polled read-only.
        run_stage('upload',[str(PYTHON),str(KIT/'submit/v2d_submit.py'),str(packed),'--track','track_3',
                           '-m','Track 3 first real scored baseline','--receipt',str(receipt)],folder,1200)
    update('wall_budget_exhausted')
    return 3


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        if '--run-stage' in sys.argv:
            try:
                descriptor=Path(sys.argv[sys.argv.index('--run-stage')+1])
                desc=read(descriptor,{})
                save(Path(desc['exit_file']),dict(name=desc.get('name'),exit_code=125,error_type=type(error).__name__,completed_utc=now()))
            except Exception:
                pass
            raise SystemExit(125)
        prior=read(STATUS,{})
        prior.update(state='wall_budget_exhausted' if isinstance(error,TimeoutError) else 'coordinator_error',
                     error_type=type(error).__name__,updated_utc=now(),
                     attention_reason='Inspect durable stage logs and receipt; restarting never repeats an upload intent.')
        save(STATUS,prior)
        raise SystemExit(2)
