"""Sequential server research worker; experiments remain in logged tmux jobs.

Uses the existing Codex login and automatic approval review. No raw model/tool
transcripts are saved: only the structured handoff and supervisor heartbeat.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / 'notes'
STATE = NOTES / 'track3-continuation-state.json'
STOP = NOTES / 'track3-continuation.stop'
REQUEST = NOTES / 'track3-publication-request.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-cycles', type=int, default=0)
    args = parser.parse_args()
    lock = (NOTES / '.track3-continuation.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Continuation worker already active; no duplicate.')
    schema = {
        'type': 'object',
        'properties': {
            'step_title': {'type': 'string'},
            'status': {'type': 'string', 'enum': [
                'completed_step', 'running_experiment', 'needs_external_input',
                'ready_for_submission', 'failed_step']},
            'progress_made': {'type': 'boolean'},
            'summary': {'type': 'string'},
            'artifact_paths': {'type': 'array', 'items': {'type': 'string'}},
            'next_step': {'type': 'string'},
            'needs_attention': {'type': 'boolean'},
        },
        'required': ['step_title', 'status', 'progress_made', 'summary',
                     'artifact_paths', 'next_step', 'needs_attention'],
        'additionalProperties': False,
    }
    schemafile = NOTES / 'track3-continuation-result.schema.json'
    save(schemafile, schema)
    previous = None
    if STATE.exists():
        previous = json.loads(STATE.read_text()).get('last_result')
    cycles = 0
    no_progress = 0
    state = {'state': 'starting', 'heartbeat_utc': now()}
    while not STOP.exists():
        cycles += 1
        started = now()
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        state = {'state': 'running', 'cycle': cycles, 'started_utc': started,
                 'heartbeat_utc': started, 'supervisor_pid': os.getpid(),
                 'worker_pid': None, 'last_result': previous,
                 'policy': 'sequential Track3 only; workspace-write + automatic approval review; no raw transcripts'}
        save(STATE, state)
        print(f'Continuation cycle {cycles} started {stamp}', flush=True)
        prompt = (
            '继续用户已授权的Track3研究，不等待用户再次发送继续。先完整读取 '
            '/home/tianyuan/projects/v2d/notes/track3-continuation-instructions.md，'
            '核查真实任务和产物，然后完成一个可验证的下一步。质量失败要修具体原因，'
            '不能仅更新状态就结束。保持日志/checkpoint/恢复命令。'
        )
        if previous:
            prompt += '\n上一轮结构化交接：\n' + json.dumps(previous, ensure_ascii=False)
        with tempfile.TemporaryDirectory(prefix='v2d-continuation-') as temporary:
            final = Path(temporary) / 'result.json'
            argv = ['codex', 'exec', '--approve-for-me', '--skip-git-repo-check',
                    '--ephemeral', '--cd', str(ROOT), '--output-schema', str(schemafile),
                    '--output-last-message', str(final), '-']
            try:
                process = subprocess.Popen(argv, stdin=subprocess.PIPE,
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                           text=True, start_new_session=True)
                state['worker_pid'] = process.pid
                save(STATE, state)
                process.stdin.write(prompt)
                process.stdin.close()
                deadline = time.monotonic() + 2700
                while process.poll() is None and time.monotonic() < deadline:
                    state['heartbeat_utc'] = now()
                    state['stop_after_current_cycle_requested'] = STOP.exists()
                    save(STATE, state)
                    time.sleep(10)
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise TimeoutError('Worker exceeded 45 minutes; detached experiment jobs are retained.')
                result = json.loads(final.read_text()) if final.exists() else {
                    'step_title': 'Worker handoff', 'status': 'failed_step',
                    'progress_made': False, 'summary': 'No structured handoff was returned.',
                    'artifact_paths': [], 'next_step': 'Inspect actual job/artifact state and continue safely.',
                    'needs_attention': True,
                }
                result['exit_code'] = process.returncode
            except Exception as exc:
                result = {'step_title': 'Worker execution', 'status': 'failed_step',
                          'progress_made': False, 'summary': type(exc).__name__,
                          'artifact_paths': [], 'next_step': 'Audit running jobs and resume the next useful step.',
                          'needs_attention': True, 'exit_code': 124 if isinstance(exc, TimeoutError) else 1}
        result.update(started_utc=started, completed_utc=now(), cycle=cycles)
        resultfile = NOTES / 'continuation' / f'result-{stamp}.json'
        resultfile.parent.mkdir(exist_ok=True)
        save(resultfile, result)
        previous = result
        no_progress = 0 if result.get('progress_made') else no_progress + 1
        state.update(state='continuing', worker_pid=None, heartbeat_utc=now(),
                     last_result=result, last_result_file=str(resultfile),
                     consecutive_no_progress=no_progress)
        save(STATE, state)
        save(REQUEST, {'requested_utc': now(), 'reason': 'continuation step handoff',
                       'result_file': str(resultfile), 'cycle': cycles})
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if args.max_cycles and cycles >= args.max_cycles:
            break
        # Back off on repeated tool/auth/no-progress failures without inventing success.
        delay = 900 if no_progress >= 6 else 300 if no_progress >= 3 else 15
        if result.get('status') == 'running_experiment':
            delay = max(delay, 60)
        state.update(state='backoff' if delay >= 300 else 'continuing', next_cycle_delay_seconds=delay)
        save(STATE, state)
        deadline = time.monotonic() + delay
        while not STOP.exists() and time.monotonic() < deadline:
            time.sleep(min(10, max(.1, deadline - time.monotonic())))
    state.update(state='stopped', heartbeat_utc=now(), stop_marker_present=STOP.exists())
    save(STATE, state)


if __name__ == '__main__':
    main()
