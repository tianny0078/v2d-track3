"""Launch one durable continuation supervisor, without duplicating experiments."""
from datetime import datetime
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
lock = (ROOT / 'notes/.track3-continuation-launch.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX)
prefix = 'v2d-t3-continuation-'
sessions = subprocess.run(['tmux', 'list-sessions', '-F', '#{session_name}'],
                          capture_output=True, text=True)
active = [s for s in sessions.stdout.splitlines() if s.startswith(prefix)]
if active:
    print('Already running: ' + ', '.join(active))
    raise SystemExit(0)
if (ROOT / 'notes/track3-continuation.stop').exists():
    raise SystemExit('Stop marker exists; preserve it until the user resumes.')
stamp = datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d-%H%M%S')
session = prefix + stamp
log = ROOT / f'logs/track3-continuation-{stamp}.log'
exitfile = log.with_suffix('.exit')
argv = [sys.executable, '-u', str(ROOT / 'scripts/run_track3_continuation.py')]
job = {'session': session, 'argv': argv, 'log': str(log), 'exit_file': str(exitfile),
       'attach': 'tmux attach -t ' + session,
       'resume': 'cd ' + str(ROOT) + ' && python3 scripts/launch_track3_continuation.py',
       'stop_after_current_cycle': 'touch ' + str(ROOT / 'notes/track3-continuation.stop'),
       'scope': 'Sequential Track3 research, single episode before scored cohort; no Kaggle uploads',
       'policy': 'workspace-write and automatic approval review; existing configured model; no sandbox bypass',
       'heartbeat': str(ROOT / 'notes/track3-continuation-state.json'),
       'instructions': str(ROOT / 'notes/track3-continuation-instructions.md')}
shell = (shlex.join(argv) + ' >> ' + shlex.quote(str(log)) + ' 2>&1; '
         'task_rc=$?; printf "%s\\n" "$task_rc" > ' + shlex.quote(str(exitfile)) + '; exit "$task_rc"')
subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', str(ROOT), shell], check=True)
(ROOT / 'notes/track3-continuation-job.json').write_text(json.dumps(job, indent=2) + '\n')
print(json.dumps(job, indent=2))
