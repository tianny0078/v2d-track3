"""Launch the bounded diagnostic baseline in a dedicated server tmux session."""
from pathlib import Path
import datetime as dt
import json
import shlex
import socket
import subprocess

ROOT = Path(__file__).resolve().parents[1]
prefix = 'v2d-t3-baseline-probe-'
check = subprocess.run(['tmux', 'list-sessions', '-F', '#{session_name}'], capture_output=True, text=True)
if any(line.startswith(prefix) for line in check.stdout.splitlines()):
    raise SystemExit('A matching baseline probe is already running')
stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
session = prefix + stamp
log = ROOT / 'logs' / f'track3-baseline-probe-{stamp}.log'
exitfile = log.with_suffix('.exit')
argv = ['timeout', '--signal=TERM', '--kill-after=10', '1180', str(ROOT / '.deps/flash-chord-venv/bin/python'), '-u', str(ROOT / 'scripts/agent_baseline_probe.py')]
shell = shlex.join(argv) + ' > ' + shlex.quote(str(log)) + ' 2>&1; code=$?; printf "%s\\n" "$code" > ' + shlex.quote(str(exitfile))
job = {'state': 'starting', 'session': session, 'hostname': socket.gethostname(), 'log': str(log), 'exit_file': str(exitfile),
       'attach': f'tmux attach -t {session}', 'resume': 'python3 scripts/agent_baseline_probe_launch.py',
       'argv': argv, 'hard_timeout_seconds': 1180, 'physical_gpu': 3, 'scope': 'diagnostic baseline only',
       'training_ready': False, 'task_success_claimed': False, 'scored_episode': False}
(ROOT / 'notes/track3-baseline-probe-job.json').write_text(json.dumps(job, indent=2) + '\n')
subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', str(ROOT), 'bash', '-c', shell], check=True)
print(json.dumps(job, indent=2))
