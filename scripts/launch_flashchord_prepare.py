"""Idempotent tmux launcher for the official, isolated FlashCHORD environment."""
import json
import shlex
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'v2d-t3-flashchord-prepare'
jobfile = ROOT/'notes/track3-flashchord-prepare-job.json'
reportfile = ROOT/'notes/track3-flashchord-environment.json'
sessions = subprocess.run(['tmux', 'list-sessions', '-F', '#{session_name}'], capture_output=True, text=True)
active = [s for s in sessions.stdout.splitlines() if s.startswith(PREFIX)]
if active:
    print('Already running:', ', '.join(active)); raise SystemExit(0)
if reportfile.exists() and json.loads(reportfile.read_text()).get('state') == 'completed':
    print('Environment and GPU smoke already completed; reuse existing environment'); raise SystemExit(0)
stamp = datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d-%H%M%S')
log = ROOT/f'logs/track3-flashchord-prepare-{stamp}.log'
exitfile = ROOT/f'logs/track3-flashchord-prepare-{stamp}.exit'
session = PREFIX+'-'+stamp
argv = [sys.executable, '-u', str(ROOT/'scripts/prepare_flashchord.py')]
shell = shlex.join(argv)+' >> '+shlex.quote(str(log))+' 2>&1; task_rc=$?; printf "%s\\n" "$task_rc" > '+shlex.quote(str(exitfile))+'; exit "$task_rc"'
job = {'session': session, 'hostname': subprocess.check_output(['hostname'], text=True).strip(),
       'argv': argv, 'log': str(log), 'exit_file': str(exitfile), 'training': False,
       'attach': 'tmux attach -t '+session, 'tail': 'tail -n 50 '+str(log),
       'resume': 'cd '+str(ROOT)+' && python3 scripts/launch_flashchord_prepare.py',
       'checkpoint': 'Package cache, project-local environment, exact uv.lock, and GPU smoke report; no policy checkpoint yet.'}
jobfile.write_text(json.dumps(job, indent=2)+'\n')
subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', str(ROOT), shell], check=True)
print(json.dumps(job, indent=2))
