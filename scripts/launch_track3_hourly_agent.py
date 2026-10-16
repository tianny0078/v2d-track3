"""Start the user-requested hourly reporting agent in a persistent server tmux."""
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1];prefix='v2d-t3-hourly-progress-'
sessions=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True)
active=[s for s in sessions.stdout.splitlines() if s.startswith(prefix)]
if active:print('Already running:',','.join(active));raise SystemExit(0)
cap=json.loads((ROOT/'notes/track3-hourly-capability.json').read_text())
assert cap.get('local_notes_readable') and cap.get('site_verified'),cap
if (ROOT/'notes/track3-hourly-agent.stop').exists():raise SystemExit('Stop marker exists; explicit resume needed.')
stamp=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d-%H%M%S');session=prefix+stamp
log=ROOT/f'logs/track3-hourly-progress-{stamp}.log';exitfile=log.with_suffix('.exit')
argv=[sys.executable,'-u',str(ROOT/'scripts/run_track3_hourly_agent.py')]
job={'session':session,'argv':argv,'log':str(log),'exit_file':str(exitfile),'interval_seconds':3600,
     'attach':'tmux attach -t '+session,'tail':'tail -n 30 '+str(log),
     'resume':'cd '+str(ROOT)+' && python3 scripts/launch_track3_hourly_agent.py',
     'stop_after_current_run':'touch '+str(ROOT/'notes/track3-hourly-agent.stop'),
     'site':'https://v2d-track3-results-20261015.sharp-spice-1328.chatgpt.site/',
     'note':'Immediate first verification run, then every hour; server must remain powered on. No cloud-only task is used for local NAS access.'}
(ROOT/'notes/track3-hourly-agent-job.json').write_text(json.dumps(job,indent=2)+'\n')
shell=shlex.join(argv)+' >> '+shlex.quote(str(log))+' 2>&1; task_rc=$?; printf "%s\\n" "$task_rc" > '+shlex.quote(str(exitfile))+'; exit "$task_rc"'
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),shell],check=True)
print(json.dumps(job,indent=2))
