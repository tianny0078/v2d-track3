"""Start one resumable model/dependency download in server-side tmux."""
import json,shlex,subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
root=Path(__file__).resolve().parents[1];status=root/'notes/track3-foundation-stereo-prepare-status.json'
if status.exists() and json.loads(status.read_text()).get('state')=='completed':
    print('Model and isolated dependencies ready; no duplicate download');raise SystemExit(0)
sessions=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],text=True,capture_output=True).stdout.splitlines()
if any(s.startswith('v2d-t3-foundation-prepare-') for s in sessions):
    print('Matching prepare task active');raise SystemExit(0)
stamp=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d-%H%M%S')
r={'session':'v2d-t3-foundation-prepare-'+stamp,'log':str(root/'logs/track3-foundation-stereo-prepare.log'),
   'exit_file':str(root/'logs/track3-foundation-stereo-prepare.exit'),
   'resume':f'cd {root} && python3 scripts/launch_foundation_stereo_prepare.py','training':False}
r['attach']='tmux attach -t '+r['session']
shell='python3 -u scripts/prepare_foundation_stereo.py >> '+shlex.quote(r['log'])+' 2>&1; task_rc=$?; printf "%s\\n" "$task_rc" > '+shlex.quote(r['exit_file'])+'; exit "$task_rc"'
r['shell_command']=shell
(root/'notes/track3-foundation-stereo-prepare-job.json').write_text(json.dumps(r,indent=2)+'\n')
subprocess.run(['tmux','new-session','-d','-s',r['session'],'-c',str(root),shell],check=True);print('Started',r['session'])
