"""Start one public-resource download in tmux, with completion and duplicate guards."""
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

root=Path(__file__).resolve().parents[1]
status=root/'notes/track3-reconstruction-download-status.json'
if status.exists() and json.loads(status.read_text()).get('state')=='completed':
    print('Resources already complete; reusing existing files')
    raise SystemExit(0)
sessions=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],text=True,capture_output=True).stdout.splitlines()
if any(s.startswith('v2d-t3-recon-download-') for s in sessions):
    print('Public resource download already active; no duplicate started')
    raise SystemExit(0)
stamp=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d-%H%M%S')
r={'session':'v2d-t3-recon-download-'+stamp,'log':str(root/'logs/track3-reconstruction-download.log'),
   'exit_file':str(root/'logs/track3-reconstruction-download.exit'),'resume':f'cd {root} && python3 scripts/launch_reconstruction_download.py'}
r['attach']='tmux attach -t '+r['session']
cmd='python3 -u scripts/download_reconstruction_resources.py >> '+shlex.quote(r['log'])+' 2>&1; task_rc=$?; printf "%s\\n" "$task_rc" > '+shlex.quote(r['exit_file'])+'; exit "$task_rc"'
r['shell_command']=cmd
(root/'notes/track3-reconstruction-download-job.json').write_text(json.dumps(r,indent=2)+'\n')
subprocess.run(['tmux','new-session','-d','-s',r['session'],'-c',str(root),cmd],check=True)
print('Started',r['session'])
