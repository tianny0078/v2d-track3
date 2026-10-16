"""Run temporal hand correction diagnostics in tmux with duplicate protection."""
import json,shlex,subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
root=Path(__file__).resolve().parents[1];out=root/'outputs/track3/episode_000011/hand_registration_temporal'
if (out/'report.json').exists() and json.loads((out/'report.json').read_text()).get('state')=='completed':
    print('Completed temporal correction diagnostic exists');raise SystemExit(0)
sessions=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True).stdout.splitlines()
if any(s.startswith('v2d-t3-hand-temporal-') for s in sessions):
    print('Existing temporal correction active');raise SystemExit(0)
stamp=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d-%H%M%S')
r={'session':'v2d-t3-hand-temporal-'+stamp,'log':str(root/'logs/track3-hand-temporal.log'),
   'exit_file':str(root/'logs/track3-hand-temporal.exit'),'training':False,
   'resume':f'cd {root} && python3 scripts/launch_hand_regularization.py'}
r['attach']='tmux attach -t '+r['session']
shell='OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python3 -u scripts/regularize_hand_registration.py >> '+shlex.quote(r['log'])+' 2>&1; task_rc=$?; printf "%s\\n" "$task_rc" > '+shlex.quote(r['exit_file'])+'; exit "$task_rc"'
r['shell_command']=shell
(root/'notes/track3-hand-temporal-job.json').write_text(json.dumps(r,indent=2)+'\n')
subprocess.run(['tmux','new-session','-d','-s',r['session'],'-c',str(root),shell],check=True);print('Started',r['session'])
