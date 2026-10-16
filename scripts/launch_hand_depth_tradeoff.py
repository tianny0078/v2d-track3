"""Launch one bounded three-setting stereo/hand trade-off sweep in server tmux."""
import json,shlex,subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
root=Path(__file__).resolve().parents[1];base=root/'outputs/track3/episode_000011';out=base/'hand_depth_tradeoff'
if (out/'report.json').exists() and json.loads((out/'report.json').read_text()).get('state')=='completed':
    print('Completed trade-off sweep exists; no duplicate started');raise SystemExit(0)
for tag in ['hand_surface_depth_aligned','hand_surface_depth_sequence']:
    if json.loads((base/tag/'report.json').read_text()).get('state')!='completed':raise SystemExit('A prerequisite audit has not completed')
sessions=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],text=True,capture_output=True).stdout.splitlines()
if any(s.startswith('v2d-t3-hand-depth-tradeoff-') for s in sessions):
    print('Existing trade-off task active');raise SystemExit(0)
stamp=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d-%H%M%S')
r={'session':'v2d-t3-hand-depth-tradeoff-'+stamp,'log':str(root/'logs/track3-hand-depth-tradeoff.log'),
   'exit_file':str(root/'logs/track3-hand-depth-tradeoff.exit'),'training':False,
   'resume':f'cd {root} && python3 scripts/launch_hand_depth_tradeoff.py','partial_resume':'Completed alpha/frame results are reused after verifying source hashes.'}
r['attach']='tmux attach -t '+r['session']
shell='OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python3 -u scripts/run_hand_depth_tradeoff.py >> '+shlex.quote(r['log'])+' 2>&1; task_rc=$?; printf "%s\\n" "$task_rc" > '+shlex.quote(r['exit_file'])+'; exit "$task_rc"'
r['shell_command']=shell
(root/'notes/track3-hand-depth-tradeoff-job.json').write_text(json.dumps(r,indent=2)+'\n')
subprocess.run(['tmux','new-session','-d','-s',r['session'],'-c',str(root),shell],check=True);print('Started',r['session'])
