"""Launch a single CPU stereo/odometry episode in a persistent tmux session."""
import json,shlex,subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

root=Path(__file__).resolve().parents[1];report=root/'outputs/track3/episode_000011/stereo_sequence/report.json'
if report.exists() and json.loads(report.read_text()).get('state')=='completed':
    print('Completed stereo/odometry episode exists; no duplicate started');raise SystemExit(0)
sessions=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],text=True,capture_output=True).stdout.splitlines()
if any(s.startswith('v2d-t3-stereo-sequence-') for s in sessions):
    print('Stereo/odometry episode is already active');raise SystemExit(0)
stamp=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d-%H%M%S')
r={'session':'v2d-t3-stereo-sequence-'+stamp,'log':str(root/'logs/track3-stereo-sequence.log'),
   'exit_file':str(root/'logs/track3-stereo-sequence.exit'),'resume':f'cd {root} && python3 scripts/launch_stereo_sequence.py',
   'training':False,'partial_resume':'Partial depth/pose outputs persist; an interrupted VO chain restarts from frame 0.'}
r['attach']='tmux attach -t '+r['session']
cmd='python3 -u scripts/run_stereo_sequence.py >> '+shlex.quote(r['log'])+' 2>&1; task_rc=$?; printf "%s\\n" "$task_rc" > '+shlex.quote(r['exit_file'])+'; exit "$task_rc"'
r['shell_command']=cmd
(root/'notes/track3-stereo-sequence-job.json').write_text(json.dumps(r,indent=2)+'\n')
subprocess.run(['tmux','new-session','-d','-s',r['session'],'-c',str(root),cmd],check=True);print('Started',r['session'])
