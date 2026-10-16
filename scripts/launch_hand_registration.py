"""Launch one bounded hand-registration diagnostic in persistent tmux."""
import argparse,json,shlex,subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sequence',action='store_true');args=p.parse_args()
root=Path(__file__).resolve().parents[1];tag='hand-registration-'+('sequence' if args.sequence else 'sample')
out=root/'outputs/track3/episode_000011'/tag.replace('-','_');path=out/'report.json'
if path.exists() and json.loads(path.read_text()).get('state')=='completed':
    print('Completed diagnostic exists; no duplicate started');raise SystemExit(0)
sessions=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True).stdout.splitlines()
if any(s.startswith('v2d-t3-'+tag+'-') for s in sessions):
    print('Matching diagnostic active');raise SystemExit(0)
stamp=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d-%H%M%S')
r={'session':'v2d-t3-'+tag+'-'+stamp,'log':str(root/f'logs/track3-{tag}.log'),
   'exit_file':str(root/f'logs/track3-{tag}.exit'),'training':False,
   'resume':f'cd {root} && python3 scripts/launch_hand_registration.py'+(' --sequence' if args.sequence else '')}
r['attach']='tmux attach -t '+r['session']
command='python3 -u scripts/register_hand_landmarks.py'+(' --sequence' if args.sequence else '')
shell=command+' >> '+shlex.quote(r['log'])+' 2>&1; task_rc=$?; printf "%s\\n" "$task_rc" > '+shlex.quote(r['exit_file'])+'; exit "$task_rc"'
r['shell_command']=shell
(root/f'notes/track3-{tag}-job.json').write_text(json.dumps(r,indent=2)+'\n')
subprocess.run(['tmux','new-session','-d','-s',r['session'],'-c',str(root),shell],check=True);print('Started',r['session'])
