"""Run the full hand-surface consistency audit in persistent CPU tmux."""
import argparse,json,shlex,subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--depth-candidate',action='store_true');args=p.parse_args()
tag='hand-surface-depth-'+('aligned' if args.depth_candidate else 'sequence')
root=Path(__file__).resolve().parents[1];base=root/'outputs/track3/episode_000011';out=base/tag.replace('-','_')
if (out/'report.json').exists() and json.loads((out/'report.json').read_text()).get('state')=='completed':
    print('Full hand-surface audit already completed');raise SystemExit(0)
source=base/'hand_masks_sequence/report.json'
if not source.exists() or json.loads(source.read_text()).get('state')!='completed':raise SystemExit('Full hand masks have not completed')
sessions=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],text=True,capture_output=True).stdout.splitlines()
if any(s.startswith('v2d-t3-hand-surface-depth-') for s in sessions):
    print('Hand-surface audit already active');raise SystemExit(0)
stamp=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d-%H%M%S')
r={'session':'v2d-t3-'+tag+'-'+stamp,'log':str(root/f'logs/track3-{tag}.log'),
   'exit_file':str(root/f'logs/track3-{tag}.exit'),'training':False,
   'resume':f'cd {root} && python3 scripts/launch_hand_surface_depth.py'+(' --depth-candidate' if args.depth_candidate else '')}
r['attach']='tmux attach -t '+r['session']
shell='python3 -u scripts/check_hand_surface_depth.py --sequence'+(' --depth-candidate' if args.depth_candidate else '')+' >> '+shlex.quote(r['log'])+' 2>&1; task_rc=$?; printf "%s\\n" "$task_rc" > '+shlex.quote(r['exit_file'])+'; exit "$task_rc"'
r['shell_command']=shell
(root/f'notes/track3-{tag}-job.json').write_text(json.dumps(r,indent=2)+'\n')
subprocess.run(['tmux','new-session','-d','-s',r['session'],'-c',str(root),shell],check=True);print('Started',r['session'])
