"""Run cache-based metric diagnostics once in tmux, with persistent logs."""
import json,shlex,subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

root=Path(__file__).resolve().parents[1];report=root/'outputs/track3/episode_000011/metric_scene_diagnostic/report.json'
if report.exists() and json.loads(report.read_text()).get('state')=='completed':
    print('Completed metric scene diagnostic exists; no duplicate started');raise SystemExit(0)
sessions=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],text=True,capture_output=True).stdout.splitlines()
if any(s.startswith('v2d-t3-metric-scene-') for s in sessions):
    print('Existing metric scene diagnostic active');raise SystemExit(0)
stamp=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d-%H%M%S')
r={'session':'v2d-t3-metric-scene-'+stamp,'log':str(root/'logs/track3-metric-scene.log'),
   'exit_file':str(root/'logs/track3-metric-scene.exit'),'resume':f'cd {root} && python3 scripts/launch_metric_scene.py','training':False}
r['attach']='tmux attach -t '+r['session']
command='python3 -u scripts/check_metric_scene.py >> '+shlex.quote(r['log'])+' 2>&1; task_rc=$?; printf "%s\\n" "$task_rc" > '+shlex.quote(r['exit_file'])+'; exit "$task_rc"'
r['shell_command']=command
(root/'notes/track3-metric-scene-job.json').write_text(json.dumps(r,indent=2)+'\n')
subprocess.run(['tmux','new-session','-d','-s',r['session'],'-c',str(root),command],check=True);print('Started',r['session'])
