"""Create the cached scene figures in a persistent offline CPU container."""
import argparse,json,os,shlex,subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--refresh-figures',action='store_true');args=p.parse_args()
root=Path(__file__).resolve().parents[1];base=root/'outputs/track3/episode_000011';out=base/'stereo_scene_quality'
if not args.refresh_figures and (out/'report.json').exists() and json.loads((out/'report.json').read_text()).get('state')=='completed':
    print('Completed figures already exist');raise SystemExit(0)
if json.loads((base/'foundation_stereo_sequence/report.json').read_text()).get('state')!='completed':
    raise SystemExit('Full learned stereo still running; summary not started')
name='v2d-t3-scene-summary'
if subprocess.check_output(['docker','ps','--filter','name=^/'+name+'$','--format','{{.Names}}'],text=True).strip():
    print('Existing figure job active');raise SystemExit(0)
out.mkdir(exist_ok=True);image='sha256:367f7ab879e29469b7d222635f5797f962906ef0afd01e2586556850cc9aec32'
cmd=['docker','run','--rm','--pull=never','--name',name,'--network=none','--read-only','--tmpfs','/tmp:rw,size=1g',
     '--user',f'{os.getuid()}:{os.getgid()}','-e','MPLCONFIGDIR=/tmp/matplotlib',
     '--mount',f'type=bind,src={root}/scripts,dst=/work/scripts,readonly',
     '--mount',f'type=bind,src={base},dst=/work/outputs/track3/episode_000011,readonly',
     '--mount',f'type=bind,src={out},dst=/work/outputs/track3/episode_000011/stereo_scene_quality',
     '--entrypoint','python',image,'-u','/work/scripts/summarize_stereo_scene.py']
stamp=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d-%H%M%S')
r={'session':'v2d-t3-scene-summary-'+stamp,'argv':cmd,'image_id':image,'log':str(root/'logs/track3-scene-summary.log'),
   'exit_file':str(root/'logs/track3-scene-summary.exit'),'resume':f'cd {root} && python3 scripts/launch_stereo_scene_summary.py','training':False}
r['attach']='tmux attach -t '+r['session']
shell=shlex.join(cmd)+' >> '+shlex.quote(r['log'])+' 2>&1; task_rc=$?; printf "%s\\n" "$task_rc" > '+shlex.quote(r['exit_file'])+'; exit "$task_rc"'
(root/'notes/track3-scene-summary-job.json').write_text(json.dumps(r,indent=2)+'\n')
subprocess.run(['tmux','new-session','-d','-s',r['session'],'-c',str(root),shell],check=True);print('Started',r['session'])
