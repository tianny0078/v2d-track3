"""Run bundled MediaPipe hand tracking on the single public episode in tmux."""
import argparse,json,os,shlex,subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];base=a.output_root.resolve();out=base/'hand_detector_mediapipe_sequence'
manifest=json.loads((base/'calibrated_sequence/manifest.json').read_text());assert manifest['state']=='completed'
if (out/'report.json').exists() and json.loads((out/'report.json').read_text()).get('state')=='completed':
    print('Completed sequence detector exists; no duplicate started');raise SystemExit(0)
name='v2d-t3-hands-'+base.name
if subprocess.check_output(['docker','ps','--filter','name=^/'+name+'$','--format','{{.Names}}'],text=True).strip():
    print('Sequence detector already active');raise SystemExit(0)
out.mkdir(parents=True,exist_ok=True)
image=subprocess.check_output(['docker','image','inspect','v2d_mediapipe:latest','--format','{{.Id}}'],text=True).strip()
cmd=['docker','run','--rm','--pull=never','--name',name,'--network=none','--read-only','--tmpfs','/tmp:rw,exec,size=2g',
     '--user',f'{os.getuid()}:{os.getgid()}','-e','TF_NUM_INTRAOP_THREADS=4','-e','TF_NUM_INTEROP_THREADS=2']
for src,dst,ro in [(root/'scripts','/work/scripts',True),(base/'calibrated_sequence','/episode/calibrated_sequence',True),
                   (out,'/episode/hand_detector_mediapipe_sequence',False)]:
    cmd+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
cmd+=['--entrypoint','python',image,'-u','/work/scripts/run_mediapipe_hand_check.py','--sequence','--base','/episode']
cmd=['timeout','300']+cmd
stamp=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d-%H%M%S')
r={'session':name,'argv':cmd,'image_id':image,
   'log':str(base/'hands-job.log'),'exit_file':str(base/'hands-job.exit'),
   'resume':shlex.join(['python3',str(Path(__file__).resolve()),'--output-root',str(base)]),'training':False,'timeout_seconds':300,'checkpoint':str(out/'report.json'),'checkpoint_semantics':'Completed report protected; interrupted detection restarts sequence','source_frame_count':len(manifest['frames'])}
r['attach']='tmux attach -t '+r['session']
shell=shlex.join(cmd)+' >> '+shlex.quote(r['log'])+' 2>&1; task_rc=$?; printf "%s\\n" "$task_rc" > '+shlex.quote(r['exit_file'])+'; exit "$task_rc"'
(base/'hands-job.json').write_text(json.dumps(r,indent=2)+'\n')
subprocess.run(['tmux','new-session','-d','-s',r['session'],'-c',str(root),shell],check=True);print('Started',r['session'])
