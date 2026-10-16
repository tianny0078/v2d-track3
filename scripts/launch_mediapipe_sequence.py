"""Run bundled MediaPipe hand tracking on the single public episode in tmux."""
import json,os,shlex,subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

root=Path(__file__).resolve().parents[1];out=root/'outputs/track3/episode_000011/hand_detector_mediapipe_sequence'
if (out/'report.json').exists() and json.loads((out/'report.json').read_text()).get('state')=='completed':
    print('Completed sequence detector exists; no duplicate started');raise SystemExit(0)
name='v2d-t3-hand-detector-sequence'
if subprocess.check_output(['docker','ps','--filter','name=^/'+name+'$','--format','{{.Names}}'],text=True).strip():
    print('Sequence detector already active');raise SystemExit(0)
out.mkdir(parents=True,exist_ok=True)
image=subprocess.check_output(['docker','image','inspect','v2d_mediapipe:latest','--format','{{.Id}}'],text=True).strip()
cmd=['docker','run','--rm','--pull=never','--name',name,'--network=none','--read-only','--tmpfs','/tmp:rw,exec,size=2g',
     '--user',f'{os.getuid()}:{os.getgid()}','-e','TF_NUM_INTRAOP_THREADS=4','-e','TF_NUM_INTEROP_THREADS=2']
for src,dst,ro in [(root/'scripts','/work/scripts',True),(root/'outputs/track3/episode_000011/calibrated_sequence','/work/outputs/track3/episode_000011/calibrated_sequence',True),
                   (out,'/work/outputs/track3/episode_000011/hand_detector_mediapipe_sequence',False)]:
    cmd+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
cmd+=['--entrypoint','python',image,'-u','/work/scripts/run_mediapipe_hand_check.py','--sequence']
stamp=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d-%H%M%S')
r={'session':'v2d-t3-hand-detector-sequence-'+stamp,'argv':cmd,'image_id':image,
   'log':str(root/'logs/track3-hand-detector-sequence.log'),'exit_file':str(root/'logs/track3-hand-detector-sequence.exit'),
   'resume':f'cd {root} && python3 scripts/launch_mediapipe_sequence.py','training':False}
r['attach']='tmux attach -t '+r['session']
shell=shlex.join(cmd)+' >> '+shlex.quote(r['log'])+' 2>&1; task_rc=$?; printf "%s\\n" "$task_rc" > '+shlex.quote(r['exit_file'])+'; exit "$task_rc"'
(root/'notes/track3-hand-detector-sequence-job.json').write_text(json.dumps(r,indent=2)+'\n')
subprocess.run(['tmux','new-session','-d','-s',r['session'],'-c',str(root),shell],check=True);print('Started',r['session'])
