"""Launch a single offline SAM2 hand mask diagnostic in persistent tmux."""
import argparse,json,os,shlex,subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sequence',action='store_true');args=p.parse_args()
root=Path(__file__).resolve().parents[1];base=root/'outputs/track3/episode_000011';tag='hand-masks-'+('sequence' if args.sequence else 'sample');out=base/tag.replace('-','_')
if (out/'report.json').exists() and json.loads((out/'report.json').read_text()).get('state')=='completed':
    print('Completed hand masks exist; no duplicate');raise SystemExit(0)
# Keep experimental GPU jobs sequential, including across different module images.
active=subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()
if any(n.startswith('v2d-t3-') for n in active):raise SystemExit('An existing Track3 GPU task is active; hand mask job not started')
out.mkdir(exist_ok=True);image='sha256:bc01a90c3cf71e0336e2baa312dda34f983778ba12472902ce6b4eb5ef1b4d52';name='v2d-t3-'+tag
cmd=['docker','run','--rm','--pull=never','--name',name,'--network=none','--read-only','--tmpfs','/tmp:rw,exec,size=4g','--gpus','device=0',
     '--user',f'{os.getuid()}:{os.getgid()}','-e','TORCH_HOME=/tmp/torch','-e','CUDA_CACHE_PATH=/tmp/cuda']
for src,dst,ro in [(root/'scripts','/work/scripts',True),(root/'weights/sam2','/work/weights/sam2',True),
                   (base,'/work/outputs/track3/episode_000011',True),(out,'/work/outputs/track3/episode_000011/'+tag.replace('-','_'),False)]:
    cmd+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
cmd+=['--entrypoint','python',image,'-u','/work/scripts/run_hand_masks.py']
if args.sequence:cmd+=['--sequence']
stamp=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d-%H%M%S')
r={'session':'v2d-t3-'+tag+'-'+stamp,'argv':cmd,'image_id':image,'log':str(root/f'logs/track3-{tag}.log'),
   'exit_file':str(root/f'logs/track3-{tag}.exit'),'training':False,'resume':f'cd {root} && python3 scripts/launch_hand_masks.py'+(' --sequence' if args.sequence else '')}
r['attach']='tmux attach -t '+r['session']
shell=shlex.join(cmd)+' >> '+shlex.quote(r['log'])+' 2>&1; task_rc=$?; printf "%s\\n" "$task_rc" > '+shlex.quote(r['exit_file'])+'; exit "$task_rc"'
(root/f'notes/track3-{tag}-job.json').write_text(json.dumps(r,indent=2)+'\n')
subprocess.run(['tmux','new-session','-d','-s',r['session'],'-c',str(root),shell],check=True);print('Started',r['session'])
