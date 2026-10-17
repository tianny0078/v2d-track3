"""Launch bounded, offline detection and full temporal masks for a scored episode."""
import argparse,hashlib,json,os,shlex,subprocess
from datetime import datetime,timezone
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode',type=int,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--gpu',type=int,default=2);a=p.parse_args()
root=Path(__file__).resolve().parents[1];base=a.output_root.resolve();session=f'v2d-t3-masks-ep{a.episode:06d}'
if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:print('Already running',session);raise SystemExit(0)
rp=base/'object_masks_temporal/report.json'
if rp.exists() and json.loads(rp.read_text()).get('state')=='completed':print('Completed output protected');raise SystemExit(0)
manifest=json.loads((base/'calibrated_sequence/manifest.json').read_text());assert manifest['state']=='completed' and manifest['episode']==a.episode
names=json.loads((root/'toolkit/8d2bce64/v2d_submission_kit/data/track_3_evaluation_objects.json').read_text())['episodes'][str(a.episode)]
(base/'object_detection').mkdir(exist_ok=True);(base/'object_masks_temporal').mkdir(exist_ok=True)
usage=subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip();assert int(usage)<1000,'Selected GPU busy'
commands=[]
for stage,image,script,extra in [('detect','sha256:6b1b94dd1fa0418b1aad2b3e164b35745624e9d9d42c6404dd5754c990d5e05c','run_track3_episode_detection.py',['--objects',*names]),('masks','sha256:bc01a90c3cf71e0336e2baa312dda34f983778ba12472902ce6b4eb5ef1b4d52','run_track3_episode_masks.py',[])]:
    cmd=['timeout','600','docker','run','--rm','--pull=never','--name',session+'-'+stage,'--network=none','--read-only','--tmpfs','/tmp:rw,exec,size=4g','--gpus',f'device={a.gpu}','--user',f'{os.getuid()}:{os.getgid()}']
    for env in ['HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','HF_HOME=/tmp/hf','TORCH_HOME=/tmp/torch','CUDA_CACHE_PATH=/tmp/cuda-cache']:cmd+=['-e',env]
    for src,dst,ro in [(root/'scripts','/work/scripts',True),(root/'weights','/work/weights',True),(base,'/episode',False)]:cmd+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
    cmd+=['--entrypoint','python',image,'-u','/work/scripts/'+script,'--base','/episode',*extra];commands.append(cmd)
log=base/'masks-job.log';exitfile=base/'masks-job.exit'
shell='( '+ ' && '.join(shlex.join(c) for c in commands)+' ) >> '+shlex.quote(str(log))+' 2>&1; task_rc=$?; printf "%s\\n" "$task_rc" > '+shlex.quote(str(exitfile))+'; exit "$task_rc"'
record=dict(state='launched',episode=a.episode,object_order=names,session=session,argv=commands,log=str(log),exit_file=str(exitfile),checkpoint=str(base/'object_masks_temporal/report.json'),checkpoint_semantics='Masks saved per frame; temporal memory restarts from zero after interruption; completed output protected.',resume_command=shlex.join(['python3',str(Path(__file__).resolve()),'--episode',str(a.episode),'--output-root',str(base),'--gpu',str(a.gpu)]),timeout_seconds_per_stage=600,started_utc=datetime.now(timezone.utc).isoformat(),source_sha256={n:hashlib.sha256((root/'scripts'/n).read_bytes()).hexdigest() for n in ['launch_track3_episode_masks.py','run_track3_episode_detection.py','run_track3_episode_masks.py']})
(base/'masks-job.json').write_text(json.dumps(record,indent=2)+'\n');subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(root),shell],check=True);print('Started',session)
