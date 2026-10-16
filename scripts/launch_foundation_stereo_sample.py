"""Launch one isolated, offline learned-stereo three-frame comparison in tmux."""
import argparse,json,os,shlex,subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sequence',action='store_true');args=p.parse_args()
tag='foundation-stereo-'+('sequence' if args.sequence else 'sample')
root=Path(__file__).resolve().parents[1];base=root/'outputs/track3/episode_000011';out=base/tag.replace('-','_')
if (out/'report.json').exists() and json.loads((out/'report.json').read_text()).get('state')=='completed':
    print('Completed learned stereo sample exists; no duplicate started');raise SystemExit(0)
status=root/'notes/track3-foundation-stereo-prepare-status.json'
if not status.exists() or json.loads(status.read_text()).get('state')!='completed':
    raise SystemExit('Public model/dependency preparation has not completed')
name='v2d-t3-'+tag
if subprocess.check_output(['docker','ps','--filter','name=^/'+name+'$','--format','{{.Names}}'],text=True).strip():
    print('Matching learned stereo experiment active');raise SystemExit(0)
out.mkdir(exist_ok=True);image='sha256:367f7ab879e29469b7d222635f5797f962906ef0afd01e2586556850cc9aec32'
cmd=['docker','run','--rm','--pull=never','--name',name,'--network=none','--read-only','--tmpfs','/tmp:rw,exec,size=4g',
     '--gpus','device=0','--user',f'{os.getuid()}:{os.getgid()}','-e','PYTHONPATH=/work/deps','-e','CUDA_CACHE_PATH=/tmp/cuda',
     '-e','MPLCONFIGDIR=/tmp/matplotlib','-e','HF_HUB_OFFLINE=1']
for src,dst,ro in [(root/'scripts','/work/scripts',True),(root/'weights/foundation_stereo','/work/weights/foundation_stereo',True),
                   (root/'.deps/stereo-py311','/work/deps',True),(root/'data/track3_sample','/work/data/track3_sample',True),
                   (base,'/work/outputs/track3/episode_000011',True),(out,'/work/outputs/track3/episode_000011/'+tag.replace('-','_'),False)]:
    cmd+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
cmd+=['--entrypoint','python',image,'-u','/work/scripts/run_foundation_stereo_sample.py']
if args.sequence:cmd+=['--sequence']
stamp=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d-%H%M%S')
r={'session':'v2d-t3-'+tag+'-'+stamp,'argv':cmd,'image_id':image,
   'log':str(root/f'logs/track3-{tag}.log'),'exit_file':str(root/f'logs/track3-{tag}.exit'),
   'resume':f'cd {root} && python3 scripts/launch_foundation_stereo_sample.py'+(' --sequence' if args.sequence else ''),'training':False}
r['attach']='tmux attach -t '+r['session']
shell=shlex.join(cmd)+' >> '+shlex.quote(r['log'])+' 2>&1; task_rc=$?; printf "%s\\n" "$task_rc" > '+shlex.quote(r['exit_file'])+'; exit "$task_rc"'
(root/f'notes/track3-{tag}-job.json').write_text(json.dumps(r,indent=2)+'\n')
subprocess.run(['tmux','new-session','-d','-s',r['session'],'-c',str(root),shell],check=True);print('Started',r['session'])
