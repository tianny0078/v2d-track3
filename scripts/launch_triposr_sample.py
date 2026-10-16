"""Start one offline two-object mesh experiment in an existing GPU image."""
import json,os,shlex,subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

root=Path(__file__).resolve().parents[1];out=root/'outputs/track3/episode_000011/object_meshes_triposr'
if (out/'report.json').exists() and json.loads((out/'report.json').read_text()).get('state')=='completed':
    print('Reusing completed meshes');raise SystemExit(0)
name='v2d-t3-triposr-sample'
if subprocess.check_output(['docker','ps','--filter','name=^/'+name+'$','--format','{{.Names}}'],text=True).strip():
    print('Existing mesh task is active; no duplicate started');raise SystemExit(0)
out.mkdir(parents=True,exist_ok=True)
image='sha256:367f7ab879e29469b7d222635f5797f962906ef0afd01e2586556850cc9aec32'
cmd=['docker','run','--rm','--pull=never','--name',name,'--network=none','--read-only','--tmpfs','/tmp:rw,exec,size=4g',
     '--gpus','device=0','--user',f'{os.getuid()}:{os.getgid()}','-e','HF_HUB_OFFLINE=1','-e','HF_HOME=/tmp/hf',
     '-e','TORCH_HOME=/tmp/torch','-e','CUDA_CACHE_PATH=/tmp/cuda-cache','-e','PYTORCH_KERNEL_CACHE_PATH=/tmp/torch-kernels',
     '-e','MPLCONFIGDIR=/tmp/matplotlib']
for src,dst,ro in [(root/'scripts','/work/scripts',True),(root/'weights/triposr','/work/weights/triposr',True),
                   (root/'.deps/triposr-py311','/work/runtime_deps',True),
                   (root/'vendor/TripoSR','/work/vendor/TripoSR',True),
                   (root/'outputs/track3/episode_000011/mesh_inputs','/work/outputs/track3/episode_000011/mesh_inputs',True),
                   (out,'/work/outputs/track3/episode_000011/object_meshes_triposr',False)]:
    cmd+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
cmd+=['--entrypoint','python',image,'-u','/work/scripts/run_triposr_sample.py']
stamp=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d-%H%M%S')
r={'session':'v2d-t3-mesh-'+stamp,'argv':cmd,'image_id':image,'log':str(root/'logs/track3-mesh-sample.log'),
   'exit_file':str(root/'logs/track3-mesh-sample.exit'),'resume':f'cd {root} && python3 scripts/launch_triposr_sample.py','training':False}
r['attach']='tmux attach -t '+r['session']
shell=shlex.join(cmd)+' >> '+shlex.quote(r['log'])+' 2>&1; task_rc=$?; printf "%s\\n" "$task_rc" > '+shlex.quote(r['exit_file'])+'; exit "$task_rc"'
(root/'notes/track3-mesh-job.json').write_text(json.dumps(r,indent=2)+'\n')
subprocess.run(['tmux','new-session','-d','-s',r['session'],'-c',str(root),shell],check=True);print('Started',r['session'])
