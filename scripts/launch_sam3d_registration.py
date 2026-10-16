"""Resume bounded SAM3D registration in server tmux; no duplicate container."""
from datetime import datetime
import json,os,shlex,subprocess
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/track3/episode_000011';OUT=BASE/'sam3d_metric_registration'
OUT.mkdir(exist_ok=True);name='v2d-t3-sam3d-register'
if (OUT/'report.json').exists() and json.loads((OUT/'report.json').read_text()).get('state')=='completed':raise SystemExit('Already completed; no duplicate')
if subprocess.check_output(['docker','ps','--filter','name=^/'+name+'$','--format','{{.Names}}'],text=True).strip():raise SystemExit('Already active; no duplicate')
cmd=['docker','run','--rm','--pull=never','--name',name,'--gpus','device=2','--network=none','--read-only','--cpus','8','--memory','24g',
     '--tmpfs','/tmp:rw,size=2g','--user',f'{os.getuid()}:{os.getgid()}',
     '--mount',f'type=bind,src={ROOT}/scripts,dst=/work/scripts,readonly',
     '--mount',f'type=bind,src={BASE},dst=/work/outputs/track3/episode_000011,readonly',
     '--mount',f'type=bind,src={OUT},dst=/work/outputs/track3/episode_000011/sam3d_metric_registration',
     '-e','NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility','-e','PYOPENGL_PLATFORM=egl','-e','PYTHONDONTWRITEBYTECODE=1',
     '-e','OPENBLAS_NUM_THREADS=4','-e','OMP_NUM_THREADS=4',
     '--entrypoint','python','sha256:7ada1f16b351db919fed82a8f28e1bdb3593d0be05c169a94f3a5b5c81b88727','-u','/work/scripts/register_sam3d_candidates.py']
stamp=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d-%H%M%S');session=name+'-'+stamp
log=ROOT/f'logs/track3-sam3d-register-{stamp}.log';exitfile=log.with_suffix('.exit')
job={'session':session,'argv':cmd,'log':str(log),'exit_file':str(exitfile),'attach':'tmux attach -t '+session,
     'resume':'cd '+str(ROOT)+' && python3 scripts/launch_sam3d_registration.py','checkpoint':'Per-object source-hashed registration JSON','budget_seconds':540}
(ROOT/'notes/track3-sam3d-registration-job.json').write_text(json.dumps(job,indent=2)+'\n')
shell=shlex.join(cmd)+' >> '+shlex.quote(str(log))+' 2>&1; task_rc=$?; printf "%s\\n" "$task_rc" > '+shlex.quote(str(exitfile))+'; exit "$task_rc"'
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),shell],check=True);print(json.dumps(job,indent=2))
