"""Inspect completed SAM3D candidates in a persistent offline renderer job."""
from datetime import datetime
import json,os,shlex,subprocess
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/track3/episode_000011';OUT=BASE/'object_meshes_sam3d_local'
assert json.loads((OUT/'report.json').read_text())['state']=='completed'
if (OUT/'render_report.json').exists() and json.loads((OUT/'render_report.json').read_text()).get('state')=='completed':raise SystemExit('Already rendered; no duplicate')
name='v2d-t3-sam3d-render'
if subprocess.check_output(['docker','ps','--filter','name=^/'+name+'$','--format','{{.Names}}'],text=True).strip():raise SystemExit('Render active; no duplicate')
cmd=['docker','run','--rm','--pull=never','--name',name,'--gpus','device=2','--network=none','--read-only',
     '--tmpfs','/tmp:rw,size=2g','--user',f'{os.getuid()}:{os.getgid()}',
     '--mount',f'type=bind,src={ROOT}/scripts,dst=/work/scripts,readonly',
     '--mount',f'type=bind,src={BASE},dst=/work/outputs/track3/episode_000011,readonly',
     '--mount',f'type=bind,src={OUT},dst=/work/outputs/track3/episode_000011/object_meshes_sam3d_local',
     '-e','NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility','-e','PYOPENGL_PLATFORM=egl','-e','PYTHONDONTWRITEBYTECODE=1',
     '--entrypoint','python','sha256:7ada1f16b351db919fed82a8f28e1bdb3593d0be05c169a94f3a5b5c81b88727','-u','/work/scripts/render_sam3d_candidates.py']
stamp=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d-%H%M%S');session=name+'-'+stamp
log=ROOT/f'logs/track3-sam3d-render-{stamp}.log';exitfile=log.with_suffix('.exit')
job={'session':session,'argv':cmd,'log':str(log),'exit_file':str(exitfile),'attach':'tmux attach -t '+session,
     'resume':'cd '+str(ROOT)+' && python3 scripts/launch_sam3d_render.py'}
(ROOT/'notes/track3-sam3d-render-job.json').write_text(json.dumps(job,indent=2)+'\n')
shell=shlex.join(cmd)+' >> '+shlex.quote(str(log))+' 2>&1; task_rc=$?; printf "%s\\n" "$task_rc" > '+shlex.quote(str(exitfile))+'; exit "$task_rc"'
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),shell],check=True);print(json.dumps(job,indent=2))
