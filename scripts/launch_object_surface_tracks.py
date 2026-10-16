"""Launch one offline visible-surface rigid-track experiment in persistent tmux."""
import json,os,shlex,subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
root=Path(__file__).resolve().parents[1];base=root/'outputs/track3/episode_000011';out=base/'object_surface_tracks'
if (out/'report.json').exists() and json.loads((out/'report.json').read_text()).get('state')=='completed':
    print('Completed surface-track experiment exists; no duplicate');raise SystemExit(0)
name='v2d-t3-object-surface-tracks'
if subprocess.check_output(['docker','ps','--filter','name=^/'+name+'$','--format','{{.Names}}'],text=True).strip():
    print('Surface-track experiment already active');raise SystemExit(0)
out.mkdir(exist_ok=True);image='sha256:367f7ab879e29469b7d222635f5797f962906ef0afd01e2586556850cc9aec32'
cmd=['docker','run','--rm','--pull=never','--name',name,'--network=none','--read-only','--tmpfs','/tmp:rw,size=1g',
     '--user',f'{os.getuid()}:{os.getgid()}','-e','OPENBLAS_NUM_THREADS=2','-e','OMP_NUM_THREADS=2',
     '--mount',f'type=bind,src={root}/scripts,dst=/work/scripts,readonly',
     '--mount',f'type=bind,src={base},dst=/work/outputs/track3/episode_000011,readonly',
     '--mount',f'type=bind,src={out},dst=/work/outputs/track3/episode_000011/object_surface_tracks',
     '--entrypoint','python',image,'-u','/work/scripts/run_object_surface_tracks.py']
stamp=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d-%H%M%S')
r={'session':'v2d-t3-object-surface-tracks-'+stamp,'argv':cmd,'image_id':image,'training':False,
   'log':str(root/'logs/track3-object-surface-tracks.log'),'exit_file':str(root/'logs/track3-object-surface-tracks.exit'),
   'resume':f'cd {root} && python3 scripts/launch_object_surface_tracks.py','partial_resume':'Per-frame pose/report checkpoints; source hashes checked before resume.'}
r['attach']='tmux attach -t '+r['session']
shell=shlex.join(cmd)+' >> '+shlex.quote(r['log'])+' 2>&1; task_rc=$?; printf "%s\\n" "$task_rc" > '+shlex.quote(r['exit_file'])+'; exit "$task_rc"'
(root/'notes/track3-object-surface-tracks-job.json').write_text(json.dumps(r,indent=2)+'\n')
subprocess.run(['tmux','new-session','-d','-s',r['session'],'-c',str(root),shell],check=True);print('Started',r['session'])
