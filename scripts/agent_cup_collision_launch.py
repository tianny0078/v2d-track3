"""Persistent bounded compound-cup collision backend experiment."""
from pathlib import Path
import json,subprocess,shlex,datetime,socket,sys
ROOT=Path(__file__).resolve().parents[1]
existing=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True)
if any(s.startswith('v2d-t3-agent-cup-collision-') for s in existing.stdout.splitlines()):raise SystemExit('Cup collision job already running')
stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S');session='v2d-t3-agent-cup-collision-'+stamp
log=ROOT/'logs'/f'track3-agent-cup-collision-{stamp}.log';exitfile=log.with_suffix('.exit');cache=ROOT/'outputs/track3/episode_000011/agent_cup_collision_smoke/warp_cache';cache.mkdir(parents=True,exist_ok=True)
script='agent_cup_collision_urdf.py' if '--urdf-only' in sys.argv else 'agent_cup_collision_smoke.py'
argv=['env','CUDA_VISIBLE_DEVICES=','OPENBLAS_NUM_THREADS=4','OMP_NUM_THREADS=4','WP_CACHE_DIR='+str(cache),str(ROOT/'.deps/flash-chord-venv/bin/python'),'-u',str(ROOT/'scripts'/script)]
shell=shlex.join(argv)+' > '+shlex.quote(str(log))+' 2>&1; code=$?; printf "%s\\n" "$code" > '+shlex.quote(str(exitfile))
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),'bash','-c',shell],check=True)
job={'session':session,'argv':argv,'log':str(log),'exit_file':str(exitfile),'hostname':socket.gethostname(),'report':'outputs/track3/episode_000011/agent_cup_collision_smoke/report.json','attach':f'tmux attach -t {session}','resume':'python3 scripts/agent_cup_collision_launch.py','compute_budget_seconds':540,'checkpoint':'Per-wall candidate OBJ/MJCF/static-query and dynamics traces; completed stages inspectable','scope':'CPU actual collision/dynamics asset smoke only; no recovered trajectory validation'}
note=ROOT/'notes/track3-agent-cup-collision-job.json'
if note.exists():
    previous=json.loads(note.read_text());job['previous_runs']=previous.pop('previous_runs',[])+[previous]
job['mode']='urdf_export_import_only' if '--urdf-only' in sys.argv else 'native_or_newton_resume'
note.write_text(json.dumps(job,indent=2));print(json.dumps(job,indent=2))
