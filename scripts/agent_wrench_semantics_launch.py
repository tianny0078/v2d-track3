from pathlib import Path
import json,subprocess,shlex,datetime,socket
ROOT=Path(__file__).resolve().parents[1];prefix='v2d-t3-wrench-origin-';existing=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True)
if any(x.startswith(prefix) for x in existing.stdout.splitlines()):raise SystemExit('Matching wrench job already running')
stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S');session=prefix+stamp;log=ROOT/'logs'/f'track3-wrench-origin-{stamp}.log';exitfile=log.with_suffix('.exit');argv=['timeout','570','env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu','OPENBLAS_NUM_THREADS=4','OMP_NUM_THREADS=4',str(ROOT/'.deps/flash-chord-venv/bin/python'),'-u',str(ROOT/'scripts/agent_wrench_semantics_probe.py')];shell=shlex.join(argv)+' > '+shlex.quote(str(log))+' 2>&1; code=$?; printf "%s\\n" "$code" > '+shlex.quote(str(exitfile));subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),'bash','-c',shell],check=True)
note=ROOT/'notes/track3-wrench-origin-job.json';r={'state':'running','session':session,'log':str(log),'exit_file':str(exitfile),'hostname':socket.gethostname(),'argv':argv,'report':'outputs/track3/episode_000011/wrench_origin_semantics/report.json','resume':'python3 scripts/agent_wrench_semantics_launch.py','attach':f'tmux attach -t {session}','hard_timeout_seconds':570,'scope':'CPU COM transform consistency with actual contacts and unknown semantics prototype; no policy training','checkpoint':'physical_contacts.npz; deterministic source hash; completed result reuse'}
if note.exists():
 old=json.loads(note.read_text());r['previous_runs']=old.pop('previous_runs',[])+[old]
note.write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
