"""Persistent bounded official reference/scene preflight, no training."""
from pathlib import Path
import json,subprocess,shlex,datetime,socket
ROOT=Path(__file__).resolve().parents[1];prefix='v2d-t3-reference-preflight-'
existing=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True)
if any(s.startswith(prefix) for s in existing.stdout.splitlines()):raise SystemExit('Matching preflight already running')
stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S');session=prefix+stamp;log=ROOT/'logs'/f'track3-reference-preflight-{stamp}.log';exitfile=log.with_suffix('.exit')
argv=['timeout','570','env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu','OPENBLAS_NUM_THREADS=4','OMP_NUM_THREADS=4',str(ROOT/'.deps/flash-chord-venv/bin/python'),'-u',str(ROOT/'scripts/agent_reference_preflight.py')]
shell=shlex.join(argv)+' > '+shlex.quote(str(log))+' 2>&1; code=$?; printf "%s\\n" "$code" > '+shlex.quote(str(exitfile))
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),'bash','-c',shell],check=True)
note=ROOT/'notes/track3-reference-preflight-job.json';job={'state':'running','session':session,'log':str(log),'exit_file':str(exitfile),'hostname':socket.gethostname(),'report':'outputs/track3/episode_000011/flashchord_reference_preflight/report.json','resume':'python3 scripts/agent_reference_launch.py','attach':f'tmux attach -t {session}','argv':argv,'hard_timeout_seconds':570,'scope':'Official single-row ManoSharpaData load + CPU single-world Newton scene/solver compilation; no training or dynamics rollout','checkpoint':'reference_checkpoint.npz, parquet and source hashes; scene_checkpoint.npz when successful'}
if note.exists():
 previous=json.loads(note.read_text());job['previous_runs']=previous.pop('previous_runs',[])+[previous]
note.write_text(json.dumps(job,indent=2));print(json.dumps(job,indent=2))
