"""Launch the bounded cup pilot in its own persistent tmux session."""
from pathlib import Path
import json,subprocess,shlex,datetime,socket
ROOT=Path(__file__).resolve().parents[1]
existing=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True)
if any(s.startswith('v2d-t3-agent-cup-') for s in existing.stdout.splitlines()):raise SystemExit('Cup job already running; inspect it instead.')
stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S');session='v2d-t3-agent-cup-'+stamp
log=ROOT/'logs'/f'track3-agent-cup-{stamp}.log';exitfile=log.with_suffix('.exit');log.parent.mkdir(exist_ok=True)
argv=['env','OPENBLAS_NUM_THREADS=4','OMP_NUM_THREADS=4',str(ROOT/'.deps/flash-chord-venv/bin/python'),'-u',str(ROOT/'scripts/agent_cup_reconstruct.py')]
shell=shlex.join(argv)+' > '+shlex.quote(str(log))+' 2>&1; code=$?; printf "%s\\n" "$code" > '+shlex.quote(str(exitfile))
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),'bash','-c',shell],check=True)
info={'session':session,'argv':argv,'log':str(log),'exit_file':str(exitfile),'report':'outputs/track3/episode_000011/agent_cup_reconstruction/report.json','hostname':socket.gethostname(),'attach':f'tmux attach -t {session}','resume':'python3 scripts/agent_cup_launch.py','cache':'outputs/track3/episode_000011/agent_cup_reconstruction/training_clouds.npz','scope':'Only code-generated hollow cup geometry and validation; no policy training'}
(ROOT/'notes/track3-agent-cup-job.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))
