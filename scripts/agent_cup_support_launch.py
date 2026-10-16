"""Launch CPU-only table-feasible temporal cup windows with persistence."""
from pathlib import Path
import json,subprocess,shlex,datetime,socket
ROOT=Path(__file__).resolve().parents[1]
existing=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True)
if any(s.startswith('v2d-t3-agent-cup-support-') for s in existing.stdout.splitlines()):raise SystemExit('Supported cup job already running')
stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S');session='v2d-t3-agent-cup-support-'+stamp
log=ROOT/'logs'/f'track3-agent-cup-support-{stamp}.log';exitfile=log.with_suffix('.exit')
argv=['env','OPENBLAS_NUM_THREADS=4','OMP_NUM_THREADS=4',str(ROOT/'.deps/flash-chord-venv/bin/python'),'-u',str(ROOT/'scripts/agent_cup_support_refine.py')]
shell=shlex.join(argv)+' > '+shlex.quote(str(log))+' 2>&1; code=$?; printf "%s\\n" "$code" > '+shlex.quote(str(exitfile))
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),'bash','-c',shell],check=True)
job={'session':session,'argv':argv,'log':str(log),'exit_file':str(exitfile),'hostname':socket.gethostname(),'report':'outputs/track3/episode_000011/agent_cup_pose_supported/report.json','attach':f'tmux attach -t {session}','resume':'python3 scripts/agent_cup_support_launch.py','compute_budget_seconds':540,'checkpoint':'Per-window selected poses plus checkpoint_poses.npz, source-hash checked','scope':'CPU fixed cup shape, hard table nonpenetration, joint temporal pose windows only'}
(ROOT/'notes/track3-agent-cup-support-job.json').write_text(json.dumps(job,indent=2));print(json.dumps(job,indent=2))
