from pathlib import Path
import subprocess,shlex,json,datetime
root=Path(__file__).resolve().parents[1];session='v2d-t3-masked-physics'
check=subprocess.run(['tmux','has-session','-t',session],capture_output=True)
if check.returncode==0:raise SystemExit('Existing masked-objective job running')
log=root/'logs/track3-masked-physics.log';exitfile=log.with_suffix('.exit')
argv=['timeout','600','env','CUDA_VISIBLE_DEVICES=3','JAX_PLATFORMS=cpu','OPENBLAS_NUM_THREADS=2','OMP_NUM_THREADS=2',str(root/'.deps/flash-chord-venv/bin/python'),'-u',str(root/'scripts/verify_track3_masked_physics.py')]
cmd=shlex.join(argv)+' > '+shlex.quote(str(log))+' 2>&1; rc=$?; printf "%s\\n" "$rc" > '+shlex.quote(str(exitfile))
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(root),'bash','-c',cmd],check=True)
(root/'notes/track3-masked-physics-job.json').write_text(json.dumps(dict(state='running',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),session=session,argv=argv,log=str(log),exit_file=str(exitfile),resume='python3 scripts/launch_track3_masked_physics.py',checkpoint='outputs/track3/episode_000011/masked_physics_integration/checkpoint.npz',report='outputs/track3/episode_000011/masked_physics_integration/report.json',hard_timeout_seconds=600,production_integrated=False),indent=2))
