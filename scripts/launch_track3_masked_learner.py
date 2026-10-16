from pathlib import Path
import subprocess, shlex, json, datetime
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/track3/episode_000011/masked_learner'
OUT.mkdir(exist_ok=True,parents=True)
session='v2d-t3-masked-learner'
if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0:raise SystemExit('already running')
py=str(ROOT/'.deps/flash-chord-venv/bin/python')
config=OUT/'config.yaml'
subprocess.run([py,'-c',"from omegaconf import OmegaConf;import sys;c=OmegaConf.load(sys.argv[1]);c.scene.world_count=1;c.training.world_count=1;c.env.auto_reset=True;c.output_dir=sys.argv[3];OmegaConf.save(c,sys.argv[2])",str(ROOT/'outputs/track3/episode_000011/reference_alignment/learner_config.yaml'),str(config),str(OUT/'checkpoints')],check=True)
commands=[]
for resume in (False,True):
 argv=['timeout','600','env','CUDA_VISIBLE_DEVICES=3','OPENBLAS_NUM_THREADS=2','OMP_NUM_THREADS=2','XLA_PYTHON_CLIENT_PREALLOCATE=false',py,'-u',str(ROOT/'scripts/train_track3_masked.py'),'--config',str(config)]
 if resume:argv+=['--resume']
 commands.append(shlex.join(argv))
log=ROOT/'logs/track3-masked-learner.log';exitfile=log.with_suffix('.exit');exitfile.unlink(missing_ok=True)
cmd=' && '.join(commands)
cmd='('+cmd+') > '+shlex.quote(str(log))+' 2>&1; rc=$?; printf "%s\\n" "$rc" > '+shlex.quote(str(exitfile))
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),'bash','-c',cmd],check=True)
(ROOT/'notes/track3-masked-learner-job.json').write_text(json.dumps(dict(state='running',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),session=session,log=str(log),exit_file=str(exitfile),resume='python3 scripts/launch_track3_masked_learner.py',commands=commands,checkpoint=str(OUT/'checkpoints'),hard_timeout_seconds=1200,training_ready=False),indent=2))
