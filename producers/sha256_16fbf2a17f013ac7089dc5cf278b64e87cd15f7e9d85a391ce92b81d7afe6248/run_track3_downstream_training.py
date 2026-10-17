"""Single-GPU, checkpoint-resumable diagnostic learner and real policy evaluation."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import time
import fcntl
import argparse

ROOT=Path('/home/tianyuan/projects/v2d')
OUT=ROOT/'outputs/track3/episode_000011/downstream_training_v1'
JOB=ROOT/'notes/track3-downstream-training-job.json'
PY=ROOT/'.deps/flash-chord-venv/bin/python'


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=['proxy','contact_off'],default='proxy')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--job-path', type=Path)
    parser.add_argument('--reference-checkpoint', type=Path,
                        default=ROOT/'outputs/track3/episode_000011/downstream_retarget_v1/reference_checkpoint.npz')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--train-steps', type=int, default=50000)
    parser.add_argument('--resume-steps', type=int, default=60000)
    parser.add_argument('--episode', type=int, default=11)
    parser.add_argument('--reference-mode', choices=['geometry_proxy','ground_truth_unknown'], default='geometry_proxy')
    parser.add_argument('--proxy-object-names', nargs='+', default=['blue_cup','beige_cup'])
    args=parser.parse_args()
    if bool(args.output_dir) != bool(args.job_path):
        parser.error('Custom output requires both --output-dir and --job-path')
    if args.train_steps <= 0 or args.resume_steps <= args.train_steps:
        parser.error('Expected 0 < train-steps < resume-steps')
    global OUT,JOB
    if args.variant=='contact_off':
        OUT=ROOT/'outputs/track3/episode_000011/downstream_training_contact_off_v1'
        JOB=ROOT/'notes/track3-downstream-contact-off-job.json'
    if args.output_dir:
        OUT=args.output_dir.resolve(); JOB=args.job_path.resolve()
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'.training-supervisor.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Training supervisor already running; no duplicate.')
    report=json.loads((OUT/'report.json').read_text()) if (OUT/'report.json').exists() else dict(
        state='running',started_utc=now(),physical_gpu=args.gpu,world_count=8,requested_training_steps=args.train_steps,
        resume_terminal_steps=args.resume_steps,scope='Current reconstructed geometry + explicit positive geometry proxy; approximate robot retarget diagnostic',
        ground_truth_force_known=False,object_assistance=False,production_ready=False,
        robot_reference_fully_feasible=False,stages=[])
    report['reward_variant']=args.variant
    if report.get('state')=='completed' and (OUT/'evaluation.json').exists():
        print('Completed training, restore and evaluation exist; reuse without rerun.',flush=True)
        return
    report['state']='running'
    config=OUT/'config.yaml';snapshot=OUT/'producer-source'
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(args.gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',
        OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',WANDB_MODE='disabled',
        PYTHONPATH=str(snapshot),JAX_COMPILATION_CACHE_DIR=str(OUT/'jax_cache'))
    report['resume_semantics']='Latest common state/replay snapshot restores optimizer/replay/RNG via FlashSAC; filesystem report is a supervisor heartbeat.'

    def update():
        report['updated_utc']=now()
        report['checkpoint_files']=[p.name for p in sorted((OUT/'checkpoints').glob('*.safetensors'))]
        save(OUT/'report.json',report)
        old=json.loads(JOB.read_text())
        old.update(state=report['state'],updated_utc=report['updated_utc'],stage=report.get('stage'),
            report=str(OUT/'report.json'),checkpoint_files=report['checkpoint_files'],pid=os.getpid())
        save(JOB,old)

    def run(name,argv,timeout):
        stage=dict(name=name,argv=list(map(str,argv)),started_utc=now(),log=str(OUT/f'{name}.log'),state='running')
        report['stages'].append(stage);report['stage']=name;update()
        with (OUT/f'{name}.log').open('a') as output:
            process=subprocess.Popen(list(map(str,argv)),cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
            stage['pid']=process.pid;start=time.monotonic()
            while process.poll() is None:
                if time.monotonic()-start>timeout:
                    process.terminate()
                    try:process.wait(timeout=20)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                    stage['state']='timed_out';break
                update();time.sleep(5)
            stage.update(exit_code=process.returncode,completed_utc=now())
            if stage['state']=='running':stage['state']='completed' if process.returncode==0 else 'failed'
        (OUT/f'{name}.exit').write_text(str(process.returncode)+'\n');update()
        if process.returncode:raise RuntimeError(f'{name} failed: {stage["log"]}')

    common=lambda: sorted(int(p.stem.split('_')[-1]) for p in (OUT/'checkpoints').glob('state_*.safetensors')
        if (OUT/'checkpoints'/('replay_'+p.stem.split('_')[-1]+'.safetensors')).exists())
    def learner(target,label):
        current=common();latest=current[-1] if current else None
        if latest is not None and latest>=target:return
        learner_argv=[PY,'-u',snapshot/'train_track3_masked.py','--config',config,
            '--reference-mode',args.reference_mode,'--reference-checkpoint',
            args.reference_checkpoint.resolve(),
            '--total-environment-steps',str(target)]
        if args.reference_mode=='geometry_proxy':
            learner_argv+=['--proxy-object-names',*args.proxy_object_names]
        if latest is not None:learner_argv+=['--resume','--resume-step',str(latest)]
        run(label,learner_argv,3600)

    try:
        update();learner(args.train_steps,f'train_{args.train_steps}');learner(args.resume_steps,f'resume_{args.resume_steps}')
        report['actual_resume_verified']=(OUT/'resume_entry.json').exists()
        evaluator=ROOT/'scripts/evaluate_track3_downstream.py'
        run('policy_rollout',[PY,'-u',evaluator,'--checkpoint',OUT/f'checkpoints/policy_{args.resume_steps}.safetensors',
                             '--output',OUT,'--episode',str(args.episode)],600)
        report.update(state='completed',completed_utc=now(),policy_rollout_performed=True,
            evaluation_file=str(OUT/'evaluation.json'))
    except Exception as exc:
        report.update(state='failed',failure=str(exc),completed_utc=now())
        raise
    finally:
        update()
        save(ROOT/'notes/track3-publication-request.json',dict(requested_utc=now(),reason='downstream training phase update',
             artifact_paths=[str((OUT/'report.json').relative_to(ROOT))]))


if __name__=='__main__':main()
