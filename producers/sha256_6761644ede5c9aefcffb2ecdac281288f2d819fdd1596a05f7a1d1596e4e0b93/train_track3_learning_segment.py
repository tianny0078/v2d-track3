"""One real FlashSAC segment with local reward/loss logs and resumable state."""
import argparse,json,math,os,signal,time
from pathlib import Path

def atomic(p,v):
 p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');q.replace(p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--reference-checkpoint',required=True)
 ap.add_argument('--target',type=int,required=True);ap.add_argument('--resume-step',type=int);ap.add_argument('--output',type=Path,required=True)
 a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise ValueError('This isolated trial is assigned physical GPU1 only')
 os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
 import jax
 from dataclasses import replace
 from omegaconf import OmegaConf
 import train_track3_masked as entry
 from flash_chord.training.flash_sac import runner
 cfg=OmegaConf.load(a.config);cfg.training.total_environment_steps=a.target
 if a.resume_step is not None:
  cfg.training.resume=True;cfg.training.checkpoint_path=str(Path(cfg.output_dir)/f'state_{a.resume_step}.safetensors')
  cfg.training.checkpoint.load_replay_path=str(Path(cfg.output_dir)/f'replay_{a.resume_step}.safetensors')
 report={'state':'running','target':a.target,'resume_step':a.resume_step,'is_real_sac':True,'object_assistance':False,
  'contact_reference':'unknown GT force/contact; saved geometry proxy not measured force','latest_metrics':None}
 begin=time.monotonic();latest={};original_run=entry.run_training;original_compile=runner._compile_executables
 class Logger:
  def log(self,metrics,*,step):
   clean={k:float(v) if math.isfinite(float(v)) else None for k,v in metrics.items()}
   row=dict(environment_steps=int(step),elapsed_seconds=time.monotonic()-begin,metrics=clean)
   with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
   report.update(actual_environment_steps=int(step),latest_metrics=clean,elapsed_seconds=row['elapsed_seconds']);atomic(a.output/'report.json',report)
   if any(v is None for v in clean.values()):raise FloatingPointError('Nonfinite SAC metric; preserve failure without claiming completion')
  def close(self):pass
 def capture_compile(*pos,**kw):
  compiled=original_compile(*pos,**kw);consume=compiled.consume_transition
  def capture(*args,**kwargs):
   ret=consume(*args,**kwargs);latest['learner']=ret[0];return ret
  return replace(compiled,consume_transition=capture)
 def run(env,config,**kw):
  kw['logger']=Logger()
  try:return original_run(env,config,**kw)
  except KeyboardInterrupt:
   if 'learner' in latest:
    learner=latest['learner'];jax.block_until_ready(learner.global_update_step)
    paths=runner.save_learner_checkpoint(kw['checkpoint_dir'],learner,metadata=kw['checkpoint_metadata'],save_replay=True)
    report['interrupt_checkpoint']=dict(policy=str(paths.policy),state=str(paths.training_state),replay=str(paths.replay),step=int(jax.device_get(learner.environment_steps)))
   raise
 runner._compile_executables=capture_compile;entry.run_training=run
 signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
 try:
  entry.main(cfg,reference_checkpoint=a.reference_checkpoint,reference_mode='ground_truth_unknown')
  report.update(state='completed',actual_environment_steps=a.target)
 except BaseException as exc:
  report.update(state='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=f'{type(exc).__name__}: {exc}');raise
 finally:
  report['elapsed_seconds']=time.monotonic()-begin;atomic(a.output/'report.json',report)
if __name__=='__main__':main()
