"""Learning v3 runner (any episode, GPU1/2). Generalised from run_track3_ep37_learning.py.
Single ep37 training -> saved-policy evaluation at each ~5M milestone; no daemon."""
import argparse,fcntl,hashlib,json,math,os,shutil,struct,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/home/tianyuan/projects/v2d')
DEFAULT=ROOT/'outputs/track3/advice_restart_20261017/ep37_contact_learning_50m_v1'
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify_prepared_inputs(p):
 for path,digest in p['input_artifact_sha256'].items():assert sha(path)==digest,'Prepared input changed'
 for path,digest in p['binding'].items():
  if path.startswith('parameter:'):
   if path!='parameter:hand_tracking_weight':raise ValueError(f'Unknown preparation parameter: {path}')
   from omegaconf import OmegaConf
   import math
   assert isinstance(digest,(int,float)) and math.isfinite(digest) and digest>0,'Invalid bound hand weight'
   cfg=OmegaConf.to_container(OmegaConf.load(p['config']),resolve=True)
   for objective in (cfg['objective'],cfg['env']['objective']['config']):
    for term in ('hand_keypoints','hand_joint_pos'):
     assert objective[term]['weight']==digest,'Prepared base hand weight differs from bound parameter'
   for holder in (cfg,cfg['training']):
    for stage in holder.get('curriculum',{}).get('stages',[]):
     for term in ('hand_keypoints','hand_joint_pos'):
      assert stage['objective_weights'][term]==digest,'Prepared curriculum hand weight differs from bound parameter'
  elif not path.endswith('.py'):assert sha(path)==digest,'Frozen source data/asset changed'
def contact_observer_command(argv,observer):
 return [argv[0],'-u','-c',
  'import importlib.util,runpy,sys; '
  'p=sys.argv.pop(1); s=importlib.util.spec_from_file_location("track3_contact_trace",p); '
  'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
  'sys.modules[s.name]=m; p=sys.argv.pop(1); sys.argv[0]=p; '
  'runpy.run_path(p,run_name="__main__")',str(observer),*argv[2:]]
def save(p,d):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');q.replace(p)
def header(p):
 with p.open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0];assert n<32*1024**2;h=json.loads(f.read(n))
  end=max(v['data_offsets'][1] for k,v in h.items() if k!='__metadata__')
  assert p.stat().st_size==8+n+end,'Incomplete safetensors payload'
 return h['__metadata__']
def complete_sets(folder):
 good=[]
 for p in folder.glob('state_*.safetensors'):
  n=int(p.stem.split('_')[-1]);paths=[folder/f'{kind}_{n}.safetensors' for kind in ['policy','state','replay']]
  if not all(q.exists() for q in paths):continue
  try:
   meta=[header(q) for q in paths]
   assert {m['user.environment_steps'] for m in meta}=={str(n)}
   assert len({m['user.checkpoint_set_id'] for m in meta})==1
   good.append(n)
  except (OSError,ValueError,AssertionError,KeyError,struct.error):continue
 return sorted(good)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--episode',type=int,required=True);ap.add_argument('--gpu',choices=['0','1','2','3'],required=True)  # user 2026-10-23: all 4 GPUs usable
 ap.add_argument('--output',type=Path);ap.add_argument('--prepare-only',action='store_true')
 ap.add_argument('--milestone-steps',type=int,default=1_000_000);ap.add_argument('--milestones',type=int,default=5)
 ap.add_argument('--worlds',type=int,default=2048);ap.add_argument('--segment-timeout',type=int,default=7200)
 ap.add_argument('--max-new-milestones',type=int,
  help='Stop after this many newly completed train/replay milestones; resume keeps existing results.')
 ap.add_argument('--through-step',type=int,
  help='Absolute last milestone allowed, including across interrupted invocations.')
 ap.add_argument('--learning-output',type=Path,
  help='Refresh the existing learning consumer for this producer after each completed replay.')
 ap.add_argument('--checkpoint-root',type=Path,
  help='Isolated checkpoint root, forwarded on preparation and checked against an existing frozen trial.')
 ap.add_argument('--hand-tracking-weight',type=float,
  help='Explicit hand-weight ablation, forwarded on preparation and checked against an existing frozen trial.')
 ap.add_argument('--contact-observer',type=Path,
  help='Optional observation-only contact module; frozen training sources remain unchanged. Effective on supervisor restart only.')
 a=ap.parse_args()
 if a.max_new_milestones is not None and a.max_new_milestones<=0:
  ap.error('--max-new-milestones must be positive')
 if a.hand_tracking_weight is not None and (not math.isfinite(a.hand_tracking_weight) or a.hand_tracking_weight<=0):
  ap.error('--hand-tracking-weight must be finite and positive')
 a.output=a.output or ROOT/'outputs/track3'/os.environ.get('TRACK3_LEARNING_TREE','learning_v3')/f'episode_{a.episode:06d}/learning'
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 lock=(out/'.trial.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 py=ROOT/'.deps/flash-chord-venv/bin/python';prep=out/'preparation.json'
 if not prep.exists() or json.loads(prep.read_text()).get('state')!='prepared':
  prepare_argv=[str(py),'-u',str(ROOT/'scripts/prepare_track3_learning_v3.py'),'--episode',str(a.episode),'--output',str(out),'--worlds',str(a.worlds),
   '--milestone-steps',str(a.milestone_steps),'--milestones',str(a.milestones)]
  if a.hand_tracking_weight is not None:prepare_argv+=['--hand-tracking-weight',str(a.hand_tracking_weight)]
  if a.checkpoint_root is not None:prepare_argv+=['--checkpoint-root',str(a.checkpoint_root.resolve())]
  with (out/'prepare.log').open('a') as log:
   subprocess.run(prepare_argv,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
 p=json.loads(prep.read_text());assert p['world_count']==a.worlds and p['episode']==a.episode
 if a.through_step is not None and a.through_step not in p['milestones']:
  raise ValueError('--through-step must be a prepared milestone')
 if a.checkpoint_root is not None and Path(p['checkpoint_root']).resolve()!=a.checkpoint_root.resolve():
  raise ValueError('Requested checkpoint root differs from frozen preparation; use a separate output directory')
 if a.hand_tracking_weight is not None and p['binding'].get('parameter:hand_tracking_weight')!=a.hand_tracking_weight:
  raise ValueError('Requested hand weight differs from frozen preparation; use a separate output directory')
 # Runtime scripts are read from the frozen snapshot; mutable working scripts
 # need not remain identical after preparation. Data and assets must remain so.
 verify_prepared_inputs(p)
 # Validate the actual executable snapshot before reporting readiness or
 # touching NAS. Working scripts may evolve; the bound snapshot may not.
 for name,digest in p['runtime_source_sha256'].items():
  assert sha(out/'producer-source'/name)==digest,f'Frozen runtime changed: {name}'
 if a.prepare_only:
  print(json.dumps(dict(state='prepared_inputs_verified',output=str(out),world_count=p['world_count'],fingerprint=p['fingerprint'])))
  return
 nas=Path(p['checkpoint_root']);training=nas/'training';checkpoints=training/'checkpoints';snapshot=training/'producer-source'
 checkpoints.mkdir(parents=True,exist_ok=True);snapshot.mkdir(exist_ok=True)
 # Real write probe; only this trial directory. Checkpoint/replay bytes never go on local disk.
 probe=nas/'.write_probe';probe.write_text('track3 trial write check');probe.unlink()
 for name,digest in p['runtime_source_sha256'].items():
  src=out/'producer-source'/name;assert sha(src)==digest
  dst=snapshot/name
  if dst.exists():assert sha(dst)==digest,'Frozen runtime changed'
  else:shutil.copy2(src,dst)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu,  # Track3 owns GPU1/GPU2 (agreed split)
 XLA_PYTHON_CLIENT_PREALLOCATE='false',WANDB_MODE='disabled',
  OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONPATH=str(snapshot),
  JAX_COMPILATION_CACHE_DIR=str(out/'jax_cache'))
 observer=None
 if a.contact_observer is not None:
  source=a.contact_observer.resolve();digest=sha(source)
  folder=out/'observation-source'/digest;folder.mkdir(parents=True,exist_ok=True)
  observer=folder/'track3_contact_trace.py'
  if observer.exists():assert sha(observer)==digest,'Observation source changed'
  else:shutil.copy2(source,observer)
  save(folder/'binding.json',dict(source=str(source),sha256=digest,
   scope='Evaluation contact readout only; no training, action or physics modification'))
 report_path=out/'report.json';report=json.loads(report_path.read_text()) if report_path.exists() else dict(state='running',started_utc=now(),episode=a.episode,gpu=a.gpu,milestones=[],production_pass=False,submission_performed=False)
 def update(**kw):report.update(kw,updated_utc=now());save(report_path,report)
 def run(label,argv,timeout):
  update(state='running',stage=label)
  child_env=env.copy()
  if observer is not None and label.startswith('evaluate_'):
   # An evaluator executed by filename puts its frozen directory first in
   # sys.path, so PYTHONPATH alone cannot override its sibling contact module.
   # Preload only the explicitly selected observer, then run the frozen evaluator.
   argv=contact_observer_command(argv,observer)
   update(evaluation_contact_observer=dict(path=str(observer),sha256=sha(observer)))
  with (out/f'{label}.log').open('a') as log:
   proc=subprocess.Popen([str(v) for v in argv],cwd=ROOT,env=child_env,stdout=log,stderr=subprocess.STDOUT)
   update(child_pid=proc.pid)
   try:rc=proc.wait(timeout=timeout)
   except subprocess.TimeoutExpired:
    proc.terminate()
    try:proc.wait(timeout=180)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()
    raise RuntimeError(f'{label}: bounded timeout; last complete checkpoint retained')
  (out/f'{label}.exit').write_text(str(rc)+'\n')
  if rc:raise RuntimeError(f'{label} failed exit {rc}; see {out/label}.log')
 try:
  new_milestones=0
  def consume(target):
   # Commit the retry intent before invoking the consumer. A failed refresh
   # must not be skipped merely because its simulator replay is complete.
   update(pending_learning_step=target)
   subprocess.run([str(py),str(ROOT/'scripts/summarize_track3_learning_curve.py'),
    '--base',str(out),'--output',str(a.learning_output.resolve())],check=True,cwd=ROOT)
   update(pending_learning_step=None)
  pending=report.get('pending_learning_step')
  if a.through_step is not None and pending is not None and pending>a.through_step:
   raise ValueError('Pending learning step exceeds absolute budget')
  if pending is not None:
   if a.learning_output is None:raise ValueError('Pending learning consumer requires --learning-output')
   assert any(r['step']==pending and r['state']=='completed' for r in report['milestones'])
   consume(pending)
   new_milestones+=1
   if pending==p['actual_step_cap']:
    update(state='completed_step_cap',actual_environment_steps=pending);return
   if a.max_new_milestones is not None and new_milestones>=a.max_new_milestones:
    update(state='stopped_between_segments',stop_reason='bounded_milestone_budget',
     completed_new_milestones=new_milestones);return
  for target in p['milestones']:
   if a.through_step is not None and target>a.through_step:
    update(state='stopped_between_segments',stop_reason='absolute_step_budget',
     through_step=a.through_step)
    return
   prior=next((r for r in report['milestones'] if r['step']==target and r['state']=='completed'),None)
   if prior:continue
   if (out/'stop-after-segment').exists():update(state='stopped_between_segments');return
   groups=complete_sets(checkpoints);latest=groups[-1] if groups else None
   if latest is None or latest<target:
    segment=out/f'segments/step_{target}';segment.mkdir(parents=True,exist_ok=True)
    argv=[py,'-u',snapshot/'train_track3_learning_segment.py','--config',p['config'],
      '--reference-checkpoint',p['reference_checkpoint'],'--target',target,'--output',segment]
    if latest is not None:argv+=['--resume-step',latest]
    run(f'train_{target}',argv,a.segment_timeout)
   assert target in complete_sets(checkpoints),'Missing matching policy/state/replay snapshot'
   evaluation=out/f'evaluations/step_{target}'
   replay_report=evaluation/'replay-report.json'
   if not replay_report.exists():
    run(f'evaluate_{target}',[py,'-u',snapshot/'evaluate_track3_canonical_checkpoint.py',
      '--checkpoint',checkpoints/f'policy_{target}.safetensors','--object-manifest',p['object_manifest'],
      '--episode',a.episode,'--output',evaluation,'--substeps',25,'--record-contacts'],1200)
   rr=json.loads(replay_report.read_text());assert rr['policy_rollout_performed'] and rr['runtime']['actual_policy_calls']==p['frames']
   segment_report=out/f'segments/step_{target}/report.json'
   sr=json.loads(segment_report.read_text()) if segment_report.exists() else {}
   entry={'state':'completed','step':target,'checkpoint':str(checkpoints/f'policy_{target}.safetensors'),
    'reward_metrics':sr.get('latest_metrics'),'proxy_metrics':rr['metrics'],'evaluation_report':str(replay_report),
    'contact_trace':rr['runtime'].get('contact_trace'),'completed_utc':now(),
    'proxy_is_self_reconstruction_not_GT':True}
   report['milestones']=[r for r in report['milestones'] if r['step']!=target]+[entry]
   # Publish the completed replay and its consume intent together. Interruption
   # during checkpoint pruning must not turn this replay into a skipped result.
   completion=dict(actual_environment_steps=target)
   if a.learning_output is not None:completion['pending_learning_step']=target
   update(**completion)
   # Validate the newest complete set first; retain two recovery sets and all policies.
   groups=complete_sets(checkpoints)
   removed=[]
   for n in groups[:-2]:
    for kind in ['state','replay']:
     path=checkpoints/f'{kind}_{n}.safetensors';path.unlink();removed.append(str(path))
   if removed:update(last_pruned_own_recovery_files=removed)
   if a.learning_output is not None:
    consume(target)
   new_milestones+=1
   if a.max_new_milestones is not None and new_milestones>=a.max_new_milestones and target<p['actual_step_cap']:
    update(state='stopped_between_segments',stop_reason='bounded_milestone_budget',
     completed_new_milestones=new_milestones)
    return
  update(state='completed_step_cap',actual_environment_steps=p['actual_step_cap'])
 except BaseException as exc:
  update(state='failed',error=f'{type(exc).__name__}: {exc}');raise
 finally:
  pass
if __name__=='__main__':main()
