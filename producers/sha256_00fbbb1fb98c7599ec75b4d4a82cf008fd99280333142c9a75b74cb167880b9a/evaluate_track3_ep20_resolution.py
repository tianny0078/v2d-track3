"""Frozen actor CPU probe; export failures before scoring; explicit substep ablation."""
import sys,json,hashlib,datetime,traceback,argparse,copy
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--substeps',type=int,default=25);p.add_argument('--zero-action',action='store_true');a=p.parse_args()
b=Path('outputs/track3/episode_000020/training').resolve();o=Path('outputs/track3/scored_baseline_20261017/ep20-resolution-v1');stem=f'ep20_cpu_substeps{a.substeps}'+('_zero' if a.zero_action else '');sys.path.insert(0,str(b/'producer-source'))
import track3_evaluation_harness as h
original=h.checkpoint_config
actual={}
def config(metadata):
 c=copy.deepcopy(original(metadata));c['training']['device']='cpu:0';actual['original_substeps']=c['env']['sim']['substeps'];c['env']['sim']['substeps']=a.substeps;return c
h.checkpoint_config=config
from flash_chord.envs.rl import RLEnv
init=RLEnv.__init__
def cpu_init(self,*args,**kwargs):
 kwargs['device']='cpu';return init(self,*args,**kwargs)
RLEnv.__init__=cpu_init;RLEnv.capture_step=lambda self:None
cp=b/'checkpoints/policy_60000.safetensors'
report=dict(checkpoint=str(cp),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),device='cpu',capture=False,policy_complete=False,reference_promoted=False,substeps=a.substeps,object_assistance=False)
record=h.record_object_pose_rollout
def capture(*args,**kwargs):
 env=args[0]; ref=kwargs['reference']; ids=env.command.body_ids_w.numpy().reshape(-1).astype(int)
 pre=env.state_0.body_q.numpy()[ids].copy()
 expected=np.concatenate([ref.object_body_pos_w()[0],ref.object_body_quat_w()[0][:,[1,2,3,0]]],axis=-1)
 reset_error=np.linalg.norm(pre[:,:3]-expected[:,:3],axis=-1)
 (o/(stem+'_reset.json')).write_text(json.dumps(dict(body_ids=ids.tolist(),body_names=list(ref.object_body_names()),reset_pose_xyzw=pre.tolist(),reference_pose_xyzw=expected.tolist(),reset_position_errors_m=reset_error.tolist(),object_root_offsets=env.scene.object_root_position_offsets_w.tolist(),sample_timing='rollout records after physics step; reset captured before first policy action'),indent=2))
 if a.zero_action:
  import jax.numpy as jnp
  call=list(args); original_action=call[2]
  def zero_action(*policy_args,**policy_kwargs):
   result,key=original_action(*policy_args,**policy_kwargs)
   return jnp.zeros_like(result),key
  call[2]=zero_action;args=tuple(call)
 r=record(*args,**kwargs)
 # Upstream replaces invalid-world achieved poses with zero; preserve that fact and flags.
 np.savez_compressed(o/(stem+'_checkpoint.npz'),achieved_pose_w=r.achieved_pose_w,reference_pose_w=r.reference_pose_w,tracking_error=r.tracking_error,non_finite_worlds=np.array(r.non_finite_worlds),completion_step=r.completion.completion_step)
 provenance=h.evaluation_provenance(checkpoint=cp,parquet=str(args[0].config.task.parquet) if hasattr(args[0].config,'task') else str(b/'config.yaml'),eval_script=Path(__file__))
 h.write_object_trajectories_parquet(o/(stem+'.parquet'),r,provenance=provenance)
 report.update(rollout_steps=r.achieved_pose_w.shape[0],non_finite_worlds=list(r.non_finite_worlds),trajectory_exported=True,invalid_world_poses_zero_filled_by_upstream=bool(r.non_finite_worlds),tracking_error_names=list(r.tracking_error_names),nonfinite_error_frame_ids=np.flatnonzero(~np.isfinite(r.tracking_error).all(axis=(1,2))).tolist())
 if len(r.non_finite_worlds)==r.achieved_pose_w.shape[1]:raise RuntimeError('all_worlds_nonfinite: trajectory saved, MPPE undefined; invalid zero-filled poses excluded')
 return r
h.record_object_pose_rollout=capture
try:
 r=h.evaluate_checkpoint(cp,world_count=1,start_frame=0,reset_mode='explicit',object_trajectories_output=o/(stem+'.parquet'),eval_script=Path(__file__));h.write_evaluation_report(o/(stem+'_evaluation.json'),r);report['policy_complete']=True
except Exception as e:
 report['error_type']=type(e).__name__;report['error']=str(e);traceback.print_exc()
finally:
 report.update(actual);report['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(o/(stem+'_attempt.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
sys.exit(0 if report['policy_complete'] else 2)
