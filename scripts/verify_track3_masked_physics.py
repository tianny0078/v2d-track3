"""Two-world native GPU RLEnv integration diagnostic, no learner or assistance."""
from pathlib import Path
import datetime,json,hashlib,traceback
import numpy as np
import warp as wp
from omegaconf import OmegaConf
from hydra.utils import instantiate
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/track3/episode_000011/masked_physics_integration'
PRE=ROOT/'outputs/track3/episode_000011/flashchord_reference_preflight'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 report=dict(started_utc=now(),state='running',training_ready=False,production_integrated=False,policy_training=False)
 try:
  from flash_chord.data.loader import load_reference
  from flash_chord.scene.builder import build_scene
  from flash_chord.envs.rl import RLEnv
  from track3_masked_objective import MaskedTrackingObjective
  cfg=OmegaConf.load(ROOT/'outputs/track3/episode_000011/baseline_probe/composed_training.yaml')
  ref=load_reference(PRE/'episode11_interface_only.parquet');z=np.load(PRE/'reference_checkpoint.npz')
  # GT validity is absent for ALL source slots: no proximity labels promoted.
  assert not z['contact_ground_truth_known'].any()
  sides=tuple(ref.sides);known=np.zeros((ref.num_frames,len(sides),2),np.int32)
  cfg.env.sim.njmax=4096;cfg.env.sim.nconmax=2048
  for name in ('object_keypoints','hand_keypoints','hand_joint_pos','contact_wrench_support'):
   cfg.env.objective[name].weight=1.0
  cfg.env.objective.missed_contact.weight=-1.0;cfg.env.objective.unintended_contact.weight=-1.0
  cfg.env.objective=OmegaConf.create(dict(_target_='track3_masked_objective.MaskedObjectiveSpec',config=OmegaConf.to_container(cfg.env.objective,resolve=True),reference_known=known.tolist(),provenance='reference_checkpoint.npz contact_ground_truth_known: all slots unknown; no GT contact supervision',body_com_local=z['canonical_COM'].tolist(),com_radius=z['object_bounding_radius_COM'].tolist()))
  cfg.scene.world_count=2;cfg.env.auto_reset=False
  cfg.task.motion_speed=1.0 # Mask is indexed on original 20Hz source frames; no implicit retiming.
  OmegaConf.save(cfg,OUT/'composed_training.yaml')
  with wp.ScopedDevice('cuda:0'):
   scene=build_scene(instantiate(cfg.embodiment),ref,world_count=2,decompose_objects=False,contact_friction=.5,collision=instantiate(cfg.collision))
   actual=scene.model.body_com.numpy()[[b.root.body_id for b in scene.objects]]
   np.testing.assert_allclose(actual,z['canonical_COM'],atol=1e-6)
   env=RLEnv(scene,ref,config=instantiate(cfg.env,_convert_='all'),device='cuda:0')
   assert isinstance(env.objective,MaskedTrackingObjective)
   frames=wp.array([0,68],dtype=wp.int32);env.reset(reset_frame=frames)
   rows=[];snapshots={}
   for i in range(3):
    evaluated=env.timestep.numpy().copy()
    obs,reward,done,trunc=env.step()
    terms=env.objective.terms.numpy().reshape(2,-1).copy()
    assert np.isfinite(obs.numpy()).all() and np.isfinite(reward.numpy()).all() and np.isfinite(terms).all()
    assert np.any(np.abs(reward.numpy())>1e-8), 'nonzero tracking weights must reach reward'
    np.testing.assert_array_equal(env.objective.known.numpy(),0)
    np.testing.assert_array_equal(terms[:,[3,4,5,10]],0)
    raw=env.contact.contact_pos_o.numpy().copy()
    env.objective.evaluate(env.state_0,wp.array(evaluated,dtype=wp.int32),env.terminated)
    np.testing.assert_array_equal(env.contact.contact_pos_o.numpy(),raw)
    rows.append(dict(control_step=i,evaluated_frames=evaluated.tolist(),reward=reward.numpy().tolist(),done=done.numpy().tolist(),active_contact_slots=int(env.contact.contact_active.numpy().sum()),max_contact_force_N=float(np.linalg.norm(env.contact.contact_force_w.numpy(),axis=-1).max()),contact_terms=terms[:,[3,4,5,10]].tolist()))
   # Independent selected reset must leave every state coordinate in world 1 unchanged.
   q=env.state_0.joint_q.numpy().copy();qd=env.state_0.joint_qd.numpy().copy();t=env.timestep.numpy().copy()
   env.reset(reset_frame=wp.array([12,100],dtype=wp.int32),reset_mask=wp.array([1,0],dtype=wp.int32))
   nq=env.model.joint_coord_count//2;nv=env.model.joint_dof_count//2
   np.testing.assert_array_equal(env.state_0.joint_q.numpy()[nq:],q[nq:]);np.testing.assert_array_equal(env.state_0.joint_qd.numpy()[nv:],qd[nv:]);assert env.timestep.numpy()[1]==t[1] and env.timestep.numpy()[0]==12
   # Counterfactual temporal mask test isolated from actual GT source and saved config.
   counter=np.zeros_like(known);counter[12,0,0]=1;counter[int(t[1]),1,1]=1
   env.objective.bind_known(counter,'counterfactual batching/reset diagnostic only; not dataset annotation')
   env.command.gather(env.timestep);env.objective.evaluate(env.state_0,env.timestep,env.terminated)
   expected=counter[env.timestep.numpy()].ravel();np.testing.assert_array_equal(env.objective.known.numpy(),expected)
   env.objective.bind_known(known,'restored actual all-unknown GT source');env.objective.evaluate(env.state_0,env.timestep,env.terminated)
   np.savez_compressed(OUT/'checkpoint.npz',joint_q=env.state_0.joint_q.numpy(),joint_qd=env.state_0.joint_qd.numpy(),body_q=env.state_0.body_q.numpy(),timestep=env.timestep.numpy(),reference_known=known,body_com_local=actual,com_radius=z['object_bounding_radius_COM'],terms=env.objective.terms.numpy(),contact_pos_link=env.contact.contact_pos_o.numpy(),contact_pos_com=env.objective.contact_pos_com.numpy())
   report.update(state='completed',world_count=2,control_steps=3,solver_calls=3*env.config.sim.substeps,world_physics_steps=2*3*env.config.sim.substeps,sim_dt=env.sim_dt,rows=rows,independent_reset_pass=True,counterfactual_temporal_mask_gather_pass=True,actual_GT_known_cells=0,configured_spec_called_by_RLEnv=True,native_GPU_contact_update=True,raw_contact_preservation_pass=True,COM_model_matches_reference=True,radius_source='reference_checkpoint.object_bounding_radius_COM: max mesh vertex distance from canonical URDF COM',runtime_environment_integrated=True,production_integrated=False,limitations=['GPU RLEnv diagnostic only; learner entry not yet exercised with project config','source trajectory geometry gates remain failed; no training readiness','checkpoint is an inspection snapshot, resume command deterministically rebuilds diagnostic; not exact solver restore','all actual GT contacts unknown; mixed-mask checks are isolated counterfactuals'])
 except Exception:
  report.update(state='failed',error=traceback.format_exc());raise
 finally:
  report['completed_utc']=now();report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'scripts/track3_masked_objective.py',ROOT/'scripts/track3_com_reference.py',PRE/'reference_checkpoint.npz',OUT/'composed_training.yaml'] if p.exists()}
  (OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
