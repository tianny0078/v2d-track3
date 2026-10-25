"""Learning v3: prepare one episode (any of the 20) with FoundationPose v2 object trajectories; no CUDA training here.
Generalised from prepare_track3_ep37_learning.py (frame count/objects from data; no engineering-pass gate)."""
import argparse, copy, hashlib, json, math, shlex, time, xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf
import build_track3_scored_reference as ref
import initialize_sharpa_retarget as ik

ROOT=Path('/home/tianyuan/projects/v2d')
BASE=ROOT/'outputs/track3/advice_restart_20261017'
DEFAULT=None
import os
TREE=os.environ.get('TRACK3_LEARNING_TREE','learning_v3')  # e.g. learning_v4 (objects from build_track3_v2_object_manifests --out)
V3=ROOT/'outputs/track3'/TREE
SOURCES=['prepare_track3_learning_v3.py','build_track3_v2_object_manifests.py','run_track3_learning_v3.py','train_track3_learning_segment.py',
 'track3_weighted_hand_objective.py','train_track3_masked.py','track3_reference_alignment.py',
 'track3_masked_objective.py','track3_masked_contact.py','track3_com_reference.py','track3_wrench_adapter.py',
 'track3_evaluation_harness.py','evaluate_track3_canonical_checkpoint.py','track3_contact_trace.py',
 'build_track3_scored_reference.py','initialize_sharpa_retarget.py','run_track3_fastlane_reference.py']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def repair_hand_tracking_weights(cfg, weight):
 """Apply an explicit ablation to both base terms and curriculum overrides."""
 if not math.isfinite(weight) or weight <= 0:raise ValueError('hand tracking weight must be finite and positive')
 for objective in (cfg['objective'],cfg['env']['objective']['config']):
  for term in ('hand_keypoints','hand_joint_pos'):objective[term]['weight']=weight
 for holder in (cfg,cfg['training']):
  for stage in holder.get('curriculum',{}).get('stages',[]):
   for term in ('hand_keypoints','hand_joint_pos'):stage['objective_weights'][term]=weight
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');q.replace(p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--episode',type=int,required=True);ap.add_argument('--output',type=Path);ap.add_argument('--worlds',type=int,default=2048)
 ap.add_argument('--milestone-steps',type=int,default=1_000_000);ap.add_argument('--milestones',type=int,default=5)
 ap.add_argument('--checkpoint-root',type=Path)
 ap.add_argument('--hand-tracking-weight',type=float,default=None)
 a=ap.parse_args();E=f'episode_{a.episode:06d}'
 a.output=a.output or V3/E/'learning';a.checkpoint_root=a.checkpoint_root or Path('/mnt/nas3/tianyuan/v2d-track3')/TREE/E
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 if a.hand_tracking_weight is not None:
  if not math.isfinite(a.hand_tracking_weight) or a.hand_tracking_weight<=0:raise ValueError('Invalid hand tracking weight')
 assert a.worlds in (1024,2048,4096)
 hands=(ROOT/'outputs/track3'/os.environ['TRACK3_HANDS_ROOT'] if os.environ.get('TRACK3_HANDS_ROOT') else BASE/'dbt-full20-motion-consumer-v1/candidates')/E/'hands_world.npz'  # v5: ORB-world hands
 weights=BASE/f'dbt-full20-motion-weights-v1/{E}/motion_weights.npz'
 objects=V3/E/'objects/objects.json'
 template=BASE/'gpu-throughput-w4096-v1/config.yaml'
 asset_manifest=json.loads(objects.read_text())
 dependencies=[Path(asset_manifest['poses_path'])]
 for item in asset_manifest['objects']:
  dependencies.extend([Path(item['mesh_path']),Path(item['urdf_path'])])
  for mesh in ET.parse(item['urdf_path']).getroot().iter('mesh'):
   path=Path(mesh.attrib['filename'])
   dependencies.append(path if path.is_absolute() else Path(item['urdf_path']).parent/path)
 for side in ref.SIDES:
  dependencies.extend([ik.ASSETS/f'{side}_sharpawave.xml',ref.ASSETS/f'{side}_sharpawave.urdf'])
 binding={str(p.resolve()):sha(p) for p in [hands,weights,objects,template,*dependencies,*[ROOT/'scripts'/n for n in SOURCES]]}
 if a.hand_tracking_weight is not None:binding['parameter:hand_tracking_weight']=a.hand_tracking_weight
 fingerprint=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()
 rp=out/'preparation.json'
 if rp.exists():
  old=json.loads(rp.read_text());assert old['fingerprint']==fingerprint and old['world_count']==a.worlds,'Use a new variant for changed inputs/code'
  if old['state']=='prepared':print('Exact preparation reused');return
 report={'state':'preparing','episode':a.episode,'fingerprint':fingerprint,'binding':binding,'world_count':a.worlds,
  'scope':'experimental actual-contact policy learning; not throughput/far-hand diagnostic','production_pass':False}
 save(rp,report)
 z=ref.npz_dict(hands);wz=ref.npz_dict(weights)
 np.testing.assert_array_equal(z['frame_index'],wz['frame_index']);np.testing.assert_array_equal(z['track_id'],wz['track_id'])
 w=wz['soft_weight'];np.testing.assert_allclose(z['soft_weight'],w,atol=1e-12)
 args=SimpleNamespace(hands=hands,object_manifest=objects,episode_id=a.episode,fps=None,ik_budget_seconds=3600.)
 data=ref.read_inputs(args);T=data['T'];B=len(data['names'])
 reference=out/'reference';reference.mkdir(exist_ok=True)
 eligible=data['present']&np.isfinite(data['joints']).all(axis=(-1,-2))
 filtered=data['joints'].copy()
 for f in range(1,data['T']):
  for h in range(2):
   if eligible[f,h] and eligible[f-1,h] and z['track_id'][f,h]==z['track_id'][f-1,h]:
    filtered[f,h]=w[f,h]*data['joints'][f,h]+(1-w[f,h])*filtered[f-1,h]
 # Original human source remains untouched. Only the explicit robot target is filtered.
 def weighted_robot(d,o,budget):
  models=[ref.kinematics(s,o) for s in ref.SIDES];T=d['T'];q=np.full((T,2,22),np.nan);wr=np.full((T,2,7),np.nan);done=np.zeros((T,2),bool);rows=[]
  cp=o/'weighted_ik.npz'
  if cp.exists():
   c=ref.npz_dict(cp);assert str(c['fingerprint'])==fingerprint;q=c['q'];wr=c['wrist_world'];done=c['done'];rows=json.loads(str(c['rows_json']))
  begin=time.monotonic();original_ls=ik.least_squares
  for f in range(T):
   for h in range(2):
    if done[f,h] or not eligible[f,h]:continue
    m,md,ids,_=models[h];same=bool(f and done[f-1,h] and z['track_id'][f,h]==z['track_id'][f-1,h]);previous=q[f-1,h] if same else None;weight=float(w[f,h])
    def weighted(fun,x0,**kw):
     def residual(x):
      r=np.sqrt(weight)*fun(x)
      return np.r_[r,np.sqrt(1-weight)*(x[6:]-previous)/.25] if previous is not None else r
     return least_squares(residual,x0,**kw)
    ik.least_squares=weighted
    try:r=ik.solve(m,md,ids,filtered[f,h],previous)
    finally:ik.least_squares=original_ls
    q[f,h]=r['qpos'];wr[f,h]=r['wrist_pose'];done[f,h]=True
    rows.append(dict(frame=f,side=ref.SIDES[h],weight=weight,**r['row']))
   ref.save_npz(cp,q=q,wrist_world=wr,done=done,fingerprint=fingerprint,rows_json=json.dumps(rows))
   if f%10==0:print('weighted IK',f,'solved',int(done.sum()),flush=True)
  planned_q=q.copy();planned_w=wr.copy();source=np.full((T,2),-1,np.int64)
  for h in range(2):
   good=np.flatnonzero(done[:,h])
   if not len(good):raise ValueError('No finite candidate for a hand; choose an explicit new trial rather than far-hand 50M')
   nearest=good[np.argmin(abs(np.arange(T)[:,None]-good),axis=1)];planned_q[:,h]=q[nearest,h];planned_w[:,h]=wr[nearest,h];source[:,h]=nearest
   A=d['table_T_world'];planned_w[:,h,:3]=planned_w[:,h,:3]@A[:3,:3].T+A[:3,3]
   rr=Rotation.from_quat(planned_w[:,h,3:],scalar_first=True).as_matrix()
   planned_w[:,h,3:]=Rotation.from_matrix(A[:3,:3]@rr).as_quat(scalar_first=True)
  return planned_q,planned_w,source,done,models,dict(solved=int(done.sum()),eligible=int(eligible.sum()),full_sequence=True,
   elapsed_s=time.monotonic()-begin,soft_weights_applied=True,explicit_retreat_sides=[],position_only=True,
   causal_target_filter='w*current_raw+(1-w)*previous_filtered on same track',contact_optimized=False,
   fingertip_error_p50_p95_m=np.quantile([v for r in rows for v in r['fingertip_error_m']],[.5,.95]).tolist())
 ref.build_robot=weighted_robot
 parquet,stats=ref.build_reference(args,data,reference)
 cp=ref.npz_dict(reference/'reference_checkpoint.npz')
 # Preserve true provenance: every hand is a model estimate, not measured GT.
 cp.update(model_reference_present=eligible,observed_hand=z['observed'],hand_reference_weight=w,
  side_identity_known=z['side_identity_known'],source=z['source'],track_id=z['track_id'],
  force_ground_truth_known=np.zeros((T,2,B),bool),force_reference_is_measured=np.array(False))
 # Geometry proxy is positive-only proximity. It is saved, not enabled as measured-force supervision.
 import trimesh
 from scipy.spatial import cKDTree
 trees=[cKDTree(np.asarray(trimesh.load(r['mesh_path'],force='mesh',process=False).vertices)) for r in data['records']]
 human_table=filtered@data['table_T_world'][:3,:3].T+data['table_T_world'][:3,3]
 distances=np.full((T,2,B,5),np.nan)
 for f in range(T):
  for b in range(B):
   P=data['table_poses'][f,b]
   local=((human_table[f][:,[4,8,12,16,20]]-P[:3,3])@P[:3,:3]).reshape(-1,3)
   fin=np.isfinite(local).all(1);dist=np.full(len(local),np.nan)  # missing hand frames stay NaN
   if fin.any():dist[fin]=trees[b].query(local[fin])[0]
   distances[f,:,b]=dist.reshape(2,5)
 proxy=(np.nan_to_num(np.nanmin(distances,axis=-1),nan=9.)<.012)&eligible[...,None]
 distances=np.nanmin(distances,axis=2)
 cp.update(geometry_proxy_positive=proxy,geometry_proxy_known=proxy,
  geometry_proxy_fingertip_vertex_distance_m=distances,geometry_proxy_is_measured_force=np.array(False))
 ref.save_npz(reference/'reference_checkpoint.npz',**cp)
 table=pq.read_table(parquet);row=table.to_pylist()[0];row['observed_hand']=z['observed'].tolist();row['model_reference_present']=eligible.tolist()
 row['geometry_proxy_known']=proxy.tolist();row['force_reference_is_measured']=False
 row['contact_reference_provenance']='GT force/contact unknown. Positive-only 12mm model-tip/visual-mesh vertex proximity proxy saved separately (upper bound on surface distance), contact objectives off. Model-confidence-weighted hand targets; no object assistance.'
 pq.write_table(pa.Table.from_pylist([row]),parquet)
 cfg=OmegaConf.to_container(OmegaConf.load(template),resolve=True)
 cfg['task'].update(parquet=str(parquet),motion_speed=1.,source_frame_playback=False,motion_start_frame=0,motion_end_frame=-1)
 cfg['scene'].update(world_count=a.worlds,object_scale_min=1.,object_scale_max=1.,decompose_objects=False)
 # Remove the throughput-specific exact far-hand reset implementation completely.
 for key in ['reset']:
  if key in cfg:cfg[key]['_target_']='flash_chord.lifecycle.reset.ReferenceResetConfig'
 cfg['env']['reset'].update(_target_='flash_chord.lifecycle.reset.ReferenceResetConfig',reset_finger_openness=.7,
  always_reset_to_first_frame=False,reset_to_first_frame_probability=.5,voc_decay_steps=0)
 cfg['env']['objective'].update(_target_='track3_weighted_hand_objective.WeightedHandObjectiveSpec',
  reference_known=np.zeros((T,2,B),int).tolist(),body_com_local=data['canonical_COM'].tolist(),com_radius=data['radius'].tolist(),
  hand_reference_weight=w[:,[ref.SIDES.index(s) for s in cfg['env']['objective']['config']['sides']]].tolist(),
  provenance='Model DBT weighted hand reference; GT contact/force unknown; geometry proxy diagnostic only; no external assistance.')
 steps=[(n*a.milestone_steps//a.worlds)*a.worlds for n in range(1,a.milestones+1)]
 cfg['training'].update(world_count=a.worlds,total_environment_steps=steps[-1],resume=False,checkpoint_path=None)
 cfg['training']['optimization']['learning_rate_schedule_updates']=int(steps[-1]//a.worlds*cfg['training']['updates_per_collection'])
 cfg['training']['checkpoint'].update(save_interval_environment_steps=steps[-1],save_replay=True,load_replay_path=None)
 for key in ['curriculum']:
  for holder in [cfg,cfg['training']]:
   cur=holder.get(key)
   if cur:
    cur['thresholds']=[max(a.worlds,(int(v)*steps[-1]//250003456//a.worlds)*a.worlds) for v in cur['thresholds']]
    for stage in cur['stages']:stage['voc_scale']=0.
 cfg['output_dir']=str(a.checkpoint_root.resolve()/'training/checkpoints')
 if a.hand_tracking_weight is not None:repair_hand_tracking_weights(cfg,a.hand_tracking_weight)
 cfg['logging'].update(mode='disabled',upload_checkpoints=False,upload_training_state=False,log_interval=25)
 cfg['voc_scale_override']=0.;cfg['env']['voc_scale']=0.
 cfg['env']['voc']['max_force']=0.;cfg['env']['voc']['max_torque']=0.
 OmegaConf.save(OmegaConf.create(cfg),out/'config.yaml')
 snapshot=out/'producer-source';snapshot.mkdir(exist_ok=True)
 for name in SOURCES:(snapshot/name).write_bytes((ROOT/'scripts'/name).read_bytes())
 report.update(state='prepared',frames=T,ik=stats,object_manifest=str(objects),
  reference_checkpoint=str(reference/'reference_checkpoint.npz'),parquet=str(parquet),config=str(out/'config.yaml'),
  checkpoint_root=str(a.checkpoint_root.resolve()),milestones=steps,actual_step_cap=steps[-1],requested_step_cap=steps[-1],
  reference_mode='ground_truth_unknown',contact_reward_enabled=False,force_ground_truth_known_cells=0,
  positive_geometry_proxy_cells=int(proxy.sum()),geometry_proxy_threshold_m=.012,
  human_tip_vertex_min_m=float(np.nanmin(distances)) if np.isfinite(distances).any() else None,human_tip_vertex_median_m=float(np.nanmedian(distances)) if np.isfinite(distances).any() else None,
  geometry_proxy_method='nearest visual vertex: upper bound on surface distance; positive-only proximity, not measured force',
  far_hand_reference=False,object_assistance=False,physical_gpu=None,training_started=False,
  runtime_source_sha256={name:sha(snapshot/name) for name in SOURCES},
  input_artifact_sha256={str(p):sha(p) for p in [out/'config.yaml',parquet,reference/'reference_checkpoint.npz']})
 save(rp,report);print(json.dumps({k:report[k] for k in ['state','frames','ik','positive_geometry_proxy_cells','human_tip_vertex_min_m','actual_step_cap']},indent=2))
if __name__=='__main__':main()
