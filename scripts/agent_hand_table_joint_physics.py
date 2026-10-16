"""Actual native finite-table smoke for only locally accepted rigid hand candidates."""
from pathlib import Path
import json,time,hashlib,traceback
from types import SimpleNamespace
import numpy as np,warp as wp,newton,mujoco
from scipy.spatial.transform import Rotation
from agent_cup_reconstruct import ROOT,BASE,save,now
from agent_scene_contact_attribution import contacts_snapshot
from agent_hand_table_joint_fit import progress
PARENT=BASE/'hand_table_joint_fit/v2_keypoint_balanced';OUT=PARENT/'physics';PRE=BASE/'flashchord_reference_preflight'
def main():
 OUT.mkdir(exist_ok=True);start=time.monotonic();paths=[Path(__file__),PARENT/'hands_world_candidate.npz',PARENT/'report.json',PRE/'episode11_interface_only.parquet',BASE/'scene_contact_attribution/finite_table_candidate/finite_scene.xml'];sha={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};key=hashlib.sha256(json.dumps(sha,sort_keys=True).encode()).hexdigest();r={'state':'running','source_hash':key,'source_sha256':sha,'training':False,'training_ready':False,'frames':[28,86],'scope':'Only locally accepted keyframes; global candidate remains rejected. Exact rigid wrist/FK transfer, unchanged local finger angles. Native finite tabletop candidate, not validated scene reconstruction.','results':[]};save(OUT/'report.json',r);progress('v2全段仍拒绝；仅28/86局部门槛通过，正验证刚体delta匹配Sharpa FK并做实际finite-table native接触对照。')
 try:
  report=json.loads((PARENT/'report.json').read_text());candidate=np.load(PARENT/'hands_world_candidate.npz');A=np.load(PRE/'reference_checkpoint.npz')['table_T_world'];poses=np.load(BASE/'agent_cup_pose_supported/candidate_poses.npz')['world_T_canonical_refined']
  for f in r['frames']:
   e=next(x for x in report['evaluation'] if x['frame']==f);c=e['candidate'];b=e['alpha_050'];assert c['finite_table_max_penetration_m']<=.001 and b['mask_iou']-c['mask_iou']<=.02 and c['absolute_depth_p50_p95_m'][0]<=.01 and c['visible_keypoint_error_p50_p95_px'][0]-b['visible_keypoint_error_p50_p95_px'][0]<=2
  from flash_chord.data.loader import load_reference
  from flash_chord.embodiments.sharpa_hands import SharpaHands
  from flash_chord.scene.builder import build_scene
  from flash_chord.scene.collision import CollisionPolicy
  wp.init();archive=[]
  with wp.ScopedDevice('cpu'):
   scene=build_scene(SharpaHands(),load_reference(PRE/'episode11_interface_only.parquet'),world_count=1,decompose_objects=False,contact_friction=.5,collision=CollisionPolicy(robot_support_collision=False));model=scene.model;solver=newton.solvers.SolverMuJoCo(model,use_mujoco_cpu=True,use_mujoco_contacts=True,update_data_interval=1,nconmax=4096,njmax=12000);native=mujoco.MjModel.from_xml_path(str(BASE/'scene_contact_attribution/finite_table_candidate/finite_scene.xml'));assert [mujoco.mj_id2name(native,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(native.njnt)]==[mujoco.mj_id2name(solver.mj_model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(solver.mj_model.njnt)];native.opt.timestep=.001;labels=list(model.joint_label);qs=model.joint_q_start.numpy();ds=model.joint_qd_start.numpy();leftidx=[labels.index('left_wrist_'+x) for x in ['x','y','z','rotvec']];leftbody=[i for i in range(native.nbody) if (mujoco.mj_id2name(native,mujoco.mjtObj.mjOBJ_BODY,i) or '').startswith('left_sharpa')]
   for f in r['frames']:
    q=model.joint_q.numpy().copy();q[:scene.layout.num_joint_q]=scene.robot_reference.joint_q[f];T=A@poses[f]
    for oi,binding in enumerate(scene.objects):
     ids=list(binding.root.free_q_ids);q[ids[:3]]=T[oi,:3,3];q[ids[3:]]=Rotation.from_matrix(T[oi,:3,:3]).as_quat()
    target=np.zeros(model.joint_dof_count,np.float32);target[:scene.layout.num_joint_dof]=scene.robot_reference.joint_target[f];Q=A@candidate['candidate_delta_world'][f,0]@np.linalg.inv(A);pos=np.array([q[qs[j]] for j in leftidx[:3]]);qrot=q[qs[leftidx[3]]:qs[leftidx[3]]+4];newpos=Q[:3,:3]@pos+Q[:3,3];newrot=Q[:3,:3]@Rotation.from_quat(qrot).as_matrix();qc=q.copy();tc=target.copy()
    for k,j in enumerate(leftidx[:3]):qc[qs[j]]=newpos[k];tc[ds[j]]=newpos[k]
    qc[qs[leftidx[3]]:qs[leftidx[3]]+4]=Rotation.from_matrix(newrot).as_quat();tc[ds[leftidx[3]]:ds[leftidx[3]]+3]=Rotation.from_matrix(newrot).as_rotvec();pair={};kin={}
    for label,qq,tt in [('alpha_050',q,target),('candidate',qc,tc)]:
     a=model.state();a.joint_q.assign(qq);a.joint_qd.zero_();newton.eval_fk(model,a.joint_q,a.joint_qd,a);solver._update_mjc_data(solver.mj_data,model,a);control=model.control();control.joint_target_pos.assign(tt);solver._apply_mjc_control(model,a,control,solver.mj_data);d=mujoco.MjData(native);d.qpos[:]=solver.mj_data.qpos.copy();d.ctrl[:]=solver.mj_data.ctrl.copy();mujoco.mj_forward(native,d);kin[label]=(d.xpos[leftbody].copy(),d.xmat[leftbody].copy().reshape(-1,3,3));probe=SimpleNamespace(mj_model=native,mj_data=d);before=contacts_snapshot(probe,'matched_reference_static');archive.append({'frame':f,'variant':label,'q_newton':qq.copy(),'target_newton':tt.copy(),'qpos_native':d.qpos.copy(),'ctrl_native':d.ctrl.copy()});peak=0.
     for step in range(20):
      mujoco.mj_step(native,d)
      for ci in range(d.ncon):
       ff=np.zeros(6);mujoco.mj_contactForce(native,d,ci,ff);peak=max(peak,float(np.linalg.norm(ff[:3])))
     pair[label]={'before':before,'after':contacts_snapshot(probe,'20_native_1ms_steps_constant_initial_matching_control'),'peak_native_force_N':peak,'finite_state':bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())}
    poserr=np.linalg.norm(kin['candidate'][0]-(kin['alpha_050'][0]@Q[:3,:3].T+Q[:3,3]),axis=1);roterr=np.max(abs(kin['candidate'][1]-Q[:3,:3][None]@kin['alpha_050'][1]));assert poserr.max()<1e-6 and roterr<1e-5,(poserr.max(),roterr);row={'frame':f,'rigid_FK_position_max_error_m':float(poserr.max()),'rigid_FK_orientation_matrix_max_error':float(roterr),'world_delta':candidate['candidate_delta_world'][f,0].tolist(),'table_delta':Q.tolist(),'variants':pair};save(OUT/f'frame_{f:06d}.json',row);r['results'].append({'frame':f,'rigid_FK_position_max_error_m':float(poserr.max()),'rigid_FK_orientation_matrix_max_error':float(roterr),'variants':{k:{'static_categories':v['before']['categories'],'peak_native_force_N':v['peak_native_force_N'],'finite_state':v['finite_state']} for k,v in pair.items()}});save(OUT/'report.json',r);print('PHYSICS',f,{k:{cat:round(val['max_penetration_m']*1000,3) for cat,val in v['before']['categories'].items()} for k,v in pair.items()},flush=True)
  np.savez_compressed(OUT/'matched_robot_keyframe_reference.npz',frames=np.array([x['frame'] for x in archive]),variant=np.array([x['variant'] for x in archive]),q_newton=np.stack([x['q_newton'] for x in archive]),joint_target_newton=np.stack([x['target_newton'] for x in archive]),qpos_native=np.stack([x['qpos_native'] for x in archive]),ctrl_native=np.stack([x['ctrl_native'] for x in archive]));r.update(state='completed',completed_utc=now(),elapsed_seconds=time.monotonic()-start,actual_physics_steps=80,gates={'rigid_fk_matches_candidate':True,'native_states_finite':all(v['finite_state'] for x in r['results'] for v in x['variants'].values()),'all_candidate_penetrations_under_1mm':all(z['max_penetration_m']<=.001 for x in r['results'] for z in x['variants']['candidate']['static_categories'].values()),'training_ready':False},limitations=['Constant initial native control for20ms; not an official policy rollout or long tracking test.','Robot finger joints unchanged because candidate is whole-hand rigid SE3; all physical hand body FK is checked under same transform.','Contact forces are numerical model reactions, not measured human forces.','Cup poses/geometry, right hand and unknown contact semantics unchanged; whole-sequence candidate still rejected.']);save(OUT/'report.json',r);progress('28/86匹配Sharpa FK与80个真实native步已完成；局部接触报告已保存，整体候选仍拒绝。','completed')
 except Exception:
  r.update(state='failed',error=traceback.format_exc(),elapsed_seconds=time.monotonic()-start);save(OUT/'report.json',r);raise
if __name__=='__main__':main()
