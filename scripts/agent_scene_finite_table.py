"""Independent finite tabletop candidate: preserve hand/object references and vendor."""
from pathlib import Path
import json, time, hashlib, traceback, xml.etree.ElementTree as ET
from types import SimpleNamespace
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw
import mujoco, warp as wp, newton
from agent_cup_reconstruct import ROOT,BASE,CAM,K,NAMES,save,now
from agent_scene_contact_attribution import OUT,PRE,GEO,FRAMES,contacts_snapshot,progress
DEST=OUT/'finite_table_candidate'
# Visible rim annotations: training frame 0 only fits shape; holdouts never fit it.
PIX={0:[[529,441],[574,435],[614,438],[409,491],[379,527],[351,588],[334,650],[331,700],[336,744],[797,568],[824,621],[840,681],[838,735]],10:[[540,435],[591,433],[626,440],[329,650],[330,700],[337,744],[800,570],[825,624],[840,681],[837,737]],60:[[568,431],[625,434],[682,452],[770,510],[823,573],[850,632],[865,691],[862,739],[402,535],[374,591],[359,649],[360,710],[370,753]],124:[[520,418],[574,414],[622,424],[670,445],[752,515],[793,573],[817,632],[825,686],[822,739],[365,533],[340,586],[324,645],[325,706],[334,754]]}

def main():
 DEST.mkdir(exist_ok=True);start=time.monotonic();paths=[Path(__file__),PRE/'episode11_interface_only.parquet',PRE/'reference_checkpoint.npz',BASE/'agent_cup_pose_supported/candidate_poses.npz',GEO/'report.json',BASE/'stereo_sequence/camera_poses.npz'];sha={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};key=hashlib.sha256(json.dumps(sha,sort_keys=True).encode()).hexdigest()
 if (DEST/'report.json').exists():
  prior=json.loads((DEST/'report.json').read_text())
  if prior.get('state')=='failed':save(DEST/f'failure_{int(time.time())}.json',prior)
  if prior.get('source_hash')==key and prior.get('state')=='completed':print('Reusing completed finite candidate');return
 report={'state':'running','source_sha256':sha,'source_hash':key,'started_utc':now(),'training':False,'training_ready':False,'human_or_cup_pose_changed':False,'vendor_changed':False,'results':[]};save(DEST/'report.json',report);progress('无限地面导致圆桌外手部假碰撞；正验证有限桌面候选，杯与手参考保持不变。')
 try:
  cp=np.load(PRE/'reference_checkpoint.npz');A=cp['table_T_world'];W=np.linalg.inv(A);geo=json.loads((GEO/'report.json').read_text());n=np.array(geo['table_plane']['normal_world']);offset=geo['table_plane']['offset_m'];supported=np.load(BASE/'agent_cup_pose_supported/candidate_poses.npz')['world_T_canonical_refined']
  def backproject(f,pix):
   rays=np.c_[pix,np.ones(len(pix))]@np.linalg.inv(K).T@CAM[f,:3,:3].T;c=CAM[f,:3,3];t=-(c@n+offset)/(rays@n);v=c+t[:,None]*rays;return v@A[:3,:3].T+A[:3,3]
  xy=backproject(0,np.array(PIX[0]))[:,:2];sol=least_squares(lambda q:np.linalg.norm(xy-q[:2],axis=1)-q[2],[0,0,.23],loss='soft_l1',f_scale=.003);center=sol.x[:2];radius=sol.x[2];theta=np.linspace(0,2*np.pi,1441);circle=np.c_[center[0]+radius*np.cos(theta),center[1]+radius*np.sin(theta),np.zeros(len(theta))];world=circle@W[:3,:3].T+W[:3,3];fit=[]
  for f,p in PIX.items():
   p=np.array(p);xyf=backproject(f,p)[:,:2];radial=np.linalg.norm(xyf-center,axis=1)-radius;camera=(world-CAM[f,:3,3])@CAM[f,:3,:3];uv=(camera@K.T);uv=uv[:,:2]/uv[:,2:];px=cKDTree(uv).query(p)[0];im=Image.open(BASE/f'calibrated_sequence/frames/{f:06d}.jpg').convert('RGB');dr=ImageDraw.Draw(im);dr.line([tuple(x) for x in uv],fill=(0,255,0),width=3)
   for x,y in p:dr.ellipse((x-4,y-4,x+4,y+4),fill=(255,40,40))
   dr.text((12,12),f'Finite table circle: green; manual visible rim: red; frame {f}',fill='yellow');im.save(DEST/f'rim_overlay_{f:06d}.jpg');fit.append({'frame':f,'split':'train' if f==0 else 'holdout','manual_pixels':p.tolist(),'radial_residual_m':radial.tolist(),'absolute_radial_p95_m':float(np.quantile(abs(radial),.95)),'image_distance_p95_px':float(np.quantile(px,.95))})
  report['table_candidate']={'center_table_xy_m':center.tolist(),'radius_m':float(radius),'thickness_m':.03,'thickness_status':'unmeasured assumption; tabletop top is z=0; thickness extends downward','radius_sensitivity_m':[-.01,0,.01],'circle_prior':'manual visible boundary fit in train frame 0 on calibrated plane; approximate round tabletop','human_annotation_uncertainty':'approximate single-pixel selections, no segmentation GT; independent heldout residuals below','plane_world':{'normal':n.tolist(),'offset_m':offset},'table_T_world':A.tolist(),'fit':fit,'real_floor':'not measured; no infinite plane at tabletop level in finite candidate'};save(DEST/'report.json',report)
  from flash_chord.data.loader import load_reference
  from flash_chord.embodiments.sharpa_hands import SharpaHands
  from flash_chord.scene.builder import build_scene
  from flash_chord.scene.collision import CollisionPolicy
  wp.init()
  with wp.ScopedDevice('cpu'):
   scene=build_scene(SharpaHands(),load_reference(PRE/'episode11_interface_only.parquet'),world_count=1,decompose_objects=False,contact_friction=.5,collision=CollisionPolicy(robot_support_collision=False));model=scene.model;solver=newton.solvers.SolverMuJoCo(model,use_mujoco_cpu=True,use_mujoco_contacts=True,update_data_interval=1,nconmax=4096,njmax=12000,save_to_mjcf=str(DEST/'infinite_scene.xml'));infinite=mujoco.MjModel.from_xml_path(str(DEST/'infinite_scene.xml'));tree=ET.parse(DEST/'infinite_scene.xml');g=tree.find('.//geom[@name="ground_plane_240"]');assert g is not None
   g.set('type','cylinder');g.set('size',f'{radius:.12g} .015');g.set('pos',f'{center[0]:.12g} {center[1]:.12g} -.015');g.set('quat','1 0 0 0');tree.write(DEST/'finite_scene.xml');finite=mujoco.MjModel.from_xml_path(str(DEST/'finite_scene.xml'));assert finite.nq==infinite.nq==solver.mj_model.nq;assert np.array_equal(finite.jnt_qposadr,solver.mj_model.jnt_qposadr);assert [mujoco.mj_id2name(finite,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(finite.njnt)]==[mujoco.mj_id2name(solver.mj_model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(solver.mj_model.njnt)]
   all_steps=0
   for f in FRAMES:
    path=DEST/f'frame_{f:06d}.json'
    if path.exists():
     old=json.loads(path.read_text())
     if old.get('source_hash')==key:report['results'].append(old['summary']);continue
    a=model.state();q=model.joint_q.numpy().copy();q[:scene.layout.num_joint_q]=scene.robot_reference.joint_q[f];P=A@supported[f]
    for oi,binding in enumerate(scene.objects):
     ids=list(binding.root.free_q_ids);q[ids[:3]]=P[oi,:3,3];q[ids[3:]]=Rotation.from_matrix(P[oi,:3,:3]).as_quat()
    a.joint_q.assign(q);a.joint_qd.zero_();newton.eval_fk(model,a.joint_q,a.joint_qd,a);solver._update_mjc_data(solver.mj_data,model,a);control=model.control();target=np.zeros(model.joint_dof_count,np.float32);target[:scene.layout.num_joint_dof]=scene.robot_reference.joint_target[f];control.joint_target_pos.assign(target);solver._apply_mjc_control(model,a,control,solver.mj_data);q_native=solver.mj_data.qpos.copy();ctrl=solver.mj_data.ctrl.copy();rows={};summary={'frame':f,'human_observed':cp['observed_hand'][f].tolist(),'variant':'supported_poses_identical'}
    for label,m in [('infinite',infinite),('finite',finite)]:
     d=mujoco.MjData(m);d.qpos[:]=q_native;d.ctrl[:]=ctrl;m.opt.timestep=.001;mujoco.mj_forward(m,d);proxy=SimpleNamespace(mj_model=m,mj_data=d);before=contacts_snapshot(proxy,'exact_reference_static');initial_positions=d.xpos.copy();peak=0.
     for i in range(20):
      mujoco.mj_step(m,d);all_steps+=1
      for ci in range(d.ncon):
       ff=np.zeros(6);mujoco.mj_contactForce(m,d,ci,ff);peak=max(peak,float(np.linalg.norm(ff[:3])))
     after=contacts_snapshot(proxy,'after_20_native_1ms_steps');rows[label]={'before':before,'after':after,'max_native_contact_force_over_20ms_N':peak,'finite_state':bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()),'max_body_position_change_m':float(np.linalg.norm(d.xpos-initial_positions,axis=1).max())};summary[label]={'static':before['categories'],'after_20ms':after['categories'],'max_native_contact_force_over_20ms_N':peak,'finite_state':rows[label]['finite_state']}
    ground=rows['infinite']['before']['contacts'];ground=[r for r in ground if r['category']=='hand_table' and r['penetration_m']>.001];summary['infinite_hand_table_contacts_over_1mm_outside_radius']={str(delta):{'outside':int(sum(np.linalg.norm(np.array(r['point_table_m'][:2])-center)>radius+delta for r in ground)),'total':len(ground)} for delta in [-.01,0,.01]};save(path,{'source_hash':key,'summary':summary,'native':rows});report['results'].append(summary);save(DEST/'report.json',report);print('FINITE',f,{label:{cat:round(v['max_penetration_m']*1000,3) for cat,v in summary[label]['static'].items()} for label in ['infinite','finite']},flush=True)
  parent=json.loads((OUT/'report.json').read_text());report['unchanged_object_image_validation']=parent['holdout_summary']['supported'];report['unchanged_object_geometry_gates']=parent['existing_supported_gates'];report['validation_invariance']='All cup/hand geometry, camera and poses byte-identical; table extent only changed, so prior raw object depth/mask renders and material/cavity gates are identical. Raw holdout metrics freshly rerun in parent attribution report.';report['gates']={'finite_native_scene_compiles':True,'native_20ms_states_finite':all(r[l]['finite_state'] for r in report['results'] for l in ['infinite','finite']),'heldout_table_rim_p95_under_15px':all(r['image_distance_p95_px']<=15 for r in fit if r['split']=='holdout'),'heldout_table_rim_p95_under_10mm':all(r['absolute_radial_p95_m']<=.010 for r in fit if r['split']=='holdout'),'finite_table_hand_penetration_under_1mm':all(r['finite']['static'].get('hand_table',{}).get('max_penetration_m',0)<=.001 for r in report['results']),'all_collision_penetration_under_1mm':all(v['max_penetration_m']<=.001 for r in report['results'] for v in r['finite']['static'].values()),'cup_geometry_and_image_gate':False,'training_ready':False};report.update(state='completed',completed_utc=now(),elapsed_seconds=time.monotonic()-start,actual_physics_steps_this_run=all_steps,physics_backend='native MuJoCo CPU: both infinite control and finite candidate recompiled from same official Newton scene export; matching joint names/order asserted; finite table not integrated into official Newton builder',automatic_promotion=False);save(DEST/'report.json',report);parent['finite_table_candidate']={'report':str((DEST/'report.json').relative_to(ROOT)),'gates':report['gates'],'radius_m':float(radius),'hand_cup_reference_unchanged':True};save(OUT/'report.json',parent);progress('有限桌面候选实际物理检查完成；桌外假接触与杯杯/手物真实残留已分离，整体门槛仍失败。','completed')
 except Exception:
  report.update(state='failed',error=traceback.format_exc(),elapsed_seconds=time.monotonic()-start);save(DEST/'report.json',report);raise
if __name__=='__main__':main()
