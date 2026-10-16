"""Bounded pairwise collision attribution at observed/planned keyframes; no training."""
from pathlib import Path
import json,time,hashlib,traceback
import numpy as np
import mujoco,warp as wp,newton
from scipy.spatial.transform import Rotation
from agent_cup_reconstruct import ROOT,BASE,NAMES,HOLDOUT,save,now,CAM,K
from agent_cup_pose_refine import full_metrics,aggregate,samples,overlap,comparison
from agent_cup_support_refine import clearance
from agent_cup_support_audit import cavity_diagnostic

OUT=BASE/'scene_contact_attribution';PRE=BASE/'flashchord_reference_preflight';GEO=BASE/'agent_cup_reconstruction';FRAMES=[0,28,68,86,113,136]

def progress(detail,state='running'):
    p=ROOT/'notes/track3-live-progress.json'
    if not p.exists():return
    r=json.loads(p.read_text());r['scene_contact_attribution']={'state':state,'detail':detail,'report':str((OUT/'report.json').relative_to(ROOT)),'checked_at':now()}
    save(p,r)

def role(name):
    if name=='world':return 'table'
    for n in NAMES:
        if n in name:return n
    if 'left' in name:return 'left_hand'
    if 'right' in name:return 'right_hand'
    return 'other'

def category(a,b):
    r=[role(a),role(b)];hands=[x.endswith('_hand') for x in r];objects=[x in NAMES for x in r]
    if 'table' in r:return 'hand_table' if any(hands) else 'object_table' if any(objects) else 'other_table'
    if all(hands):return 'hand_self' if r[0]==r[1] else 'hand_hand'
    if all(objects):return 'object_object'
    if any(hands) and any(objects):return 'hand_object'
    return 'other'

def contacts_snapshot(solver,phase):
    m=solver.mj_model;d=solver.mj_data;rows=[]
    for ci,c in enumerate(d.contact[:d.ncon]):
        bs=[int(m.geom_bodyid[g]) for g in [c.geom1,c.geom2]];bn=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b) or 'world' for b in bs];gn=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(g)) for g in [c.geom1,c.geom2]];f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f)
        rows.append({'contact_id':ci,'category':category(*bn),'bodies':bn,'body_ids':bs,'geoms':gn,'geom_ids':[int(c.geom1),int(c.geom2)],'geom_types':[int(m.geom_type[g]) for g in [c.geom1,c.geom2]],'distance_m':float(c.dist),'penetration_m':max(0.,float(-c.dist)),'point_table_m':np.asarray(c.pos).tolist(),'normal_geom1_to_geom2_table':np.asarray(c.frame).reshape(3,3)[0].tolist(),'native_force_norm_N':float(np.linalg.norm(f[:3]))})
    rows.sort(key=lambda x:x['distance_m']);groups={}
    for cat in sorted(set(x['category'] for x in rows)):
        rr=[x for x in rows if x['category']==cat];groups[cat]={'count':len(rr),'penetrating_over_1mm':sum(x['penetration_m']>.001 for x in rr),'max_penetration_m':max(x['penetration_m'] for x in rr),'max_native_force_N':max(x['native_force_norm_N'] for x in rr),'worst_pairs':rr[:5]}
    return {'phase':phase,'native_contact_count':len(rows),'categories':groups,'contacts':rows}

def geom_min_z(m,d):
    out={}
    for gi in range(m.ngeom):
        bi=int(m.geom_bodyid[gi]);bn=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,bi) or 'world';rr=role(bn)
        if rr=='table':continue
        typ=int(m.geom_type[gi]);R=d.geom_xmat[gi].reshape(3,3);p=d.geom_xpos[gi]
        if typ==int(mujoco.mjtGeom.mjGEOM_MESH):
            mid=int(m.geom_dataid[gi]);v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]];z=(v@R.T+p)[:,2].min()
        elif typ==int(mujoco.mjtGeom.mjGEOM_SPHERE):z=p[2]-m.geom_size[gi,0]
        elif typ in [int(mujoco.mjtGeom.mjGEOM_CAPSULE),int(mujoco.mjtGeom.mjGEOM_CYLINDER)]:z=p[2]-abs(R[2,2])*m.geom_size[gi,1]-m.geom_size[gi,0]
        else:z=p[2]-abs(R[2])@m.geom_size[gi]
        if rr not in out or z<out[rr]['min_z_m']:out[rr]={'min_z_m':float(z),'body':bn,'geom':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,gi)}
    return out

def main():
    OUT.mkdir(exist_ok=True);start=time.monotonic();source_paths=[Path(__file__),PRE/'episode11_interface_only.parquet',PRE/'reference_checkpoint.npz',GEO/'candidate_poses.npz',GEO/'report.json',BASE/'agent_cup_pose_supported/candidate_poses.npz',BASE/'agent_cup_pose_supported/report.json',BASE/'sharpa_keypoint_initializer_v2/checkpoint.npz',BASE/'hand_depth_tradeoff/alpha_050/hands_world.npz'];sha={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths};key=hashlib.sha256(json.dumps(sha,sort_keys=True).encode()).hexdigest();old=json.loads((OUT/'report.json').read_text()) if (OUT/'report.json').exists() else {}
    if old.get('source_hash')==key and old.get('state')=='completed':print('Matching completed attribution reused');return
    report={'state':'running','started_utc':now(),'source_hash':key,'source_sha256':sha,'budget_seconds':1140,'training':False,'training_ready':False,'new_pose_or_geometry_candidate_created':False,'human_observations_modified':False,'contact_unknown_semantics_changed':False,'frames':FRAMES,'collision_policy':'Baseline recipe: robot_scope all, hand_self true, robot_object true, robot_support false, ground true','friction_assumed':.5,'physics_dt_s':.001,'results':[]};save(OUT/'report.json',report);progress('逐对定位手物、杯杯、桌面与自碰撞；比较initial和supported，未修改观测。')
    try:
        from flash_chord.data.loader import load_reference
        from flash_chord.embodiments.sharpa_hands import SharpaHands
        from flash_chord.scene.builder import build_scene
        from flash_chord.scene.collision import CollisionPolicy
        ref=load_reference(PRE/'episode11_interface_only.parquet');cp=np.load(PRE/'reference_checkpoint.npz');A=cp['table_T_world'];initial=np.load(GEO/'candidate_poses.npz')['world_T_canonical_initial'];supported=np.load(BASE/'agent_cup_pose_supported/candidate_poses.npz')['world_T_canonical_refined'];geo=json.loads((GEO/'report.json').read_text());shapes=[geo['objects'][n]['shape'] for n in NAMES];clouds=[samples(s,.003) for s in shapes];n=np.array(geo['table_plane']['normal_world']);offset=geo['table_plane']['offset_m'];hands=np.load(BASE/'hand_depth_tradeoff/alpha_050/hands_world.npz')
        wp.init()
        with wp.ScopedDevice('cpu'):
            scene=build_scene(SharpaHands(),ref,world_count=1,decompose_objects=False,contact_friction=.5,collision=CollisionPolicy(robot_support_collision=False));model=scene.model;solver=newton.solvers.SolverMuJoCo(model,use_mujoco_cpu=True,use_mujoco_contacts=True,update_data_interval=1,nconmax=4096,njmax=12000);contacts=model.contacts()
            for label,poses in [('initial',initial),('supported',supported)]:
                for frame in FRAMES:
                    artifact=OUT/f'{label}_frame_{frame:06d}.json'
                    if artifact.exists():
                        r=json.loads(artifact.read_text())
                        if r.get('source_hash')==key:report['results'].append({k:v for k,v in r.items() if k not in ['before_step','after_step']});continue
                    a=model.state();b=model.state();q=model.joint_q.numpy().copy();q[:scene.layout.num_joint_q]=scene.robot_reference.joint_q[frame];P=A@poses[frame]
                    for oi,binding in enumerate(scene.objects):
                        ids=list(binding.root.free_q_ids);q[ids[:3]]=P[oi,:3,3];q[ids[3:]]=Rotation.from_matrix(P[oi,:3,:3]).as_quat()
                    a.joint_q.assign(q);a.joint_qd.zero_();newton.eval_fk(model,a.joint_q,a.joint_qd,a);solver._update_mjc_data(solver.mj_data,model,a);mujoco.mj_forward(solver.mj_model,solver.mj_data);before=contacts_snapshot(solver,'static_mj_forward_at_exact_reference');bounds=geom_min_z(solver.mj_model,solver.mj_data)
                    control=model.control();target=np.zeros(model.joint_dof_count,np.float32);target[:scene.layout.num_joint_dof]=scene.robot_reference.joint_target[frame];control.joint_target_pos.assign(target);a.clear_forces();solver.step(a,b,control,contacts,.001);after=contacts_snapshot(solver,'native_contacts_from_one_1ms_step')
                    human=[]
                    for hi,side in enumerate(['left','right']):
                        observed=bool(hands['present'][frame,hi]);v=hands['vertices_world'][frame,hi]@A[:3,:3].T+A[:3,3] if observed else None;human.append({'side':side,'observed':observed,'robot_only_planned_hold':not observed,'human_mesh_min_table_z_m':float(v[:,2].min()) if observed else None,'human_mesh_fraction_below_plane':float((v[:,2]<0).mean()) if observed else None})
                    r={'source_hash':key,'pose_variant':label,'frame':frame,'heldout':frame in HOLDOUT,'human':human,'robot_and_cup_bounds':bounds,'analytic_table_clearance_m':[clearance(poses[frame,oi],shapes[oi],n,offset) for oi in range(2)],'analytic_cup_material':overlap(poses[frame],shapes,.003,clouds),'cavity':cavity_diagnostic(poses[frame],shapes,.003) if frame>=68 else None,'before_step':before,'after_step':after,'static_categories':before['categories'],'step_categories':after['categories'],'artifact':str(artifact.relative_to(ROOT))};save(artifact,r);report['results'].append({k:v for k,v in r.items() if k not in ['before_step','after_step']});save(OUT/'report.json',report);print(label,frame,{k:round(v['max_penetration_m']*1000,3) for k,v in before['categories'].items()},'human',human,flush=True)
                    progress(f'已归因 {len(report["results"])}/12 个关键帧场景；initial/supported并列，手与未知状态保持原样。')
        report.update(state='attribution_completed',elapsed_seconds=time.monotonic()-start);save(OUT/'report.json',report)
        # Re-evaluate unchanged candidates against the same raw holdout observation gates.
        evaluation=[]
        for frame in sorted(HOLDOUT):
            path=OUT/f'image_holdout_{frame:06d}.json'
            if path.exists() and json.loads(path.read_text()).get('source_hash')==key:evaluation.append(json.loads(path.read_text()));continue
            rows={label:full_metrics(frame,P[frame],shapes,.003)[0] for label,P in [('initial',initial),('supported',supported)]};r={'source_hash':key,'frame':frame,**rows};save(path,r);evaluation.append(r)
        report['holdout_summary']={label:aggregate([item for r in evaluation for item in r[label]]) for label in ['initial','supported']};report['heldout_directly_optimized']=False;report['existing_supported_gates']=json.loads((BASE/'agent_cup_pose_supported/report.json').read_text())['gates'];report.update(state='completed',completed_utc=now(),elapsed_seconds=time.monotonic()-start,physics_steps_total=12,automatic_promotion=False);save(OUT/'report.json',report);progress('关键帧逐对碰撞归因与原留出图像门槛复核完成；结果待审查，仍不训练就绪。','completed')
    except Exception:
        report.update(state='failed',error=traceback.format_exc(),elapsed_seconds=time.monotonic()-start);save(OUT/f'failure_{int(time.time())}.json',report);save(OUT/'report.json',report);progress('碰撞归因遇到接口错误，失败证据已保留；正在审查。','failed');raise

if __name__=='__main__':main()
