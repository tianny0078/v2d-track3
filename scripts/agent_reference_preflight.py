"""Episode 11 official ManoSharpa loader/scene preflight, explicitly not training-ready."""
from pathlib import Path
import json,time,hashlib,traceback,xml.etree.ElementTree as ET
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import mujoco,trimesh
from scipy.spatial.transform import Rotation
from agent_cup_reconstruct import ROOT,BASE,NAMES,save,now

OUT=BASE/'flashchord_reference_preflight';IK=BASE/'sharpa_keypoint_initializer_v2';GEO=BASE/'agent_cup_reconstruction';COL=BASE/'agent_cup_collision_smoke'
OFFICIAL=ROOT/'vendor/video_to_data/robotic_grounding/flash_chord/src/flash_chord'
SIDES=['left','right'];BUDGET=540

def quat(R):return Rotation.from_matrix(R.reshape(-1,3,3)).as_quat(scalar_first=True).reshape(R.shape[:-2]+(4,))
def transform_pose(p,Q,A):
    outp=p@A[:3,:3].T+A[:3,3];R=Rotation.from_quat(Q.reshape(-1,4),scalar_first=True).as_matrix().reshape(Q.shape[:-1]+(3,3))
    return outp,quat(A[:3,:3]@R)

def robot_frames(side,q,wrist):
    """Full URDF FK with actual joint origins/axes, including fixed fingertip frames."""
    path=OFFICIAL/'assets/urdfs/sharpawave'/f'{side}_sharpawave.urdf';xml=ET.parse(path).getroot();links=[e.attrib['name'] for e in xml.findall('link')];joints=xml.findall('joint');children={j.find('child').attrib['link'] for j in joints};root=next(x for x in links if x not in children)
    names=json.loads((IK/'report.json').read_text())['joint_names'][side];out=np.empty((len(q),len(links),7));memo=[]
    for j in joints:
        origin=j.find('origin');a=origin.attrib if origin is not None else {};T=np.eye(4);T[:3,3]=np.fromstring(a.get('xyz','0 0 0'),sep=' ');T[:3,:3]=Rotation.from_euler('xyz',np.fromstring(a.get('rpy','0 0 0'),sep=' ')).as_matrix();axis=j.find('axis');v=np.fromstring(axis.attrib['xyz'],sep=' ') if axis is not None else np.array([1,0,0]);memo.append((j.find('parent').attrib['link'],j.find('child').attrib['link'],T,v,j.attrib['type'],names.index(j.attrib['name']) if j.attrib['name'] in names else None))
    for f in range(len(q)):
        W=np.eye(4);W[:3,3]=wrist[f,:3];W[:3,:3]=Rotation.from_quat(wrist[f,3:],scalar_first=True).as_matrix();poses={root:W};pending=list(memo)
        while pending:
            progress=False
            for item in pending[:]:
                parent,child,T,axis,typ,qi=item
                if parent not in poses:continue
                M=np.eye(4)
                if typ!='fixed':
                    if qi is None:raise ValueError(f'Unmapped joint for {child}')
                    if typ in ('revolute','continuous'):M[:3,:3]=Rotation.from_rotvec(axis*q[f,qi]).as_matrix()
                    elif typ=='prismatic':M[:3,3]=axis*q[f,qi]
                    else:raise ValueError(typ)
                poses[child]=poses[parent]@T@M;pending.remove(item);progress=True
            if not progress:raise ValueError('Disconnected URDF FK tree')
        for i,name in enumerate(links):out[f,i]=np.r_[poses[name][:3,3],quat(poses[name][:3,:3])]
    # Independent saved MJCF site FK, including true orientations.
    m=mujoco.MjModel.from_xml_path(str(IK/f'{side}_kinematics_only.xml'));d=mujoco.MjData(m);sites=np.empty((len(q),m.nsite,7));site_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_SITE,i) for i in range(m.nsite)]
    for f in range(len(q)):
        d.qpos[:]=q[f];mujoco.mj_forward(m,d);R=Rotation.from_quat(wrist[f,3:],scalar_first=True).as_matrix();sites[f,:,:3]=d.site_xpos@R.T+wrist[f,:3];sites[f,:,3:]=quat(R@d.site_xmat.reshape(-1,3,3))
    return links,out,site_names,sites

def generate(report):
    data=np.load(IK/'checkpoint.npz');observed=data['present'];q=data['qpos'].copy();w=data['wrist_pose'].copy();source_frame=np.tile(np.arange(137)[:,None],(1,2))
    for hi in range(2):
        good=np.flatnonzero(observed[:,hi]);bad=np.flatnonzero(~observed[:,hi]);nearest=good[np.argmin(abs(bad[:,None]-good),axis=1)];q[bad,hi]=q[nearest,hi];w[bad,hi]=w[nearest,hi];source_frame[bad,hi]=nearest
    params=json.loads((GEO/'report.json').read_text());n=np.array(params['table_plane']['normal_world']);d=params['table_plane']['offset_m'];poses=np.load(GEO/'candidate_poses.npz')['world_T_canonical_initial']
    center=poses[0,:,:3,3].mean(0);origin=center-(n@center+d)*n;x=np.array([1.,0,0]);x-=n*(x@n);x/=np.linalg.norm(x);y=np.cross(n,x);A=np.eye(4);A[:3,:3]=np.stack([x,y,n]);A[:3,3]=-A[:3,:3]@origin
    for hi in range(2):w[:,hi,:3],w[:,hi,3:]=transform_pose(w[:,hi,:3],w[:,hi,3:],A)
    zposes=A@poses;objp=zposes[:,:,:3,3];objq=quat(zposes[:,:,:3,:3]);props=json.loads((COL/'report.json').read_text())['candidates']['wall_3000um']['geometry_quality'];com=np.array([p['properties']['com_m'] for p in props]);objcom=objp+np.einsum('tbij,bj->tbi',zposes[:,:,:3,:3],com)
    proxy=np.load(BASE/'contact_wrench_probe/provisional_reference.npz');distance=proxy['nearest_link_distance_m'];cp=np.zeros((137,2,16,3));cn=np.zeros_like(cp);pid=np.zeros((137,2,16),np.int32);available=np.zeros_like(pid,bool);selected_distance=np.full(pid.shape,np.nan)
    for f in range(137):
        for hi in range(2):
            for li in range(16):
                dd=distance[f,hi,:,li];finite=np.isfinite(dd)
                if not finite.any():continue
                oi=int(np.argmin(np.where(finite,dd,np.inf)));available[f,hi,li]=True;selected_distance[f,hi,li]=dd[oi]
                if dd[oi]>=.01:continue
                local=proxy['contact_pos_object_COM'][f,hi,oi,li]+proxy['canonical_COM_m'][oi]
                cp[f,hi,li]=zposes[f,oi,:3,:3]@local+objp[f,oi];cn[f,hi,li]=zposes[f,oi,:3,:3]@proxy['contact_normal_inward'][f,hi,oi,li];cn[f,hi,li]/=np.linalg.norm(cn[f,hi,li]);pid[f,hi,li]=oi+1
    meshes=[GEO/name/'wall_03mm_3000um.obj' for name in NAMES];radii=[float(np.linalg.norm(trimesh.load(path,process=False).vertices-com[i],axis=1).max()) for i,path in enumerate(meshes)];urdfs=[COL/'wall_3000um'/name/f'{name}.urdf' for name in NAMES]
    row={'fps':20.,'source_dataset':'v2d_track3_interface_preflight','sequence_id':'episode_000011','robot_name':'sharpa_wave','coord_frame':'table_Z_up_meters','object_name':'two_nested_cups','object_body_names':NAMES,'object_mesh_paths':list(map(str,meshes)),'object_urdf_paths':list(map(str,urdfs)),'object_mesh_radius':radii,'object_body_position':objp.tolist(),'object_body_wxyz':objq.tolist(),'object_articulation':np.zeros((137,1)).tolist(),'observed_hand':observed.tolist(),'planned_reference':(~observed).tolist(),'contact_ground_truth_known':np.zeros_like(pid,bool).tolist(),'interface_only_not_training_ready':True}
    fkextra={};ikreport=json.loads((IK/'report.json').read_text())
    for hi,side in enumerate(SIDES):
        names,frames,snames,sites=robot_frames(side,q[:,hi],w[:,hi]);fkextra[side+'_frame_names']=np.array(names);fkextra[side+'_frames']=frames;fkextra[side+'_site_names']=np.array(snames);fkextra[side+'_sites']=sites
        row.update({f'robot_{side}_wrist_position':w[:,hi,:3].tolist(),f'robot_{side}_wrist_wxyz':w[:,hi,3:].tolist(),f'robot_{side}_finger_joints':q[:,hi].tolist(),f'{side}_robot_finger_joint_names':ikreport['joint_names'][side],f'robot_{side}_frames':frames.tolist(),f'{side}_robot_frame_names':names,f'mano_{side}_object_contact_positions':cp[:,hi].tolist(),f'mano_{side}_object_contact_normals':cn[:,hi].tolist(),f'mano_{side}_object_contact_part_ids':pid[:,hi].tolist()})
    parquet=OUT/'episode11_interface_only.parquet';pq.write_table(pa.Table.from_pylist([row]),parquet)
    unknown_cp=cp.copy();unknown_cn=cn.copy();unknown_cp[~available]=np.nan;unknown_cn[~available]=np.nan
    np.savez_compressed(OUT/'reference_checkpoint.npz',table_T_world=A,world_T_table=np.linalg.inv(A),observed_hand=observed,planned_reference=~observed,planned_source_frame=source_frame,observed_wrist_world=data['wrist_pose'],observed_finger_q=data['qpos'],observed_sites_world=data['sites_world'],planned_wrist_table=w,planned_finger_q=q,object_link_pose_table=zposes,object_COM_table=objcom,canonical_COM=com,object_bounding_radius_COM=radii,proxy_contact_available=available,contact_ground_truth_known=np.zeros_like(pid,bool),proxy_contact_position_table_unknown_nan=unknown_cp,proxy_contact_normal_table_unknown_nan=unknown_cn,proxy_contact_part_ids=pid,proxy_selected_distance_m=selected_distance,**fkextra)
    report.update(parquet=str(parquet),stage='parquet_generated',frames=137,fps=20.,fps_evidence='ffprobe source ego_cam_a: 20/1,137 frames,6.85s',observed_hand_instances=int(observed.sum()),planned_missing_robot_instances=int((~observed).sum()),human_missing_preserved_nan=bool(np.isnan(data['wrist_pose'][~observed]).all() and np.isnan(data['qpos'][~observed]).all()),object_pose_source='agent_cup_reconstruction/candidate_poses.npz world_T_canonical_initial; chosen to match existing contact proxy exactly',object_pose_quality_pass=False,planned_reference_policy='Nearest observed initializer frame held for missing robot frames only; no inferred human observation and no collision-safe motion claim',contact_policy='Nearest object per MANO link, 16 slots per hand,10mm threshold,negative mesh outward normal; unavailable schema slots use0 exclusively as interface placeholders',contact_gt_known=False,contact_unknown_schema_supported=False,contact_placeholder_training_gate=False,active_proxy_slots=int((pid>0).sum()),table_T_world=A.tolist(),plane_mapping_error_m=float(abs((A@np.r_[origin,1])[2])),canonical_COM_m=com.tolist(),wrench_COM_contract_pass=False,wrench_COM_issue='Official scene spawns link origin from object_body_position; official compute_reference_supports subtracts same position as COM. Nonzero URDF body_com requires explicit downstream adaptation or consistently recentered assets; neither done here.',wrench_origin_error_m=np.linalg.norm(com,axis=1).tolist(),mesh_radius_about_URDF_COM_m=radii)
    save(OUT/'report.json',report);return parquet

def main():
    OUT.mkdir(exist_ok=True);start=time.monotonic();sources=[Path(__file__),IK/'checkpoint.npz',IK/'report.json',GEO/'candidate_poses.npz',GEO/'report.json',BASE/'contact_wrench_probe/provisional_reference.npz',OFFICIAL/'data/mano_sharpa.py',OFFICIAL/'data/loader.py',OFFICIAL/'scene/objects.py',OFFICIAL/'objectives/wrench.py']+[IK/f'{s}_kinematics_only.xml' for s in SIDES]+[OFFICIAL/'assets/urdfs/sharpawave'/f'{s}_sharpawave.urdf' for s in SIDES]+[COL/'wall_3000um'/n/f'{n}.urdf' for n in NAMES];hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources};cache=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest();old=json.loads((OUT/'report.json').read_text()) if (OUT/'report.json').exists() else {}
    if old.get('source_hash')==cache and old.get('state')=='completed':print('Completed matching preflight reused');return
    report={'state':'running','started_utc':now(),'source_hash':cache,'source_sha256':hashes,'budget_seconds':BUDGET,'training_ready':False,'policy_training':False,'gpu_tested':False}
    if old.get('source_hash')==cache and (OUT/'reference_checkpoint.npz').exists():report.update(old,state='running');parquet=Path(report['parquet']);print('Reused parquet checkpoint',flush=True)
    else:parquet=generate(report)
    try:
        import warp as wp,newton
        from flash_chord.data.loader import load_reference
        from flash_chord.embodiments.sharpa_hands import SharpaHands
        from flash_chord.scene.builder import build_scene
        from flash_chord.scene.collision import CollisionPolicy
        ref=load_reference(parquet);report['load_reference_pass']=True;report['stage']='reference_loaded';save(OUT/'report.json',report);print('Official loader passed',ref.num_frames,ref.fps,flush=True)
        wp.init()
        with wp.ScopedDevice('cpu'):scene=build_scene(SharpaHands(),ref,world_count=1,decompose_objects=False,contact_friction=.5,collision=CollisionPolicy())
        report['stage']='scene_built';report['scene_build_pass']=True;save(OUT/'report.json',report);print('Official scene built',scene.model.body_count,scene.model.shape_count,flush=True)
        checkpoint=np.load(OUT/'reference_checkpoint.npz');frame_checks=[]
        for side in SIDES:
            bound=scene.robot_reference.hand(side);names=checkpoint[side+'_frame_names'].tolist();frames=checkpoint[side+'_frames'];target_names=[side+'_hand_C_MC']+[side+'_'+digit+'_DP' for digit in ['thumb','index','middle','ring','pinky']]+[side+'_'+digit+'_fingertip' for digit in ['thumb','index','middle','ring','pinky']];idx=[names.index(n) for n in target_names];expected=frames[:,idx];actual_p=np.concatenate([bound.wrist_pos_w[:,None],bound.dp_pos_w,bound.fingertip_pos_w],axis=1);actual_q=np.concatenate([bound.wrist_quat_w[:,None],bound.dp_quat_w,bound.fingertip_quat_w],axis=1);position_error=np.linalg.norm(actual_p-expected[:,:,:3],axis=-1);dots=abs(np.sum(actual_q*expected[:,:,3:],axis=-1));angle=2*np.arccos(np.clip(dots,0,1));frame_checks.append({'side':side,'names':target_names,'position_error_max_m':float(position_error.max()),'orientation_error_max_rad':float(angle.max())})
        model=scene.model;state=model.state();q=model.joint_q.numpy();q[:scene.layout.num_joint_q]=scene.robot_reference.joint_q[0];state.joint_q.assign(q);newton.eval_fk(model,state.joint_q,state.joint_qd,state);bodyq=state.body_q.numpy();coms=model.body_com.numpy();object_checks=[]
        for i,binding in enumerate(scene.objects):
            bi=binding.root.body_id;R=Rotation.from_quat(bodyq[bi,3:]).as_matrix();worldcom=bodyq[bi,:3]+R@coms[bi];object_checks.append({'name':binding.name,'body_id':bi,'link_position_error_m':float(np.linalg.norm(bodyq[bi,:3]-ref.object_body_pos_w()[0,i])),'COM_position_error_m':float(np.linalg.norm(worldcom-checkpoint['object_COM_table'][0,i])),'body_COM_local_m':coms[bi].tolist(),'shape_count_including_visual':len(binding.shapes)})
        solver=newton.solvers.SolverMuJoCo(model,use_mujoco_cpu=True,use_mujoco_contacts=True,nconmax=4096,njmax=12000,save_to_mjcf=str(OUT/'official_scene_export.xml'));report['solver_compile_pass']=True
        np.savez_compressed(OUT/'scene_checkpoint.npz',body_q_frame0=bodyq,body_com=coms,body_mass=model.body_mass.numpy(),joint_q_frame0=q,robot_joint_q=scene.robot_reference.joint_q,robot_joint_target=scene.robot_reference.joint_target,body_labels=np.array(model.body_label))
        report.update(scene={'device':'cpu','world_count':scene.world_count,'body_count':model.body_count,'shape_count':model.shape_count,'joint_q_count':model.joint_coord_count,'joint_dof_count':model.joint_dof_count,'robot_joint_q_count':scene.layout.num_joint_q,'robot_joint_dof_count':scene.layout.num_joint_dof,'ground':'inferred tabletop transformed to z=0; no support USDA','decompose_objects':False,'contact_friction_assumed':.5,'native_mujoco_nbody':solver.mj_model.nbody,'native_mujoco_ngeom':solver.mj_model.ngeom,'physics_steps':0},frame_checks=frame_checks,object_checks=object_checks,frame_FK_gate=all(x['position_error_max_m']<1e-4 and x['orientation_error_max_rad']<.002 for x in frame_checks),object_link_COM_gate=all(x['link_position_error_m']<1e-6 and x['COM_position_error_m']<1e-6 for x in object_checks),interface_preflight_pass=True,state='completed',completed_utc=now(),elapsed_seconds=time.monotonic()-start)
    except Exception:
        report.update(state='failed',stage_error=traceback.format_exc(),updated_utc=now(),elapsed_seconds=time.monotonic()-start);save(OUT/'failure.json',report);print(report['stage_error'],flush=True)
    report['training_ready']=False;save(OUT/'report.json',report);print('RESULT',json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','table_T_world']}),flush=True)

if __name__=='__main__':main()
