"""Bounded COM wrench adapter and unknown-contact semantics probe; no training."""
from pathlib import Path
import json,time,traceback,hashlib
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation
from types import SimpleNamespace
from agent_cup_reconstruct import ROOT,BASE,NAMES,save,now
from track3_wrench_adapter import supports,convert_origin,world_to_origin,runtime_link_to_com,bridge_mujoco_cpu_contacts,masked_contact_terms,transport_known_support

OUT=BASE/'wrench_origin_semantics';PRE=BASE/'flashchord_reference_preflight';OFFICIAL=ROOT/'vendor/video_to_data/robotic_grounding/flash_chord/src/flash_chord';FRAMES=[0,68,136];MU=.3

def official_terms(ref,cur):
    from flash_chord.objectives.contact import contact_wrench_support_objective,missed_contact_penalty,unintended_contact_penalty,force_closure_objective
    W,H,B,K=ref.shape;rd=wp.array(ref.astype(np.float32).ravel(),dtype=wp.float32);cd=wp.array(cur.astype(np.float32).ravel(),dtype=wp.float32);out=[]
    for kernel,extra in [(contact_wrench_support_objective,[.1,1.]),(missed_contact_penalty,[]),(unintended_contact_penalty,[]),(force_closure_objective,[.01])]:
        result=wp.zeros(W,dtype=wp.float32);wp.launch(kernel,dim=W,inputs=[rd,cd,H,B,K,*extra],outputs=[result]);out.append(result.numpy())
    return np.stack(out,axis=-1)

def physical_scene(report):
    import newton
    from flash_chord.data.loader import load_reference
    from flash_chord.embodiments.sharpa_hands import SharpaHands
    from flash_chord.scene.builder import build_scene
    from flash_chord.runtime.contact import ContactTracker
    ref=load_reference(PRE/'episode11_interface_only.parquet');scene=build_scene(SharpaHands(),ref,world_count=1,decompose_objects=False,contact_friction=MU);model=scene.model
    solver=newton.solvers.SolverMuJoCo(model,use_mujoco_cpu=True,use_mujoco_contacts=True,update_data_interval=1,nconmax=4096,njmax=12000)
    contacts=model.contacts();tracker=ContactTracker.build(scene.layout,tuple(b.root.body_id for b in scene.objects),1,model.body_count,scene.collision_layout.shape_count,sides=('left','right'),device='cpu')
    try:solver.update_contacts(contacts);native_bridge_required=False
    except NotImplementedError:native_bridge_required=True
    report['newton_CPU_update_contacts_implemented']=not native_bridge_required
    object_ids=np.array(tracker.layout.slot_object_ids);hand_ids=np.array(tracker.layout.slot_hand_ids);com=model.body_com.numpy()[[b.root.body_id for b in scene.objects]];rows=[];samples=[]
    for frame in FRAMES:
        a=model.state();b=model.state();q=model.joint_q.numpy();q[:scene.layout.num_joint_q]=scene.robot_reference.joint_q[frame]
        for oi,binding in enumerate(scene.objects):
            ids=list(binding.root.free_q_ids);q[ids[:3]]=ref.object_body_pos_w()[frame,oi];q[ids[3:]]=np.roll(ref.object_body_quat_w()[frame,oi],-1)
        a.joint_q.assign(q);a.joint_qd.zero_();newton.eval_fk(model,a.joint_q,a.joint_qd,a);control=model.control();target=np.zeros(model.joint_dof_count,np.float32);target[:scene.layout.num_joint_dof]=scene.robot_reference.joint_target[frame];control.joint_target_pos.assign(target)
        a.clear_forces();solver.step(a,b,control,contacts,.001);stats=bridge_mujoco_cpu_contacts(solver,contacts);tracker.update(model,b,contacts)
        # Warp CPU .numpy() may share storage: snapshots MUST own their memory.
        p_link=tracker.contact_pos_o.numpy().copy();p_world=tracker.contact_pos_w.numpy().copy();force=tracker.contact_force_w.numpy().copy();direction_world=force/np.maximum(np.linalg.norm(force,axis=-1,keepdims=True),1e-20);direction_link=tracker.contact_force_direction_o.numpy().copy();active=tracker.contact_active.numpy().astype(bool)
        device_com=wp.empty_like(tracker.contact_pos_o);wp.launch(runtime_link_to_com,dim=len(p_link),inputs=[tracker.contact_pos_o,tracker.slot_object_ids,wp.array(com,dtype=wp.vec3)],outputs=[device_com]);p_com=device_com.numpy().copy();bodyq=b.body_q.numpy()[[x.root.body_id for x in scene.objects]].copy()
        p_ref,n_ref=world_to_origin(p_world,direction_world,bodyq[object_ids,:3],bodyq[object_ids,3:],com[object_ids],target_origin='com')
        p_ref[~active]=0.;n_ref[~active]=0.;p_com[~active]=0.;direction_link[~active]=0.;p_link[~active]=0.
        valid=active&(np.linalg.norm(force,axis=-1)>1e-8)
        # Independent host force summation from native MuJoCo, not bridge output.
        import mujoco
        expected_force=np.zeros_like(force,dtype=np.float64);mapping=solver.mjc_geom_to_newton_shape.numpy()[0];shape_body=model.shape_body.numpy();layout=tracker.layout
        for ci,c in enumerate(solver.mj_data.contact[:solver.mj_data.ncon]):
            s0=int(mapping[c.geom1]);s1=int(mapping[c.geom2]);b0=int(shape_body[s0]);b1=int(shape_body[s1]);h0=layout.shape_hand_ids[s0] if s0<len(layout.shape_hand_ids) else -1;h1=layout.shape_hand_ids[s1] if s1<len(layout.shape_hand_ids) else -1;o0=layout.body_object_ids[b0] if b0>=0 else -1;o1=layout.body_object_ids[b1] if b1>=0 else -1
            if h0>=0 and o1>=0:hi,oi,li,sign=h0,o1,layout.shape_link_ids[s0],1.
            elif h1>=0 and o0>=0:hi,oi,li,sign=h1,o0,layout.shape_link_ids[s1],-1.
            else:continue
            raw=np.zeros(6);mujoco.mj_contactForce(solver.mj_model,solver.mj_data,ci,raw)
            slot=layout.hand_slot_starts[hi]+oi*layout.link_counts[hi]+li;expected_force[slot]+=sign*np.asarray(c.frame).reshape(3,3).T@raw[:3]
        stats.update(frame=frame,newton_steps=1,dt=.001,active_contact_slots=int(active.sum()),nonzero_force_slots=int(valid.sum()),per_cup_nonzero_force_slots=[int((valid&(object_ids==oi)).sum()) for oi in range(2)],world_vs_tracker_COM_position_error_max_m=float(np.linalg.norm(p_ref[valid]-p_com[valid],axis=-1).max(initial=0.)),world_vs_tracker_direction_error_max=float(np.linalg.norm(n_ref[valid]-direction_link[valid],axis=-1).max(initial=0.)),native_vs_tracker_force_relative_error=float(np.linalg.norm(expected_force-force)/max(np.linalg.norm(expected_force),1e-12)))
        samples.append(dict(frame=frame,p_link=p_link,p_com=p_com,p_ref=p_ref,n_link=direction_link,n_ref=n_ref,p_world=p_world,n_world=direction_world,force_world=force,bodyq=bodyq,active=active));rows.append(stats);print('PHYSICS',json.dumps(stats),flush=True)
        save(OUT/'physics_progress.json',{'rows':rows})
    np.savez_compressed(OUT/'physical_contacts.npz',frame_index=FRAMES,object_ids=object_ids,hand_ids=hand_ids,body_com=com,radii=ref.object_mesh_radius(),**{k:np.array([s[k] for s in samples]) for k in samples[0] if k!='frame'})
    report['physics']=rows;report['contact_bridge']='Project host bridge from SolverMuJoCo CPU native contact/force arrays into Newton contacts, followed by unmodified official ContactTracker.update; not production integration';report['physics_steps_total']=3;report['physics_is_reconstruction_validation']=False;report['scene_checkpoint']=str(OUT/'physical_contacts.npz');save(OUT/'report.json',report)

def check_contacts(report):
    z=np.load(OUT/'physical_contacts.npz');oid=z['object_ids'];hid=z['hand_ids'];com=z['body_com'];rc=z['radii'];K=max(np.bincount(hid*2+oid));cells=[];allpoints={k:[] for k in ['p_link','p_com','p_ref','n_link','n_ref','p_rotated','n_rotated']};rigid=Rotation.from_rotvec([.37,-.21,.19]);Q=rigid.as_matrix();t=np.array([.31,-.22,.47]);transform_error=[]
    for f in range(len(z['frame_index'])):
        bq=z['bodyq'][f];newpos=bq[:,:3]@Q.T+t;newquat=(rigid*Rotation.from_quat(bq[:,3:])).as_quat();p_rot,n_rot=world_to_origin(z['p_world'][f]@Q.T+t,z['n_world'][f]@Q.T,newpos[oid],newquat[oid],com[oid],target_origin='com');p_rot[~z['active'][f]]=0.;n_rot[~z['active'][f]]=0.
        for h in range(2):
            for o in range(2):
                ids=np.flatnonzero((hid==h)&(oid==o));cells.append(o)
                for key in allpoints:
                    value=p_rot if key=='p_rotated' else n_rot if key=='n_rotated' else z[key][f];arr=np.zeros((K,3));arr[:len(ids)]=value[ids];allpoints[key].append(arr)
    sp={}
    for label,p,n in [('reference_COM','p_ref','n_ref'),('runtime_COM','p_com','n_link'),('runtime_link','p_link','n_link'),('rigid_world_COM','p_rotated','n_rotated')]:sp[label]=supports(np.array(allpoints[p]),np.array(allpoints[n]),rc[cells],cells,mu=MU)
    delta=abs(sp['reference_COM']-sp['runtime_COM']);rigiddelta=abs(sp['reference_COM']-sp['rigid_world_COM']);origin_delta=abs(sp['runtime_link']-sp['runtime_COM']);n=sp['runtime_COM'].shape[1];rows=[]
    for o,name in enumerate(NAMES):
        selected=np.array(cells)==o;rows.append({'object':name,'same_contact_reference_runtime_support_max_error':float(delta[selected].max()),'world_rigid_transform_support_max_error':float(rigiddelta[selected].max()),'COM_vs_link_support_max_difference':float(origin_delta[selected].max()),'COM_vs_link_changed_basis_values_over_1e_5':int((origin_delta[selected]>1e-5).sum())})
    np.savez_compressed(OUT/'physical_supports.npz',cell_object_id=cells,shared_basis_radius_argument=1.,kernel_radius=rc[cells],**sp)
    report['support_checks']=rows;report['COM_adapter_physical_gate']=bool(all(r['same_contact_reference_runtime_support_max_error']<1e-5 and r['world_rigid_transform_support_max_error']<1e-5 and r['COM_vs_link_support_max_difference']>1e-4 for r in rows));report['normal_changed_by_origin_translation']=False;report['radius_normalization']='sample_wrench_basis(512, rc=1, seed=object_index), exactly one moment / COM_radius inside official wrench_support';return sp['runtime_COM'].reshape(len(z['frame_index']),2,2,n)

def reference_proxy_check(report):
    from flash_chord.data.loader import load_reference
    from flash_chord.objectives.wrench import compute_reference_supports
    ref=load_reference(PRE/'episode11_interface_only.parquet');z=np.load(PRE/'reference_checkpoint.npz');com=z['canonical_COM'];Q=Rotation.from_quat(ref.object_body_quat_w().reshape(-1,4),scalar_first=True).as_matrix().reshape(137,2,3,3);comworld=ref.object_body_pos_w()+np.einsum('tbij,bj->tbi',Q,com)
    class WrenchCOMView:
        def __getattr__(self,name):return getattr(ref,name)
        def object_body_pos_w(self):return comworld
    official=compute_reference_supports(WrenchCOMView(),mu=MU,device='cpu');p=[];n=[];ids=[]
    for f in range(137):
        for hi,side in enumerate(['left','right']):
            for oi in range(2):
                pp=(ref.contact_pos_w(side)[f]-ref.object_body_pos_w()[f,oi])@Q[f,oi];nn=ref.contact_normal_w(side)[f]@Q[f,oi];pp=convert_origin(pp,com[oi],source_origin='link',target_origin='com');nn*=((ref.contact_part_ids(side)[f]==oi+1)[:,None]);p.append(pp);n.append(nn);ids.append(oi)
    runtime= supports(p,n,np.asarray(ref.object_mesh_radius())[ids],ids,mu=MU).reshape(137,2,2,512);error=float(abs(official-runtime).max());known=z['contact_ground_truth_known'].any(axis=-1)[:,:,None]*np.ones((1,1,2),bool);unknown=np.where(known[...,None],official,np.nan)
    np.savez_compressed(OUT/'proxy_supports_with_unknown.npz',reference_COM=official,runtime_COM=runtime,ground_truth_known=known,ground_truth_support_unknown_nan=unknown,observed_hand=z['observed_hand'],proxy_available=z['proxy_contact_available'],object_COM_world=comworld)
    report['proxy_both_sides_COM_support_max_error']=error;report['proxy_COM_adapter_gate']=error<1e-5;report['proxy_is_GT']=False;report['actual_GT_known_cells']=int(known.sum());report['human_NaN_sidecar_preserved']=bool(np.isnan(z['observed_wrist_world'][~z['observed_hand']]).all())

def semantics(report,current):
    zero=np.zeros_like(current);known=np.ones(current.shape[:-1],bool);cases=[]
    for name,r,c in [('known_no_contact',zero,current),('known_match',current,current),('known_missed',current,zero)]:
        official=official_terms(r,c);adapted=masked_contact_terms(r,c,known);cases.append({'case':name,'official':official.tolist(),'masked':adapted['terms'].tolist(),'max_error':float(abs(official-adapted['terms']).max())})
    unknown=masked_contact_terms(zero,current,np.zeros_like(known));reference_only=official_terms(zero,current)
    # A single known body must keep its penalty when a second unknown body is added.
    positive=current.reshape(-1,512);v=positive[np.argmax(np.linalg.norm(positive,axis=1))];c=np.stack([v,np.zeros(512)])[None,None];r=np.zeros_like(c);mask=np.array([[[True,False]]]);mixed=masked_contact_terms(r,c,mask);naive=official_terms(r,c);standalone=official_terms(r[:,:,:1],c[:,:,:1]);perturbed=c.copy();perturbed[:,:,1]=1e5;invariant=masked_contact_terms(r,perturbed,mask)
    report['unknown_semantics']={'official_part_id_zero':'No object contact; produces zero reference support, interpreted as inactive command by unintended_contact; ManoSharpaReference.contact_active also false.','all_known_kernel_parity':cases,'zero_reference_only_unintended_penalties':reference_only[:,2].tolist(),'unknown_both_sides_masked_terms':unknown['terms'].tolist(),'unknown_terms_valid':unknown['valid'].tolist(),'mixed_known_unknown_unintended':float(mixed['terms'][0,2]),'naive_both_sides_zero_unintended':float(naive[0,2]),'known_cell_standalone_unintended':float(standalone[0,2]),'unknown_current_perturbation_max_delta':float(abs(invariant['terms']-mixed['terms']).max()),'all_unknown_reward_has_no_valid_supervision':bool(not unknown['valid'].any()),'known_mask_source':'Counterfactual semantics tests only; actual data contact_ground_truth_known remains all false.','production_env_reference_buffer_hooked':False,'production_reward_hooked':False,'GPU_known_mask_kernel_tested':False,'masked_reduction_is_host_semantic_prototype':True}
    report['unknown_semantic_probe_gate']=bool(np.max(current)>1e-3 and reference_only[:,2].max()>0 and max(x['max_error'] for x in cases)<2e-6 and not unknown['valid'].any() and abs(mixed['terms'][0,2]-standalone[0,2])<2e-6 and abs(invariant['terms']-mixed['terms']).max()==0)

def main():
    OUT.mkdir(exist_ok=True);start=time.monotonic();paths=[Path(__file__),ROOT/'scripts/track3_wrench_adapter.py',PRE/'episode11_interface_only.parquet',PRE/'reference_checkpoint.npz',OFFICIAL/'objectives/wrench.py',OFFICIAL/'objectives/contact.py',OFFICIAL/'runtime/contact.py',OFFICIAL/'objectives/tracking.py',OFFICIAL/'data/mano_sharpa.py',OFFICIAL/'objectives/recon_body.py'];hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest();old=json.loads((OUT/'report.json').read_text()) if (OUT/'report.json').exists() else {}
    if old.get('source_hash')==key and old.get('state')=='completed':print('Matching completed bounded probe retained');return
    report={'state':'running','started_utc':now(),'source_hash':key,'source_sha256':hashes,'training_ready':False,'policy_training':False,'GPU_tested':False,'friction_assumed':MU,'reference_frame_subset':FRAMES,'budget_seconds':540};save(OUT/'report.json',report)
    try:
        wp.init()
        with wp.ScopedDevice('cpu'):
            if old.get('source_hash')==key and (OUT/'physical_contacts.npz').exists():report.update(old,state='running');print('Reusing actual contact checkpoint',flush=True)
            else:physical_scene(report)
            current=check_contacts(report);save(OUT/'report.json',report);reference_proxy_check(report);save(OUT/'report.json',report);semantics(report,current)
        report.update(state='completed',completed_utc=now(),elapsed_seconds=time.monotonic()-start,production_training_integration=False,remaining_gates=['Full environment reference/reward known-mask integration absent.','All current contact ground-truth labels remain unknown; proximity proxy is not measured force.','Three independent 1ms contact readouts do not validate reconstructed trajectories or dynamics stability.','Current shape/pose/hand geometry quality remains failed.','CPU contact bridge is experimental; GPU and official native CPU update_contacts unverified/unsupported.'])
    except Exception:
        report.update(state='failed',error=traceback.format_exc(),elapsed_seconds=time.monotonic()-start);save(OUT/f'failure_{int(time.time())}.json',report);print(report['error'],flush=True)
    save(OUT/'report.json',report);print('FINAL',json.dumps({k:v for k,v in report.items() if k!='source_sha256'}),flush=True)

if __name__=='__main__':main()
