"""Audit final pose transforms, temporal steps, table clearance and hard failures."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from agent_cup_reconstruct import ROOT,save
from agent_cup_pose_refine import OUT,SOURCE,NAMES,now,full_metrics,comparison

def main():
    r=json.loads((OUT/'report.json').read_text())
    if r['state']!='completed':raise RuntimeError('Wait for the pose experiment to complete')
    s=json.loads((SOURCE/'report.json').read_text());z=np.load(OUT/'candidate_poses.npz');old=z['world_T_canonical_initial'];new=z['world_T_canonical_refined']
    assert old.shape==new.shape==(137,2,4,4)
    for array in [old,new]:
        assert np.isfinite(array).all();assert np.allclose(np.linalg.det(array[:,:,:3,:3]),1,atol=1e-6)
        assert np.allclose(array[:,:,:3,:3].swapaxes(-1,-2)@array[:,:,:3,:3],np.eye(3),atol=1e-6)
    delta=new@np.linalg.inv(old)
    # Existing normalized correction is pre-Slerp/interpolation metadata; this
    # exact matrix is the authoritative final rigid correction for any consumer.
    np.savez_compressed(OUT/'exact_world_rigid_delta.npz',frame_index=np.arange(137),object_names=NAMES,world_rigid_delta=delta)
    n=np.array(s['table_plane']['normal_world']);offset=s['table_plane']['offset_m'];angle=np.arange(128)*2*np.pi/128
    audit={};violations=[]
    for variant,P in [('before',old),('after',new)]:
        objects={}
        for oi,name in enumerate(NAMES):
            shape=s['objects'][name]['shape'];rr=shape['bottom_outer_radius_m'];bottom=np.column_stack((rr*np.cos(angle),rr*np.sin(angle),np.zeros(128)))
            world=np.einsum('fij,kj->fki',P[:,oi,:3,:3],bottom)+P[:,oi,None,:3,3]
            plane_distance=world@n+offset
            translation=np.linalg.norm(np.diff(P[:,oi,:3,3],axis=0),axis=1)
            # Tilt change ignores arbitrary axisymmetric yaw.
            axis=P[:,oi,:3,2];axis_step=np.rad2deg(np.arccos(np.clip((axis[1:]*axis[:-1]).sum(-1),-1,1)))
            objects[name]={'table_clearance_min_m':float(plane_distance.min()),'table_clearance_per_frame_min_m':plane_distance.min(1).tolist(),'translation_step_p50_p95_max_m':np.quantile(translation,[.5,.95,1]).tolist(),'axis_step_p50_p95_max_deg':np.quantile(axis_step,[.5,.95,1]).tolist(),'frames_below_table_by_over_3mm':np.flatnonzero(plane_distance.min(1)<-.003).tolist()}
        T=np.linalg.inv(P[-1,0])@P[-1,1]
        audit[variant]={'objects':objects,'final_axis_angle_deg':float(np.rad2deg(np.arccos(np.clip(T[2,2],-1,1)))),'final_bottom_eccentricity_m':float(np.linalg.norm(T[:2,3])),'final_beige_base_in_blue_m':T[:3,3].tolist()}
    r['temporal_and_support_audit']=audit
    ev=r['evaluation_frames'];mean_before=[x['material_before']['sampled_mean_material_penetration_m'] for x in ev];mean_after=[x['material_after']['sampled_mean_material_penetration_m'] for x in ev]
    frac_before=[x['material_before']['fraction_samples_inside_other_material'] for x in ev];frac_after=[x['material_after']['fraction_samples_inside_other_material'] for x in ev]
    r['material_overlap_summary']={'before':{'median_mean_penetration_m':float(np.median(mean_before)),'worst_mean_penetration_m':float(max(mean_before)),'worst_fraction_samples_inside':float(max(frac_before))},'after':{'median_mean_penetration_m':float(np.median(mean_after)),'worst_mean_penetration_m':float(max(mean_after)),'worst_fraction_samples_inside':float(max(frac_after))}}
    rows=[dict(y,frame=x['frame']) for x in ev for y in x['after']]
    worst_depth=sorted([x for x in rows if x['depth_abs_p50_p95_m']],key=lambda x:x['depth_abs_p50_p95_m'][1],reverse=True)[:5]
    worst_overlap=sorted(ev,key=lambda x:x['material_after']['fraction_samples_inside_other_material'],reverse=True)[:5]
    r['failure_examples']={'largest_depth_p95':worst_depth,'largest_material_overlap':[{'frame':x['frame'],'heldout':x['heldout'],'after':x['material_after']} for x in worst_overlap],'table_penetration_frame':{n:int(np.argmin(o['table_clearance_per_frame_min_m'])) for n,o in audit['after']['objects'].items()},'interpretation':'Depth tails may reflect cup-interior stereo failures or assumed internal geometry; exclude neither without an independent consistency check. Table penetration is a missing objective constraint, not a tolerable small gate miss.'}
    shapes=[s['objects'][n]['shape'] for n in NAMES]
    for f in [28,86,113]:
        _,bi,_=full_metrics(f,old[f],shapes,.003);_,ai,_=full_metrics(f,new[f],shapes,.003)
        comparison(f,bi,ai,OUT/'previews'/f'failure_comparison_{f:06d}.jpg')
    r['search_gates_before_audit']=dict(r['gates'])
    material_ok=max(frac_after)<.01
    table_ok=all(o['table_clearance_min_m']>=-.003 for o in audit['after']['objects'].values())
    r['gates']['material_nonpenetration_all_evaluated_frames']=material_ok
    r['gates']['table_support_all_frames_within_3mm']=table_ok
    r['gates']['geometric_reference_ready']=bool(r['gates']['geometric_reference_ready'] and material_ok and table_ok)
    r['final_pose_metadata_note']='Use world_T_canonical_refined or exact_world_rigid_delta.npz. normalized_correction is TRAIN-fit/pre-Slerp metadata and is not the exact transform on interpolated frames.'
    r['depth_reliability_definition']=s.get('depth_reliability_definition')
    r['final_audit_utc']=now();r['source_sha256']['scripts/agent_cup_pose_audit.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    r['completed_stage_compute_seconds']=r['elapsed_seconds']+13.45355859119445
    save(OUT/'report.json',r)
    j=ROOT/'notes/track3-agent-cup-pose-job.json';job=json.loads(j.read_text());job.update(state='completed',exit_code=int(Path(job['exit_file']).read_text()),quality_gate_pass=r['gates']['geometric_reference_ready'],post_audit='scripts/agent_cup_pose_audit.py',actual_combined_pilot_and_extension_seconds=r['completed_stage_compute_seconds']);save(j,job)
    print(json.dumps({'gates':r['gates'],'summaries':r['summaries'],'material_overlap_summary':r['material_overlap_summary'],'final_pose':{k:{x:y for x,y in v.items() if x!='objects'} for k,v in audit.items()},'table_min':{name:o['table_clearance_min_m'] for name,o in audit['after']['objects'].items()},'runtime_seconds':r['completed_stage_compute_seconds']},indent=2))

if __name__=='__main__':main()
