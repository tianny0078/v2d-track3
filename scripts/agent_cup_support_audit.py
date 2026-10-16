"""Finalize bounded supported-pose experiment with uncertainty and failure audits."""
import json,hashlib
from pathlib import Path
import numpy as np
from agent_cup_reconstruct import ROOT,NAMES,save,HOLDOUT
from agent_cup_support_refine import OUT,SHAPE_SOURCE,clearance,now
from agent_cup_pose_refine import full_metrics,comparison

def cavity_diagnostic(P,shapes,t):
    # Diagnostic for frames visually identified as nesting. Not an optimization
    # constraint and not a proof that a different pose cannot be feasible.
    blue,beige=shapes;z=np.linspace(0,beige['height_m'],80);a=np.arange(128)*2*np.pi/128
    r=beige['bottom_outer_radius_m']+beige['taper_dr_dz']*z
    q=np.stack([r[:,None]*np.cos(a),r[:,None]*np.sin(a),np.broadcast_to(z[:,None],(80,128))],axis=-1).reshape(-1,3)
    T=np.linalg.inv(P[0])@P[1];q=q@T[:3,:3].T+T[:3,3]
    v=q[(q[:,2]>=.003)&(q[:,2]<=blue['height_m'])]
    gap=blue['bottom_outer_radius_m']+blue['taper_dr_dz']*v[:,2]-t-np.linalg.norm(v[:,:2],axis=1)
    return {'samples_in_blue_height_range':len(v),'cavity_radial_clearance_min_m':float(gap.min()) if len(gap) else None,'fraction_outside_assumed_inner_radius':float((gap<0).mean()) if len(gap) else None,'interpretation':'Negative clearance at this pose contradicts full containment of the inserted portion; it does not prove pose-only infeasibility, and does not distinguish shape error from pose error.'}

def main():
    report=json.loads((OUT/'report.json').read_text());assert report['state']=='completed'
    shapesource=json.loads((SHAPE_SOURCE/'report.json').read_text());shapes=[shapesource['objects'][n]['shape'] for n in NAMES]
    n=np.array(shapesource['table_plane']['normal_world']);offset=shapesource['table_plane']['offset_m'];z=np.load(OUT/'candidate_poses.npz');P=z['world_T_canonical_refined']
    assert np.isfinite(P).all() and np.allclose(np.linalg.det(P[:,:,:3,:3]),1,atol=1e-6)
    assert all(not set(w['frames'])&HOLDOUT for w in report['window_results'])
    gaps=np.array([[clearance(P[f,oi],shapes[oi],n,offset) for oi in range(2)] for f in range(137)])
    assert gaps.min()>=-1e-7
    if not np.allclose(z['world_rigid_delta']@z['world_T_canonical_before'],P,atol=1e-8):raise AssertionError('Rigid delta composition mismatch')
    report['artifact_validation']={'all_137_poses_finite_SO3':True,'world_rigid_delta_composition_pass':True,'heldout_frames_not_directly_optimized':True,'all_137_frames_table_clearance_min_m':float(gaps.min()),'comparison_baseline':'Previous render-optimized (and partly table-penetrating) poses, not original v3.'}
    report['nesting_cavity_diagnostic']=[dict(frame=f,wall_thickness_m=t,**cavity_diagnostic(P[f],shapes,t)) for f in [68,86,113,136] for t in [.0015,.003,.005]]
    projected=z['world_T_canonical_table_projected'];before=z['world_T_canonical_before'];cost=[]
    for f in [28,68,86,113,136]:
        rows,_,material=full_metrics(f,projected[f],shapes,.003)
        cost.append({'frame':f,'projection_only_image_metrics':rows,'projection_only_material':material,'translation_applied_m':np.linalg.norm(projected[f,:,:3,3]-before[f,:,:3,3],axis=1).tolist()})
    report['table_projection_only_cost']=cost
    # Retain views of the actual worst failures, not just successful keyframes.
    allrows=[dict(y,frame=x['frame']) for x in report['evaluation_frames'] for y in x['after']]
    worst=sorted([x for x in allrows if x['depth_abs_p50_p95_m']],key=lambda x:x['depth_abs_p50_p95_m'][1],reverse=True)[:5]
    report['failure_frames']['depth_p95_worst']=worst
    fdepth=worst[0]['frame'];fmaterial=report['failure_frames']['material_worst'][0]['frame']
    for f in sorted({fdepth,fmaterial}):
        _,bi,_=full_metrics(f,before[f],shapes,.003);_,ai,_=full_metrics(f,P[f],shapes,.003)
        comparison(f,bi,ai,OUT/'previews'/f'failure_{f:06d}.jpg')
    report['final_audit_utc']=now();report['source_sha256']['scripts/agent_cup_support_audit.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['next_action']='Use failure windows and cavity clearance to separate pose ambiguity, inner-geometry uncertainty and stereo outliers. Table feasibility passes but does not establish collision-ready geometry.'
    save(OUT/'report.json',report)
    path=ROOT/'notes/track3-agent-cup-support-job.json';job=json.loads(path.read_text());job.update(state='completed',exit_code=int(Path(job['exit_file']).read_text()),elapsed_seconds=report['elapsed_seconds'],quality_gate_pass=report['gates']['geometric_reference_ready'],table_nonpenetration_pass=report['gates']['table_nonpenetration_hard_pass'],post_audit='scripts/agent_cup_support_audit.py');save(path,job)
    print(json.dumps({'gates':report['gates'],'summaries':report['summaries'],'material':report['material_summary'],'pose_audit':{k:{x:y for x,y in v.items() if x!='per_frame_table_clearance_m'} for k,v in report['temporal_and_table_audit'].items()},'elapsed_seconds':report['elapsed_seconds'],'accepted_windows':sum(x['accepted'] for x in report['window_results'])},indent=2))

if __name__=='__main__':main()
