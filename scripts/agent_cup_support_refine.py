"""Fixed-shell pose windows with a hard world-table feasibility projection.

Three control poses jointly explain each five-frame TRAIN window. No object is
attracted to the table: only infeasible negative clearance is projected upward.
Heldout frames receive TRAIN-pose interpolation and the same geometric plane prior.
"""
from __future__ import annotations
import json,time,hashlib,traceback
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation,Slerp
from agent_cup_reconstruct import ROOT,BASE,HOLDOUT,NAMES,save
from agent_cup_pose_refine import (FrameObservation,perturb,samples,overlap,full_metrics,
                                 comparison,aggregate,now,SCALE,BudgetEnd)

OUT=BASE/'agent_cup_pose_supported'
SHAPE_SOURCE=BASE/'agent_cup_reconstruction'
POSE_SOURCE=BASE/'agent_cup_pose_refinement'
BUDGET=540
WALL=.003

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clearance(P,s,n,offset):
    v=P[:3,:3].T@n;radial=np.linalg.norm(v[:2]);h=s['height_m'];r=s['bottom_outer_radius_m'];k=s['taper_dr_dz']
    return float(n@P[:3,3]+offset+min(-r*radial,h*v[2]-(r+k*h)*radial))
def feasible(P,shapes,n,offset):
    P=P.copy();shifts=[]
    for oi in range(2):
        gap=clearance(P[oi],shapes[oi],n,offset);shift=max(0.,.0002-gap) if gap<0 else 0.
        P[oi,:3,3]+=shift*n;shifts.append(shift)
    return P,shifts

def windows():
    groups=[];current=[]
    for f in range(137):
        if f in HOLDOUT:
            if current:groups.append(current);current=[]
        else:current.append(f)
    if current:groups.append(current)
    result=[]
    for g in groups:
        for i in range(0,len(g),5):result.append(g[i:i+5])
    priority=[]
    for f in [28,68,86,113,136]:priority.extend([w for w in result if f in w])
    return priority+[w for w in result if w not in priority]

def interpolate_controls(frames,knots,controls):
    result=np.tile(np.eye(4),(len(frames),2,1,1))
    for oi in range(2):
        for j in range(3):result[:,oi,j,3]=np.interp(frames,knots,controls[:,oi,j,3])
        if len(knots)>1:result[:,oi,:3,:3]=Slerp(knots,Rotation.from_matrix(controls[:,oi,:3,:3]))(frames).as_matrix()
        else:result[:,oi,:3,:3]=controls[0,oi,:3,:3]
    return result

def solve_window(frames,current,shapes,n,offset,deadline):
    knots=np.unique([frames[0],frames[len(frames)//2],frames[-1]]);base=current[knots].copy()
    observations=[FrameObservation(f,stride=4) for f in frames];clouds=[samples(s,WALL) for s in shapes]
    initial=np.stack([feasible(current[f],shapes,n,offset)[0] for f in frames]);start=time.monotonic();calls=0
    best={'loss':np.inf,'x':np.zeros(len(knots)*10),'poses':initial}
    def make(x):
        controls=np.stack([perturb(base[j],v) for j,v in enumerate(np.asarray(x).reshape(len(knots),10))])
        poses=interpolate_controls(np.array(frames),knots,controls)
        return np.stack([feasible(P,shapes,n,offset)[0] for P in poses])
    def terms(poses,x):
        images=[];details=[];coll=[]
        for obs,P in zip(observations,poses):
            score,rows=obs.score(P,shapes,WALL);images.append(score);details.append(rows);coll.append(overlap(P,shapes,WALL,clouds))
        image=float(np.mean(images));collision=15*float(np.mean([c['sampled_mean_material_penetration_m'] for c in coll]))/.001
        # Within-window acceleration. Short windows may still retain true curvature.
        temporal=0.;axis_temporal=0.
        if len(frames)>=3:
            acc=np.diff(poses[:,:,:3,3],n=2,axis=0);temporal=.025*float(np.mean(np.sum((acc/.005)**2,axis=-1)))
            axes=poses[:,:,:3,2];axis_temporal=.015*float(np.mean(np.sum((np.diff(axes,n=2,axis=0)/.05)**2,axis=-1)))
        boundary=0.
        for ii,neighbor,neighbor2 in [(0,frames[0]-1,frames[0]-2),(-1,frames[-1]+1,frames[-1]+2)]:
            if 0<=neighbor<137 and 0<=neighbor2<137 and neighbor not in HOLDOUT and neighbor2 not in HOLDOUT:
                target=2*current[neighbor,:,:3,3]-current[neighbor2,:,:3,3]
                boundary+=.01*float(np.mean(np.sum(((poses[ii,:,:3,3]-target)/.01)**2,axis=-1)))
        prior=.003*float(np.mean(np.asarray(x)**2))
        return {'total':image+collision+temporal+axis_temporal+boundary+prior,'image':image,'material':collision,'translation_acceleration':temporal,'axis_acceleration':axis_temporal,'boundary':boundary,'pose_prior':prior,'frames':[{'frame':f,'objects':rows,'material_overlap':c} for f,rows,c in zip(frames,details,coll)]}
    def objective(x):
        nonlocal calls
        if time.monotonic()>deadline:raise BudgetEnd()
        calls+=1;P=make(x);d=terms(P,x)
        if d['total']<best['loss']:best.update(loss=d['total'],x=np.asarray(x).copy(),poses=P.copy())
        return d['total']
    before=terms(initial,np.zeros(len(knots)*10));status='completed'
    try:
        objective(np.zeros(len(knots)*10))
        # The free seed may align axes about the visible rim, but axis equality
        # is never an objective or acceptance criterion.
        seed=np.zeros((len(knots),10))
        for j,P in enumerate(base):
            axis=P[1,:3,:3].T@P[0,:3,2];seed[j,8:10]=np.clip(np.array([-axis[1],axis[0]])/SCALE[8:10],-2,2)
            Q=perturb(P,seed[j]);rim=np.array([0.,0.,shapes[1]['height_m']]);seed[j,5:8]=np.clip(((P[1,:3,:3]-Q[1,:3,:3])@rim)/.01,-2,2)
        objective(seed.ravel())
        opt=minimize(objective,best['x'],method='Powell',bounds=[(-2,2)]*(10*len(knots)),options={'maxiter':6,'maxfev':850,'xtol':.025,'ftol':.0005})
        optimizer={'success':bool(opt.success),'message':str(opt.message),'nfev':int(opt.nfev)}
    except BudgetEnd:
        status='budget_stopped';optimizer={'success':False,'message':'Shared 540s optimization budget; retain best evaluated feasible candidate.'}
    candidate=best['poses'];after=terms(candidate,best['x']);reasons=[]
    if after['total']>=before['total']-1e-5:reasons.append('No window objective improvement over table-feasible baseline')
    # Compare worst and mean image quality; hard table gate is independent.
    br=[y for f in before['frames'] for y in f['objects']];ar=[y for f in after['frames'] for y in f['objects']]
    if np.mean([y['silhouette_iou'] for y in ar])<np.mean([y['silhouette_iou'] for y in br])-.025:reasons.append('Mean silhouette worsened >0.025')
    if min(y['silhouette_iou'] for y in ar)<min(y['silhouette_iou'] for y in br)-.05:reasons.append('Worst silhouette worsened >0.05')
    if min(y['coverage'] for y in ar)<min(y['coverage'] for y in br)-.05:reasons.append('Worst coverage worsened >0.05')
    bd=[y['depth_p50_p95_m'][0] for y in br if y['depth_p50_p95_m']];ad=[y['depth_p50_p95_m'][0] for y in ar if y['depth_p50_p95_m']]
    if np.median(ad)>np.median(bd)+.003:reasons.append('Median depth worsened >3mm')
    gaps=[clearance(P[oi],shapes[oi],n,offset) for P in candidate for oi in range(2)]
    if min(gaps)<-1e-7:reasons.append('Hard table feasibility failed')
    accepted=not reasons;selected=candidate if accepted else initial
    return {'frames':frames,'control_frames':knots.tolist(),'state':status,'accepted':accepted,'rejection_reasons':reasons,'before_projected':before,'after_proposed':after,'selected_world_T_canonical':selected.tolist(),'proposed_control_corrections':best['x'].tolist(),'elapsed_seconds':time.monotonic()-start,'objective_evaluations':calls,'optimizer':optimizer,'min_proposed_table_clearance_m':min(gaps)}

def audit_sequence(P,shapes,n,offset):
    table=np.array([[clearance(P[f,oi],shapes[oi],n,offset) for oi in range(2)] for f in range(137)])
    objects={}
    for oi,name in enumerate(NAMES):
        step=np.linalg.norm(np.diff(P[:,oi,:3,3],axis=0),axis=1);a=P[:,oi,:3,2];astep=np.rad2deg(np.arccos(np.clip((a[1:]*a[:-1]).sum(-1),-1,1)))
        objects[name]={'table_clearance_min_m':float(table[:,oi].min()),'min_table_clearance_frame':int(table[:,oi].argmin()),'frames_table_penetration_over_1mm':np.flatnonzero(table[:,oi]<-.001).tolist(),'translation_step_p50_p95_max_m':np.quantile(step,[.5,.95,1]).tolist(),'axis_step_p50_p95_max_deg':np.quantile(astep,[.5,.95,1]).tolist()}
    T=np.linalg.inv(P[-1,0])@P[-1,1]
    return {'objects':objects,'final_axis_angle_deg':float(np.rad2deg(np.arccos(np.clip(T[2,2],-1,1)))),'final_bottom_eccentricity_m':float(np.linalg.norm(T[:2,3])),'per_frame_table_clearance_m':table.tolist()}

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'windows').mkdir(exist_ok=True);(OUT/'previews').mkdir(exist_ok=True)
    sources={str(p.relative_to(ROOT)):digest(p) for p in [SHAPE_SOURCE/'report.json',POSE_SOURCE/'candidate_poses.npz',ROOT/'scripts/agent_cup_pose_refine.py',Path(__file__)]};key=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    source=json.loads((SHAPE_SOURCE/'report.json').read_text());shapes=[source['objects'][n]['shape'] for n in NAMES];n=np.array(source['table_plane']['normal_world']);offset=source['table_plane']['offset_m'];assert abs(np.linalg.norm(n)-1)<1e-8
    before=np.load(POSE_SOURCE/'candidate_poses.npz')['world_T_canonical_refined'];current=np.stack([feasible(P,shapes,n,offset)[0] for P in before]);projected=current.copy()
    start=time.monotonic();deadline=start+BUDGET;ordered=windows();report={'state':'running','started_utc':now(),'compute_budget_seconds':BUDGET,'source_sha256':sources,'cache_key':key,'method':'Fixed v3 cups; 3-control SE3 trajectories jointly fitted inside up-to-5-frame TRAIN windows, image/depth/collision/acceleration/boundary terms, hard table projection','training':False,'holdout_frames':sorted(HOLDOUT),'window_order':ordered,'table_plane':source['table_plane'],'table_convention':'n dot x_world + offset >= 0; n points upward toward initial camera. Source plane points were transformed by world_T_rgb[0] before fit.','hard_table_projection':'Minimum world translation along n only when the analytic full-frustum envelope penetrates; 0.2mm positive margin. No attraction/contact constraint for lifted cups.','optional_table_support_enabled':False,'wall_thickness_m':WALL,'wall_thickness_measured':False,'limitations':['Table finite extent and marker feet are not modeled.','Window joins have soft boundary regularization, not a full-sequence bundle adjustment.','Same validation windows reused for selection; not an unbiased test.','Analytic sampled cup-shell overlap is not a physics collision-backend test.','Depth confidence uses eroded masks/MAD; not independently verified stereo uncertainty.','Inner wall/base dimensions remain assumptions; infeasibility may be geometry rather than pose.']}
    save(OUT/'report.json',report);results=[]
    for wi,frames in enumerate(ordered):
        if time.monotonic()>deadline:break
        path=OUT/'windows'/f'{frames[0]:06d}_{frames[-1]:06d}.json'
        if path.exists() and json.loads(path.read_text()).get('cache_key')==key:result=json.loads(path.read_text())
        else:result=solve_window(frames,current,shapes,n,offset,deadline);result['cache_key']=key;save(path,result)
        current[frames]=np.asarray(result['selected_world_T_canonical']);results.append(result);report['window_results']=[{k:v for k,v in x.items() if k!='selected_world_T_canonical'} for x in results]
        np.savez_compressed(OUT/'checkpoint_poses.npz',world_T_canonical=current,completed_window_count=len(results));save(OUT/'report.json',report)
        print('window',frames,'accepted',result['accepted'],'loss',result['before_projected']['total'],'->',result['after_proposed']['total'],'sec',result['elapsed_seconds'],flush=True)
        if wi==4:print('KEY_WINDOW_PILOT_COMPLETE','elapsed',time.monotonic()-start,'accepted',sum(r['accepted'] for r in results),flush=True)
    train=np.array(sorted(set(range(137))-HOLDOUT));interpolated=interpolate_controls(np.arange(137),train,current[train])
    after=np.stack([feasible(P,shapes,n,offset)[0] for P in interpolated])
    np.savez_compressed(OUT/'candidate_poses.npz',frame_index=np.arange(137),object_names=NAMES,world_T_canonical_before=before,world_T_canonical_table_projected=projected,world_T_canonical_refined=after,world_rigid_delta=after@np.linalg.inv(before),heldout=np.array([f in HOLDOUT for f in range(137)]))
    evaluations=[]
    for f in range(137):
        b,bi,bc=full_metrics(f,before[f],shapes,WALL);a,ai,ac=full_metrics(f,after[f],shapes,WALL)
        evaluations.append({'frame':f,'heldout':f in HOLDOUT,'before':b,'after':a,'material_before':bc,'material_after':ac})
        if f in [0,10,28,55,60,68,86,113,124,125,136]:comparison(f,bi,ai,OUT/'previews'/f'comparison_{f:06d}.jpg')
    report['evaluation_frames']=evaluations;report['summaries']={}
    for split in ['train_in_sample','heldout_interpolated']:
        items=[x for x in evaluations if x['heldout']==(split=='heldout_interpolated')];report['summaries'][split]={side:aggregate([r for x in items for r in x[side]]) for side in ['before','after']}
    report['temporal_and_table_audit']={variant:audit_sequence(P,shapes,n,offset) for variant,P in [('before',before),('table_projected_only',projected),('after',after)]}
    report['wall_sensitivity']=[]
    for t in [.0015,.003,.005]:
        clouds=[samples(s,t) for s in shapes]
        for f in [68,86,113,136]:report['wall_sensitivity'].append({'frame':f,'wall_thickness_m':t,'material':overlap(after[f],shapes,t,clouds)})
    hm=report['summaries']['heldout_interpolated']['after'];maxover=max(x['material_after']['fraction_samples_inside_other_material'] for x in evaluations)
    mingap=min(o['table_clearance_min_m'] for o in report['temporal_and_table_audit']['after']['objects'].values())
    report['gates']={'table_nonpenetration_hard_pass':mingap>=-1e-7,'cup_material_worst_frame_pass':maxover<.01,'image_reference_pass':hm['silhouette_iou_median']>=.9 and hm['silhouette_iou_worst']>=.8 and hm['depth_p95_worst_m']<=.03 and hm['depth_render_coverage_min']>=.8,'geometric_reference_ready':False,'interpretation':'Engineering gates; no automatic promotion. Tiny gate misses do not justify unbounded reruns.'};report['gates']['geometric_reference_ready']=all(report['gates'][k] for k in ['table_nonpenetration_hard_pass','cup_material_worst_frame_pass','image_reference_pass'])
    report['material_summary']={side:{'median_mean_penetration_m':float(np.median([x[f'material_{side}']['sampled_mean_material_penetration_m'] for x in evaluations])),'worst_fraction_samples_inside':float(max(x[f'material_{side}']['fraction_samples_inside_other_material'] for x in evaluations))} for side in ['before','after']}
    report['failure_frames']={'material_worst':[{'frame':x['frame'],'fraction':x['material_after']['fraction_samples_inside_other_material']} for x in sorted(evaluations,key=lambda x:x['material_after']['fraction_samples_inside_other_material'],reverse=True)[:8]],'heldout_silhouette_worst':[dict(y,frame=x['frame']) for x in evaluations if x['heldout'] for y in x['after'] if y['silhouette_iou_occlusion_aware']<.8]}
    report.update(state='completed',completed_utc=now(),elapsed_seconds=time.monotonic()-start,completed_windows=len(results),total_windows=len(ordered),optimization_budget_exhausted=any(x['state']=='budget_stopped' for x in results))
    save(OUT/'report.json',report);print('COMPLETED',report['elapsed_seconds'],report['gates'],report['summaries'],flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        if (OUT/'report.json').exists():
            r=json.loads((OUT/'report.json').read_text());r.update(state='failed',error=traceback.format_exc(),updated_utc=now());save(OUT/'report.json',r)
        raise
