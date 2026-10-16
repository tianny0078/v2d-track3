"""Ten-minute CPU pose pilot: fixed hollow cups, render/depth/collision/temporal loss.

Only explicit TRAIN keyframes are optimized. Heldout corrections are interpolated
from those fitted keys. This is an AgentSTAR-inspired numerical pose search, not
the upstream AgentSTAR harness, and is not independent of upstream RGB-D tracking.
"""
from __future__ import annotations
import json,time,hashlib,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scipy.ndimage import binary_dilation,binary_erosion,distance_transform_edt
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation,Slerp
from PIL import Image,ImageDraw
from agent_cup_reconstruct import BASE,K,CAM,load_frame,save,NAMES,HOLDOUT,raycast,metrics,ROOT

OUT=BASE/'agent_cup_pose_refinement'
SOURCE=BASE/'agent_cup_reconstruction'
KEYFRAMES=sorted(set(range(137))-HOLDOUT)
NOMINAL_WALL=.003
BUDGET_SECONDS=540
SCALE=np.tile([.01,.01,.01,np.deg2rad(5),np.deg2rad(5)],2)
BOUNDS=[(-2,2)]*10
WEIGHTS={'silhouette_iou':3.,'silhouette_distance':.4,'depth':.5,'coverage':1.,'material_overlap':10.,'pose_prior':.005,'temporal_correction':.02}

def now():return datetime.now(timezone.utc).isoformat()
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def perturb(initial,x):
    result=initial.copy();v=np.asarray(x).reshape(2,5)*SCALE.reshape(2,5)
    for oi in range(2):
        result[oi,:3,3]+=v[oi,:3]
        result[oi,:3,:3]=initial[oi,:3,:3]@Rotation.from_rotvec(np.r_[v[oi,3:],0]).as_matrix()
    return result

def analytic_rays(shape,t,T,directions):
    R=T[:3,:3];o=-T[:3,3]@R;d=directions@R
    result=np.full(len(d),np.inf);h=shape['height_m'];r=shape['bottom_outer_radius_m'];k=shape['taper_dr_dz']
    A=d[:,0]**2+d[:,1]**2-(k*d[:,2])**2;safe=np.where(abs(A)>1e-12,A,1e-12)
    for intercept,zlo in [(r,0),(r-t,.003)]:
        B=2*(o[0]*d[:,0]+o[1]*d[:,1]-(intercept+k*o[2])*k*d[:,2]);C=o[0]**2+o[1]**2-(intercept+k*o[2])**2
        disc=B*B-4*A*C;sq=np.sqrt(np.maximum(disc,0))
        for sign in [-1,1]:
            tau=(-B+sign*sq)/(2*safe);z=o[2]+tau*d[:,2]
            good=(disc>=0)&(tau>.01)&(z>=zlo)&(z<=h)
            np.minimum(result,np.where(good,tau,np.inf),out=result)
    for z,lo,hi in [(0,0,r),(.003,0,r+k*.003-t),(h,r+k*h-t,r+k*h)]:
        tau=(z-o[2])/np.where(abs(d[:,2])>1e-12,d[:,2],1e-12);rr=np.linalg.norm(o[:2]+tau[:,None]*d[:,:2],axis=1)
        good=(tau>.01)&(rr>=lo)&(rr<=hi);np.minimum(result,np.where(good,tau,np.inf),out=result)
    return result

def samples(shape,t):
    z=np.linspace(.001,shape['height_m']-.001,24);a=np.arange(64)*2*np.pi/64;r=shape['bottom_outer_radius_m']+shape['taper_dr_dz']*z
    clouds=[]
    for rad in [r,r-t]:clouds.append(np.stack([rad[:,None]*np.cos(a),rad[:,None]*np.sin(a),np.broadcast_to(z[:,None],(24,64))],axis=-1).reshape(-1,3))
    for zz in [.0005,.0025]:
        rad=np.linspace(0,shape['bottom_outer_radius_m'],10)
        clouds.append(np.stack([rad[:,None]*np.cos(a),rad[:,None]*np.sin(a),np.full((10,64),zz)],axis=-1).reshape(-1,3))
    return np.concatenate(clouds)

def inside_material(q,s,t):
    z=q[:,2];rho=np.linalg.norm(q[:,:2],axis=1);r=s['bottom_outer_radius_m']+s['taper_dr_dz']*z
    inside=(z>0)&(z<s['height_m'])&(rho<r)&((z<.003)|(rho>r-t))
    outer=(r-rho)/np.sqrt(1+s['taper_dr_dz']**2);inner=(rho-(r-t))/np.sqrt(1+s['taper_dr_dz']**2)
    cavity=np.where(rho<r-t,.003-z,np.maximum(inner,.003-z))
    dist=np.minimum.reduce([z,s['height_m']-z,outer,cavity])
    return np.where(inside,np.maximum(dist,0),0)

def overlap(poses,shapes,t,clouds):
    p=[]
    for oi in range(2):
        j=1-oi;T=np.linalg.inv(poses[j])@poses[oi];q=clouds[oi]@T[:3,:3].T+T[:3,3]
        p.append(inside_material(q,shapes[j],t))
    p=np.concatenate(p)
    return {'sampled_mean_material_penetration_m':float(p.mean()),'sampled_max_material_penetration_m':float(p.max()),'fraction_samples_inside_other_material':float((p>1e-5).mean()),'sample_count':len(p),'samples_inside':int((p>1e-5).sum())}

class FrameObservation:
    def __init__(self,f,stride=4):
        self.f=f;d,m,h=load_frame(f);y,x=np.where(m.any(0));self.bounds=[max(0,x.min()-60),max(0,y.min()-60),min(1024,x.max()+61),min(767,y.max()+61)]
        x0,y0,x1,y1=self.bounds;yy,xx=np.mgrid[y0:y1:stride,x0:x1:stride];self.shape=yy.shape
        self.rays=np.column_stack(((xx.ravel()-K[0,2])/K[0,0],(yy.ravel()-K[1,2])/K[1,1],np.ones(xx.size)))
        self.depth=d[yy,xx].ravel();self.mask=m[:,yy,xx].reshape(2,-1);self.hand=binary_dilation(h,iterations=2)[yy,xx].ravel();self.valid=[];self.reliable=[];self.dist=[]
        for oi in range(2):
            valid=~self.hand&~self.mask[1-oi];reliable=binary_erosion(m[oi],iterations=3)[yy,xx].ravel()&valid&np.isfinite(self.depth)&(self.depth>.15)&(self.depth<1.2)
            z=self.depth[reliable]
            if len(z):
                med=np.median(z);mad=np.median(abs(z-med));reliable &=abs(self.depth-med)<max(.025,4*mad)
            mask=self.mask[oi].reshape(self.shape);dt=distance_transform_edt(~mask)+distance_transform_edt(mask)
            self.dist.append(np.minimum(dt.ravel()*stride/8.,3));self.valid.append(valid);self.reliable.append(reliable)
        self.camera_inv=np.linalg.inv(CAM[f])

    def score(self,poses,shapes,t):
        pred=[analytic_rays(shapes[i],t,self.camera_inv@poses[i],self.rays) for i in range(2)];terms=[];rows=[]
        for oi in range(2):
            m=self.mask[oi];valid=self.valid[oi];render=np.isfinite(pred[oi])&~(np.isfinite(pred[1-oi])&(pred[1-oi]+1e-4<pred[oi]));union=(render|m)&valid;paired=self.reliable[oi]&render
            iou=float((render&m&valid).sum()/max(union.sum(),1));coverage=float(paired.sum()/max(self.reliable[oi].sum(),1));err=abs(pred[oi][paired]-self.depth[paired])
            # Pixel mismatch distance and bounded depth residual cannot replace coverage.
            chamfer=float((self.dist[oi]*((render^m)&valid)).sum()/max(union.sum(),1))
            depth=float(np.minimum(err/.005,4).mean()) if len(err) else 4.
            terms.append(WEIGHTS['silhouette_iou']*(1-iou)+WEIGHTS['silhouette_distance']*chamfer+WEIGHTS['depth']*depth+WEIGHTS['coverage']*(1-coverage))
            rows.append({'object':NAMES[oi],'silhouette_iou':iou,'coverage':coverage,'depth_p50_p95_m':np.quantile(err,[.5,.95]).tolist() if len(err) else None,'depth_pixels':len(err)})
        return float(np.mean(terms)),rows

class BudgetEnd(Exception):pass

def optimize_frame(f,initial,shapes,previous,deadline):
    obs=FrameObservation(f);clouds=[samples(s,NOMINAL_WALL) for s in shapes];calls=0;best={'x':np.zeros(10),'loss':np.inf};start=time.monotonic()
    def objective(x,details=False):
        nonlocal calls
        if not details and time.monotonic()>deadline:raise BudgetEnd()
        calls+=1;poses=perturb(initial,x);image_loss,im=obs.score(poses,shapes,NOMINAL_WALL);coll=overlap(poses,shapes,NOMINAL_WALL,clouds)
        collision=WEIGHTS['material_overlap']*coll['sampled_mean_material_penetration_m']/.001
        prior=WEIGHTS['pose_prior']*float(np.mean(np.asarray(x)**2))
        temporal=0.
        if previous is not None:
            pf,px=previous;temporal=WEIGHTS['temporal_correction']*float(np.mean((np.asarray(x)-px)**2))*min(1.,20/max(f-pf,1))
        loss=image_loss+collision+prior+temporal
        if loss<best['loss']:best.update(x=np.array(x).copy(),loss=loss)
        if details:return {'total':loss,'image_loss':image_loss,'collision_loss':collision,'pose_prior_loss':prior,'temporal_correction_loss':temporal,'objects':im,'material_overlap':coll}
        return loss
    before=objective(np.zeros(10),True);status='completed'
    try:
        if previous is not None:objective(previous[1])
        # Axis alignment is only a free-search initialization, not a target or
        # acceptance condition. Rotate around the beige rim to preserve its location.
        local_axis=initial[1,:3,:3].T@initial[0,:3,2]
        tilt=np.array([-local_axis[1],local_axis[0]])/SCALE[8:10]
        for fraction in [.5,1.]:
            seed=best['x'].copy();seed[8:10]=np.clip(fraction*tilt,-2,2)
            posed=perturb(initial,seed);rim=np.array([0.,0.,shapes[1]['height_m']])
            shift=(initial[1,:3,:3]-posed[1,:3,:3])@rim
            seed[5:8]=np.clip(shift/.01,-2,2);objective(seed)
        # Powell handles coarse-raster visibility changes without false gradients.
        result=minimize(objective,best['x'],method='Powell',bounds=BOUNDS,options={'maxiter':10,'maxfev':1000,'xtol':.015,'ftol':.0003})
        optimizer={'success':bool(result.success),'message':str(result.message),'nfev':int(result.nfev)}
    except BudgetEnd:
        status='budget_stopped';optimizer={'success':False,'message':'Shared 600-second budget reached; best evaluated candidate retained.'}
    x=best['x'];after=objective(x,True)
    # Multi-criterion TRAIN acceptance; no holdout data choose these corrections.
    reasons=[]
    if after['total']>=before['total']-1e-5:reasons.append('No joint-objective improvement')
    for b,a in zip(before['objects'],after['objects']):
        if a['silhouette_iou']<b['silhouette_iou']-.025:reasons.append(a['object']+' silhouette worsened >0.025')
        if a['coverage']<b['coverage']-.04:reasons.append(a['object']+' coverage worsened >0.04')
        if a['depth_p50_p95_m'] and b['depth_p50_p95_m'] and a['depth_p50_p95_m'][0]>b['depth_p50_p95_m'][0]+.002:reasons.append(a['object']+' median depth worsened >2mm')
    if after['material_overlap']['sampled_mean_material_penetration_m']>before['material_overlap']['sampled_mean_material_penetration_m']+.00005:reasons.append('Mean sampled material overlap worsened >0.05mm')
    accepted=not reasons
    return {'frame':f,'state':status,'training_in_sample':True,'accepted':accepted,'rejection_reasons':reasons,'x_proposed':x.tolist(),'x_selected':x.tolist() if accepted else [0.]*10,'parameter_units':'xyz world translation / 0.01m, local x/y tilt / 5deg; yaw fixed','before':before,'after_proposed':after,'after_selected':after if accepted else before,'elapsed_seconds':time.monotonic()-start,'objective_evaluations':calls,'optimizer':optimizer,'raster_bounds_xyxy':list(map(int,obs.bounds)),'raster_stride':4}

def full_metrics(f,poses,shapes,t):
    pred=[raycast(shapes[oi],t,np.linalg.inv(CAM[f])@poses[oi]) for oi in range(2)];rows=[];rasters=[]
    for oi in range(2):
        row,render,m,valid=metrics(f,oi,pred[oi],pred[1-oi]);row['object']=NAMES[oi];rows.append(row);rasters.append((render,m,valid))
    coll=overlap(poses,shapes,t,[samples(s,t) for s in shapes])
    return rows,rasters,coll

def comparison(f,before,after,dest):
    rgb=np.asarray(Image.open(BASE/f'calibrated_sequence/frames/{f:06d}.jpg').resize((512,384)))
    output=Image.new('RGB',(1024,768),'white')
    for column,rasters in enumerate([before,after]):
        for oi,(render,m,valid) in enumerate(rasters):
            im=rgb.copy();im[render&valid]=(im[render&valid]*.45+np.array([30,160,250])*.55).astype(np.uint8);im[m^binary_erosion(m)]=[255,190,20]
            image=Image.fromarray(im);ImageDraw.Draw(image).text((8,8),f'{"before" if column==0 else "after"} f={f} {NAMES[oi]}',fill='white');output.paste(image,(column*512,oi*384))
    output.save(dest)

def aggregate(rows):
    valid=[r for r in rows if r['depth_abs_p50_p95_m']]
    return {'frame_object_count':len(rows),'silhouette_iou_median':float(np.median([r['silhouette_iou_occlusion_aware'] for r in rows])),'silhouette_iou_worst':float(min(r['silhouette_iou_occlusion_aware'] for r in rows)),'depth_p50_median_m':float(np.median([r['depth_abs_p50_p95_m'][0] for r in valid])),'depth_p95_median_m':float(np.median([r['depth_abs_p50_p95_m'][1] for r in valid])),'depth_p95_worst_m':float(max(r['depth_abs_p50_p95_m'][1] for r in valid)),'depth_render_coverage_min':float(min(r['depth_render_coverage'] for r in rows))}

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'keyframes').mkdir(exist_ok=True);(OUT/'previews').mkdir(exist_ok=True)
    sources={str(p.relative_to(ROOT)):digest(p) for p in [SOURCE/'report.json',SOURCE/'candidate_poses.npz',Path(__file__),ROOT/'scripts/agent_cup_reconstruct.py']}
    key=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest();src=json.loads((SOURCE/'report.json').read_text());shapes=[src['objects'][n]['shape'] for n in NAMES]
    initial=np.load(SOURCE/'candidate_poses.npz')['world_T_canonical_initial'];start=time.monotonic();deadline=start+BUDGET_SECONDS
    report={'state':'running','started_utc':now(),'compute_budget_seconds':BUDGET_SECONDS,'source_sha256':sources,'cache_key':key,'method':'Fixed-v3 metric hollow cups; bounded 10-DOF paired Powell pose search with render/depth/material-overlap/temporal-correction objective','shape_frozen':True,'nominal_wall_thickness_m':NOMINAL_WALL,'wall_thickness_measured':False,'training_keyframes':KEYFRAMES,'holdout_frames':sorted(HOLDOUT),'objective_weights':WEIGHTS,'training':False,'pose_parameter_bounds':{'translation_per_world_axis_m':[-.02,.02],'tilt_per_local_axis_degrees':[-10,10],'yaw':'fixed (axisymmetric cup)'},'limitations':['Not upstream AgentSTAR reproduction.','Initial poses were computed by upstream tracker using all frames; holdout isolates this pose-update fitting only.','Temporal term regularizes correction changes between training keys, not a full temporal bundle adjustment.','Analytic sampled material overlap is not complete collision detection or Newton validation.','Geometry, wall thickness, masks and learned stereo carry uncertainty.','No force/retarget/policy training; no automatic promotion.']}
    save(OUT/'report.json',report);previous=None;results=[]
    for f in KEYFRAMES:
        if time.monotonic()>deadline:break
        path=OUT/'keyframes'/f'{f:06d}.json'
        if path.exists() and json.loads(path.read_text()).get('cache_key')==key:result=json.loads(path.read_text())
        else:
            result=optimize_frame(f,initial[f],shapes,previous,deadline);result['cache_key']=key;save(path,result)
        results.append(result);previous=(f,np.array(result['x_selected']));report['keyframe_results']=results;save(OUT/'report.json',report)
        print('keyframe',f,'accepted',result['accepted'],'loss',result['before']['total'],'->',result['after_proposed']['total'],'seconds',result['elapsed_seconds'],'evaluations',result['objective_evaluations'],flush=True)
    if not results:raise RuntimeError('No optimized/cached keyframes')
    frames=np.array([r['frame'] for r in results]);values=np.array([r['x_selected'] for r in results]);correction=np.column_stack([np.interp(np.arange(137),frames,values[:,j]) for j in range(10)])
    refined=np.stack([perturb(initial[f],correction[f]) for f in range(137)])
    # Holdout poses use only neighboring fitted TRAIN absolute poses, eliminating
    # inherited within-gap tracker jitter. No heldout image/depth enters this step.
    for oi in range(2):
        for j in range(3):refined[:,oi,j,3]=np.interp(np.arange(137),frames,refined[frames,oi,j,3])
        refined[:,oi,:3,:3]=Slerp(frames,Rotation.from_matrix(refined[frames,oi,:3,:3]))(np.clip(np.arange(137),frames[0],frames[-1])).as_matrix()
    np.savez_compressed(OUT/'candidate_poses.npz',frame_index=np.arange(137),object_names=np.array(NAMES),world_T_canonical_initial=initial,world_T_canonical_refined=refined,normalized_correction=correction,directly_optimized=np.array([f in frames for f in range(137)]),heldout=np.array([f in HOLDOUT for f in range(137)]))
    # Full stride-2 validation occurs after all TRAIN candidate selection.
    allrows=[];evalframes=sorted(HOLDOUT|set(frames.tolist()));sens=[]
    for f in evalframes:
        b,bi,bc=full_metrics(f,initial[f],shapes,.003);a,ai,ac=full_metrics(f,refined[f],shapes,.003)
        allrows.append({'frame':f,'heldout':f in HOLDOUT,'direct_optimization_in_sample':f in frames,'before':b,'after':a,'material_before':bc,'material_after':ac})
        if f in {0,20,55,68,115,136,10,60,124,125}:comparison(f,bi,ai,OUT/'previews'/f'comparison_{f:06d}.jpg')
    for t in [.0015,.003,.005]:
        for f in [68,136]:
            clouds=[samples(s,t) for s in shapes];sens.append({'wall_thickness_m':t,'frame':f,'before':overlap(initial[f],shapes,t,clouds),'after':overlap(refined[f],shapes,t,clouds)})
    report['wall_thickness_overlap_sensitivity']=sens;report['evaluation_frames']=allrows
    report['five_keyframe_pilot']={'directory':'five_keyframe_pilot','elapsed_seconds':13.45355859119445,'reason_for_extension':'Sparse correction interpolation retained upstream within-gap jitter; stronger sampled collision term and rim-preserving axis seed explored as hypotheses.'}
    report['heldout_pose_method']='Linear world translation and Slerp rotation of neighboring optimized TRAIN absolute poses; no direct heldout optimization.'
    report['holdout_status']='Reused selection-validation windows, not a fresh unbiased test set.'
    report['summaries']={}
    for split in ['train_in_sample','heldout_interpolated']:
        items=[x for x in allrows if x['heldout']==(split=='heldout_interpolated')]
        report['summaries'][split]={side:aggregate([r for x in items for r in x[side]]) for side in ['before','after']}
    hb=report['summaries']['heldout_interpolated']['before'];ha=report['summaries']['heldout_interpolated']['after'];end=next(x for x in allrows if x['frame']==136) if 136 in frames else None
    report['gates']={'proposal_heldout_improved':bool(ha['silhouette_iou_median']>=hb['silhouette_iou_median'] and ha['depth_p50_median_m']<=hb['depth_p50_median_m']+.001),'geometric_reference_ready':bool(ha['silhouette_iou_median']>=.90 and ha['silhouette_iou_worst']>=.80 and ha['depth_p95_worst_m']<=.03 and ha['depth_render_coverage_min']>=.80 and end is not None and end['material_after']['fraction_samples_inside_other_material']<.01),'threshold_interpretation':'Engineering proposal only; averages, worst cases and full frame results reported. Small gate misses are diagnostic, not grounds for unbounded optimization.'}
    report.update(state='completed',completed_utc=now(),elapsed_seconds=time.monotonic()-start,optimization_budget_exhausted=any(r['state']=='budget_stopped' for r in results),completed_training_keyframes=frames.tolist())
    save(OUT/'report.json',report);print('completed',report['elapsed_seconds'],'gates',report['gates'],'summaries',report['summaries'],flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        if (OUT/'report.json').exists():
            r=json.loads((OUT/'report.json').read_text());r.update(state='failed',error=traceback.format_exc(),updated_utc=now());save(OUT/'report.json',r)
        raise
