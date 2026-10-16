"""Bounded coding-agent cup pilot. Metric data, explicit priors, blocked holdouts.

AgentSTAR-inspired code geometry/render comparison; not an AgentSTAR reproduction.
No learned model downloads, GT gauge alignment, retargeting or policy training.
"""
from __future__ import annotations
import json, os, hashlib, time, traceback
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw
import trimesh

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/track3/episode_000011'
OUT=BASE/'agent_cup_reconstruction'
HOLDOUT=set(range(8,13))|set(range(58,63))|set(range(122,127))
NAMES=['blue_cup','beige_cup']
K=np.load(BASE/'moge_sample_calibrated/frame_000000/undistorted_input_K.npy')
CAM=np.load(BASE/'stereo_sequence/camera_poses.npz')['world_T_rgb']
TRACK=np.load(BASE/'object_surface_tracks/poses.npz')['world_T_reference']
THICKNESSES=[.0015,.003,.005]

def now():return datetime.now(timezone.utc).isoformat()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_frame(f):
    with np.load(BASE/f'foundation_stereo_sequence/frame_{f:06d}/prediction.npz') as z:d=z['depth_rgb']
    with np.load(BASE/f'object_masks_temporal/masks/{f:06d}.npz') as z:m=z['masks']
    with np.load(BASE/f'hand_masks_sequence/frame_{f:06d}/masks.npz') as z:h=z['masks'].any(axis=0)
    return d,m,h
def unproject(d,mask):
    y,x=np.where(mask);z=d[y,x]
    return np.column_stack(((x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z))
def cloud(f,oi,max_points=1200):
    d,m,h=load_frame(f)
    good=binary_erosion(m[oi],iterations=3)&~binary_dilation(h,iterations=2)&np.isfinite(d)&(d>.15)&(d<1.2)
    v=d[good]
    if len(v)==0:return np.empty((0,3))
    med=np.median(v);mad=np.median(abs(v-med));good &=abs(d-med)<max(.025,4*mad)
    p=unproject(d,good)
    if len(p)>max_points:p=p[np.linspace(0,len(p)-1,max_points).astype(int)]
    return p@CAM[f,:3,:3].T+CAM[f,:3,3]

def table_plane():
    # Manually bounded tabletop ROI; provenance recorded, not a semantic GT mask.
    d,m,h=load_frame(0);y,x=np.indices(d.shape)
    mask=(x>350)&(x<820)&(y>450)&(y<750)&~binary_dilation(m.any(0)|h,iterations=10)&np.isfinite(d)&(d>.2)&(d<1)
    p=unproject(d,mask);p=p@CAM[0,:3,:3].T+CAM[0,:3,3]
    p=p[::max(1,len(p)//7000)]
    rng=np.random.default_rng(11);best=np.zeros(len(p),bool)
    for _ in range(350):
        s=p[rng.choice(len(p),3,replace=False)];n=np.cross(s[1]-s[0],s[2]-s[0]);n/=max(np.linalg.norm(n),1e-12)
        inside=abs((p-s[0])@n)<.004
        if inside.sum()>best.sum():best=inside
    c=p[best].mean(0);_,_,v=np.linalg.svd(p[best]-c,full_matrices=False);n=v[-1]
    if n@(c-CAM[0,:3,3])>0:n=-n
    residual=abs((p-c)@n);good=residual<.004
    return n,float(-n@c),{'frame':0,'roi_xyxy':[350,450,820,750],'normal_world':n.tolist(),'offset_m':float(-n@c),'inlier_fraction':float(good.mean()),'inlier_residual_p50_p95_m':np.quantile(residual[good],[.5,.95]).tolist(),'manual_roi':True}

def initial_shape(oi,n,offset,train_clouds):
    ex=np.array([1.,0,0]);ex-=n*(ex@n);ex/=np.linalg.norm(ex);ey=np.cross(n,ex);basis=np.column_stack((ex,ey,n))
    ref0=TRACK[0,oi,:3,3];origin=ref0-(n@ref0+offset)*n
    # Fit only separated pre-manipulation training views; later frames are used for pose alignment.
    frames=[f for f in range(0,22,2) if f not in HOLDOUT]
    pts=[]
    for f in frames:
        ref=(train_clouds[f,oi]-TRACK[f,oi,:3,3])@TRACK[f,oi,:3,:3]
        world0=ref@TRACK[0,oi,:3,:3].T+ref0
        pts.append((world0-origin)@basis)
    frame_pts=pts;pts=np.concatenate(pts);height=float(np.median([np.quantile(p[:,2],.995) for p in frame_pts]))
    fit=pts[(pts[:,2]>.02)&(pts[:,2]<height*.78)]
    init=np.array([0.,0.,.035,.06,0.,0.]) # cx cy bottom_radius taper tilt_x tilt_y
    def residual(v):
        rot=Rotation.from_rotvec([v[4],v[5],0]).as_matrix();q=(fit-np.array([v[0],v[1],0]))@rot
        rho=np.linalg.norm(q[:,:2],axis=1);r=v[2]+v[3]*q[:,2]
        return np.r_[rho-r, np.array([v[4],v[5]])*.025]
    opt=least_squares(residual,init,bounds=([-.08,-.08,.015,-.1,-.14,-.14],[.08,.08,.065,.3,.14,.14]),loss='soft_l1',f_scale=.002,max_nfev=150)
    v=opt.x;rot=Rotation.from_rotvec([v[4],v[5],0]).as_matrix();B=basis@rot
    c=origin+basis@np.r_[v[:2],0]
    frame_heights=[float(np.quantile(((p-np.r_[v[:2],0])@rot)[:,2],.995)) for p in frame_pts]
    height=float(np.median(frame_heights))
    T=np.eye(4);T[:3,:3]=B;T[:3,3]=c
    ref_T=np.linalg.inv(TRACK[0,oi])@T
    return {'height_m':height,'bottom_outer_radius_m':float(v[2]),'taper_dr_dz':float(v[3]),'fit_frames':frames,'fit_points':len(fit),'fit_radial_p50_p95_m':np.quantile(abs(residual(v)[:len(fit)]),[.5,.95]).tolist(),'axis_tilt_from_table_deg':float(np.rad2deg(np.linalg.norm(v[4:]))),'reference_T_canonical':ref_T.tolist(),'axis_prior':'Table normal plus bounded +/-8deg x/y correction; circular symmetry assumed','scale':'Fixed calibrated stereo meters; no GT scaling','height_estimator':'Median of per-frame 99.5th axial percentiles from static TRAIN frames 0-20; avoids moving-track height contamination','training_frame_heights_m':frame_heights}

def cup_mesh(s,t,bottom=.003,segments=128):
    h=s['height_m'];r=s['bottom_outer_radius_m'];k=s['taper_dr_dz'];top=r+k*h
    # Cross section traverses outer bottom->outer top->inner top->inner bottom.
    section=np.array([[r,0],[top,h],[top-t,h],[r+k*bottom-t,bottom]])
    theta=np.arange(segments)*2*np.pi/segments
    verts=np.concatenate([np.column_stack((rr*np.cos(theta),rr*np.sin(theta),np.full(segments,zz))) for rr,zz in section])
    verts=np.vstack([verts,[0,0,0],[0,0,bottom]]);faces=[]
    for j in range(3):
        for i in range(segments):
            a=j*segments+i;b=j*segments+(i+1)%segments;c=(j+1)*segments+(i+1)%segments;d=(j+1)*segments+i
            faces.extend([[a,b,c],[a,c,d]])
    for i in range(segments):
        faces.append([4*segments,(i+1)%segments,i])
        faces.append([4*segments+1,3*segments+i,3*segments+(i+1)%segments])
    m=trimesh.Trimesh(vertices=verts,faces=faces,process=False);m.fix_normals()
    return m

def raycast(s,t,T_cam_cup,height=767,width=1024,stride=2):
    y,x=np.mgrid[0:height:stride,0:width:stride];directions=np.stack(((x-K[0,2])/K[0,0],(y-K[1,2])/K[1,1],np.ones_like(x)),axis=-1)
    R=T_cam_cup[:3,:3];o=-T_cam_cup[:3,3]@R;d=directions@R
    result=np.full(x.shape,np.inf);h=s['height_m'];r=s['bottom_outer_radius_m'];k=s['taper_dr_dz'];bottom=.003
    def side(intercept,zlo,zhi):
        A=d[...,0]**2+d[...,1]**2-(k*d[...,2])**2
        B=2*(o[0]*d[...,0]+o[1]*d[...,1]-(intercept+k*o[2])*k*d[...,2])
        C=o[0]**2+o[1]**2-(intercept+k*o[2])**2
        disc=B*B-4*A*C;sq=np.sqrt(np.maximum(disc,0));safe=np.where(abs(A)>1e-12,A,1e-12)
        for sign in [-1,1]:
            tau=(-B+sign*sq)/(2*safe);z=o[2]+tau*d[...,2]
            ok=(disc>=0)&(tau>.01)&(z>=zlo)&(z<=zhi)
            np.minimum(result,np.where(ok,tau,np.inf),out=result)
    side(r,0,h);side(r-t,bottom,h)
    for z,lo,hi in [(0,0,r),(bottom,0,r+k*bottom-t),(h,r+k*h-t,r+k*h)]:
        tau=(z-o[2])/np.where(abs(d[...,2])>1e-12,d[...,2],1e-12);xx=o[0]+tau*d[...,0];yy=o[1]+tau*d[...,1];rr=np.sqrt(xx*xx+yy*yy)
        ok=(tau>.01)&(rr>=lo)&(rr<=hi);np.minimum(result,np.where(ok,tau,np.inf),out=result)
    return result

def refine_pose(s,oi,clouds,initial):
    # Single bounded shape->pose pass. Holdout corrections interpolated from TRAIN only.
    deltas=np.zeros((137,3));records=[]
    for f in sorted(set(range(137))-HOLDOUT):
        T=initial[f];q=(clouds[f,oi]-T[:3,3])@T[:3,:3]
        if len(q)<100:continue
        h=s['height_m'];r=s['bottom_outer_radius_m'];k=s['taper_dr_dz']
        q=q[(q[:,2]>.015)&(q[:,2]<h*.8)]
        if len(q)<60:continue
        def residual(v):
            u=q-v;dist=np.linalg.norm(u[:,:2],axis=1)-(r+k*u[:,2])
            return np.r_[dist,v*.4]
        fit=least_squares(residual,np.zeros(3),bounds=(-.01,.01),loss='soft_l1',f_scale=.003,max_nfev=35)
        deltas[f]=fit.x
        records.append({'frame':f,'delta_canonical_m':fit.x.tolist(),'at_bound':bool(np.any(abs(fit.x)>.0099))})
    known=np.array(sorted(set(range(137))-HOLDOUT))
    for j in range(3):deltas[:,j]=np.interp(np.arange(137),known,deltas[known,j])
    poses=initial.copy();poses[:,:3,3]+=np.einsum('fij,fj->fi',initial[:,:3,:3],deltas)
    return poses,records,deltas

def metrics(f,oi,pred,other):
    depth,masks,hands=load_frame(f);d=depth[::2,::2];m=masks[oi,::2,::2];othermask=masks[1-oi,::2,::2]
    hand=binary_dilation(hands,iterations=2)[::2,::2]
    # Other object and hand pixels are don't-care, avoiding false silhouette penalties.
    valid=~hand&~othermask;render=np.isfinite(pred)&~(np.isfinite(other)&(other+1e-4<pred))
    union=(render|m)&valid;inter=render&m&valid
    reliable=binary_erosion(masks[oi],iterations=3)[::2,::2]&valid&np.isfinite(d)&(d>.15)&(d<1.2)
    v=d[reliable]
    if len(v):
        median=np.median(v);mad=np.median(abs(v-median));reliable &=abs(d-median)<max(.025,4*mad)
    paired=reliable&render;err=abs(pred[paired]-d[paired]);q=np.quantile(err,[.5,.95]).tolist() if len(err) else None
    return {'frame':f,'heldout':f in HOLDOUT,'silhouette_iou_occlusion_aware':float(inter.sum()/max(union.sum(),1)),'reliable_depth_pixels':int(reliable.sum()),'paired_depth_pixels':int(paired.sum()),'depth_render_coverage':float(paired.sum()/max(reliable.sum(),1)),'depth_abs_p50_p95_m':q},render,m,valid

def overlay(f,render,m,valid,dest,label):
    im=np.asarray(Image.open(BASE/f'calibrated_sequence/frames/{f:06d}.jpg').resize((512,384))).copy()[:384]
    # ray raster has 384 rows, same as resized image.
    im[render&valid]=(im[render&valid]*.4+np.array([20,170,250])*.6).astype(np.uint8)
    edge=m^binary_erosion(m);im[edge]=[255,180,30]
    pil=Image.fromarray(im);ImageDraw.Draw(pil).text((8,8),label,fill='white');pil.save(dest)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source_files=[BASE/'object_surface_tracks/poses.npz',BASE/'stereo_sequence/camera_poses.npz',BASE/'foundation_stereo_sequence/report.json',BASE/'object_masks_temporal/report.json',BASE/'hand_masks_sequence/report.json',Path(__file__)]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in source_files}
    report={'state':'running','started_utc':now(),'source_sha256':hashes,'method':'Coding-agent analytic hollow frustum, metric stereo shape fit and one bounded pose-refinement pass; AgentSTAR-inspired, not full reproduction','references':['https://github.com/makezur/agenticSTAR','https://arxiv.org/html/2609.24487'],'holdout_frames':sorted(HOLDOUT),'training_frames':sorted(set(range(137))-HOLDOUT),'assumed_wall_thickness_m':THICKNESSES,'assumed_bottom_thickness_m':.003,'training':False,'limitations':['Predicted stereo/masks/initial object tracking, no GT.','Initial tracking used the whole sequence, so holdouts test geometry fitting and TRAIN-only pose updates, not an independent upstream tracker.','Wall and bottom thickness are assumed candidates, not measured; mass density unknown.','Rotational symmetry, no marker geometry; small boundary mismatches expected.','Analytic geometry sweep is not a Newton collision-backend validation.']}
    save(OUT/'report.json',report);print('started',now(),flush=True)
    cache=OUT/'training_clouds.npz'
    # Cache per source hash avoids recomputation after interruption.
    key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if cache.exists() and (OUT/'cache.json').exists() and json.loads((OUT/'cache.json').read_text()).get('key')==key:
        z=np.load(cache);clouds={tuple(map(int,k.split('_'))):z[k] for k in z.files}
    else:
        clouds={}
        for f in report['training_frames']:
            for oi in range(2):clouds[f,oi]=cloud(f,oi)
        np.savez_compressed(cache,**{f'{f}_{oi}':v for (f,oi),v in clouds.items()});save(OUT/'cache.json',{'key':key})
    n,offset,table=table_plane();report['table_plane']=table;save(OUT/'report.json',report)
    shapes=[];initial=[];refined=[];pose_records=[];report['objects']={}
    for oi,name in enumerate(NAMES):
        shape=initial_shape(oi,n,offset,clouds);shapes.append(shape)
        poses=TRACK[:,oi]@np.asarray(shape['reference_T_canonical']);initial.append(poses)
        after,recs,delta=refine_pose(shape,oi,clouds,poses);refined.append(after);pose_records.append(recs)
        folder=OUT/name;folder.mkdir(exist_ok=True);save(folder/'parameters.json',shape)
        candidates=[]
        for t in THICKNESSES:
            m=cup_mesh(shape,t);tag=f'wall_{round(t*1000):02d}mm_{int(t*1e6)}um';m.export(folder/f'{tag}.ply');m.export(folder/f'{tag}.obj')
            _,cnt=np.unique(m.edges_sorted,axis=0,return_counts=True)
            candidates.append({'id':tag,'wall_thickness_m':t,'bottom_thickness_m':.003,'watertight':bool(m.is_watertight),'winding_consistent':bool(m.is_winding_consistent),'boundary_edges':int((cnt==1).sum()),'nonmanifold_edges':int((cnt>2).sum()),'vertices':len(m.vertices),'faces':len(m.faces),'volume_m3':float(m.volume),'uniform_material_COM_canonical_m':m.center_mass.tolist(),'COM_assumption':'Uniform shell density; actual COM unmeasured','open_mouth_radius_m':float(shape['bottom_outer_radius_m']+shape['taper_dr_dz']*shape['height_m']-t),'mesh_path':str((folder/f'{tag}.ply').relative_to(ROOT))})
        report['objects'][name]={'shape':shape,'candidates':candidates,'pose_refinement':recs};save(OUT/'report.json',report)
        print(name,'shape',shape,flush=True)
    initial=np.stack(initial,axis=1);refined=np.stack(refined,axis=1)
    np.savez_compressed(OUT/'candidate_poses.npz',frame_index=np.arange(137),object_names=np.array(NAMES),world_T_canonical_initial=initial,world_T_canonical_refined=refined,reference_T_canonical=np.array([s['reference_T_canonical'] for s in shapes]),heldout=np.array([f in HOLDOUT for f in range(137)]))
    evaluations=[];frames=sorted(HOLDOUT|{0,20,35,45,68,85,105,136});preview=OUT/'previews';preview.mkdir(exist_ok=True)
    for variant,poses in [('initial',initial),('train_refined',refined)]:
        for t in THICKNESSES:
            rows=[]
            for f in frames:
                pred=[raycast(shapes[oi],t,np.linalg.inv(CAM[f])@poses[f,oi]) for oi in range(2)]
                for oi in range(2):
                    stat,r,m,v=metrics(f,oi,pred[oi],pred[1-oi]);stat['object']=NAMES[oi];rows.append(stat)
                    if t==.003 and f in [0,10,60,68,124,136]:overlay(f,r,m,v,preview/f'{variant}_{NAMES[oi]}_{f:06d}.jpg',f'{variant} {NAMES[oi]} f={f}; cyan model, amber mask')
            held=[r for r in rows if r['heldout']];good=[r for r in held if r['depth_abs_p50_p95_m']]
            summary={'variant':variant,'wall_thickness_m':t,'heldout_frame_object_count':len(held),'median_heldout_iou':float(np.median([r['silhouette_iou_occlusion_aware'] for r in held])),'median_heldout_depth_p50_m':float(np.median([r['depth_abs_p50_p95_m'][0] for r in good])),'median_heldout_depth_p95_m':float(np.median([r['depth_abs_p50_p95_m'][1] for r in good])),'min_heldout_render_coverage':float(min(r['depth_render_coverage'] for r in held)),'frames':rows}
            summary['proposed_gate_pass']=bool(summary['median_heldout_iou']>=.90 and summary['median_heldout_depth_p50_m']<=.005 and summary['median_heldout_depth_p95_m']<=.015 and summary['min_heldout_render_coverage']>=.9)
            evaluations.append(summary);report['evaluations']=evaluations;save(OUT/'report.json',report);print('evaluation',{k:v for k,v in summary.items() if k!='frames'},flush=True)
    # Coaxial cavity clearance sweep, both thickness hypotheses and analytic surfaces.
    sweep=[];blue,beige=shapes
    for t in THICKNESSES:
        for insertion in np.linspace(0,min(blue['height_m'],beige['height_m']),101):
            zbottom=blue['height_m']-insertion;z=np.linspace(max(.003,zbottom),min(blue['height_m'],zbottom+beige['height_m']),150)
            gap=blue['bottom_outer_radius_m']+blue['taper_dr_dz']*z-t-(beige['bottom_outer_radius_m']+beige['taper_dr_dz']*(z-zbottom))
            sweep.append({'wall_thickness_m':t,'insertion_m':float(insertion),'min_radial_clearance_m':float(gap.min()),'bottom_clearance_m':float(zbottom-.003),'admissible':bool(gap.min()>=0 and zbottom>=.003)})
    save(OUT/'coaxial_insertion_sweep.json',{'assumption':'Coaxial analytic frusta; excludes attached markers and backend contact margins. Not measured insertion success.','samples':sweep})
    report['insertion_sweep']={'path':'coaxial_insertion_sweep.json','max_admissible_insertion_by_wall_m':{str(t):max([r['insertion_m'] for r in sweep if r['wall_thickness_m']==t and r['admissible']],default=0.) for t in THICKNESSES}}
    report['state']='completed';report['completed_utc']=now();report['quality_pass']=any(e['proposed_gate_pass'] for e in evaluations)
    report['next_action']='Inspect failure overlays; refine rim/axis/pose using training frames before any contact or policy training.' if not report['quality_pass'] else 'Validate physical collision assets and uncertain inner geometry before contact extraction.'
    save(OUT/'report.json',report);print('completed quality_pass',report['quality_pass'],flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        if (OUT/'report.json').exists():
            r=json.loads((OUT/'report.json').read_text());r.update(state='failed',error=traceback.format_exc(),updated_utc=now());save(OUT/'report.json',r)
        raise
