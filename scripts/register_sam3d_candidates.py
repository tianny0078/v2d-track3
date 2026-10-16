"""Bounded metric SAM3D registration; generation-frame fit, separate pose propagation audit.

No GT input. Geometry is fixed except one uniform metric-scale correction. Masks,
depth and tracker poses are predictions. Numerical render fitting is not SAM3D
training or an AgentSTAR reproduction.
"""
import os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
import cv2,numpy as np,pyrender,trimesh
from scipy.ndimage import binary_erosion,binary_dilation,distance_transform_edt
from scipy.optimize import least_squares,minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/track3/episode_000011'
SOURCE=BASE/'object_meshes_sam3d_local';OUT=BASE/'sam3d_metric_registration'
NAMES=['blue_cup','beige_cup'];K=np.load(BASE/'moge_sample_calibrated/frame_000000/undistorted_input_K.npy')
CAM=np.load(BASE/'stereo_sequence/camera_poses.npz')['world_T_rgb']
TRACK=np.load(BASE/'object_surface_tracks/poses.npz')['world_T_reference']
GEOM=json.loads((BASE/'agent_cup_reconstruction/report.json').read_text())
TABLE_N=np.array(GEOM['table_plane']['normal_world']);TABLE_D=GEOM['table_plane']['offset_m']
SCALE=np.array([.01,.01,.01,*([np.deg2rad(5)]*3),.05]);BOUNDS=([-6]*3+[-5]*3+[-5],[6]*3+[5]*3+[5])

def save(p,r):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def observation(f,oi):
    d=np.load(BASE/f'foundation_stereo_sequence/frame_{f:06d}/prediction.npz')['depth_rgb']
    masks=np.load(BASE/f'object_masks_temporal/masks/{f:06d}.npz')['masks'].astype(bool)
    hand=np.load(BASE/f'hand_masks_sequence/frame_{f:06d}/masks.npz')['masks'].any(0)
    good=binary_erosion(masks[oi],iterations=3)&~binary_dilation(hand,iterations=2)&np.isfinite(d)&(d>.15)&(d<1.2)
    z=d[good];med=np.median(z);mad=np.median(abs(z-med));good &=abs(d-med)<max(.025,4*mad)
    yy,xx=np.where(good);z=d[yy,xx];p=np.c_[(xx-K[0,2])*z/K[0,0],(yy-K[1,2])*z/K[1,1],z]
    if len(p)>4000:p=p[np.linspace(0,len(p)-1,4000).astype(int)]
    return {'points':p@CAM[f,:3,:3].T+CAM[f,:3,3], 'mask':masks[oi,::2,::2],
            'valid':(~binary_dilation(hand,iterations=2)&~masks[1-oi])[::2,::2],
            'reliable':good[::2,::2], 'depth':d[::2,::2]}

def posed(base,x):
    v=np.asarray(x)*SCALE;T=base.copy();T[:3,:3]=Rotation.from_rotvec(v[3:6]).as_matrix()@base[:3,:3]*np.exp(v[6]);T[:3,3]+=v[:3];return T

def icp(base,points,tree,surface):
    x=np.zeros(7);rows=[]
    for _ in range(16):
        T=posed(base,x);s=np.cbrt(np.linalg.det(T[:3,:3]));q=(points-T[:3,3])@T[:3,:3]/(s*s)
        dist,idx=tree.query(q,workers=4);cut=max(.003/s,min(.035/s,np.quantile(dist,.80)));good=dist<cut
        target=points[good];matched=surface[idx[good]]
        if len(target)<60:break
        def residual(v):
            mat=posed(base,v);pred=matched@mat[:3,:3].T+mat[:3,3]
            return np.r_[(pred-target).ravel(),v[3:6]*.0002,v[6]*.0005]
        opt=least_squares(residual,x,bounds=BOUNDS,loss='soft_l1',f_scale=.003,max_nfev=18)
        rows.append({'pairs':len(target),'trimmed_distance_median_m':float(np.median(dist[good])*s)})
        if np.linalg.norm(opt.x-x)<.002:x=opt.x;break
        x=opt.x
    return posed(base,x),rows

class RenderScore:
    def __init__(self,mesh,oi):
        self.renderer=pyrender.OffscreenRenderer(512,384)
        self.scene=pyrender.Scene(bg_color=[0,0,0,0],ambient_light=[.7,.7,.7])
        self.node=self.scene.add(pyrender.Mesh.from_trimesh(mesh,smooth=True))
        view=np.diag([1.,-1.,-1.,1.]);self.scene.add(pyrender.IntrinsicsCamera(K[0,0]/2,K[1,1]/2,K[0,2]/2,K[1,2]/2,znear=.05,zfar=3),pose=view)
        self.scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2),pose=view)
        self.oi=oi;self.observation={};self.vertices=np.asarray(mesh.vertices)[::max(1,len(mesh.vertices)//7000)]

    def evaluate(self,T,f=0,return_image=False):
        if f not in self.observation:self.observation[f]=observation(f,self.oi)
        o=self.observation[f];self.scene.set_pose(self.node,np.linalg.inv(CAM[f])@T)
        rgba,depth=self.renderer.render(self.scene);pred=depth>0;valid=o['valid'];mask=o['mask'];union=(pred|mask)&valid
        iou=float((pred&mask&valid).sum()/max(union.sum(),1));paired=o['reliable']&pred
        err=abs(depth[paired]-o['depth'][paired]);coverage=float(paired.sum()/max(o['reliable'].sum(),1))
        q=np.quantile(err,[.5,.95]).tolist() if len(err) else None
        world=self.vertices@T[:3,:3].T+T[:3,3];tableheight=world@TABLE_N+TABLE_D;low=float(np.quantile(tableheight,.001))
        loss=3*(1-iou)+.6*(np.minimum(err/.005,5).mean() if len(err) else 5)+(1-coverage)
        # First frame both cups stand on table. This support prior is not applied
        # to subsequent manipulated frames in the independent propagation audit.
        if f==0:loss+=4*max(0,-low-.001)/.005+.2*min(abs(low)/.005,4)
        result={'frame':f,'object':NAMES[self.oi],'silhouette_iou_occlusion_aware':iou,'depth_abs_p50_p95_m':q,
                'depth_render_coverage':coverage,'paired_depth_pixels':int(paired.sum()),'near_lowest_surface_table_height_m':low,
                'objective':float(loss),'in_sample':f==0}
        return (result,rgba,pred) if return_image else result

def main():
    OUT.mkdir(exist_ok=True);start=time.monotonic();deadline=start+540
    report={'state':'running','started_utc':datetime.now(timezone.utc).isoformat(),'method':'Fixed SAM3D mesh; trimmed stereo-point similarity ICP followed by bounded depth/silhouette/table render search',
            'fit_frames':[0],'propagation_audit_frames':[8,10,12,60,124,136],'objects':{},'training':False,
            'limitations':['Generation frame 0 fitted in sample; later frames audit shape plus inherited tracker, not independent GT.',
                           'One uniform scale correction is fitted from calibrated stereo meters, no GT scale.',
                           'Surface thickness and rear geometry remain model predictions; watertight does not establish physical correctness.',
                           'Table support assumption only for the visibly resting generation frame.']}
    save(OUT/'report.json',report);poses=[]
    for oi,name in enumerate(NAMES):
        output=OUT/name;output.mkdir(exist_ok=True);path=output/'result.json'
        mesh=trimesh.load(SOURCE/name/'mesh.glb',force='mesh',process=False)
        source_hash=hashlib.sha256((SOURCE/name/'mesh.glb').read_bytes()+Path(__file__).read_bytes()).hexdigest()
        if path.exists() and json.loads(path.read_text()).get('source_hash')==source_hash:
            result=json.loads(path.read_text());poses.append(np.array(result['selected_world_T_canonical_with_scale']));report['objects'][name]=result;continue
        tr=json.loads((SOURCE/name/'transform.json').read_text());T=np.eye(4);T[:3,:3]=Rotation.from_quat(tr['rotation'],scalar_first=True).as_matrix()@np.diag(tr['scale']);T[:3,3]=tr['translation'];T=CAM[0]@T
        np.random.seed(42+oi);surface,_=trimesh.sample.sample_surface(mesh,160000);surface=np.r_[surface,mesh.vertices[::4]];tree=cKDTree(surface);obs=observation(0,oi)
        seeds=[T];aligned=T.copy();delta=Rotation.align_vectors(TABLE_N[None],(T[:3,1]/np.linalg.norm(T[:3,1]))[None])[0].as_matrix();aligned[:3,:3]=delta@T[:3,:3];seeds.append(aligned)
        scorer=RenderScore(mesh,oi);initial=scorer.evaluate(T);candidates=[]
        for j,seed in enumerate(seeds):
            fit,history=icp(seed,obs['points'],tree,surface);score=scorer.evaluate(fit)
            candidates.append((fit,score));save(output/f'icp_seed_{j}.json',{'history':history,'score':score,'world_T_canonical_with_scale':fit.tolist()})
            print(name,'ICP',j,score,flush=True)
        candidates.append((T,initial));base,score=min(candidates,key=lambda pair:pair[1]['objective']);best={'matrix':base.copy(),'score':score,'x':np.zeros(7)};calls=0
        def objective(x):
            nonlocal calls
            if time.monotonic()>deadline:raise TimeoutError('Shared 540 second budget exhausted')
            mat=posed(base,x);r=scorer.evaluate(mat);calls+=1;loss=r['objective']+.002*float(np.mean(np.asarray(x)**2))
            if loss<best['score']['objective']:best.update(matrix=mat,score=r,x=np.array(x).copy())
            return loss
        status='completed'
        try:minimize(objective,np.zeros(7),method='Powell',bounds=list(zip(*BOUNDS)),options={'maxfev':520,'maxiter':8,'xtol':.015,'ftol':.0003})
        except TimeoutError:status='budget_stopped'
        selected=best['matrix'];final,rgba,pred=scorer.evaluate(selected,return_image=True)
        rgb=cv2.imread(str(BASE/'calibrated_sequence/frames/000000.jpg'));rgb=cv2.resize(rgb,(512,384));rgb[pred]=(rgb[pred]*.4+cv2.cvtColor(rgba[:,:,:3],cv2.COLOR_RGB2BGR)[pred]*.6).astype(np.uint8)
        cv2.imwrite(str(output/'frame0_registered_overlay.jpg'),rgb)
        ref_T=np.linalg.inv(TRACK[0,oi])@selected;sequence=TRACK[:,oi]@ref_T
        audit=[scorer.evaluate(sequence[f],f) for f in report['propagation_audit_frames']];scorer.renderer.delete()
        result={'state':status,'source_hash':source_hash,'initial':initial,'selected':final,'evaluations':calls,
                'selected_world_T_canonical_with_scale':selected.tolist(),'reference_T_canonical_with_scale':ref_T.tolist(),
                'scale_m_per_canonical_unit':float(np.cbrt(np.linalg.det(selected[:3,:3]))),'sam3d_initial_scale':tr['scale'][0],
                'propagation_audit':audit,'watertight':bool(mesh.is_watertight),'automatic_promotion':False}
        save(path,result);poses.append(selected);report['objects'][name]=result;save(OUT/'report.json',report);print(name,'completed',result,flush=True)
    sequence=np.stack([TRACK[:,oi]@np.linalg.inv(TRACK[0,oi])@poses[oi] for oi in range(2)],axis=1)
    np.savez_compressed(OUT/'candidate_poses.npz',frame_index=np.arange(137),object_names=np.array(NAMES),world_T_canonical_with_scale=sequence)
    report.update(state='completed',elapsed_seconds=time.monotonic()-start,completed_utc=datetime.now(timezone.utc).isoformat())
    save(OUT/'report.json',report);print('completed',report['elapsed_seconds'],flush=True)

if __name__=='__main__':main()
