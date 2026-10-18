"""Bounded original-stereo metric registration. No GT/monitor/model generation.
Reuse public26's verified perspective raster and fixed7DoF loss, adapted only
for heterogeneous original image crops/K and independent per-object budgets.
"""
from pathlib import Path
import argparse,ctypes,datetime,hashlib,json,subprocess,sys,time,traceback
from types import SimpleNamespace
import cv2,numpy as np,trimesh
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
R=Path(__file__).resolve().parents[1]
O=R/'outputs/track3/advice_restart_20261017/canonical9_metric_v1'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');q.replace(p)
def load_manifest():
 d=json.loads((O/'input-manifest.json').read_text());protocol=json.loads((O/'protocol.json').read_text())
 assert sha(__file__)==protocol['source_hashes'][str(Path(__file__).resolve())]
 assert sha(O/'input-manifest.json')==protocol['input_manifest_sha256']
 for row in d['objects']:assert row['frame']not in row['excluded_monitor_frames']
 return d,protocol
def verify(paths,manifest):
 for p in paths:assert sha(p)==manifest['input_hashes'][str(p)],'Changed frozen input: '+str(p)
def verify_cached(stage,key,manifest):
 folder=O/('stereo'if stage=='stereo'else'objects')/key
 report=json.loads((folder/'report.json').read_text())
 if stage=='stereo'and report['state']=='completed':
  assert sha(folder/'evidence.npz')==report['evidence_sha256'],'Changed stereo evidence'
  assert sha(O/'protocol.json')==report['protocol_sha256'],'Changed stereo protocol'
 elif stage=='fit'and report['state']=='fitted_frozen_not_validated':
  frozen=json.loads((folder/'freeze.json').read_text());row=next(r for r in manifest['objects']if r['name']==key)
  for path,digest in [(folder/'metric_canonical.ply',frozen['metric_mesh_sha256']),(folder/'registration.npz',frozen['transform_sha256']),(O/'protocol.json',frozen['protocol_sha256']),(O/'stereo'/row['stereo_key']/'evidence.npz',frozen['evidence_sha256']),(Path(row['metric_shape_source']),frozen['source_mesh_sha256'])]:
   assert sha(path)==digest,'Changed completed fit asset: '+str(path)
 return report['state']
def read_frame(path,frame):
 cap=cv2.VideoCapture(str(path));assert cap.isOpened(),path;cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,im=cap.read();cap.release();assert ok,(path,frame);return im

def stereo(key,manifest,protocol):
 row=next(r for r in manifest['objects']if r['stereo_key']==key);D=O/'stereo'/key;D.mkdir(parents=True,exist_ok=True)
 paths=[manifest['calibration'],row['original_rgb'],row['original_intrinsics'],*row['videos'].values()];verify(paths,manifest)
 if (D/'report.json').exists()and json.loads((D/'report.json').read_text()).get('state')=='completed':verify_cached('stereo',key,manifest);return
 started=time.monotonic();cal=json.loads(Path(manifest['calibration']).read_text());pair=next(p for p in cal['stereo_pairs']if p['camera']=='ego');assert pair['left']=='ego_cam_c'and pair['right']=='ego_cam_b';assert cal['translation_units']=='meters';cs=cal['cameras'];lc,rc=cs['ego_cam_c'],cs['ego_cam_b'];T=np.asarray(pair['left_to_right'],float);size=tuple(lc['resolution']);R1,R2,P1,P2,Q,_,_=cv2.stereoRectify(np.array(lc['intrinsic_matrix']),np.array(lc['distortion_coefficients']),np.array(rc['intrinsic_matrix']),np.array(rc['distortion_coefficients']),size,T[:3,:3].copy(),T[:3,3].reshape(3,1).copy(),flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
 images=[]
 for name,c,rr,pp in [('ego_cam_c',lc,R1,P1),('ego_cam_b',rc,R2,P2)]:
  mp=cv2.initUndistortRectifyMap(np.array(c['intrinsic_matrix']),np.array(c['distortion_coefficients']),rr,pp,size,cv2.CV_32FC1);raw=read_frame(row['videos'][name],row['frame']);assert tuple(raw.shape[1::-1])==size;images.append(cv2.remap(cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY),*mp,cv2.INTER_LINEAR))
 K=np.load(row['original_intrinsics']);rgb=cv2.imread(row['original_rgb']);assert rgb is not None;h,w=rgb.shape[:2];camera=cs['ego_cam_a'];ks=np.array(camera['intrinsic_matrix']);ks[0]*=w/camera['resolution'][0];ks[1]*=h/camera['resolution'][1];expected,_=cv2.getOptimalNewCameraMatrix(ks,np.array(camera['distortion_coefficients']),(w,h),0,(w,h));assert np.allclose(K,expected,atol=1e-6),'Saved calibrated K differs from expected grid'
 # Check the actual RGB decode/resize/undistort grid without changing it.
 raw=read_frame(row['videos']['ego_cam_a'],row['frame']);ok,enc=cv2.imencode('.jpg',raw);assert ok;im=cv2.imdecode(enc,cv2.IMREAD_COLOR);im=cv2.resize(im,(w,h),interpolation=cv2.INTER_AREA);maps=cv2.initUndistortRectifyMap(ks,np.array(camera['distortion_coefficients']),None,K,(w,h),cv2.CV_32FC1);im=cv2.remap(im,*maps,cv2.INTER_LINEAR);ok,enc=cv2.imencode('.jpg',im);assert ok;decoded=cv2.imdecode(enc,cv2.IMREAD_COLOR);grid_mae=float(np.abs(decoded.astype(float)-rgb).mean());assert grid_mae<5.,'RGB source/grid mismatch exceeds5 intensity levels; do not fit'
 l,r=images;params=dict(numDisparities=192,blockSize=5,P1=200,P2=800,uniquenessRatio=10,speckleWindowSize=100,speckleRange=2,disp12MaxDiff=1);dl=cv2.StereoSGBM_create(minDisparity=0,**params).compute(l,r).astype('float32')/16;dr=cv2.StereoSGBM_create(minDisparity=-192,**params).compute(r,l).astype('float32')/16;yy,xx=np.indices(l.shape,dtype='float32');rx=xx-dl;reverse=cv2.remap(dr,rx,yy,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=-10000);lr=np.abs(dl+reverse);warped=cv2.remap(r.astype('float32'),rx,yy,cv2.INTER_LINEAR);ll=l.astype('float32');avg=lambda a:cv2.boxFilter(a,-1,(7,7),normalize=True);al,ar=avg(ll),avg(warped);vl=np.maximum(avg(ll*ll)-al*al,0);vr=np.maximum(avg(warped*warped)-ar*ar,0);ncc=(avg(ll*warped)-al*ar)/np.sqrt(vl*vr+1e-6);texture=np.minimum(np.sqrt(vl),np.sqrt(vr));points=cv2.reprojectImageTo3D(dl,Q);valid=(dl>0)&(rx>=3)&(rx<size[0]-4)&(lr<1.5)&(texture>=4)&(ncc>=.7)&np.isfinite(points).all(-1)&(points[:,:,2]>0)
 a_T_c=np.linalg.inv(np.asarray(camera['extrinsics_to_stereo_left']))@np.asarray(lc['extrinsics_to_stereo_left']);pts=points[valid]@R1;pts=pts@a_T_c[:3,:3].T+a_T_c[:3,3];project=pts@K.T;uv=project[:,:2]/project[:,2:];good=(pts[:,2]>0)&np.isfinite(uv).all(-1)&(uv[:,0]>=0)&(uv[:,0]<w-1)&(uv[:,1]>=0)&(uv[:,1]<h-1);pts=pts[good];uv=uv[good];pix=np.rint(uv).astype(int);depth=np.full(h*w,np.inf,np.float32);np.minimum.at(depth,pix[:,1]*w+pix[:,0],pts[:,2]);depth=depth.reshape(h,w);depth[~np.isfinite(depth)]=np.nan
 orb=cv2.ORB_create(nfeatures=3000);kl,de1=orb.detectAndCompute(l,None);kr,de2=orb.detectAndCompute(r,None);dy=[]
 if de1 is not None and de2 is not None:
  for matches in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(de1,de2,k=2):
   if len(matches)==2 and matches[0].distance<.7*matches[1].distance:
    m=matches[0];dy.append(abs(kl[m.queryIdx].pt[1]-kr[m.trainIdx].pt[1]))
 np.savez_compressed(D/'evidence.npz',depth_m=depth,valid_rectified=valid,disparity=dl,reverse_disparity=dr,lr_error_px=lr,texture_std=texture,zncc=ncc,points_RGB_camera_m=pts,original_rgb_uv=uv,K=K,R1=R1,R2=R2,P1=P1,P2=P2,Q=Q,a_T_c=a_T_c)
 save(D/'report.json',dict(state='completed',completed_utc=now(),elapsed_seconds=time.monotonic()-started,episode=row['episode'],frame=row['frame'],grid=[w,h],original_saved_rgb_reconstruction_MAE=grid_mae,valid_stereo_fraction=float(valid.mean()),RGB_supported_pixels=int(np.isfinite(depth).sum()),epipolar_matches=len(dy),epipolar_vertical_px_p50_p90=np.percentile(dy,[50,90]).tolist()if dy else None,GT_read=False,monitor_read=False,metric_scale_validated=False,input_paths=paths,evidence_sha256=sha(D/'evidence.npz'),protocol_sha256=sha(O/'protocol.json')))

class BudgetStop(Exception):pass
class Raster:
 def __init__(self,path):
  self.P=ctypes.POINTER(ctypes.c_double);self.IP=ctypes.POINTER(ctypes.c_int);self.fn=ctypes.CDLL(str(path)).raster;self.fn.argtypes=[self.P,self.IP,ctypes.c_int,self.P,ctypes.c_int,ctypes.c_int,self.P]
 def __call__(self,v,f,k,w,h):
  v=np.ascontiguousarray(v,dtype='float64');f=np.ascontiguousarray(f,dtype='int32');k=np.ascontiguousarray(k,dtype='float64');z=np.empty((h,w));self.fn(v.ctypes.data_as(self.P),f.ctypes.data_as(self.IP),len(f),k.ctypes.data_as(self.P),w,h,z.ctypes.data_as(self.P));return z

def fit(name,manifest,protocol):
 row=next(r for r in manifest['objects']if r['name']==name);A=O/'objects'/name;A.mkdir(parents=True,exist_ok=True);paths=[row[k]for k in ['original_rgb','original_mask','original_intrinsics','metric_shape_source','model_transform','candidate_freeze']];verify(paths,manifest)
 if (A/'freeze.json').exists():
  verify_cached('fit',name,manifest);return
 evidence_path=O/'stereo'/row['stereo_key']/'evidence.npz';e=np.load(evidence_path);maskdata=np.load(row['original_mask']);assert maskdata['object_names'].tolist()[row['mask_index']]==name;mask=maskdata['masks'][row['mask_index']].astype(bool);rgb=cv2.imread(row['original_rgb']);assert mask.shape==rgb.shape[:2];K=np.load(row['original_intrinsics']);assert np.allclose(K,e['K']);interior=cv2.erode(mask.astype('uint8'),np.ones((7,7),np.uint8))>0;depth=e['depth_m'];vv,uu=np.where(interior&np.isfinite(depth));count_all=len(uu)
 support=dict(interior_pixels=int(interior.sum()),supported_pixels=count_all,support_fraction=float(count_all/max(interior.sum(),1)),depth_p05_p50_p95_m=np.percentile(depth[vv,uu],[5,50,95]).tolist()if count_all else None)
 if count_all<100:
  save(A/'report.json',dict(state='unsupported_metric_scale',reason='Fewer than100 original strict-stereo interior points; no category-scale fallback',support=support,shape_quality_pass=False,metric_scale_validated=False,production_pass=False,known_geometry_defect=row['known_geometry_defect']));return
 rng=np.random.default_rng(42);tile=(vv//32)*10000+uu//32;ids=[]
 for t in np.unique(tile):
  ix=np.where(tile==t)[0];ids.extend(rng.choice(ix,min(32,len(ix)),replace=False))
 ids=np.asarray(ids);uu,vv=uu[ids],vv[ids];obs=depth[vv,uu];sy,sx=np.where(mask);x0=max(0,int(sx.min())-20);x1=min(mask.shape[1],int(sx.max())+21);y0=max(0,int(sy.min())-20);y1=min(mask.shape[0],int(sy.max())+21);factor=min(1.,320/max(x1-x0,y1-y0));w=max(1,round((x1-x0)*factor));h=max(1,round((y1-y0)*factor));fx=w/(x1-x0);fy=h/(y1-y0);Kc=K.copy();Kc[0,2]-=x0;Kc[1,2]-=y0;Kc[0]*=fx;Kc[1]*=fy;Kc[0,2]+=(fx-1)/2;Kc[1,2]+=(fy-1)/2;target=cv2.resize(mask[y0:y1,x0:x1].astype('uint8'),(w,h),interpolation=cv2.INTER_NEAREST_EXACT)>0;pu=np.rint((uu-x0+.5)*fx-.5).astype(int);pv=np.rint((vv-y0+.5)*fy-.5).astype(int);assert np.all((pu>=0)&(pu<w)&(pv>=0)&(pv<h));baseline=abs(e['P2'][0,3]/e['P2'][0,0]);sigma=np.maximum(.005,1.5*obs**2/(e['P1'][0,0]*baseline));boundary=target&~cv2.erode(target.astype('uint8'),np.ones((3,3),np.uint8)).astype(bool);dt_target=cv2.distanceTransform((~boundary).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE);mz=float(np.median(obs));camera_center=np.linalg.solve(K,[sx.mean(),sy.mean(),1.])*mz
 mesh=trimesh.load(row['metric_shape_source'],force='mesh',process=False);freeze=json.loads(Path(row['candidate_freeze']).read_text());assert sha(row['metric_shape_source'])==freeze['mesh_sha256'];v=mesh.vertices;faces=np.ascontiguousarray(mesh.faces,dtype='int32');q=json.loads(Path(row['model_transform']).read_text())['rotation'];R0=Rotation.from_quat(q[1:]+q[:1]).as_matrix();center=(mesh.bounds[0]+mesh.bounds[1])/2;extent=np.ptp((v-center)@R0.T,axis=0);s0=float(np.median([(sx.max()-sx.min())*mz/(K[0,0]*extent[0]),(sy.max()-sy.min())*mz/(K[1,1]*extent[1])]));bounds=[(-np.deg2rad(25),np.deg2rad(25))]*3+[(-.2*mz,.2*mz)]*3+[(np.log(.5),np.log(2))];raster=Raster(O/'raster.so');np.savez_compressed(A/'fit-evidence.npz',u=uu,v=vv,depth=obs,sigma=sigma,render_u=pu,render_v=pv,render_K=Kc,render_target=target,crop_xyxy=[x0,y0,x1,y1]);resume=json.loads((A/'progress.json').read_text())if (A/'progress.json').exists()else {};evaluations=int(resume.get('evaluations',0));elapsed_before=float(resume.get('optimizer_elapsed_seconds',0));xstart=np.asarray(resume.get('best_x',np.zeros(7)),float);best=dict(cost=float('inf'),x=xstart.copy());started=time.monotonic();log=(A/'evaluations.jsonl').open('a')
 def transform(x):
  rot=Rotation.from_rotvec(x[:3]).as_matrix()@R0;s=s0*np.exp(x[6]);t=camera_center+x[3:6]-s*(rot@center);return s,rot,t
 def evaluate(x,record=True):
  nonlocal evaluations
  if record and (evaluations>=600 or elapsed_before+time.monotonic()-started>=90):raise BudgetStop()
  s,rot,t=transform(x);z=raster(s*(v@rot.T)+t,faces,Kc,w,h);pred=z<1e20;iou=np.count_nonzero(pred&target)/max(np.count_nonzero(pred|target),1);pb=pred&~cv2.erode(pred.astype('uint8'),np.ones((3,3),np.uint8)).astype(bool);dt=cv2.distanceTransform((~pb).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE);bd=.5*(np.minimum(dt_target[pb],10).mean()if pb.any()else 10)+.5*np.minimum(dt[boundary],10).mean();pz=z[pv,pu];hit=pz<1e20;norm=np.abs(pz-obs)/sigma;loss=np.where(norm<=1,.5*norm**2,np.minimum(norm,6)-.5);loss[~hit]=3;cost=.75*(1-iou)+.25*bd/10+.25*loss.mean();err=np.abs(pz[hit]-obs[hit]);metrics=dict(cost=float(cost),iou=float(iou),boundary_distance_render_px=float(bd),depth_mean_abs_mm=float(err.mean()*1000)if len(err)else None,depth_p95_abs_mm=float(np.percentile(err,95)*1000)if len(err)else None,depth_coverage=float(hit.mean()),sample_count=len(obs),scale_m_per_unit=float(s))
  if record:
   evaluations+=1
   if cost<best['cost']:best.update(cost=float(cost),x=np.asarray(x).copy())
   save(A/'progress.json',dict(evaluations=evaluations,optimizer_elapsed_seconds=elapsed_before+time.monotonic()-started,best_x=best['x'].tolist(),best_cost=best['cost'],evidence_sha256=sha_evidence))
   if evaluations%25==0:log.write(json.dumps(dict(evaluation=evaluations,**metrics))+'\n');log.flush()
  return cost,metrics,z
 sha_evidence=sha(evidence_path)
 if resume:assert resume['evidence_sha256']==sha_evidence
 _,initial,zinit=evaluate(np.zeros(7),False);c,_,_=evaluate(xstart,False);best.update(cost=float(c),x=xstart.copy())
 try:
  remaining=max(0,600-evaluations);res=minimize(lambda x:evaluate(x)[0],xstart,method='Powell',bounds=bounds,options=dict(maxfev=remaining,xtol=.001,ftol=.0001))if remaining else SimpleNamespace(success=False,message='Saved evaluation budget exhausted')
 except BudgetStop:res=SimpleNamespace(success=False,message='Fixed evaluation or90s optimizer wall budget reached; freezing best')
 x=best['x'];_,final,zfinal=evaluate(x,False);s,rot,t=transform(x);T=np.eye(4);T[:3,:3]=rot;T[:3,3]=t;np.savez_compressed(A/'registration.npz',camera_T_canonical=T,scale_m_per_unit=s,delta_parameters=x,initial_scale=s0,initial_rotation=R0,initial_camera_center=camera_center,original_K=K);metric=mesh.copy();metric.vertices=v*s;metric.export(A/'metric_canonical.ply');props=metric.mass_properties;mass=float(props['mass']);inertia=np.asarray(props['inertia']);com=np.asarray(props['center_mass']);save(A/'mass-properties-UNVALIDATED.json',dict(volume_m3=float(props['volume']),canonical_COM_m=com.tolist(),inertia_per_unit_mass_m2=(inertia/mass).tolist()if mass>0 else None,physical_mass_kg=None,convention='Geometric uniform-density estimate only; disconnected/unknown shell thickness retained, not measured physical mass/inertia'))
 canvas=cv2.resize(rgb[y0:y1,x0:x1],(w,h));ims=[]
 for label,z in [('initial',zinit),('fitted',zfinal)]:
  im=canvas.copy();contours,_=cv2.findContours((z<1e20).astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,contours,-1,(0,0,255),1);cv2.putText(im,label,(4,14),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1);ims.append(im)
 cv2.imwrite(str(A/'train-overlay.jpg'),np.concatenate(ims,axis=1));low=np.array([b[0]for b in bounds]);high=np.array([b[1]for b in bounds]);near=(np.minimum(x-low,high-x)/(high-low)<.01);report=dict(state='fitted_frozen_not_validated',completed_utc=now(),name=name,episode=row['episode'],frame=row['frame'],evaluations=evaluations,optimizer_elapsed_seconds=elapsed_before+time.monotonic()-started,optimizer_success=bool(res.success),optimizer_message=str(res.message),near_search_bound_parameters=np.where(near)[0].tolist(),initial=initial,fitted=final,support=support,known_geometry_defect=row['known_geometry_defect'],shape_quality_pass=False,metric_scale_validated=False,production_pass=False,GT_read=False,monitor_read=False,coordinate_frame=f'original undistorted ego_cam_a episode{row["episode"]}frame{row["frame"]}',scale_convention='metric_canonical vertices already scaled once; camera_T_canonical is rigid link-origin transform, COM separate',camera_K_used=K.tolist(),render_K=Kc.tolist(),crop_xyxy=[x0,y0,x1,y1],source_shape_frozen=True);save(A/'report.json',report);save(A/'freeze.json',dict(frozen_utc=now(),source_mesh_sha256=freeze['mesh_sha256'],metric_mesh_sha256=sha(A/'metric_canonical.ply'),transform_sha256=sha(A/'registration.npz'),protocol_sha256=sha(O/'protocol.json'),evidence_sha256=sha_evidence,GT_read=False,monitor_read=False,production_pass=False));log.close();print(json.dumps(report),flush=True)

def coordinator(manifest,protocol):
 for p,digest in protocol['source_hashes'].items():assert sha(p)==digest,'Changed batch source'
 verify(list(manifest['input_hashes']),manifest);subprocess.run(['g++','-O3','-shared','-fPIC','-std=c++17',str(O/'raster-source.cpp'),'-o',str(O/'raster.so')],check=True);raster=Raster(O/'raster.so');test=raster(np.array([[0,0,1],[4,0,1],[0,4,2.]]),np.array([[0,1,2]]),np.eye(3),5,5);assert abs(test[1,1]-4/3)<1e-10
 keys=sorted(set(r['stereo_key']for r in manifest['objects']));tasks=[('stereo',k,30)for k in keys]+[('fit',r['name'],105)for r in manifest['objects']];rows=[]
 for stage,key,limit in tasks:
  out=O/('stereo'if stage=='stereo'else'objects')/key;out.mkdir(parents=True,exist_ok=True);rp=out/'report.json';terminal={'completed','fitted_frozen_not_validated','unsupported_metric_scale','failed','timeout'}
  if rp.exists()and json.loads(rp.read_text()).get('state')in terminal:rows.append(dict(stage=stage,key=key,cached=True,state=verify_cached(stage,key,manifest)));continue
  if stage=='fit':
   row=next(r for r in manifest['objects']if r['name']==key);sp=O/'stereo'/row['stereo_key']/'report.json'
   if not sp.exists()or json.loads(sp.read_text()).get('state')!='completed':save(rp,dict(state='unsupported_metric_scale',reason='Original stereo stage failed; no metric fallback',production_pass=False));rows.append(dict(stage=stage,key=key,state='unsupported_metric_scale'));continue
  save(O/'job.json',dict(state='running',active_stage=stage,active_key=key,heartbeat_utc=now(),completed_tasks=len(rows)))
  start=time.monotonic()
  with open(out/'run.log','a')as log:
   try:code=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--stage',stage,'--key',key],stdout=log,stderr=subprocess.STDOUT,timeout=limit).returncode
   except subprocess.TimeoutExpired:code=124
  (out/'run.exit').write_text(str(code)+'\n')
  if code and not rp.exists():save(rp,dict(state='timeout'if code==124 else'failed',exit_code=code,stage=stage,wall_cap_seconds=limit,checkpoint_preserved=True,automatic_retry=False,production_pass=False))
  rows.append(dict(stage=stage,key=key,exit_code=code,elapsed_seconds=time.monotonic()-start,state=json.loads(rp.read_text())['state']if rp.exists()else'missing_report'));save(O/'progress.json',dict(tasks=rows,updated_utc=now()))
 objects={r['name']:json.loads((O/'objects'/r['name']/'report.json').read_text())for r in manifest['objects']};save(O/'report.json',dict(state='completed_bounded_batch',completed_utc=now(),tasks=rows,objects=objects,fitted=sum(r['state']=='fitted_frozen_not_validated'for r in objects.values()),expected=9,GT_read=False,monitor_read=False,shape_frozen=True,metric_scale_validated=False,production_pass=False,next_step='Review fixed-budget original TRAIN overlays/depth and known shape defects; freeze before independent monitors. No automatic local search or regeneration.'));save(O/'job.json',dict(state='completed_bounded_batch',completed_utc=now(),report=str(O/'report.json')))

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['batch','stereo','fit'],default='batch');p.add_argument('--key');args=p.parse_args();cv2.setNumThreads(2);manifest,protocol=load_manifest()
 if args.stage=='batch':coordinator(manifest,protocol)
 elif args.stage=='stereo':stereo(args.key,manifest,protocol)
 else:fit(args.key,manifest,protocol)
if __name__=='__main__':main()
