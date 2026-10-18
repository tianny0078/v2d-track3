"""Independent table-nonpenetrating 6D registration variant of frozen canonical scene producer.
Full visual/collision support is projected analytically; original scale/loss/settle gates remain.
Project-side experimental candidate. No GT, generated shape/scale updates, policy,
unknown-hand relabeling, altered gravity, or overwrite of submitted references.
"""
from pathlib import Path
import argparse,copy,datetime,hashlib,json,math,os,subprocess,sys,time,traceback
import xml.etree.ElementTree as ET
import cv2,numpy as np,trimesh,mujoco
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from track3_canonical9_metric_batch import Raster
from build_track3_coarse_objects import export_asset
R=Path(__file__).resolve().parents[1]
B=R/'outputs/track3/advice_restart_20261017'
O=B/'canonical_supported_registration_v1'
SOURCE=Path(__file__).resolve()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');q.replace(p)
def tf(v,T):return np.asarray(v)@T[:3,:3].T+T[:3,3]
def Tof(q):
 T=np.eye(4);T[:3,3]=q[:3];T[:3,:3]=Rotation.from_quat(q[3:],scalar_first=True).as_matrix();return T
def pose(T):return np.r_[T[:3,3],Rotation.from_matrix(T[:3,:3]).as_quat(scalar_first=True)]
def inputs():return read(B/'all20-registration-inputs-v1/input-manifest.json')['rows']
def frozen(p,key):
 if not p.exists():return None
 d=read(p)
 if d.get('cache_key')!=key:raise RuntimeError('Frozen cache input/source changed: '+str(p))
 for path,h in d.get('output_sha256',{}).items():
  if sha(path)!=h:raise RuntimeError('Frozen output changed: '+path)
 return d

def asset_source(name):
 if name=='wooden_spoon':
  g=read(B/'spoon_canonical_v1/sam3d_physics_v1/geometry-report.json')['variants']['native_transform']
  return Path(g['asset']['mesh_path']),None,g
 if name=='white_pot':return B/'public26_pot_blind_v1/metric_registration_v1/original/metric_canonical.ply',B/'public26_pot_blind_v1/metric_registration_v1/original/registration.npz',None
 return B/'canonical9_metric_v1/objects'/name/'metric_canonical.ply',B/'canonical9_metric_v1/objects'/name/'registration.npz',None

def asset(name):
 import coacd
 source,reg,reuse=asset_source(name);D=O/'assets'/name;D.mkdir(parents=True,exist_ok=True)
 key=sha(SOURCE)+sha(source)+(sha(reg)if reg else sha(B/'spoon_canonical_v1/sam3d_physics_v1/geometry-report.json'))
 cached=frozen(D/'report.json',key)
 if cached:return cached
 mesh=trimesh.load(source,force='mesh',process=False)
 assert np.isfinite(mesh.vertices).all() and len(mesh.faces)>0
 params=dict(threshold=.001,real_metric=True,max_convex_hull=64,preprocess_mode='off',resolution=1000,mcts_nodes=12,mcts_iterations=20,mcts_max_depth=3,pca=False,merge=True,decimate=False,seed=18102026)
 start=time.monotonic()
 if reuse:
  old=Path(reuse['asset']['urdf_path']);link=ET.parse(old).getroot().find('link')
  parts=[trimesh.load(old.parent/c.find('geometry/mesh').get('filename'),force='mesh',process=False)for c in link.findall('collision')]
  mass=float(reuse['asset']['mass_kg']);method='reuse_frozen_spoon_64part_CoACD'
 else:
  # Frozen geometric shape is never repaired, scaled, simplified, or recentered.
  raw=coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices),np.asarray(mesh.faces)),**params)
  parts=[]
  for v,f in raw:
   p=trimesh.Trimesh(v,f,process=False)
   if p.volume<0:p.invert()
   assert p.is_watertight and p.volume>0
   parts.append(p)
  assert len(parts)>1,'Refuse a single global hull for these generated objects'
  # Unmeasured mass hypothesis, shared across occurrences; copied from prior asset.
  row=next(x for x in inputs()if x['name']==name)
  old=read(R/'outputs/track3'/row['episode']/'objects/objects.json')['objects'][row['object_index']]
  mass=float(old.get('mass_kg',.15));method='fixed_parameter_CoACD'
 a=export_asset(D,name,mesh,parts,mass)
 # Uniform explicit material avoids Newton interpreting OBJ materials as many shapes.
 u=ET.parse(a['urdf_path']);vis=u.getroot().find('link/visual');mat=ET.SubElement(vis,'material',name='canonical_uniform');ET.SubElement(mat,'color',rgba='0.65 0.62 0.55 1');u.write(a['urdf_path'])
 a.update(reconstruction_source='frozen_self_reconstructed_SAM3D_original_RGB',source_inputs=[str(source)],coordinate_origin='unchanged metric canonical link origin',scale_applied_again=False)
 collision_volume=sum(abs(p.volume)for p in parts)
 # 3-axis deterministic sections measure decomposition occupancy additions, not GT fidelity.
 rng=np.random.default_rng(1810);section=[]
 for axis in range(3):
  others=[k for k in range(3)if k!=axis];lo,hi=mesh.bounds;xy=rng.uniform(lo[others],hi[others],(24,2));added=0;tested=0
  for xy0 in xy:
   def intervals(m):
    tri=m.triangles;u=tri[:,:,others[0]];v=tri[:,:,others[1]];den=(v[:,1]-v[:,2])*(u[:,0]-u[:,2])+(u[:,2]-u[:,1])*(v[:,0]-v[:,2]);aa=np.divide((v[:,1]-v[:,2])*(xy0[0]-u[:,2])+(u[:,2]-u[:,1])*(xy0[1]-v[:,2]),den,out=np.full(len(tri),np.nan),where=abs(den)>1e-15);bb=np.divide((v[:,2]-v[:,0])*(xy0[0]-u[:,2])+(u[:,0]-u[:,2])*(xy0[1]-v[:,2]),den,out=np.full(len(tri),np.nan),where=abs(den)>1e-15);cc=1-aa-bb;ok=(aa>=0)&(bb>=0)&(cc>=0);s=np.sort((aa*tri[:,0,axis]+bb*tri[:,1,axis]+cc*tri[:,2,axis])[ok]);return s
   s=intervals(mesh);xs=np.linspace(lo[axis],hi[axis],80);visual=np.searchsorted(s,xs)%2==1;coll=np.zeros(80,bool)
   for part in parts:
    sp=intervals(part)
    if len(sp)>=2:coll|=(xs>=sp.min())&(xs<=sp.max())
   added+=int((coll&~visual).sum());tested+=int(coll.sum())
  section.append(dict(axis=axis,added_occupancy_fraction=added/max(tested,1),sampled_collision_points=tested))
 paths=[Path(a['mesh_path']),Path(a['urdf_path']),*D.glob('collision_*.obj')]
 report=dict(state='completed',created_utc=now(),cache_key=key,asset=a,source_mesh=str(source),source_mesh_sha256=sha(source),registration=str(reg)if reg else None,method=method,parameters=params if not reuse else'reused_frozen',elapsed_seconds=time.monotonic()-start,extents_m=mesh.extents.tolist(),oversized_max_axis_gt_0_5m=bool(mesh.extents.max()>.5),mesh_watertight=bool(mesh.is_watertight),collision_parts=len(parts),collision_volume_sum_m3=collision_volume,visual_volume_m3=float(mesh.volume),collision_section_diagnostic=section,cavity_preservation_validated=False,topology_is_not_fidelity=True,production_pass=False,output_sha256={str(p):sha(p)for p in paths})
 save(D/'report.json',report);print('asset',name,'parts',len(parts),flush=True);return report

def observation(ep,name,frame):
 E=R/'outputs/track3'/ep;config=read(E/'coarse-input.json');m=np.load(config['masks_pattern'].format(frame=frame));names=m['object_names'].tolist();mask=m['masks'][names.index(name)].astype(bool);other=np.any(np.delete(m['masks'],names.index(name),axis=0),axis=0)if len(names)>1 else np.zeros_like(mask)
 # Overlapping SAM masks cannot identify a surface; preserve overlap ambiguity.
 ambiguous=mask&other
 dep_path=Path(config['depth_pattern'].format(frame=frame));z=np.load(dep_path)[config['depth_key']];valid=np.isfinite(z)&(z>.05)&(z<5)&mask
 depth=np.where(valid,z,np.nan)
 return dict(mask=mask,other=other,ambiguous=ambiguous,depth=depth,K=np.load(config['K_path']),depth_path=dep_path,mask_path=Path(config['masks_pattern'].format(frame=frame)),rgb_path=Path(config['rgb_pattern'].format(frame=frame)))

def render_context(obs,mesh,side=200):
 mask=obs['mask'];y,x=np.where(mask)
 if not len(x):raise RuntimeError('Object has no mask support')
 x0=max(0,x.min()-25);x1=min(mask.shape[1],x.max()+26);y0=max(0,y.min()-25);y1=min(mask.shape[0],y.max()+26);factor=min(1.,side/max(x1-x0,y1-y0));w=round((x1-x0)*factor);h=round((y1-y0)*factor);fx=w/(x1-x0);fy=h/(y1-y0);K=obs['K'].copy();K[0,2]-=x0;K[1,2]-=y0;K[0]*=fx;K[1]*=fy;K[0,2]+=(fx-1)/2;K[1,2]+=(fy-1)/2
 target=cv2.resize(mask[y0:y1,x0:x1].astype('uint8'),(w,h),interpolation=cv2.INTER_NEAREST_EXACT)>0
 ignore=cv2.resize((obs['other']&~mask)[y0:y1,x0:x1].astype('uint8'),(w,h),interpolation=cv2.INTER_NEAREST_EXACT)>0
 # Read existing LR-stereo depth as uncertain observation, not GT. No new model.
 inside=cv2.erode(mask.astype('uint8'),np.ones((5,5),np.uint8)).astype(bool);yy,xx=np.where(inside&np.isfinite(obs['depth']));rng=np.random.default_rng(1810)
 if len(xx)>1200:ix=rng.choice(len(xx),1200,replace=False);yy,xx=yy[ix],xx[ix]
 u=np.clip(np.rint((xx-x0+.5)*fx-.5).astype(int),0,w-1);v=np.clip(np.rint((yy-y0+.5)*fy-.5).astype(int),0,h-1);dz=obs['depth'][yy,xx];raster=Raster(B/'canonical9_metric_v1/raster.so')
 def evaluate(T):
  z=raster(tf(mesh.vertices,T),mesh.faces,K,w,h);pred=(z>0)&(z<1e20);union=(pred|target)&~ignore;inter=pred&target&~ignore;iou=inter.sum()/max(1,union.sum());hit=pred[v,u];err=abs(z[v[hit],u[hit]]-dz[hit]);coverage=hit.mean()if len(hit)else 0.;depthloss=float(np.minimum(err/.02,6).mean())if len(err)else 6.
  metrics=dict(iou=float(iou),depth_MAE_m=float(err.mean())if len(err)else None,depth_p95_m=float(np.percentile(err,95))if len(err)else None,depth_coverage=float(coverage),depth_support=len(dz),mask_overlap_ambiguous_fraction=float(obs['ambiguous'].sum()/max(mask.sum(),1)))
  return .8*(1-iou)+.1*depthloss+.1*(1-coverage),metrics,pred
 return evaluate,(x0,y0,x1,y1),target

def register(ep,row,a):
 name=row['name'];D=O/'episodes'/ep/'registrations'/name;D.mkdir(parents=True,exist_ok=True);E=R/'outputs/track3'/ep
 old=np.load(E/'objects/poses.npz');frames=old['frame_index'];j=old['object_names'].tolist().index(name);oldT=old['world_T_canonical'][:,j];cams=np.load(E/'stereo_sequence/camera_poses.npz');A=np.load(E/'reference/reference_checkpoint.npz')['table_T_world'];source,reg,reuse=asset_source(name)
 key=sha(SOURCE)+sha(E/'objects/poses.npz')+sha(E/'stereo_sequence/camera_poses.npz')+sha(E/'reference/reference_checkpoint.npz')+sha(source)+(sha(reg)if reg else '')
 cfg=read(E/'coarse-input.json');ff=int(row['frozen_anchor_candidate']['frame'])if row.get('frozen_anchor_candidate')else int(row['frame'])
 for pp in [Path(cfg['masks_pattern'].format(frame=ff)),Path(cfg['depth_pattern'].format(frame=ff)),Path(cfg['K_path'])]:key+=sha(pp)
 support_urdf=Path(a['urdf_path']);key+=sha(support_urdf)+sha(Path(a['mesh_path']))
 for shape in ET.parse(support_urdf).getroot().iter('mesh'):key+=sha((support_urdf.parent/shape.get('filename')).resolve())
 cached=frozen(D/'report.json',key)
 if cached:return np.load(D/'registration.npz')['world_T_canonical'],cached
 mesh=trimesh.load(a['mesh_path'],force='mesh',process=False);anchor=row.get('frozen_anchor_candidate');f=int(row['frame']);method='fixed_metric_shape_6D_render_registration';fitted=True
 support_vertices=[np.asarray(mesh.vertices)];urdf=Path(a['urdf_path']);link=ET.parse(urdf).getroot().find('link')
 for collision in link.findall('collision'):
  # Frozen generated assets use identity collision origins and unit scale.
  origin=collision.find('origin');assert origin is None or (np.allclose(np.fromstring(origin.get('xyz','0 0 0'),sep=' '),0)and np.allclose(np.fromstring(origin.get('rpy','0 0 0'),sep=' '),0))
  shape=collision.find('geometry/mesh');assert np.allclose(np.fromstring(shape.get('scale','1 1 1'),sep=' '),1)
  part=trimesh.load(urdf.parent/shape.get('filename'),force='mesh',process=False);support_vertices.append(np.asarray(part.vertices))
 support_vertices=np.concatenate(support_vertices);support_evidence=dict(method='nearest_nonpenetrating_table_normal_projection',clearance_m=.001,vertex_count=len(support_vertices),uses_all_visual_and_collision_vertices=True,table_transform_source=str(E/'reference/reference_checkpoint.npz'),table_transform_sha256=sha(E/'reference/reference_checkpoint.npz'),URDF_sha256=sha(urdf),applied_to_optimization=fitted,measured_contact=False)
 unconstrained_initial=None;projection_lifts=[]
 if name=='wooden_spoon'and ep=='episode_000037':
  world=np.array(reuse['world_T_link']);f=0;method='reuse_native_SAM3D_ep37';fitted=False
 elif anchor:
  f=int(anchor['frame']);z=np.load(R/anchor['path']);world=np.linalg.inv(A)@z['table_T_canonical'][0];method='reuse_frozen_same_episode_anchor';fitted=False
 ix=int(np.flatnonzero(frames==f)[0]);ci=int(np.flatnonzero(cams['frame_index']==f)[0]);cam=cams['world_T_rgb'][ci];obs=observation(ep,name,f);evaluate,crop,target=render_context(obs,mesh)
 def support_projection(T):
  table_T_link=A@cam@T;raw_min=float(np.min(support_vertices@table_T_link[2,:3])+table_T_link[2,3]);lift=max(0.,.001-raw_min)
  projected=T.copy();projected[:3,3]+=np.linalg.inv(A@cam)[:3,2]*lift
  check=A@cam@projected;new_min=float(np.min(support_vertices@check[2,:3])+check[2,3]);assert new_min>=.001-1e-8
  return projected,dict(unprojected_min_z_m=raw_min,table_normal_lift_m=lift,projected_min_z_m=new_min)
 if not fitted:T=np.linalg.inv(cam)@world;_,initial,_=evaluate(T);final=initial;calls=0;elapsed=0.;support_evidence['applied_to_optimization']=False
 else:
  if reg:T0=np.load(reg)['camera_T_canonical'].copy()
  else:T0=np.linalg.inv(np.load(R/'outputs/track3/episode_000037/stereo_sequence/camera_poses.npz')['world_T_rgb'][0])@np.array(reuse['world_T_link'])
  vv,uu=np.where(obs['mask']&np.isfinite(obs['depth']));depth=np.median(obs['depth'][vv,uu])if len(uu)else np.nan
  if not np.isfinite(depth):raise RuntimeError('No finite depth; no invented scale/position fallback')
  center=(mesh.bounds[0]+mesh.bounds[1])/2;y,x=np.where(obs['mask']);tc=np.linalg.solve(obs['K'],[np.median(x),np.median(y),1])*depth;T0[:3,3]=tc-T0[:3,:3]@center
  _,unconstrained_initial,_=evaluate(T0);T0,initial_support=support_projection(T0);support_evidence['initial']=initial_support;method='fixed_metric_shape_6D_render_registration_with_table_nonpenetration'
  start=time.monotonic();calls=0;best=[float('inf'),T0.copy()]
  def trial(x):
   T=T0.copy();T[:3,:3]=Rotation.from_rotvec(x[:3]).as_matrix()@T0[:3,:3];T[:3,3]=tc+x[3:]-T[:3,:3]@center;T,projection=support_projection(T);projection_lifts.append(projection['table_normal_lift_m']);return T
  class Stop(Exception):pass
  def loss(x):
   nonlocal calls
   if calls>=400 or time.monotonic()-start>45:raise Stop()
   calls+=1;T=trial(x);c,_,_=evaluate(T)
   if c<best[0]:best[:]=[c,T.copy()]
   return c
  _,initial,_=evaluate(T0)
  try:minimize(loss,np.zeros(6),method='Powell',bounds=[(-np.pi,np.pi)]*3+[(-.15,.15)]*3,options=dict(maxfev=400,xtol=.002,ftol=.0002))
  except Stop:pass
  T=best[1];_,final,_=evaluate(T);world=cam@T;elapsed=time.monotonic()-start
 selected_table=A@world;support_evidence['selected_min_z_m']=float(np.min(support_vertices@selected_table[2,:3])+selected_table[2,3]);support_evidence['projected_trial_count']=sum(x>0 for x in projection_lifts);support_evidence['max_trial_lift_m']=max(projection_lifts,default=0.);support_evidence['selected_supported']=bool(support_evidence['selected_min_z_m']>=.001-1e-8)
 C=np.linalg.inv(oldT[ix])@world;new=oldT@C
 np.testing.assert_allclose(new[ix],world,atol=1e-8)
 # Preserve all world-relative object motions; origins are explicitly changed once.
 np.savez_compressed(D/'registration.npz',frame_index=frames,world_T_canonical=new,oldlink_T_newlink=C,fit_frame=f,world_T_rgb=cam,table_T_world=A,observed=False)
 _,_,pred=evaluate(T);im=cv2.imread(str(obs['rgb_path']));x0,y0,x1,y1=crop;canvas=cv2.resize(im[y0:y1,x0:x1],(pred.shape[1],pred.shape[0]));contours,_=cv2.findContours(pred.astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(canvas,contours,-1,(0,0,255),1);cv2.imwrite(str(D/'registration-overlay.jpg'),canvas)
 report=dict(state='completed_candidate',cache_key=key,method=method,fit_frame=f,evaluations=calls,elapsed_seconds=elapsed,initial=initial,unconstrained_initial=unconstrained_initial,support_constraint=support_evidence,fitted=final,source_mesh=str(source),shape_and_scale_frozen=True,oldlink_T_newlink=C.tolist(),relative_motion_preserved=True,world_pose_observed=False,production_pass=False,source_depth=str(obs['depth_path']),depth_is_ground_truth=False,output_sha256={str(D/'registration.npz'):sha(D/'registration.npz')})
 save(D/'report.json',report);return new,report

def scene(ep,assets,worldTs):
 E=R/'outputs/track3'/ep;D=O/'episodes'/ep;tree=ET.parse(E/'reference/scene_cpu.xml');root=tree.getroot();A=np.load(E/'reference/reference_checkpoint.npz')['table_T_world'];names=list(assets)
 for side,sign in [('left',-1),('right',1)]:
  hand=next(x for x in root.iter('body')if x.get('name')==side+'_wrist_px_body');p=np.fromstring(hand.get('pos','0 0 0'),sep=' ');hand.set('pos',' '.join(map(str,p+[sign*3,0,2])))
 for name,a in assets.items():
  body=next(x for x in root.iter('body')if x.get('name')==name+'_'+name);u=Path(a['urdf_path']);link=ET.parse(u).getroot().find('link');template=copy.deepcopy(body.find('geom'))
  for g in list(body.findall('geom')):body.remove(g)
  inert=link.find('inertial');tensor=inert.find('inertia').attrib;old=body.find('inertial')
  if old is None:old=ET.SubElement(body,'inertial')
  old.attrib.clear();old.attrib.update(pos=inert.find('origin').get('xyz'),mass=inert.find('mass').get('value'),fullinertia=' '.join(tensor[k]for k in ['ixx','iyy','izz','ixy','ixz','iyz']))
  for i,c in enumerate(link.findall('collision')):
   mn='canonical_'+name+'_'+str(i);ET.SubElement(root.find('asset'),'mesh',name=mn,file=str((u.parent/c.find('geometry/mesh').get('filename')).resolve()));g=copy.deepcopy(template);g.set('name',mn);g.set('mesh',mn);g.set('type','mesh');g.set('pos','0 0 0');g.set('quat','1 0 0 0');g.attrib.pop('size',None);body.append(g)
 path=D/'scene.xml';tree.write(path);m=mujoco.MjModel.from_xml_path(str(path));m.opt.timestep=.001;d=mujoco.MjData(m);q=np.load(E/'reference/scene_check_checkpoint.npz')['initial_joint_q'].copy()
 for j in range(m.njnt):
  if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE:
   k=int(m.jnt_qposadr[j]);q[k+3:k+7]=q[k+3:k+7][[3,0,1,2]]
 ids=[]
 for name in names:
  bid=m.body(name+'_'+name).id;ids.append(bid);j=m.body_jntadr[bid];k=m.jnt_qposadr[j];q[k:k+7]=pose(A@worldTs[name][0])
 np.savez_compressed(D/'scene-initial.npz',initial_qpos_wxyz=q,table_T_world=A,object_names=names,world_T_canonical=np.stack([worldTs[n]for n in names],axis=1))
 return m,q,ids

def simulate(ep,m,q,ids,arm,duration):
 D=O/'episodes'/ep/arm;D.mkdir(exist_ok=True);key=sha(SOURCE)+sha(O/'episodes'/ep/'scene.xml')+hashlib.sha256(q.tobytes()).hexdigest()+str(duration);cached=frozen(D/'report.json',key)
 if cached:return cached,np.load(D/'checkpoint.npz')['qpos']
 d=mujoco.MjData(m);d.qpos[:]=q;d.qvel[:]=0;d.ctrl[:]=0;mujoco.mj_forward(m,d);startcom=d.xipos[ids].copy();startrot=d.xquat[ids].copy();n=math.ceil(duration/m.opt.timestep);history=[];start=0
 def measure():
  pen=np.zeros(len(ids));hand=False
  for c in d.contact:
   b1,b2=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
   for i,b in enumerate(ids):
    if b in (b1,b2):
     pen[i]=max(pen[i],-float(c.dist));other=b2 if b1==b else b1;bn=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,other)or'';hand|=bn.startswith(('left_','right_'))
  com=np.linalg.norm(d.xipos[ids]-startcom,axis=1);rot=np.degrees(2*np.arccos(np.clip(abs(np.sum(d.xquat[ids]*startrot,axis=1)),0,1)))
  return np.c_[com,rot,pen],hand
 initial,hand0=measure();history=[initial.tolist()];ck=D/'checkpoint.npz';finite=True;noreset=True;hand=hand0
 if ck.exists():
  z=np.load(ck);assert str(z['cache_key'])==key;d.qpos[:]=z['qpos'];d.qvel[:]=z['qvel'];d.time=float(z['time']);history=z['history'].tolist();start=int(z['steps']);hand=bool(z['hand_contact']);mujoco.mj_forward(m,d)
 begin=time.monotonic()
 for i in range(start,n):
  prior=d.time;mujoco.mj_step(m,d);mujoco.mj_forward(m,d);stats,h=measure();hand|=h;history.append(stats.tolist());finite=bool(np.isfinite(d.qpos).all()and np.isfinite(d.qvel).all());noreset&=bool(d.time>prior)
  if (i+1)%500==0 or i+1==n or not finite or not noreset:
   np.savez_compressed(ck,cache_key=key,qpos=d.qpos,qvel=d.qvel,time=d.time,steps=i+1,history=np.asarray(history),initial_qpos_wxyz=q,hand_contact=hand)
   print(ep,arm,i+1,n,flush=True)
  if not finite or not noreset:break
 H=np.asarray(history);warnings=[int(w.number)for w in d.warning];full=(len(H)-1)==n;gates=dict(COM_lt10mm=bool(H[:,:,0].max()<.01),penetration_lt5mm=bool(H[:,:,2].max()<.005),finite=finite,no_reset=noreset,no_warning_overflow=not any(warnings),full_duration=full,no_hand_contact=not hand,no_assistance=True)
 result=dict(state='completed',cache_key=key,arm=arm,required_duration_s=duration,actual_simulated_s=float(d.time),steps=len(H)-1,max_COM_drift_m=H[:,:,0].max(axis=0).tolist(),max_rotation_deg=H[:,:,1].max(axis=0).tolist(),max_penetration_m=H[:,:,2].max(axis=0).tolist(),initial_penetration_m=H[0,:,2].tolist(),gates=gates,engineering_pass=bool(all(gates.values())),endpoint_qvel_norm=float(np.linalg.norm(d.qvel)),gravity=m.opt.gravity.tolist(),hand_roots_shifted_m=[[-3,0,2],[3,0,2]],zero_action=True,zero_initial_velocity_assumption=True,object_external_force=False,rotation_veto=False,elapsed_seconds=time.monotonic()-begin,production_pass=False,output_sha256={str(ck):sha(ck)})
 save(D/'report.json',result);return result,d.qpos.copy()

def objects_manifest(ep,assets,Ts,A,frames,regs,where):
 D=O/'episodes'/ep/where;D.mkdir(parents=True,exist_ok=True);names=list(assets);W=np.stack([Ts[n]for n in names],axis=1);Q=np.array([[pose(T)for T in f]for f in W]);np.savez_compressed(D/'poses.npz',frame_index=frames,object_names=names,world_T_canonical=W,object_pose_wxyz=Q,observed=np.zeros((len(frames),len(names)),bool),planned_reference=np.ones((len(frames),len(names)),bool))
 value=dict(schema_version=1,episode=ep,episode_id=int(ep.split('_')[1]),fps=20.,frame_index=frames.tolist(),coordinate_frame='world',coord_frame='world',table_T_world=A.tolist(),object_names=names,pose_semantics='link_origin',objects=[assets[n]for n in names],poses_path=str(D/'poses.npz'),object_pose_wxyz=Q.tolist(),reconstruction_source='frozen_canonical11_SAM3D_original_RGB_fixed_metric',registrations=regs,experimental=True,observed=False,production_pass=False)
 save(D/'objects.json',value);return str(D/'objects.json')

def episode(ep):
 rows=[r for r in inputs()if r['episode']==ep];rows.sort(key=lambda r:r['object_index']);E=R/'outputs/track3'/ep;D=O/'episodes'/ep;D.mkdir(parents=True,exist_ok=True)
 key=sha(SOURCE)+sha(E/'objects/poses.npz')+''.join(sha(O/'assets'/r['name']/'report.json')for r in rows)
 cached=frozen(D/'report.json',key)
 if cached:return cached
 assets={r['name']:copy.deepcopy(read(O/'assets'/r['name']/'report.json')['asset'])for r in rows};Ts={};regs={}
 video=read(E/'calibrated_sequence/manifest.json')['source_video']
 for name,a in assets.items():
  a['source_inputs']+= [str(R/video),str(E/'calibrated_sequence/manifest.json'),str(E/'coarse-input.json')]
  a['source_usage']='Original episode RGB derived from recorded source MP4 and existing calibrated stereo used for rigid scene registration; canonical shape frozen from its recorded source.'
 for r in rows:Ts[r['name']],regs[r['name']]=register(ep,r,assets[r['name']])
 ref=np.load(E/'reference/reference_checkpoint.npz');frames=ref['frame_index'];A=ref['table_T_world'];duration=len(frames)/20.
 manifest=objects_manifest(ep,assets,Ts,A,frames,regs,'objects');m,q,ids=scene(ep,assets,Ts);first,endpoint=simulate(ep,m,q,ids,'original_initial',duration)
 # Exactly one settled-endpoint comparison; no angle/height/settle-duration search.
 settledTs={};obsrows=[];camz=np.load(E/'stereo_sequence/camera_poses.npz');cam=camz['world_T_rgb'][int(np.flatnonzero(camz['frame_index']==frames[0])[0])]
 for name,bid in zip(assets,ids):
  k=m.jnt_qposadr[m.body_jntadr[bid]];settled_world=np.linalg.inv(A)@Tof(endpoint[k:k+7]);C=np.linalg.inv(Ts[name][0])@settled_world;settledTs[name]=Ts[name]@C
  obs=observation(ep,name,int(frames[0]));mesh=trimesh.load(assets[name]['mesh_path'],force='mesh',process=False);ev,_,_=render_context(obs,mesh,256);_,before,_=ev(np.linalg.inv(cam)@Ts[name][0]);_,after,_=ev(np.linalg.inv(cam)@settled_world)
  db=before['depth_MAE_m'];da=after['depth_MAE_m'];dpb=before['depth_p95_m'];dpa=after['depth_p95_m'];ok=after['iou']>=before['iou']-.02 and db is not None and da is not None and da<=db+.005 and dpa<=dpb+.01 and after['depth_coverage']>=before['depth_coverage']-.02
  obsrows.append(dict(name=name,before=before,settled=after,observation_non_degradation=bool(ok),thresholds=dict(iou_decrease_max=.02,depth_MAE_increase_max_m=.005,depth_p95_increase_max_m=.01,coverage_decrease_max=.02),pose_change_translation_m=float(np.linalg.norm(settled_world[:3,3]-Ts[name][0,:3,3]))))
 settledmanifest=objects_manifest(ep,assets,settledTs,A,frames,regs,'settled_objects');second,_=simulate(ep,m,endpoint,ids,'settled_initial',duration);obsok=all(r['observation_non_degradation']for r in obsrows);adopt=bool(obsok and second['engineering_pass'])
 save(D/'settled-observation.json',dict(rows=obsrows,pass_non_degradation=obsok,GT_read=False,only_one_endpoint=True,zeroed_endpoint_velocity_declared=True))
 result=dict(state='completed',cache_key=key,episode=ep,object_names=list(assets),frames=len(frames),duration_s=duration,objects_json=manifest,settled_objects_json=settledmanifest,recommended_objects_json=settledmanifest if adopt else manifest,settled_initial_recommended=adopt,original_engineering_pass=first['engineering_pass'],settled_engineering_pass=second['engineering_pass'],settled_observation_pass=obsok,engineering_pass=bool(first['engineering_pass']or adopt),initial=first,settled=second,source_scene=str(E/'reference/scene_cpu.xml'),scene_xml=str(D/'scene.xml'),reference_parquet=None,parquet_adapter_owner='watchdog_review',shape_quality_pass=False,production_pass=False,registrations=regs,output_sha256={str(D/'objects/objects.json'):sha(D/'objects/objects.json'),str(D/'settled_objects/objects.json'):sha(D/'settled_objects/objects.json'),str(D/'scene.xml'):sha(D/'scene.xml')})
 save(D/'report.json',result);return result

def execute(stage,key,limit):
 D=O/('assets'if stage=='asset'else'episodes')/key;D.mkdir(parents=True,exist_ok=True);failure=D/'failure.json'
 if failure.exists():return dict(stage=stage,key=key,state='preserved_failure',failure=read(failure))
 with open(D/'run.log','a')as log:
  try:code=subprocess.run([sys.executable,str(SOURCE),'--stage',stage,'--key',key,'--output',str(O)],stdout=log,stderr=subprocess.STDOUT,timeout=limit).returncode
  except subprocess.TimeoutExpired:code=124
 (D/'run.exit').write_text(str(code)+'\n')
 if code:
  save(failure,dict(state='timeout'if code==124 else'failed',exit_code=code,wall_cap_seconds=limit,checkpoint_preserved=True,automatic_retry=False,existing_best_fallback='Original frozen self-built episode assets remain available; no unlabelled geometry substitution'))
 return dict(stage=stage,key=key,exit_code=code,state='completed'if code==0 else'failed',report=str(D/'report.json'))

def batch(stage):
 rows=inputs();names=sorted(set(r['name']for r in rows),key=lambda n:(n!='wooden_spoon',-sum(r['name']==n for r in rows),n));episodes=sorted(set(r['episode']for r in rows));tasks=[];O.mkdir(parents=True,exist_ok=True)
 protocol=dict(source_sha256=sha(SOURCE),input_manifest_sha256=sha(B/'all20-registration-inputs-v1/input-manifest.json'),fixed_shape=True,fixed_metric_scale=True,registration_max_calls_per_occurrence=400,registration_max_seconds=45,asset_max_seconds=240,episode_max_seconds=360,static_timestep_s=.001,full_duration='original reference T/20',settle='one first-full-duration endpoint, zero declared velocity, compare original f0, one second full duration',GT_read=False,production_pass=False)
 if (O/'protocol.json').exists():assert read(O/'protocol.json')==protocol,'Frozen batch protocol changed'
 else:save(O/'protocol.json',protocol)
 for name in names:
  tasks.append(execute('asset',name,240));save(O/'job.json',dict(state='running',phase='assets',heartbeat_utc=now(),tasks=tasks))
 if stage=='assets':return
 completed=[]
 for ep in episodes:
  missing=[r['name']for r in rows if r['episode']==ep and not(O/'assets'/r['name']/'report.json').exists()]
  if missing:
   tasks.append(dict(stage='episode',key=ep,state='asset_failure_fallback_old_scene',missing=missing));continue
  tasks.append(execute('episode',ep,360));p=O/'episodes'/ep/'report.json'
  if p.exists():completed.append(read(p))
  save(O/'job.json',dict(state='running',phase='scenes',heartbeat_utc=now(),completed_episodes=len(completed),tasks=tasks))
  save(O/'manifest.json',dict(state='running',episodes=[dict(episode=r['episode'],objects_json=r['recommended_objects_json'],object_manifest=r['recommended_objects_json'],original_objects_json=r['objects_json'],settled_objects_json=r['settled_objects_json'],reference_parquet=None,engineering_pass=r['engineering_pass'],production_pass=False,scene_xml=r['scene_xml'])for r in completed]))
 ar={n:read(O/'assets'/n/'report.json')for n in names if (O/'assets'/n/'report.json').exists()};oversize=sum(bool(ar[r['name']]['oversized_max_axis_gt_0_5m'])for r in rows if r['name']in ar)
 save(O/'report.json',dict(state='completed_bounded_batch',completed_utc=now(),canonical_assets_completed=len(ar),expected_assets=11,episodes_completed=len(completed),expected_episodes=20,object_instances=len(rows),oversized_instances=oversize,oversize_checked_instances=sum(r['name']in ar for r in rows),original_engineering_pass_count=sum(r['original_engineering_pass']for r in completed),settled_engineering_pass_count=sum(r['settled_engineering_pass']for r in completed),settled_observation_pass_count=sum(r['settled_observation_pass']for r in completed),engineering_pass_count=sum(r['engineering_pass']for r in completed),tasks=tasks,GT_read=False,production_pass=False,policy_rollout=False,rotation_veto=False))
 m=read(O/'manifest.json')if(O/'manifest.json').exists()else dict(episodes=[]);
 done={x['episode']for x in m['episodes']}
 for ep in episodes:
  if ep not in done:m['episodes'].append(dict(episode=ep,objects_json=str(R/'outputs/track3'/ep/'objects/objects.json'),object_manifest=str(R/'outputs/track3'/ep/'objects/objects.json'),reference_parquet=None,engineering_pass=False,production_pass=False,canonical_replacement=False,fallback='frozen_previous_self_built_assets_due_to_bounded_stage_failure'))
 m['state']='completed_bounded_batch';save(O/'manifest.json',m);save(O/'job.json',dict(state='completed_bounded_batch',completed_utc=now(),report=str(O/'report.json')))

def main():
 global O
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['batch','assets','asset','episode'],default='batch');p.add_argument('--key');p.add_argument('--output',type=Path,default=O);a=p.parse_args();O=a.output.resolve();O.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(2)
 if a.stage=='asset':asset(a.key)
 elif a.stage=='episode':episode(a.key)
 else:batch(a.stage)
if __name__=='__main__':main()
