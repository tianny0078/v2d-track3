"""Metric observation-supported coarse assets; no model inference, GT scans or training.

Generic visual surfaces preserve missing regions. Local finite-volume collision
patches do not globally convexify open pots/cups. Unknown poses remain NaN in the
measurement channel; finite held plans are separately marked for robot reference.
"""
from __future__ import annotations
import argparse, hashlib, json, math, time
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False));t.replace(p)
def transform(p,T): return p@T[:3,:3].T+T[:3,3]
def rigid(a,b):
 ca=a.mean(0);cb=b.mean(0);u,_,vt=np.linalg.svd((a-ca).T@(b-cb));R=vt.T@u.T
 if np.linalg.det(R)<0:vt[-1]*=-1;R=vt.T@u.T
 T=np.eye(4);T[:3,:3]=R;T[:3,3]=cb-R@ca;return T
def points(d,K,good):
 y,x=np.where(good);z=d[y,x];return np.c_[(x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z]
def subsample(p,n=1200):
 if len(p)>n:p=p[np.linspace(0,len(p)-1,n).astype(int)]
 return p
def register(template,observed,initial):
 if len(observed)<40:return initial,{'accepted':False,'reason':'insufficient_depth','points':len(observed)}
 tree=cKDTree(observed);T=initial.copy();# centroid displacement seeds translations while retaining previous orientation
 moved=transform(template,T);T[:3,3]+=np.median(observed,axis=0)-np.median(moved,axis=0)
 for _ in range(18):
  moved=transform(template,T);dist,ix=tree.query(moved,workers=1);good=dist<.06
  if good.sum()<30:break
  good&=dist<=np.quantile(dist[good],.75)
  if good.sum()<20:break
  delta=rigid(moved[good],observed[ix[good]]);T=delta@T
  if np.linalg.norm(delta[:3,3])<1e-5 and Rotation.from_matrix(delta[:3,:3]).magnitude()<1e-4:break
 dist,_=tree.query(transform(template,T));good=dist<.03;fraction=float(good.mean());res=np.quantile(dist[good],[.5,.95]).tolist() if good.any() else None
 ok=bool(good.sum()>=30 and fraction>=.2 and res[0]<.02 and np.isfinite(T).all())
 return T,{'accepted':ok,'reason':'trimmed_partial_ICP' if ok else 'ICP_support_failure','points':len(observed),'template_support_fraction':fraction,'residual_p50_p95_m':res}

class Inputs:
 def __init__(self,path):
  self.path=Path(path).resolve();self.cfg=json.loads(self.path.read_text());self.hashes={str(self.path):sha(self.path)}
  self.names=self.cfg['object_names'];assert self.names and len(set(self.names))==len(self.names)
  for name in self.names:
   if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):raise ValueError('Unsafe object name')
  z=np.load(self.file(self.cfg['camera_poses_path']),allow_pickle=False);self.cam=np.asarray(z[self.cfg.get('camera_poses_key','world_T_rgb')]);indices=z['frame_index'].tolist() if 'frame_index' in z else list(range(len(self.cam)));self.map={int(f):i for i,f in enumerate(indices)}
  self.frames=self.cfg.get('frame_ids',indices);self.K=np.load(self.file(self.cfg['K_path']),allow_pickle=False);self.fps=float(self.cfg['fps']);assert self.fps>0
  if any(int(f) not in self.map for f in self.frames):raise ValueError('Camera sequence incomplete for requested frames')
  if any(b<=a for a,b in zip(self.frames,self.frames[1:])):raise ValueError('frame_ids must be strictly increasing')
  for T in self.cam:
   if not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]) or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-3):raise ValueError('Invalid camera SE3')
 def path_for(self,key,f):
  value=self.cfg.get(key)
  if value is None:return None
  p=Path(value.format(frame=int(f)));return p if p.is_absolute() else self.path.parent/p
 def file(self,p):
  p=Path(p);p=p if p.is_absolute() else self.path.parent/p;p=p.resolve();self.hashes[str(p)]=sha(p);return p
 def load(self,f):
  dp=self.path_for('depth_pattern',f);mp=self.path_for('masks_pattern',f);rp=self.path_for('rgb_pattern',f)
  if any(p is None or not p.exists() for p in [dp,mp,rp]):return None
  d=np.load(self.file(dp),allow_pickle=False)[self.cfg.get('depth_key','depth')];z=np.load(self.file(mp),allow_pickle=False);names=z['object_names'].tolist()
  if set(names)!=set(self.names):raise ValueError(f'Object names mismatch at {f}: {names}')
  masks=np.asarray(z['masks'],bool)[[names.index(n) for n in self.names]]
  with Image.open(self.file(rp)) as image:size=image.size
  if d.shape!=masks.shape[1:] or size!=(d.shape[1],d.shape[0]):raise ValueError('RGB/depth/mask grid mismatch')
  hand=np.zeros(d.shape,bool);hp=self.path_for('hand_masks_pattern',f)
  if hp is not None and hp.exists():
   h=np.load(self.file(hp),allow_pickle=False)['masks'];hand=np.asarray(h,bool).any(axis=0) if h.ndim==3 else h.astype(bool)
   if hand.shape!=d.shape:raise ValueError('Hand grid mismatch')
  j=self.map[int(f)];K=self.K if self.K.ndim==2 else self.K[j]
  return d,masks,binary_dilation(hand,iterations=1),K,self.cam[j]
 def cloud(self,data,i):
  d,m,h,K,C=data;good=binary_erosion(m[i],iterations=1)&~h&~np.delete(m,i,axis=0).any(0)&np.isfinite(d)&(d>.1)&(d<3.)
  xyz=points(d,K,good)
  if len(xyz)>40:
   lo,hi=np.quantile(xyz[:,2],[.01,.99]);xyz=xyz[(xyz[:,2]>=lo)&(xyz[:,2]<=hi)]
  return subsample(transform(xyz,C)),good

def table_transform(inp,data):
 if inp.cfg.get('table_T_world') is not None:
  A=np.asarray(inp.cfg['table_T_world'],float)
  if A.shape!=(4,4) or not np.isfinite(A).all() or not np.allclose(A[:3,:3]@A[:3,:3].T,np.eye(3),atol=1e-5) or np.linalg.det(A[:3,:3])<.99:raise ValueError('Invalid table_T_world')
  return A,{'source':'explicit_input','measured_truth':False}
 d,m,h,K,C=data;y,x=np.where(m.any(0));yy,xx=np.indices(d.shape);roi=(xx>=max(0,x.min()-100))&(xx<=x.max()+100)&(yy>=max(0,y.min()-70))&(yy<=y.max()+130)
 valid=roi&~binary_dilation(m.any(0)|h,iterations=6)&np.isfinite(d)&(d>.1)&(d<2.)
 p=subsample(transform(points(d,K,valid),C),5000)
 if len(p)<100:raise ValueError('Insufficient background plane support; provide table_T_world')
 rng=np.random.default_rng(11);best=np.zeros(len(p),bool)
 for _ in range(200):
  s=p[rng.choice(len(p),3,replace=False)];n=np.cross(s[1]-s[0],s[2]-s[0]);n/=max(np.linalg.norm(n),1e-12);mask=abs((p-s[0])@n)<.008
  if mask.sum()>best.sum():best=mask
 if best.sum()<60:raise ValueError('No background plane consensus; provide table_T_world')
 c=p[best].mean(0);_,_,v=np.linalg.svd(p[best]-c,full_matrices=False);n=v[-1]
 if n@(C[:3,3]-c)<0:n=-n
 d0=float(-n@c);ex=np.eye(3)[np.argmin(abs(n))];ex=ex-(ex@n)*n;ex/=np.linalg.norm(ex);ey=np.cross(n,ex);A=np.eye(4);A[:3,:3]=np.stack([ex,ey,n]);A[2,3]=d0
 return A,{'source':'masked_background_RANSAC','assumed_table_not_semantically_verified':True,'normal_world':n.tolist(),'offset_m':d0,'points':len(p),'inliers':int(best.sum()),'residual_p50_p95_m':np.quantile(abs(p[best]@n+d0),[.5,.95]).tolist()}

def surface_mesh(data,i,T,stride=4):
 d,m,h,K,C=data;valid=m[i]&~h&~np.delete(m,i,axis=0).any(0)&np.isfinite(d)&(d>.1)&(d<3.);yy,xx=np.mgrid[0:d.shape[0]:stride,0:d.shape[1]:stride];z=d[yy,xx];ok=valid[yy,xx];verts=np.stack([(xx-K[0,2])*z/K[0,0],(yy-K[1,2])*z/K[1,1],z],-1).reshape(-1,3)
 ids=np.arange(z.size).reshape(z.shape);a=ids[:-1,:-1].ravel();b=ids[:-1,1:].ravel();c=ids[1:,1:].ravel();e=ids[1:,:-1].ravel();faces=np.vstack([np.c_[a,c,b],np.c_[a,e,c]]);keep=ok.ravel()[faces].all(1);faces=faces[keep]
 if len(faces):
  tri=verts[faces];edges=np.linalg.norm(tri-np.roll(tri,1,axis=1),axis=2);faces=faces[edges.max(1)<.025]
 if not len(faces):return None
 used,inv=np.unique(faces,return_inverse=True);v=transform(transform(verts[used],C),np.linalg.inv(T));mesh=trimesh.Trimesh(v,inv.reshape(-1,3),process=True);mesh.update_faces(mesh.nondegenerate_faces());mesh.remove_unreferenced_vertices();return mesh if len(mesh.faces) else None

def hollow_shell(s,t,bottom,segments=96):
 h=float(s['height_m']);r=float(s['bottom_outer_radius_m']);k=float(s['taper_dr_dz']);assert h>bottom>0 and min(r,r+k*h)>t>0
 a=np.arange(segments)*2*np.pi/segments;section=[(r,0),(r+k*h,h),(r+k*h-t,h),(r+k*bottom-t,bottom)];v=np.concatenate([np.c_[rr*np.cos(a),rr*np.sin(a),np.full(segments,z)]for rr,z in section]);v=np.vstack([v,[0,0,0],[0,0,bottom]]);f=[]
 for j in range(3):
  for i in range(segments):a0=j*segments+i;b=j*segments+(i+1)%segments;c=(j+1)*segments+(i+1)%segments;e=(j+1)*segments+i;f.extend([[a0,b,c],[a0,c,e]])
 for i in range(segments):f.extend([[4*segments,(i+1)%segments,i],[4*segments+1,3*segments+i,3*segments+(i+1)%segments]])
 mesh=trimesh.Trimesh(v,f,process=True);mesh.fix_normals();return mesh
def hollow_collisions(s,t,bottom):
 h=s['height_m'];r=s['bottom_outer_radius_m'];k=s['taper_dr_dz'];parts=[]
 for i in range(64):
  a=np.array([i,i+1])*2*np.pi/64;v=[[rr*np.cos(q),rr*np.sin(q),z]for z in [bottom,h]for rr in [r+k*z-t,r+k*z]for q in a];parts.append(trimesh.convex.convex_hull(v))
 a=np.arange(64)*2*np.pi/64;parts.append(trimesh.convex.convex_hull(np.concatenate([np.c_[(r+k*z)*np.cos(a),(r+k*z)*np.sin(a),np.full(64,z)]for z in [0,bottom]])));return parts
def surface_collisions(mesh,max_parts=128,thickness=.004):
 centers=mesh.triangles_center;voxel=max(.004,float(np.linalg.norm(mesh.extents))/24)
 while True:
  cells=np.floor(centers/voxel).astype(int);_,inv=np.unique(cells,axis=0,return_inverse=True)
  if inv.max()+1<=max_parts:break
  voxel*=1.2
 parts=[]
 for j in range(inv.max()+1):
  p=mesh.triangles[inv==j].reshape(-1,3);c=p.mean(0);_,_,vt=np.linalg.svd(p-c,full_matrices=False);R=vt.T
  if np.linalg.det(R)<0:R[:,-1]*=-1
  q=(p-c)@R;lo=q.min(0);hi=q.max(0);ext=np.maximum(hi-lo,[.002,.002,thickness]);T=np.eye(4);T[:3,:3]=R;T[:3,3]=c+R@((lo+hi)/2);parts.append(trimesh.creation.box(extents=ext,transform=T))
 return parts,{'method':'observation_local_PCA_boxes','parts':len(parts),'voxel_m':voxel,'minimum_slab_thickness_m':thickness,'unknown_backside_completed':False,'limitations':'Finite local collision slabs may overlap and thicken narrow rims; no global hull, no cavity cap, unobserved surfaces remain absent.'}

def export_asset(dest,name,mesh,parts,mass):
 dest.mkdir(parents=True,exist_ok=True);mesh.export(dest/'visual.obj');vol=np.array([abs(p.volume)for p in parts]);assert np.all(vol>0);weights=vol/vol.sum()*mass;centers=np.array([p.center_mass for p in parts]);com=(weights[:,None]*centers).sum(0)/mass;I=np.zeros((3,3))
 for p,w,c,v in zip(parts,weights,centers,vol):
  delta=c-com;I+=np.asarray(p.moment_inertia)*(w/v)+w*((delta@delta)*np.eye(3)-np.outer(delta,delta))
 assert np.linalg.eigvalsh(I).min()>0
 robot=ET.Element('robot',name=name);link=ET.SubElement(robot,'link',name=name);inertial=ET.SubElement(link,'inertial');ET.SubElement(inertial,'origin',xyz=' '.join(map(str,com)),rpy='0 0 0');ET.SubElement(inertial,'mass',value=str(mass));ET.SubElement(inertial,'inertia',**{k:str(v)for k,v in zip(['ixx','ixy','ixz','iyy','iyz','izz'],[I[0,0],I[0,1],I[0,2],I[1,1],I[1,2],I[2,2]])})
 visual=ET.SubElement(link,'visual');ET.SubElement(ET.SubElement(visual,'geometry'),'mesh',filename='visual.obj')
 for j,p in enumerate(parts):
  file=f'collision_{j:03d}.obj';p.export(dest/file);coll=ET.SubElement(link,'collision');ET.SubElement(ET.SubElement(coll,'geometry'),'mesh',filename=file)
 ET.indent(robot);ET.ElementTree(robot).write(dest/'object.urdf',encoding='unicode')
 vertices=np.vstack([mesh.vertices]+[p.vertices for p in parts]);radius=float(np.linalg.norm(vertices-com,axis=1).max());physics={'mass_assumed_kg':mass,'inertia_eigenvalues':np.linalg.eigvalsh(I).tolist(),'collision_parts':len(parts),'volume_sum_m3':float(vol.sum()),'COM_is_collision_volume_weighted':True,'overlap_mass_double_counting_possible':True}
 try:
  import mujoco
  model=mujoco.MjModel.from_xml_path(str(dest/'object.urdf'));data=mujoco.MjData(model);mujoco.mj_forward(model,data);physics.update(mujoco_urdf_compiles=True,mujoco_geoms=int(model.ngeom),finite_state=bool(np.isfinite(data.xpos).all()),dynamics_tested=False)
 except ImportError:physics.update(mujoco_urdf_compiles=None,dynamics_tested=False)
 return {'name':name,'mesh_path':str((dest/'visual.obj').resolve()),'urdf_path':str((dest/'object.urdf').resolve()),'canonical_COM':com.tolist(),'bounding_radius_m':radius,'mass_kg':mass,'physics':physics,'geometry_quality':{'vertices':len(mesh.vertices),'faces':len(mesh.faces),'area_m2':float(mesh.area),'watertight':bool(mesh.is_watertight),'winding_consistent':bool(mesh.is_winding_consistent),'unobserved_geometry_unknown':True}}

def run(args):
 start=time.monotonic();deadline=start+args.max_seconds;inp=Inputs(args.manifest);out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True);key=sha(inp.path)+sha(__file__)
 if (out/'objects.json').exists() and not args.overwrite:raise ValueError('Completed output exists; use a new output directory or explicit --overwrite')
 Tn=len(inp.frames);B=len(inp.names);pose=np.full((Tn,B,4,4),np.nan);measured=pose.copy();observed=np.zeros((Tn,B),bool);accepted=observed.copy();confidence=np.zeros((Tn,B));clouds={};stats=[];anchors={};table=None;plane=None
 # One pass extracts metric point support; no learned inference and no dense point cloud cache.
 for j,f in enumerate(inp.frames):
  if time.monotonic()>deadline:raise TimeoutError('Input extraction budget exhausted; increase --max-seconds')
  data=inp.load(f)
  if data is None:continue
  if table is None and data[1].any():table,plane=table_transform(inp,data)
  for i in range(B):
   p,_=inp.cloud(data,i);clouds[j,i]=p;observed[j,i]=len(p)>=40
   if len(p)>=60 and i not in anchors:anchors[i]=j
 if len(anchors)!=B:raise ValueError('Object lacks >=60 measured depth points; no synthetic primitive substituted')
 save(out/'input_support.json',{'frame_index':inp.frames,'object_names':inp.names,'anchors':{inp.names[i]:inp.frames[j]for i,j in anchors.items()},'depth_points':{f'{j}:{i}':len(p)for (j,i),p in clouds.items()},'input_sha256':inp.hashes})
 assets=[]
 for i,name in enumerate(inp.names):
  j0=anchors[i];p=clouds[j0,i];prior=inp.cfg.get('same_object_assets',{}).get(name);initial=np.eye(4)
  if prior:
   if prior['object_name']!=name:raise ValueError('Same-object reuse identity mismatch')
   provenance=inp.file(prior['provenance']);record=json.loads(provenance.read_text());source_objects=record.get('objects',{})
   if not isinstance(source_objects,dict) or name not in source_objects or 'shape' not in source_objects[name]:raise ValueError('Prior provenance must identify this exact object and its self-built shape')
   s=prior['shape'];reference_shape=source_objects[name]['shape']
   if any(not np.isclose(float(s[k]),float(reference_shape[k]),rtol=1e-8,atol=1e-10) for k in ['height_m','bottom_outer_radius_m','taper_dr_dz']):raise ValueError('Prior shape does not match named provenance; use a documented new reconstruction')
   wall=float(prior.get('wall_m',.003));bottom=float(prior.get('bottom_m',.003));visual=hollow_shell(s,wall,bottom);template=subsample(visual.vertices)
   if prior.get('initial_world_T_canonical') is not None:
    initial=np.asarray(prior['initial_world_T_canonical'],float)
    if initial.shape!=(4,4) or not np.isfinite(initial).all() or not np.allclose(initial[3],[0,0,0,1]) or not np.allclose(initial[:3,:3].T@initial[:3,:3],np.eye(3),atol=1e-5) or np.linalg.det(initial[:3,:3])<.99:raise ValueError('Prior initial pose must be rigid SE3 for this episode')
   else:
    initial[:3,:3]=table[:3,:3].T;initial[:3,3]=np.median(p,axis=0)-initial[:3,:3]@np.array([0,0,s['height_m']/2])
   initial,stat=register(template,p,initial)
  else:
   initial[:3,3]=np.median(p,axis=0);template=p-initial[:3,3];stat={'accepted':True,'reason':'observed_anchor_reference','points':len(p),'rotation_observability':'initial world axes convention; not measured orientation'}
  current=initial.copy()
  for j,f in enumerate(inp.frames):
   if j<j0:continue
   if time.monotonic()>deadline:stat={'accepted':False,'reason':'tracking_budget_exhausted','points':len(clouds.get((j,i),[]))};candidate=current
   elif j==j0:candidate=initial
   else:candidate,stat=register(template,clouds.get((j,i),np.empty((0,3))),current)
   ok=bool(stat['accepted']);accepted[j,i]=ok
   if ok:current=candidate.copy();measured[j,i]=current;confidence[j,i]=min(.7,stat.get('template_support_fraction',1.)*.7)
   pose[j,i]=current;stats.append({'frame':int(f),'object':name,**stat,'confidence_is_heuristic_not_probability':True})
  pose[:j0,i]=initial
  if not accepted[:,i].any():raise ValueError(f'{name}: no accepted registration; prior asset unregistered')
  selected=np.flatnonzero(accepted[:,i]);selected=selected[np.unique(np.linspace(0,len(selected)-1,min(8,len(selected))).astype(int))]
  if prior:parts=hollow_collisions(s,wall,bottom);source={'method':'same_named_object_self_built_hollow_shell_registered_to_episode','provenance':prior['provenance'],'shape':s,'wall_m':wall,'bottom_m':bottom,'thickness_is_assumption':True};support_frames=[int(inp.frames[j0])]
  else:
   meshes=[];support_frames=[]
   for j in selected:
    data=inp.load(inp.frames[j]);mesh=surface_mesh(data,i,pose[j,i],stride=args.surface_stride)
    if mesh is not None:meshes.append(mesh);support_frames.append(int(inp.frames[j]))
   if not meshes:raise ValueError(f'{name}: no valid observed triangles; no primitive substituted')
   visual=trimesh.util.concatenate(meshes);visual.merge_vertices(digits_vertex=5);visual.update_faces(visual.unique_faces());visual.update_faces(visual.nondegenerate_faces());visual.remove_unreferenced_vertices();parts,source=surface_collisions(visual,args.max_collision_parts)
  if not np.isfinite(visual.vertices).all() or len(visual.faces)==0 or visual.area<=0:raise ValueError('Invalid observed mesh')
  asset=export_asset(out/'assets'/name,name,visual,parts,float(inp.cfg.get('object_mass_kg',{}).get(name,.15)));asset.update(geometry_source=source,support_frame_ids=support_frames,accepted_pose_count=int(accepted[:,i].sum()),observed_depth_frame_count=int(observed[:,i].sum()));assets.append(asset);save(out/f'{name}-scorecard.json',asset)
  print(name,'triangles',len(visual.faces),'accepted',int(accepted[:,i].sum()),'/',Tn,flush=True)
 if not np.isfinite(pose).all():raise ValueError('Nonfinite planned pose')
 q=Rotation.from_matrix(pose[:,:,:3,:3].reshape(-1,3,3)).as_quat().reshape(Tn,B,4);wxyz=np.concatenate([pose[:,:,:3,3],q[...,3:],q[...,:3]],axis=-1)
 np.savez_compressed(out/'poses.npz',frame_index=np.asarray(inp.frames),object_names=np.asarray(inp.names),object_pose_wxyz=wxyz,world_T_canonical=pose,measured_world_T_canonical=measured,observed=observed,accepted=accepted,confidence=confidence,planned_reference=~accepted)
 result={'schema_version':1,'state':'completed','episode_id':inp.cfg['episode_id'],'fps':inp.fps,'frame_index':inp.frames,'coordinate_frame':'world','table_T_world':table.tolist(),'table_plane_evidence':plane,'object_names':inp.names,'pose_semantics':'link_origin','poses_path':str(out/'poses.npz'),'objects':assets,'training_ready':False,'baseline_assets_valid':all(a['physics'].get('mujoco_urdf_compiles') is not False for a in assets),'geometry_complete':False,'source_manifest':str(out/'input_manifest.json'),'limitations':['Coarse video-derived assets, no GT scans; partial surfaces and rotation ambiguity remain.','Rejected/missing measurements remain NaN; finite pose holds are explicit robot-reference plans.','Surface collision slabs are incomplete and can overlap; mass .15kg default is an assumption.','Image projection/heldout/physical contact quality is diagnostic, not claimed passed.']}
 save(out/'input_manifest.json',{'input':inp.cfg,'source_sha256':dict(inp.hashes,**{str(Path(__file__).resolve()):sha(__file__)}),'schema_cache_key':key});save(out/'tracking_scorecard.json',stats);save(out/'objects.json',result);save(out/'manifest.json',result);save(out/'report.json',dict(result,completed_utc=now(),elapsed_s=time.monotonic()-start));return result
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--max-seconds',type=float,default=300);p.add_argument('--surface-stride',type=int,default=4);p.add_argument('--max-collision-parts',type=int,default=128);p.add_argument('--overwrite',action='store_true');a=p.parse_args()
 if a.surface_stride<1 or a.max_collision_parts<1 or a.max_seconds<=0:p.error('positive budgets required')
 try:run(a)
 except Exception as e:save(Path(a.output)/'failure.json',{'state':'failed','time':now(),'error_type':type(e).__name__,'error':str(e),'command':' '.join(__import__('sys').argv)});raise
if __name__=='__main__':main()
