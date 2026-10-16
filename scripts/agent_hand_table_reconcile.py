"""Bounded raw stereo / table / hand-coordinate diagnosis. No reference replacement."""
from pathlib import Path
import json,time,hashlib,traceback,ast
import numpy as np
from scipy.ndimage import binary_erosion,binary_dilation
from scipy.optimize import least_squares
from PIL import Image,ImageDraw
import trimesh
from agent_cup_reconstruct import ROOT,BASE,CAM,K,HOLDOUT,save,now
OUT=BASE/'hand_table_reconcile';TRAIN=[0,20,28,40,68,86,100,113,136];EVAL=sorted(set([28,68,86])|HOLDOUT)

def progress(detail,state='running'):
 p=ROOT/'notes/track3-live-progress.json';r=json.loads(p.read_text());r['hand_table_reconcile']={'state':state,'detail':detail,'report':str((OUT/'report.json').relative_to(ROOT)),'checked_at':now()};save(p,r)
def stats(v):
 v=np.asarray(v);v=v[np.isfinite(v)];return {'n':len(v),'p05_p50_p95':np.quantile(v,[.05,.5,.95]).tolist(),'min':float(v.min()),'max':float(v.max())} if len(v) else {'n':0,'p05_p50_p95':None,'min':None,'max':None}
def xyz(f,d,mask):
 y,x=np.where(mask&np.isfinite(d));z=d[y,x];v=np.c_[(x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z];return v@CAM[f,:3,:3].T+CAM[f,:3,3]
def plane_fit(p,n0):
 rng=np.random.default_rng(42);subset=p[::max(1,len(p)//6000)];best=np.zeros(len(subset),bool)
 for _ in range(300):
  tri=subset[rng.choice(len(subset),3,replace=False)];nn=np.cross(tri[1]-tri[0],tri[2]-tri[0]);nn/=max(np.linalg.norm(nn),1e-12);rr=(subset-tri[0])@nn;inside=abs(rr)<.006
  if inside.sum()>best.sum():best=inside
 good=subset[best];c=good.mean(0);_,_,vh=np.linalg.svd(good-c,full_matrices=False);n=vh[-1];n=n if n@n0>0 else -n;d=-n@c
 for _ in range(6):
  res=p@n+d;w=np.minimum(1,.004/np.maximum(abs(res),1e-9))*(abs(res)<.020);c=np.average(p,axis=0,weights=w);_,_,vh=np.linalg.svd((p-c)*np.sqrt(w[:,None]),full_matrices=False);n=vh[-1];n=n if n@n0>0 else -n;d=-n@c
 return n,float(d)
def main():
 OUT.mkdir(exist_ok=True);start=time.monotonic();paths=[Path(__file__),ROOT/'scripts/run_foundation_stereo_sample.py',ROOT/'scripts/check_hand_surface_depth.py',BASE/'hand_depth_tradeoff/alpha_050/hands_world.npz',BASE/'hand_registration_temporal/hands_world.npz',BASE/'hand_registration_depth/hands_world.npz',BASE/'metric_scene_diagnostic/hands_world.npz',BASE/'hot3d_hand_tracking/pipeline/mesh.npz',BASE/'hot3d_hand_tracking/pipeline/annotations.jsonl',BASE/'agent_cup_reconstruction/report.json',BASE/'stereo_sequence/camera_poses.npz',BASE/'flashchord_reference_preflight/reference_checkpoint.npz'];sha={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};key=hashlib.sha256(json.dumps(sha,sort_keys=True).encode()).hexdigest()
 if (OUT/'report.json').exists():
  old=json.loads((OUT/'report.json').read_text())
  if old.get('source_hash')==key and old.get('state')=='completed':print('Matching completed diagnosis reused');return
  if old.get('state')=='failed':save(OUT/f'failure_{int(time.time())}.json',old)
 report={'state':'running','started_utc':now(),'source_sha256':sha,'source_hash':key,'training':False,'training_ready':False,'reference_changed':False,'new_hand_pose_candidate':False,'frames':[],'train_plane_frames':TRAIN,'heldout_frames':sorted(HOLDOUT),'limitations':['Both stereo methods and segmentation are estimates, not GT.','RGB depth is LR-consistency-filtered and nearest-z projected from a different camera; visibility disagreement and holes remain.','Table ROI follows conservative interior of an approximate finite circle; this is not semantic table GT.','Hand hidden surfaces are inferred MANO geometry; visible-depth agreement alone cannot prove collision validity.']};save(OUT/'report.json',report);progress('核对raw stereo手内点、训练桌面背景平面、既有手候选及HOT3D帧编号；保持unknown。')
 try:
  geo=json.loads((BASE/'agent_cup_reconstruction/report.json').read_text());n=np.array(geo['table_plane']['normal_world']);d=geo['table_plane']['offset_m'];A=np.load(BASE/'flashchord_reference_preflight/reference_checkpoint.npz')['table_T_world'];W=np.linalg.inv(A);table=json.loads((BASE/'scene_contact_attribution/finite_table_candidate/report.json').read_text())['table_candidate'];center=np.array(table['center_table_xy_m']);radius=table['radius_m'];yy,xx=np.indices((767,1024));rays=np.stack([(xx-K[0,2])/K[0,0],(yy-K[1,2])/K[1,1],np.ones_like(xx)],-1)
  report['coordinate_audit']={'world':'rectified ego_cam_c frame0','camera':'calibrated RGB pinhole x-right/y-down/z-forward; metric camera-depth z, not Euclidean ray range','table':'right-handed Z-up, table z = n dot world_point + d','table_rotation_det':float(np.linalg.det(A[:3,:3])),'table_row_normal_max_error':float(abs(A[2,:3]-n).max()),'table_offset_error_m':float(abs(A[2,3]-d)),'roundtrip_error':float(abs(W@A-np.eye(4)).max()),'rgb_frame0_not_identity_max_difference':float(abs(CAM[0]-np.eye(4)).max()),'hot3d_mesh_frame_rule':'mesh.frame = annotations.frame_index + 1, explicit subtraction for comparison; hand 0=left/1=right','units':'metres; no scale refit','existing_reference_uses_hot3d':False}
  cache={};traincloud={'learned':[],'sgbm':[]}
  for f in sorted(set(TRAIN)|set(EVAL)):
   masks=np.load(BASE/f'hand_masks_sequence/frame_{f:06d}/masks.npz');cup=np.load(BASE/f'object_masks_temporal/masks/{f:06d}.npz')['masks'].any(0);hand=masks['masks'].any(0);depths={'learned':np.load(BASE/f'foundation_stereo_sequence/frame_{f:06d}/prediction.npz')['depth_rgb'],'sgbm':np.load(BASE/f'stereo_sequence/depth_rgb/{f:06d}.npz')['depth']};rw=rays@CAM[f,:3,:3].T;origin=CAM[f,:3,3];dist=-(origin@n+d)/(rw@n);onplane=origin+rw*dist[...,None];tab=onplane@A[:3,:3].T+A[:3,3];roi=(np.linalg.norm(tab[...,:2]-center,axis=-1)<radius-.035)&(dist>0)&~binary_dilation(hand|cup,iterations=12);cache[f]=(masks,cup,depths,roi);entry={'frame':f,'split':'holdout' if f in HOLDOUT else 'train','roi_pixels':int(roi.sum()),'methods':{}}
   for method,depth in depths.items():
    p=xyz(f,depth,roi&(depth>.15)&(depth<2));p=p[::max(1,len(p)//3000)];entry['methods'][method]={'valid_background_points':len(p),'original_plane_signed_residual_m':stats(p@n+d)}
    if len(p)>30:
     nf,df=plane_fit(p,n);entry['methods'][method]['local_fit']={'normal_world':nf.tolist(),'offset_m':df,'tilt_from_original_deg':float(np.degrees(np.arccos(np.clip(n@nf,-1,1)))),'residual_m':stats(p@nf+df)}
    if f in TRAIN:traincloud[method].append(p)
   save(OUT/f'background_{f:06d}.json',entry)
  fits={}
  for method,cloud in traincloud.items():
   p=np.concatenate(cloud);nf,df=plane_fit(p,n);fits[method]=(nf,df);report.setdefault('train_background_planes',{})[method]={'normal_world':nf.tolist(),'offset_m':df,'train_points':len(p),'tilt_from_original_deg':float(np.degrees(np.arccos(np.clip(n@nf,-1,1)))),'original_residual_m':stats(p@n+d),'fitted_residual_m':stats(p@nf+df),'consensus_under_6mm_fraction':float((abs(p@nf+df)<.006).mean()),'consensus_residual_m':stats((p@nf+df)[abs(p@nf+df)<.006])}
  for f,(_,_,depths,roi) in cache.items():
   row=json.loads((OUT/f'background_{f:06d}.json').read_text())
   for method,dep in depths.items():
    p=xyz(f,dep,roi&(dep>.15)&(dep<2));nf,df=fits[method];row['methods'][method]['train_plane_signed_residual_m']=stats(p@nf+df);row['methods'][method]['train_plane_inlier_fraction_6mm']=float((abs(p@nf+df)<.006).mean());row['methods'][method]['train_plane_consensus_residual_m']=stats((p@nf+df)[abs(p@nf+df)<.006])
   save(OUT/f'background_{f:06d}.json',row)
  save(OUT/'report.json',report);print('PLANE',report['train_background_planes'],flush=True)
  # Reuse exact project raster function without importing its cv2/argparse runner.
  src=ast.parse((ROOT/'scripts/check_hand_surface_depth.py').read_text());ns={'np':np};exec(compile(ast.Module(body=[x for x in src.body if isinstance(x,ast.FunctionDef) and x.name=='raster_depth'],type_ignores=[]),'project_raster','exec'),ns);render=ns['raster_depth'];faces=trimesh.load(BASE/'hand_sequence_hybrid/frame_000000/left_0.ply',process=False).faces
  candidates={label:np.load(BASE/path/'hands_world.npz') for label,path in [('metric_baseline','metric_scene_diagnostic'),('temporal','hand_registration_temporal'),('alpha_050','hand_depth_tradeoff/alpha_050'),('depth_full','hand_registration_depth')]};hot=np.load(BASE/'hot3d_hand_tracking/pipeline/mesh.npz');annotations={x['frame_index']:x for x in map(json.loads,(BASE/'hot3d_hand_tracking/pipeline/annotations.jsonl').read_text().splitlines())}
  for f in EVAL:
   path=OUT/f'frame_{f:06d}.json'
   if path.exists():
    old=json.loads(path.read_text())
    if old.get('source_hash')==key:report['frames'].append(old);continue
   masks,cup,depths,roi=cache[f];row={'source_hash':key,'frame':f,'split':'holdout' if f in HOLDOUT else 'train','hands':[]}
   for side,mask in zip(masks['is_right'],masks['masks']):
    hi=int(side);other=masks['masks'][masks['is_right']!=hi].any(0);region=binary_erosion(mask,iterations=3)&~binary_dilation(cup|other,iterations=3);both=region&np.isfinite(depths['learned'])&np.isfinite(depths['sgbm']);agree=both&(abs(depths['learned']-depths['sgbm'])<.01);handrow={'side':hi,'interior_pixels':int(region.sum()),'excluded_other_hand_overlap_pixels':int((binary_erosion(mask,iterations=3)&binary_dilation(other,iterations=3)).sum()),'both_methods_pixels':int(both.sum()),'agreement_under_10mm_pixels':int(agree.sum()),'cross_method_depth_difference_m':stats(depths['learned'][both]-depths['sgbm'][both]),'raw_stereo':{},'candidates':{}}
    for method,dep in depths.items():
     valid=region&np.isfinite(dep)&(dep>.15)&(dep<2);p=xyz(f,dep,valid);nf,df=fits[method];handrow['raw_stereo'][method]={'valid_pixels':int(valid.sum()),'invalid_fraction':float(1-valid.sum()/max(1,region.sum())),'signed_height_original_plane_m':stats(p@n+d),'signed_height_train_plane_m':stats(p@nf+df),'fraction_below_original_plane':float((p@n+d<-.001).mean()) if len(p) else None,'agreed_pixels_height_original_m':stats(xyz(f,dep,agree)@n+d)}
    variants={label:(z['vertices_world'][f,hi],{}) for label,z in candidates.items() if z['present'][f,hi]};ix=np.flatnonzero((hot['frame']==f+1)&(hot['hand']==hi));ah=[h for h in annotations[f]['hands'] if h['handedness']==('right' if hi else 'left')]
    if len(ix)==1 and len(ah)==1:variants['hot3d_independent']=(hot['verts'][ix[0]]@CAM[f,:3,:3].T+CAM[f,:3,3],{'frames_since_detection':ah[0].get('frames_since_detection'),'interpolated':ah[0].get('interpolated',False),'not_accepted_as_observed_reference':True})
    for label,(v,extra) in variants.items():
     camera=(v-CAM[f,:3,3])@CAM[f,:3,:3];rd=render(camera,faces,K);foot=np.isfinite(rd);height=v@n+d;cv={'mesh_min_table_z_m':float(height.min()),'mesh_vertices_below_1mm':int((height<-.001).sum()),'mask_iou':float((foot&mask).sum()/max(1,(foot|mask).sum())),'min_z_with_background_train_plane_m':{method:float((v@nf+df).min()) for method,(nf,df) in fits.items()},**extra}
     for method,dep in depths.items():
      valid=region&foot&np.isfinite(dep);cv[method]={'rendered_minus_depth_m':stats(rd[valid]-dep[valid]),'agreed_pixels_rendered_minus_depth_m':stats(rd[agree&foot]-dep[agree&foot])}
     handrow['candidates'][label]=cv
    row['hands'].append(handrow)
   save(path,row);report['frames'].append(row);save(OUT/'report.json',report);print('FRAME',f,[(h['side'],h['raw_stereo']['learned']['signed_height_original_plane_m']['p05_p50_p95'],{k:round(v['mesh_min_table_z_m']*1000,1) for k,v in h['candidates'].items()}) for h in row['hands']],flush=True);progress(f'已完成{len(report["frames"])}/{len(EVAL)}帧手mask有效双目、5类手候选与独立桌背景检查。')
   if f in [28,68,86]:
    im=Image.open(BASE/f'calibrated_sequence/frames/{f:06d}.jpg').convert('RGB');draw=ImageDraw.Draw(im);hmask=masks['masks'][np.flatnonzero(masks['is_right']==0)[0]];region=binary_erosion(hmask,iterations=3)&~binary_dilation(cup,iterations=3);y,x=np.where(region&np.isfinite(depths['learned']));p=xyz(f,depths['learned'],region);height=p@n+d
    for xx0,yy0,z0 in zip(x[::12],y[::12],height[::12]):draw.ellipse((xx0-1,yy0-1,xx0+1,yy0+1),fill=(240,40,40) if z0<0 else (40,240,80))
    draw.text((8,8),'Raw LR-filtered learned stereo: green above original table, red below; no GT',fill='yellow');im.save(OUT/f'raw_hand_depth_{f:06d}.jpg')
  report.update(state='completed',completed_utc=now(),elapsed_seconds=time.monotonic()-start,automatic_promotion=False,physics_rerun=False,physics_rerun_reason='No coordinate/scale bug or validated replacement established in this diagnostic; existing unchanged native contact evidence remains applicable.');report['gates']={'coordinate_transform_consistent':bool(report['coordinate_audit']['table_offset_error_m']<1e-8 and report['coordinate_audit']['table_row_normal_max_error']<1e-8),'raw_depth_and_table_crosscheck_completed':True,'heldout_candidate_comparison_completed':True,'reference_repaired':False,'geometry_gate':False,'training_ready':False};report['next_step']={'kind':'bounded_joint_hand_depth_and_table_consistency_fit','scope':'Use training frames in windows 24–32,64–72,82–90. Fit a small MANO translation/rotation candidate using perspective-correct hand surface residual on stereo-agreed visible pixels, independent keypoint/mask projections, temporal smoothness, and soft finite-table nonpenetration. Preserve scale/shape initially; inspect hidden palm violation separately. Freeze candidate settings before holdout 8–12/58–62/122–126 evaluation.','required_before_promotion':'Compare mask IoU/depth residual/plane residual versus alpha.5 and full-depth, preserve human missing NaNs, then rebuild matching robot reference and run native keyframe contacts. Reject if improvement relies on moving table against background evidence or worsens image gates.','do_not':'Do not select full-depth solely because its table penetration is smaller; do not treat learned stereo or HOT3D as GT.'};save(OUT/'report.json',report);progress('手-桌归因完成；坐标审计与原始双目/背景交叉检查已落盘，未自动替换手参考，下一步联合约束优化已明确。','completed')
 except Exception:
  report.update(state='failed',error=traceback.format_exc(),elapsed_seconds=time.monotonic()-start);save(OUT/'report.json',report);raise
if __name__=='__main__':main()
