"""Independent SE3 hand candidate with rendering, stereo, 2D, time and finite table."""
from pathlib import Path
import ast,json,time,traceback,hashlib
import numpy as np,warp as wp,trimesh
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from scipy.ndimage import binary_erosion,binary_dilation
from PIL import Image,ImageDraw
from agent_cup_reconstruct import ROOT,BASE,K,CAM,HOLDOUT,save,now
from agent_hand_table_raster import triangles
OUT=BASE/'hand_table_joint_fit/v2_keypoint_balanced';TRAIN=list(range(24,33))+list(range(64,73))+list(range(82,91));PILOT=[28,68,86];WINDOWS=[list(range(24,33)),list(range(64,73)),list(range(82,91))]
def progress(detail,state='running'):
 p=ROOT/'notes/track3-live-progress.json';r=json.loads(p.read_text());r['hand_table_joint_fit']={'state':state,'detail':detail,'report':str((OUT/'report.json').relative_to(ROOT)),'checked_at':now()};save(p,r)
def huber(x):
 a=abs(np.asarray(x));return np.where(a<=1,.5*a*a,a-.5)
def main():
 OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic();paths=[Path(__file__),ROOT/'scripts/agent_hand_table_raster.py',BASE/'hand_depth_tradeoff/alpha_050/hands_world.npz',BASE/'hand_registration_depth/hands_world.npz',BASE/'hand_detector_mediapipe_sequence/report.json',BASE/'hand_sequence_hybrid/identity_audit.json',BASE/'agent_cup_reconstruction/report.json',BASE/'stereo_sequence/camera_poses.npz'];sha={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};key=hashlib.sha256(json.dumps(sha,sort_keys=True).encode()).hexdigest();old=json.loads((OUT/'report.json').read_text()) if (OUT/'report.json').exists() else {}
 if old.get('state')=='completed' and old.get('source_hash')==key:print('Matching completed joint fit reused');return
 if old.get('state')=='failed':save(OUT/f'failure_{int(time.time())}.json',old)
 report={'state':'running','started_utc':now(),'source_hash':key,'source_sha256':sha,'training':False,'training_ready':False,'reference_replaced':False,'candidate_is_estimate_not_GT':True,'fit_hands':['left'],'unchanged_right_hand':True,'train_frames':TRAIN,'heldout_frames':sorted(HOLDOUT),'fixed_settings':{'resolution_factor':4,'depth_agreement_m':.01,'translation_component_bound_m':.08,'rotation_component_bound_rad':.35,'depth_weight':2.,'mask_iou_weight':8.,'keypoint_weight':2.,'table_max_penetration_weight':3.,'table_scale_m':.005,'temporal_weight':.2,'pilot_maxfev_per_seed':900,'expanded_maxfev_per_frame':500,'reverse_smooth_maxfev':250,'keypoint_median_increase_gate_px':2.},'budget_seconds':1050,'pilot':[],'fit_progress':[],'gates':{'reference_repaired':False,'training_ready':False}};save(OUT/'report.json',report);progress('开始固定MANO形状/尺度的SE3联合拟合；先28/68/86三帧双初始化pilot，留出不参与。')
 try:
  a=np.load(BASE/'hand_depth_tradeoff/alpha_050/hands_world.npz');full=np.load(BASE/'hand_registration_depth/hands_world.npz');geo=json.loads((BASE/'agent_cup_reconstruction/report.json').read_text());n=np.array(geo['table_plane']['normal_world']);d=geo['table_plane']['offset_m'];A=np.load(BASE/'flashchord_reference_preflight/reference_checkpoint.npz')['table_T_world'];table=json.loads((BASE/'scene_contact_attribution/finite_table_candidate/report.json').read_text())['table_candidate'];tc=np.array(table['center_table_xy_m']);radius=table['radius_m'];faces=np.asarray(trimesh.load(BASE/'hand_sequence_hybrid/frame_000000/left_0.ply',process=False).faces,np.int32);mp=json.loads((BASE/'hand_detector_mediapipe_sequence/report.json').read_text());assignment=json.loads((BASE/'hand_sequence_hybrid/identity_audit.json').read_text())['assignments'];raw=json.loads((BASE/'hand_sequence_hybrid/report.json').read_text());wp.init();facewp=wp.array(faces,dtype=wp.int32,device='cpu');vw=wp.empty(778,dtype=wp.vec3,device='cpu');dbuf={};data={}
  def raster(v,scale=4):
   offset=2 if scale==4 else 0;shape=(len(range(offset,767,scale)),len(range(offset,1024,scale)));h,w=shape
   if scale not in dbuf:dbuf[scale]=wp.empty(h*w,dtype=float,device='cpu')
   depth=dbuf[scale];depth.fill_(float('inf'));vw.assign(np.asarray(v,np.float32));wp.launch(triangles,len(faces),inputs=[vw,facewp,depth,w,h,float(K[0,0]/scale),float(K[1,1]/scale),float((K[0,2]-offset)/scale),float((K[1,2]-offset)/scale)],device='cpu');r=depth.numpy().copy().reshape(shape);r[~np.isfinite(r)]=np.nan;return r
  for f in sorted(set(TRAIN)|HOLDOUT):
   masks=np.load(BASE/f'hand_masks_sequence/frame_{f:06d}/masks.npz');mask=masks['masks'][np.flatnonzero(masks['is_right']==0)[0]];other=masks['masks'][masks['is_right']!=0].any(0);cups=np.load(BASE/f'object_masks_temporal/masks/{f:06d}.npz')['masks'].any(0);dl=np.load(BASE/f'foundation_stereo_sequence/frame_{f:06d}/prediction.npz')['depth_rgb'];ds=np.load(BASE/f'stereo_sequence/depth_rgb/{f:06d}.npz')['depth'];valid=binary_erosion(mask,iterations=3)&~binary_dilation(cups|other,iterations=3)&np.isfinite(dl)&np.isfinite(ds)&(abs(dl-ds)<.01)&(dl>.15)&(dl<2);target=np.full_like(dl,np.nan);target[valid]=.5*(dl[valid]+ds[valid]);R=CAM[f,:3,:3];t=CAM[f,:3,3];v=(a['vertices_world'][f,0]-t)@R;j=(a['joints_world'][f,0]-t)@R;c=v.mean(0);kp=None
   if raw['frames'][f]['box_source'].startswith('MediaPipe'):
    for side,index in zip(assignment[f]['suggested_is_right'],assignment[f]['source_indices']):
     if side==0:kp=np.asarray(mp['frames'][f]['hands'][index]['normalized_landmarks'])[:,:2]*[1024,767]
   data[f]={'v':v,'j':j,'center':c,'mask':mask,'small_mask':mask[2::4,2::4],'valid':valid,'target':target,'small_target':target[2::4,2::4],'other':other|cups,'kp':kp,'full_seed':np.r_[((full['vertices_world'][f,0]-a['vertices_world'][f,0]).mean(0))@R,np.zeros(3)]}
  # Validate accelerated renderer against the existing NumPy full-resolution reference.
  src=ast.parse((ROOT/'scripts/check_hand_surface_depth.py').read_text());ns={'np':np};exec(compile(ast.Module(body=[x for x in src.body if isinstance(x,ast.FunctionDef) and x.name=='raster_depth'],type_ignores=[]),'numpy_raster','exec'),ns);ref=ns['raster_depth'](data[28]['v'],faces,K)[2::4,2::4];test=raster(data[28]['v']);both=np.isfinite(ref)&np.isfinite(test);val={'support_disagreement_pixels':int((np.isfinite(ref)!=np.isfinite(test)).sum()),'max_common_depth_difference_m':float(abs(ref[both]-test[both]).max()),'pass':bool((np.isfinite(ref)!=np.isfinite(test)).sum()<=4 and abs(ref[both]-test[both]).max()<.0001)};save(OUT/'raster_validation.json',val);assert val['pass'],val
  def transform(f,p):
   D=data[f];rot=Rotation.from_rotvec(p[3:]).as_matrix();return (D['v']-D['center'])@rot.T+D['center']+p[:3],(D['j']-D['center'])@rot.T+D['center']+p[:3]
  calls=0
  def evaluate(f,p,neighbor=None,scale=4,metrics=False):
   nonlocal calls;calls+=1;D=data[f];v,j=transform(f,p);rendered=raster(v,scale);mask=D['small_mask'] if scale==4 else D['mask'];target=D['small_target'] if scale==4 else D['target'];foot=np.isfinite(rendered);valid=np.isfinite(target);support=foot&valid;res=rendered[support]-target[support];iou=float((foot&mask).sum()/max(1,(foot|mask).sum()));coverage=float(support.sum()/max(1,valid.sum()));world=v@CAM[f,:3,:3].T+CAM[f,:3,3];tab=world@A[:3,:3].T+A[:3,3];inside=np.linalg.norm(tab[:,:2]-tc,axis=1)<radius;pen=np.maximum(-tab[inside,2],0);maxpen=float(pen.max()) if len(pen) else 0.;kp=D['kp'];kperr=np.array([])
   if kp is not None:
    uv=j@K.T;uv=uv[:,:2]/uv[:,2:];vis=(kp[:,0]>=4)&(kp[:,0]<1020)&(kp[:,1]>=4)&(kp[:,1]<763);kperr=np.linalg.norm(uv[vis]-kp[vis],axis=1)
   terms={'depth':2*float(np.mean(huber(res/.01))) if len(res)>30 else 100.,'mask':8*(1-iou),'coverage':6*max(0,.8-coverage)**2,'keypoints':2.*float(np.mean(huber(kperr/8))) if len(kperr) else 0.,'table':3*(maxpen/.005)**2+.5*float(np.mean((pen/.005)**2)) if len(pen) else 0.,'prior':.05*float(np.sum((p[:3]/.06)**2)+np.sum((p[3:]/.25)**2)),'temporal':.2*float(np.mean(((p-neighbor)/np.r_[[.01]*3,[.10]*3])**2)) if neighbor is not None else 0.};loss=sum(terms.values())
   if not metrics:return loss
   return {'loss':loss,'terms':terms,'mask_iou':iou,'valid_depth_coverage':coverage,'supported_pixels':int(support.sum()),'absolute_depth_p50_p95_m':np.quantile(abs(res),[.5,.95]).tolist() if len(res) else None,'signed_depth_median_m':float(np.median(res)) if len(res) else None,'visible_keypoint_error_p50_p95_px':np.quantile(kperr,[.5,.95]).tolist() if len(kperr) else None,'finite_table_max_penetration_m':maxpen,'mesh_min_table_z_m':float(tab[:,2].min()),'table_inside_vertices':int(inside.sum()),'parameters_camera_translation_rotvec':p.tolist()}
  bounds=[(-.08,.08)]*3+[(-.35,.35)]*3;params={f:data[f]['full_seed'].copy() for f in TRAIN}
  def fit(f,seed,neighbor,maxfev,label):
   dest=OUT/f'{label}_{f:06d}.json'
   if dest.exists():
    rr=json.loads(dest.read_text())
    if rr.get('source_hash')==key:return np.array(rr['parameters']),rr
   before=time.monotonic();rr0=evaluate(f,seed,neighbor,metrics=True);result=minimize(lambda p:evaluate(f,p,neighbor),seed,method='Powell',bounds=bounds,options={'maxfev':maxfev,'maxiter':12,'xtol':.0005,'ftol':.002});chosen=result.x if result.fun<rr0['loss'] else seed;rr={'source_hash':key,'frame':f,'label':label,'parameters':chosen.tolist(),'before':rr0,'after':evaluate(f,chosen,neighbor,metrics=True),'nfev':int(result.nfev),'optimizer_success':bool(result.success),'message':str(result.message),'elapsed_seconds':time.monotonic()-before};save(dest,rr);return chosen,rr
  for f in PILOT:
   trials=[]
   for label,seed in [('pilot_alpha',np.zeros(6)),('pilot_full',data[f]['full_seed']),('pilot_full_rxminus',data[f]['full_seed']+np.array([0,0,0,-.20,0,0])),('pilot_full_rxplus',data[f]['full_seed']+np.array([0,0,0,.20,0,0]))]:
    p,rr=fit(f,seed,None,900,label);trials.append((p,rr))
   p,rr=min(trials,key=lambda x:x[1]['after']['loss']);params[f]=p;report['pilot'].append({'frame':f,'selected_seed':rr['label'],'elapsed_seconds':sum(x[1]['elapsed_seconds'] for x in trials),'before_alpha':evaluate(f,np.zeros(6),metrics=True),'before_full':evaluate(f,data[f]['full_seed'],metrics=True),'selected':evaluate(f,p,metrics=True)});save(OUT/'report.json',report);print('PILOT',report['pilot'][-1],flush=True);progress(f'三帧联合拟合pilot已完成{len(report["pilot"])}/3；图像、深度与桌面项共同选择，不按深度单独选。')
  # Interpolate pilot anchors inside each TRAIN window as seeds, then temporal block updates.
  for window,pilot in zip(WINDOWS,PILOT):
   for f in window:
    if f!=pilot:params[f]=params[pilot].copy()
  for passid,order,maxfev in [('forward',TRAIN,500),('reverse',list(reversed(TRAIN)),250)]:
   for f in order:
    if time.monotonic()-start>900:raise TimeoutError('Compute guard reached; checkpoints retained')
    neigh=[params[g] for g in [f-1,f+1] if g in params];neighbor=np.mean(neigh,axis=0) if neigh else None;p,rr=fit(f,params[f],neighbor,maxfev,passid);params[f]=p;report['fit_progress'].append({'pass':passid,'frame':f,'after':rr['after'],'elapsed_seconds':rr['elapsed_seconds']});np.savez_compressed(OUT/'train_checkpoint.npz',frames=np.array(TRAIN),parameters=np.stack([params[f] for f in TRAIN]));save(OUT/'report.json',report)
   print('PASS',passid,'elapsed',time.monotonic()-start,flush=True);progress(f'已完成{passid}训练窗口时序联合块优化，正在保留固定超参的验证。')
  # Fixed transfer rule: interpolate camera-frame SE3 corrections between TRAIN anchors;
  # outside TRAIN support taper linearly to zero over 12 frames, avoiding arbitrary distant extrapolation.
  field=np.zeros((137,6));xp=np.array(TRAIN);fp=np.stack([params[f] for f in TRAIN])
  for k in range(6):field[:,k]=np.interp(np.arange(137),xp,fp[:,k])
  for f in range(137):
   if f<min(TRAIN):field[f]*=max(0,1-(min(TRAIN)-f)/12)
   if f>max(TRAIN):field[f]*=max(0,1-(f-max(TRAIN))/12)
  vertices=a['vertices_world'].copy();joints=a['joints_world'].copy();delta=a['world_rigid_delta'].copy();extra=np.tile(np.eye(4),(137,2,1,1))
  for f in range(137):
   if not a['present'][f,0]:continue
   R=CAM[f,:3,:3];t=CAM[f,:3,3];v=(a['vertices_world'][f,0]-t)@R;j=(a['joints_world'][f,0]-t)@R;c=v.mean(0);rr=Rotation.from_rotvec(field[f,3:]).as_matrix();trans=c+field[f,:3]-rr@c;B=np.eye(4);B[:3,:3]=rr;B[:3,3]=trans;Q=CAM[f]@B@np.linalg.inv(CAM[f]);extra[f,0]=Q;vertices[f,0]=a['vertices_world'][f,0]@Q[:3,:3].T+Q[:3,3];joints[f,0]=a['joints_world'][f,0]@Q[:3,:3].T+Q[:3,3];delta[f,0]=Q@delta[f,0]
  extra[~a['present']]=np.nan;assert np.isnan(vertices[~a['present']]).all();assert np.array_equal(vertices[:,1],a['vertices_world'][:,1],equal_nan=True);np.savez_compressed(OUT/'hands_world_candidate.npz',frame_index=a['frame_index'],present=a['present'],vertices_world=vertices,joints_world=joints,world_rigid_delta=delta,candidate_delta_world=extra,parameters_camera_translation_rotvec=field,optimized_train_frame=np.isin(np.arange(137),TRAIN),heldout_frame=np.isin(np.arange(137),list(HOLDOUT)))
  report['transfer_rule']='Camera-frame SE3 interpolation across TRAIN anchors, outside [24,90] linearly tapers to zero over12 frames. Rule fixed before heldout evaluation. Old baseline estimates are not GT; heldout corrections never directly optimized.';evaluation=[]
  for f in sorted(set(TRAIN)|HOLDOUT):
   rows={label:evaluate(f,p,scale=1,metrics=True) for label,p in [('alpha_050',np.zeros(6)),('depth_full',data[f]['full_seed']),('candidate',field[f])]};item={'frame':f,'split':'holdout' if f in HOLDOUT else 'train',**rows};evaluation.append(item);save(OUT/f'eval_{f:06d}.json',item)
   if f in PILOT or f in [10,60,124]:
    im=Image.open(BASE/f'calibrated_sequence/frames/{f:06d}.jpg').convert('RGB');dr=ImageDraw.Draw(im)
    for label,p,col in [('alpha',np.zeros(6),(255,80,80)),('candidate',field[f],(40,230,80))]:
     v,j=transform(f,p);uv=j@K.T;uv=uv[:,:2]/uv[:,2:]
     for x,y in uv:dr.ellipse((x-2,y-2,x+2,y+2),fill=col)
    dr.text((8,8),f'frame{f}: red alpha.5 / green SE3 candidate; candidate, not GT',fill='yellow');im.save(OUT/f'overlay_{f:06d}.jpg')
  report['evaluation']=evaluation;report['evaluations_calls']=calls;g={}
  for split in ['train','holdout']:
   rows=[r for r in evaluation if r['split']==split];g[split]={'max_finite_table_penetration_m':max(x['candidate']['finite_table_max_penetration_m'] for x in rows),'worst_mask_iou_drop':max(x['alpha_050']['mask_iou']-x['candidate']['mask_iou'] for x in rows),'worst_abs_depth_median_m':max(x['candidate']['absolute_depth_p50_p95_m'][0] for x in rows),'median_abs_depth_median_m':float(np.median([x['candidate']['absolute_depth_p50_p95_m'][0] for x in rows])),'minimum_valid_depth_coverage':min(x['candidate']['valid_depth_coverage'] for x in rows),'worst_keypoint_median_increase_px':max(x['candidate']['visible_keypoint_error_p50_p95_px'][0]-x['alpha_050']['visible_keypoint_error_p50_p95_px'][0] for x in rows if x['candidate']['visible_keypoint_error_p50_p95_px'] is not None)};g[split]['pass']=bool(g[split]['max_finite_table_penetration_m']<=.001 and g[split]['worst_mask_iou_drop']<=.02 and g[split]['worst_abs_depth_median_m']<=.01 and g[split]['worst_keypoint_median_increase_px']<=2.)
  report['gates'].update(g);report['gates']['human_NaNs_and_right_hand_preserved']=True;report['gates']['candidate_image_geometry_pass']=bool(g['train']['pass'] and g['holdout']['pass']);report['robot_reference_rebuilt']=False;report['physics_rerun']=False;report['physics_reason']='Only rebuild matched robot reference after candidate image/geometry gates pass; failed candidate is retained rather than silently used.';report.update(state='completed',completed_utc=now(),elapsed_seconds=time.monotonic()-start,automatic_promotion=False);report['next_step']={'kind':'review_failed_gate_then_bounded_articulated_hand_fit','instruction':'Read per-frame evals and identify whether rigid hand SE3 can meet both mask and table. If not, use a small MANO wrist/finger articulation candidate with fixed metric scale and independent 2D/depth constraints, retain hidden-surface uncertainty, train-only selection and same holdouts. Do not relax gates or move table. Rebuild robot/physics only for a validated matching candidate.'};save(OUT/'report.json',report);progress('联合手SE3候选与独立留出评估完成；真实gate已记录，未替换参考，未训练。','completed');print('GATES',g,flush=True)
 except Exception:
  report.update(state='failed',error=traceback.format_exc(),elapsed_seconds=time.monotonic()-start);save(OUT/'report.json',report);progress('联合拟合接口或预算中止，检查点和失败证据已保留。','failed');raise
if __name__=='__main__':main()
