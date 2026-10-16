"""Fit a bounded, temporally regularized stereo depth correction to diagnostic hands."""
import json,time
from pathlib import Path
import numpy as np
from download_track3_sample import now,save

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/track3/episode_000011';OUT=BASE/'hand_registration_depth'
OUT.mkdir(exist_ok=True);started=time.perf_counter()
source=np.load(BASE/'hand_registration_temporal/hands_world.npz',allow_pickle=False)
poses=np.load(BASE/'stereo_sequence/camera_poses.npz',allow_pickle=False)['world_T_rgb']
depth=json.loads((BASE/'hand_surface_depth_sequence/report.json').read_text());assert depth['state']=='completed'
quality=json.loads((BASE/'hand_registration_temporal/quality_report.json').read_text())
bad={(r['frame_index'],r['is_right']) for r in quality['frames'] if r['quality_flags']}
joints=source['joints_world'].copy();vertices=source['vertices_world'].copy();present=source['present']
target=np.zeros((137,2,3));weight=np.zeros((137,2));rejected=[]
for f in depth['frames']:
    frame=f['frame_index'];pose=poses[frame]
    for hand in f['hands']:
        side=hand['is_right'];stats=hand['candidates']['temporal'];nn=stats['learned'];sg=stats['sgbm']
        reason=None
        if (frame,side) in bad:reason='Existing 2D/edge quality flag'
        elif stats['rendered_mask_iou_with_sam2']<.5 or nn['supported_pixels']<100 or sg['supported_pixels']<100:reason='Insufficient mask/depth support'
        else:
            low,median,high=nn['rendered_minus_stereo_depth_m_p10_p50_p90'];sg_median=sg['rendered_minus_stereo_depth_m_p10_p50_p90'][1]
            if abs(median)>.10:reason='Required depth shift exceeds 10cm'
            elif high-low>.06:reason='Depth residual spread exceeds 6cm'
            elif abs(median-sg_median)>.02:reason='Learned and classical stereo disagree by over 2cm'
        if reason:rejected.append({'frame_index':frame,'is_right':side,'reason':reason});continue
        center=joints[frame,side].mean(axis=0);camera=(center-pose[:3,3])@pose[:3,:3]
        target[frame,side]=pose[:3,:3]@(-median*camera/camera[2])
        weight[frame,side]=stats['rendered_mask_iou_with_sam2']

# Only evidence-supported frames are adjusted; corrections taper to zero at adjacent rejected observations.
shift=np.zeros_like(target);settings={'baseline_anchor':.05,'velocity_penalty':8.,'acceleration_penalty':4.}
segments=[]
for side in [0,1]:
    indices=np.flatnonzero(weight[:,side]>0)
    for group in np.split(indices,np.flatnonzero(np.diff(indices)>1)+1):
        if not len(group):continue
        n=len(group);w=weight[group,side];matrix=np.diag(w+settings['baseline_anchor'])
        for order,key in [(1,'velocity_penalty'),(2,'acceleration_penalty')]:
            if n>order:
                diff=np.diff(np.eye(n),n=order,axis=0);matrix+=settings[key]*diff.T@diff
        if group[0]>0 and present[group[0]-1,side]:matrix[0,0]+=settings['velocity_penalty']
        if group[-1]<136 and present[group[-1]+1,side]:matrix[-1,-1]+=settings['velocity_penalty']
        shift[group,side]=np.linalg.solve(matrix,w[:,None]*target[group,side]);segments.append({'is_right':side,'first_frame':int(group[0]),'last_frame':int(group[-1])})
delta=source['world_rigid_delta'].copy()
for frame,side in zip(*np.where(present)):
    joints[frame,side]+=shift[frame,side];vertices[frame,side]+=shift[frame,side];delta[frame,side,:3,3]+=shift[frame,side]
assert np.array_equal(present,source['present']) and np.isnan(joints[~present]).all() and np.isnan(vertices[~present]).all()
assert np.isfinite(joints[present]).all() and np.isfinite(vertices[present]).all()
assert np.max(np.linalg.norm(shift,axis=2))<.2
np.savez_compressed(OUT/'hands_world.npz',frame_index=np.arange(137),present=present,joints_world=joints,vertices_world=vertices,
                    world_rigid_delta=delta,depth_shift_world=shift,source_depth_weight=weight)
report={'state':'completed','completed_utc':now(),'elapsed_seconds':time.perf_counter()-started,'training':False,
        'method':'Robust rendered surface depth, SGBM/learned agreement gates, centroid viewing-ray translation, regularized within evidence-supported segments',
        'settings':settings,'depth_anchored_instances':int((weight>0).sum()),'unchanged_or_rejected':rejected,'segments':segments,
        'world_shift_norm_m_p50_p95_max':np.percentile(np.linalg.norm(shift[present],axis=1),[50,95,100]).tolist(),
        'wrist_continuity':{},'limitations':['Depth and 2D targets are model predictions, not GT; fitted residual improvement is not held-out accuracy.',
                                         'No hand scale or finger pose is changed; original and earlier candidates are retained.',
                                         'Rejected observations receive zero correction; accepted neighbours taper to zero at those boundaries.',
                                         'No physical object contact or collision optimization is included.']}
for side,name in enumerate(['left','right']):
    valid=present[1:,side]&present[:-1,side];step=np.linalg.norm(np.diff(joints[:,side,0],axis=0),axis=1)
    report['wrist_continuity'][name]={'wrist_step_m_p50_p95_max':np.percentile(step[valid],[50,95,100]).tolist(),'steps_over_8cm':(np.flatnonzero(valid&(step>.08))+1).tolist()}
save(OUT/'report.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['unchanged_or_rejected','segments']},indent=2),flush=True)
