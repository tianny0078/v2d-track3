"""Smooth bounded rigid correction candidates over contiguous visible hand tracks."""
import json,time
from pathlib import Path
import cv2
import numpy as np
from download_track3_sample import now,save

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/track3/episode_000011';OUT=BASE/'hand_registration_temporal'
OUT.mkdir(exist_ok=True);cv2.setNumThreads(4)
base=np.load(BASE/'metric_scene_diagnostic/hands_world.npz',allow_pickle=False)
camera=np.load(BASE/'stereo_sequence/camera_poses.npz',allow_pickle=False)['world_T_rgb']
source=json.loads((BASE/'hand_registration_sequence/report.json').read_text());assert source['state']=='completed'
present=base['present'];joints=base['joints_world'].copy();vertices=base['vertices_world'].copy()
target=np.zeros((137,2,6));weight=np.zeros((137,2));transforms=np.tile(np.eye(4),(137,2,1,1))
for f in source['frames']:
    frame=f['frame_index'];world_r=camera[frame,:3,:3]
    for h in f['hands']:
        if h['registration_accepted']:
            side=h['is_right'];registered=np.load(BASE/f'hand_registration_sequence/frame_{frame:06d}'/('right.npz' if side else 'left.npz'),allow_pickle=False)
            delta=np.asarray(h['camera_delta']);rv=cv2.Rodrigues(world_r@delta[:3,:3]@world_r.T)[0][:,0]
            shift=registered['joints_world'][0]-joints[frame,side,0]
            target[frame,side]=np.r_[shift,rv]
            weight[frame,side]=h['detector_score']*(h['visible_target_joints']/21)**2*h['pnp_inliers']/h['visible_target_joints']

started=time.perf_counter();settings={'baseline_anchor':.15,'correction_velocity_penalty':6.,'correction_acceleration_penalty':2.}
smooth=np.zeros_like(target);segments=[]
for side in [0,1]:
    indices=np.flatnonzero(present[:,side]);groups=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
    for group in groups:
        if not len(group):continue
        n=len(group);w=weight[group,side];matrix=np.diag(w+settings['baseline_anchor'])
        if n>1:
            diff=np.diff(np.eye(n),axis=0);matrix+=settings['correction_velocity_penalty']*diff.T@diff
        if n>2:
            diff=np.diff(np.eye(n),n=2,axis=0);matrix+=settings['correction_acceleration_penalty']*diff.T@diff
        smooth[group,side]=np.linalg.solve(matrix,w[:,None]*target[group,side]);segments.append({'is_right':side,'first_frame':int(group[0]),'last_frame':int(group[-1])})
for frame in range(137):
    for side in [0,1]:
        if present[frame,side]:
            shift,rv=smooth[frame,side,:3],smooth[frame,side,3:];r=cv2.Rodrigues(rv)[0];wrist=joints[frame,side,0].copy()
            vertices[frame,side]=(vertices[frame,side]-wrist)@r.T+wrist+shift
            joints[frame,side]=(joints[frame,side]-wrist)@r.T+wrist+shift
            transforms[frame,side,:3,:3]=r;transforms[frame,side,:3,3]=wrist+shift-r@wrist
            assert np.linalg.norm(rv)<=np.deg2rad(25)+1e-8 and np.linalg.norm(shift)<=.15+1e-8
assert np.isnan(joints[~present]).all() and np.isnan(vertices[~present]).all()
np.savez_compressed(OUT/'hands_world.npz',frame_index=np.arange(137),present=present,joints_world=joints,vertices_world=vertices,
                    world_rigid_delta=transforms,source_candidate_weight=weight)
report={'state':'completed','completed_utc':now(),'elapsed_seconds':time.perf_counter()-started,'training':False,'settings':settings,
        'segments':segments,'total_hands':int(present.sum()),'wrist_continuity':{},
        'limitations':['Candidate temporal regularization, not a GT hand-pose accuracy result.',
                       'Smoothing corrects only rigid deltas; source finger shape remains fixed.',
                       'No missing hands are created or gaps bridged; each visible segment is independent.',
                       'No depth/contact constraints yet; keep both original and candidate outputs.']}
for side,name in enumerate(['left','right']):
    valid=present[1:,side]&present[:-1,side];distance=np.linalg.norm(np.diff(joints[:,side,0],axis=0),axis=1)
    report['wrist_continuity'][name]={'adjacent_visible_pairs':int(valid.sum()),
          'wrist_step_m_p50_p95_max':np.percentile(distance[valid],[50,95,100]).tolist(),
          'steps_over_8cm':(np.flatnonzero(valid&(distance>.08))+1).tolist()}
save(OUT/'report.json',report);print(json.dumps(report,indent=2),flush=True)
