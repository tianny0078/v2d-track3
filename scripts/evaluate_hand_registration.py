"""Compare baseline and rigid hand candidates to the same independent 2D observations."""
import argparse,json
from pathlib import Path
import numpy as np
from download_track3_sample import now,save

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/track3/episode_000011'
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--depth-candidate',action='store_true');args=p.parse_args()
original=np.load(BASE/'metric_scene_diagnostic/hands_world.npz',allow_pickle=False)
temporal=np.load(BASE/'hand_registration_temporal/hands_world.npz',allow_pickle=False)
if args.depth_candidate:depth_candidate=np.load(BASE/'hand_registration_depth/hands_world.npz',allow_pickle=False)
camera=np.load(BASE/'stereo_sequence/camera_poses.npz',allow_pickle=False)['world_T_rgb']
k=np.load(BASE/'moge_sample_calibrated/frame_000000/undistorted_input_K.npy',allow_pickle=False)
source=json.loads((BASE/'hand_sequence_hybrid/report.json').read_text());mp=json.loads((BASE/'hand_detector_mediapipe_sequence/report.json').read_text())
audit=json.loads((BASE/'hand_sequence_hybrid/identity_audit.json').read_text());assign={f['frame_index']:f for f in audit['assignments']}
rows=[]
for frame in range(137):
    if not source['frames'][frame]['box_source'].startswith('MediaPipe'):continue
    for side,idx in zip(assign[frame]['suggested_is_right'],assign[frame]['source_indices']):
        obs=mp['frames'][frame]['hands'][idx];target=np.asarray(obs['normalized_landmarks'])[:,:2]*[1024,767]
        valid=(target[:,0]>=4)&(target[:,0]<1020)&(target[:,1]>=4)&(target[:,1]<763)
        if not valid.any():continue
        independent=np.load(BASE/f'hand_registration_sequence/frame_{frame:06d}'/('right.npz' if side else 'left.npz'),allow_pickle=False)['joints_world']
        row={'frame_index':frame,'is_right':side,'visible_landmarks':int(valid.sum()),'detector_score':obs['score']}
        candidates=[('baseline',original['joints_world'][frame,side]),('independent_rigid',independent),('temporal_rigid',temporal['joints_world'][frame,side])]
        if args.depth_candidate:candidates.append(('depth_rigid',depth_candidate['joints_world'][frame,side]))
        for name,world in candidates:
            xyz=(world-camera[frame,:3,3])@camera[frame,:3,:3];uv=xyz@k.T;uv=uv[:,:2]/uv[:,2:3]
            err=np.linalg.norm(uv[valid]-target[valid],axis=1);row[name+'_reprojection_pixels_p50_p95']=np.percentile(err,[50,95]).tolist()
        row['quality_flags']=[]
        if row['visible_landmarks']<15:row['quality_flags'].append('partial_image_edge_observation')
        key=('depth_rigid' if args.depth_candidate else 'temporal_rigid')+'_reprojection_pixels_p50_p95'
        if row[key][0]>30:row['quality_flags'].append('large_2d_model_disagreement')
        if obs['score']<.5:row['quality_flags'].append('low_detector_confidence')
        rows.append(row)
report={'state':'completed','completed_utc':now(),'evaluated_hand_observations':len(rows),'frames':rows,'metrics':{},
        'interpretation':'Same 2D predictions used for fitting; reports model agreement, not held-out or GT accuracy. All candidates remain diagnostic until depth/contact checks.',
        'flagged_observations':sum(bool(r['quality_flags']) for r in rows)}
for group,subset in [('all',rows),('at_least_15_visible_landmarks',[r for r in rows if r['visible_landmarks']>=15]),('partial_image_edge',[r for r in rows if r['visible_landmarks']<15])]:
    report['metrics'][group]={'observations':len(subset)}
    for name in ['baseline','independent_rigid','temporal_rigid']+(['depth_rigid'] if args.depth_candidate else []):
        values=[r[name+'_reprojection_pixels_p50_p95'][0] for r in subset]
        report['metrics'][group][name+'_median_reprojection_pixels_p5_p50_p95']=np.percentile(values,[5,50,95]).tolist() if values else None
save(BASE/('hand_registration_depth' if args.depth_candidate else 'hand_registration_temporal')/'quality_report.json',report)
print(json.dumps({k:v for k,v in report.items() if k!='frames'},indent=2))
