"""Diagnostic rigid hand registration to independent, visible MediaPipe landmarks."""
import argparse,json,time
from pathlib import Path
import cv2
import numpy as np
from download_track3_sample import now,save

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/track3/episode_000011'
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--sequence',action='store_true');args=parser.parse_args()
OUT=BASE/('hand_registration_sequence' if args.sequence else 'hand_registration_sample');OUT.mkdir(exist_ok=True)
cv2.setNumThreads(4);cv2.setRNGSeed(37)
index=json.loads((BASE/'hand_sequence_reviewed/report.json').read_text())
original=json.loads((BASE/'hand_sequence_hybrid/report.json').read_text())
audit=json.loads((BASE/'hand_sequence_hybrid/identity_audit.json').read_text());amap={f['frame_index']:f for f in audit['assignments']}
mp=json.loads((BASE/'hand_detector_mediapipe_sequence/report.json').read_text())
k=np.load(BASE/'moge_sample_calibrated/frame_000000/undistorted_input_K.npy',allow_pickle=False)
poses=np.load(BASE/'stereo_sequence/camera_poses.npz',allow_pickle=False)['world_T_rgb']
report={'state':'running','started_utc':now(),'training':False,'frames':[],
        'method':'RANSAC PnP + LM, fixed WiLoR articulated shape, visible independent 2D landmarks only',
        'gates':{'minimum_visible_joints':8,'minimum_classifier_score':.5,'minimum_inlier_fraction':.6,
                 'max_rotation_degrees':25,'max_wrist_displacement_m':.15,'median_improvement_fraction':.25,
                 'max_median_reprojection_pixels':15,'max_p95_reprojection_pixels':25},
        'limitations':['MediaPipe is an independent model prediction, not ground truth.',
                       'Fitting and evaluation use the same 2D landmarks, so improvement is not held-out accuracy.',
                       '3D articulated shape and camera intrinsics are fixed; only a rigid delta is fitted.',
                       'Missing hands remain missing. Raw MANO parameters are not rewritten; delta transforms must be composed.']}
save(OUT/'report.json',report);start=time.perf_counter();accepted=0;tried=0;before=[];after=[]
for frame in (range(137) if args.sequence else [0,68,136]):
    folder=BASE/index['frames'][frame]['source'];hands=json.loads((folder/'predictions.json').read_text())
    image=cv2.imread(str(BASE/f'calibrated_sequence/frames/{frame:06d}.jpg'));observations=mp['frames'][frame]['hands']
    used_mp=original['frames'][frame]['box_source'].startswith('MediaPipe')
    source_indices=amap[frame]['source_indices'];frame_report={'frame_index':frame,'hands':[]}
    for hi,hand in enumerate(hands):
        p=hand['wilor_preds'];t=np.asarray(p['pred_cam_t_full']).reshape(1,3)
        joints=np.asarray(p['pred_keypoints_3d']).reshape(21,3)+t;vertices=np.asarray(p['pred_vertices']).reshape(778,3)+t
        delta=np.eye(4);side=int(hand['is_right']);entry={'is_right':side,'registration_accepted':False}
        color=(40,220,60) if side else (220,70,200)
        target=None;reason='No independent MediaPipe observation for this frame'
        if used_mp:
            obs=observations[source_indices[hi]];xy=np.asarray(obs['normalized_landmarks'])[:,:2]*[1024,767]
            visible=(xy[:,0]>=4)&(xy[:,0]<1020)&(xy[:,1]>=4)&(xy[:,1]<763)
            entry.update(detector_score=obs['score'],visible_target_joints=int(visible.sum()))
            target=xy
            if obs['score']<.5:reason='Low detector confidence'
            elif visible.sum()<8:reason='Insufficient visible joints'
            else:
                tried+=1
                objects=np.ascontiguousarray(joints[visible],np.float64);pixels=np.ascontiguousarray(xy[visible],np.float64)
                proj=joints@k.T;proj=proj[:,:2]/proj[:,2:3]
                old_error=np.linalg.norm(proj[visible]-pixels,axis=1)
                ok,rv,tv,inliers=cv2.solvePnPRansac(objects,pixels,k,None,np.zeros((3,1)),np.zeros((3,1)),True,
                    iterationsCount=300,reprojectionError=12,confidence=.999,flags=cv2.SOLVEPNP_ITERATIVE)
                count=len(inliers) if inliers is not None else 0;reason='PnP inlier gate'
                entry['pnp_inliers']=count
                if ok and count>=8 and count/len(objects)>=.6:
                    rv,tv=cv2.solvePnPRefineLM(objects[inliers[:,0]],pixels[inliers[:,0]],k,None,rv,tv)
                    r=cv2.Rodrigues(rv)[0];new=joints@r.T+tv.reshape(1,3);proj=new@k.T;proj=proj[:,:2]/proj[:,2:3]
                    new_error=np.linalg.norm(proj[visible]-pixels,axis=1)
                    old50=float(np.median(old_error));new50=float(np.median(new_error));new95=float(np.percentile(new_error,95))
                    angle=float(np.rad2deg(np.linalg.norm(rv)));distance=float(np.linalg.norm(new[0]-joints[0]))
                    good=(angle<=25 and distance<=.15 and np.all(new[:,2]>.1) and new50<old50*.75 and new50<15 and new95<25)
                    entry.update(initial_reprojection_pixels_p50_p95=np.percentile(old_error,[50,95]).tolist(),
                                 fitted_reprojection_pixels_p50_p95=[new50,new95],rotation_delta_degrees=angle,
                                 wrist_displacement_m=distance,camera_origin_translation_delta_norm_m=float(np.linalg.norm(tv)),
                                 registration_accepted=bool(good))
                    before.append(old50);after.append(new50 if good else old50)
                    reason='Accepted diagnostic rigid fit' if good else 'Rigid correction or reprojection gate'
                    if good:delta[:3,:3]=r;delta[:3,3]=tv[:,0];accepted+=1
        entry.update(reason=reason,camera_delta=delta.tolist())
        transformed=joints@delta[:3,:3].T+delta[:3,3];vnew=vertices@delta[:3,:3].T+delta[:3,3]
        world=poses[frame]
        dest=OUT/f'frame_{frame:06d}';dest.mkdir(exist_ok=True)
        np.savez_compressed(dest/('right.npz' if side else 'left.npz'),joints_camera=transformed,vertices_camera=vnew,
                            joints_world=transformed@world[:3,:3].T+world[:3,3],vertices_world=vnew@world[:3,:3].T+world[:3,3],camera_delta=delta)
        for obj,c,radius in [(joints,(70,70,250),2),(transformed,color,3)]:
            proj=obj@k.T;uv=proj[:,:2]/proj[:,2:3]
            for x,y in uv:
                if 0<=x<1024 and 0<=y<767:cv2.circle(image,(int(x),int(y)),radius,c,-1)
        if target is not None:
            for x,y in target:
                if 0<=x<1024 and 0<=y<767:cv2.circle(image,(int(x),int(y)),4,(255,220,0),1)
        frame_report['hands'].append(entry)
    cv2.putText(image,'red: original; cyan rings: MP; green/purple: fitted',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
    cv2.imwrite(str(OUT/f'frame_{frame:06d}'/'registration_overlay.jpg'),image)
    report['frames'].append(frame_report);save(OUT/'report.json',report)
report.update(state='completed',completed_utc=now(),elapsed_seconds=time.perf_counter()-start,
              attempted_hands=tried,accepted_corrections=accepted,
              matched_hand_median_reprojection_pixels_before_after=[float(np.median(before)),float(np.median(after))] if before else None)
save(OUT/'report.json',report)
print(json.dumps({k:v for k,v in report.items() if k!='frames'},indent=2),flush=True)
