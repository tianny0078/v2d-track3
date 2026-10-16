"""Bounded full-sequence depth/2D trade-off sweep, preserving every candidate."""
import json,time
from pathlib import Path
import numpy as np
import cv2
from check_hand_surface_depth import load_faces,raster_depth
from download_track3_sample import digest,now,save

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/track3/episode_000011';OUT=BASE/'hand_depth_tradeoff'


def main():
    cv2.setNumThreads(4);OUT.mkdir(exist_ok=True)
    paths=[BASE/'hand_registration_temporal/hands_world.npz',BASE/'hand_registration_depth/hands_world.npz']
    source_hashes=[digest(p) for p in paths];temporal=np.load(paths[0],allow_pickle=False);aligned=np.load(paths[1],allow_pickle=False)
    pose=np.load(BASE/'stereo_sequence/camera_poses.npz',allow_pickle=False)['world_T_rgb']
    k=np.load(BASE/'moge_sample_calibrated/frame_000000/undistorted_input_K.npy',allow_pickle=False)
    faces=load_faces(BASE/'hand_sequence_hybrid/frame_000000/left_0.ply')
    mp=json.loads((BASE/'hand_detector_mediapipe_sequence/report.json').read_text());raw=json.loads((BASE/'hand_sequence_hybrid/report.json').read_text())
    assignment=json.loads((BASE/'hand_sequence_hybrid/identity_audit.json').read_text())['assignments']
    result={'state':'running','started_utc':now(),'source_sha256':source_hashes,'alphas':{},'training':False,
            'method':'Interpolate only the depth translation at alpha 0.25/0.50/0.75; cache endpoints 0/1 from existing diagnostics; no MANO shape/scale change',
            'limitations':['Model-prediction agreement, not held-out or GT accuracy.',
                           'No automatic policy training or replacement of the original trajectory.',
                           'Every candidate and per-frame result is retained; no missing hands are interpolated.']}
    save(OUT/'report.json',result);started=time.perf_counter()
    for alpha in [.25,.5,.75]:
        tag=f'alpha_{int(alpha*100):03d}';dest=OUT/tag;dest.mkdir(exist_ok=True);report_path=dest/'report.json'
        prior=json.loads(report_path.read_text()) if report_path.exists() else {}
        if prior and prior.get('source_sha256')!=source_hashes:raise RuntimeError('Trade-off inputs changed; refusing mixed cached results')
        if prior.get('state')=='completed':result['alphas'][str(alpha)]=prior['summary'];save(OUT/'report.json',result);continue
        rows=prior.get('frames',[]);done={row['frame_index'] for row in rows}
        report={'state':'running','started_utc':now(),'alpha':alpha,'source_sha256':source_hashes,'frames':rows}
        joints=temporal['joints_world']+alpha*(aligned['joints_world']-temporal['joints_world'])
        vertices=temporal['vertices_world']+alpha*(aligned['vertices_world']-temporal['vertices_world'])
        delta=temporal['world_rigid_delta'].copy();delta[:,:,:3,3]+=alpha*aligned['depth_shift_world']
        assert np.isnan(joints[~temporal['present']]).all() and np.isfinite(joints[temporal['present']]).all()
        np.savez_compressed(dest/'hands_world.npz',frame_index=np.arange(137),present=temporal['present'],joints_world=joints,vertices_world=vertices,world_rigid_delta=delta)
        save(report_path,report)
        for frame in range(137):
            if frame in done:continue
            masks=np.load(BASE/f'hand_masks_sequence/frame_{frame:06d}/masks.npz',allow_pickle=False)
            cups=np.load(BASE/f'object_masks_temporal/masks/{frame:06d}.npz',allow_pickle=False)['masks'].any(axis=0)
            depth=np.load(BASE/f'foundation_stereo_sequence/frame_{frame:06d}/prediction.npz',allow_pickle=False)['depth_rgb']
            targets={}
            if raw['frames'][frame]['box_source'].startswith('MediaPipe'):
                for side,index in zip(assignment[frame]['suggested_is_right'],assignment[frame]['source_indices']):
                    targets[side]=np.asarray(mp['frames'][frame]['hands'][index]['normalized_landmarks'])[:,:2]*[1024,767]
            item={'frame_index':frame,'hands':[]};r,t=pose[frame,:3,:3],pose[frame,:3,3]
            for side,mask in zip(masks['is_right'],masks['masks']):
                world=vertices[frame,side];camera=(world-t)@r;rendered=raster_depth(camera,faces,k);footprint=np.isfinite(rendered)
                region=cv2.erode(mask.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)&~cups
                valid=region&footprint&np.isfinite(depth);residual=rendered[valid]-depth[valid]
                hand={'is_right':int(side),'mask_iou':float((footprint&mask).sum()/max((footprint|mask).sum(),1)),
                      'depth_supported_pixels':int(valid.sum()),'depth_residual_median_m':float(np.median(residual)) if len(residual) else None}
                if side in targets:
                    target=targets[side];visible=(target[:,0]>=4)&(target[:,0]<1020)&(target[:,1]>=4)&(target[:,1]<763)
                    camera=(joints[frame,side]-t)@r;proj=camera@k.T;uv=proj[:,:2]/proj[:,2:3]
                    hand['visible_2d_landmarks']=int(visible.sum());hand['reprojection_median_pixels']=float(np.median(np.linalg.norm(uv[visible]-target[visible],axis=1))) if visible.any() else None
                item['hands'].append(hand)
            report['frames'].append(item);save(report_path,report)
            if frame%10==0:print('alpha',alpha,'frame',frame,flush=True)
        entries=[h for row in report['frames'] for h in row['hands']];residual=[h['depth_residual_median_m'] for h in entries if h['depth_residual_median_m'] is not None]
        error=[h['reprojection_median_pixels'] for h in entries if h.get('reprojection_median_pixels') is not None]
        summary={'alpha':alpha,'frames':len(report['frames']),'hand_instances':len(entries),'median_mask_iou':float(np.median([h['mask_iou'] for h in entries])),
                 'absolute_surface_depth_residual_m_p50_p95':np.percentile(np.abs(residual),[50,95]).tolist(),
                 'median_2d_reprojection_pixels_p50_p95':np.percentile(error,[50,95]).tolist(),'wrist_continuity':{}}
        for side,name in enumerate(['left','right']):
            valid=temporal['present'][1:,side]&temporal['present'][:-1,side];step=np.linalg.norm(np.diff(joints[:,side,0],axis=0),axis=1)
            summary['wrist_continuity'][name]={'max_step_m':float(step[valid].max()),'steps_over_8cm':(np.flatnonzero(valid&(step>.08))+1).tolist()}
        report.update(state='completed',completed_utc=now(),summary=summary);save(report_path,report)
        result['alphas'][str(alpha)]=summary;save(OUT/'report.json',result);print(json.dumps(summary),flush=True)
    result.update(state='completed',completed_utc=now(),elapsed_seconds=time.perf_counter()-started,
                  endpoint_reports=['hand_surface_depth_sequence/quality_report.json','hand_surface_depth_aligned/quality_report.json'])
    save(OUT/'report.json',result)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if (OUT/'report.json').exists():
            r=json.loads((OUT/'report.json').read_text());r.update(state='failed',error_type=type(exc).__name__,updated_utc=now());save(OUT/'report.json',r)
        raise
