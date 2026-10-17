"""Metric stereo depth and conservative background-only stereo visual odometry."""
import argparse,json,time
from pathlib import Path
import cv2
import numpy as np
from download_track3_sample import now,save

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/track3/episode_000011'
OUT=BASE/'stereo_sequence'


def main():
    global BASE,OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',type=int,default=11)
    parser.add_argument('--input-root',type=Path,default=ROOT/'data/track3_sample/track_3/public')
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--max-frames',type=int)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--allow-missing-masks',action='store_true')
    args=parser.parse_args()
    if args.episode<0 or (args.max_frames is not None and args.max_frames<1):parser.error('Invalid episode/frame budget')
    BASE=args.output_root or ROOT/f'outputs/track3/episode_{args.episode:06d}'
    OUT=BASE/'stereo_sequence'
    if (OUT/'report.json').exists() and json.loads((OUT/'report.json').read_text()).get('state')=='completed':
        raise RuntimeError('Completed output exists; choose a new output root')
    cv2.setNumThreads(args.threads);OUT.mkdir(parents=True,exist_ok=True);(OUT/'depth_rgb').mkdir(exist_ok=True)
    cal_path=args.calibration or args.input_root/'meta/camera_calibration.json'
    cal=json.loads(cal_path.read_text())
    pair=next(p for p in cal['stereo_pairs'] if p['camera']=='ego')
    assert pair['left']=='ego_cam_c' and pair['right']=='ego_cam_b'
    lc,rc=[cal['cameras'][pair[k]] for k in ['left','right']]
    transform=np.asarray(pair['left_to_right'])
    r1,r2,p1,p2,q,_,_=cv2.stereoRectify(np.asarray(lc['intrinsic_matrix']),np.asarray(lc['distortion_coefficients']),np.asarray(rc['intrinsic_matrix']),np.asarray(rc['distortion_coefficients']),tuple(lc['resolution']),transform[:3,:3].copy(),transform[:3,3].reshape(3,1).copy(),flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
    np.savez_compressed(OUT/'rectification.npz',R1=r1,R2=r2,P1=p1,P2=p2,Q=q)
    size=(1280,800);maps=[]
    for name,r,p in [('ego_cam_c',r1,p1),('ego_cam_b',r2,p2)]:
        c=cal['cameras'][name]
        maps.append(cv2.initUndistortRectifyMap(np.asarray(c['intrinsic_matrix']),np.asarray(c['distortion_coefficients']),r,p,size,cv2.CV_32FC1))
    video_root=args.input_root/'videos/chunk-000'
    left=cv2.VideoCapture(str(video_root/f'observation.images.ego_cam_c/episode_{args.episode:06d}.mp4'))
    right=cv2.VideoCapture(str(video_root/f'observation.images.ego_cam_b/episode_{args.episode:06d}.mp4'))
    assert left.isOpened() and right.isOpened(), 'Missing or unreadable stereo videos'
    counts=[int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in [left,right]]
    assert counts[0]==counts[1] and counts[0]>0, 'Stereo frame-count mismatch'
    frame_count=min(counts[0],args.max_frames or counts[0])
    params=dict(numDisparities=192,blockSize=5,P1=8*25,P2=32*25,uniquenessRatio=10,speckleWindowSize=100,speckleRange=2,disp12MaxDiff=1)
    matcher_left=cv2.StereoSGBM_create(minDisparity=0,**params)
    matcher_right=cv2.StereoSGBM_create(minDisparity=-192,**params)
    sift=cv2.SIFT_create(nfeatures=2200);matcher=cv2.BFMatcher()
    feature_mask=np.zeros((800,1280),np.uint8);feature_mask[:320,:]=255
    b_T_c=np.asarray(cal['cameras']['ego_cam_c']['extrinsics_to_stereo_left'])
    b_T_a=np.asarray(cal['cameras']['ego_cam_a']['extrinsics_to_stereo_left'])
    a_T_c=np.linalg.inv(b_T_a)@b_T_c
    rect_T_raw=np.eye(4);rect_T_raw[:3,:3]=r1
    rect_T_a=rect_T_raw@np.linalg.inv(a_T_c)
    rgb_k=np.asarray(cal['cameras']['ego_cam_a']['intrinsic_matrix'],dtype=float).copy()
    rw,rh=cal['cameras']['ego_cam_a']['resolution']
    rgb_k[0]*=1024/rw;rgb_k[1]*=767/rh
    rgb_k,_=cv2.getOptimalNewCameraMatrix(rgb_k,np.asarray(cal['cameras']['ego_cam_a']['distortion_coefficients']),(1024,767),0,(1024,767))
    np.save(OUT/'rgb_K.npy',rgb_k)
    h,w=767,1024;grid_y,grid_x=np.indices((800,1280))
    world_T_rect=np.eye(4);poses=[];rgb_poses=[];prev=None
    report={'state':'running','started_utc':now(),'frames':[],'method':'SGBM LR consistency + background SIFT/PnP',
            'world_frame':'rectified ego_cam_c at frame 0','training':False,'lr_consistency_px':1.5,
            'feature_region':'top 40% of stereo-left image; excludes the manipulation area conservatively',
            'limitations':['Depth/calibration/odometry are not ground-truth validated.',
                           'RANSAC failure holds the last accepted pose and marks the frame; no hidden interpolation.',
                           'Object visible-surface centres are not rigid object-centre poses.']}
    report.update(episode=args.episode,input_root=str(args.input_root),calibration=str(cal_path),source_frame_count=counts[0],requested_frames=frame_count,full_sequence=frame_count==counts[0],missing_masks_allowed=args.allow_missing_masks)
    save(OUT/'report.json',report);started=time.perf_counter()
    for frame in range(frame_count):
        ok1,im1=left.read();ok2,im2=right.read()
        if not (ok1 and ok2):raise RuntimeError('Stereo sequence ended early')
        im1=cv2.remap(cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY),*maps[0],cv2.INTER_LINEAR)
        im2=cv2.remap(cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY),*maps[1],cv2.INTER_LINEAR)
        disp=matcher_left.compute(im1,im2).astype(np.float32)/16
        reverse=matcher_right.compute(im2,im1).astype(np.float32)/16
        match_x=np.clip(np.rint(grid_x-disp).astype(int),0,1279)
        lr=np.abs(disp+reverse[grid_y,match_x])<1.5
        valid=(disp>0)&(grid_x-disp>=0)&lr
        points=cv2.reprojectImageTo3D(disp,q)
        valid&=np.isfinite(points).all(axis=2)&(points[:,:,2]>.15)&(points[:,:,2]<10)
        key,desc=sift.detectAndCompute(im1,feature_mask)
        item={'frame_index':frame,'stereo_valid_fraction':float(valid.mean()),'background_features':len(key),'pose_accepted':frame==0,'objects':{}}
        if prev is not None and desc is not None and prev['desc'] is not None:
            matches=[m for m,n in matcher.knnMatch(prev['desc'],desc,k=2) if m.distance<.7*n.distance]
            objects=[];pixels=[]
            for match in matches:
                x,y=np.rint(prev['key'][match.queryIdx].pt).astype(int)
                if 0<=x<1280 and 0<=y<800 and prev['valid'][y,x]:
                    objects.append(prev['points'][y,x]);pixels.append(key[match.trainIdx].pt)
            item.update(pnp_reference_frame=prev['frame'],pnp_correspondences=len(objects))
            if len(objects)>=30:
                pts=np.asarray(objects,np.float32);pix=np.asarray(pixels,np.float32)
                success,rv,tv,inliers=cv2.solvePnPRansac(pts,pix,p1[:3,:3],None,iterationsCount=300,reprojectionError=3.,confidence=.999)
                count=len(inliers) if inliers is not None else 0
                item['pnp_inliers']=count
                if success and count>=20 and count/max(len(objects),1)>.5:
                    rv,tv=cv2.solvePnPRefineLM(pts[inliers[:,0]],pix[inliers[:,0]],p1[:3,:3],None,rv,tv)
                    angle=float(np.linalg.norm(rv));translation=float(np.linalg.norm(tv))
                    item.update(relative_rotation_rad=angle,relative_translation_m=translation)
                    if angle<np.deg2rad(15) and translation<.15:
                        current_T_prev=np.eye(4);current_T_prev[:3,:3]=cv2.Rodrigues(rv)[0];current_T_prev[:3,3]=tv[:,0]
                        world_T_rect=prev['world_T_rect']@np.linalg.inv(current_T_prev);item['pose_accepted']=True
        if item['pose_accepted']:
            prev={'frame':frame,'key':key,'desc':desc,'points':points,'valid':valid,'world_T_rect':world_T_rect.copy()}
        world_T_a=world_T_rect@rect_T_a
        poses.append(world_T_rect.copy());rgb_poses.append(world_T_a.copy())
        c_points=points[valid]@r1
        a_points=c_points@a_T_c[:3,:3].T+a_T_c[:3,3]
        proj=a_points@rgb_k.T;uv=proj[:,:2]/proj[:,2:3]
        keep=(a_points[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w-1)&(uv[:,1]>=0)&(uv[:,1]<h-1)
        px=np.rint(uv[keep]).astype(int);z=a_points[keep,2]
        buffer=np.full(h*w,np.inf,np.float32);np.minimum.at(buffer,px[:,1]*w+px[:,0],z)
        depth=buffer.reshape(h,w);depth[~np.isfinite(depth)]=np.nan
        np.savez_compressed(OUT/'depth_rgb'/f'{frame:06d}.npz',depth=depth)
        mask_path=BASE/f'object_masks_temporal/masks/{frame:06d}.npz'
        if mask_path.exists():mask=np.load(mask_path,allow_pickle=False)
        elif args.allow_missing_masks:
            mask={'object_names':np.asarray([],dtype=str),'masks':[]};item['object_masks_known']=False
        else:raise FileNotFoundError(mask_path)
        for name,region in zip(mask['object_names'].tolist(),mask['masks']):
            vv,uu=np.where(region&np.isfinite(depth))
            if len(uu):
                zz=depth[vv,uu];xyz=np.stack([(uu-rgb_k[0,2])*zz/rgb_k[0,0],(vv-rgb_k[1,2])*zz/rgb_k[1,1],zz],axis=1)
                centre=xyz.mean(axis=0);world=world_T_a[:3,:3]@centre+world_T_a[:3,3]
                item['objects'][name]={'supported_pixels':len(uu),'mask_coverage':float(len(uu)/max(region.sum(),1)),
                                      'depth_m_p5_p50_p95':np.percentile(zz,[5,50,95]).tolist(),
                                      'visible_surface_centroid_camera_m':centre.tolist(),'visible_surface_centroid_world_m':world.tolist()}
        report['frames'].append(item)
        if frame%10==0 or frame==frame_count-1:
            save(OUT/'report.json',report)
            np.savez_compressed(OUT/'camera_poses.npz',frame_index=np.arange(len(poses)),world_T_rect=np.asarray(poses),world_T_rgb=np.asarray(rgb_poses),pose_accepted=np.asarray([f['pose_accepted'] for f in report['frames']]))
            print('frame',frame,'valid',round(item['stereo_valid_fraction'],3),'pose accepted',item['pose_accepted'],flush=True)
    left.release();right.release()
    report.update(state='completed',completed_utc=now(),elapsed_seconds=time.perf_counter()-started,
                  accepted_camera_poses=sum(f['pose_accepted'] for f in report['frames']))
    save(OUT/'report.json',report)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if (OUT/'report.json').exists():
            r=json.loads((OUT/'report.json').read_text());r.update(state='failed',error_type=type(exc).__name__,error_message=str(exc).split('\n')[0]);save(OUT/'report.json',r)
        raise
