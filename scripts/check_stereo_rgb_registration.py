"""Check factory rig transforms by registering cached stereo depth into RGB."""
import json
from datetime import datetime,timezone
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/track3/episode_000011'
OUT=BASE/'stereo_rgb_diagnostic'


def main():
    cv2.setNumThreads(4);OUT.mkdir(exist_ok=True)
    cal=json.loads((ROOT/'data/track3_sample/track_3/public/meta/camera_calibration.json').read_text())
    cached=np.load(BASE/'validation/stereo_diagnostic.npz',allow_pickle=False)
    disparity=cached['disparity'];q=cached['Q'];r1=cached['R1']
    points_rect=cv2.reprojectImageTo3D(disparity,q)
    valid=(disparity>0)&np.isfinite(points_rect).all(axis=2)&(points_rect[:,:,2]>.15)&(points_rect[:,:,2]<10)
    # Both per-camera transforms target the original physical stereo-left CAM_B.
    # The distributed ego pair instead calls CAM_C its image-left after rotation.
    b_T_c=np.asarray(cal['cameras']['ego_cam_c']['extrinsics_to_stereo_left'])
    b_T_a=np.asarray(cal['cameras']['ego_cam_a']['extrinsics_to_stereo_left'])
    a_T_c=np.linalg.inv(b_T_a)@b_T_c
    c_points=points_rect[valid]@r1
    a_points=c_points@a_T_c[:3,:3].T+a_T_c[:3,3]
    pred=BASE/'moge_sample_calibrated/frame_000068'
    image=cv2.imread(str(pred/'input_undistorted.jpg'));h,w=image.shape[:2]
    k=np.load(pred/'undistorted_input_K.npy',allow_pickle=False)
    projected=a_points@k.T
    uv=projected[:,:2]/projected[:,2:3]
    keep=(a_points[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w-1)&(uv[:,1]>=0)&(uv[:,1]<h-1)
    pixel=np.rint(uv[keep]).astype(int);z=a_points[keep,2]
    buffer=np.full(h*w,np.inf,np.float32)
    np.minimum.at(buffer,pixel[:,1]*w+pixel[:,0],z)
    registered=buffer.reshape(h,w);registered[~np.isfinite(registered)]=np.nan
    np.save(OUT/'stereo_depth_in_rgb_camera.npy',registered)
    mask_data=np.load(BASE/'object_masks_temporal/masks/000068.npz',allow_pickle=False)
    mono=np.load(pred/'depth.npy',allow_pickle=False)
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'frame_index':68,
            'transform_interpretation':'a_T_c = inverse(b_T_a) @ b_T_c; rectified C points returned to native C using R1.T',
            'a_T_c':a_T_c.tolist(),'registered_depth_pixel_fraction':float(np.isfinite(registered).mean()),
            'source_stereo_valid_pixels':int(valid.sum()),'projected_points_in_rgb':int(keep.sum()),'objects':{},
            'limitations':['SGBM stereo and MoGe predictions are both unvalidated; their ratio is not ground-truth error.',
                           'Rig extrinsic interpretation is provisional pending visual/image-feature checks.',
                           'Different camera viewpoints may introduce visibility differences and projection holes.']}
    for name,mask in zip(mask_data['object_names'].tolist(),mask_data['masks']):
        support=mask&np.isfinite(registered)
        ratios=mono[support]/registered[support]
        report['objects'][name]={'mask_pixels':int(mask.sum()),'registered_stereo_pixels':int(support.sum()),
                                'coverage_fraction':float(support.sum()/max(mask.sum(),1)),
                                'stereo_depth_m_p5_p50_p95':np.percentile(registered[support],[5,50,95]).tolist() if support.any() else None,
                                'moge_to_stereo_depth_ratio_p5_p50_p95':np.percentile(ratios,[5,50,95]).tolist() if support.any() else None}
    scaled=np.clip((np.nan_to_num(registered,nan=0)-.3)/2.7,0,1)
    colored=cv2.applyColorMap(np.uint8(scaled*255),cv2.COLORMAP_TURBO)
    good=np.isfinite(registered)
    overlay=image.copy();overlay[good]=(.55*image[good]+.45*colored[good]).astype(np.uint8)
    canvas=np.hstack([image,overlay])
    cv2.imwrite(str(OUT/'stereo_registered_overlay.jpg'),canvas)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['objects'],indent=2))


if __name__=='__main__':main()
