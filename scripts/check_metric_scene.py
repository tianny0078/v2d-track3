"""Fuse cached predictions in the first stereo-camera frame for diagnostics only."""
import hashlib,json,time
from pathlib import Path
import cv2
import numpy as np
from download_track3_sample import now,save

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/track3/episode_000011';OUT=BASE/'metric_scene_diagnostic'


def export_points(path,xyz,colors):
    with path.open('w') as stream:
        stream.write('ply\nformat ascii 1.0\nelement vertex '+str(len(xyz))+'\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for p,c in zip(xyz,colors):stream.write(' '.join(f'{x:.6f}' for x in p)+' '+' '.join(str(int(x)) for x in c)+'\n')


def main():
    cv2.setNumThreads(4);OUT.mkdir(exist_ok=True)
    index_path=BASE/'hand_sequence_reviewed/report.json';index=json.loads(index_path.read_text())
    poses=np.load(BASE/'stereo_sequence/camera_poses.npz',allow_pickle=False)
    k=np.load(BASE/'moge_sample_calibrated/frame_000000/undistorted_input_K.npy',allow_pickle=False)
    assert index['state']=='completed' and len(index['frames'])==137
    n=137;present=np.zeros((n,2),bool);joints_camera=np.full((n,2,21,3),np.nan);joints_world=joints_camera.copy()
    vertices_world=np.full((n,2,778,3),np.nan);vertices_camera=vertices_world.copy()
    report={'state':'running','started_utc':now(),'training':False,'frames':[],
            'units':'metres','world_frame':'rectified ego_cam_c at frame 0',
            'source_hand_index_sha256':hashlib.sha256(index_path.read_bytes()).hexdigest(),
            'limitations':['This is an unoptimized coordinate fusion diagnostic, not policy-ready motion.',
                           'No hand/object physical contact or depth correction is applied.',
                           'Stereo residuals at projected hand vertices are sparse proxies, not GT pose error.',
                           'Cups are masked visible surfaces; they are not full meshes or 6D object poses.']}
    save(OUT/'report.json',report);started=time.perf_counter()
    for row in index['frames']:
        frame=row['frame_index'];predictions=json.loads((BASE/row['source']/'predictions.json').read_text())
        depth=np.load(BASE/f'stereo_sequence/depth_rgb/{frame:06d}.npz',allow_pickle=False)['depth']
        masks=np.load(BASE/f'object_masks_temporal/masks/{frame:06d}.npz',allow_pickle=False)
        # Reject cup pixels in this hand-specific comparison, including a small boundary margin.
        cup_pixels=cv2.dilate(np.any(masks['masks'],axis=0).astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
        pose=poses['world_T_rgb'][frame];r,t=pose[:3,:3],pose[:3,3]
        item={'frame_index':frame,'hands':[]};cloud=[];colors=[]
        for hand in predictions:
            side=int(hand['is_right']);p=hand['wilor_preds'];translation=np.asarray(p['pred_cam_t_full']).reshape(1,3)
            joint=np.asarray(p['pred_keypoints_3d']).reshape(21,3)+translation
            verts=np.asarray(p['pred_vertices']).reshape(778,3)+translation
            assert np.isfinite(joint).all() and np.isfinite(verts).all()
            present[frame,side]=True;joints_camera[frame,side]=joint;joints_world[frame,side]=joint@r.T+t
            vertices_camera[frame,side]=verts;vertices_world[frame,side]=verts@r.T+t
            proj=joint@k.T;proj=proj[:,:2]/proj[:,2:3]
            predicted_pixels=np.asarray(p['pred_keypoints_2d']).reshape(21,2)
            reprojection=np.linalg.norm(proj-predicted_pixels,axis=1)
            projv=verts@k.T;uv=projv[:,:2]/projv[:,2:3];pixels=np.rint(uv).astype(int)
            valid=(verts[:,2]>0)&(pixels[:,0]>=0)&(pixels[:,0]<depth.shape[1])&(pixels[:,1]>=0)&(pixels[:,1]<depth.shape[0])
            ix=np.flatnonzero(valid);px=pixels[ix];support=np.isfinite(depth[px[:,1],px[:,0]])&~cup_pixels[px[:,1],px[:,0]]
            ix=ix[support];px=px[support];residual=verts[ix,2]-depth[px[:,1],px[:,0]]
            entry={'is_right':side,'projected_joint_error_pixels_p50_p95':np.percentile(reprojection,[50,95]).tolist(),
                   'vertices_inside_image':int(valid.sum()),'stereo_supported_vertices_outside_cups':len(ix),
                   'stereo_depth_residual_m_p10_p50_p90':np.percentile(residual,[10,50,90]).tolist() if len(ix) else None,
                   'wrist_camera_m':joint[0].tolist(),'wrist_world_m':joints_world[frame,side,0].tolist()}
            item['hands'].append(entry)
            if frame in [0,68,136]:
                cloud.append(vertices_world[frame,side]);colors.append(np.tile([60,220,80] if side else [220,60,180],(778,1)))
        if frame in [0,68,136]:
            for name,mask in zip(masks['object_names'].tolist(),masks['masks']):
                yy,xx=np.where(mask&np.isfinite(depth));zz=depth[yy,xx]
                xyz=np.stack([(xx-k[0,2])*zz/k[0,0],(yy-k[1,2])*zz/k[1,1],zz],axis=1)
                world=xyz@r.T+t;cloud.append(world);colors.append(np.tile([60,100,240] if name=='blue_cup' else [215,195,150],(len(world),1)))
                export_points(OUT/f'frame_{frame:06d}_{name}_visible.ply',world,colors[-1])
            export_points(OUT/f'frame_{frame:06d}_hands_and_cups.ply',np.concatenate(cloud),np.concatenate(colors))
        report['frames'].append(item)
    np.savez_compressed(OUT/'hands_world.npz',frame_index=np.arange(n),present=present,
                        joints_camera=joints_camera,joints_world=joints_world,vertices_camera=vertices_camera,vertices_world=vertices_world)
    errors=[hand['projected_joint_error_pixels_p50_p95'][0] for f in report['frames'] for hand in f['hands']]
    residuals=[hand['stereo_depth_residual_m_p10_p50_p90'][1] for f in report['frames'] for hand in f['hands'] if hand['stereo_supported_vertices_outside_cups']>=30]
    jumps={}
    for side,name in enumerate(['left','right']):
        valid=present[1:,side]&present[:-1,side]
        distance=np.linalg.norm(np.diff(joints_world[:,side,0],axis=0),axis=1)
        jumps[name]={'adjacent_visible_pairs':int(valid.sum()),'wrist_step_m_p50_p95_max':np.percentile(distance[valid],[50,95,100]).tolist(),
                     'steps_over_8cm':[int(f) for f in np.flatnonzero(valid&(distance>.08))+1]}
    report.update(state='completed',completed_utc=now(),elapsed_seconds=time.perf_counter()-started,
                  total_hands=int(present.sum()),joint_reprojection_median_pixels=float(np.median(errors)),
                  supported_hand_depth_comparisons=len(residuals),
                  hand_depth_residual_m_p5_p50_p95=np.percentile(residuals,[5,50,95]).tolist() if residuals else None,
                  wrist_continuity=jumps)
    save(OUT/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='frames'},indent=2),flush=True)


if __name__=='__main__':main()
