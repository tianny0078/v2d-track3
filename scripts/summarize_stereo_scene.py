"""Scientific figures and quality flags for cached stereo/hand diagnostics."""
import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from download_track3_sample import now,save

ROOT=Path('/work');BASE=ROOT/'outputs/track3/episode_000011';OUT=BASE/'stereo_scene_quality'
OUT.mkdir(exist_ok=True);cv2.setNumThreads(4)
learned=json.loads((BASE/'foundation_stereo_sequence/report.json').read_text())
if learned.get('state')!='completed' or len(learned['frames'])!=137:raise RuntimeError('Full learned stereo has not completed')
sgbm=json.loads((BASE/'stereo_sequence/report.json').read_text())
hands=np.load(BASE/'metric_scene_diagnostic/hands_world.npz',allow_pickle=False)
temporal=np.load(BASE/'hand_registration_temporal/hands_world.npz',allow_pickle=False)
temporal_report=json.loads((BASE/'hand_registration_temporal/report.json').read_text())
temporal_quality=json.loads((BASE/'hand_registration_temporal/quality_report.json').read_text())
registration=json.loads((BASE/'hand_registration_sequence/report.json').read_text())
reg_quality=json.loads((BASE/'hand_registration_sequence/quality_report.json').read_text())
poses=np.load(BASE/'stereo_sequence/camera_poses.npz',allow_pickle=False)
k=np.load(BASE/'moge_sample_calibrated/frame_000000/undistorted_input_K.npy',allow_pickle=False)
frames=np.arange(137)
report={'state':'running','created_utc':now(),'training':False,'world_frame':'rectified ego_cam_c at frame 0','objects':{},
        'limitations':['Coverage, cross-model agreement and reprojection do not measure GT accuracy.',
                       'Point clouds show visible surfaces, not complete object geometry or centres.',
                       'Registered hands are candidates only; the reviewed original trajectory remains the baseline.',
                       'World frame is the first stereo camera frame, not the challenge mocap frame.']}
save(OUT/'report.json',report)
fig,axes=plt.subplots(4,1,figsize=(12,11),sharex=True,constrained_layout=True)
sgvalid=np.asarray([f['stereo_valid_fraction'] for f in sgbm['frames']]);nnvalid=np.asarray([f['stereo_valid_fraction'] for f in learned['frames']])
axes[0].plot(frames,sgvalid,label='SGBM');axes[0].plot(frames,nnvalid,label='C-FoundationStereo');axes[0].set_ylabel('Stereo valid fraction');axes[0].legend(loc='lower right')
report['stereo_valid_fraction_p5_p50_p95']={'sgbm':np.percentile(sgvalid,[5,50,95]).tolist(),'learned':np.percentile(nnvalid,[5,50,95]).tolist()}
for ax,name in zip(axes[1:3],['blue_cup','beige_cup']):
    old=np.asarray([f['objects'][name]['mask_coverage'] for f in sgbm['frames']])
    new=np.asarray([f['objects'][name]['mask_depth_coverage'] for f in learned['frames']])
    disagreement=np.asarray([f['objects'][name]['learned_minus_sgbm_depth_m_p5_p50_p95'][1] for f in learned['frames']])
    ax.plot(frames,old,label='SGBM');ax.plot(frames,new,label='C-FoundationStereo');ax.set_ylabel(name+'\nmask depth coverage')
    report['objects'][name]={'sgbm_mask_coverage_p5_p50_p95':np.percentile(old,[5,50,95]).tolist(),
                             'learned_mask_coverage_p5_p50_p95':np.percentile(new,[5,50,95]).tolist(),
                             'frames_with_lower_learned_coverage':np.flatnonzero(new<old).tolist(),
                             'per_frame_shared_pixel_depth_difference_m_p5_p50_p95':np.percentile(disagreement,[5,50,95]).tolist()}
for side,name,color in [(0,'left','#b5489c'),(1,'right','#379d58')]:
    orig=hands['joints_world'][:,side,0];new=np.full_like(orig,np.nan)
    for f in registration['frames']:
        if any(h['is_right']==side for h in f['hands']):
            new[f['frame_index']]=np.load(BASE/f'hand_registration_sequence/frame_{f["frame_index"]:06d}'/('right.npz' if side else 'left.npz'),allow_pickle=False)['joints_world'][0]
    valid=hands['present'][1:,side]&hands['present'][:-1,side]
    for pts,style,label in [(orig,'-',name+' baseline'),(new,':',name+' 2D fitted candidate'),
                            (temporal['joints_world'][:,side,0],'--',name+' temporal candidate')]:
        distance=np.linalg.norm(np.diff(pts,axis=0),axis=1);distance[~valid]=np.nan
        axes[3].plot(frames[1:],distance,style,label=label,color=color,alpha=.9)
axes[3].axhline(.08,color='gray',linestyle='--',linewidth=1);axes[3].set_ylabel('Adjacent wrist step (m)');axes[3].set_xlabel('Frame index (no missing-hand interpolation)');axes[3].legend(ncol=2,fontsize=8)
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('Episode 11: internal consistency diagnostics, no GT accuracy claims')
fig.savefig(OUT/'sequence_quality.png',dpi=150);fig.savefig(OUT/'sequence_quality.pdf');plt.close(fig)
connections=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
fig=plt.figure(figsize=(15,14),constrained_layout=True)
for col,frame in enumerate([0,68,136]):
    ax=fig.add_subplot(3,3,col+1);image=cv2.cvtColor(cv2.imread(str(BASE/f'calibrated_sequence/frames/{frame:06d}.jpg')),cv2.COLOR_BGR2RGB)
    ax.imshow(image);ax.axis('off');ax.set_title(f'Frame {frame}')
    xy_ax=fig.add_subplot(3,3,col+4);xz_ax=fig.add_subplot(3,3,col+7)
    depth=np.load(BASE/f'foundation_stereo_sequence/frame_{frame:06d}/prediction.npz',allow_pickle=False)['depth_rgb']
    masks=np.load(BASE/f'object_masks_temporal/masks/{frame:06d}.npz',allow_pickle=False)
    pose=poses['world_T_rgb'][frame];bounds=[]
    for name,region,color in zip(masks['object_names'].tolist(),masks['masks'],['#456ed4','#c6b27e']):
        yy,xx=np.where(region&np.isfinite(depth));yy=yy[::4];xx=xx[::4];zz=depth[yy,xx]
        xyz=np.stack([(xx-k[0,2])*zz/k[0,0],(yy-k[1,2])*zz/k[1,1],zz],axis=1);xyz=xyz@pose[:3,:3].T+pose[:3,3]
        xy_ax.scatter(xyz[:,0],xyz[:,1],s=.5,c=color,alpha=.5,label=name+' visible')
        xz_ax.scatter(xyz[:,0],xyz[:,2],s=.5,c=color,alpha=.5,label=name+' visible')
    for side,name,color in [(0,'left','#b5489c'),(1,'right','#379d58')]:
        if hands['present'][frame,side]:
            pts=hands['joints_world'][frame,side]
            for ax,d in [(xy_ax,1),(xz_ax,2)]:
                for a,b in connections:ax.plot(pts[[a,b],0],pts[[a,b],d],color=color,linewidth=1)
                ax.scatter(pts[:,0],pts[:,d],s=4,c=color,label=name+' hand (baseline)')
    for ax in [xy_ax,xz_ax]:
        ax.set_xlim(-.4,.3);ax.set_xlabel('Initial-camera x (m)');ax.grid(alpha=.2);ax.set_aspect('equal',adjustable='box')
    xy_ax.set_ylim(.43,-.04);xy_ax.set_ylabel('Initial-camera y (m), down')
    xz_ax.set_ylim(.3,.75);xz_ax.set_ylabel('Initial-camera z (m), forward');xz_ax.legend(fontsize=7,loc='upper right')
fig.suptitle('Baseline hands + learned-stereo cup surfaces (fixed orthographic views)\nFar stereo outliers lie outside the shown depth range; raw arrays are preserved. No contact optimization.')
fig.savefig(OUT/'three_frame_scene.png',dpi=150);fig.savefig(OUT/'three_frame_scene.pdf');plt.close(fig)
report.update(state='completed',completed_utc=now(),registered_hand_continuity=reg_quality,
              temporal_hand_continuity=temporal_report['wrist_continuity'],temporal_hand_2d_agreement=temporal_quality['metrics'],
              learned_frames=len(learned['frames']),cached_pilot_frames=learned.get('reused_frame_count'),
              next_step='Constrain 2D hand registration jointly in time and depth; improve own full meshes and estimate object rigid poses before policy training.')
save(OUT/'report.json',report)
print(json.dumps(report,indent=2),flush=True)
