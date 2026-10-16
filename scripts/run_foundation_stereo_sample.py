"""Offline three-frame learned stereo, with left/right consistency and RGB registration."""
import argparse,collections,json,shutil,time
from pathlib import Path
import cv2
import numpy as np
import torch  # Load the existing matching CUDA/cuDNN libraries before ONNX Runtime.
import onnxruntime as ort
from download_track3_sample import digest,now,save

ROOT=Path('/work');BASE=ROOT/'outputs/track3/episode_000011';OUT=BASE/'foundation_stereo_sample'
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--sequence',action='store_true');args=parser.parse_args()
if args.sequence:OUT=BASE/'foundation_stereo_sequence'


def preprocess(image):
    # Match the official v2d foundation_stereo normalization and bottom/right padding.
    h,w=image.shape[:2];scale=min(960/w,576/h);nw,nh=int(w*scale),int(h*scale)
    resized=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_AREA)
    padded=cv2.copyMakeBorder(resized,0,576-nh,0,960-nw,cv2.BORDER_REPLICATE)
    rgb=cv2.cvtColor(padded,cv2.COLOR_BGR2RGB).astype(np.float32)
    normalized=(rgb-np.asarray([123.675,116.28,103.53],np.float32))/np.asarray([58.395,57.12,57.375],np.float32)
    return np.ascontiguousarray(normalized.transpose(2,0,1)[None]),(nw,nh,w,h)


def depth_preview(depth):
    scaled=np.clip((depth-.25)/.95,0,1);scaled[~np.isfinite(scaled)]=0
    image=cv2.applyColorMap((255*scaled).astype(np.uint8),cv2.COLORMAP_TURBO);image[~np.isfinite(depth)]=0
    return image


def main():
    cv2.setNumThreads(4);torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    model=ROOT/'weights/foundation_stereo/deployable_foundationstereo_small_576x960_v2.0.onnx'
    reuse={}
    for source in [BASE/'foundation_stereo_sample',OUT]:
        if (source/'report.json').exists():
            old=json.loads((source/'report.json').read_text())
            if old.get('model_sha256')==digest(model):
                for f in old['frames']:
                    if (source/f'frame_{f["frame_index"]:06d}/prediction.npz').exists():reuse[f['frame_index']]=(source,f)
    report={'state':'running','started_utc':now(),'training':False,'network':'disabled','frames':[],
            'onnxruntime':ort.__version__,'torch':torch.__version__,'model_sha256':digest(model),
            'source':json.loads((model.parent/'source.json').read_text()),'lr_consistency_pixels':1.5,
            'preprocessing':'Official ImageNet RGB normalization, aspect-preserving resize, bottom/right replicate padding; disparity restored with actual horizontal scale',
            'limitations':['Learned stereo and comparison to SGBM are not GT depth accuracy.',
                           'No new camera extrinsics, hand alignment, object poses or completed meshes are inferred here.',
                           'This run uses the ONNX model with CUDA EP, not the official TensorRT wrapper.']}
    save(OUT/'report.json',report);started=time.perf_counter()
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=2
    options.enable_profiling=not args.sequence;options.profile_file_prefix=str(OUT/'ort-profile')
    session=ort.InferenceSession(str(model),sess_options=options,
        providers=[('CUDAExecutionProvider',{'device_id':0,'gpu_mem_limit':48*1024**3,'arena_extend_strategy':'kSameAsRequested',
                                            'cudnn_conv_algo_search':'HEURISTIC','use_tf32':0}),'CPUExecutionProvider'])
    report.update(session_load_seconds=time.perf_counter()-started,active_providers=session.get_providers(),
                  inputs=[{'name':p.name,'shape':p.shape,'type':p.type} for p in session.get_inputs()],
                  outputs=[{'name':p.name,'shape':p.shape,'type':p.type} for p in session.get_outputs()]);save(OUT/'report.json',report)
    if 'CUDAExecutionProvider' not in session.get_providers():raise RuntimeError('CUDA provider did not initialize; refusing silent CPU inference')
    if {p.name for p in session.get_inputs()}!={'left_image','right_image'}:raise RuntimeError('Unexpected stereo model inputs')
    def infer(left,right):
        a,meta=preprocess(left);b,_=preprocess(right)
        output=session.run(['disparity'],{'left_image':a,'right_image':b})[0]
        raw=np.squeeze(output);nw,nh,w,h=meta
        if raw.shape!=(576,960):raise RuntimeError('Unexpected disparity dimensions')
        return cv2.resize(raw[:nh,:nw],(w,h),interpolation=cv2.INTER_LINEAR)*(w/nw)
    cal=json.loads((ROOT/'data/track3_sample/track_3/public/meta/camera_calibration.json').read_text())
    cache=np.load(BASE/'validation/stereo_diagnostic.npz',allow_pickle=False)
    r1,r2,p1,p2,q=[cache[name] for name in ['R1','R2','P1','P2','Q']];maps=[]
    for name,r,p in [('ego_cam_c',r1,p1),('ego_cam_b',r2,p2)]:
        c=cal['cameras'][name];maps.append(cv2.initUndistortRectifyMap(np.asarray(c['intrinsic_matrix']),np.asarray(c['distortion_coefficients']),r,p,(1280,800),cv2.CV_32FC1))
    b_T_c=np.asarray(cal['cameras']['ego_cam_c']['extrinsics_to_stereo_left']);b_T_a=np.asarray(cal['cameras']['ego_cam_a']['extrinsics_to_stereo_left'])
    a_T_c=np.linalg.inv(b_T_a)@b_T_c;k=np.load(BASE/'moge_sample_calibrated/frame_000000/undistorted_input_K.npy',allow_pickle=False)
    vr=ROOT/'data/track3_sample/track_3/public/videos/chunk-000';caps=[cv2.VideoCapture(str(vr/f'observation.images.ego_cam_{cam}/episode_000011.mp4')) for cam in ['c','b']]
    yy,xx=np.indices((800,1280));sg_report=json.loads((BASE/'stereo_sequence/report.json').read_text())
    for frame in (range(137) if args.sequence else [0,68,136]):
        dest=OUT/f'frame_{frame:06d}';dest.mkdir(exist_ok=True);images=[]
        if frame in reuse:
            source,item=reuse[frame];path=source/f'frame_{frame:06d}/prediction.npz'
            with np.load(path,allow_pickle=False) as cached:
                if cached['depth_rgb'].shape!=(767,1024) or cached['stereo_valid'].shape!=(800,1280):raise RuntimeError('Cached stereo shapes differ')
            if path!=dest/'prediction.npz':shutil.copyfile(path,dest/'prediction.npz')
            item=dict(item,cache_reused_from=str(path),new_inference_seconds=0)
            report['frames'].append(item);save(OUT/'report.json',report)
            print('frame',frame,'reused validated stereo cache',flush=True);continue
        for cap,mapping in zip(caps,maps):
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,raw=cap.read()
            if not ok:raise RuntimeError('Missing stereo frame')
            images.append(cv2.remap(raw,*mapping,cv2.INTER_LINEAR))
        start=time.perf_counter();disparity=infer(*images)
        # Reversed pair plus horizontal flips gives a positive right-view disparity.
        right_disparity=np.ascontiguousarray(infer(np.ascontiguousarray(images[1][:,::-1]),np.ascontiguousarray(images[0][:,::-1]))[:,::-1])
        match_x=np.clip(np.rint(xx-disparity).astype(int),0,1279)
        lr=np.abs(disparity-right_disparity[yy,match_x])<1.5
        points=cv2.reprojectImageTo3D(disparity,q)
        valid=np.isfinite(disparity)&(disparity>0)&(xx-disparity>=0)&lr&np.isfinite(points).all(axis=2)&(points[:,:,2]>.15)&(points[:,:,2]<10)
        c_points=points[valid]@r1;a_points=c_points@a_T_c[:3,:3].T+a_T_c[:3,3]
        proj=a_points@k.T;uv=proj[:,:2]/proj[:,2:3]
        keep=(a_points[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<1023)&(uv[:,1]>=0)&(uv[:,1]<766)
        pixels=np.rint(uv[keep]).astype(int);buffer=np.full(767*1024,np.inf,np.float32)
        np.minimum.at(buffer,pixels[:,1]*1024+pixels[:,0],a_points[keep,2]);depth=buffer.reshape(767,1024);depth[~np.isfinite(depth)]=np.nan
        np.savez_compressed(dest/'prediction.tmp.npz',disparity=disparity,right_disparity=right_disparity,stereo_valid=valid,depth_rgb=depth)
        (dest/'prediction.tmp.npz').replace(dest/'prediction.npz')
        previous=np.load(BASE/f'stereo_sequence/depth_rgb/{frame:06d}.npz',allow_pickle=False)['depth']
        masks=np.load(BASE/f'object_masks_temporal/masks/{frame:06d}.npz',allow_pickle=False)
        item={'frame_index':frame,'dual_direction_inference_seconds':time.perf_counter()-start,
              'stereo_valid_fraction':float(valid.mean()),'sgbm_stereo_valid_fraction':sg_report['frames'][frame]['stereo_valid_fraction'],'objects':{}}
        for name,mask in zip(masks['object_names'].tolist(),masks['masks']):
            support=mask&np.isfinite(depth);common=support&np.isfinite(previous)
            item['objects'][name]={'mask_depth_coverage':float(support.sum()/mask.sum()),
                                  'sgbm_mask_depth_coverage':sg_report['frames'][frame]['objects'][name]['mask_coverage'],
                                  'depth_m_p5_p50_p95':np.percentile(depth[support],[5,50,95]).tolist() if support.any() else None,
                                  'shared_pixels_with_sgbm':int(common.sum()),
                                  'learned_minus_sgbm_depth_m_p5_p50_p95':np.percentile(depth[common]-previous[common],[5,50,95]).tolist() if common.any() else None}
        if frame in [0,20,40,60,68,80,100,120,136]:
            rgb=cv2.imread(str(BASE/f'calibrated_sequence/frames/{frame:06d}.jpg'))
            panels=[rgb,depth_preview(previous),depth_preview(depth)];titles=['RGB','SGBM, LR consistent','C-FoundationStereo, LR consistent']
            for panel,title in zip(panels,titles):cv2.putText(panel,title,(12,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2,cv2.LINE_AA)
            cv2.imwrite(str(dest/'comparison.jpg'),np.concatenate(panels,axis=1))
        report['frames'].append(item);save(OUT/'report.json',report);print(json.dumps(item),flush=True)
    for cap in caps:cap.release()
    if not args.sequence:
        profile=session.end_profiling();events=json.loads(Path(profile).read_text())
        counts=collections.Counter(e.get('args',{}).get('provider') for e in events if e.get('cat')=='Node' and e.get('args',{}).get('provider'))
        report.update(profile_file=profile,profile_node_execution_counts=dict(counts))
    else:report['profiling_note']='Profiled the three-frame pilot only; full sequence profiling disabled to avoid a large trace.'
    report.update(state='completed',completed_utc=now(),elapsed_seconds=time.perf_counter()-started,reused_frame_count=len(reuse))
    save(OUT/'report.json',report)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if (OUT/'report.json').exists():
            r=json.loads((OUT/'report.json').read_text());r.update(state='failed',updated_utc=now(),error_type=type(exc).__name__,error_message=str(exc).split('\n')[0]);save(OUT/'report.json',r)
        raise
