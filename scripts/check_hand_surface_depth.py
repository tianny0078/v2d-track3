"""Compare rendered hand surfaces to stereo only within independently segmented hand pixels."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from download_track3_sample import now,save

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/track3/episode_000011';OUT=BASE/'hand_surface_depth_diagnostic'
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--sequence',action='store_true');parser.add_argument('--depth-candidate',action='store_true');args=parser.parse_args()
if args.sequence:OUT=BASE/'hand_surface_depth_sequence'
if args.depth_candidate:
    if not args.sequence:raise ValueError('Depth candidate requires --sequence')
    OUT=BASE/'hand_surface_depth_aligned'


def load_faces(path):
    # All input meshes were generated here by the same trimesh PLY exporter.
    data=path.read_bytes();header,body=data.split(b'end_header\n',1);text=header.decode('ascii')
    if 'format binary_little_endian 1.0' not in text:raise RuntimeError('Unexpected source PLY format')
    if 'property float x\nproperty float y\nproperty float z' not in text:raise RuntimeError('Unexpected source vertex layout')
    n=int(next(l.split()[-1] for l in text.splitlines() if l.startswith('element vertex ')))
    m=int(next(l.split()[-1] for l in text.splitlines() if l.startswith('element face ')))
    if len(body)!=n*12+m*13:raise RuntimeError('Unexpected PLY body size')
    entries=np.frombuffer(body[n*12:],dtype=np.dtype([('count','u1'),('indices','<i4',(3,))]),count=m)
    assert np.all(entries['count']==3)
    return entries['indices'].astype(np.int64)


def raster_depth(vertices,faces,k):
    proj=vertices@k.T;uv=proj[:,:2]/proj[:,2:3];z=vertices[:,2]
    depth=np.full((767,1024),np.inf,np.float32)
    for face in faces:
        if np.min(z[face])<=.05:continue
        tri=uv[face];lo=np.maximum(np.floor(tri.min(axis=0)).astype(int),[0,0]);hi=np.minimum(np.ceil(tri.max(axis=0)).astype(int),[1023,766])
        if np.any(lo>hi):continue
        a,b,c=tri;denom=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(denom)<1e-7:continue
        yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        wa=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/denom
        wb=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/denom;wc=1-wa-wb
        inside=(wa>=0)&(wb>=0)&(wc>=0);inverse=wa/z[face[0]]+wb/z[face[1]]+wc/z[face[2]]
        candidate=np.full_like(inverse,np.inf);np.divide(1.,inverse,out=candidate,where=inside&(inverse>0))
        region=depth[lo[1]:hi[1]+1,lo[0]:hi[0]+1];np.minimum(region,candidate,out=region)
    depth[~np.isfinite(depth)]=np.nan
    return depth


def main():
    cv2.setNumThreads(4);OUT.mkdir(exist_ok=True)
    mask_dir=BASE/('hand_masks_sequence' if args.sequence else 'hand_masks_sample')
    mask_report=json.loads((mask_dir/'report.json').read_text());assert mask_report['state']=='completed'
    faces=load_faces(BASE/'hand_sequence_hybrid/frame_000000/left_0.ply')
    baseline=np.load(BASE/'metric_scene_diagnostic/hands_world.npz',allow_pickle=False)
    temporal=np.load(BASE/'hand_registration_temporal/hands_world.npz',allow_pickle=False)
    if args.depth_candidate:aligned=np.load(BASE/'hand_registration_depth/hands_world.npz',allow_pickle=False)
    poses=np.load(BASE/'stereo_sequence/camera_poses.npz',allow_pickle=False)['world_T_rgb']
    k=np.load(BASE/'moge_sample_calibrated/frame_000000/undistorted_input_K.npy',allow_pickle=False)
    report={'state':'running','started_utc':now(),'frames':[],
            'method':'Perspective-correct triangle z-buffer; compare inside eroded SAM2 hand mask and outside cup masks',
            'limitations':['Both masks and stereo are model predictions, not GT surface error.',
                           'Mask overlap diagnoses geometry, not hand detection accuracy.',
                           'No depth correction is applied.']}
    reused={}
    if args.sequence:
        for source in ([OUT] if args.depth_candidate else [BASE/'hand_surface_depth_diagnostic',OUT]):
            if (source/'report.json').exists():
                cached=json.loads((source/'report.json').read_text())
                if cached.get('method')==report['method']:
                    for row in cached['frames']:reused[row['frame_index']]=row
    save(OUT/'report.json',report)
    for row in mask_report['frames']:
        frame=row['frame_index'];masks=np.load(mask_dir/f'frame_{frame:06d}/masks.npz',allow_pickle=False)
        if frame in reused:
            item=reused[frame];assert sorted(h['is_right'] for h in item['hands'])==sorted(masks['is_right'].tolist())
            report['frames'].append(item);save(OUT/'report.json',report);print('frame',frame,'reused surface diagnostic',flush=True);continue
        cups=np.load(BASE/f'object_masks_temporal/masks/{frame:06d}.npz',allow_pickle=False)['masks'].any(axis=0)
        sg=np.load(BASE/f'stereo_sequence/depth_rgb/{frame:06d}.npz',allow_pickle=False)['depth']
        nn=np.load(BASE/f'foundation_stereo_sequence/frame_{frame:06d}/prediction.npz',allow_pickle=False)['depth_rgb']
        pose=poses[frame];item={'frame_index':frame,'hands':[]}
        for side,mask,mask_stats in zip(masks['is_right'],masks['masks'],row['hands']):
            region=cv2.erode(mask.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)&~cups
            hand={'is_right':int(side),'mask_quality_flag':mask_stats['quality_flag'],'segmented_hand_interior_pixels':int(region.sum()),'candidates':{}}
            candidates=[('depth_aligned',aligned['vertices_world'][frame,side])] if args.depth_candidate else [('baseline',baseline['vertices_world'][frame,side]),('temporal',temporal['vertices_world'][frame,side])]
            for name,world in candidates:
                camera=(world-pose[:3,3])@pose[:3,:3];rendered=raster_depth(camera,faces,k);footprint=np.isfinite(rendered)
                stats={'rendered_mask_iou_with_sam2':float((footprint&mask).sum()/max((footprint|mask).sum(),1))}
                for method,depth in [('sgbm',sg),('learned',nn)]:
                    support=region&footprint&np.isfinite(depth);residual=rendered[support]-depth[support]
                    stats[method]={'supported_pixels':int(support.sum()),'rendered_minus_stereo_depth_m_p10_p50_p90':np.percentile(residual,[10,50,90]).tolist() if len(residual) else None}
                hand['candidates'][name]=stats
            item['hands'].append(hand)
        report['frames'].append(item);save(OUT/'report.json',report);print(json.dumps(item),flush=True)
    report.update(state='completed',completed_utc=now(),reused_frame_count=len(reused));save(OUT/'report.json',report)


if __name__=='__main__':main()
