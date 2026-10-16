"""Offline hand reconstruction on the already-calibrated three-frame sample."""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

ROOT=Path('/work')
OUT=ROOT/'outputs/track3/episode_000011/hand_sample'


def encode(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {k:encode(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [encode(v) for v in value]
    return value


def save(report):
    temp=OUT/'report.json.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(OUT/'report.json')


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--low-confidence',action='store_true')
    parser.add_argument('--mediapipe-boxes',action='store_true')
    parser.add_argument('--sequence',action='store_true')
    parser.add_argument('--identity-repair',action='store_true')
    args=parser.parse_args()
    confidence=.1 if args.low_confidence else .3
    if args.low_confidence:OUT=ROOT/'outputs/track3/episode_000011/hand_sample_low_threshold'
    if args.mediapipe_boxes:OUT=ROOT/'outputs/track3/episode_000011/hand_sample_mediapipe_guided'
    if args.sequence:
        if not args.mediapipe_boxes:raise ValueError('Sequence mode requires the validated hybrid detector')
        OUT=ROOT/'outputs/track3/episode_000011/hand_sequence_hybrid'
    repair_map={}
    if args.identity_repair:
        if not args.sequence:raise ValueError('Identity repair requires --sequence')
        audit_path=ROOT/'outputs/track3/episode_000011/hand_sequence_hybrid/identity_audit.json'
        audit=json.loads(audit_path.read_text())
        if audit.get('review_status')!='visually_reviewed':raise ValueError('Review the identity audit before repair')
        repair_map={f['frame_index']:f for f in audit['changes']}
        OUT=ROOT/'outputs/track3/episode_000011/hand_sequence_identity_repair'
    cv2.setNumThreads(4);torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    first=ROOT/'outputs/track3/episode_000011/moge_sample_calibrated/frame_000000'
    k=np.load(first/'undistorted_input_K.npy',allow_pickle=False)
    focal_config=float(k[0,0])*256/1024
    report={'state':'running','started_utc':datetime.now(timezone.utc).isoformat(),'training':False,'network':'disabled',
            'hand_detector_confidence':confidence,
            'mediapipe_box_guidance':args.mediapipe_boxes,
            'full_sequence':args.sequence,
            'identity_repair_only':args.identity_repair,
            'frames':[],'torch':torch.__version__,'actual_focal_pixels':float(k[0,0]),'wilor_focal_config':focal_config,
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'limitations':['Hands are camera-relative; global motion and object-contact alignment are not solved.',
                           'Occluded or image-edge hands may be absent; no ground-truth hand error is available.']}
    save(report)
    started=time.perf_counter()
    pipe=WiLorHandPose3dEstimationPipeline(device=torch.device('cuda'),dtype=torch.float16,
                                         wilor_pretrained_dir='/work/weights/wilor',focal_length=focal_config,verbose=False)
    report['model_load_seconds']=time.perf_counter()-started
    faces=np.asarray(pipe.wilor_model.mano.faces)
    detections_map={}
    if args.mediapipe_boxes:
        mp_subdir='hand_detector_mediapipe_sequence' if args.sequence else 'hand_detector_mediapipe'
        mp_report=json.loads((ROOT/f'outputs/track3/episode_000011/{mp_subdir}/report.json').read_text())
        detections_map={f['frame_index']:f['hands'] for f in mp_report['frames']}
    if args.identity_repair:report['audit_sha256']=hashlib.sha256(audit_path.read_bytes()).hexdigest();save(report)
    frames=sorted(repair_map) if args.identity_repair else (range(137) if args.sequence else [0,68,136])
    for frame in frames:
        dest=OUT/f'frame_{frame:06d}';dest.mkdir(exist_ok=True)
        path=ROOT/f'outputs/track3/episode_000011/calibrated_sequence/frames/{frame:06d}.jpg' if args.sequence else ROOT/f'outputs/track3/episode_000011/moge_sample_calibrated/frame_{frame:06d}/input_undistorted.jpg'
        image=cv2.imread(str(path))
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
        if args.mediapipe_boxes:
            detections=detections_map.get(frame,[])
        else:
            detections=[]
        if args.identity_repair:
            repair=repair_map[frame]
            predictions=pipe.predict_with_bboxes(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),
                bboxes=np.asarray(repair['boxes'],dtype=np.float32),is_rights=np.asarray(repair['suggested_is_right']))
        elif detections:
            boxes=[];flags=[]
            for det in detections:
                box=np.asarray(det['bbox_from_landmarks_pixels']);center=(box[:2]+box[2:])/2;half=(box[2:]-box[:2])*.6
                boxes.append(np.concatenate([center-half,center+half]))
                # Raw MP assumes mirrored input. This mapping agrees with the independent frame-0 WiLoR left-hand detection.
                flags.append(int(det['raw_label']=='Left'))
            predictions=pipe.predict_with_bboxes(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),bboxes=np.asarray(boxes,dtype=np.float32),is_rights=np.asarray(flags))
        else:
            predictions=pipe.predict(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),hand_conf=confidence)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        (dest/'predictions.json').write_text(json.dumps(encode(predictions),indent=2)+'\n')
        summary=[]
        for i,pred in enumerate(predictions):
            right=bool(int(pred['is_right']));name=('right' if right else 'left')+f'_{i}'
            p=pred['wilor_preds']
            vertices=np.asarray(p['pred_vertices']).reshape(-1,3)+np.asarray(p['pred_cam_t_full']).reshape(1,3)
            joints=np.asarray(p['pred_keypoints_2d']).reshape(-1,2)
            if not np.isfinite(vertices).all() or not np.isfinite(joints).all():raise RuntimeError('Nonfinite hand outputs')
            mesh_faces=faces if right else faces[:,[0,2,1]]
            trimesh.Trimesh(vertices=vertices,faces=mesh_faces,process=False).export(dest/(name+'.ply'))
            for x,y in joints:
                if -10000<x<10000 and -10000<y<10000:
                    cv2.circle(image,(int(round(x)),int(round(y))),3,(20,210,30) if right else (220,80,180),-1)
            box=[int(round(x)) for x in pred['hand_bbox']]
            cv2.rectangle(image,tuple(box[:2]),tuple(box[2:]),(20,210,30) if right else (220,80,180),2)
            cv2.putText(image,name,(box[0],max(20,box[1]-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,(20,210,30) if right else (220,80,180),2,cv2.LINE_AA)
            summary.append({'name':name,'is_right':right,'vertices':len(vertices),'faces':len(faces),
                            'cam_translation_m':np.asarray(p['pred_cam_t_full']).reshape(-1).tolist(),
                            'scaled_focal_pixels':float(p['scaled_focal_length']),
                            'field_shapes':{k:list(np.asarray(v).shape) for k,v in p.items()}})
        cv2.imwrite(str(dest/'joints_overlay.jpg'),image)
        item={'frame_index':frame,'detected_hands':len(predictions),'hands':summary,'inference_seconds':elapsed,
              'box_source':('Reviewed temporal identity repair, reused padded boxes' if args.identity_repair else
                            ('MediaPipe landmark bounds, 20% padding' if detections else 'WiLoR YOLO')),
              'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2}
        report['frames'].append(item);save(report);print(json.dumps(item),flush=True)
    report.update(state='completed',completed_utc=datetime.now(timezone.utc).isoformat());save(report)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        if (OUT/'report.json').exists():
            failed=json.loads((OUT/'report.json').read_text())
            failed.update(state='failed',error_type=type(exc).__name__,error_message=str(exc).split('\n')[0])
            save(failed)
        raise
