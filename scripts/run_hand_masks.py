"""Offline hand masks from reviewed detections and independent visible landmarks."""
import argparse,json,shutil,time
from pathlib import Path
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from download_track3_sample import now,save

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sequence',action='store_true');args=p.parse_args()
ROOT=Path('/work');BASE=ROOT/'outputs/track3/episode_000011';OUT=BASE/('hand_masks_sequence' if args.sequence else 'hand_masks_sample')
OUT.mkdir(exist_ok=True);cv2.setNumThreads(4);torch.set_num_threads(4)
index=json.loads((BASE/'hand_sequence_reviewed/report.json').read_text());source=json.loads((BASE/'hand_sequence_hybrid/report.json').read_text())
audit=json.loads((BASE/'hand_sequence_hybrid/identity_audit.json').read_text());assign={f['frame_index']:f for f in audit['assignments']}
mp=json.loads((BASE/'hand_detector_mediapipe_sequence/report.json').read_text())
report={'state':'running','started_utc':now(),'training':False,'network':'disabled','frames':[],
        'prompt_source':'Reviewed automatic boxes; image-visible MP keypoints outside cup masks as positive points; cup interiors as negative points',
        'limitations':['SAM2 score is model confidence, not GT mask accuracy.',
                       'No GT prompts, missing hand invention, mask clipping or physical contact fitting.',
                       'Optional SAM2 CUDA hole-filling extension may be absent, as in the existing cup runs.']}
reuse={}
if args.sequence:
    for source_dir in [BASE/'hand_masks_sample',OUT]:
        if (source_dir/'report.json').exists():
            cached=json.loads((source_dir/'report.json').read_text())
            if cached.get('prompt_source')==report['prompt_source']:
                for row in cached['frames']:
                    frame=row['frame_index'];folder=source_dir/f'frame_{frame:06d}'
                    if (folder/'masks.npz').exists():reuse[frame]=(folder,row)
save(OUT/'report.json',report);start=time.perf_counter()
model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',str(ROOT/'weights/sam2/sam2.1_hiera_large.pt'),device='cuda')
predictor=SAM2ImagePredictor(model)
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    for frame in (range(137) if args.sequence else [0,68,136]):
        dest=OUT/f'frame_{frame:06d}';dest.mkdir(exist_ok=True)
        if frame in reuse:
            cached,row=reuse[frame]
            with np.load(cached/'masks.npz',allow_pickle=False) as array:
                assert sorted(array['is_right'].tolist())==sorted(index['frames'][frame]['present_is_right'])
                assert array['masks'].shape[1:]==(767,1024)
            if cached!=dest:
                for name in ['masks.npz','overlay.jpg']:shutil.copyfile(cached/name,dest/name)
            report['frames'].append(dict(row,cache_reused_from=str(cached)));save(OUT/'report.json',report)
            print('frame',frame,'reused reviewed hand-mask cache',flush=True);continue
        image=cv2.imread(str(BASE/f'calibrated_sequence/frames/{frame:06d}.jpg'));preview=image.copy()
        hands=json.loads((BASE/index['frames'][frame]['source']/'predictions.json').read_text())
        cups=np.load(BASE/f'object_masks_temporal/masks/{frame:06d}.npz',allow_pickle=False)['masks'];cup_union=np.any(cups,axis=0)
        negative=[]
        for region in cups:
            distance=cv2.distanceTransform(region.astype(np.uint8),cv2.DIST_L2,5);y,x=np.unravel_index(np.argmax(distance),distance.shape)
            if distance[y,x]>3:negative.append([x,y])
        predictor.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));regions=[];sides=[];stats=[]
        for i,hand in enumerate(hands):
            side=int(hand['is_right']);box=np.asarray(hand['hand_bbox']).copy();box[[0,2]]=np.clip(box[[0,2]],0,1023);box[[1,3]]=np.clip(box[[1,3]],0,766)
            positives=[]
            if source['frames'][frame]['box_source'].startswith('MediaPipe'):
                detection=mp['frames'][frame]['hands'][assign[frame]['source_indices'][i]]
                xy=np.asarray(detection['normalized_landmarks'])[:,:2]*[1024,767]
                for x,y in xy:
                    if 0<=x<1023 and 0<=y<766 and not cup_union[int(y),int(x)]:positives.append([float(x),float(y)])
            coords=np.asarray(positives+negative,np.float32) if positives else None
            labels=np.asarray([1]*len(positives)+[0]*len(negative),np.int32) if positives else None
            masks,scores,_=predictor.predict(box=box.astype(np.float32),point_coords=coords,point_labels=labels,multimask_output=True)
            choice=int(np.argmax(scores));mask=masks[choice].astype(bool);area=int(mask.sum());overlap=float((mask&cup_union).sum()/max(area,1))
            regions.append(mask);sides.append(side)
            stats.append({'is_right':side,'mask_pixels':area,'cup_overlap_fraction':overlap,'selected_predicted_iou':float(scores[choice]),
                          'positive_prompt_count':len(positives),'negative_prompt_count':len(negative) if positives else 0,
                          'quality_flag':'review_overlap' if overlap>.2 else ('empty_mask' if not area else 'candidate')})
            color=np.asarray([40,210,60] if side else [220,70,190]);preview[mask]=(.55*preview[mask]+.45*color).astype(np.uint8)
            contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(preview,contours,-1,color.tolist(),2)
        np.savez_compressed(dest/'masks.npz',is_right=np.asarray(sides),masks=np.asarray(regions,dtype=bool))
        cv2.imwrite(str(dest/'overlay.jpg'),preview);item={'frame_index':frame,'hands':stats}
        report['frames'].append(item);save(OUT/'report.json',report);print(json.dumps(item),flush=True)
report.update(state='completed',completed_utc=now(),elapsed_seconds=time.perf_counter()-start,reused_frame_count=len(reuse));save(OUT/'report.json',report)
