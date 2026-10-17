"""Offline first-frame SAM2 initialization and full episode temporal propagation."""
import argparse, hashlib, json, time
from itertools import chain
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--seed-frame',type=int,default=0);p.add_argument('--detection-dir',default='object_detection');p.add_argument('--output-dir',default='object_masks_temporal');a=p.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(4)
    base=a.base;out=base/a.output_dir;out.mkdir(exist_ok=True);(out/'masks').mkdir(exist_ok=True)
    seq=base/'calibrated_sequence';manifest=json.loads((seq/'manifest.json').read_text());count=len(manifest['frames'])
    det=json.loads((base/a.detection_dir/'report.json').read_text());names=det['object_order']
    initial=next(f for f in det['frames'] if f['frame_index']==a.seed_frame)
    report=dict(state='running',started_utc=datetime.now(timezone.utc).isoformat(),training=False,network='disabled',object_order=names,frames=[],source_manifest_sha256=hashlib.sha256((seq/'manifest.json').read_bytes()).hexdigest(),detector_report_sha256=hashlib.sha256((base/a.detection_dir/'report.json').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seed_frame=a.seed_frame,prompt_source='Selected-anchor GroundingDINO boxes -> SAM2 image masks -> bidirectional SAM2 video propagation',limitations=['Automatic candidates, not GT; identity and occlusions unverified.','Overlap and empty masks retained and flagged, not silently repaired.','Interrupted temporal memory restarts from frame zero.'],reference_promoted=False)
    def save():
        tmp=out/'report.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(out/'report.json')
    save();start=time.perf_counter();config='configs/sam2.1/sam2.1_hiera_l.yaml';weight='/work/weights/sam2/sam2.1_hiera_large.pt'
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        image=cv2.imread(str(seq/'frames'/f'{a.seed_frame:06d}.jpg'));model=build_sam2(config,weight,device='cuda');predictor=SAM2ImagePredictor(model);predictor.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        seeds=[];stats={}
        for name in names:
            chosen=initial['objects'][name]['selected']
            if chosen is None:raise RuntimeError('No initial detection for '+name+'; no invented mask')
            masks,scores,_=predictor.predict(box=np.asarray(chosen['xyxy_pixels'],np.float32),multimask_output=True)
            best=int(np.argmax(scores));mask=masks[best].astype(bool)
            if not mask.any():raise RuntimeError('Empty seed for '+name)
            seeds.append(mask);stats[name]=dict(mask_pixels=int(mask.sum()),predicted_iou=float(scores[best]),box=chosen['xyxy_pixels'])
        np.savez_compressed(out/'initial-masks.npz',masks=np.stack(seeds),object_names=np.asarray(names))
        report['initial_objects']=stats;save();del predictor,model;torch.cuda.empty_cache()
        video=build_sam2_video_predictor(config,weight,device='cuda')
        state=video.init_state(str(seq/'frames'),offload_video_to_cpu=True,offload_state_to_cpu=True)
        for oid,mask in enumerate(seeds,1):video.add_new_mask(state,frame_idx=a.seed_frame,obj_id=oid,mask=mask)
        seen=set()
        streams=chain(video.propagate_in_video(state,start_frame_idx=a.seed_frame),video.propagate_in_video(state,start_frame_idx=a.seed_frame,reverse=True))
        for frame,ids,logits in streams:
            if frame in seen:continue
            seen.add(frame)
            actual=[names[int(i)-1] for i in ids];masks=(logits[:,0]>0).cpu().numpy()
            np.savez_compressed(out/'masks'/f'{frame:06d}.npz',masks=masks,object_names=np.asarray(actual))
            overlap=int((masks.sum(axis=0)>1).sum());row=dict(frame_index=int(frame),objects={n:dict(mask_pixels=int(m.sum()),quality_flag='candidate' if m.any() else 'empty_mask') for n,m in zip(actual,masks)},overlap_pixels=overlap)
            report['frames'].append(row)
            if frame in [0,count//2,count-1]:
                preview=cv2.imread(str(seq/'frames'/f'{frame:06d}.jpg'))
                for j,(n,m) in enumerate(zip(actual,masks)):
                    color=np.asarray([(70+97*j)%255,(200+53*j)%255,(40+119*j)%255]);preview[m]=(.55*preview[m]+.45*color).astype(np.uint8)
                    cv2.imwrite(str(out/f'frame_{frame:06d}_{n}.png'),m.astype(np.uint8)*255)
                cv2.imwrite(str(out/f'frame_{frame:06d}_overlay.jpg'),preview)
            if frame%10==0:save();print('frame',frame,'overlap',overlap,flush=True)
    report['frames'].sort(key=lambda f:f['frame_index'])
    assert len(report['frames'])==count
    report.update(state='completed',completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start,total_frames=count);save()
if __name__=='__main__':main()
