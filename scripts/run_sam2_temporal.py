"""Track the two first-frame masks through the public episode without GT prompts."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

ROOT = Path('/work')
OUT = ROOT/'outputs/track3/episode_000011/object_masks_temporal'


def save(report):
    temp = OUT/'report.json.tmp'
    temp.write_text(json.dumps(report, indent=2)+'\n')
    temp.replace(OUT/'report.json')


def main():
    cv2.setNumThreads(4)
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'masks').mkdir(exist_ok=True)
    sequence = ROOT/'outputs/track3/episode_000011/calibrated_sequence'
    names = {1: 'blue_cup', 2: 'beige_cup'}
    report = {'state': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'training': False, 'network': 'disabled', 'prompt_source': 'First-frame GroundingDINO+SAM2 masks only',
              'object_ids': names, 'frames': [], 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limitations': ['Temporal identity requires visual review; no mask ground truth is available.',
                              'A partial inference restarts from frame 0 to reconstruct temporal memory; no training occurs.']}
    save(report)
    predictor = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml', str(ROOT/'weights/sam2/sam2.1_hiera_large.pt'), device='cuda')
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        state = predictor.init_state(str(sequence/'frames'), offload_video_to_cpu=True, offload_state_to_cpu=False)
        for obj_id, name in names.items():
            mask = cv2.imread(str(ROOT/f'outputs/track3/episode_000011/object_masks/frame_000000/{name}.png'), cv2.IMREAD_GRAYSCALE)
            if mask is None or not (mask > 0).any():
                raise RuntimeError('Initial independent mask missing')
            predictor.add_new_mask(state, frame_idx=0, obj_id=obj_id, mask=mask > 0)
        for frame, object_ids, logits in predictor.propagate_in_video(state):
            masks = (logits[:, 0] > 0).cpu().numpy()
            actual_names = [names[int(i)] for i in object_ids]
            np.savez_compressed(OUT/'masks'/f'{frame:06d}.npz', masks=masks, object_names=np.asarray(actual_names))
            a, b = masks
            union = int((a | b).sum())
            metrics = {'frame_index': int(frame), 'objects': {name: {'mask_pixels': int(mask.sum())} for name, mask in zip(actual_names, masks)},
                       'inter_object_mask_iou': float((a & b).sum()/union) if union else None}
            report['frames'].append(metrics)
            if frame in [0, 68, 136]:
                image = cv2.imread(str(sequence/'frames'/f'{frame:06d}.jpg'))
                for name, mask, color in zip(actual_names, masks, [(255, 110, 30), (0, 160, 255)]):
                    image[mask] = (.55*image[mask].astype(np.float32)+.45*np.asarray(color)).astype(np.uint8)
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(image, contours, -1, color, 2)
                    cv2.imwrite(str(OUT/f'frame_{frame:06d}_{name}.png'), mask.astype(np.uint8)*255)
                cv2.imwrite(str(OUT/f'frame_{frame:06d}_overlay.jpg'), image)
                print('Sample',json.dumps(metrics),flush=True)
            if frame % 10 == 0:
                save(report)
        torch.cuda.synchronize()
    report.update(state='completed', completed_utc=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=time.perf_counter()-started,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                  total_frames=len(report['frames']))
    if report['total_frames'] != 137:
        raise RuntimeError('Incomplete propagation')
    save(report)
    print('Completed temporal masks for', report['total_frames'], 'frames',flush=True)


if __name__ == '__main__':
    main()
