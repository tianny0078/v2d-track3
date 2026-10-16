"""Offline image masks for the same three frames, using detector boxes only."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

ROOT = Path('/work')
OUT = ROOT / 'outputs/track3/episode_000011/object_masks'


def save(report):
    tmp = OUT / 'report.json.tmp'
    tmp.write_text(json.dumps(report, indent=2) + '\n')
    tmp.replace(OUT / 'report.json')


def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {'state': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'torch': torch.__version__, 'network': 'disabled', 'training': False,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'frames': [], 'prompt_source': 'GroundingDINO boxes; no manual/GT prompts',
              'limitations': ['SAM2 predicted IoU is a model confidence, not ground-truth accuracy.',
                              'Masks are per image; temporal identity and occlusion handling remain unverified.']}
    save(report)
    start = time.perf_counter()
    model = build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml', str(ROOT/'weights/sam2/sam2.1_hiera_large.pt'), device='cuda')
    predictor = SAM2ImagePredictor(model)
    report['model_load_seconds'] = time.perf_counter() - start
    colors = {'blue_cup': (255, 110, 30), 'beige_cup': (0, 160, 255)}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for frame in [0, 68, 136]:
            det_path = ROOT/f'outputs/track3/episode_000011/object_detection/frame_{frame:06d}.json'
            detection = json.loads(det_path.read_text())
            path = ROOT/f'outputs/track3/episode_000011/moge_sample_calibrated/frame_{frame:06d}/input_undistorted.jpg'
            image = cv2.imread(str(path))
            preview = image.copy()
            dest = OUT / f'frame_{frame:06d}'
            dest.mkdir(exist_ok=True)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            masks_by_name = {}
            stats = {}
            for name in ['blue_cup', 'beige_cup']:
                box = detection['objects'][name]['selected']
                if box is None:
                    stats[name] = {'state': 'no_detection'}
                    continue
                masks, scores, logits = predictor.predict(box=np.asarray(box['xyxy_pixels'], dtype=np.float32), multimask_output=True)
                choice = int(np.argmax(scores))
                mask = masks[choice].astype(bool)
                masks_by_name[name] = mask
                y, x = np.where(mask)
                stats[name] = {'state': 'completed' if len(x) else 'empty_mask',
                               'selected_predicted_iou': float(scores[choice]),
                               'candidate_predicted_ious': scores.tolist(), 'mask_pixels': int(mask.sum()),
                               'fraction_of_image': float(mask.mean()),
                               'centroid_pixels': [float(x.mean()), float(y.mean())] if len(x) else None,
                               'detector_score': box['score'], 'box_prompt_xyxy': box['xyxy_pixels']}
                cv2.imwrite(str(dest / (name + '.png')), mask.astype(np.uint8) * 255)
                color = np.asarray(colors[name], dtype=np.float32)
                preview[mask] = (.55 * preview[mask].astype(np.float32) + .45 * color).astype(np.uint8)
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(preview, contours, -1, colors[name], 2)
                if len(x):
                    cv2.putText(preview, name, (int(x.mean()), max(20, int(y.min()) - 8)), cv2.FONT_HERSHEY_SIMPLEX, .55, colors[name], 2, cv2.LINE_AA)
            torch.cuda.synchronize()
            item = {'frame_index': frame, 'objects': stats, 'inference_seconds': time.perf_counter() - start,
                    'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
                    'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            if len(masks_by_name) == 2:
                a, b = masks_by_name.values()
                union = (a | b).sum()
                item['inter_object_mask_iou'] = float((a & b).sum() / union) if union else None
                np.savez_compressed(dest/'masks.npz', object_names=np.asarray(list(masks_by_name)), masks=np.stack(list(masks_by_name.values())))
            cv2.imwrite(str(dest/'overlay.jpg'), preview)
            report['frames'].append(item)
            save(report)
            print('Frame', frame, json.dumps(item), flush=True)
    report.update(state='completed', completed_utc=datetime.now(timezone.utc).isoformat(),
                  all_objects_nonempty=all(o['state'] == 'completed' for f in report['frames'] for o in f['objects'].values()))
    save(report)


if __name__ == '__main__':
    main()
