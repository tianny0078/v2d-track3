"""Offline object-name detections on three anchors of any calibrated episode."""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import groundingdino
import numpy as np
import torch
from groundingdino.models import build_model
from groundingdino.util.inference import load_image, predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from torchvision.ops import box_convert

ROOT = Path('/work')
OUT = ROOT / 'outputs/track3/episode_000011/object_detection'


def save(report):
    tmp = OUT / 'report.json.tmp'
    tmp.write_text(json.dumps(report, indent=2) + '\n')
    tmp.replace(OUT / 'report.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--objects', nargs='+', required=True)
    parser.add_argument('--prompt-overrides', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    global OUT
    OUT = args.output_dir or args.base/'object_detection'
    manifest=json.loads((args.base/'calibrated_sequence/manifest.json').read_text())
    count=len(manifest['frames'])
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    config = SLConfig.fromfile(str(Path(groundingdino.__file__).parent / 'config/GroundingDINO_SwinT_OGC.py'))
    config.device = args.device
    config.text_encoder_type = '/work/weights/bert'
    report = {'state': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'device': args.device, 'torch': torch.__version__, 'network': 'disabled',
              'box_threshold': .25, 'text_threshold': .2, 'frames': [],
              'object_order': args.objects, 'training': False,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save(report)
    started = time.perf_counter()
    model = build_model(config)
    checkpoint = torch.load(ROOT / 'weights/grounding_dino/groundingdino_swint_ogc.pth', map_location='cpu', weights_only=True)
    status = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    report['load_state'] = {'missing_keys': status.missing_keys, 'unexpected_keys': status.unexpected_keys}
    model.eval().to(args.device)
    report['model_load_seconds'] = time.perf_counter() - started
    prompts = {name: name.replace('_', ' ')+'.' for name in args.objects}
    if args.prompt_overrides: prompts.update(json.loads(args.prompt_overrides.read_text()))
    report['prompts']=prompts
    colors = {name: (40, 160, 230) for name in args.objects}
    for frame in sorted(set([0,count//2,count-1])):
        path = args.base/f'calibrated_sequence/frames/{frame:06d}.jpg'
        original, image = load_image(str(path))
        height, width = original.shape[:2]
        annotated = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        detections = {}
        for name, prompt in prompts.items():
            if args.device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            boxes, scores, phrases = predict(model=model, image=image, caption=prompt,
                                            box_threshold=.25, text_threshold=.2, device=args.device)
            if args.device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            xyxy = box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy') * torch.tensor([width, height, width, height])
            choices = sorted([{'xyxy_pixels': box.tolist(), 'score': float(score), 'phrase': phrase}
                              for box, score, phrase in zip(xyxy, scores, phrases)], key=lambda x: -x['score'])
            detections[name] = {'prompt': prompt, 'candidates': choices, 'selected': choices[0] if choices else None,
                                'inference_seconds': elapsed}
            if choices:
                x0, y0, x1, y1 = [int(round(x)) for x in choices[0]['xyxy_pixels']]
                cv2.rectangle(annotated, (x0, y0), (x1, y1), colors[name], 2)
                cv2.putText(annotated, f'{name} {choices[0]["score"]:.2f}', (x0, max(20, y0 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, colors[name], 2, cv2.LINE_AA)
        item = {'frame_index': frame, 'image_size': [width, height], 'objects': detections,
                'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        (OUT / f'frame_{frame:06d}.json').write_text(json.dumps(item, indent=2) + '\n')
        cv2.imwrite(str(OUT / f'frame_{frame:06d}_boxes.jpg'), annotated)
        report['frames'].append(item)
        save(report)
        print('Frame', frame, {n: d['selected'] for n, d in detections.items()}, flush=True)
    report.update(state='completed', completed_utc=datetime.now(timezone.utc).isoformat(),
                  all_objects_detected=all(d['selected'] is not None for f in report['frames'] for d in f['objects'].values()))
    save(report)


if __name__ == '__main__':
    main()
