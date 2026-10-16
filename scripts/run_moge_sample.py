"""Three-frame offline MoGe inference in the existing GPU container; no training."""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from moge.model.v2 import MoGeModel

ROOT = Path('/work')
OUT = ROOT / 'outputs/track3/episode_000011/moge_sample'
SOURCE = ROOT / 'outputs/track3/episode_000011/validation'


def write(value):
    tmp = OUT / 'report.json.tmp'
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(OUT / 'report.json')


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--centered-intrinsics', action='store_true',
                        help='Undistort into a centred camera with equal fx/fy matching MoGe projection.')
    args = parser.parse_args()
    if args.centered_intrinsics:
        OUT = ROOT / 'outputs/track3/episode_000011/moge_sample_calibrated'
    cv2.setNumThreads(4)
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    c = json.loads((ROOT / 'data/track3_sample/track_3/public/meta/camera_calibration.json').read_text())['cameras']['ego_cam_a']
    provenance = json.loads((ROOT / 'weights/moge/source.json').read_text())
    report = {'state': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'model': provenance, 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(0), 'frames': [], 'training': False,
              'num_tokens': 800, 'network': 'disabled',
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'centered_intrinsics': args.centered_intrinsics,
              'limitations': ['Predicted depth is not ground truth.',
                              'No object mesh, object tracking, hand reconstruction or policy rollout yet.',
                              'Factory intrinsics used as a focal-length prior after undistortion.']}
    write(report)
    start = time.perf_counter()
    model = MoGeModel.from_pretrained(str(ROOT / 'weights/moge/model.pt')).to('cuda').eval()
    torch.cuda.synchronize()
    report['model_load_seconds'] = time.perf_counter() - start
    print('Loaded model in', report['model_load_seconds'], 'seconds', flush=True)
    for frame in [0, 68, 136]:
        dest = OUT / f'frame_{frame:06d}'
        dest.mkdir(exist_ok=True)
        image = cv2.imread(str(SOURCE / f'cam_a_{frame:06d}.jpg'))
        height0, width0 = image.shape[:2]
        width, height = 1024, round(height0 * 1024 / width0)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        k = np.asarray(c['intrinsic_matrix'], dtype=np.float64)
        k[0, :] *= width / width0
        k[1, :] *= height / height0
        dist = np.asarray(c['distortion_coefficients'], dtype=np.float64)
        if args.centered_intrinsics:
            focal = float(max(k[0, 0], k[1, 1]))
            new_k = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
        else:
            new_k, roi = cv2.getOptimalNewCameraMatrix(k, dist, (width, height), 0, (width, height))
        map_x, map_y = cv2.initUndistortRectifyMap(k, dist, None, new_k, (width, height), cv2.CV_32FC1)
        source_valid = ((map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1))
        image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        cv2.imwrite(str(dest / 'input_undistorted.jpg'), image)
        fov = float(np.degrees(2 * np.arctan(width / (2 * new_k[0, 0]))))
        tensor = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()[None].to('cuda') / 255
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            prediction = model.infer(tensor, num_tokens=800, fov_x=fov, use_fp16=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        depth = prediction['depth'][0].float().cpu().numpy()
        mask = prediction['mask'][0].cpu().numpy().astype(bool)
        intrinsics = prediction['intrinsics'][0].float().cpu().numpy()
        pixel_k = intrinsics.astype(np.float64).copy()
        pixel_k[0, :] *= width
        pixel_k[1, :] *= height
        k_error = float(np.max(np.abs(pixel_k - new_k)))
        if args.centered_intrinsics and (k_error > 1e-3 or source_valid.mean() < .999):
            raise RuntimeError(f'Camera contract failed: K error={k_error}, valid remap={source_valid.mean()}')
        valid = mask & np.isfinite(depth) & (depth > 0)
        if not valid.any():
            raise RuntimeError(f'No valid positive depth for frame {frame}')
        np.save(dest / 'depth.npy', depth)
        np.save(dest / 'mask.npy', mask)
        np.save(dest / 'intrinsics_normalized.npy', intrinsics)
        np.save(dest / 'undistorted_input_K.npy', new_k)
        cv2.imwrite(str(dest / 'mask.png'), mask.astype(np.uint8) * 255)
        lo, hi = np.percentile(depth[valid], [2, 98])
        scaled = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
        scaled[~valid] = 0
        colored = cv2.applyColorMap(np.uint8(scaled * 255), cv2.COLORMAP_TURBO)
        colored[~valid] = 0
        cv2.imwrite(str(dest / 'depth_preview.jpg'), colored)
        r = {'frame_index': frame, 'input_resolution': [width, height],
             'inference_seconds': elapsed, 'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
             'peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2,
             'valid_positive_depth_fraction': float(valid.mean()),
             'depth_m_percentiles_5_50_95': np.percentile(depth[valid], [5, 50, 95]).tolist(),
             'input_fov_x_deg': fov, 'output_intrinsics_normalized': intrinsics.tolist(),
             'input_intrinsics_pixels': new_k.tolist(), 'output_intrinsics_pixels': pixel_k.tolist(),
             'intrinsics_max_abs_difference_pixels': k_error,
             'valid_undistortion_source_fraction': float(source_valid.mean())}
        report['frames'].append(r)
        write(report)
        print(json.dumps(r), flush=True)
    report.update(state='completed', completed_utc=datetime.now(timezone.utc).isoformat(),
                  total_seconds=time.perf_counter() - start)
    write(report)
    print('Completed offline three-frame inference.', flush=True)


if __name__ == '__main__':
    main()
