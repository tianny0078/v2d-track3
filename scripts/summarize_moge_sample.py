"""Inspect completed MoGe arrays and render a shared-scale diagnostic figure."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/track3/episode_000011/moge_sample_calibrated'


def main():
    cv2.setNumThreads(4)
    run = json.loads((OUT / 'report.json').read_text())
    if run['state'] != 'completed':
        raise RuntimeError('Inference has not completed; no summary written')
    low, high = 0.3, 6.0
    tile_w, tile_h = 640, 479
    margin, header, row_h, footer = 20, 40, 535, 90
    canvas = np.full((header + row_h * 3 + footer, tile_w * 2 + margin * 3, 3), 245, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'frames': [],
              'source_report': str(OUT / 'report.json'), 'shared_display_depth_range_m': [low, high],
              'limitations': ['Depth predictions have not been compared with metric depth ground truth.',
                              'This is a depth diagnostic, not a complete object reconstruction or policy rollout.',
                              'Colormap limits clip the preview only; raw depth arrays are preserved.']}
    for index, info in enumerate(run['frames']):
        frame = info['frame_index']
        p = OUT / f'frame_{frame:06d}'
        image = cv2.imread(str(p / 'input_undistorted.jpg'))
        depth = np.load(p / 'depth.npy', allow_pickle=False)
        mask = np.load(p / 'mask.npy', allow_pickle=False)
        intrinsic = np.load(p / 'intrinsics_normalized.npy', allow_pickle=False).astype(np.float64)
        known = np.load(p / 'undistorted_input_K.npy', allow_pickle=False)
        if depth.shape != image.shape[:2] or mask.shape != depth.shape:
            raise RuntimeError(f'Image/depth/mask dimensions differ: frame {frame}')
        intrinsic[0] *= depth.shape[1]
        intrinsic[1] *= depth.shape[0]
        valid = mask & np.isfinite(depth) & (depth > 0)
        check = {'frame_index': frame, 'shape': list(depth.shape), 'dtype': str(depth.dtype),
                 'all_finite': bool(np.isfinite(depth).all()), 'all_positive': bool((depth > 0).all()),
                 'valid_mask_fraction': float(valid.mean()),
                 'intrinsics_max_abs_difference_pixels': float(np.max(np.abs(intrinsic - known))),
                 'depth_min_max_m': [float(depth[valid].min()), float(depth[valid].max())],
                 'preview_clipped_fraction': float(((depth[valid] < low) | (depth[valid] > high)).mean()),
                 'files': {}}
        for path in sorted(p.iterdir()):
            if path.is_file():
                check['files'][path.name] = {'size_bytes': path.stat().st_size,
                                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        if not check['all_finite'] or not check['all_positive'] or check['intrinsics_max_abs_difference_pixels'] > 1e-3:
            raise RuntimeError(f'Depth/camera validation failed: frame {frame}')
        normalized = np.clip((depth - low) / (high - low), 0, 1)
        colored = cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_TURBO)
        colored[~valid] = 0
        cv2.imwrite(str(p / 'depth_preview_shared_scale.jpg'), colored)
        y = header + index * row_h
        cv2.putText(canvas, f'Frame {frame:03d}: undistorted RGB', (margin, y + 23), font, .65, (35, 35, 35), 1, cv2.LINE_AA)
        cv2.putText(canvas, 'MoGe predicted depth (metres)', (tile_w + margin * 2, y + 23), font, .65, (35, 35, 35), 1, cv2.LINE_AA)
        canvas[y + 34:y + 34 + tile_h, margin:margin + tile_w] = cv2.resize(image, (tile_w, tile_h))
        canvas[y + 34:y + 34 + tile_h, tile_w + margin * 2:tile_w * 2 + margin * 2] = cv2.resize(colored, (tile_w, tile_h))
        result['frames'].append(check)
    y = header + row_h * 3 + 8
    ramp = np.tile(np.arange(256, dtype=np.uint8), (20, 1))
    bar = cv2.resize(cv2.applyColorMap(ramp, cv2.COLORMAP_TURBO), (tile_w, 20))
    canvas[y:y + 20, tile_w + margin * 2:tile_w * 2 + margin * 2] = bar
    cv2.putText(canvas, '0.3 m (near)', (tile_w + margin * 2, y + 42), font, .55, (35, 35, 35), 1, cv2.LINE_AA)
    cv2.putText(canvas, '6.0 m (far)', (tile_w * 2 + margin * 2 - 100, y + 42), font, .55, (35, 35, 35), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Shared colour scale. Predictions, not ground truth.', (margin, 27), font, .65, (35, 35, 35), 1, cv2.LINE_AA)
    figure = OUT / 'three_frame_comparison.jpg'
    if not cv2.imwrite(str(figure), canvas):
        raise RuntimeError('Failed to save comparison figure')
    result['figure'] = str(figure)
    result['checks_passed'] = True
    (OUT / 'quality_report.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Passed array/camera checks for', len(result['frames']), 'frames')
    print('Figure:', figure)


if __name__ == '__main__':
    main()
