"""Compare mask methods and export diagnostic visible-surface point clouds."""
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/track3/episode_000011'
OUT = BASE/'object_masks_temporal'


def ply(path, xyz, rgb):
    vertex = np.empty(len(xyz), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                       ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    for i, key in enumerate(['x', 'y', 'z']):
        vertex[key] = xyz[:, i]
    for i, key in enumerate(['red', 'green', 'blue']):
        vertex[key] = rgb[:, i]
    header = ('ply\nformat binary_little_endian 1.0\ncomment predicted visible surface in camera coordinates, metres\n'
              f'element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    with path.open('wb') as stream:
        stream.write(header.encode('ascii'))
        vertex.tofile(stream)


def main():
    cv2.setNumThreads(4)
    report = json.loads((OUT/'report.json').read_text())
    if report.get('state') != 'completed' or report['total_frames'] != 137:
        raise RuntimeError('Temporal masks incomplete')
    independent = json.loads((BASE/'object_masks/report.json').read_text())
    values = np.asarray([f['inter_object_mask_iou'] for f in report['frames']])
    summary = {'created_utc': datetime.now(timezone.utc).isoformat(), 'total_frames': 137,
               'inter_object_mask_iou_median': float(np.median(values)),
               'inter_object_mask_iou_max': float(values.max()),
               'frames_iou_over_point8': int((values > .8).sum()),
               'sample_frames': [], 'empty_frames_by_object': {},
               'limitations': ['Overlap measures identity collapse only; it is not segmentation accuracy.',
                               'Point clouds contain visible predicted surfaces, not complete object meshes or rigid poses.',
                               'Camera coordinates +X right, +Y down, +Z forward; no camera-to-world transform is known.',
                               'Depth scale has not been validated against metric depth ground truth.']}
    for name in ['blue_cup', 'beige_cup']:
        summary['empty_frames_by_object'][name] = sum(f['objects'][name]['mask_pixels'] == 0 for f in report['frames'])
    surface = BASE/'visible_surface_diagnostic'
    surface.mkdir(exist_ok=True)
    width, height, margin, row_height = 640, 479, 20, 545
    canvas = np.full((3*row_height+60, 2*width+3*margin, 3), 245, np.uint8)
    for row, frame in enumerate([0, 68, 136]):
        old = cv2.imread(str(BASE/f'object_masks/frame_{frame:06d}/overlay.jpg'))
        new = cv2.imread(str(OUT/f'frame_{frame:06d}_overlay.jpg'))
        old_info = next(f for f in independent['frames'] if f['frame_index'] == frame)
        new_info = next(f for f in report['frames'] if f['frame_index'] == frame)
        y = row*row_height+40
        for column, image in enumerate([old, new]):
            x = margin+column*(width+margin)
            canvas[y+28:y+28+height, x:x+width] = cv2.resize(image, (width, height))
        cv2.putText(canvas, f'Frame {frame:03d}: independent masks', (margin, y+18), cv2.FONT_HERSHEY_SIMPLEX, .65, (30,30,30), 1, cv2.LINE_AA)
        cv2.putText(canvas, f'Frame {frame:03d}: temporal identities', (width+2*margin, y+18), cv2.FONT_HERSHEY_SIMPLEX, .65, (30,30,30), 1, cv2.LINE_AA)
        data = np.load(OUT/'masks'/f'{frame:06d}.npz', allow_pickle=False)
        prediction = BASE/f'moge_sample_calibrated/frame_{frame:06d}'
        depth = np.load(prediction/'depth.npy', allow_pickle=False)
        k = np.load(prediction/'intrinsics_normalized.npy', allow_pickle=False)
        rgb = cv2.cvtColor(cv2.imread(str(prediction/'input_undistorted.jpg')), cv2.COLOR_BGR2RGB)
        image_h, image_w = depth.shape
        fy, fx = float(k[1,1])*image_h, float(k[0,0])*image_w
        cx, cy = float(k[0,2])*image_w, float(k[1,2])*image_h
        item = {'frame_index': frame, 'independent_mask_iou': old_info['inter_object_mask_iou'],
                'temporal_mask_iou': new_info['inter_object_mask_iou'], 'objects': {}}
        for name, mask in zip(data['object_names'].tolist(), data['masks']):
            if mask.shape != depth.shape:
                raise RuntimeError('Mask/depth camera dimensions differ')
            v, u = np.where(mask & np.isfinite(depth) & (depth > 0))
            z = depth[v,u]
            xyz = np.stack([(u+.5-cx)*z/fx, (v+.5-cy)*z/fy, z], axis=1).astype(np.float32)
            file = surface/f'frame_{frame:06d}_{name}.ply'
            ply(file, xyz[::2], rgb[v,u][::2])
            item['objects'][name] = {
                'valid_depth_pixels': len(z), 'exported_points': len(xyz[::2]), 'path': str(file),
                'depth_m_percentiles_5_50_95': np.percentile(z,[5,50,95]).tolist(),
                'visible_surface_centroid_camera_m': xyz.mean(axis=0).tolist(),
                'projection_pixel_centers': 'u+0.5,v+0.5 in MoGe normalized-coordinate convention'}
        summary['sample_frames'].append(item)
    cv2.putText(canvas, 'Blue outline: blue cup. Orange outline: beige cup. Predictions, no mask GT.',
                (margin, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (30,30,30), 1, cv2.LINE_AA)
    figure = OUT/'independent_vs_temporal.jpg'
    cv2.imwrite(str(figure), canvas)
    summary['figure'] = str(figure)
    summary['optional_postprocessing'] = 'sam2._C extension unavailable in existing image; hole-filling was skipped by upstream fallback.'
    (OUT/'quality_report.json').write_text(json.dumps(summary,indent=2)+'\n')
    (surface/'README.md').write_text('# Visible-surface diagnostics\n\nThese camera-frame point clouds use predicted MoGe depth and tracked object masks. They are not complete object meshes, not object-centre poses, and not valid submission artifacts. Depth scale is not yet ground-truth validated.\n')
    print('Checked all 137 mask pairs; exported 6 visible-surface point clouds')
    print('Figure:', figure)


if __name__ == '__main__':
    main()
