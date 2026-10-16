"""CPU input validation and stereo diagnostics; no GT camera alignment or inference."""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.deps' / 'track3'))
import cv2
import numpy as np
import pyarrow.parquet as pq

DATA = ROOT / 'data' / 'track3_sample' / 'track_3' / 'public'
OUT = ROOT / 'outputs' / 'track3' / 'episode_000011' / 'validation'


def stats(values):
    a = np.asarray(values, dtype=float)
    if not a.size:
        return None
    return {'min': float(a.min()), 'median': float(np.median(a)), 'max': float(a.max())}


def main():
    cv2.setNumThreads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(DATA / 'data/chunk-000/episode_000011.parquet')
    rows = table.to_pylist()
    n = len(rows)
    sample_ids = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
    metadata = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    orders = [tuple(o['name'] for o in r['observation.objects']) for r in rows]
    frames = [r['frame_index'] for r in rows]
    ts = np.asarray([r['timestamp'] for r in rows])
    captures = np.asarray([r['capture_time'] for r in rows])
    report = {
        'episode_index': 11, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'rows': n, 'schema_metadata': metadata, 'object_order_actual': list(orders[0]),
        'object_order_constant': len(set(orders)) == 1,
        'object_order_matches_schema': list(orders[0]) == json.loads(metadata['objects']),
        'frame_index_contiguous': frames == list(range(n)),
        'timestamp_delta_seconds': stats(np.diff(ts)),
        'capture_time_delta_seconds': stats(np.diff(captures)),
        'capture_time_strictly_increasing': bool(np.all(np.diff(captures) > 0)),
        'objects': {}, 'videos': {},
        'limitations': ['No camera-to-mocap-world extrinsic estimated.',
                        'Stereo statistics are image geometry diagnostics, not reconstruction scores.',
                        'Video PTS alignment does not prove hardware exposure synchronization.'],
    }
    for name in orders[0]:
        observations = [next(o for o in r['observation.objects'] if o['name'] == name) for r in rows]
        pose = np.asarray([o['pose'] for o in observations], dtype=np.float64)
        visible = np.asarray([o['visible'] for o in observations])
        valid = pose[visible]
        converted = np.concatenate([valid[:, :3], valid[:, 4:], valid[:, 3:4]], axis=1)
        roundtrip = np.concatenate([converted[:, :3], converted[:, 6:7], converted[:, 3:6]], axis=1)
        report['objects'][name] = {
            'visible_frames': int(visible.sum()), 'invisible_frames': int((~visible).sum()),
            'invisible_poses_all_zero': bool(np.all(pose[~visible] == 0)),
            'all_finite': bool(np.isfinite(pose).all()),
            'visible_quaternion_norms': stats(np.linalg.norm(valid[:, 3:], axis=1)),
            'wxyz_xyzw_roundtrip_max_abs': float(np.max(np.abs(valid - roundtrip))) if valid.size else None,
            'visible_translation_min_m': valid[:, :3].min(axis=0).tolist() if valid.size else None,
            'visible_translation_max_m': valid[:, :3].max(axis=0).tolist() if valid.size else None,
        }
    images = {}
    pts = {}
    for camera in ['a', 'b', 'c']:
        path = DATA / f'videos/chunk-000/observation.images.ego_cam_{camera}/episode_000011.mp4'
        probe = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams',
            '-show_frames', '-show_entries',
            'stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration:frame=best_effort_timestamp_time',
            '-of', 'json', str(path),
        ], text=True))
        pts[camera] = np.asarray([float(f['best_effort_timestamp_time']) for f in probe['frames']])
        cap = cv2.VideoCapture(str(path))
        count = 0
        images[camera] = {}
        while True:
            ok, im = cap.read()
            if not ok:
                break
            if count in sample_ids:
                images[camera][count] = im
                cv2.imwrite(str(OUT / f'cam_{camera}_{count:06d}.jpg'), im)
            count += 1
        cap.release()
        report['videos'][camera] = {
            'stream': probe['streams'][0], 'fully_decoded_frames': count,
            'matches_parquet_rows': count == n, 'pts_delta_seconds': stats(np.diff(pts[camera])),
            'pts_vs_parquet_max_abs_seconds': float(np.max(np.abs(pts[camera] - ts))) if len(pts[camera]) == n else None,
        }
    report['max_cross_stream_pts_difference_seconds'] = float(max(
        np.max(np.abs(pts['a'] - pts[c])) for c in ['b', 'c']))
    calibration = json.loads((DATA / 'meta/camera_calibration.json').read_text())
    pair = next(p for p in calibration['stereo_pairs'] if p['camera'] == 'ego')
    left = calibration['cameras'][pair['left']]
    right = calibration['cameras'][pair['right']]
    k1, k2 = np.asarray(left['intrinsic_matrix']), np.asarray(right['intrinsic_matrix'])
    d1, d2 = np.asarray(left['distortion_coefficients']), np.asarray(right['distortion_coefficients'])
    transform = np.asarray(pair['left_to_right'])
    width, height = left['resolution']
    r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(
        k1, d1, k2, d2, (width, height), transform[:3, :3], transform[:3, 3],
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    sift = cv2.SIFT_create(nfeatures=2400)
    matcher = cv2.BFMatcher()
    hypotheses = []
    for rotate in [False, True]:
        results = []
        for frame in sample_ids:
            im1 = images[pair['left'][-1]][frame]
            im2 = images[pair['right'][-1]][frame]
            if rotate:
                im1, im2 = cv2.rotate(im1, cv2.ROTATE_180), cv2.rotate(im2, cv2.ROTATE_180)
            a, da = sift.detectAndCompute(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY), None)
            b, db = sift.detectAndCompute(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY), None)
            matches = [m for m, n2 in matcher.knnMatch(da, db, k=2) if m.distance < .7 * n2.distance]
            x1 = np.float64([a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            x2 = np.float64([b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            u1 = cv2.undistortPoints(x1, k1, d1, R=r1, P=p1).reshape(-1, 2)
            u2 = cv2.undistortPoints(x2, k2, d2, R=r2, P=p2).reshape(-1, 2)
            dy = np.abs(u1[:, 1] - u2[:, 1])
            disparity = u1[:, 0] - u2[:, 0]
            results.append({'frame': frame, 'ratio_test_matches': len(matches),
                            'vertical_residual_median_px': float(np.median(dy)),
                            'vertical_residual_p90_px': float(np.percentile(dy, 90)),
                            'fraction_vertical_residual_under_2px': float(np.mean(dy < 2)),
                            'fraction_positive_disparity': float(np.mean(disparity > 0)),
                            'disparity_median_px': float(np.median(disparity))})
        hypotheses.append({'undo_video_rotation_180': rotate, 'frames': results,
                           'median_vertical_residual_px': float(np.median([r['vertical_residual_median_px'] for r in results]))})
    report['stereo'] = {'pair': pair, 'extrinsic_translation_norm_m': float(np.linalg.norm(transform[:3, 3])),
                        'hypotheses': hypotheses,
                        'rectified_intrinsics_left': p1.tolist(), 'rectified_intrinsics_right': p2.tolist()}
    winning = min(hypotheses, key=lambda h: h['median_vertical_residual_px'])
    report['stereo']['best_tested_hypothesis'] = winning['undo_video_rotation_180']
    report['stereo']['orientation_is_provisional'] = True
    maps1 = cv2.initUndistortRectifyMap(k1, d1, r1, p1, (width, height), cv2.CV_32FC1)
    maps2 = cv2.initUndistortRectifyMap(k2, d2, r2, p2, (width, height), cv2.CV_32FC1)
    middle = sample_ids[len(sample_ids) // 2]
    im1, im2 = images[pair['left'][-1]][middle], images[pair['right'][-1]][middle]
    if winning['undo_video_rotation_180']:
        im1, im2 = cv2.rotate(im1, cv2.ROTATE_180), cv2.rotate(im2, cv2.ROTATE_180)
    rect1 = cv2.remap(im1, *maps1, cv2.INTER_LINEAR)
    rect2 = cv2.remap(im2, *maps2, cv2.INTER_LINEAR)
    panel = np.hstack([rect1, rect2])
    for y in range(0, height, 50):
        cv2.line(panel, (0, y), (2 * width - 1, y), (0, 255, 0), 1)
    cv2.imwrite(str(OUT / 'stereo_rectified_diagnostic.jpg'), panel)
    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=192, blockSize=5,
                                  P1=8 * 25, P2=32 * 25, uniquenessRatio=10,
                                  speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1)
    disp = stereo.compute(cv2.cvtColor(rect1, cv2.COLOR_BGR2GRAY),
                          cv2.cvtColor(rect2, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 16
    valid = disp > 0
    points = cv2.reprojectImageTo3D(disp, q)
    finite = valid & np.isfinite(points).all(axis=2) & (points[:, :, 2] > 0)
    report['stereo']['sgbm_diagnostic_middle_frame'] = {
        'frame': middle, 'positive_disparity_fraction_full_image': float(valid.mean()),
        'positive_finite_depth_fraction_full_image': float(finite.mean()),
        'depth_m_percentiles_5_50_95': np.percentile(points[:, :, 2][finite], [5, 50, 95]).tolist() if finite.any() else None,
        'note': 'Provisional calibration hypothesis, no GT scale/depth validation; full-image fraction includes SGBM left margin.',
    }
    np.savez_compressed(OUT / 'stereo_diagnostic.npz', disparity=disp, Q=q, R1=r1, R2=r2, P1=p1, P2=p2)
    colored = cv2.applyColorMap(np.uint8(np.clip(disp, 0, 128) * 255 / 128), cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    cv2.imwrite(str(OUT / 'disparity_diagnostic.jpg'), colored)
    assert report['frame_index_contiguous'] and report['object_order_constant']
    assert all(v['matches_parquet_rows'] for v in report['videos'].values())
    assert all(v['all_finite'] and v['invisible_poses_all_zero'] for v in report['objects'].values())
    path = OUT / 'report.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('Report:', path, flush=True)
    print('Frames:', n, 'actual object order:', orders[0], 'schema agrees:', report['object_order_matches_schema'], flush=True)
    for h in hypotheses:
        print('Undo 180:', h['undo_video_rotation_180'], 'median rectified vertical residual px:', h['median_vertical_residual_px'], flush=True)


if __name__ == '__main__':
    main()
