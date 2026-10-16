"""Prepare all 137 frames using the verified three-frame camera geometry."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/track3/episode_000011/calibrated_sequence'
cached = OUT/'manifest.json'
if cached.exists() and json.loads(cached.read_text()).get('state') == 'completed':
    print('Reusing calibrated sequence:', OUT)
    raise SystemExit(0)
cv2.setNumThreads(4)
frames = OUT/'frames'
frames.mkdir(parents=True, exist_ok=True)
sample = ROOT/'outputs/track3/episode_000011/moge_sample_calibrated'
camera = json.loads((ROOT/'data/track3_sample/track_3/public/meta/camera_calibration.json').read_text())['cameras']['ego_cam_a']
w, h = 1024, 767
k = np.asarray(camera['intrinsic_matrix'])
k[0] *= w/2028
k[1] *= h/1520
target_k = np.load(sample/'frame_000000/undistorted_input_K.npy', allow_pickle=False)
maps = cv2.initUndistortRectifyMap(k, np.asarray(camera['distortion_coefficients']), None, target_k, (w, h), cv2.CV_32FC1)
video = ROOT/'data/track3_sample/track_3/public/videos/chunk-000/observation.images.ego_cam_a/episode_000011.mp4'
cap = cv2.VideoCapture(str(video))
entries = []
while True:
    ok, image = cap.read()
    if not ok:
        break
    i = len(entries)
    dest = frames/f'{i:06d}.jpg'
    if i in [0, 68, 136]:
        # Reuse the exact tested model input, including its JPEG representation.
        dest.write_bytes((sample/f'frame_{i:06d}/input_undistorted.jpg').read_bytes())
    else:
        # Match the validation-JPEG -> resize -> undistort -> JPEG pipeline.
        ok, encoded = cv2.imencode('.jpg', image)
        if not ok:
            raise RuntimeError('JPEG encode failed')
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        image = cv2.remap(image, *maps, cv2.INTER_LINEAR)
        if not cv2.imwrite(str(dest), image):
            raise RuntimeError('Failed to save frame')
    entries.append({'frame_index': i, 'filename': dest.name, 'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()})
cap.release()
if len(entries) != 137:
    raise RuntimeError('Unexpected source frame count')
report = {'state': 'completed', 'frames': entries, 'resolution': [w, h], 'target_K': target_k.tolist(),
          'sample_inputs_reused_exactly': [0, 68, 136],
          'source_video': str(video), 'frame_alignment': 'frame_index; no temporal resampling'}
cached.write_text(json.dumps(report, indent=2)+'\n')
print('Prepared',len(entries),'calibrated frames')
