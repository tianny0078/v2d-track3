"""Independent bundled-model hand detector check, without downloading new assets."""
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--sequence',action='store_true')
parser.add_argument('--base',type=Path,help='Episode output directory containing calibrated_sequence/manifest.json')
args=parser.parse_args()
ROOT=Path('/work');BASE=args.base or ROOT/'outputs/track3/episode_000011'
OUT=BASE/('hand_detector_mediapipe_sequence' if args.sequence else 'hand_detector_mediapipe')
frame_ids=[0,68,136]
if args.sequence:
    manifest=json.loads((BASE/'calibrated_sequence/manifest.json').read_text())
    assert manifest['state']=='completed'
    frame_ids=[r['frame_index'] for r in manifest['frames']]
    assert frame_ids==list(range(len(frame_ids)))
OUT.mkdir(parents=True,exist_ok=True)
report={'state':'running','started_utc':datetime.now(timezone.utc).isoformat(),'mediapipe':mp.__version__,
        'training':False,'network':'disabled','frames':[],
        'handedness_note':'Raw MediaPipe handedness assumes a mirrored input; do not export it as final robot handedness without calibration.'}
with mp.solutions.hands.Hands(static_image_mode=not args.sequence,max_num_hands=2,model_complexity=1,min_detection_confidence=.15,min_tracking_confidence=.2) as detector:
    for frame in frame_ids:
        path=BASE/f'calibrated_sequence/frames/{frame:06d}.jpg' if args.sequence else ROOT/f'outputs/track3/episode_000011/moge_sample_calibrated/frame_{frame:06d}/input_undistorted.jpg'
        image=cv2.imread(str(path))
        h,w=image.shape[:2];start=time.perf_counter();result=detector.process(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        hands=[]
        for landmarks,handedness in zip(result.multi_hand_landmarks or [],result.multi_handedness or []):
            xyz=np.asarray([[p.x,p.y,p.z] for p in landmarks.landmark]);xy=xyz[:,:2]*np.asarray([w,h])
            raw=handedness.classification[0]
            hands.append({'raw_label':raw.label,'score':raw.score,'normalized_landmarks':xyz.tolist(),
                          'bbox_from_landmarks_pixels':[float(xy[:,0].min()),float(xy[:,1].min()),float(xy[:,0].max()),float(xy[:,1].max())]})
            mp.solutions.drawing_utils.draw_landmarks(image,landmarks,mp.solutions.hands.HAND_CONNECTIONS)
        item={'frame_index':frame,'detected_hands':len(hands),'hands':hands,'inference_seconds':time.perf_counter()-start}
        report['frames'].append(item);cv2.imwrite(str(OUT/f'frame_{frame:06d}.jpg'),image)
        (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('frame',frame,'hands',len(hands),flush=True)
report.update(state='completed',completed_utc=datetime.now(timezone.utc).isoformat())
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
