"""Audit duplicate hands and identity continuity using persistent track memory."""
import itertools,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/track3/episode_000011'
source=json.loads((BASE/'hand_sequence_hybrid/report.json').read_text())
mp=json.loads((BASE/'hand_detector_mediapipe_sequence/report.json').read_text())
mp_map={f['frame_index']:f['hands'] for f in mp['frames']}

def iou(a,b):
    inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
    union=(a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-inter
    return float(inter/union) if union else 0.


tracks={};changes=[];duplicates=[];assignments=[]
for f in source['frames']:
    frame=f['frame_index'];pred=json.loads((BASE/f'hand_sequence_hybrid/frame_{frame:06d}/predictions.json').read_text())
    boxes=[np.asarray(p['hand_bbox']) for p in pred];raw=[int(p['is_right']) for p in pred]
    confidence=[float(d['score']) for d in mp_map[frame]] if f['box_source'].startswith('MediaPipe') else [.95]*len(pred)
    keep=list(range(len(pred)))
    if len(keep)==2 and raw[0]==raw[1] and iou(boxes[0],boxes[1])>.5:
        chosen=int(np.argmax(confidence));keep=[chosen]
        duplicates.append({'frame_index':frame,'kept_source_index':chosen,'dropped_source_index':1-chosen,'box_iou':iou(boxes[0],boxes[1])})
    xy=[(boxes[i][:2]+boxes[i][2:])/2/1024 for i in keep]
    best=None
    for state in itertools.permutations([0,1],len(keep)):
        cost=0.
        for j,(index,side) in enumerate(zip(keep,state)):
            c=confidence[index];cost-=np.log(max(.001,c if side==raw[index] else 1-c))
            if side in tracks:
                track=tracks[side];gap=frame-track['frame']
                predicted=track['xy']+track['velocity']*min(gap,5)
                sigma=min(.15,.035+.02*np.sqrt(gap))
                delta=xy[j]-predicted;cost+=float(delta@delta)/sigma**2
        if best is None or cost<best[0]:best=(cost,list(state))
    chosen=best[1]
    for j,side in enumerate(chosen):
        if side in tracks:
            old=tracks[side];gap=frame-old['frame'];velocity=.7*old['velocity']+.3*(xy[j]-old['xy'])/gap
            speed=np.linalg.norm(velocity)
            if speed>.05:velocity*=.05/speed
        else:velocity=np.zeros(2)
        tracks[side]={'frame':frame,'xy':xy[j],'velocity':velocity}
    info={'frame_index':frame,'source_indices':keep,'raw_is_right':[raw[i] for i in keep],
          'suggested_is_right':chosen,'boxes':[boxes[i].tolist() for i in keep]}
    assignments.append(info)
    if len(keep)!=len(pred) or chosen!=info['raw_is_right']:changes.append(info)
result={'method':'Same-class duplicate NMS IoU>0.5; persistent hand memory with classifier scores and bounded-velocity image continuity',
        'changes':changes,'duplicate_detections':duplicates,'assignments':assignments,
        'limitations':['Heuristic assignments require visual review and are not ground-truth labels.',
                       'Changed handedness must rerun WiLoR crop mirroring; no relabel-only repair.',
                       'No missing hands or invisible frames are invented/interpolated.']}
(BASE/'hand_sequence_hybrid/identity_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print('Duplicate candidates removed:',[(d['frame_index'],d['box_iou']) for d in duplicates])
print('Recommended changed frames:',[(c['frame_index'],c['source_indices'],c['raw_is_right'],c['suggested_is_right']) for c in changes])
