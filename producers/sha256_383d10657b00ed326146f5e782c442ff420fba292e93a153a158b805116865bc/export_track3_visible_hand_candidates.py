"""Export image-valid detector candidates; hand identity and metric depth stay unknown."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('output_root',type=Path);a=p.parse_args();b=a.output_root/'hand_detector_mediapipe_sequence';src=b/'report.json';r=json.loads(src.read_text());assert r['state']=='completed'
n=len(r['frames']);xy=np.full((n,2,21,2),np.nan,np.float32);valid=np.zeros((n,2,21),bool);present=np.zeros((n,2),bool);scores=np.full((n,2),np.nan,np.float32);excluded=0; overflow=[]
for f in r['frames']:
 hands=f['hands']
 if len(hands)>2: overflow.append(dict(frame_index=f['frame_index'],raw_count=len(hands),excluded_count=len(hands)-2))
 selected=sorted(enumerate(hands),key=lambda item:(-item[1]['score'],item[0]))[:2] if len(hands)>2 else list(enumerate(hands))
 for j,(_,h) in enumerate(selected):
  i=f['frame_index'];v=np.asarray(h['normalized_landmarks'])[:,:2];mask=np.isfinite(v).all(axis=1)&(v>=0).all(axis=1)&(v<=1).all(axis=1)
  present[i,j]=True;valid[i,j]=mask;scores[i,j]=h['score'];xy[i,j,mask]=v[mask]*[1024,767];excluded+=int((~mask).sum())
np.savez_compressed(b/'visible_candidates.npz',xy_pixels=xy,valid=valid,candidate_present=present,detector_score=scores,calibrated_is_right=np.full((n,2),np.nan),metric_xyz=np.full((n,2,21,3),np.nan,np.float32))
checks={'finite_iff_valid':bool(np.array_equal(np.isfinite(xy).all(axis=-1),valid)),'absent_all_unknown':bool(np.isnan(xy[~present]).all()),'inside_image':bool(((xy[valid]>=[0,0])&(xy[valid]<=[1024,767])).all())}
report=dict(all_passed=all(checks.values()),checks=checks,frame_count=n,overflow_frames=overflow,overflow_selection='top two detector scores; raw report retained; slots have no identity',excluded_outside_image_landmarks=excluded,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),candidate_slots_are_temporally_identified=False,calibrated_handedness_known=False,metric_depth_known=False,ground_truth_known=False,scope='Image-valid detector candidates only; raw outputs retained without clipping or invented observations')
(b/'visible-candidates-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));assert report['all_passed']
