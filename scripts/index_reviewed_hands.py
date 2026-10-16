"""Index corrected frames without overwriting the original hand experiment."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/track3/episode_000011'
original=BASE/'hand_sequence_hybrid';repair=BASE/'hand_sequence_identity_repair'
audit=json.loads((original/'identity_audit.json').read_text())
report=json.loads((repair/'report.json').read_text())
assert report['state']=='completed'
assert report['audit_sha256']==hashlib.sha256((original/'identity_audit.json').read_bytes()).hexdigest()
changed={row['frame_index'] for row in audit['changes']}
assert changed=={row['frame_index'] for row in report['frames']}
frames=[]
for frame in range(137):
    folder=(repair if frame in changed else original)/f'frame_{frame:06d}'
    path=folder/'predictions.json';hands=json.loads(path.read_text());sides=[int(p['is_right']) for p in hands]
    assert len(set(sides))==len(sides),f'Duplicate identity remains in frame {frame}'
    frames.append({'frame_index':frame,'source':str(folder.relative_to(BASE)),
                   'predictions_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                   'present_is_right':sides,'reconstructed_hands':len(hands),'identity_repaired':frame in changed})
out=BASE/'hand_sequence_reviewed';out.mkdir(exist_ok=True)
r={'state':'completed','created_utc':datetime.now(timezone.utc).isoformat(),'frames':frames,
   'total_hands':sum(f['reconstructed_hands'] for f in frames),'two_hand_frames':sum(f['reconstructed_hands']==2 for f in frames),
   'one_hand_frames':sum(f['reconstructed_hands']==1 for f in frames),'repair_frames':sorted(changed),
   'limitations':['Human visual quality review of temporal heuristic labels; no GT hand labels.',
                  'An absent hand remains absent; no interpolation or physical alignment.',
                  'Reviewed identity is not a hand-pose accuracy guarantee.']}
(out/'report.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k!='frames'},indent=2))
