"""Per-object motion of a learning replay: achieved vs reference displacement from frame 0 (cm), mean position error (cm),
mean orientation error (deg), and whether mean errors meet the SP thresholds (10 cm, 0.5 rad).
Usage: python scripts/summarize_track3_object_motion.py <tree> <episode> [step]  (default: every evaluated step)."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

R = Path(__file__).resolve().parents[1]

def summarize(ck, names):
    z = np.load(ck); A = z['achieved_pose_w'][:, 0]; G = z['reference_pose_w']; out = {}
    for b, n in enumerate(names):
        a, g = A[:, b], G[:, b]
        rel = Rotation.from_quat(a[:, [4, 5, 6, 3]]).inv()*Rotation.from_quat(g[:, [4, 5, 6, 3]])  # poses are xyz + wxyz
        out[n] = dict(moved_cm=round(float(np.linalg.norm(a[:, :3]-a[:1, :3], axis=1).max()*100), 1),
                      ref_moved_cm=round(float(np.linalg.norm(g[:, :3]-g[:1, :3], axis=1).max()*100), 1),
                      pos_err_cm=round(float(np.linalg.norm(a[:, :3]-g[:, :3], axis=1).mean()*100), 1),
                      ori_err_deg=round(float(np.degrees(rel.magnitude()).mean()), 1))
        out[n]['sp_ok'] = bool(out[n]['pos_err_cm'] <= 10) and bool(out[n]['ori_err_deg'] <= np.degrees(.5))
    return out

if __name__ == '__main__':
    tree, ep = sys.argv[1], int(sys.argv[2]); L = R/'outputs/track3'/tree/f'episode_{ep:06d}/learning'
    names = [o['name'] for o in json.loads((R/'outputs/track3'/tree/f'episode_{ep:06d}/objects/objects.json').read_text())['objects']]
    steps = [sys.argv[3]] if len(sys.argv) > 3 else sorted((p.name.split('_')[1] for p in (L/'evaluations').glob('step_*')), key=int)
    for s in steps:
        print(tree, ep, s, json.dumps(summarize(L/f'evaluations/step_{s}/rollout_checkpoint.npz', names)))
