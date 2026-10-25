"""Multi-view silhouette agreement of current vs procedural geometry on every scored episode (validation only).
For each (episode, object): every 5th frame with a finite FoundationPose v4 camera_T_link and a SAM2 mask of >= 1500 px:
IoU and rendered/mask area ratio of (a) the current SAM3D mesh (decimated to 20k faces) and (b) the procedural
<=4096-face mesh, both at the same pose (scaled by the FP SOURCE scale). Hands occlude the masks, so absolute IoU is
low while an object is held; the comparison is paired (same frames).
Writes outputs/track3/procedural_geometry_v1/eval/silhouettes.json.
"""
import json, sys
from pathlib import Path
import numpy as np, trimesh, fast_simplification
sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, EPS, load_base, occurrences
from silhouette import compare


def main(only=None):
    occ = occurrences(); f = OUT/'eval/silhouettes.json'; rep = json.loads(f.read_text()) if (only and f.exists()) else {}
    cache = {}
    for name, items in occ.items():
        if only and name not in only: continue
        old = load_base(name)
        old = trimesh.Trimesh(*fast_simplification.simplify(np.asarray(old.vertices, np.float32), np.asarray(old.faces, np.int32),
                                                            target_reduction=max(0.0, 1 - 20000/len(old.faces))))
        new = trimesh.load(OUT/name/'visual_4096.obj', force='mesh')
        for ep, s, _ in items:
            used, out = compare(ep, name, {'current': old, 'procedural': new}, step=5)
            r = dict(frames=len(used))
            for k, v in out.items():
                if v: r[k] = dict(iou_median=float(np.median([x['iou'] for x in v])), iou_mean=float(np.mean([x['iou'] for x in v])),
                                  area_ratio_median=float(np.median([x['area_ratio'] for x in v])))
            if out['current']:
                d = np.array([b['iou'] - a['iou'] for a, b in zip(out['current'], out['procedural'])])
                r['paired_iou_delta_median'] = float(np.median(d)); r['frames_procedural_better'] = int((d > 0).sum())
            rep[f'{ep}:{name}'] = r; print(ep, name, json.dumps(r), flush=True)
    (OUT/'eval').mkdir(exist_ok=True); (OUT/'eval/silhouettes.json').write_text(json.dumps(rep, indent=1))


if __name__ == '__main__': main(sys.argv[1:] or None)
