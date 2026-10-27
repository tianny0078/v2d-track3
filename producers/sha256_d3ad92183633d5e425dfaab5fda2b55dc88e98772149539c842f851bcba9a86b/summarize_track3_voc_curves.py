"""P0-3 (advice-v2): per tree/episode training curves around VOC removal from segments/step_*/metrics.jsonl.
Prints, per 1M-step bin: voc_scale, stage, object_keypoints reward/step, reference progress, object-termination share."""
import json, sys
from pathlib import Path
import numpy as np
T = Path('/home/tianyuan/projects/v2d/outputs/track3')
K = dict(voc='curriculum/voc_scale', stage='curriculum/stage_index', okp='reward_components/object_keypoints_weighted_mean_per_env_step',
         prog='episode/reference_progress_mean_per_completed_episode', oterm='termination/object_fraction_per_completed_episode',
         len='episode/length_mean_per_completed_episode')
def curve(tree, ep):
    rows = []
    for f in sorted((T/tree/f'episode_{ep:06d}/learning/segments').glob('step_*/metrics.jsonl')):
        for l in f.read_text().splitlines():
            d = json.loads(l); rows.append((d['environment_steps'], {k: d['metrics'].get(v, np.nan) for k, v in K.items()}))
    rows.sort(key=lambda r: r[0]); out = []
    for b in range(10):
        sel = [m for s, m in rows if b*1e6 <= s < (b+1)*1e6]
        if sel: out.append((b, {k: float(np.nanmean([m[k] for m in sel])) for k in K}))
    return out
if __name__ == '__main__':
    tree = sys.argv[1]
    for ep in map(int, sys.argv[2:]):
        print(f'== {tree} ep{ep}  (bin M: voc stage okp prog oterm len)')
        for b, m in curve(tree, ep):
            print(f"  {b}-{b+1}M voc {m['voc']:.2f} st {m['stage']:.0f} okp {m['okp']:.3f} prog {m['prog']:.3f} oterm {m['oterm']:.2f} len {m['len']:.1f}")
