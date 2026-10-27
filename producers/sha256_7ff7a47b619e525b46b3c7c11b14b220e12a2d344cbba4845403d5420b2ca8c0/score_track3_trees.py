"""Score every evaluated rollout of the given learning trees with the #3 selection proxy (reference = learning_v7_geom,
frame-0 Kabsch scene alignment; see select_track3_submission_candidates.py) plus per-object follow fraction
(frames where the reference object is > 5 cm from its start: share with position error < 5 cm).
Usage: score_track3_trees.py OUT_DIR TREE [TREE ...]  -> OUT_DIR/scores.json, OUT_DIR/scores.md"""
import glob, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import select_track3_submission_candidates as s

def follow(P, ref, verts):
    a0 = np.concatenate([p[0] for p in s.pts(P[:1], verts)]); r0 = np.concatenate([p[0] for p in s.pts(ref[:1], verts)])
    P = np.einsum('ij,fbjk->fbik', s.kabsch(a0, r0), P)
    err = np.linalg.norm(P[..., :3, 3]-ref[..., :3, 3], axis=-1); moving = np.linalg.norm(ref[..., :3, 3]-ref[:1, :, :3, 3], axis=-1) > .05
    return [round(float((err[moving[:, b], b] < .05).mean()), 2) if moving[:, b].any() else None for b in range(P.shape[1])]

def main(out, trees):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); res = {}
    for ep in s.EPS:
        names, ref, verts = s.reference(ep); F = len(ref); r = res[ep] = dict(objects=names, static=s.score(np.repeat(ref[:1], F, 0), ref, verts)['mppe_cm'], runs={})
        for t in trees:
            for p in sorted(glob.glob(str(s.T/t/f'episode_{ep:06d}/learning/evaluations/step_*/episode_{ep:06d}.parquet'))):
                P, valid = s.load_rollout(p, names)
                if len(P) != F or not valid or not np.isfinite(P).all(): continue
                r['runs'][f"{t}@{Path(p).parent.name.split('_')[1]}"] = dict(path=p, follow=follow(P, ref, verts), **s.score(P, ref, verts))
        print(ep, r['static'], min(((k, v['mppe_cm']) for k, v in r['runs'].items()), key=lambda x: x[1], default=None), flush=True)
    md = ['| ep | 物体 | 静止 | ' + ' | '.join(f'{t} 最佳 (步) · SP · 跟随' for t in trees) + ' |', '|---|---|---|' + '---|'*len(trees)]
    for ep, r in res.items():
        cells = []
        for t in trees:
            c = {k: v for k, v in r['runs'].items() if k.startswith(t+'@')}
            if not c: cells.append('–'); continue
            k = min(c, key=lambda k: (not c[k]['sp'], c[k]['mppe_cm'])); v = c[k]
            cells.append(f"{v['mppe_cm']} ({int(k.split('@')[1])/1e6:.0f}M) · {'✓' if v['sp'] else '✗'} · {'/'.join('–' if f is None else str(f) for f in v['follow'])}")
        md.append(f"| {ep} | {'+'.join(r['objects'])} | {r['static']} | " + ' | '.join(cells) + ' |')
    (out/'scores.json').write_text(json.dumps(res, indent=1)); (out/'scores.md').write_text('\n'.join(md)+'\n'); print('\n'.join(md))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
