"""Same check as scripts/diag_track3_initial_penetration.py, but reading the object reference from objects/poses.npz
(table_T_world @ world_T_canonical) instead of learning/reference/reference_checkpoint.npz (a manifest-only tree has no
learning dir). Ground: table plane z=0, frame 0 and whole clip; object pairs: frame 0. Geometry = URDF collision parts.
Usage: .deps/flash-chord-venv/bin/python scripts/procedural_geometry/diag_penetration_v7.py <tree> [eps...]
Writes outputs/track3/<tree>/penetration_report.json (merged per episode) or --out <path> (use it for read-only trees such as learning_v6).
"""
import json, sys
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R/'scripts'))
_argv = sys.argv; sys.argv = sys.argv[:1]
from diag_track3_initial_penetration import parts, xf, depth_into  # noqa: E402  (module reads argv at import)
sys.argv = _argv
EPS = [1, 3, 5, 6, 8, 10, 14, 15, 16, 19, 20, 25, 28, 30, 32, 33, 35, 36, 37, 38]


def run(tree, ep):
    E = R/'outputs/track3'/tree/f'episode_{ep:06d}/objects'; man = json.loads((E/'objects.json').read_text())
    z = np.load(man['poses_path']); A = np.asarray(man['table_T_world'], float)
    names = [str(x) for x in z['object_names']]; P = np.einsum('ij,fbjk->fbik', A, z['world_T_canonical'])
    items = {o['name']: o for o in man['objects']}; geo = {n: parts(items[n]['urdf_path']) for n in names}; r = {}
    for b, n in enumerate(names):
        minz = np.array([min(xf(P[f, b], v)[:, 2].min() for v, _, _ in geo[n]) for f in range(len(P))])
        r[n] = dict(f0_min_z_mm=round(float(minz[0])*1000, 1), f0_ground_pen_mm=round(max(0., -float(minz[0]))*1000, 1),
                    frames_below_5mm=int((minz < -.005).sum()), frames=len(P), worst_ground_pen_mm=round(max(0., -float(minz.min()))*1000, 1),
                    mass_kg=items[n].get('mass_kg'), collision_parts=len(geo[n]))
    pairs = {}
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a = np.concatenate([xf(P[0, i], v) for v, _, _ in geo[names[i]]]); c = np.concatenate([xf(P[0, j], v) for v, _, _ in geo[names[j]]])
            d = max(depth_into(a, geo[names[j]], P[0, j]), depth_into(c, geo[names[i]], P[0, i]))
            pairs[f'{names[i]}|{names[j]}'] = dict(f0_pen_mm=round(d*1000, 1))
    return dict(objects=r, pairs=pairs)


if __name__ == '__main__':
    a = sys.argv[1:]; out = None
    if '--out' in a: i = a.index('--out'); out = Path(a[i+1]); del a[i:i+2]
    tree = a[0]; eps = list(map(int, a[1:])) or EPS
    if out is None: out = R/'outputs/track3'/tree/'penetration_report.json'
    if 'learning_v6' in str(out.parent) and 'learning_v6' == out.parent.name: raise SystemExit('refusing to write into learning_v6; pass --out')
    rep = json.loads(out.read_text()) if out.exists() else {}
    for e in eps:
        rep[str(e)] = run(tree, e); print(e, json.dumps(rep[str(e)]), flush=True)
    out.write_text(json.dumps(rep, indent=1))
