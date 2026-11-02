"""Unified candidate selection for Track3 submission #3 (review §2.2).

Every evaluated policy rollout (official evaluator parquet, VOC=0) of every tree is scored against ONE reference: the
learning_v7_geom object trajectories (our estimate; no GT, no public scan). Scoring mimics the host protocol without
any scan: one rigid scene alignment at the first frame (Kabsch on all objects' frame-0 mesh points, achieved -> reference),
then per-point error on our v7 4096-face mesh vertices (MPPE proxy, cm) and per-object mean position / orientation error
(SP proxy: <= 10 cm and <= 0.5 rad for every object). Symmetry is ignored (conservative for cups/pot).
Also scores the 'static' trajectory (reference frame 0 held) as the do-nothing yardstick; it is NOT a candidate
(submissions must be real policy rollouts).
Writes outputs/track3/submission_v3/selection.json and selection.md.
"""
import glob, json
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, trimesh
from scipy.spatial.transform import Rotation

R = Path('/home/tianyuan/projects/v2d'); T = R/'outputs/track3'; OUT = T/'submission_v3'
EPS = [1, 3, 5, 6, 8, 10, 14, 15, 16, 19, 20, 25, 28, 30, 32, 33, 35, 36, 37, 38]
TREES = ['learning_v3', 'learning_v4', 'learning_v5', 'learning_v6', 'learning_v6i', 'learning_v7r', 'learning_v7t']
SUB2 = T/'scored_baseline_20261017/canonical-checkpoint-v2/results'

def reference(ep):
    m = json.loads((T/f'learning_v7_geom/episode_{ep:06d}/objects/objects.json').read_text()); z = np.load(m['poses_path'])
    A = np.asarray(m['table_T_world']); P = np.einsum('ij,fbjk->fbik', A, z['world_T_canonical'])
    names = [str(n) for n in z['object_names']]; items = {o['name']: o for o in m['objects']}
    verts = []
    for n in names:
        v = np.asarray(trimesh.load(items[n].get('mesh_path_4096') or items[n]['mesh_path'], force='mesh').vertices)
        verts.append(v[::max(1, len(v)//800)])
    return names, P, verts

def load_rollout(path, names):
    t = pq.read_table(path).to_pandas(); t = t[t.environment_id == 0].sort_values(['step', 'object_body_id'])
    order = [str(x) for x in t[t.step == t.step.min()].object_body_name]
    S = t.step.nunique(); B = len(order)
    pos = t[['achieved_position_x', 'achieved_position_y', 'achieved_position_z']].to_numpy().reshape(S, B, 3)
    q = t[['achieved_quaternion_x', 'achieved_quaternion_y', 'achieved_quaternion_z', 'achieved_quaternion_w']].to_numpy().reshape(S, B, 4)
    idx = [order.index(n) for n in names]
    P = np.tile(np.eye(4), (S, len(names), 1, 1)); P[..., :3, 3] = pos[:, idx]
    P[..., :3, :3] = Rotation.from_quat(q[:, idx].reshape(-1, 4)).as_matrix().reshape(S, len(names), 3, 3)
    return P, bool(t.environment_valid.all())

def kabsch(X, Y):
    mx, my = X.mean(0), Y.mean(0); U, _, Vt = np.linalg.svd((X-mx).T @ (Y-my)); d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1, 1, d]) @ U.T; M = np.eye(4); M[:3, :3] = Rm; M[:3, 3] = my - Rm @ mx; return M

def pts(P, verts):  # (F,B,4,4) -> list of (F,N,3)
    return [np.einsum('fij,nj->fni', P[:, b, :3, :3], v) + P[:, b, None, :3, 3] for b, v in enumerate(verts)]

def score(P, ref, verts):
    F = min(len(P), len(ref)); P, ref = P[:F], ref[:F]
    a0 = np.concatenate([p[0] for p in pts(P[:1], verts)]); r0 = np.concatenate([p[0] for p in pts(ref[:1], verts)])
    P = np.einsum('ij,fbjk->fbik', kabsch(a0, r0), P)
    pa, pr = pts(P, verts), pts(ref, verts)
    mppe = float(np.mean([np.linalg.norm(a-r, axis=-1).mean() for a, r in zip(pa, pr)]))*100
    pe = np.linalg.norm(P[..., :3, 3]-ref[..., :3, 3], axis=-1).mean(0)*100
    oe = np.degrees(Rotation.from_matrix((np.swapaxes(P[..., :3, :3], -1, -2) @ ref[..., :3, :3]).reshape(-1, 3, 3)).magnitude().reshape(F, -1).mean(0))
    moved = np.linalg.norm(P[..., :3, 3]-P[:1, :, :3, 3], axis=-1).max(0)*100
    return dict(mppe_cm=round(mppe, 2), pos_err_cm=np.round(pe, 1).tolist(), ori_err_deg=np.round(oe, 1).tolist(), moved_cm=np.round(moved, 1).tolist(),
                sp=bool((pe <= 10).all() and (np.radians(oe) <= .5).all()), frames=F)

def main():
    OUT.mkdir(parents=True, exist_ok=True); sel = {}; md = ['# Track3 #3 候选统一打分（参考 = learning_v7_geom 物体轨迹，自参考代理指标，非 GT）', '',
        '| ep | 物体 | 静止 MPPE | #2 MPPE | 最佳候选 | 最佳 MPPE | 最佳 SP | 实际/参考位移 cm | 候选数 |', '|---|---|---|---|---|---|---|---|---|']
    for ep in EPS:
        names, ref, verts = reference(ep); F = len(ref)
        refmove = (np.linalg.norm(ref[..., :3, 3]-ref[:1, :, :3, 3], axis=-1).max(0)*100).round(1).tolist()
        cands = [(f'sub2', SUB2/f'episode_{ep:06d}.parquet')]
        for t in TREES:
            for p in sorted(glob.glob(str(T/t/f'episode_{ep:06d}/learning/evaluations/step_*/episode_{ep:06d}.parquet'))):
                cands.append((f"{t}@{Path(p).parent.name.split('_')[1]}", Path(p)))
        res = {}
        for name, p in cands:
            try:
                P, valid = load_rollout(p, names)
            except Exception as e:  # noqa: BLE001
                res[name] = dict(error=str(e)); continue
            if len(P) != F or not valid or not np.isfinite(P).all():
                res[name] = dict(error=f'frames {len(P)} vs {F}, valid {valid}'); continue
            res[name] = dict(path=str(p), **score(P, ref, verts))
        static = score(np.repeat(ref[:1], F, 0), ref, verts)
        ok = {k: v for k, v in res.items() if 'error' not in v}
        # The official packer accepts one evaluator build per submission; #2 rollouts were made by an older
        # evaluator script (eval_script_sha256 7d026983...), so they are scored for comparison but not selectable.
        best = min((k for k in ok if k != 'sub2'), key=lambda k: (not ok[k]['sp'], ok[k]['mppe_cm']))
        sel[ep] = dict(objects=names, ref_moved_cm=refmove, static=static, best=best, best_score=ok[best], candidates=res)
        b = ok[best]
        md.append(f"| {ep} | {'+'.join(names)} | {static['mppe_cm']} | {ok.get('sub2', {}).get('mppe_cm', '–')} | {best} | {b['mppe_cm']} | {'✓' if b['sp'] else '✗'} | "
                  f"{'/'.join(map(str, b['moved_cm']))} vs {'/'.join(map(str, refmove))} | {len(ok)} |")
        print(md[-1], flush=True)
    m = lambda k: np.mean([sel[e][k]['mppe_cm'] if k != 'sub2' else sel[e]['candidates']['sub2']['mppe_cm'] for e in EPS])
    md += ['', f"均值 MPPE 代理：静止 {m('static'):.2f}，#2 {m('sub2'):.2f}，#3 选择 {m('best_score'):.2f}；"
           f"SP 代理：#2 {sum(sel[e]['candidates']['sub2'].get('sp', False) for e in EPS)}/20，#3 {sum(sel[e]['best_score']['sp'] for e in EPS)}/20"]
    (OUT/'selection.json').write_text(json.dumps(sel, indent=1)); (OUT/'selection.md').write_text('\n'.join(md)+'\n'); print(md[-1])

if __name__ == '__main__':
    main()
