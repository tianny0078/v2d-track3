"""Track3 v5 candidate selection with the NEW meshes (procedural_geometry_v8 / multiep_recon_v1), same proxy as #3/#4.

Reference = learning_v8_geom object trajectories + v8 4096-face meshes (self-reference proxy, NOT GT): frame-0 rigid scene
alignment (Kabsch over all objects' mesh points) then per-point error (MPPE proxy, cm), per-object mean position /
orientation error (SP proxy: <= 10 cm and <= 0.5 rad), follow fraction. Functions reused unchanged from
select_track3_submission_candidates.py / score_track3_trees.py; only the reference tree differs.

Candidates (eligible = rolled out in the v8 scene, so trajectory and submitted mesh are consistent):
  reeval:<tree>@<step>  outputs/track3/learning_v8_reeval/<tree>/episode_X/step_N/episode_X.parquet (old checkpoints, v8 scene)
  lv8r@<step>           outputs/track3/learning_v8r/episode_X/learning/evaluations/step_N/episode_X.parquet
Shown for comparison only: v4 = the submitted #4 rollout (v7 scene, old meshes) scored against the v8 reference, and its
#4-time proxy (v7 reference). Episodes whose v8 scene is not ready (19/20/30 until retrack_v8) fall back to the v4 rollout
WITH its v4 meshes (old geometry, still mutually consistent) and are flagged.
Writes outputs/track3/submission_v5_candidate/{selection.json, selection.md, assembly.json}.
"""
import glob, json, sys
from pathlib import Path
import numpy as np, trimesh
sys.path.insert(0, str(Path(__file__).parent))
import select_track3_submission_candidates as s
from score_track3_trees import follow

T = s.T; OUT = T/'submission_v5_candidate'; REF = 'learning_v8_geom'; V4 = T/'submission_v4'


def reference(ep):
    m = json.loads((T/REF/f'episode_{ep:06d}/objects/objects.json').read_text()); z = np.load(m['poses_path'])
    A = np.asarray(m['table_T_world']); P = np.einsum('ij,fbjk->fbik', A, z['world_T_canonical'])
    names = [str(n) for n in z['object_names']]; items = {o['name']: o for o in m['objects']}
    verts = []; mesh4096 = {}
    for n in names:
        p = items[n].get('mesh_path_4096') or items[n]['mesh_path']; mesh4096[n] = p
        v = np.asarray(trimesh.load(p, force='mesh').vertices); verts.append(v[::max(1, len(v)//800)])
    return names, P, verts, mesh4096


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    v4sel = json.loads((V4/'selection.json').read_text()); v4asm = json.loads((V4/'assembly.json').read_text())
    sel = {}; asm = {}
    md = ['# Track3 v5 候选（新网格 v8 场景；参考 = learning_v8_geom 自参考代理，非 GT）', '',
          '代理 = 第 0 帧整体刚性对齐 + v8 4096 网格点误差（cm），与 #3/#4 同一函数，只换参考树。可入选 = 在 v8 场景中回放（轨迹与提交网格一致）。', '',
          '| ep | 物体 | 静止(v8) | v4 回放 @v8参考 | v4 @v7参考(#4时) | 最佳旧ckpt重评(v8) | 最佳 lv8r | 入选 | 入选 MPPE | SP | 跟随 |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for ep in s.EPS:
        E = f'episode_{ep:06d}'
        ready = (T/REF/E/'objects/poses.npz').exists()
        v4p = Path(v4asm[E]['rollout']); v4name = v4asm[E]['candidate']
        if not ready:
            sel[ep] = dict(ready=False, chosen=f'v4:{v4name}', note='v8 scene not ready (waiting for retrack_v8): v4 rollout + v4 meshes (old geometry)')
            asm[E] = dict(candidate=f'v4:{v4name}', rollout=str(v4p), meshes={n: dict(src=str(V4/'meshes'/E/f'{n}.obj'), geometry='v4 (old)') for n in v4asm[E]['meshes']},
                          consistent_with_rollout=True, new_geometry=False)
            md.append(f'| {ep} | – | – | – | {v4sel[str(ep)]["best_score"]["mppe_cm"]} | – | – | v4 回退（等待重跟踪） | – | – | – |'); continue
        names, ref, verts, mesh4096 = reference(ep); F = len(ref)
        static = s.score(np.repeat(ref[:1], F, 0), ref, verts)['mppe_cm']
        res = {}
        def add(name, p, eligible):
            try: P, valid = s.load_rollout(p, names)
            except Exception as e: res[name] = dict(error=str(e)[:200]); return  # noqa: BLE001
            if len(P) != F or not valid or not np.isfinite(P).all(): res[name] = dict(error=f'frames {len(P)} vs {F}, valid {valid}'); return
            res[name] = dict(path=str(p), eligible=eligible, follow=follow(P, ref, verts), **s.score(P, ref, verts))
        add(f'v4:{v4name}', v4p, False)
        for p in sorted(glob.glob(str(T/f'learning_v8_reeval/*/{E}/step_*/{E}.parquet'))):
            add(f"reeval:{Path(p).parents[2].name}@{Path(p).parent.name.split('_')[1]}", p, True)
        for p in sorted(glob.glob(str(T/f'learning_v8r/{E}/learning/evaluations/step_*/{E}.parquet'))):
            add(f"lv8r@{Path(p).parent.name.split('_')[1]}", p, True)
        ok = {k: v for k, v in res.items() if v.get('eligible')}
        best = lambda pre: min((k for k in ok if k.startswith(pre)), key=lambda k: ok[k]['mppe_cm'], default=None)
        br, b8 = best('reeval:'), best('lv8r@')
        chosen = min((k for k in (br, b8) if k), key=lambda k: ok[k]['mppe_cm'], default=None)
        v4k = f'v4:{v4name}'; v4s = res.get(v4k, {})
        sel[ep] = dict(ready=True, objects=names, static=static, runs=res, best_reeval=br, best_lv8r=b8, chosen=chosen,
                       v4=dict(name=v4name, mppe_v8ref=v4s.get('mppe_cm'), mppe_v7ref_at_submission=v4sel[str(ep)]['best_score']['mppe_cm']))
        if chosen:
            asm[E] = dict(candidate=chosen, rollout=ok[chosen]['path'], meshes={n: dict(src=mesh4096[n], geometry='v8') for n in names},
                          consistent_with_rollout=True, new_geometry=True)
        f = lambda k: '–' if not k else f"{ok[k]['mppe_cm']} ({k.split(':')[-1]})"
        c = ok.get(chosen, {})
        md.append(f"| {ep} | {'+'.join(names)} | {static} | {v4s.get('mppe_cm', '–')} | {v4sel[str(ep)]['best_score']['mppe_cm']} | {f(br)} | {f(b8)} | "
                  f"{chosen or '–'} | {c.get('mppe_cm', '–')} | {'✓' if c.get('sp') else '✗'} | {'/'.join('–' if x is None else str(x) for x in c.get('follow', []))} |")
    rdy = [e for e in sel if sel[e]['ready'] and sel[e].get('chosen')]
    if rdy:
        m = lambda xs: round(float(np.mean(xs)), 2)
        md += ['', f'就绪 {len(rdy)}/20 段均值：静止 {m([sel[e]["static"] for e in rdy])}，v4 回放@v8参考 {m([sel[e]["v4"]["mppe_v8ref"] for e in rdy if sel[e]["v4"]["mppe_v8ref"] is not None])}，'
               f'入选 {m([sel[e]["runs"][sel[e]["chosen"]]["mppe_cm"] for e in rdy])}；入选代理 SP {sum(sel[e]["runs"][sel[e]["chosen"]]["sp"] for e in rdy)}/{len(rdy)}',
               '注：v4 回放是在 v7 场景（旧网格）里滚出来的，用 v8 参考打分只作对照，不可入选（网格与轨迹不一致）。']
    (OUT/'selection.json').write_text(json.dumps(sel, indent=1, default=str)); (OUT/'assembly.json').write_text(json.dumps(asm, indent=1))
    (OUT/'selection.md').write_text('\n'.join(md) + '\n'); print('\n'.join(md))


if __name__ == '__main__':
    main()
