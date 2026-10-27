"""Tables for notes/track3-hand-contact-v1.md: geometry before/after (human + robot-proxy hands), robot kinematic contact,
zero-residual replay vs rt20 (= learning_fm_rt recipe) and gc2 g200_x1. Writes outputs/track3/hand_contact_v1_fm/summary.{md,json}."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import T3, WORK, EPS, ep_name, save_json  # noqa

FM = T3/'hand_contact_v1_fm'; GC2 = json.loads((T3/'grasp_construct_v2/summary.json').read_text())


def jl(p):
    return json.loads(p.read_text()) if p.exists() else None


def fmt_obj(o):
    if o is None: return '–'
    f = '–' if o['follow_frac'] is None else o['follow_frac']
    return f"{o['moved_cm']}/{o['ref_moved_cm']} · {f} · {o['pos_err_cm']}/{o['rot_err_deg']}{' ✈' if o['flung'] else ''}"


def agg(rows):
    """rows: list of replay dicts (one per episode) -> need-to-move object stats."""
    fol = []; err = []; flung = 0; n = 0; allpass = 0
    for r in rows:
        ok = True; had = False
        for o in r['objects'].values():
            flung += o['flung']
            if o['follow_frac'] is None: continue
            had = True; n += 1; fol.append(o['follow_frac']); err.append(o['pos_err_cm'])
            ok &= o['follow_frac'] >= .5 and not o['flung']
        allpass += had and ok and not any(o['flung'] for o in r['objects'].values())
    fol = np.array(fol)
    return dict(n_obj=n, follow_ge_05=int((fol >= .5).sum()), mean_follow=round(float(fol.mean()), 3) if n else None,
                pos_err_median_cm=round(float(np.median(err)), 1) if n else None, flung=int(flung), episodes_all_pass=int(allpass), episodes=len(rows))


def main():
    md = ['# hand_contact_v1 汇总（自动生成：scripts/hand_contact_v1/summarize.py）', '']
    out = dict(geometry={}, kin={}, replay={})
    # ---------------- geometry
    md += ['## 1. 人手几何（运动帧；src = learning_v5 源手，hc = hands_contact_v1 人手级，hcr = hands_contact_v1r 机器人代理级）', '',
           '接触 = 两手 5 指尖+掌心皮肤到表面最小距离 cm p50（p90）；两点 = 被分配手的拇指与另一指尖都贴上的距离 p50；穿透 = 运动帧 max(r−sd) mm p50/p95/max（容器空腔按实心）；'
           '2D = 与源 2D / MediaPipe(独立) 像素误差 p50；深度 = |z − 双目| cm p50；地面 = 胶囊点低于桌面 >3 mm 的帧数（两手合计）；加速度 mm/帧² p50。', '',
           '| ep | 物体 | 接触 src→hc→hcr | |接触| src→hc→hcr | 合格帧比例 src→hc→hcr | 两点 src→hc→hcr | 穿透 src→hc→hcr | 2D 源/MP src→hc→hcr | 深度 src→hc | 地面 src→hc→hcr | 加速度 src→hc |', '|---|---|---|---|---|---|---|---|---|---|---|']
    acc = {k: dict(contact=[], two=[], pen95=[], penmax=[], ok=[], pen50=[]) for k in ('src', 'hc', 'hcr')}
    for ep in EPS:
        E = ep_name(ep); g = dict(src=jl(WORK/'source_geometry'/f'{E}.json'), hc=jl(T3/'hands_contact_v1'/E/'geometry_report.json'),
                                  hcr=jl(T3/'hands_contact_v1r'/E/'geometry_report.json'))
        if g['src'] is None or g['hc'] is None: continue
        out['geometry'][ep] = g
        for n in g['src']['objects']:
            cell = lambda key, i: '→'.join(str(g[k]['objects'][n][key][i]) if g[k] and g[k]['objects'][n][key] else '–' for k in ('src', 'hc', 'hcr'))  # noqa
            for k in acc:
                if g[k] and g[k]['objects'][n]['contact_cm']:
                    o = g[k]['objects'][n]; acc[k]['contact'].append(o['contact_cm'][1]); acc[k]['two'].append(o['assigned_two_point_cm'][1] if o['assigned_two_point_cm'] else np.nan)
                    acc[k]['pen95'].append(o['pen_mm'][1]); acc[k]['penmax'].append(o['pen_mm'][2]); acc[k]['pen50'].append(o['pen_mm'][0]); acc[k]['ok'].append(o.get('ok_frac', np.nan))
            okc = '→'.join(str(g[k]['objects'][n].get('ok_frac', '–')) if g[k] else '–' for k in ('src', 'hc', 'hcr'))
            md.append(f"| {ep} | {n} ({g['src']['objects'][n]['moving_frames']}f) | {cell('contact_cm', 1)} | {cell('contact_abs_cm', 1) if 'contact_abs_cm' in g['src']['objects'][n] else '–'} | {okc} | {cell('assigned_two_point_cm', 1)} | "
                      f"{cell('pen_mm', 0)} / {cell('pen_mm', 1)} | "
                      f"{g['src']['px_vs_mediapipe'][0]}→{g['hc']['px_vs_source'][0]}/{g['hc']['px_vs_mediapipe'][0]}→{(g['hcr'] or {}).get('px_vs_source', ['–'])[0]}/{(g['hcr'] or {}).get('px_vs_mediapipe', ['–'])[0]} | "
                      f"{g['src']['depth_resid_cm_abs'][0]}→{g['hc']['depth_resid_cm_abs'][0]} | {g['src']['ground_frames_below_3mm']}→{g['hc']['ground_frames_below_3mm']}→{(g['hcr'] or {}).get('ground_frames_below_3mm', '–')} | "
                      f"{g['src']['accel_mm'][0]}→{g['hc']['accel_mm'][0]} |")
    md += ['', '汇总（所有运动物体，逐物体中位数的中位数）：']
    for k, v in acc.items():
        if v['contact']:
            md.append(f"- {k}: 接触 p50 中位 {np.nanmedian(v['contact']):.2f} cm（<1 cm 的物体 {int((np.array(v['contact']) < 1).sum())}/{len(v['contact'])}），"
                      f"两点 {np.nanmedian(v['two']):.2f} cm，穿透 p50 ≤3 mm 的物体 {int((np.array(v['pen50']) <= 3).sum())}/{len(v['pen50'])}，合格帧比例中位 {np.nanmedian(v['ok']):.2f}，穿透 p95 中位 {np.nanmedian(v['pen95']):.1f} mm（≤3 mm 的物体 {int((np.array(v['pen95']) <= 3).sum())}/{len(v['pen95'])}）")
    # ---------------- robot kinematic
    md += ['', '## 2. 机器人级运动学检查（参考关节目标 + 参考物体位姿，CPU MuJoCo 碰撞，RL 场景模型）', '',
           '持握距离 = 运动帧里任一手连杆到该物体的最小有符号距离 mm p50（<0 = 穿透，上限 30）；穿透 p95；手-地穿透 p95 mm。', '',
           '| ep | 物体 | 持握距离 rt20→hc→hcr | 物体穿透 p95 rt20→hc→hcr | 手-地 p95 rt20→hc→hcr |', '|---|---|---|---|---|']
    for ep in EPS:
        E = ep_name(ep); k = {s: jl(FM/E/f'{s}_kin.json') for s in ('rt20', 'hc', 'hcr')}
        if k['rt20'] is None: continue
        out['kin'][ep] = k
        for n in k['rt20']['objects']:
            c = lambda key, i: '→'.join(str(k[s]['objects'][n][key][i]) if k[s] else '–' for s in ('rt20', 'hc', 'hcr'))  # noqa
            md.append(f"| {ep} | {n} | {c('hold_dist_mm_p10_50_90', 1)} | {c('pen_mm_p50_p95_max', 1)} | "
                      + '→'.join(str(k[s]['hand_ground_pen_mm_p95_max'][0]) if k[s] else '–' for s in ('rt20', 'hc', 'hcr')) + ' |')
    # ---------------- replay
    md += ['', '## 3. 零残差回放（VOC=0、摩擦 1.0、1 world、explicit reset(0)；feasibility_matrix_zero_residual.run 原样）', '',
           '物体格式：实际/参考位移 cm · 跟随比例 · 平均位置误差 cm/姿态误差 °（✈ = 打飞）。gc2 = grasp_construct_v2 g200_x1（物体坐标系构造抓取 + ILC）。', '']
    srcs = [('rt20', 'g25'), ('hc', 'g25'), ('hcr', 'g25'), ('rt20s04ff', 'g25'), ('hcs04ff', 'g25'), ('hcrs04ff', 'g25'), ('rt20', 'g200'), ('hc', 'g200'), ('hcr', 'g200')]
    rows = {s: [] for s in srcs+[('gc2', 'g200')]}
    for ep in EPS:
        E = ep_name(ep); rs = {s: jl(FM/E/f'{s[0]}_{s[1]}.json') for s in srcs}
        if rs[('rt20', 'g25')] is None: continue
        gc2 = next((x for x in GC2.get(str(ep), []) if x['source'] == 'gc2 g200_x1'), None)
        rs[('gc2', 'g200')] = gc2
        names = list(rs[('rt20', 'g25')]['objects'])
        md += [f'### ep{ep}', '', '| 来源 | 增益 | '+' | '.join(names)+' | 腕滞后 cm | 手-物穿透 max mm | 手-地 max mm |', '|---|---|'+'---|'*len(names)+'---|---|---|']
        for s, r in rs.items():
            if r is None: continue
            md.append(f"| {s[0]} | {s[1]} | "+' | '.join(fmt_obj(r['objects'].get(n)) for n in names)+f" | {r.get('wrist_lag_cm_mean', '–')} | {r.get('hand_obj_pen_max_mm', '–')} | {r.get('hand_ground_pen_max_mm', '–')} |")
        md.append('')
        if all(rs[s] is not None for s in rs):
            for s in rs: rows[s].append(rs[s])
            out['replay'][ep] = {f'{a}_{b}': v for (a, b), v in rs.items()}
    md += ['## 4. 20 段同口径汇总（只统计所有来源都跑完的段）', '', '| 来源 | 增益 | 段数 | 需移动物体 | 跟随 ≥0.5 | 平均跟随 | 位置误差中位 cm | 打飞 | 整段全过 |', '|---|---|---|---|---|---|---|---|---|']
    for s, rr in rows.items():
        if not rr: continue
        a = agg(rr); out.setdefault('agg', {})[f'{s[0]}_{s[1]}'] = a
        md.append(f"| {s[0]} | {s[1]} | {a['episodes']} | {a['n_obj']} | {a['follow_ge_05']} | {a['mean_follow']} | {a['pos_err_median_cm']} | {a['flung']} | {a['episodes_all_pass']} |")
    # ---------------- robot post-processing variants (squeeze / feed-forward), g25
    md += ['', '## 5. 机器人参考后处理（g25，零残差）：s04 = 持握帧手指屈曲目标 +0.4 rad；ff = 腕 PD 前馈（加速度/速度/被持物重量）；s08 = +0.8 rad', '',
           '| ep | 来源 | '+'变体 | 物体（实际/参考 · 跟随 · 误差 cm/°） |', '|---|---|---|---|']
    post = {}
    for ep in EPS:
        E = ep_name(ep)
        for base in ('rt20', 'hc', 'hcr'):
            for v in ('', 's04', 'ff', 's04ff', 's08ff'):
                r = jl(FM/E/f'{base}{v}_g25.json') if v else jl(FM/E/f'{base}_g25.json')
                if r is None or (v == '' and not (FM/E/f'{base}s04ff_g25.json').exists()): continue
                post.setdefault(f'{base}{v}', []).append(r)
                md.append(f"| {ep} | {base} | {v or '无'} | "+'; '.join(f"{n}: {fmt_obj(o)}" for n, o in r['objects'].items())+' |')
    md += ['', '| 来源+变体 | 段数 | 需移动物体 | 跟随 ≥0.5 | 平均跟随 | 位置误差中位 cm | 打飞 |', '|---|---|---|---|---|---|---|']
    for k, rr in post.items():
        a = agg(rr); out.setdefault('post_agg', {})[k] = a
        md.append(f"| {k} | {a['episodes']} | {a['n_obj']} | {a['follow_ge_05']} | {a['mean_follow']} | {a['pos_err_median_cm']} | {a['flung']} |")
    # ---------------- 16-world spread (same input, contact chaos)
    md += ['', '## 6. 同输入 16 world 的跟随分布（g25；base = 无后处理，s04ff = 屈曲 +0.4 + 前馈）', '',
           '| ep | 来源 | 物体：跟随 p10/50/90 · 位置误差中位 cm · 打飞比例 |', '|---|---|---|']
    for ep in EPS:
        E = ep_name(ep)
        for k in ('rt20base', 'hcbase', 'hcrbase', 'rt20s04ff', 'hcs04ff', 'hcrs04ff'):
            r = jl(FM/E/f'{k}_g25_w16.json')
            if r is None or 'spread' not in r: continue
            out.setdefault('w16', {}).setdefault(ep, {})[k] = r['spread']
            md.append(f"| {ep} | {k} | "+'; '.join(f"{n}: {v['follow_p10_50_90']} · {v['pos_err_cm_p50']} · {v['flung_frac']:.2f}" for n, v in r['spread'].items())+' |')
    (FM/'summary.md').write_text('\n'.join(md)+'\n'); save_json(FM/'summary.json', out); print('\n'.join(md[-12:]))


if __name__ == '__main__':
    main()
