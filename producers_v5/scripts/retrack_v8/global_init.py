"""retrack_v8 stage 1: global pose + scale initialisation from mask-only candidate frames (scored split only).

Per episode:
  candidates : 10-20 frames chosen from MASK-ONLY criteria (never from the v7 poses): mask area >= 0.4 x episode median,
               bbox >= 6 px from the image border, area jump vs +-2 neighbours <= 15 %, sharpness >= 0.5 x median, ranked by
               hand-landmark-hull pixels inside a 15 px ring around the mask and other-object-mask pixels in that ring (fewest
               first), greedy temporal spread >= 6 frames. Relaxation tiers are recorded per frame.
  search     : multi-hypothesis render-and-compare: N_rot random SO(3) samples; for each, scale s and camera-frame
               translation solved analytically from (mask area, mask centroid, median FoundationStereo depth in the eroded
               mask) given a reference render; s clamped to [s_lo, s_hi] x procedural; scored by hard IoU - 4 x depth MAE;
               top-K rotation-NMS'd hypotheses refined (Adam: soft IoU + depth Huber + back-projected points to surface).
  consistency: frames in the same untouched (hand-free) stretch must agree in WORLD pose (object static); the hypothesis
               assignment maximising score + agreement is chosen; frames whose 2nd distinct hypothesis (>25 deg away) scores
               within 0.03 are flagged ambiguous.
  scale      : one shared scale per object (all given episodes) = median over unambiguous candidates with IoU >= 0.7;
               then every candidate is re-searched with the scale fixed.
Writes outputs/track3/retrack_v8/episode_<ep>/<obj>_global_init.json (+ .npz with camT hypotheses).
Usage: python3 scripts/retrack_v8/global_init.py --obj mini_sweeper --episodes 19,30 [--gpu 0]
"""
import argparse, json, os, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--obj', required=True); ap.add_argument('--episodes', required=True); ap.add_argument('--gpu', default='0')
    ap.add_argument('--mesh', default=None); ap.add_argument('--n-rot', type=int, default=4096); ap.add_argument('--topk', type=int, default=24)
    ap.add_argument('--n-cand', type=int, default=16); ap.add_argument('--s-lo', type=float, default=0.6); ap.add_argument('--s-hi', type=float, default=1.5)
    ap.add_argument('--iters', type=int, default=120); ap.add_argument('--res', type=int, default=256)
    a = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import numpy as np, torch
    import common as C
    eps = [int(x) for x in a.episodes.split(',')]
    a.mesh = a.mesh or str(C.T3/f'procedural_geometry_v1/{a.obj}/visual_4096.obj')
    (C.ROOT/'logs/retrack_v8').mkdir(parents=True, exist_ok=True)
    C.LOGF = open(C.ROOT/f'logs/retrack_v8/global_init_{a.obj}.log', 'a')
    C.say('args', json.dumps(vars(a)))
    cx = C.Cx(); mesh = C.Mesh(cx, a.mesh)
    Rall = C.random_rotations(a.n_rot, seed=0)

    def candidates(e):
        n = e.n; A = e.area.astype(float); med = np.median(A[A > 0])
        rows = []
        for f in range(n):
            r = dict(frame=f, area=int(A[f]), hand_near=int(e.hand_near[f]), other_near=int(e.other_near[f]), border=bool(e.border[f]))
            nb = np.r_[A[max(0, f-2):f], A[f+1:f+3]]; nb = nb[nb > 0]
            r['jump'] = float(abs(A[f]/np.median(nb) - 1)) if len(nb) and A[f] > 0 else 1.0
            r['sharp_rel'] = float(e.sharp[f]/np.nanmedian(e.sharp)) if np.isfinite(e.sharp[f]) else 0.0
            hard = []
            if A[f] < max(1500, 0.4*med): hard.append('small')
            if e.border[f]: hard.append('border')
            if r['jump'] > 0.15: hard.append('area_jump')
            if r['sharp_rel'] < 0.5: hard.append('blur')
            if e.dmed[f] <= 0: hard.append('no_depth')
            r['reject'] = hard
            # soft ranking: tier 0 = no hand / other-object pixels in the ring, 1 = <= 3 % of mask area, 2 = <= 10 %, 3 = more
            q = max(r['hand_near'], 0) + max(r['other_near'], 0); fr = q/max(A[f], 1)
            r['ring_frac'] = float(fr); r['tier'] = 0 if q == 0 else (1 if fr <= 0.03 else (2 if fr <= 0.10 else 3))
            rows.append(r)
        ok = [r for r in rows if not r['reject']]
        sel = []
        for tier in range(4):
            pool = sorted([r for r in ok if r['tier'] == tier], key=lambda r: (r['ring_frac'], -r['sharp_rel']))
            for r in pool:
                if len(sel) >= a.n_cand: break
                if all(abs(r['frame'] - s['frame']) >= 6 for s in sel): sel.append(r)
            if len(sel) >= max(10, a.n_cand*3//4): break
        return rows, sorted(sel, key=lambda r: r['frame'])

    def search(e, f, s_fixed=None, extra_R=None, n_keep=6):
        Rs = Rall if extra_R is None else np.concatenate([Rall, extra_R])
        return C.global_search(cx, mesh, e, f, Rs, s_fixed=s_fixed, s_lo=a.s_lo, s_hi=a.s_hi, topk=a.topk, iters=a.iters, res=a.res, n_keep=n_keep)

    E = {ep: C.Episode(cx, ep, a.obj) for ep in eps}
    res = {}
    # ---------------- pass 1: free scale
    for ep, e in E.items():
        rows, sel = candidates(e)
        C.say(f'ep{ep}: candidates {[r["frame"] for r in sel]} tiers {[r["tier"] for r in sel]}')
        cand = []
        for r in sel:
            f = r['frame']; hs = search(e, f)
            v7 = C.eval_full(cx, mesh, e, f, e.camT_v7[f], e.scale_v7)
            r.update(v7_iou=float(v7['iou_known'][0]), v7_iou_recon=float(v7['iou'][0]), v7_iou_plain=float(v7['iou_plain'][0]), v7_dmae_mm=float(v7['dmae'][0]*1000))
            C.say(f'ep{ep} f{f}: v7 IoU {r["v7_iou"]:.3f} -> best {hs[0]["iou"]:.3f} (s {hs[0]["s"]:.3f}, dep {hs[0]["dmae_mm"]:.1f} mm); '
                  f'alts ' + ', '.join(f'{h["iou"]:.3f}/s{h["s"]:.2f}/{C.rot_angle_deg(hs[0]["R"], h["R"]):.0f}deg' for h in hs[1:4]))
            cand.append(dict(row=r, hyps=hs))
        res[ep] = dict(rows=rows, cand=cand)
    # ---------------- shared scale
    ss, ws = [], []
    for ep in eps:
        for c in res[ep]['cand']:
            h = c['hyps']; amb = len(h) > 1 and h[1]['score'] > h[0]['score'] - 0.03 and C.rot_angle_deg(h[0]['R'], h[1]['R']) > 25
            if h[0]['iou'] >= 0.7 and not amb: ss.append(h[0]['s']); ws.append(h[0]['iou'])
    if len(ss) < 3:
        ss = [c['hyps'][0]['s'] for ep in eps for c in res[ep]['cand']]
    s_shared = float(np.median(ss))
    C.say(f'shared scale {s_shared:.4f} from {len(ss)} frames (p10 {np.percentile(ss,10):.3f} p90 {np.percentile(ss,90):.3f}); v7 scale {E[eps[0]].scale_v7:.3f}')
    # ---------------- pass 2: fixed scale, seeded additionally with the pass-1 rotations of all candidates
    seedR = np.stack([h['R'] for ep in eps for c in res[ep]['cand'] for h in c['hyps'][:3]])
    for ep, e in E.items():
        for c in res[ep]['cand']:
            f = c['row']['frame']
            c['hyps_fixed'] = search(e, f, s_fixed=s_shared, extra_R=seedR)
            h = c['hyps_fixed']
            c['ambiguous'] = bool(len(h) > 1 and h[1]['score'] > h[0]['score'] - 0.03 and C.rot_angle_deg(h[0]['R'], h[1]['R']) > 25)
            C.say(f'ep{ep} f{f} fixed s: best IoU {h[0]["iou"]:.3f} dep {h[0]["dmae_mm"]:.1f} mm (signed {h[0]["depth_signed_mm"]:+.1f}); ambiguous {c["ambiguous"]}; alt '
                  + ', '.join(f'{x["iou"]:.3f}/{C.rot_angle_deg(h[0]["R"], x["R"]):.0f}deg' for x in h[1:3]))
    # ---------------- cross-frame consistency in static (hand-free) stretches
    for ep, e in E.items():
        cand = res[ep]['cand']
        touched = e.hand_near > 0
        # stretch id: candidates with no touched frame between them share a stretch (only untouched candidates)
        sid, cur = [], -1; last = None
        for c in cand:
            f = c['row']['frame']
            if touched[f]: sid.append(None); continue
            if last is None or touched[last:f+1].any(): cur += 1
            sid.append(cur); last = f
        for c, s_ in zip(cand, sid): c['static_stretch'] = s_
        for st in sorted({s for s in sid if s is not None}):
            mem = [c for c, s_ in zip(cand, sid) if s_ == st]
            if len(mem) < 2: continue
            # world poses of all hypotheses; choose per-member hypothesis maximising score + agreement with others' choice
            Wh = [[e.wTc[c['row']['frame']] @ h['camT'] for h in c['hyps_fixed']] for c in mem]
            best = None
            for i0, c0 in enumerate(mem):          # try each hypothesis of each member as the reference world pose
                for h0 in Wh[i0]:
                    tot, choice = 0.0, []
                    for i, c in enumerate(mem):
                        sc = [c['hyps_fixed'][k]['score'] - (0.0 if (C.rot_angle_deg(Wk[:3, :3], h0[:3, :3]) < 10 and np.linalg.norm(Wk[:3, 3] - h0[:3, 3]) < 0.02) else 0.15)
                              for k, Wk in enumerate(Wh[i])]
                        k = int(np.argmax(sc)); tot += sc[k]; choice.append(k)
                    if best is None or tot > best[0]: best = (tot, choice)
            for c, k in zip(mem, best[1]):
                if k != 0:
                    C.say(f'ep{ep} f{c["row"]["frame"]}: static-stretch {st} consistency picks hypothesis {k} (IoU {c["hyps_fixed"][k]["iou"]:.3f} vs {c["hyps_fixed"][0]["iou"]:.3f})')
                c['chosen'] = k; c['consistency'] = f'static stretch {st} ({len(mem)} frames)'
            Wc = [Wh[i][k] for i, k in enumerate(best[1])]
            spread = max(C.rot_angle_deg(Wc[0][:3, :3], W_[:3, :3]) for W_ in Wc), max(np.linalg.norm(Wc[0][:3, 3] - W_[:3, 3]) for W_ in Wc)
            for c in mem: c['stretch_spread_deg_m'] = [float(spread[0]), float(spread[1])]
    # ---------------- write
    for ep, e in E.items():
        od = C.OUTR/f'episode_{ep:06d}'; od.mkdir(parents=True, exist_ok=True)
        cand = res[ep]['cand']; js = []
        camT = []; frames = []
        for c in cand:
            k = c.get('chosen', 0); h = c['hyps_fixed'][k]; r = c['row']
            js.append(dict(frame=r['frame'], tier=r['tier'], ring_frac=r['ring_frac'], area=r['area'], sharp_rel=r['sharp_rel'], jump=r['jump'],
                           v7_iou=r['v7_iou'], v7_iou_plain=r['v7_iou_plain'], v7_dmae_mm=r['v7_dmae_mm'],
                           free_scale_best=dict(s=c['hyps'][0]['s'], iou=c['hyps'][0]['iou'], dmae_mm=c['hyps'][0]['dmae_mm']),
                           init_iou=h['iou'], init_iou_recon=h['iou_recon'], init_iou_plain=h['iou_plain'], init_dmae_mm=h['dmae_mm'],
                           init_depth_signed_mm=h['depth_signed_mm'], v7_iou_recon=r['v7_iou_recon'], chosen_hypothesis=k,
                           ambiguous=c['ambiguous'], static_stretch=c.get('static_stretch'), stretch_spread_deg_m=c.get('stretch_spread_deg_m'),
                           alternatives=[dict(iou=x['iou'], dmae_mm=x['dmae_mm'], score=x['score'], rot_from_best_deg=C.rot_angle_deg(h['R'], x['R'])) for x in c['hyps_fixed']]))
            camT.append(h['camT']); frames.append(r['frame'])
        iv = np.array([j['v7_iou'] for j in js]); ia = np.array([j['init_iou'] for j in js])
        summ = dict(object=a.obj, episode=ep, mesh=a.mesh, shared_scale=s_shared, scale_v7=e.scale_v7, n_candidates=len(js),
                    v7_iou_median=float(np.median(iv)), init_iou_median=float(np.median(ia)), n_ambiguous=int(sum(j['ambiguous'] for j in js)),
                    scale_samples=[float(x) for x in ss], candidates=js,
                    criteria='mask-only candidate selection (see global_init.py docstring); scored-split inputs only',
                    iou_definition='init_iou / v7_iou = hard IoU with KNOWN occluders as do-not-care (hand-landmark hulls outside the mask, '
                                   'other objects masks, outside image); *_recon = multiep_recon protocol (anything in front by depth = do not care); '
                                   '*_plain = mask vs render')
        (od/f'{a.obj}_global_init.json').write_text(json.dumps(summ, indent=1))
        np.savez(od/f'{a.obj}_global_init.npz', frames=np.array(frames), camera_T_object=np.stack(camT), world_T_canonical=np.stack([e.wTc[f] @ T for f, T in zip(frames, camT)]),
                 scale=s_shared, mesh_path=a.mesh)
        C.say(f'ep{ep}: candidates {len(js)}, IoU median v7 {np.median(iv):.3f} -> init {np.median(ia):.3f}; ambiguous {summ["n_ambiguous"]}')


if __name__ == '__main__':
    main()
