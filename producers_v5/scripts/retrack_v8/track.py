"""retrack_v8 stage 2: full-sequence re-tracking of one object in one scored episode (scored-split inputs only).

Inputs: global-init anchors (global_init.py npz: camera_T_object + shared scale) or any full-sequence pose file (--init, e.g.
a previous retrack with another mesh), object_masks_v3, FoundationStereo depth, ORB world_T_rgb, mediapipe hand hulls.
Per frame: render-and-compare refinement (Adam on rotation + translation, scale FIXED) of a few hypotheses
  {previous world pose (static), constant world velocity, anchor at this frame}; loss = occlusion-aware soft IoU (hand hulls,
  other objects / anything in front by depth, unknown depth = don't care) + depth Huber + back-projected points-to-surface;
  selection score = IoU - 4 x depth MAE - temporal penalty (0.0015/deg, 1/m vs the prediction). If the best IoU < 0.45 on a
  usable mask, a fixed-scale global search (1024 random rotations + 24 cube-group flips of the prediction) is added.
  Frames with mask < 400 px are not measured (interpolated later, flagged).
Forward and backward passes -> per-frame 2-state Viterbi (unary -score, pairwise motion cost) -> static prior: runs of >= 5
untouched frames (no hand-hull pixels within 15 px of the mask) get ONE joint world pose (accepted only if the median IoU of the
run drops by <= 0.03) -> moving frames: score-weighted SE(3) smoothing (+-2 frames) + short tethered re-refinement ->
full-image metrics (occlusion-aware IoU as multiep_recon eval_frames, plain IoU, depth MAE) for v7 and new poses.
Output: outputs/track3/retrack_v8/episode_<ep>/<obj>_poses<tag>.npz + <obj>_track<tag>.json. Does not modify any existing output.
Usage: python3 scripts/retrack_v8/track.py --obj mini_sweeper --ep 19 [--init <npz>] [--mesh <obj>] [--scale s] [--tag _x]
"""
import argparse, json, os, sys, math, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--obj', required=True); ap.add_argument('--ep', type=int, required=True); ap.add_argument('--gpu', default='0')
    ap.add_argument('--mesh', default=None); ap.add_argument('--scale', type=float, default=None); ap.add_argument('--init', default=None)
    ap.add_argument('--tag', default=''); ap.add_argument('--iters', type=int, default=40); ap.add_argument('--res', type=int, default=256)
    ap.add_argument('--min-area', type=int, default=400); ap.add_argument('--anchor-min-iou', type=float, default=0.5)
    a = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import numpy as np, torch
    from scipy.spatial.transform import Rotation as Rot
    import common as C
    od = C.OUTR/f'episode_{a.ep:06d}'; od.mkdir(parents=True, exist_ok=True)
    C.LOGF = open(C.ROOT/f'logs/retrack_v8/track_{a.obj}_ep{a.ep}{a.tag}.log', 'a')
    C.say('args', json.dumps(vars(a)))
    init = a.init or str(od/f'{a.obj}_global_init.npz')
    Z = np.load(init, allow_pickle=True)
    s = float(a.scale if a.scale is not None else (Z['scale'] if 'scale' in Z.files else Z['mesh_scale']))
    mesh_path = a.mesh or str(Z['mesh_path'])
    cx = C.Cx(); mesh = C.Mesh(cx, mesh_path); e = C.Episode(cx, a.ep, a.obj); n = e.n; cc = mesh.cc
    wTc = e.wTc; Winv = np.linalg.inv(wTc)
    # anchors (camera_T_object for mesh = s x canonical)
    anchors = {}
    if 'frames' in Z.files:   # global init
        J = json.loads((od/f'{a.obj}_global_init.json').read_text()) if (od/f'{a.obj}_global_init.json').exists() else None
        for k, f in enumerate(Z['frames']):
            ok = J is None or J['candidates'][k]['init_iou'] >= a.anchor_min_iou
            if ok: anchors[int(f)] = Z['camera_T_object'][k]
    else:                     # full-sequence pose file: every frame is an anchor (world_T_canonical -> camera frame)
        for f in range(n): anchors[f] = Winv[f] @ Z['world_T_canonical'][f]
    C.say(f'{len(anchors)} anchors, scale {s:.4f}, mesh {mesh_path}')
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    lo, hi = mesh.Vn.min(0), mesh.Vn.max(0); corners = lo + corners*(hi - lo)

    def proj_box(T):
        X = (s*corners) @ T[:3, :3].T + T[:3, 3]
        if (X[:, 2] < 0.05).any(): return None
        uv = (X @ e.K.T)[:, :2]/X[:, 2:3]
        return np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])

    def rdiff(A, B): return C.rot_angle_deg(A[:3, :3], B[:3, :3])

    flips = Rot.from_rotvec(np.concatenate([np.zeros((1, 3)), np.eye(3)*np.pi/2, np.eye(3)*np.pi, np.eye(3)*3*np.pi/2,
                                            np.array([[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1]])/np.sqrt(2)*np.pi])).as_matrix()
    Rrand = C.random_rotations(1024, seed=1)

    def step(f, inits, pred_world):
        """inits: list of camera_T_object. Returns dict(camT, score, iou, dmae, sel, searched) or None if not measurable."""
        if e.area[f] < a.min_area or e.dmed[f] <= 0: return None
        pb = proj_box(inits[0]) if inits else None
        if pb is not None and (pb[2] - pb[0]) < 3*W_ and (pb[3] - pb[1]) < 3*H_: box = e.crop_box(f, extra_box=pb)
        else: box = e.crop_box(f)
        ob = e.sample(f, box, a.res); pts = e.pts[f]
        def run(Ts, tether_deg=20.0):
            R0 = cx.T(np.stack([T[:3, :3] for T in Ts])); p0 = cx.T(np.stack([T[:3, 3] + s*T[:3, :3] @ cc for T in Ts]))
            s0 = cx.T([s]*len(Ts))
            R, p, _, tm = C.refine(cx, mesh, ob, R0, p0, s0, pts, iters=a.iters, opt_scale=False, lr=(0.02, 0.004, 0.0),
                                   tether=(R0, p0, math.radians(tether_deg), 0.03))
            Rn, pn = R.cpu().numpy(), p.cpu().numpy(); out = []
            for j in range(len(Ts)):
                T = C.camT_from(Rn[j], pn[j], s, cc); sc = float(tm['iou'][j] - 4*tm['dmae'][j].clamp(max=0.05))
                pen = 0.0
                if pred_world is not None:
                    Wj = wTc[f] @ T; pen = 0.0015*rdiff(Wj, pred_world) + 1.0*np.linalg.norm(Wj[:3, 3] - pred_world[:3, 3])
                out.append(dict(camT=T, score=sc, iou=float(tm['iou'][j]), dmae=float(tm['dmae'][j]), sel=sc - pen))
            return out
        cands = run(inits)
        best = max(cands, key=lambda c: c['sel']); searched = False
        if best['iou'] < 0.45 and e.area[f] >= 1500:
            searched = True
            base = inits[0][:3, :3]
            Rs = np.concatenate([base @ flips, Rrand])
            hs = C.global_search(cx, mesh, e, f, Rs, s_fixed=s, topk=12, iters=60, n_keep=4)
            if hs:
                cands += run([h['camT'] for h in hs], tether_deg=10.0)
                best = max(cands, key=lambda c: c['sel'])
        best['searched'] = searched
        return best

    W_, H_ = C.W, C.H

    def track(order):
        res = [None]*n; prevW = None; prev2W = None; last_meas = None
        for f in order:
            inits = []
            if prevW is not None:
                inits.append(Winv[f] @ prevW)
                if prev2W is not None:
                    dW = prevW @ np.linalg.inv(prev2W); inits.append(Winv[f] @ dW @ prevW)
            if f in anchors: inits.append(anchors[f])
            if prevW is None:
                # start: nearest anchors' world poses (object assumed static) + global search
                near = sorted(anchors, key=lambda k: abs(k - f))[:3]
                inits += [Winv[f] @ wTc[k] @ anchors[k] for k in near]
            pred = prevW if prevW is None or prev2W is None else (prevW @ np.linalg.inv(prev2W)) @ prevW
            r = step(f, inits, pred if prevW is not None else None)
            if r is None:
                res[f] = None
                continue
            res[f] = r; Wf = wTc[f] @ r['camT']
            prev2W, prevW = (prevW if last_meas is not None and abs(f - last_meas) == 1 else None), Wf
            last_meas = f
        return res

    t0 = time.time()
    cache = od/f'_{a.obj}_fwbw{a.tag}.npz'
    if cache.exists():
        q = np.load(cache, allow_pickle=True); fw, bw = list(q['fw']), list(q['bw'])
    else:
        fw = track(range(n)); C.say(f'forward pass {time.time()-t0:.0f}s')
        bw = track(range(n - 1, -1, -1)); C.say(f'backward pass {time.time()-t0:.0f}s')
        np.savez(cache, fw=np.array(fw, dtype=object), bw=np.array(bw, dtype=object))
    # ---------------- 2-state Viterbi over (fw, bw)
    meas = [f for f in range(n) if fw[f] is not None and bw[f] is not None]
    st = {f: [fw[f], bw[f]] for f in meas}
    cost = {meas[0]: np.array([-st[meas[0]][0]['score'], -st[meas[0]][1]['score']])}; back = {}
    for i in range(1, len(meas)):
        f, g = meas[i], meas[i-1]; cur = np.zeros(2); bp = np.zeros(2, int)
        for k in range(2):
            Wk = wTc[f] @ st[f][k]['camT']
            tr = [cost[g][j] + (0.0015*rdiff(Wk, wTc[g] @ st[g][j]['camT']) + np.linalg.norm(Wk[:3, 3] - (wTc[g] @ st[g][j]['camT'])[:3, 3]))/max(1, f - g)
                  for j in range(2)]
            bp[k] = int(np.argmin(tr)); cur[k] = tr[bp[k]] - st[f][k]['score']
        cost[f] = cur; back[f] = bp
    choice = {}; k = int(np.argmin(cost[meas[-1]]))
    for i in range(len(meas) - 1, -1, -1):
        f = meas[i]; choice[f] = k
        if i: k = back[f][k]
    Wsel = np.full((n, 4, 4), np.nan); score = np.full(n, np.nan); fb_dis = np.full(n, np.nan)
    for f in meas:
        r = st[f][choice[f]]; Wsel[f] = wTc[f] @ r['camT']; score[f] = r['score']
        fb_dis[f] = rdiff(wTc[f] @ fw[f]['camT'], wTc[f] @ bw[f]['camT'])
    C.say(f'viterbi: {len(meas)} measured frames, bw chosen {sum(choice.values())}, fw/bw disagree >30deg on {(fb_dis > 30).sum()} frames')
    measured = np.isfinite(score)

    # ---------------- interpolate unmeasured frames (slerp in world)
    def interp(Wa, Wb, t):
        from scipy.spatial.transform import Slerp
        sl = Slerp([0, 1], Rot.from_matrix(np.stack([Wa[:3, :3], Wb[:3, :3]])))
        T = np.eye(4); T[:3, :3] = sl([t]).as_matrix()[0]; T[:3, 3] = (1 - t)*Wa[:3, 3] + t*Wb[:3, 3]; return T
    mi = np.nonzero(measured)[0]
    for f in range(n):
        if measured[f]: continue
        L = mi[mi < f]; Rr = mi[mi > f]
        if len(L) and len(Rr): Wsel[f] = interp(Wsel[L[-1]], Wsel[Rr[0]], (f - L[-1])/(Rr[0] - L[-1]))
        else: Wsel[f] = Wsel[L[-1]] if len(L) else Wsel[Rr[0]]

    # ---------------- static prior on untouched runs
    touched = e.hand_near > 0
    runs = []; f = 0
    while f < n:
        if not touched[f]:
            g = f
            while g + 1 < n and not touched[g + 1]: g += 1
            if g - f + 1 >= 5: runs.append((f, g))
            f = g + 1
        else: f += 1
    static_id = np.full(n, -1); static_log = []
    for ri, (f0, f1) in enumerate(runs):
        fr = [f for f in range(f0, f1 + 1) if measured[f]]
        if len(fr) < 3: continue
        sub = fr if len(fr) <= 24 else [fr[i] for i in np.linspace(0, len(fr) - 1, 24).round().astype(int)]
        ref = Wsel[max(fr, key=lambda f: score[f])]
        wv = torch.zeros(3, device=cx.dev, requires_grad=True); dt = torch.zeros(3, device=cx.dev, requires_grad=True)
        opt = torch.optim.Adam([dict(params=[wv], lr=0.01), dict(params=[dt], lr=0.002)])
        obs = {f: e.sample(f, e.crop_box(f), a.res) for f in sub}
        Rref = cx.T(ref[:3, :3]); tref = cx.T(ref[:3, 3]); ccT = cx.T(cc)
        def world_pose():
            R = C.so3(torch, wv[None])[0] @ Rref
            c_w = tref + Rref @ (s*ccT)          # rotate about the object centre in world
            t = c_w + dt - R @ (s*ccT); return R, t
        for it in range(60):
            for g in opt.param_groups: g['lr'] = g['lr']  # constant
            R, t = world_pose(); loss = 0
            for f in sub:
                Wi = cx.T(Winv[f]); Rc = Wi[:3, :3] @ R; tc = Wi[:3, :3] @ t + Wi[:3, 3]
                p = tc + s*Rc @ ccT
                tm = C.terms(cx, mesh, obs[f], (s*Rc)[None], p[None], e.pts[f], soft=True)  # known-occluder do-not-care
                loss = loss + (tm['sil'] + tm['dep'] + 0.5*tm['pts']).sum()
            opt.zero_grad(); (loss/len(sub)).backward(); opt.step()
        with torch.no_grad():
            R, t = world_pose(); Wj = np.eye(4); Wj[:3, :3] = R.cpu().numpy(); Wj[:3, 3] = t.cpu().numpy()
        ev_j = [C.eval_full(cx, mesh, e, f, Winv[f] @ Wj, s)['iou_known'][0] for f in sub]
        ev_p = [C.eval_full(cx, mesh, e, f, Winv[f] @ Wsel[f], s)['iou_known'][0] for f in sub]
        acc = bool(np.median(ev_j) >= np.median(ev_p) - 0.03)
        static_log.append(dict(frames=[int(f0), int(f1)], n_measured=len(fr), iou_joint_median=float(np.median(ev_j)),
                               iou_perframe_median=float(np.median(ev_p)), accepted=acc,
                               perframe_spread_deg=float(max(rdiff(Wsel[f], Wj) for f in fr)),
                               perframe_spread_m=float(max(np.linalg.norm(Wsel[f][:3, 3] - Wj[:3, 3]) for f in fr))))
        C.say(f'static run {f0}-{f1}: joint IoU {np.median(ev_j):.3f} vs per-frame {np.median(ev_p):.3f} -> {"accepted" if acc else "rejected"}')
        if acc:
            for f in range(f0, f1 + 1): Wsel[f] = Wj; static_id[f] = ri

    # ---------------- smoothing of moving frames + tethered re-refinement
    Wfin = Wsel.copy()
    wts = np.where(measured, np.clip(np.nan_to_num(score, nan=0.0), 0.05, None), 0.02)
    for f in range(n):
        if static_id[f] >= 0: continue
        nb = [g for g in range(f - 2, f + 3) if 0 <= g < n and static_id[g] < 0]
        ww = np.array([wts[g]*math.exp(-0.5*((g - f)/1.0)**2) for g in nb])
        Rm = Rot.from_matrix(np.stack([Wsel[g][:3, :3] for g in nb])).mean(weights=ww).as_matrix()
        Wfin[f][:3, :3] = Rm; Wfin[f][:3, 3] = (ww[:, None]*np.stack([Wsel[g][:3, 3] for g in nb])).sum(0)/ww.sum()
    refined = np.zeros(n, bool)
    for f in range(n):
        if static_id[f] >= 0 or not measured[f]: continue
        T = Winv[f] @ Wfin[f]; ob = e.sample(f, e.crop_box(f, extra_box=proj_box(T)), a.res)
        R0 = cx.T(T[None, :3, :3]); p0 = cx.T((T[:3, 3] + s*T[:3, :3] @ cc)[None])
        R, p, _, tm = C.refine(cx, mesh, ob, R0, p0, cx.T([s]), e.pts[f], iters=25, opt_scale=False, lr=(0.006, 0.0015, 0.0),
                               tether=(R0, p0, math.radians(3), 0.005))
        T2 = C.camT_from(R.cpu().numpy()[0], p.cpu().numpy()[0], s, cc); Wfin[f] = wTc[f] @ T2; refined[f] = True
    C.say(f'smoothing + re-refinement done {time.time()-t0:.0f}s')

    # ---------------- full-image metrics: v7 vs new
    rows = []
    v7m = C.Mesh(cx, str(C.T3/f'procedural_geometry_v1/{a.obj}/visual_4096.obj'))
    for f in range(n):
        Tn = Winv[f] @ Wfin[f]; ev = C.eval_full(cx, mesh, e, f, Tn, s); e7 = C.eval_full(cx, v7m, e, f, e.camT_v7[f], e.scale_v7)
        r = dict(frame=f, mask_px=int(e.area[f]), touched=bool(touched[f]), measured=bool(measured[f]), static_run=int(static_id[f]),
                 iou=float(ev['iou_known'][0]), iou_recon=float(ev['iou'][0]), iou_plain=float(ev['iou_plain'][0]), depth_mae_mm=float(ev['dmae'][0]*1000),
                 depth_signed_mm=float(ev['depth_signed_median'][0]*1000) if np.isfinite(ev['depth_signed_median'][0]) else None,
                 v7_iou=float(e7['iou_known'][0]), v7_iou_recon=float(e7['iou'][0]), v7_iou_plain=float(e7['iou_plain'][0]),
                 v7_depth_mae_mm=float(e7['dmae'][0]*1000),
                 fw_bw_disagree_deg=None if not np.isfinite(fb_dis[f]) else float(fb_dis[f]),
                 fw_bw_score_gap=None if not np.isfinite(fb_dis[f]) else float(abs(fw[f]['score'] - bw[f]['score'])),
                 searched=bool((fw[f] or {}).get('searched', False) or (bw[f] or {}).get('searched', False)))
        why = []
        if not measured[f]: why.append('not_measured(mask<%d px)' % a.min_area)
        if r['iou'] < 0.5 and e.area[f] >= 1500: why.append('low_iou')
        if r['fw_bw_disagree_deg'] is not None and r['fw_bw_disagree_deg'] > 30 and r['fw_bw_score_gap'] < 0.03: why.append('fw_bw_ambiguous')
        if e.area[f] < 1500: why.append('small_mask')
        r['uncertain'] = why; rows.append(r)
    ok = [r for r in rows if r['mask_px'] >= 1500]
    def med(k, rr=ok): return float(np.median([r[k] for r in rr])) if rr else None
    summ = dict(object=a.obj, episode=a.ep, mesh=mesh_path, mesh_scale=s, init=init, n_frames=n, n_anchors=len(anchors),
                frames_scored=len(ok), iou_median=med('iou'), iou_recon_median=med('iou_recon'), iou_plain_median=med('iou_plain'),
                depth_mae_mm_median=med('depth_mae_mm'), depth_signed_mm_median=float(np.median([r['depth_signed_mm'] for r in ok if r['depth_signed_mm'] is not None])),
                v7_iou_median=med('v7_iou'), v7_iou_recon_median=med('v7_iou_recon'), v7_iou_plain_median=med('v7_iou_plain'),
                v7_depth_mae_mm_median=med('v7_depth_mae_mm'),
                iou_p10=float(np.percentile([r['iou'] for r in ok], 10)), v7_iou_p10=float(np.percentile([r['v7_iou'] for r in ok], 10)),
                frac_iou_ge_065=float(np.mean([r['iou'] >= 0.65 for r in ok])), v7_frac_iou_ge_065=float(np.mean([r['v7_iou'] >= 0.65 for r in ok])),
                n_uncertain=int(sum(bool(r['uncertain']) for r in rows)), uncertain_frames=[r['frame'] for r in rows if r['uncertain']],
                static_runs=static_log, n_static_frames=int((static_id >= 0).sum()), n_searched=int(sum(r['searched'] for r in rows)),
                protocol='iou = hard IoU over the full image with KNOWN occluders as do-not-care (hand-landmark hulls outside the mask, other '
                         'objects masks, outside image); iou_recon = multiep_recon eval_frames protocol (depth-in-front / other object in front / '
                         'unknown depth = do not care; lets an object hide behind its support table, reported for comparability only); '
                         'iou_plain = mask vs render; depth MAE over eroded mask, |r| clipped 5 cm; medians over frames with mask >= 1500 px; '
                         'v7 = learning_v7_geom world_T_canonical with its mesh (procedural visual_4096 x uniform_scale_applied)',
                rows=rows)
    (od/f'{a.obj}_track{a.tag}.json').write_text(json.dumps(summ, indent=1))
    camT = np.stack([Winv[f] @ Wfin[f] for f in range(n)])
    np.savez_compressed(od/f'{a.obj}_poses{a.tag}.npz', frame_index=np.arange(n), object_name=a.obj, world_T_canonical=Wfin,
                        camera_T_object=camT, world_T_rgb=wTc, mesh_path=mesh_path, mesh_scale=s,
                        iou=np.array([r['iou'] for r in rows]), iou_plain=np.array([r['iou_plain'] for r in rows]),
                        depth_mae_mm=np.array([r['depth_mae_mm'] for r in rows]), uncertain=np.array([bool(r['uncertain']) for r in rows]),
                        measured=measured, static_run=static_id, touched=touched,
                        semantics='world_T_canonical maps the canonical frame of mesh_path SCALED BY mesh_scale (vertices = mesh_scale x '
                                  'mesh_path vertices, about the canonical origin) to the ORB world of stereo_sequence_orb_objmask '
                                  '(same world as learning_v7_geom poses.npz)')
    C.say(f'DONE ep{a.ep} {a.obj}: IoU(known-occl) median v7 {summ["v7_iou_median"]:.3f} -> {summ["iou_median"]:.3f}; recon-protocol {summ["v7_iou_recon_median"]:.3f} -> {summ["iou_recon_median"]:.3f}; (plain {summ["v7_iou_plain_median"]:.3f} -> '
          f'{summ["iou_plain_median"]:.3f}), depth MAE {summ["v7_depth_mae_mm_median"]:.1f} -> {summ["depth_mae_mm_median"]:.1f} mm, '
          f'>=0.65: {summ["v7_frac_iou_ge_065"]:.2f} -> {summ["frac_iou_ge_065"]:.2f}, uncertain {summ["n_uncertain"]}/{n}')


if __name__ == '__main__':
    main()
