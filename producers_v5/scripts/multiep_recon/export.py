"""Evaluation + export stage for multiep_recon/recon.py (held-out metrics, meshes, poses, renders, report.json)."""
import json, math, sys, time
from pathlib import Path
import numpy as np, cv2, trimesh

H, W = 767, 1024
ROOT = Path(__file__).resolve().parents[2]


def refit(cx, fs, idx_list, V, F, albedo, gain, bias, camT0, centre, a, frame_terms, Poses, sample_surface, iters):
    """Pose-only refit (shape, albedo, exposure frozen) for frames idx_list of fs. Returns camT [len,4,4]."""
    torch = cx.torch
    poses = Poses(cx, camT0, centre)
    if iters <= 0: return poses.numpy()
    opt = torch.optim.Adam([poses.rot, poses.tra], lr=1e-3)
    Vt = cx.T(V); Ft = cx.T(F, torch.int32); Fl = Ft.long(); alb = cx.T(albedo); g = cx.T(gain); b = cx.T(bias)
    gen = torch.Generator(device=cx.dev); gen.manual_seed(0)
    surf = sample_surface(torch, Vt, Fl, 6000, gen)
    n = len(idx_list); B = 8; idx_t = torch.as_tensor(idx_list, device=cx.dev)
    for it in range(iters):
        for gp in opt.param_groups: gp['lr'] = 1e-3*(0.1 + 0.9*0.5*(1+math.cos(math.pi*it/iters)))
        opt.zero_grad()
        for s in range(0, n, B):                       # full-batch gradient (accumulated), one step per epoch
            loc = torch.arange(s, min(n, s+B), device=cx.dev)
            Rn, tn = poses.get(loc)
            tm = frame_terms(cx, fs, idx_t[loc], Vt, Ft, alb, g, b, Rn, tn, surf, a)
            loss = (a.w_sil*tm['sil'] + a.w_dep*tm['dep'] + a.w_pts*tm['pts']).sum()/n   # geometry-only refit (no colour)
            loss.backward()
        (a.w_tether*poses.tether()).backward()
        opt.step()
    return poses.numpy()


def look_at(centre, d, up, dist):
    c = centre + dist*d; fwd = -d
    x = np.cross(fwd, up); x /= np.linalg.norm(x); y = np.cross(fwd, x)
    Rm = np.stack([x, y, fwd]); T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = -Rm @ c
    return T


def shaded(cx, V, F, albedo, camT, K, grey=False):
    torch = cx.torch
    Vt = cx.T(V); Ft = cx.T(F, torch.int32); Fl = Ft.long()
    fn = torch.cross(Vt[Fl[:, 1]]-Vt[Fl[:, 0]], Vt[Fl[:, 2]]-Vt[Fl[:, 0]], dim=-1)
    vn = torch.zeros_like(Vt).index_add_(0, Fl[:, 0], fn).index_add_(0, Fl[:, 1], fn).index_add_(0, Fl[:, 2], fn)
    vn = vn/vn.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    camT = np.asarray(camT); B = len(camT)
    Rn = cx.T(camT[:, :3, :3]); tn = cx.T(camT[:, :3, 3])
    Kb = cx.T(np.tile([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], (B, 1)))
    alb = cx.T(np.full((len(V), 3), 0.7)) if grey else cx.T(albedo)
    with torch.no_grad():
        _, zr, hard, att = cx.render(Vt, Ft, Rn, tn, Kb, torch.cat([alb, vn], 1), want_soft=False)
        col, nrm = att[..., :3], att[..., 3:]
        nrm = nrm/nrm.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        nc = torch.einsum('bij,bhwj->bhwi', Rn, nrm)          # camera-frame normals
        sh = 0.35 + 0.65*nc[..., 2].abs()
        img = (col*sh[..., None]).clamp(0, 1)
    return img.cpu().numpy(), hard.cpu().numpy()


def turntable(cx, rows, axis, out_png):
    """rows: list of (label, V, F, albedo, grey). 8 azimuths at 25 deg elevation + 1 top view."""
    up = axis/np.linalg.norm(axis)
    e1 = np.cross(up, [1, 0, 0]); e1 = e1 if np.linalg.norm(e1) > 0.1 else np.cross(up, [0, 0, 1]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    K = np.array([[1400., 0, W/2], [0, 1400., H/2], [0, 0, 1]])
    tiles = []
    for label, V, F, alb, grey in rows:
        c = (V.min(0) + V.max(0))/2; dirs = []
        dist = max(0.3, 1.15*float(np.linalg.norm(V - c, axis=1).max())/(185/1400))   # whole object inside the 380 px tile
        for az in range(0, 360, 45):
            el = math.radians(25); t = math.radians(az)
            dirs.append(math.cos(el)*(math.cos(t)*e1 + math.sin(t)*e2) + math.sin(el)*up)
        dirs.append(0.2*e1 + up)
        cams = []
        for d in dirs:
            d = d/np.linalg.norm(d); u2 = up if abs(np.dot(d, up)) < 0.95 else e2
            cams.append(look_at(c, d, u2, dist))
        img, hard = shaded(cx, V, F, alb, np.stack(cams), K, grey)
        row = []
        for k in range(len(cams)):
            im = np.where(hard[k][..., None], img[k], 1.0)
            im = (im[H//2-190:H//2+190, W//2-190:W//2+190]*255).astype(np.uint8)[:, :, ::-1].copy()
            row.append(im)
        r = np.hstack(row); cv2.putText(r, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2); tiles.append(r)
    cv2.imwrite(str(out_png), np.vstack(tiles))


def overlay(cx, fs, i, meshes, camTs, gains, biases, path_label):
    """Panel: RGB + mask contour | before render | after render (render 60% over RGB, render contour cyan)."""
    torch = cx.torch
    rgb = fs.RGB[i].cpu().numpy()[:, :, ::-1].copy(); M = fs.M[i].cpu().numpy()
    Kc = fs.K[i].cpu().numpy(); K = np.array([[Kc[0], 0, Kc[2]], [0, Kc[1], Kc[3]], [0, 0, 1]])
    ys, xs = np.nonzero(M); cy, cxx = (ys.min()+ys.max())//2, (xs.min()+xs.max())//2
    s = int(max(ys.max()-ys.min(), xs.max()-xs.min())*0.75) + 30
    y0, y1, x0, x1 = max(0, cy-s), min(H, cy+s), max(0, cxx-s), min(W, cxx+s)
    cs, _ = cv2.findContours(M.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    panels = []
    p0 = rgb.copy(); cv2.drawContours(p0, cs, -1, (0, 0, 255), 2); panels.append(('RGB+mask', p0))
    for (lab, V, F, alb), camT, g, b in zip(meshes, camTs, gains, biases):
        img, hard = shaded(cx, V, F, np.clip(alb*g + b, 0, 1), camT[None], K)
        im = (img[0]*255).astype(np.uint8)[:, :, ::-1]; hd = hard[0]
        p = rgb.copy(); p[hd] = (0.65*im[hd] + 0.35*rgb[hd]).astype(np.uint8)
        cr, _ = cv2.findContours(hd.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(p, cs, -1, (0, 0, 255), 1); cv2.drawContours(p, cr, -1, (255, 255, 0), 1)
        panels.append((lab, p))
    out = []
    for lab, p in panels:
        c = cv2.resize(p[y0:y1, x0:x1], (360, 360)); cv2.putText(c, lab, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        out.append(c)
    r = np.hstack(out); cv2.putText(r, path_label, (6, 352), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return r


def simplify_4096(V, F, say, remesh_mod):
    """<=4096-face watertight version: quadric (fast_simplification), else topology-preserving collapse fallback."""
    import fast_simplification
    for target in (4000, 3800, 3500, 3000):
        v, f = fast_simplification.simplify(V, F, target_count=target)
        m = trimesh.Trimesh(v, f, process=True); m.remove_unreferenced_vertices()
        if len(m.faces) <= 4096 and m.is_watertight and m.is_winding_consistent and m.euler_number == 2:
            if m.volume < 0: m.invert()
            return m, f'fast_simplification target {target}'
        say(f'fast_simplification target {target}: faces {len(m.faces)} watertight {m.is_watertight} euler {m.euler_number} -> retry')
    L = np.sqrt(trimesh.Trimesh(V, F).area/4000/0.433)
    v, f = V.copy(), F.copy()
    for k in range(30):
        v, f, _ = remesh_mod.collapse_short(v, f, None, L, 3*L, max_rounds=10)
        f = remesh_mod.flip_valence(v, f)
        if len(f) <= 4096: break
        L *= 1.1
    m = trimesh.Trimesh(v, f, process=False)
    return m, f'collapse fallback L={L*1000:.2f}mm'


def finish(a, out, cx, metas, obj, eps, init, st, fr, sel, stats, KF, HO, key_items, hold_items, camT_key0, camT_hold0, P, centre,
           say, summarise, eval_frames, frame_terms, Poses, sample_surface, rms_radius, sha256, dir_digest, FrameSet, hands):
    torch = cx.torch
    import remesh as RM
    ne = len(metas)
    Vb = init['Vp']; Fb = init['F'].astype(np.int64); alb0 = init['albedo']           # before: remeshed procedural (scale 1)
    q = trimesh.load(metas[0]['mesh_path'], process=False, force='mesh')
    if len(q.vertices) != len(P.vertices) or np.abs(q.vertices - metas[0]['scale']*P.vertices).max() > 1e-5:
        # init mesh is not the submitted one (e.g. scored-only re-initialisation): "before" = submitted mesh itself, grey albedo
        Vb = q.vertices/metas[0]['scale']; Fb = np.asarray(q.faces, np.int64); alb0 = np.full((len(Vb), 3), 0.5)
        say(f'before = submitted mesh {metas[0]["mesh_path"]} ({len(Fb)} faces)')
    before_label = f'before: submitted mesh' 
    Va, Fa, alba = fr['V'], fr['F'], fr['albedo']; ga, ba = fr['gain'], fr['bias']
    one, zero = np.ones((ne, 3)), np.zeros((ne, 3))
    efile = out/'eval.json'
    if efile.exists():
        ev = json.loads(efile.read_text())
    else:
        ev = {}
        def run_eval(fs, items, V_by_ep, F, alb, g, b, camT):
            rows = []
            for ei in range(ne):
                il = [i for i, it in enumerate(items) if it[0] == ei]
                if not il: continue
                T = cx.T(camT)
                rows += eval_frames(cx, fs, cx.T(V_by_ep[ei]), cx.T(F, torch.int32), cx.T(alb), cx.T(g), cx.T(b),
                                    T[:, :3, :3], T[:, :3, 3], a, il)
            return rows
        Vb_ep = [Vb*m['scale'] for m in metas]; Vs_ep = [init['V']]*ne; Va_ep = [Va]*ne
        t0 = time.time()
        R = {}
        R['hold_before_initpose'] = run_eval(HO, hold_items, Vb_ep, Fb, alb0, one, zero, camT_hold0)
        R['hold_sharedinit_initpose'] = run_eval(HO, hold_items, Vs_ep, init['F'].astype(np.int64), init['albedo'], one, zero, camT_hold0)
        R['hold_after_initpose'] = run_eval(HO, hold_items, Va_ep, Fa, alba, ga, ba, camT_hold0)
        # pose-only refit, identical protocol for before and after
        refits = {}; yaw_logs = {}
        for tag, Vep, F, alb, g, b in [('before', Vb_ep, Fb, alb0, one, zero), ('after', Va_ep, Fa, alba, ga, ba)]:
            camT = camT_hold0.copy()
            for ei in range(ne):
                il = [i for i, it in enumerate(hold_items) if it[0] == ei]
                if not il: continue
                c0 = camT_hold0[il]
                if getattr(a, 'yaw_axis', None) is not None:   # same symmetry re-init for before and after (each with its own mesh)
                    import recon as RC_
                    c0, yl = RC_.yaw_search(cx, HO, il, Vep[ei], F, c0, a.yaw_axis, a.yaw_centre); yaw_logs.setdefault(tag, []).extend(yl)
                camT[il] = refit(cx, HO, il, Vep[ei], F, alb, g, b, c0, centre, a, frame_terms, Poses, sample_surface, a.refit_iters)
            refits[tag] = camT
            R[f'hold_{tag}_refit'] = run_eval(HO, hold_items, Vep, F, alb, g, b, camT)
        R['key_before_initpose'] = run_eval(KF, key_items, Vb_ep, Fb, alb0, one, zero, camT_key0)
        R['key_after_optpose'] = run_eval(KF, key_items, Va_ep, Fa, alba, ga, ba, fr['camT_key'])
        say(f'eval holdout/key done in {time.time()-t0:.0f}s')
        # all frames (reference), initial poses, streamed per episode
        allb, alla = [], []
        for ei, m in enumerate(metas):
            fl = [s['frame'] for s in stats if s['ep'] == m['ep'] and s['mask_px'] >= a.min_mask_px]
            for s0 in range(0, len(fl), 120):
                items = [(ei, f) for f in fl[s0:s0+120]]
                fs = FrameSet(cx, items, metas, obj, hands)
                camT = np.stack([m['camT0'][f] for _, f in items])
                allb += run_eval(fs, items, Vb_ep, Fb, alb0, one, zero, camT)
                alla += run_eval(fs, items, Va_ep, Fa, alba, ga, ba, camT)
                del fs; torch.cuda.empty_cache()
        R['all_before_initpose'] = allb; R['all_after_initpose'] = alla
        say(f'eval all frames done ({len(allb)} frames) in {time.time()-t0:.0f}s')
        ev = dict(rows=R, refit_camT={k: v.tolist() for k, v in refits.items()}, yaw_logs=yaw_logs)
        efile.write_text(json.dumps(ev))
    R = ev['rows']; refits = {k: np.array(v) for k, v in ev['refit_camT'].items()}
    # per-episode summaries
    summ = {}
    for k, rows in R.items():
        summ[k] = {'all_episodes': summarise(rows)}
        for ei, m in enumerate(metas):
            rr = [r for r in rows if r['item'][0] == ei]
            if rr: summ[k][f'ep{m["ep"]}'] = summarise(rr)
    for ep_key in ['all_episodes'] + [f'ep{m["ep"]}' for m in metas]:
        def g(k, met):
            x = summ[k].get(ep_key, {}) or {}; x = x.get(met) or {}
            return x.get('median', float('nan')), x.get('p10', float('nan'))
        line = []
        for proto in ['initpose', 'refit']:
            b, aa = f'hold_before_{proto}', f'hold_after_{proto}'
            line.append(f'{proto}: IoU med {g(b,"iou")[0]:.3f}->{g(aa,"iou")[0]:.3f} p10 {g(b,"iou")[1]:.3f}->{g(aa,"iou")[1]:.3f} '
                        f'col {g(b,"color_mae")[0]:.1f}->{g(aa,"color_mae")[0]:.1f} dep {g(b,"depth_mae_mm")[0]:.1f}->{g(aa,"depth_mae_mm")[0]:.1f}mm')
        say(f'HOLDOUT {ep_key}: ' + ' | '.join(line))

    # ---------------- meshes
    mesh_full = trimesh.Trimesh(Va, Fa, process=False)
    col8 = (np.clip(alba, 0, 1)*255).round().astype(np.uint8)
    mesh_full.export(out/'mesh_full.obj')
    trimesh.Trimesh(Va, Fa, vertex_colors=np.concatenate([col8, np.full((len(Va), 1), 255, np.uint8)], 1), process=False).export(out/'mesh_full.ply')
    if a.cad and 'V4096' in st:
        m4 = trimesh.Trimesh(st['V4096'], st['F4096'], process=False); how = f'cad path: deformation applied to {st["cad_source"]} x sbar'
        if m4.volume < 0: m4.invert()
        name4 = 'final_4096.obj'
    else:
        m4, how = simplify_4096(Va, Fa, say, RM); name4 = 'mesh_4096.obj'
    m4.export(out/name4)
    chk = {}
    for name in ['mesh_full.obj', name4]:
        mm = trimesh.load(out/name, process=False, force='mesh'); mw = trimesh.Trimesh(mm.vertices, mm.faces, process=True)
        chk[name] = dict(faces=int(len(mm.faces)), vertices=int(len(mm.vertices)), watertight=bool(mw.is_watertight),
                         winding_consistent=bool(mw.is_winding_consistent), is_volume=bool(mw.is_volume), euler=int(mw.euler_number),
                         volume_cm3=float(mw.volume*1e6), extents_mm=(mw.extents*1000).round(2).tolist())
    from scipy.spatial import cKDTree
    s_full = trimesh.sample.sample_surface(mesh_full, 100000, seed=0)[0]; s4 = trimesh.sample.sample_surface(m4, 100000, seed=0)[0]
    chk['final_mesh_file'] = name4
    chk['mesh_4096_vs_full_chamfer_mm'] = float((cKDTree(s4).query(s_full)[0].mean() + cKDTree(s_full).query(s4)[0].mean())/2*1000)
    chk['mesh_4096_method'] = how
    say('mesh checks', json.dumps(chk))
    # scale / shape change
    rP, _ = rms_radius(P.vertices, P.faces); rA, _ = rms_radius(Va, Fa)
    scale_est = dict(shared_rms_radius_scale_vs_procedural=rA/rP, per_episode_iso_scales={f'ep{m["ep"]}': m['scale'] for m in metas},
                     geometric_mean_iso=float(init['sbar']),
                     param_stage=st.get('deform') or dict(profile_radial_scale=st['prof'], dh_mm=(np.array(st['dh'])*1000).tolist(), dt_mm=st['dt']*1000))
    pp = json.loads(Path(a.proc_params).read_text())
    if 'axis' in pp: ax = np.array(pp['axis'], float)
    else:
        ev_, E_ = np.linalg.eigh(np.cov((P.vertices - P.vertices.mean(0)).T)); ax = E_[:, -1]
    ax /= np.linalg.norm(ax)
    hA = Va @ ax; hP = P.vertices @ ax
    scale_est['height_scale_vs_procedural'] = float((np.percentile(hA, 99.9) - np.percentile(hA, 0.1))/(np.percentile(hP, 99.9) - np.percentile(hP, 0.1)))
    try:
        sys.path.insert(0, str(ROOT/'toolkit/8d2bce64/v2d_submission_kit/metric_code/track_3'))
        from MPPE import _object_chamfer_cm
        before4 = trimesh.load(T3_proc4(a), process=False, force='mesh')
        scale_est['official_cdo_cm_before4096_vs_after4096'] = float(_object_chamfer_cm((before4.vertices, before4.faces), (m4.vertices, m4.faces), 0))
    except Exception as e:
        scale_est['official_cdo_error'] = f'{type(e).__name__}: {e}'[:200]
    say('scale', json.dumps(scale_est))

    # ---------------- poses
    for ei, m in enumerate(metas):
        ki = [i for i, it in enumerate(key_items) if it[0] == ei]; hi = [i for i, it in enumerate(hold_items) if it[0] == ei]
        np.savez_compressed(out/f'poses_ep{m["ep"]:06d}.npz', camera_T_object_init=m['camT0'], world_T_rgb=m['wTc'],
                            key_frames=np.array([key_items[i][1] for i in ki]), camera_T_object_key=fr['camT_key'][ki],
                            holdout_frames=np.array([hold_items[i][1] for i in hi]), camera_T_object_holdout_refit=refits['after'][hi],
                            note='canonical frame = mesh_full.obj/mesh_4096.obj; init = learning_v7_geom world_T_canonical with ORB world_T_rgb '
                                 '(valid for the shared mesh up to the small residual gauge); key/holdout = refined/refit poses for this mesh')

    # ---------------- renders
    axis = ax
    turntable(cx, [(f'before: init mesh x {init["sbar"]:.3f} + init colours', init['V'], init['F'], init['albedo'], False),
                   ('after: multi-episode recon (albedo)', Va, Fa, alba, False),
                   ('after: geometry', Va, Fa, alba, True), ('before: submitted geometry', Vb*init['sbar'], Fb, alb0, True)], axis, out/'turntable_before_after.png')
    ov = []
    for ei, m in enumerate(metas):
        hi = [i for i, it in enumerate(hold_items) if it[0] == ei]
        if not hi: continue
        rows = []
        for i in [hi[j] for j in np.linspace(0, len(hi)-1, min(3, len(hi))).round().astype(int)]:
            rows.append(overlay(cx, HO, i, [('before', Vb*m['scale'], Fb, alb0), ('after', Va, Fa, alba)],
                                [refits['before'][i], refits['after'][i]], [1.0, ga[ei]], [0.0, ba[ei]], f'ep{m["ep"]} f{hold_items[i][1]} (holdout, refit pose)'))
        p = out/f'overlay_ep{m["ep"]:06d}.jpg'; cv2.imwrite(str(p), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88]); ov.append(str(p))

    # ---------------- provenance + report
    prov = dict(procedural_mesh=(a.proc_mesh, sha256(a.proc_mesh)), procedural_params=(a.proc_params, sha256(a.proc_params)),
                sam3d_raw_glb=(a.raw_glb, sha256(a.raw_glb)) if a.raw_glb else None,
                metric_registration=(a.metric_registration, sha256(a.metric_registration)) if a.metric_registration and Path(a.metric_registration).exists() else None,
                remesh_py=sha256(Path(__file__).parent/'remesh.py'), recon_py=sha256(Path(__file__).parent/'recon.py'), export_py=sha256(__file__),
                episodes={})
    for m in metas:
        E = m['E']; ks = sel[m['ep']]['keyframes'] + sel[m['ep']]['holdout']
        prov['episodes'][m['ep']] = dict(objects_json=(m['objects_json'], sha256(m['objects_json'])), poses=(m['poses_path'], sha256(m['poses_path'])),
                                         camera_poses=(m['camera_poses'], sha256(m['camera_poses'])), rgb_K=sha256(E/'calibrated_sequence/rgb_K.npy'),
                                         submitted_mesh=(m['mesh_path'], sha256(m['mesh_path'])),
                                         keyframe_holdout_inputs_digest=dir_digest([E/f'calibrated_sequence/frames/{f:06d}.jpg' for f in ks] +
                                                                                   [E/f'object_masks_v3/masks/{f:06d}.npz' for f in ks] +
                                                                                   [E/f'foundation_stereo_v1/depth_rgb/{f:06d}.npz' for f in ks]),
                                         hands=sha256(E/'hand_detector_mediapipe_sequence/visible_candidates.npz'))
    rep = dict(object=obj, episodes=eps, args=vars(a), selection=sel,
               keyframe_scores=[{k: v for k, v in s.items() if k != 'view_dir'} for s in stats],
               metrics_summary=summ, mesh_checks=chk, scale=scale_est,
               init=dict(sbar=float(init['sbar']), proc_mesh=a.proc_mesh, color_alignment=dict(nn_median_mm=float(init['align'][0]*1000), nn_p90_mm=float(init['align'][1]*1000))),
               exposure=dict(gain=ga.tolist(), bias=ba.tolist()), free_stage=dict(remeshes=fr['nremesh'], remesh_rejected=fr['remesh_fail']),
               outputs=dict(mesh_full=str(out/'mesh_full.obj'), mesh_full_ply=str(out/'mesh_full.ply'), mesh_4096=str(out/name4),
                            turntable=str(out/'turntable_before_after.png'), overlays=ov, keyframe_contact_sheet=str(out/'keyframes_contact_sheet.jpg'),
                            poses=[str(out/f'poses_ep{m["ep"]:06d}.npz') for m in metas]),
               protocol=dict(before='submitted per-episode mesh (procedural x uniform_scale_applied) + v7 poses; colour = SAM3D NN transfer, gain 1 bias 0',
                             after='shared mesh + learned albedo + per-episode exposure',
                             initpose='v7 initial poses (no refit)', refit=f'pose-only refit {a.refit_iters} full-batch steps per episode on holdout frames, geometry terms only (silhouette+depth+points, no colour), shape frozen',
                             holdout='clean frames (pass all selector filters) not selected as keyframes and >2 frames from any keyframe',
                             all='all frames with mask >= min_mask_px, initial poses, reference only'),
               init_pose_source=getattr(a, 'init_poses', None) or 'learning_v7_geom world_T_canonical',
               provenance=prov, public_data_read=False)
    (out/'report.json').write_text(json.dumps(rep, indent=1, default=str))
    say('DONE', out)


def T3_proc4(a):
    return str(Path(a.proc_mesh).parent/'visual_4096.obj')
