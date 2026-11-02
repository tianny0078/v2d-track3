"""Scored-only re-initialisation of the mini_sweeper procedural parameters (analogue of multiep_recon/white_pot_scored_init.py).

Why: the procedural_geometry_v1 sweeper was fitted to a SAM3D mesh hallucinated from an edge-on 86x184 px crop (lineage
notes); in the face-on scored frames its fan is ~1.7x the ring diameter long and only 1.0x wide, while the masks show a short,
wide bristle fan -> render-vs-mask IoU stays ~0.6 even with correct poses and the multiep_recon selector finds 0 keyframes.
Method: part-wise differentiable deformation of the procedural visual_4096 vertices in the canonical frame (markers rigid):
  fan  (z > body top): length factor Lf about the body top, tip width factor wt (y, linear in fan height), tip thickness tt (x)
  ring (z < body bottom): in-plane (y-z) scale r about the ring/body junction
  global isotropic scale s (the per-episode mesh_scale of the retrack poses)
jointly with per-frame pose increments, on up to 40 retrack_v8 round-1 frames (ep19 + ep30) with IoU >= 0.5, mask >= 1500 px,
spread over time; loss = known-occluder soft IoU + depth Huber + back-projected points (common.terms). Fit/validation split:
every 4th selected frame is held out (pose-only refit, shape frozen) for a before/after check.
Then the builder parameters (mini_sweeper.build) are updated: fan_z[1], fan_tip_width, fan_t_tip, ring radii/centre, and the
mesh is rebuilt with the procedural builder (geom venv) -> outputs/track3/retrack_v8/sweeper_scored_init/{visual,visual_4096}.obj.
Inputs: scored-split masks/depth/poses only. Usage: python3 scripts/retrack_v8/sweeper_scored_init.py [--gpu 0]
"""
import argparse, json, os, sys, math, subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--gpu', default='0'); ap.add_argument('--iters', type=int, default=400)
    ap.add_argument('--tag', default='')
    a = ap.parse_args(); os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import numpy as np, torch
    import common as C
    obj = 'mini_sweeper'; out = C.OUTR/'sweeper_scored_init'; out.mkdir(parents=True, exist_ok=True)
    C.LOGF = open(C.ROOT/'logs/retrack_v8/sweeper_scored_init.log', 'a')
    pg = C.T3/'procedural_geometry_v1'/obj; pp = json.loads((pg/'params.json').read_text())
    cx = C.Cx(); mesh = C.Mesh(cx, pg/'visual_4096.obj')
    V0 = mesh.Vn.copy()   # canonical (not centre-shifted)
    zb0, zb1 = pp['body_z']; xc, yc, zc = pp['ring_centre']; fx, fy = pp['fan_centre_xy']; z1 = pp['fan_z'][1]
    rigid = np.zeros(len(V0), bool)
    for mk in pp['markers']:
        c = np.array(mk['centre']); q = np.array(mk['post_to'])
        rigid |= np.linalg.norm(V0 - c, axis=1) < mk['radius']*1.3
        d = V0 - q; t = np.clip(d @ (c - q)/max(np.dot(c - q, c - q), 1e-12), 0, 1)
        rigid |= np.linalg.norm(V0 - (q + t[:, None]*(c - q)), axis=1) < 0.0035
    fanv = (V0[:, 2] > zb1) & ~rigid; ringv = (V0[:, 2] < zb0) & ~rigid
    C.say(f'vertices: fan {fanv.sum()} ring {ringv.sum()} rigid {rigid.sum()} of {len(V0)}')
    Vt = cx.T(V0); fan_t = cx.T(fanv.astype(np.float32)); ring_t = cx.T(ringv.astype(np.float32))
    th = torch.zeros(4, device=cx.dev, requires_grad=True)     # log Lf, log wt, log tt, log r

    def deform(th):
        Lf, wt, tt, r = torch.exp(th)
        x, y, z = Vt[:, 0], Vt[:, 1], Vt[:, 2]
        t = ((z - zb1)/(z1 - zb1)).clamp(0, 1)
        zf = zb1 + (z - zb1)*Lf; yf = fy + (y - fy)*(1 + (wt - 1)*t); xf = fx + (x - fx)*(1 + (tt - 1)*t)
        zr = zb0 + (z - zb0)*r; yr = yc + (y - yc)*r
        X = torch.where(fan_t > 0, xf, x); Y = torch.where(fan_t > 0, yf, torch.where(ring_t > 0, yr, y)); Z = torch.where(fan_t > 0, zf, torch.where(ring_t > 0, zr, z))
        return torch.stack([X, Y, Z], 1)

    # frames from round-1 tracking
    items = []; E = {}; P0 = {}; S = {}
    for ep in (19, 30):
        z = np.load(C.OUTR/f'episode_{ep:06d}/{obj}_poses.npz', allow_pickle=True); S[ep] = float(z['mesh_scale'])
        e = C.Episode(cx, ep, obj); E[ep] = e
        ok = [f for f in range(e.n) if z['iou'][f] >= 0.5 and e.area[f] >= 1500 and z['measured'][f]]
        pick = [ok[i] for i in np.linspace(0, len(ok) - 1, min(20, len(ok))).round().astype(int)] if ok else []
        for f in pick: items.append((ep, f)); P0[(ep, f)] = z['camera_T_object'][f]
    s0 = float(np.mean(list(S.values())))
    hold = items[1::4]; fit = [it for it in items if it not in hold]
    C.say(f'{len(fit)} fit frames, {len(hold)} held-out frames; start scale {s0:.4f}')
    obs = {it: E[it[0]].sample(it[1], E[it[0]].crop_box(it[1], pad=1.6), 256) for it in items}

    def mk_mesh(V):
        class Mm: pass
        m = Mm(); m.V = V - cx.T(mesh.cc); m.F = mesh.F; m.S = None; return m

    # surface samples for the pts term: barycentric samples of the deformed mesh (fixed face/bary choice)
    rng = np.random.default_rng(0); Fn = mesh.Fn
    A_ = np.linalg.norm(np.cross(V0[Fn[:, 1]] - V0[Fn[:, 0]], V0[Fn[:, 2]] - V0[Fn[:, 0]]), axis=1)
    fi = rng.choice(len(Fn), 3000, p=A_/A_.sum()); b = rng.dirichlet([1, 1, 1], 3000)
    fi_t = cx.T(Fn[fi], torch.long); b_t = cx.T(b)

    def surf(V): return (V[fi_t]*b_t[..., None]).sum(1) - cx.T(mesh.cc)

    def pose_params(its):
        R0 = cx.T(np.stack([P0[i][:3, :3] for i in its]))
        p0 = cx.T(np.stack([P0[i][:3, 3] + s0*P0[i][:3, :3] @ mesh.cc for i in its]))   # mesh centre in camera (round-1 scale)
        return R0, p0

    def run(its, shape_opt, iters, th_fixed=None, sb=None):
        sb = s0 if sb is None else sb
        R0, p0 = pose_params(its); n = len(its)
        w = torch.zeros(n, 3, device=cx.dev, requires_grad=True); dp = torch.zeros(n, 3, device=cx.dev, requires_grad=True)
        ls = torch.zeros(1, device=cx.dev, requires_grad=True)
        groups = [dict(params=[w], lr=0.01), dict(params=[dp], lr=0.002)]
        if shape_opt: groups += [dict(params=[th], lr=0.02), dict(params=[ls], lr=0.005)]
        opt = torch.optim.Adam(groups); base = [g['lr'] for g in groups]
        thv = th if th_fixed is None else th_fixed
        for it in range(iters):
            sc = 0.1 + 0.9*0.5*(1 + math.cos(math.pi*it/iters))
            for g, b_ in zip(opt.param_groups, base): g['lr'] = b_*sc
            idx = np.random.default_rng(it).choice(n, min(8, n), replace=False)
            V = deform(thv); m = mk_mesh(V); m.S = surf(V); s = sb*torch.exp(ls)[0]
            R = C.so3(torch, w[idx]) @ R0[idx]; p = p0[idx] + dp[idx]
            # keep the deformation about the canonical origin consistent with the mesh centre shift: X = s R (V - cc) + p
            loss = 0
            for k, j in enumerate(idx):
                tm = C.terms(cx, m, obs[its[j]], (s*R[k])[None], p[k][None], E[its[j][0]].pts[its[j][1]], soft=True)
                loss = loss + (tm['sil'] + tm['dep'] + 0.5*tm['pts']).sum()
            if shape_opt: loss = loss + ((th**2).sum()/0.5**2)*0.01
            opt.zero_grad(); (loss/len(idx)).backward(); opt.step()
            with torch.no_grad(): th.clamp_(math.log(0.4), math.log(2.5))
            if shape_opt and (it % 50 == 0 or it == iters - 1):
                C.say(f'it {it} loss {loss.item()/len(idx):.4f} Lf/wt/tt/r {np.round(torch.exp(th).tolist(), 3).tolist()} s {s.item():.4f}')
        with torch.no_grad():
            V = deform(thv); m = mk_mesh(V); s = sb*torch.exp(ls)[0]; R = C.so3(torch, w) @ R0; p = p0 + dp; ious = []
            for k, it_ in enumerate(its):
                tm = C.terms(cx, m, obs[it_], (s*R[k])[None], p[k][None], None, soft=False); ious.append(float(tm['iou'][0]))
        return ious, float(s)

    th0 = torch.zeros(4, device=cx.dev)
    hb, _ = run(hold, False, 60, th_fixed=th0); fb, _ = run(fit, False, 60, th_fixed=th0)
    C.say(f'before: fit IoU median {np.median(fb):.3f}, held-out (pose refit) {np.median(hb):.3f}')
    fa, s_new = run(fit, True, a.iters)
    thv = th.detach().clone()
    ha, _ = run(hold, False, 60, th_fixed=thv, sb=s_new)
    Lf, wt, tt, r = torch.exp(thv).tolist()
    C.say(f'after: fit IoU median {np.median(fa):.3f}, held-out {np.median(ha):.3f}; Lf {Lf:.3f} wt {wt:.3f} tt {tt:.3f} r {r:.3f} s {s_new:.4f}')
    # ---- updated builder parameters
    q = json.loads(json.dumps(pp))
    q['fan_z'] = [pp['fan_z'][0], zb1 + (z1 - zb1)*Lf]
    q['fan_tip_width'] = pp['fan_tip_width']*wt; q['fan_t_tip'] = pp['fan_t_tip']*tt
    Ro, Rh = pp['ring_R_outer']*r, pp['ring_R_hole']*r
    q['ring_R_outer'], q['ring_R_hole'] = Ro, Rh
    q['ring_minor'] = min((Ro - Rh)/2, 0.007*r); q['ring_major'] = Ro - q['ring_minor']
    q['ring_centre'] = [xc, yc, zb0 + (zc - zb0)*r]
    q['provenance_scored_init'] = dict(method=__doc__.strip().splitlines()[0], factors=dict(Lf=Lf, wt=wt, tt=tt, r=r), mesh_scale=s_new,
                                       fit_frames=[list(map(int, i)) for i in fit], holdout_frames=[list(map(int, i)) for i in hold],
                                       fit_iou_median=[float(np.median(fb)), float(np.median(fa))], holdout_iou_median=[float(np.median(hb)), float(np.median(ha))],
                                       base_params=str(pg/'params.json'), inputs='scored ep19/ep30 masks, FoundationStereo depth, retrack_v8 round-1 poses')
    (out/'params.json').write_text(json.dumps(q, indent=1)); (out/'mesh_scale.json').write_text(json.dumps(dict(mesh_scale=s_new)))
    # ---- rebuild with the procedural builder (geom venv)
    code = f"""
import sys, json; sys.path.insert(0, '{C.ROOT}/scripts/procedural_geometry')
import mini_sweeper as MS, build as B
from common import decimate
p = json.load(open('{out}/params.json'))
vis = B.assemble(MS.build(p, 'high'))[0]; lo = B.assemble(MS.build(p, 'low'))[0]
if len(lo.faces) > 4096: lo = decimate(lo, 4096)
vis.export('{out}/visual.obj'); lo.export('{out}/visual_4096.obj')
print('built', len(vis.faces), len(lo.faces), vis.is_watertight, lo.is_watertight, vis.extents, lo.extents)
"""
    r_ = subprocess.run([str(C.ROOT/'.deps/geom-venv/bin/python'), '-c', code], capture_output=True, text=True)
    C.say(r_.stdout.strip(), r_.stderr.strip()[-500:])


if __name__ == '__main__':
    main()
