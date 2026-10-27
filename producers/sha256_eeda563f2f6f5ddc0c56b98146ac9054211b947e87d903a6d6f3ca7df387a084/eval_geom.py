"""Geometry / evidence metrics of a hands tree vs the v7_geom objects (same code for source and corrected).

Moving frames = contact_plan core of each object (reference objects only, no GT).
- contact_cm: per moving frame of object b, min over both hands of the skin distance of the 5 fingertips and the palm
  centre to b's surface, max(sd - r, 0) (r = 5 mm tips, 15 mm palm centre); p10/50/90.
- assigned_contact_cm: same, but only the assigned hand(s) (opt.assignment on 2D mask proximity) and only thumb +
  nearest other tip (the two-point grasp the optimiser targets).
- pen_mm: per moving frame, max over hands/capsule samples of (r - sd)+ against b; p50/p95/max. Also over ALL frames/objects.
- ground: frames with any capsule sample below the table plane by > 3 mm.
- 2D: px error vs source 2D (the evidence in the objective) and vs MediaPipe (held out), p50/p90 over present joints.
- depth: |camera z - (stereo depth + skin offset)| median over joints with depth (gross > 15 cm excluded from both).
- jitter: joint acceleration mm/frame^2 p50/p90.
Usage: eval_geom.py --tree hands_contact_v1 8 14  (also evaluates learning_v5/hands as 'source')
"""
import argparse
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa
import evidence as EV
from opt import CFG, assignment, SKIN_OFF


def q(x, p, s=1.):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return np.round(np.quantile(x, p)*s, 2).tolist() if len(x) else None


def metrics(ep, tree):
    d = load_episode(ep, hands_root=tree); ev = EV.build(ep); T = d['T']; B = len(d['names']); J = d['J']
    grids = [sdf_grid(m) for m in d['mesh']]
    core = moving_mask(d['A'] @ d['P']); held, _ = assignment(core, ev, CFG, T, B)
    S, rad = hand_samples(J); sd = object_sd(d, grids, S)            # (T,2,N,B)
    pc = J[:, :, [0, 5, 9, 13, 17]].mean(2, keepdims=True)
    sdc = object_sd(d, grids, np.concatenate([J[:, :, TIPS], pc], 2))  # (T,2,6,B)
    skin = sdc-np.r_[[TIP_R]*5, [.015]][None, None, :, None]
    pen = np.nanmax(rad[None, None, :, None]-sd, axis=2)             # (T,2,B)
    out = dict(episode=ep, tree=tree, objects={})
    for b, n in enumerate(d['names']):
        fs = np.flatnonzero(core[:, b])
        if not len(fs): continue
        dist = np.nanmin(np.clip(skin[fs, :, :, b], 0, None).reshape(len(fs), -1), -1)
        a = held[fs, :, b] > .99
        st = skin[fs, :, :, b]
        two = np.maximum(np.clip(st[..., 0], 0, None), np.clip(np.nanmin(st[..., 1:5], -1), 0, None))  # thumb AND another tip
        ad = np.where(a, two, np.nan); ad = np.nanmin(np.where(np.isfinite(ad), ad, np.inf), -1); ad[~np.isfinite(ad)] = np.nan
        pb = np.nanmax(pen[fs, :, b], -1)
        # honest per-frame acceptance: some hand skin within 1 cm of the surface AND no sample deeper than 3 mm
        okf = (dist < .01) & (np.clip(pb, 0, None) <= .003)
        # absolute skin distance (penetration counts as distance, not as contact)
        adist = np.nanmin(np.abs(skin[fs, :, :, b]).reshape(len(fs), -1), -1)
        out['objects'][n] = dict(moving_frames=int(len(fs)), contact_cm=q(dist, [.1, .5, .9], 100), contact_abs_cm=q(adist, [.1, .5, .9], 100),
                                 ok_frac=round(float(okf.mean()), 2),
                                 assigned_frames=int(a.any(-1).sum()), assigned_two_point_cm=q(ad, [.1, .5, .9], 100),
                                 pen_mm=q(np.clip(pb, 0, None), [.5, .95, 1.], 1000))
    allpen = np.nanmax(pen, (1, 2)); allpen = allpen[np.isfinite(allpen)]
    Jt = to_table(S, d['A']); gpen = np.nanmax(rad-Jt[..., 2], axis=(-1))  # (T,2)
    uv, z = project(J, d['world_T_rgb'], d['K'])
    e_src = np.linalg.norm(uv-ev['uv_src'], axis=-1); e_mp = np.linalg.norm(uv-ev['uv_mp'], axis=-1)
    e_mp_src = np.linalg.norm(ev['uv_src']-ev['uv_mp'], axis=-1)
    dok = np.isfinite(ev['d_obs']) & (np.abs(ev['d_obs']-ev['z_src']) < .15)
    dres = np.where(dok, z-(ev['d_obs']+SKIN_OFF), np.nan)
    acc = np.linalg.norm(J[2:]-2*J[1:-1]+J[:-2], axis=-1)
    out.update(pen_all_frames_mm=q(np.clip(allpen, 0, None), [.5, .95, 1.], 1000),
               ground_frames_below_3mm=int((gpen > .003).sum()), ground_pen_mm_max=round(float(np.nanmax(gpen))*1000, 1),
               px_vs_source=q(e_src, [.5, .9]), px_vs_mediapipe=q(e_mp, [.5, .9]), px_source_vs_mediapipe=q(e_mp_src, [.5, .9]),
               depth_resid_cm_abs=q(np.abs(dres), [.5, .9], 100), depth_resid_cm_signed_median=q(dres, [.5], 100),
               accel_mm=q(acc, [.5, .9], 1000))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('episodes', type=int, nargs='+'); ap.add_argument('--tree', default='hands_contact_v1')
    ap.add_argument('--no-source', action='store_true'); a = ap.parse_args()
    import json
    for e in a.episodes:
        for t in ([] if a.no_source else ['learning_v5/hands'])+[a.tree]:
            m = metrics(e, t); print(json.dumps(m), flush=True)
            if t != 'learning_v5/hands': save_json(T3/t/ep_name(e)/'geometry_report.json', m)
            else: save_json(WORK/'source_geometry'/f'{ep_name(e)}.json', m)
