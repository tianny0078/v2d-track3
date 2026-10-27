"""Robot-level kinematic check of a reference parquet (no dynamics): robot placed exactly at the zero-residual joint
targets of every frame, objects at their reference poses, CPU MuJoCo collision on the RL scene template model with
3 cm margin (grasp_construct_v2.gc2.Probe). Per moving frame (contact_plan core) of each object:
  - hold_dist_mm: min signed distance between ANY robot hand geom and that object (<0 = penetration), capped at 30 mm
  - pen_mm: max penetration hand-object (any object) and hand-ground over all frames
Usage (venv): robot_kin.py --episode 8 --source hc   (source as in feasibility harness: rt / sp / hc / hc1 ...)
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
R = Path('/home/tianyuan/projects/v2d'); sys.path.insert(0, str(R/'scripts/grasp_construct_v2')); sys.path.insert(0, str(R/'scripts/trajopt'))
import common as gc  # noqa: E402
from gc2 import Probe  # noqa: E402
import contact_plan  # noqa: E402

OUT = R/'outputs/track3/hand_contact_v1_fm'


def run(ep, src, device='cuda:0'):
    pq_path = gc.matrix_parquet(ep, src)
    S = gc.make_sim(pq_path, 'g25', 1, device=device); pr = Probe(S, margin=.03)
    pl = contact_plan.plan(pq_path); core = pl['core']; names = pl['names']
    F, B = core.shape; tg = S.ref_joint_target
    hold = np.full((F, 2, B), .03); ground = np.zeros((F, 2))
    for f in range(F):
        for h, kind, dist, _ in pr.contacts(pr.qpos(f, tg[f])):
            if kind[0] == 'obj': hold[f, h, kind[1]] = min(hold[f, h, kind[1]], dist)
            else: ground[f, h] = max(ground[f, h], -dist)
    res = dict(episode=ep, source=src, parquet=str(pq_path), objects={})
    for b, n in enumerate(names):
        fs = np.flatnonzero(core[:, b])
        if not len(fs): continue
        best = hold[fs, :, b].min(-1)
        res['objects'][n] = dict(moving_frames=int(len(fs)), hold_dist_mm_p10_50_90=np.round(np.quantile(best, [.1, .5, .9])*1000, 1).tolist(),
                                 frac_within_5mm=round(float((best < .005).mean()), 2), pen_mm_p50_p95_max=np.round(np.quantile(np.clip(-best, 0, None), [.5, .95, 1.])*1000, 1).tolist(),
                                 ref_tip_dist_cm_p50=np.round(np.median(pl['ref_tip_dist'][fs, :, b], 0)*100, 1).tolist())
    res.update(hand_obj_pen_all_mm_p95_max=np.round(np.quantile(np.clip(-hold.min((1, 2)), 0, None), [.95, 1.])*1000, 1).tolist(),
               hand_ground_pen_mm_p95_max=np.round(np.quantile(ground.max(1), [.95, 1.])*1000, 1).tolist(),
               hand_obj_pen_f0_mm=round(float(np.clip(-hold[0].min(), 0, None))*1000, 1))
    o = OUT/f'episode_{ep:06d}'; o.mkdir(parents=True, exist_ok=True)
    (o/f'{src}_kin.json').write_text(json.dumps(res, indent=1)); np.savez_compressed(o/f'{src}_kin.npz', hold=hold, ground=ground, core=core)
    print(json.dumps(res), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--episode', type=int, required=True); ap.add_argument('--source', required=True)
    ap.add_argument('--device', default='cuda:0'); a = ap.parse_args(); run(a.episode, a.source, a.device)
