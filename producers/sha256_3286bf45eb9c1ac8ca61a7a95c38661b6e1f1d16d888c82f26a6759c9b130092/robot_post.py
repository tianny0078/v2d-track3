"""Robot-reference post-processing variants on top of a learning_fm_<base> retarget (explicit, reported separately):
  squeeze : during frames where hand h holds an object (hc_held of the corrected hands, core=1 / dilation=.5, 4-frame ramp)
            add `alpha` rad to every flexion joint target of that hand (grip force from the finger PD, kp 1.7) — the
            human-pose retarget puts fingers ON the surface, which gives ~zero normal force.
  ff      : wrist PD feed-forward for gain g (grasp_construct_v2.gc2.feedforward: (M a + kd v)/kp + carried weight/kp),
            carried mass = held object's objects.json mass x hold weight.
Replays in-process with the feasibility-matrix metrics (grasp_construct_v2.common.replay_metrics code path, targets
swapped) and optionally exports episode_reference_trajopt.parquet (only robot_* columns change, round-trip checked) for
TRACK3_TRAJOPT_ROOT=hand_contact_v1_post/<variant>.
Usage (venv): robot_post.py --episode 8 --base hcr --hands hands_contact_v1r --variant s04 --alpha .4 [--ff] --gain g25 [--export]
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
R = Path('/home/tianyuan/projects/v2d'); sys.path.insert(0, str(R/'scripts/grasp_construct_v2')); sys.path.insert(0, str(R/'scripts/trajopt'))
import common as gc  # noqa: E402
from gc2 import feedforward, apply_ff  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

OUT = R/'outputs/track3/hand_contact_v1_fm'; EXP = R/'outputs/track3/hand_contact_v1_post'


def hold_weights(ep, hands, F):
    z = np.load(R/'outputs/track3'/hands/f'episode_{ep:06d}/hands_world.npz', allow_pickle=True)
    held = z['hc_held']  # (T, 2[left,right], B)
    assert held.shape[0] == F
    w = held.max(-1)[:, ::-1].copy()  # -> sim side order (right, left)
    return uniform_filter1d(w, 5, axis=0), held[:, ::-1]


def replay(S, tg):
    S.reset(0); kin_w = np.stack([tg[:, S.L.wrist_pos_dof[h]] for h in range(2)], 1); pen0 = S.penetrations()
    obj, wr, pens, objW = [], [], [], []
    for k in range(S.F):
        S.step(tg[k].astype(np.float32), k, 0.0); op = S.object_poses(); obj.append(op[0]); objW.append(op); wr.append(S.robot_q()[0][0])
        if k % 10 == 0: pens.append(S.penetrations())
    res = gc.metrics_from(S, np.array(obj), np.array(wr), kin_w, pen0, pens)
    if S.W > 1:  # same input in every world (contact chaos / GPU non-determinism): spread of follow per object
        objW = np.array(objW); ws = [gc.metrics_objects(S, objW[:, w]) for w in range(S.W)]
        res['spread'] = {n: dict(follow_p10_50_90=np.round(np.quantile([w[n]['follow_frac'] or 0 for w in ws], [.1, .5, .9]), 2).tolist(),
                                 pos_err_cm_p50=float(np.median([w[n]['pos_err_cm'] for w in ws])), flung_frac=float(np.mean([w[n]['flung'] for w in ws])))
                         for n in ws[0]}
    return res, np.array(obj)


def main(a):
    E = f'episode_{a.episode:06d}'; base = gc.matrix_parquet(a.episode, a.base)
    S = gc.make_sim(base, a.gain, a.worlds); S.capture(); tg = S.ref_joint_target.astype(np.float64).copy(); F = S.F
    w, held = hold_weights(a.episode, a.hands, F)
    if a.alpha:
        _, _, flex = gc.finger_info(S); lo = S.model.joint_limit_lower.numpy()[:S.L.nd]; hi = S.model.joint_limit_upper.numpy()[:S.L.nd]
        for h in range(2):
            d = S.L.finger_dof[h]; tg[:, d] = np.clip(tg[:, d]+a.alpha*w[:, h, None]*flex[h][None], lo[d], hi[d])
    if a.ff:
        man = json.loads((R/'outputs/track3/learning_v7_geom'/E/'objects/objects.json').read_text())
        mass = np.array([o.get('mass_kg') or .15 for o in sorted(man['objects'], key=lambda o: man['object_names'].index(o['name']))])
        names = [b.name for b in S.scene.objects]; order = [man['object_names'].index(n) for n in names]
        for h in range(2):
            pos = tg[:, S.L.wrist_pos_dof[h]]; q = Rotation.from_rotvec(tg[:, S.L.wrist_ori_dof[h]]).as_quat()
            carried = (held[:, h][:, order]*mass[order][None]).sum(-1)*9.81
            lp, lr = feedforward(pos, q, S.sim.fps, a.gain, carried=uniform_filter1d(carried, 5))
            p2, q2 = apply_ff(pos, q, lp, lr); tg[:, S.L.wrist_pos_dof[h]] = p2; tg[:, S.L.wrist_ori_dof[h]] = Rotation.from_quat(q2).as_rotvec()
    res, obj = replay(S, tg)
    res.update(episode=a.episode, source=f'{a.base}+{a.variant}', gain=a.gain, alpha=a.alpha, ff=a.ff, hands=a.hands)
    o = OUT/E; o.mkdir(parents=True, exist_ok=True)
    (o/f'{a.base}{a.variant}_{a.gain}{"_w%d" % a.worlds if a.worlds > 1 else ""}.json').write_text(json.dumps(res, indent=1)); np.savez_compressed(o/f'{a.base}{a.variant}_{a.gain}{"_w%d" % a.worlds if a.worlds > 1 else ""}.npz', object_pose_w0=obj, targets=tg)
    print(json.dumps(res), flush=True)
    if a.export:
        dst = EXP/f'{a.base}{a.variant}_{a.gain}'/E/'episode_reference_trajopt.parquet'
        gc.export_parquet(S, tg, base, dst, f'hand_contact_v1 post: base {a.base}, squeeze alpha {a.alpha}, ff {a.ff} ({a.gain})')
        changed = gc.check_columns(base, dst)
        S2 = gc.make_sim(dst, a.gain, 1); err = float(np.abs(S2.ref_joint_target-tg).max()); assert err < 1e-4, err
        print('exported', dst, 'changed', changed, 'roundtrip max', err, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--episode', type=int, required=True); ap.add_argument('--base', default='hcr')
    ap.add_argument('--hands', default='hands_contact_v1r'); ap.add_argument('--variant', default='s'); ap.add_argument('--alpha', type=float, default=.4)
    ap.add_argument('--ff', action='store_true'); ap.add_argument('--gain', default='g25'); ap.add_argument('--export', action='store_true'); ap.add_argument('--worlds', type=int, default=1)
    main(ap.parse_args())
