"""Zero-residual replay videos: panels = sources (rt20 / hc / hcr / gc2 run), same gain; re-runs each cell with qpos recording
(grasp_construct_v2.common.replay_metrics, identical metric code) into NAS hand_contact_v1/<E>/replay_<src>_<gain>/.
Usage (venv): render_replay_cmp.py --episode 8 --gain g25 --sources rt20 hc hcr [--gc2 g200_x1]
"""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
R = Path('/home/tianyuan/projects/v2d'); sys.path.insert(0, str(R/'scripts/grasp_construct_v2')); sys.path.insert(0, str(R/'scripts/trajopt'))
import common as gc  # noqa: E402

NAS = Path('/mnt/nas3/tianyuan/v2d-track3/hand_contact_v1'); OUT = R/'outputs/track3/hand_contact_v1_fm'


def cell(ep, src, gain):
    d = NAS/f'episode_{ep:06d}'/f'replay_{src}_{gain}'
    if (d/'replay_w1.json').exists(): return d
    d.mkdir(parents=True, exist_ok=True)
    post = next((v for v in ('s04ff', 's08ff', 's04', 'ff') if src.endswith(v)), None)  # robot_post variant: targets from its npz
    p = gc.matrix_parquet(ep, src[:-len(post)] if post else src)
    S = gc.make_sim(p, gain, 1); S.capture()
    if post: S.ref_joint_target = np.load(OUT/f'episode_{ep:06d}'/f'{src}_{gain}.npz')['targets']
    res, rec, _ = gc.replay_metrics(S)
    res.update(episode=ep, source=src, gain=gain, parquet=str(p))
    np.savez_compressed(d/'replay_w1.npz', object_pose_w0=rec['object_pose_w0'], qpos_mj_w0=rec['qpos_mj_w0'], object_ref_pos=S.obj_ref_pos, object_ref_quat_xyzw=S.obj_ref_quat)
    import mujoco
    mujoco.mj_saveModel(S.runner.solver.mj_model, str(d/'scene_world0.mjb')); gc.jdump(d/'replay_w1.json', res)
    return d


def errs(z):
    e = np.linalg.norm(z['object_pose_w0'][..., :3]-z['object_ref_pos'][:len(z['object_pose_w0'])], axis=-1)
    return ['']+['err '+'/'.join(f'{100*x:.0f}' for x in row)+'cm' for row in e]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--episode', type=int, required=True); ap.add_argument('--gain', default='g25')
    ap.add_argument('--sources', nargs='+', default=['rt20', 'hc', 'hcr']); ap.add_argument('--gc2'); a = ap.parse_args()
    import render_replay as rr
    panels = []; dirs = []
    for s in a.sources:
        d = cell(a.episode, s, a.gain); z = np.load(d/'replay_w1.npz'); dirs.append(d)
        panels.append((f'{s}+{a.gain} VOC=0', z['qpos_mj_w0'], errs(z)))
    if a.gc2:
        d = gc.OUT/gc.E(a.episode)/a.gc2; z = np.load(d/'replay_w1.npz'); panels.append((f'gc2 {a.gc2} VOC=0', z['qpos_mj_w0'], errs(z)))
    z = np.load(dirs[0]/'replay_w1.npz'); ref = np.concatenate([z['object_ref_pos'], z['object_ref_quat_xyzw']], -1)
    names = list(json.loads((dirs[0]/'replay_w1.json').read_text())['objects'])
    out = NAS/f'episode_{a.episode:06d}'/f'replay_{"_".join(a.sources)}{"_gc2" if a.gc2 else ""}_{a.gain}.mp4'
    rr.render(dirs[0]/'scene_world0.mjb', names, panels, ref, out, target=np.r_[z['object_ref_pos'][..., :2].reshape(-1, 2).mean(0), .06], fps=10)
    lk = OUT/f'episode_{a.episode:06d}'/out.name
    if not lk.exists(): os.symlink(out, lk)
    print(out)
