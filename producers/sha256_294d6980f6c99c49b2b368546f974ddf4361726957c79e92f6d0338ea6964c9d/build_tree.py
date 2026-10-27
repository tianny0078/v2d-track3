"""Prepare a learning_fm_hc* tree exactly like learning_fm_rt (prepare_track3_learning_v3.py via run_track3_learning_v3.py
--prepare-only, worlds 1024, 5 x 2M milestones, TRACK3_CONTACT_FRICTION=1.0, TRACK3_VOC_CURRICULUM=1), but with
TRACK3_HANDS_ROOT = the corrected hands.

--shift 1 applies the adopted +1-frame sync compensation (notes/progress.md 10-25 06:35 #6: reference frame f = video
frame f+1, last frame held) to BOTH hands and objects: writes <hands>_s1/<E>/hands_world.npz (joints shifted; metadata
such as track_id/soft_weight/present kept so prepare's weight-file asserts hold) and a shifted objects/poses.npz in the
tree (objects.json poses_path re-pointed). Without --shift the objects dir is a verbatim copy of learning_v7_geom.
Usage: build_tree.py --hands hands_contact_v1 --tree learning_fm_hc [--shift 1] 8 14
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa


def shift_arr(x, s):
    return np.concatenate([x[s:], np.repeat(x[-1:], s, 0)], 0) if s else x


def main(ep, a):
    E = ep_name(ep); src = T3/'learning_v7_geom'/E/'objects'; dst = T3/a.tree/E/'objects'
    hands = a.hands
    if not (T3/a.tree/E/'learning/preparation.json').exists() or json.loads((T3/a.tree/E/'learning/preparation.json').read_text()).get('state') != 'prepared':
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file(): shutil.copy2(f, dst/f.name)
        if a.shift:
            man = json.loads((src/'objects.json').read_text()); z = dict(np.load(man['poses_path'], allow_pickle=True))
            for k in ('world_T_canonical', 'object_pose_wxyz', 'observed', 'planned_reference'):
                if k in z: z[k] = shift_arr(z[k], a.shift)
            np.savez(dst/'poses.npz', **z); man['poses_path'] = str(dst/'poses.npz'); man['sync_shift_frames'] = a.shift
            (dst/'objects.json').write_text(json.dumps(man, indent=1))
            hands = f'{a.hands}_s{a.shift}'
            h = dict(np.load(T3/a.hands/E/'hands_world.npz', allow_pickle=True))
            for k in ('joints_world', 'joints_camera_m'): h[k] = shift_arr(h[k], a.shift)
            h['hc_sync_shift_frames'] = np.array(a.shift)
            (T3/hands/E).mkdir(parents=True, exist_ok=True); np.savez_compressed(T3/hands/E/'hands_world.npz', **h)
        env = dict(os.environ, TRACK3_LEARNING_TREE=a.tree, TRACK3_HANDS_ROOT=hands, TRACK3_CONTACT_FRICTION='1.0', TRACK3_VOC_CURRICULUM='1')
        env.pop('TRACK3_TRAJOPT_ROOT', None)
        cmd = [str(ROOT/'.deps/flash-chord-venv/bin/python'), '-u', str(ROOT/'scripts/run_track3_learning_v3.py'), '--episode', str(ep), '--gpu', '0',
               '--prepare-only', '--worlds', '1024', '--milestone-steps', '2000000', '--milestones', '5']
        r = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True, text=True)
        print(r.stdout[-400:], r.stderr[-2000:], flush=True); r.check_returncode()
    p = json.loads((T3/a.tree/E/'learning/preparation.json').read_text())
    print(ep, p['state'], p['ik']['fingertip_error_p50_p95_m'], 'hands', [k for k in p['binding'] if 'hands_world' in k], flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('episodes', type=int, nargs='+'); ap.add_argument('--hands', default='hands_contact_v1')
    ap.add_argument('--tree', default='learning_fm_hc'); ap.add_argument('--shift', type=int, default=0); a = ap.parse_args()
    for e in a.episodes: main(e, a)
