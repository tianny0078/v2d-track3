"""Zero-residual replay of a learning_fm_<src> reference with the unchanged feasibility harness
(scripts/feasibility_matrix_zero_residual.run: fcsim = envs/base.py::_transition, VOC 0, friction 1.0, 1 world,
explicit reset(0)); results go to outputs/track3/hand_contact_v1_fm/<E>/<src>_<gain>.{json,npz} instead of feasibility_v1.
Usage (flash-chord venv): CUDA_VISIBLE_DEVICES=X replay.py --episode 8 --source hc --gain g25
"""
import argparse
import sys
from pathlib import Path
R = Path('/home/tianyuan/projects/v2d'); sys.path.insert(0, str(R/'scripts'))
import feasibility_matrix_zero_residual as fm  # noqa: E402

fm.OUT = R/'outputs/track3/hand_contact_v1_fm'
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--episode', type=int, required=True); ap.add_argument('--source', required=True)
    ap.add_argument('--gain', default='g25'); a = ap.parse_args()
    out = fm.OUT/f'episode_{a.episode:06d}'/f'{a.source}_{a.gain}.json'
    if out.exists(): print('exists', out); sys.exit(0)
    fm.run(a.episode, a.source, a.gain)
