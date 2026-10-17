"""Single frozen evaluator entry for the scored cohort; explicit physics overrides."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--episode', type=int, required=True)
    p.add_argument('--substeps', type=int)
    a = p.parse_args()
    snapshot = Path(__file__).resolve().parent / 'producer-source'
    if not snapshot.is_dir():
        raise ValueError('Missing immutable checkpoint producer')
    sys.path.insert(0, str(snapshot))
    import track3_evaluation_harness as h
    import numpy as np
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    original_config = h.checkpoint_config
    physics = {}
    def config(metadata):
        import copy
        c = copy.deepcopy(original_config(metadata))
        c['training']['device'] = 'cpu:0'
        physics['checkpoint_substeps'] = c['env']['sim']['substeps']
        if a.substeps is not None:
            if a.substeps < 1:
                raise ValueError('substeps must be positive')
            c['env']['sim']['substeps'] = a.substeps
        physics['evaluation_substeps'] = c['env']['sim']['substeps']
        return c
    h.checkpoint_config = config
    from flash_chord.envs.rl import RLEnv
    original_init = RLEnv.__init__
    def cpu_init(self, *args, **kwargs):
        kwargs['device'] = 'cpu'
        return original_init(self, *args, **kwargs)
    RLEnv.__init__ = cpu_init
    RLEnv.capture_step = lambda self: None
    original_record = h.record_object_pose_rollout
    def record(*args, **kwargs):
        r = original_record(*args, **kwargs)
        np.savez_compressed(out / 'rollout_checkpoint.npz',
            achieved_pose_w=r.achieved_pose_w, reference_pose_w=r.reference_pose_w,
            invalid_worlds=r.non_finite_worlds, tracking_error=r.tracking_error)
        if len(r.non_finite_worlds):
            raise ValueError(f'Nonfinite policy worlds: {r.non_finite_worlds}')
        return r
    h.record_object_pose_rollout = record
    result = h.evaluate_checkpoint(a.checkpoint, world_count=1, start_frame=0,
        reset_mode='explicit', object_trajectories_output=out / f'episode_{a.episode:06d}.parquet',
        eval_script=Path(__file__))
    h.write_evaluation_report(out / 'evaluation.json', result)
    report = dict(episode=a.episode, physics=physics, metrics=h.evaluation_scalars(result),
        checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        harness_sha256=hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest(),
        device='cpu', capture=False, object_assistance=False, quality_pass=False, reference_promoted=False)
    (out / 'replay-report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
