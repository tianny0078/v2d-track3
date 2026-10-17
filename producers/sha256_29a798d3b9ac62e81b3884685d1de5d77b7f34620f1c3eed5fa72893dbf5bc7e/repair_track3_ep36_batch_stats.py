"""Recalibrate failed actor BatchNorm from a full physical observation sequence.

Never changes learned parameters or overwrites the source checkpoint. This is an
explicit inference repair candidate, not additional gradient training.
"""
import dataclasses
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/track3/scored_baseline_20261017'
REPLAY = BASE/'frozen-cpu-replay-v1'
OUT = BASE/'ep36-batch-stats-repair-v1'
SOURCE = ROOT/'outputs/track3/episode_000036/training/checkpoints/policy_60000.safetensors'


def main():
    OUT.mkdir(exist_ok=True)
    sys.path.insert(0, str(REPLAY/'producer-source'))
    import track3_evaluation_harness as h
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flash_chord.training.flash_sac.evaluation import create_actor_template, compile_policy_action
    from flash_chord.training.flash_sac.checkpoint import load_actor_checkpoint, save_actor_checkpoint
    observations = []
    retained = {}

    def load(checkpoint, training, observation, action_dim, *, deterministic, device):
        actor, template = create_actor_template(training, observation.shape[1], action_dim, device=device)
        state, metadata = load_actor_checkpoint(checkpoint, template)
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(state.params)), 'nonfinite learned parameters'
        bad = [jax.tree_util.keystr(p) for p, x in jax.tree_util.tree_flatten_with_path(state.batch_stats)[0]
               if not np.isfinite(np.asarray(x)).all()]
        assert bad, 'no BatchNorm failure to repair'
        repaired = state.replace(batch_stats=template.batch_stats)
        action = compile_policy_action(actor, repaired, observation, jax.random.PRNGKey(training.seed),
                                       deterministic=True, action_transform=training.network.action_transform,
                                       action_scale=training.network.action_scale)

        def collect(s, obs, key):
            a = np.asarray(obs)
            assert np.isfinite(a).all(), 'nonfinite physical calibration observations'
            observations.append(a.copy())
            return action(s, obs, key)

        retained.update(state=repaired, actor=actor, metadata=metadata, bad=bad,
                        network=training.network, original=state)
        return repaired, collect, jax.random.PRNGKey(training.seed), metadata

    h.load_flash_sac_policy = load
    sys.argv = [str(REPLAY/'evaluate_frozen_cpu.py'), '--episode', '36', '--checkpoint', str(SOURCE),
                '--output', str(OUT/'calibration-rollout'), '--substeps', '5']
    runpy.run_path(str(REPLAY/'evaluate_frozen_cpu.py'), run_name='__main__')
    batch = jnp.asarray(np.concatenate(observations, axis=0))
    assert batch.shape[0] >= 407
    actor = retained['actor'].clone(config=dataclasses.replace(retained['network'], batch_norm_update_rate=1.0))
    state = retained['state']
    _, mutable = actor.apply({'params': state.params, 'batch_stats': state.batch_stats}, batch,
                             training=True, mutable=['batch_stats'])
    repaired = state.replace(batch_stats=mutable['batch_stats'])
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(repaired))
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in
               zip(jax.tree.leaves(repaired.params), jax.tree.leaves(retained['original'].params), strict=True))
    metadata = dict(retained['metadata'], inference_repair='empirical_batch_stats_full407',
                    source_checkpoint_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    repair_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target = OUT/'policy_60000_recalibrated.safetensors'
    save_actor_checkpoint(target, repaired, metadata=metadata)
    report = dict(source_checkpoint=str(SOURCE), repaired_checkpoint=str(target),
                  source_checkpoint_sha256=metadata['source_checkpoint_sha256'],
                  repaired_checkpoint_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  bad_batch_stat_paths=retained['bad'], calibration_observations=int(batch.shape[0]),
                  learned_parameters_unchanged=True, all_repaired_tensors_finite=True,
                  extra_gradient_updates=0, physical_replay_required=True, quality_pass=False,
                  repair_source_sha256=metadata['repair_source_sha256'])
    (OUT/'repair-report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
