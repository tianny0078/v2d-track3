# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train the reference-tracking policy with native JAX FlashSAC."""

from __future__ import annotations

import os
import time
from pathlib import Path

import hydra
import jax
import warp as wp
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from flash_chord.configuration import instantiate_typed, resolve_path, resolved_dict
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.training.checkpoint_metadata import (
    build_checkpoint_metadata,
    critic_schema,
    policy_schema,
    resolved_reference_config,
)
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.flash_sac.config import TrainingConfig
from flash_chord.training.flash_sac.runner import WandbLogger, metric_definitions, run_training
from flash_chord.training.progress import ConsoleTrainingProgress

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["TZ"] = "PST8PDT,M3.2.0,M11.1.0"
time.tzset()


def main(cfg: DictConfig) -> None:
    """Compose one FlashSAC experiment and run its compiled Warp/JAX training loop."""
    import numpy as np
    from track3_reference_alignment import align_known
    from track3_masked_objective import MaskedTrackingObjective
    source_checkpoint = Path(__file__).resolve().parents[1] / 'outputs/track3/episode_000011/flashchord_reference_preflight/reference_checkpoint.npz'
    source, frames, known, side_indices = align_known(cfg, np.load(source_checkpoint)['contact_ground_truth_known'])
    cfg.env.objective.reference_known = known.tolist()
    assert cfg.scene.world_count == cfg.training.world_count
    env_config = hydra.utils.instantiate(cfg.env, _convert_="all")
    from dataclasses import replace
    env_config = replace(env_config, objective=replace(env_config.objective, config=replace(env_config.objective.config, sides=tuple(env_config.objective.config.sides))))
    embodiment = instantiate_typed(cfg.embodiment, Embodiment)
    collision = instantiate_typed(cfg.collision, CollisionPolicy)
    training = instantiate_typed(cfg.training, TrainingConfig)
    output_dir = Path(HydraConfig.get().runtime.output_dir) if cfg.output_dir is None else resolve_path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with wp.ScopedDevice(training.device):
        setup = setup_scene(
            parquet=cfg.task.parquet,
            control_fps=env_config.sim.fps,
            motion_speed=cfg.task.motion_speed,
            embodiment=embodiment,
            collision=collision,
            world_count=cfg.scene.world_count,
            include_support=cfg.scene.support,
            decompose_objects=cfg.scene.decompose_objects,
            object_scale_min=cfg.scene.object_scale_min,
            object_scale_max=cfg.scene.object_scale_max,
            object_scale_seed=cfg.scene.object_scale_seed,
            contact_friction=cfg.scene.get("contact_friction", 1.0),
            object_free_joint_damping=cfg.scene.get("object_free_joint_damping", 0.0),
            source_frame_playback=bool(cfg.task.source_frame_playback),
            motion_start_frame=int(cfg.task.motion_start_frame),
            motion_end_frame=int(cfg.task.motion_end_frame),
        )
        parquet = setup.parquet_path
        reference = setup.reference
        scene = setup.scene
        env = RLEnv(scene, reference, config=env_config)
        assert isinstance(env.objective, MaskedTrackingObjective)
        assert reference.num_frames == len(frames)
        np.testing.assert_array_equal(reference.object_body_pos_w(), source.object_body_pos_w()[frames])
        env.reset()
        env.capture_step()
        env.reset()
        jax_env = WarpRLEnv(
            env,
            publish_diagnostics=False,
            publish_step_metrics=False,
        )
        environment_config = resolved_dict(cfg)
        environment_config["task"]["parquet"] = str(parquet)
        environment_config["reference"] = resolved_reference_config(reference, str(parquet))
        checkpoint_metadata = build_checkpoint_metadata(
            environment_config,
            policy_schema(env.action, env.observation_strategy),
            critic_schema(env.action, env.observation_strategy, jax_env.critic_context_names),
        )
        logger = None
        if cfg.logging.mode != "disabled":
            logger = WandbLogger(
                training,
                metric_definitions=metric_definitions(jax_env),
                environment_config=environment_config,
                upload_checkpoints=cfg.logging.upload_checkpoints,
                upload_training_state=cfg.logging.upload_training_state,
                mode=cfg.logging.mode,
                dir=str(output_dir),
            )
        learner = run_training(
            jax_env,
            training,
            logger=logger,
            checkpoint_dir=output_dir,
            log_interval=cfg.logging.log_interval,
            voc_scale_override=cfg.voc_scale_override,
            checkpoint_metadata=checkpoint_metadata,
            progress=ConsoleTrainingProgress(training.total_environment_steps, output_dir),
        )
    np.testing.assert_array_equal(env.objective.known.numpy(), 0)
    assert np.isfinite(env.objective.terms.numpy()).all()
    import json
    (output_dir.parent / ('resume_entry.json' if training.resume else 'train_entry.json')).write_text(json.dumps(dict(environment_steps=int(jax.device_get(learner.environment_steps)), source_side_indices=side_indices, frames=len(frames), masked_objective_called=True, actual_GT_known_cells=int(known.sum()), training_ready=False), indent=2))
    print(f"completed {int(jax.device_get(learner.environment_steps))} environment steps; checkpoints: {output_dir}")


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if args.resume:
        cfg.training.resume = True
        cfg.training.total_environment_steps = 80
        cfg.training.checkpoint_path = str(Path(cfg.output_dir) / 'state_64.safetensors')
        cfg.training.checkpoint.load_replay_path = str(Path(cfg.output_dir) / 'replay_64.safetensors')
    main(cfg)
