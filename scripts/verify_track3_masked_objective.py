# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the concrete reference-tracking objective strategy."""

from types import SimpleNamespace

import numpy as np





class _Reference:
    num_frames = 1
    fps = 50.0
    sides = ("right",)

    def wrist_pos_w(self, side):
        return np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    def wrist_quat_w(self, side):
        return np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    def finger_joint_pos(self, side):
        return np.array([[0.2]], dtype=np.float32)

    def finger_joint_names(self, side):
        return ["right_joint"]

    def object_body_pos_w(self):
        return np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)

    def object_body_quat_w(self):
        return np.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32)

    def object_articulation(self):
        return np.zeros((1, 0), dtype=np.float32)

    def contact_pos_w(self, side):
        return np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)

    def contact_normal_w(self, side):
        return np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)

    def contact_part_ids(self, side):
        return np.array([[1]], dtype=np.int32)

    def object_mesh_radius(self):
        return np.array([1.0], dtype=np.float32)

    def frame_pos_w(self, side):
        return np.array([[[0.5, 0.0, 0.0]]], dtype=np.float32)

    def frame_names(self, side):
        return ["right_index_DP"]


def _robot_layout_and_binding():
    from flash_chord.embodiments.base import BodyFrame, EmbodimentLayout, HandLayout, HandLinkGeometry
    from flash_chord.embodiments.binding import BoundHandReference, RobotReferenceBinding

    palm = BodyFrame("right_hand_C_MC", 0)
    dp = BodyFrame("right_index_DP", 1)
    fingertip = BodyFrame("right_index_fingertip", 1, body_to_frame_pos=(0.05, 0.0, 0.0))
    embodiment = EmbodimentLayout(
        num_joint_q=1,
        num_joint_dof=1,
        hands=(
            HandLayout(
                side="right",
                finger_dof_ids=(0,),
                link_geometry=(
                    HandLinkGeometry("hand_C_MC", (0,), True),
                    HandLinkGeometry("index_DP", (1,), True),
                ),
                finger_q_ids=(0,),
                finger_joint_names=("right_joint",),
                palm_frame=palm,
                dp_frames=(dp,),
                fingertip_frames=(fingertip,),
            ),
        ),
    )
    identity = np.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
    hand = BoundHandReference(
        side="right",
        wrist_pos_w=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        wrist_quat_w=identity[:, 0],
        arm_joint_pos=np.empty((1, 0), dtype=np.float32),
        finger_joint_pos=np.array([[0.2]], dtype=np.float32),
        dp_pos_w=np.array([[[0.5, 0.0, 0.0]]], dtype=np.float32),
        dp_quat_w=identity,
        fingertip_pos_w=np.array([[[0.55, 0.0, 0.0]]], dtype=np.float32),
        fingertip_quat_w=identity,
    )
    binding = RobotReferenceBinding(
        num_frames=1,
        fps=50.0,
        num_joint_q=1,
        num_joint_dof=1,
        joint_q=np.array([[0.2]], dtype=np.float32),
        joint_target=np.array([[0.2]], dtype=np.float32),
        hands=(hand,),
    )
    return embodiment, binding


def test_exact_tracking_and_contact_have_analytic_maximum():
    import warp as wp

    from flash_chord.objectives.composition import Objective, ObjectiveDiagnostics
    from flash_chord.objectives.config import (
        OBJECTIVE_TERM_NAMES,
        ContactForceObjectiveTermConfig,
        ContactSupportObjectiveTermConfig,
        ObjectiveConfig,
        ObjectiveTermConfig,
        ShapedObjectiveTermConfig,
    )
    from flash_chord.objectives.tracking import TrackingObjective
    from track3_masked_objective import MaskedObjectiveSpec, MaskedTrackingObjective
    from flash_chord.objectives.composition import ObjectiveSpec
    from flash_chord.runtime.contact import ContactTracker, ContactTrackerConfig

    embodiment, robot_reference = _robot_layout_and_binding()
    reference = _Reference()
    frame_dt = 0.02
    config = ObjectiveConfig(
        object_keypoints=ShapedObjectiveTermConfig(weight=1.0),
        hand_keypoints=ShapedObjectiveTermConfig(weight=1.0),
        hand_joint_pos=ShapedObjectiveTermConfig(weight=1.0),
        contact_wrench_support=ContactSupportObjectiveTermConfig(weight=1.0),
        missed_contact=ObjectiveTermConfig(weight=-1.0),
        unintended_contact=ObjectiveTermConfig(weight=-1.0),
        termination=ObjectiveTermConfig(weight=-1.0),
        action_rate_l2=ObjectiveTermConfig(weight=-1.0),
        action_l2=ObjectiveTermConfig(weight=-1.0),
        contact_force_l2=ContactForceObjectiveTermConfig(weight=0.0, threshold=2.0),
        num_wrench_basis=16,
        num_friction_cone_edges=4,
        sides=("right",),
    )
    with wp.ScopedDevice("cpu"):
        model = SimpleNamespace(
            body_label=["right_hand_C_MC", "right_index_DP", "object"],
            body_count=3,
            joint_coord_count=1,
        )
        command = SimpleNamespace(
            layout=SimpleNamespace(num_bodies=1),
            body_ids=wp.array([2], dtype=wp.int32),
            body_target_pos_w=wp.array([[1.0, 0.0, 0.0]], dtype=wp.vec3),
            body_target_quat_w=wp.array([[0.0, 0.0, 0.0, 1.0]], dtype=wp.quat),
            object_radius=wp.array([1.0], dtype=wp.float32),
        )
        contact = ContactTracker.build(
            embodiment,
            object_body_ids=(2,),
            world_count=1,
            bodies_per_world=3,
            shapes_per_world=3,
            config=ContactTrackerConfig(sides=("right",)),
        )
        contact.contact_pos_o.assign(np.zeros((2, 3), dtype=np.float32))
        contact.contact_force_direction_o.assign(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32))
        contact.contact_force_w.assign(np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32))
        objective = TrackingObjective.build(
            model,
            embodiment,
            robot_reference,
            reference,
            command,
            contact,
            action_rate_l2=wp.zeros(1, dtype=wp.float32),
            action_l2=wp.zeros(1, dtype=wp.float32),
            world_count=1,
            frame_dt=frame_dt,
            config=config,
        )
        official_objective = objective
        spec = MaskedObjectiveSpec(config, np.ones((1, 1, 1), dtype=np.int32), "analytic fixture: fully observed")
        assert isinstance(spec, ObjectiveSpec)
        action = SimpleNamespace()
        from dataclasses import replace
        # Disable action terms so this fixture requires no fake action protocol.
        spec = replace(spec, config=replace(config,
            action_rate_l2=replace(config.action_rate_l2, enabled=False),
            action_l2=replace(config.action_l2, enabled=False)))
        objective = spec.build(model, embodiment, robot_reference, reference, command, contact,
                               action, 1, frame_dt, device="cpu")
        assert isinstance(objective, MaskedTrackingObjective)
        state = SimpleNamespace(
            body_q=wp.array(
                [
                    wp.transform_identity(),
                    wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
                    wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
                ],
                dtype=wp.transform,
            ),
            joint_q=wp.array([0.2], dtype=wp.float32),
        )
        objective.evaluate(
            state,
            timestep=wp.zeros(1, dtype=wp.int32),
            terminated=wp.zeros(1, dtype=wp.int32),
        )

    assert isinstance(objective, Objective)
    assert isinstance(objective, ObjectiveDiagnostics)
    assert objective.term_names == OBJECTIVE_TERM_NAMES
    np.testing.assert_allclose(objective.terms.numpy()[:4], 1.0, atol=1.0e-5)
    np.testing.assert_allclose(objective.terms.numpy()[4:9], 0.0, atol=1.0e-6)
    np.testing.assert_allclose(objective.terms.numpy()[9], 9.0, atol=1.0e-6)
    np.testing.assert_allclose(objective.score.numpy(), [4.0 * frame_dt], atol=1.0e-6)



    timestep = wp.zeros(1, dtype=wp.int32, device="cpu")
    terminated = wp.zeros(1, dtype=wp.int32, device="cpu")
    official_objective.evaluate(state, timestep, terminated)
    parity = float(np.max(np.abs(objective.terms.numpy() - official_objective.terms.numpy())))
    assert parity < 1e-6
    # Unknown poison must be excluded by evaluator kernels before composition.
    objective.reference_known.zero_()
    objective.ref_wrench_support.fill_(float("nan"))
    objective.evaluate(state, timestep, terminated)
    terms = objective.terms.numpy()
    assert np.isfinite(terms).all()
    np.testing.assert_array_equal(terms[[3,4,5,10]], 0)
    np.testing.assert_allclose(objective.score.numpy(), [3 * frame_dt], atol=1e-6)
    try:
        objective.bind_known(np.ones((1,1,1)), "")
        raise AssertionError("missing provenance accepted")
    except ValueError:
        pass
    return dict(all_known_full_evaluator_max_error=parity, unknown_nan_isolated=True,
                unknown_terms_zero=True, composed_score=float(objective.score.numpy()[0]),
                objective_spec_factory_verified=True, new_physics_steps=0, world_count=1,
                production_integrated=False, training_ready=False)

if __name__ == "__main__":
    import json, hashlib, datetime
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs/track3/episode_000011/masked_objective_integration"
    out.mkdir(parents=True, exist_ok=True)
    report = test_exact_tracking_and_contact_have_analytic_maximum()
    report["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), root/"scripts/track3_masked_objective.py"]}
    np.savez_compressed(out/"checkpoint.npz", all_known_error=report["all_known_full_evaluator_max_error"], unknown_score=report["composed_score"])
    (out/"report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
