# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference-tracking objective strategy built from the objective kernels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np
import warp as wp

from flash_chord.data.reference import Reference
from flash_chord.embodiments.base import EmbodimentLayout
from flash_chord.embodiments.binding import HandKeypointFrame, RobotReferenceBinding
from flash_chord.objectives.composition import ObjectiveBuffers
from flash_chord.objectives.config import OBJECTIVE_TERM_NAMES, ObjectiveConfig
from flash_chord.objectives.contact import (
    advance_contact_force_history,
    contact_force_l2,
    contact_force_per_hand_log,
    reset_contact_force_history,
)
from flash_chord.objectives.joints import hand_joint_objective
from flash_chord.objectives.keypoints import (
    KEYPOINT_VECS_NP,
    hand_keypoints_objective,
    object_keypoints_objective,
)
from flash_chord.objectives.wrench import (
    batched_wrench_support,
    compute_reference_supports,
    friction_cone_angles,
    merge_wrench_support,
    sample_wrench_basis,
)
from flash_chord.runtime.command import CommandBuffers
from flash_chord.runtime.contact import ContactTracker

_OBJECT_KEYPOINTS = OBJECTIVE_TERM_NAMES.index("object_keypoints")
_HAND_KEYPOINTS = OBJECTIVE_TERM_NAMES.index("hand_keypoints")
_HAND_JOINT_POS = OBJECTIVE_TERM_NAMES.index("hand_joint_pos")
_CONTACT_WRENCH_SUPPORT = OBJECTIVE_TERM_NAMES.index("contact_wrench_support")
_MISSED_CONTACT = OBJECTIVE_TERM_NAMES.index("missed_contact")
_UNINTENDED_CONTACT = OBJECTIVE_TERM_NAMES.index("unintended_contact")
_CONTACT_FORCE_L2 = OBJECTIVE_TERM_NAMES.index("contact_force_l2")
_FORCE_CLOSURE = OBJECTIVE_TERM_NAMES.index("force_closure")


from flash_chord.objectives.tracking import TrackingObjective, clear_tracking_objectives, gather_objective_reference, normalize_tracking_objectives
from track3_masked_contact import contact_wrench_support_objective, force_closure_objective, missed_contact_penalty, unintended_contact_penalty, gather_known

class MaskedTrackingObjective(TrackingObjective):
    """Pinned evaluator derivative: masks are gathered before one reward composition.

    COM conversion and actual training selection remain a separate integration gate.
    """

    def bind_known(self, known, provenance):
        values = np.asarray(known)
        expected = (self.num_frames, self.num_hands, self.num_bodies)
        if values.shape != expected or not np.isin(values, [0, 1]).all():
            raise ValueError(f"known mask must be binary {expected}")
        if not isinstance(provenance, str) or not provenance.strip():
            raise ValueError("explicit observation-mask provenance required")
        self.mask_provenance = provenance
        device = self.ref_wrench_support.device
        self.reference_known = wp.array(values.astype(np.int32).ravel(), dtype=wp.int32, device=device)
        self.known = wp.zeros(self.world_count * self.num_hands * self.num_bodies, dtype=wp.int32, device=device)
        return self

    def evaluate(self, state, timestep: wp.array, terminated: wp.array) -> None:
        """Evaluate enabled terms and compose the per-world score."""
        wp.launch(gather_known, dim=(self.world_count, self.num_hands * self.num_bodies),
                  inputs=[timestep, self.reference_known, self.num_frames, self.num_hands * self.num_bodies],
                  outputs=[self.known])
        if hasattr(self, "com_slot_ids"):
            from track3_wrench_adapter import runtime_link_to_com
            wp.launch(runtime_link_to_com, dim=len(self.com_slot_ids),
                      inputs=[self.contact.contact_pos_o, self.com_slot_ids, self.com_local],
                      outputs=[self.contact_pos_com])
        enabled = self.enabled
        buffers = self.buffers
        wrench_contact_enabled = any(
            enabled[index] for index in (_CONTACT_WRENCH_SUPPORT, _MISSED_CONTACT, _UNINTENDED_CONTACT, _FORCE_CLOSURE)
        )
        tracking_enabled = any(
            enabled[index]
            for index in (
                _OBJECT_KEYPOINTS,
                _HAND_KEYPOINTS,
                _HAND_JOINT_POS,
                _CONTACT_WRENCH_SUPPORT,
                _MISSED_CONTACT,
                _UNINTENDED_CONTACT,
                _FORCE_CLOSURE,
            )
        )
        if tracking_enabled:
            wp.launch(
                clear_tracking_objectives,
                dim=self.world_count,
                inputs=[self.num_hands],
                outputs=[
                    buffers.object_keypoints,
                    buffers.hand_keypoints,
                    buffers.hand_joint_pos,
                    buffers.contact_wrench_support,
                    buffers.missed_contact,
                    buffers.unintended_contact,
                    buffers.force_closure,
                    self.hand_keypoint_sum,
                    self.hand_joint_value,
                ],
            )
        reference_gather_enabled = enabled[_HAND_KEYPOINTS] or enabled[_HAND_JOINT_POS] or wrench_contact_enabled
        if reference_gather_enabled:
            wp.launch(
                gather_objective_reference,
                dim=self.world_count,
                inputs=[
                    timestep,
                    self.num_frames,
                    self.num_hands,
                    self.num_keypoints,
                    self.num_fingers,
                    self.num_bodies,
                    self.num_basis,
                    int(enabled[_HAND_KEYPOINTS]),
                    int(enabled[_HAND_JOINT_POS]),
                    int(wrench_contact_enabled),
                    self.ref_hand_keypoints_w,
                    self.ref_finger_joint_pos,
                    self.ref_wrench_support,
                ],
                outputs=[
                    self.hand_keypoint_target_w,
                    self.finger_joint_target,
                    self.wrench_support_target,
                ],
            )
        if enabled[_OBJECT_KEYPOINTS]:
            wp.launch(
                object_keypoints_objective,
                dim=self.world_count * self.num_bodies * 6,
                inputs=[
                    state.body_q,
                    self.command.body_ids,
                    self.command.body_target_pos_w,
                    self.command.body_target_quat_w,
                    self.object_keypoint_vecs,
                    self.num_bodies,
                    self.bodies_per_world,
                    self.config.object_keypoints.var,
                    int(self.config.object_keypoints.shape),
                ],
                outputs=[buffers.object_keypoints],
            )
        if enabled[_HAND_KEYPOINTS]:
            wp.launch(
                hand_keypoints_objective,
                dim=self.world_count * self.num_hands * self.num_keypoints,
                inputs=[
                    state.body_q,
                    self.keypoint_body_ids,
                    self.keypoint_local_pos,
                    self.hand_keypoint_target_w,
                    self.num_keypoints,
                    self.num_hands,
                    self.bodies_per_world,
                    self.config.hand_keypoints.var,
                    int(self.config.hand_keypoints.shape),
                    0.0,
                ],
                outputs=[self.hand_keypoint_sum],
            )
        if enabled[_HAND_JOINT_POS]:
            wp.launch(
                hand_joint_objective,
                dim=self.world_count * self.num_hands,
                inputs=[
                    state.joint_q,
                    self.finger_q_ids,
                    self.finger_joint_target,
                    self.num_fingers,
                    self.num_hands,
                    self.num_joint_q,
                    self.config.hand_joint_pos.var,
                    int(self.config.hand_joint_pos.shape),
                    0.0,
                ],
                outputs=[self.hand_joint_value],
            )
        if wrench_contact_enabled:
            contact = self.contact
            assert contact is not None
            wp.launch(
                batched_wrench_support,
                dim=self.world_count * self.num_hands * self.num_bodies * self.num_basis,
                inputs=[
                    getattr(self, "contact_pos_com", contact.contact_pos_o),
                    contact.contact_force_direction_o,
                    self.wrench_contact_enabled,
                    contact.hand_slot_starts,
                    contact.link_counts,
                    contact.layout.slots_per_world,
                    self.num_hands,
                    self.num_bodies,
                    self.wrench_basis,
                    self.cone_cos,
                    self.cone_sin,
                    self.config.friction_coefficient,
                    self.object_radius,
                    self.num_edges,
                    self.num_basis,
                ],
                outputs=[self.current_wrench_support],
            )
            if self.unintended_wrench_support is not self.current_wrench_support:
                wp.launch(
                    batched_wrench_support,
                    dim=self.world_count * self.num_hands * self.num_bodies * self.num_basis,
                    inputs=[
                        getattr(self, "contact_pos_com", contact.contact_pos_o),
                        contact.contact_force_direction_o,
                        self.excluded_wrench_contact_enabled,
                        contact.hand_slot_starts,
                        contact.link_counts,
                        contact.layout.slots_per_world,
                        self.num_hands,
                        self.num_bodies,
                        self.wrench_basis,
                        self.cone_cos,
                        self.cone_sin,
                        self.config.friction_coefficient,
                        self.object_radius,
                        self.num_edges,
                        self.num_basis,
                    ],
                    outputs=[self.unintended_wrench_support],
                )
                wp.launch(
                    merge_wrench_support,
                    dim=self.world_count * self.num_hands * self.num_bodies * self.num_basis,
                    inputs=[self.current_wrench_support],
                    outputs=[self.unintended_wrench_support],
                )
        if enabled[_CONTACT_FORCE_L2]:
            contact = self.contact
            assert contact is not None
            if self.config.contact_force_l2.mode == "per_hand_log":
                wp.launch(
                    contact_force_per_hand_log,
                    dim=self.world_count,
                    inputs=[
                        contact.contact_force_w,
                        contact.hand_slot_starts,
                        contact.link_counts,
                        contact.layout.num_hands,
                        contact.layout.num_objects,
                        contact.layout.slots_per_world,
                        self.world_count,
                        self.config.contact_force_l2.history_length,
                        self.config.contact_force_l2.log_force_floor**2,
                        math.log(self.config.contact_force_l2.log_force_reference),
                        1.0
                        / (
                            math.log(self.config.contact_force_l2.log_force_reference)
                            - math.log(self.config.contact_force_l2.log_force_floor)
                        ),
                        self.contact_force_history_cursor,
                    ],
                    outputs=[
                        self.contact_force_squared_history,
                        buffers.contact_force_l2,
                    ],
                )
                wp.launch(
                    advance_contact_force_history,
                    dim=1,
                    inputs=[
                        self.contact_force_history_cursor,
                        self.config.contact_force_l2.history_length,
                    ],
                )
            else:
                wp.launch(
                    contact_force_l2,
                    dim=self.world_count,
                    inputs=[
                        contact.contact_force_w,
                        contact.layout.slots_per_world,
                        self.config.contact_force_l2.threshold,
                    ],
                    outputs=[buffers.contact_force_l2],
                )
        if enabled[_CONTACT_WRENCH_SUPPORT]:
            wp.launch(
                contact_wrench_support_objective,
                dim=self.world_count,
                inputs=[
                    self.wrench_support_target,
                    self.current_wrench_support,
                    self.known,
                    self.num_hands,
                    self.num_bodies,
                    self.num_basis,
                    self.config.contact_wrench_support.tolerance,
                    self.config.contact_wrench_support.var,
                ],
                outputs=[buffers.contact_wrench_support],
            )
        if enabled[_MISSED_CONTACT]:
            wp.launch(
                missed_contact_penalty,
                dim=self.world_count,
                inputs=[
                    self.wrench_support_target,
                    self.current_wrench_support,
                    self.known,
                    self.num_hands,
                    self.num_bodies,
                    self.num_basis,
                ],
                outputs=[buffers.missed_contact],
            )
        if enabled[_UNINTENDED_CONTACT]:
            wp.launch(
                unintended_contact_penalty,
                dim=self.world_count,
                inputs=[
                    self.wrench_support_target,
                    self.unintended_wrench_support,
                    self.known,
                    self.num_hands,
                    self.num_bodies,
                    self.num_basis,
                ],
                outputs=[buffers.unintended_contact],
            )
        if enabled[_FORCE_CLOSURE]:
            wp.launch(
                force_closure_objective,
                dim=self.world_count,
                inputs=[
                    self.wrench_support_target,
                    self.current_wrench_support,
                    self.known,
                    self.num_hands,
                    self.num_bodies,
                    self.num_basis,
                    self.config.force_closure.min_support,
                ],
                outputs=[buffers.force_closure],
            )
        if enabled[_OBJECT_KEYPOINTS] or enabled[_HAND_KEYPOINTS] or enabled[_HAND_JOINT_POS]:
            wp.launch(
                normalize_tracking_objectives,
                dim=self.world_count,
                inputs=[
                    self.num_hands,
                    self.num_keypoints,
                    self.num_bodies,
                    int(enabled[_OBJECT_KEYPOINTS]),
                    int(enabled[_HAND_KEYPOINTS]),
                    int(enabled[_HAND_JOINT_POS]),
                    self.hand_keypoint_sum,
                    self.hand_joint_value,
                ],
                outputs=[
                    buffers.object_keypoints,
                    buffers.hand_keypoints,
                    buffers.hand_joint_pos,
                ],
            )
        buffers.compose(terminated, self.action_rate_l2, self.action_l2)

@dataclass(frozen=True)
class MaskedObjectiveSpec:
    """Explicit BaseEnv ObjectiveSpec; select via env config.objective.

    No module monkeypatches. This intermediate spec does not claim COM integration.
    """
    config: ObjectiveConfig
    reference_known: np.ndarray
    provenance: str
    body_com_local: np.ndarray | None = None
    com_radius: np.ndarray | None = None

    def build(self, model, embodiment, robot_reference, reference, command, contact,
              action, world_count, frame_dt, device=None,
              reference_joint_q_offset=None, episode_start_frame=None):
        base = self.config.build(model, embodiment, robot_reference, reference, command,
                                 contact, action, world_count, frame_dt, device=device,
                                 reference_joint_q_offset=reference_joint_q_offset,
                                 episode_start_frame=episode_start_frame)
        result = MaskedTrackingObjective(**vars(base))
        result.bind_known(self.reference_known, self.provenance)
        if (self.body_com_local is None) != (self.com_radius is None):
            raise ValueError("COM offsets and COM radii must be supplied together")
        if self.body_com_local is not None:
            from track3_com_reference import COMSupportReference
            view = COMSupportReference(reference, self.body_com_local, self.com_radius,
                                       self.reference_known, self.config.sides)
            support = compute_reference_supports(view, num_basis=result.num_basis,
                num_edges=result.num_edges, mu=self.config.friction_coefficient,
                seed=self.config.wrench_basis_seed, sides=self.config.sides, device=device)
            result.ref_wrench_support.assign(support.ravel())
            result.object_radius = wp.array(np.tile(view.radius, world_count), dtype=wp.float32, device=device)
            if contact is None:
                raise ValueError("COM runtime wiring requires ContactTracker")
            ids = []
            for links in contact.layout.link_counts:
                for body in range(result.num_bodies):
                    ids.extend([body] * links)
            result.com_slot_ids = wp.array(np.tile(ids, world_count), dtype=wp.int32, device=device)
            result.com_local = wp.array(view.com, dtype=wp.vec3, device=device)
            result.contact_pos_com = wp.zeros(world_count * contact.layout.slots_per_world,
                                             dtype=wp.vec3, device=device)
        return result
