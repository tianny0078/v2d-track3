# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contact-wrench objectives on the per-(hand, body, basis-direction) wrench supports.

Each reads the reference and current (sim) support vectors, shaped ``[W·H·B·K]`` (K basis directions per
body). A basis direction is "active" where its support ``> 1e-3``.
"""

from __future__ import annotations

import warp as wp

_ACTIVE = wp.constant(1.0e-3)
_FORCE_SQUARED_NUMERIC_CEILING = wp.constant(1.0e30)


@wp.func
def _idx(w: int, h: int, b: int, k: int, n_hands: int, n_bodies: int, n_basis: int) -> int:
    return ((w * n_hands + h) * n_bodies + b) * n_basis + k


# Project derivative of pinned vendor contact.py; unknown cells are excluded
# before reading support payloads and from every body/hand denominator.
# This module is not yet connected to the training entrypoint.

@wp.kernel
def contact_wrench_support_objective(
    cmd_sup: wp.array(dtype=wp.float32),  # [W*H*B*K] command (demo) supports
    cur_sup: wp.array(dtype=wp.float32),  # [W*H*B*K] current (sim) supports
    known: wp.array(dtype=wp.int32),  # [W*H*B], mandatory observation validity
    n_hands: int,
    n_bodies: int,
    n_basis: int,
    tolerance: float,
    var: float,
    objective: wp.array(dtype=wp.float32),  # [W] out
) -> None:
    """Tolerance-band match of cur vs cmd supports, gated by both active; mean over basis dirs / bodies /
    active hands."""
    w = wp.tid()
    hand_sum = float(0.0)
    n_active_hands = float(0.0)
    for h in range(n_hands):
        body_sum = float(0.0)
        n_active_bodies = float(0.0)
        for b in range(n_bodies):
            if known[(w * n_hands + h) * n_bodies + b] == 0:
                continue
            k_sum = float(0.0)
            cmd_num = float(0.0)
            body_active = int(0)
            for k in range(n_basis):
                i = _idx(w, h, b, k, n_hands, n_bodies, n_basis)
                cs = cmd_sup[i]
                us = cur_sup[i]
                if cs > _ACTIVE:
                    cmd_num += 1.0
                    body_active = 1
                    if us > _ACTIVE:
                        better = wp.max((1.0 - tolerance) * cs - us, 0.0)
                        too_large = wp.max(us - (1.0 + tolerance) * cs, 0.0)
                        loss = better * better + too_large * too_large
                        k_sum += wp.exp(-loss / var)
            if cmd_num > 1.0e-6:
                body_sum += k_sum / cmd_num
            if body_active == 1:
                n_active_bodies += 1.0
        if n_active_bodies > 1.0e-6:
            hand_sum += body_sum / n_active_bodies
            n_active_hands += 1.0
    if n_active_hands > 1.0e-6:
        objective[w] = hand_sum / n_active_hands
    else:
        objective[w] = 0.0


@wp.kernel
def missed_contact_penalty(
    cmd_sup: wp.array(dtype=wp.float32),
    cur_sup: wp.array(dtype=wp.float32),
    known: wp.array(dtype=wp.int32),  # [W*H*B], mandatory observation validity
    n_hands: int,
    n_bodies: int,
    n_basis: int,
    penalty: wp.array(dtype=wp.float32),  # [W] out
) -> None:
    """Per active hand: fraction of commanded basis directions per body that the agent failed to cover,
    averaged over commanded-active bodies; summed over hands / active hands."""
    w = wp.tid()
    hand_sum = float(0.0)
    n_active_hands = float(0.0)
    for h in range(n_hands):
        body_sum = float(0.0)
        n_active_bodies = float(0.0)
        for b in range(n_bodies):
            if known[(w * n_hands + h) * n_bodies + b] == 0:
                continue
            n_expected = float(0.0)
            n_missed = float(0.0)
            for k in range(n_basis):
                i = _idx(w, h, b, k, n_hands, n_bodies, n_basis)
                if cmd_sup[i] > _ACTIVE:
                    n_expected += 1.0
                    if cur_sup[i] <= _ACTIVE:
                        n_missed += 1.0
            if n_expected > 0.0:
                body_sum += n_missed / n_expected
                n_active_bodies += 1.0
        if n_active_bodies > 1.0e-6:
            hand_sum += body_sum / n_active_bodies
            n_active_hands += 1.0
    if n_active_hands > 1.0e-6:
        penalty[w] = hand_sum / n_active_hands
    else:
        penalty[w] = 0.0


@wp.kernel
def unintended_contact_penalty(
    cmd_sup: wp.array(dtype=wp.float32),
    cur_sup: wp.array(dtype=wp.float32),
    known: wp.array(dtype=wp.int32),  # [W*H*B], mandatory observation validity
    n_hands: int,
    n_bodies: int,
    n_basis: int,
    penalty: wp.array(dtype=wp.float32),  # [W] out
) -> None:
    """Per hand: fraction of bodies with contact the reference doesn't ask for (binary) + mean squared
    support on those bodies (continuous); summed over hands (no hand normalization)."""
    w = wp.tid()
    total = float(0.0)
    for h in range(n_hands):
        unintended_count = float(0.0)
        support_sum = float(0.0)
        n_inactive = float(0.0)
        n_known = float(0.0)
        for b in range(n_bodies):
            if known[(w * n_hands + h) * n_bodies + b] == 0:
                continue
            n_known += 1.0
            cmd_active = int(0)
            cur_active = int(0)
            sup_sq = float(0.0)
            for k in range(n_basis):
                i = _idx(w, h, b, k, n_hands, n_bodies, n_basis)
                if cmd_sup[i] > _ACTIVE:
                    cmd_active = 1
                if cur_sup[i] > _ACTIVE:
                    cur_active = 1
                u = wp.max(cur_sup[i], 0.0)
                sup_sq += u * u
            if cmd_active == 0:  # demo asks for no contact on this body
                n_inactive += 1.0
                support_sum += sup_sq / float(n_basis)  # mean over K
                if cur_active == 1:
                    unintended_count += 1.0
        binary = unintended_count / wp.max(n_known, 1.0)  # mean over all bodies
        cont = support_sum / wp.max(n_inactive, 1.0e-6)
        total += binary + cont
    penalty[w] = total


@wp.kernel
def force_closure_objective(
    cmd_sup: wp.array(dtype=wp.float32),
    cur_sup: wp.array(dtype=wp.float32),
    known: wp.array(dtype=wp.int32),  # [W*H*B], mandatory observation validity
    n_hands: int,
    n_bodies: int,
    n_basis: int,
    min_support: float,
    objective: wp.array(dtype=wp.float32),
) -> None:
    """Reward per-body wrench-direction coverage on commanded-active hands."""
    w = wp.tid()
    hand_sum = float(0.0)
    n_active_hands = float(0.0)
    for h in range(n_hands):
        body_sum = float(0.0)
        n_active_bodies = float(0.0)
        for b in range(n_bodies):
            if known[(w * n_hands + h) * n_bodies + b] == 0:
                continue
            n_closed = float(0.0)
            body_active = int(0)
            for k in range(n_basis):
                i = _idx(w, h, b, k, n_hands, n_bodies, n_basis)
                if cmd_sup[i] > _ACTIVE:
                    body_active = 1
                if cur_sup[i] > min_support:
                    n_closed += 1.0
            if body_active == 1:
                body_sum += n_closed / float(n_basis)
                n_active_bodies += 1.0
        if n_active_bodies > 1.0e-6:
            hand_sum += body_sum / n_active_bodies
            n_active_hands += 1.0
    if n_active_hands > 1.0e-6:
        objective[w] = hand_sum / n_active_hands
    else:
        objective[w] = 0.0


@wp.kernel
def gather_known(timestep: wp.array(dtype=wp.int32), reference_known: wp.array(dtype=wp.int32), num_frames: int, cells: int, known: wp.array(dtype=wp.int32)):
    world, cell = wp.tid()
    frame = wp.min(wp.max(timestep[world], 0), num_frames - 1)
    known[world * cells + cell] = reference_known[frame * cells + cell]
