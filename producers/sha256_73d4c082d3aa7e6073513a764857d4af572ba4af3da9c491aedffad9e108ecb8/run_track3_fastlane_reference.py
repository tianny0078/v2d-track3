"""Freeze the corrected full reference and run bounded native action replays.

Original project implementation; imports do not load models or start jobs.
Soft quality failures produce scorecards, never suppress completed artifacts.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/track3/episode_000011"
SIDES = ("left", "right")
CATEGORIES = ("hand_table", "hand_object", "hand_hand", "hand_self", "object_table", "object_object", "other")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save_npz(path, **arrays):
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_npz(path):
    with np.load(path) as cp:
        return {k: cp[k].copy() for k in cp.files}


def urdf_frames(path, link_names, joint_names, fingers, wrists):
    """URDF FK includes fixed fingertips absent from the native body list."""
    root = ET.parse(path).getroot()
    joints = []
    children = set()
    for element in root.findall("joint"):
        origin = element.find("origin")
        T = np.eye(4)
        if origin is not None:
            T[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            T[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
        axis = element.find("axis")
        axis = np.fromstring(axis.get("xyz", "1 0 0"), sep=" ") if axis is not None else np.array([1., 0., 0.])
        parent, child = element.find("parent").get("link"), element.find("child").get("link")
        joints.append((element.get("name"), element.get("type"), parent, child, T, axis))
        children.add(child)
    root_links = [x.get("name") for x in root.findall("link") if x.get("name") not in children]
    assert len(root_links) == 1
    lookup = {name: i for i, name in enumerate(joint_names)}
    result = np.empty((len(wrists), len(link_names), 7))
    for f, wrist in enumerate(wrists):
        T = np.eye(4)
        T[:3, 3] = wrist[:3]
        T[:3, :3] = Rotation.from_quat(wrist[[4, 5, 6, 3]]).as_matrix()
        poses, todo = {root_links[0]: T}, joints.copy()
        while todo:
            remaining = []
            for name, kind, parent, child, origin, axis in todo:
                if parent not in poses:
                    remaining.append((name, kind, parent, child, origin, axis))
                    continue
                motion = np.eye(4)
                if kind != "fixed":
                    value = fingers[f, lookup[name]]
                    if kind in ("revolute", "continuous"):
                        motion[:3, :3] = Rotation.from_rotvec(axis * value).as_matrix()
                    elif kind == "prismatic":
                        motion[:3, 3] = axis * value
                    else:
                        raise ValueError(f"Unsupported URDF joint {name}: {kind}")
                poses[child] = poses[parent] @ origin @ motion
            if len(remaining) == len(todo):
                raise ValueError("Disconnected URDF graph")
            todo = remaining
        for k, name in enumerate(link_names):
            T = poses[name]
            xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
            result[f, k] = np.r_[T[:3, 3], xyzw[[3, 0, 1, 2]]]
    return result


def prepare_reference(model, qref, paths, out):
    cp = load_npz(paths["reference"])
    contract = load_npz(paths["identity"])
    row = pq.read_table(paths["parquet"]).to_pylist()[0]
    original_object_fields = {k: v for k, v in row.items() if k.startswith("object_")}
    observed = contract["present"]
    assert observed.shape == (137, 2) and not observed[5].any()
    wrists, fingers = np.empty((137, 2, 7)), np.empty((137, 2, 22))
    data = mujoco.MjData(model)
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_sharpa_wave_{s}_hand_C_MC") for s in SIDES]
    joint_ids = [[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_sharpa_wave_{n}") for n in row[f"{s}_robot_finger_joint_names"]] for s in SIDES]
    assert min(body_ids) >= 0 and min(min(x) for x in joint_ids) >= 0
    for f in range(137):
        data.qpos[:] = qref[f]
        mujoco.mj_forward(model, data)
        for hi, body in enumerate(body_ids):
            wrists[f, hi] = np.r_[data.xpos[body], data.xquat[body]]
            fingers[f, hi] = qref[f, model.jnt_qposadr[joint_ids[hi]]]
    fk_position_max, fk_angle_max = 0., 0.
    for hi, side in enumerate(SIDES):
        names = row[f"{side}_robot_frame_names"]
        frames = urdf_frames(paths[f"urdf_{side}"], names, row[f"{side}_robot_finger_joint_names"], fingers[:, hi], wrists[:, hi])
        for f in range(137):
            data.qpos[:] = qref[f]
            mujoco.mj_forward(model, data)
            for k, name in enumerate(names):
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_sharpa_wave_{name}")
                if bid < 0:
                    continue
                fk_position_max = max(fk_position_max, float(np.linalg.norm(frames[f, k, :3] - data.xpos[bid])))
                delta = np.zeros(3)
                mujoco.mju_subQuat(delta, frames[f, k, 3:], data.xquat[bid])
                fk_angle_max = max(fk_angle_max, float(np.linalg.norm(delta)))
        row.update({f"robot_{side}_wrist_position": wrists[:, hi, :3].tolist(),
                    f"robot_{side}_wrist_wxyz": wrists[:, hi, 3:].tolist(),
                    f"robot_{side}_finger_joints": fingers[:, hi].tolist(),
                    f"robot_{side}_frames": frames.tolist()})
        for suffix in ("positions", "normals", "part_ids"):
            key = f"mano_{side}_object_contact_{suffix}"
            values = np.asarray(row[key]).copy()
            values[~observed[:, hi]] = 0
            row[key] = values.tolist()
    assert fk_position_max < 1e-6 and fk_angle_max < 1e-5, (fk_position_max, fk_angle_max)
    cp.update(qpos=qref, planned_wrist_table=wrists, planned_finger_q=fingers,
              observed_hand=observed, planned_reference=contract["robot_only"],
              planned_source_frame=contract["robot_source_frame"],
              geometry_proxy_known=contract["geometry_proxy_known"],
              contact_ground_truth_known=contract["contact_ground_truth_known"])
    for key in ("proxy_part_ids", "confidence"):
        cp[key][~observed] = 0
    for key in ("proxy_position_table_unknown_nan", "proxy_normal_table_unknown_nan"):
        cp[key][~observed] = np.nan
    for key in ("observed_hand", "planned_reference", "geometry_proxy_known", "contact_ground_truth_known"):
        row[key] = cp[key].tolist()
    row.update(robot_retarget_scope="Frozen full137 native joint-absolute reference; f5 unobserved, robot-only plan; proxy is not GT",
               force_reference_is_measured=False, quality_gate_pass=False,
               downstream_diagnostic_authorized=True, fastlane_reference_version="v1")
    assert {k: row[k] for k in original_object_fields} == original_object_fields
    save_npz(out / "reference_checkpoint.npz", **cp)
    save_npz(out / "human_observations.npz", present=observed, joints_world=contract["joints_world"], vertices_world=contract["vertices_world"], frame_index=contract["frame_index"])
    pq.write_table(pa.Table.from_pylist([row]), out / "episode11_fastlane.parquet")
    save_npz(out / "frozen_qpos.npz", qpos=qref)
    return dict(frames=137, observed_hand_instances=int(observed.sum()), f5_observed=observed[5].tolist(),
                geometry_proxy_known=int(cp["geometry_proxy_known"].sum()), ground_truth_contacts=int(cp["contact_ground_truth_known"].sum()),
                fk_native_position_max_m=fk_position_max, fk_native_orientation_max_rad=fk_angle_max,
                object_fields_unchanged=True, human_observations_unchanged=True,
                parquet=str(out / "episode11_fastlane.parquet"), checkpoint=str(out / "reference_checkpoint.npz"))


def roles(model):
    result = []
    for b in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "world"
        result.append("table" if b == 0 else next((s for s in ("blue_cup", "beige_cup", "left", "right") if s in name), "other"))
    return result


def contact_category(a, b):
    hands = [r in SIDES for r in (a, b)]
    objects = [r.endswith("cup") for r in (a, b)]
    if "table" in (a, b):
        return "hand_table" if any(hands) else "object_table" if any(objects) else "other"
    if all(hands):
        return "hand_self" if a == b else "hand_hand"
    if all(objects):
        return "object_object"
    return "hand_object" if any(hands) and any(objects) else "other"


def run_variant(model, qref, out, label, deadline):
    directory = out / label
    directory.mkdir(exist_ok=True)
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    ns = mujoco.mj_stateSize(model, state_spec)
    d, target = mujoco.MjData(model), mujoco.MjData(model)
    d.qpos[:] = qref[0]
    mujoco.mj_forward(model, d)
    initial = np.empty(ns)
    mujoco.mj_getState(model, d, initial, state_spec)
    pos = np.flatnonzero(model.actuator_biasprm[:, 1] != 0)
    hinges = np.flatnonzero((model.jnt_type == 3) & model.jnt_limited.astype(bool))
    hq, hd = model.jnt_qposadr[hinges], model.jnt_dofadr[hinges]
    balls = np.flatnonzero(model.jnt_type == 1)
    objects = np.flatnonzero(model.jnt_type == 0)
    oa = model.jnt_qposadr[objects]
    od = np.concatenate([np.arange(model.jnt_dofadr[j], model.jnt_dofadr[j] + 6) for j in objects])
    slide_ids = [[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_wrist_{a}") for a in "xyz"] for s in SIDES]
    wq = model.jnt_qposadr[np.asarray(slide_ids)]
    safety = label == "allhinge_safety"
    lo, hi = model.jnt_range[hinges, 0] + .1, model.jnt_range[hinges, 1] - .1
    assert np.all(lo <= hi)
    role = roles(model)
    n = 6800
    arrays = dict(qpos=np.empty((n, model.nq)), qvel=np.empty((n, model.nv)), ctrl=np.empty((n, model.nu)),
                  qfrc_applied=np.empty((n, model.nv)), target_qpos=np.empty((n, model.nq)), time=np.empty(n),
                  wrist_error_mm=np.empty((n, 2)), object_error_mm=np.empty((n, len(oa))),
                  object_orientation_deg=np.empty((n, len(oa))), joint_excess_rad=np.empty(n),
                  penetration_m=np.empty((n, len(CATEGORIES))), contact_count=np.empty(n, dtype=np.int32),
                  object_contact_force_N=np.empty((n, len(oa), 3)), object_contact_torque_Nm=np.empty((n, len(oa), 3)))
    step = 0
    started = time.monotonic()
    status = dict(state="running", variant=label, started_utc=now(), expected_steps=n,
                  object_assistance=False, physics_timestep_s=.001, reference_fps=20,
                  controller="native position actuator targets plus wrist ball PD-like applied torque, gain25/cap45; velocity targets0",
                  safety=dict(margin_rad=.1, lookahead_s=.05, correction_gain=2.) if safety else None)

    def checkpoint():
        current = np.empty(ns)
        mujoco.mj_getState(model, d, current, state_spec)
        save_npz(directory / "checkpoint.npz", **{k: v[:step] for k, v in arrays.items()},
                 initial_state=initial, current_state=current, initial_qpos=qref[0], initial_qvel=np.zeros(model.nv),
                 integration_state_spec=np.array(int(state_spec)), completed_steps=np.array(step),
                 xfrc_applied_constant=np.zeros((model.nbody, 6)), reference_qpos=qref)
        status.update(completed_steps=step, elapsed_s=time.monotonic() - started, updated_utc=now())
        save_json(directory / "report.json", status)

    try:
        for f in range(1, 137):
            if time.monotonic() >= deadline:
                raise TimeoutError("Bounded run reached deadline; partial checkpoint saved")
            delta = np.zeros(model.nv)
            mujoco.mj_differentiatePos(model, delta, 1., qref[f - 1], qref[f])
            for sub in range(50):
                target.qpos[:] = qref[f - 1]
                mujoco.mj_integratePos(model, target.qpos, delta, (sub + 1) / 50)
                ref = target.qpos.copy()
                if safety:
                    predicted = d.qpos[hq] + .05 * d.qvel[hd]
                    correction = np.maximum(lo - predicted, 0) - np.maximum(predicted - hi, 0)
                    target.qpos[hq] = np.clip(target.qpos[hq] + 2. * correction, lo, hi)
                mujoco.mj_forward(model, target)
                d.ctrl[:] = 0
                d.ctrl[pos] = target.actuator_length[pos]
                d.qfrc_applied[:] = 0
                error = np.zeros(model.nv)
                mujoco.mj_differentiatePos(model, error, 1., d.qpos, target.qpos)
                for j in balls:
                    a = model.jnt_dofadr[j]
                    d.qfrc_applied[a:a + 3] = np.clip(25. * error[a:a + 3], -45., 45.)
                assert not d.qfrc_applied[od].any() and not d.xfrc_applied.any()
                arrays["ctrl"][step], arrays["qfrc_applied"][step], arrays["target_qpos"][step] = d.ctrl, d.qfrc_applied, target.qpos
                mujoco.mj_step(model, d)
                mujoco.mj_forward(model, d)
                if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or d.time < .001 * (step + 1) - 1e-9:
                    raise RuntimeError("Nonfinite state or MuJoCo unstable-state reset")
                arrays["qpos"][step], arrays["qvel"][step], arrays["time"][step] = d.qpos, d.qvel, d.time
                arrays["wrist_error_mm"][step] = np.linalg.norm(d.qpos[wq] - ref[wq], axis=-1) * 1000
                arrays["object_error_mm"][step] = [np.linalg.norm(d.qpos[a:a + 3] - ref[a:a + 3]) * 1000 for a in oa]
                for oi, a in enumerate(oa):
                    orientation = np.zeros(3)
                    mujoco.mju_subQuat(orientation, d.qpos[a + 3:a + 7], ref[a + 3:a + 7])
                    arrays["object_orientation_deg"][step, oi] = np.linalg.norm(orientation) * 180 / np.pi
                arrays["joint_excess_rad"][step] = max(0., float(np.maximum(model.jnt_range[hinges, 0] - d.qpos[hq], d.qpos[hq] - model.jnt_range[hinges, 1]).max()))
                penetration = np.zeros(len(CATEGORIES))
                force, torque = np.zeros((len(oa), 3)), np.zeros((len(oa), 3))
                for ci, contact in enumerate(d.contact[:d.ncon]):
                    bodies = [int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
                    category = contact_category(*(role[b] for b in bodies))
                    k = CATEGORIES.index(category)
                    penetration[k] = max(penetration[k], -float(contact.dist))
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(model, d, ci, wrench)
                    basis = contact.frame.reshape(3, 3)
                    for oi, j in enumerate(objects):
                        body = int(model.jnt_bodyid[j])
                        if body not in bodies:
                            continue
                        sign = 1. if bodies[1] == body else -1.
                        F = sign * (basis.T @ wrench[:3])
                        force[oi] += F
                        torque[oi] += np.cross(contact.pos - d.xipos[body], F) + sign * (basis.T @ wrench[3:])
                arrays["penetration_m"][step] = penetration
                arrays["contact_count"][step] = d.ncon
                arrays["object_contact_force_N"][step], arrays["object_contact_torque_Nm"][step] = force, torque
                step += 1
            if f % 20 == 0:
                checkpoint()
                print(json.dumps(dict(variant=label, completed_frames=f + 1, completed_steps=step)), flush=True)
        # Fresh data receives only the serialized initial integration state and
        # stored actions. No state assignment occurs during its rollout.
        checkpoint()
        saved = load_npz(directory / "checkpoint.npz")
        replay = mujoco.MjData(model)
        mujoco.mj_setState(model, replay, saved["initial_state"], state_spec)
        rp, rv, rt = np.empty_like(saved["qpos"]), np.empty_like(saved["qvel"]), np.empty_like(saved["time"])
        for i in range(n):
            if i % 500 == 0 and time.monotonic() >= deadline:
                raise TimeoutError("Deadline during independent action replay")
            replay.ctrl[:] = saved["ctrl"][i]
            replay.qfrc_applied[:] = saved["qfrc_applied"][i]
            replay.xfrc_applied[:] = saved["xfrc_applied_constant"]
            mujoco.mj_step(model, replay)
            mujoco.mj_forward(model, replay)
            rp[i], rv[i], rt[i] = replay.qpos, replay.qvel, replay.time
        save_npz(directory / "independent_action_replay.npz", qpos=rp, qvel=rv, time=rt, initial_state=saved["initial_state"])
        replay_metrics = dict(steps=n, qpos_max_abs=float(np.max(np.abs(rp - saved["qpos"]))),
                              qvel_max_abs=float(np.max(np.abs(rv - saved["qvel"]))),
                              exact_qpos=bool(np.array_equal(rp, saved["qpos"])), exact_qvel=bool(np.array_equal(rv, saved["qvel"])))
        metrics = {k + "_peak": float(np.max(arrays[k])) for k in ("wrist_error_mm", "object_error_mm", "object_orientation_deg", "joint_excess_rad")}
        metrics.update({k + "_rms": float(np.sqrt(np.mean(arrays[k] ** 2))) for k in ("wrist_error_mm", "object_error_mm", "object_orientation_deg")})
        metrics.update(collision_failed_steps=int(np.any(arrays["penetration_m"][:, :4] > .001, axis=1).sum()),
                       penetration_peak_m={k: float(arrays["penetration_m"][:, i].max()) for i, k in enumerate(CATEGORIES)},
                       contact_force_peak_N=float(np.linalg.norm(arrays["object_contact_force_N"], axis=-1).max()),
                       contact_torque_peak_Nm=float(np.linalg.norm(arrays["object_contact_torque_Nm"], axis=-1).max()))
        scorecard = dict(finite=True, complete137=True, independent_action_replay=bool(replay_metrics["exact_qpos"] and replay_metrics["exact_qvel"]),
                         collision_under_1mm=metrics["collision_failed_steps"] == 0, joint_limits=metrics["joint_excess_rad_peak"] <= 1e-8,
                         wrist_under_20mm=metrics["wrist_error_mm_peak"] <= 20, object_under_20mm=metrics["object_error_mm_peak"] <= 20,
                         object_orientation_under_10deg=metrics["object_orientation_deg_peak"] <= 10)
        status.update(state="completed", completed_utc=now(), metrics=metrics, independent_action_replay=replay_metrics,
                      scorecard=scorecard, quality_pass=all(scorecard.values()), training_ready=False,
                      force_reference_is_measured=False, contact_categories=list(CATEGORIES),
                      native_warning_counts={str(i): int(w.number) for i, w in enumerate(d.warning)},
                      soft_quality_failure_blocks_artifacts=False)
    except Exception:
        status.update(state="partial_or_failed", error=traceback.format_exc())
    checkpoint()
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "fastlane_reference_v1")
    parser.add_argument("--budget-seconds", type=float, default=600.)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    paths = dict(script=Path(__file__), source=BASE / "identity_joint_absolute_v1/checkpoint.npz",
                 scene=BASE / "downstream_physics_v1/scene_cpu.xml", reference=BASE / "downstream_reference_v1/reference_checkpoint.npz",
                 parquet=BASE / "downstream_reference_v1/episode11_geometry_proxy.parquet", identity=BASE / "hand_identity_contract_v1/checkpoint.npz")
    for side in SIDES:
        paths[f"urdf_{side}"] = ROOT / f"vendor/video_to_data/robotic_grounding/flash_chord/src/flash_chord/assets/urdfs/sharpawave/{side}_sharpawave.urdf"
    hashes = {k: dict(path=str(v), sha256=digest(v)) for k, v in paths.items()}
    key = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    report_path = out / "report.json"
    old = json.loads(report_path.read_text()) if report_path.exists() else {}
    if old and old.get("input_key") != key:
        raise RuntimeError("Existing artifact has different inputs/code; use a new --output directory")
    if old.get("state") == "completed":
        print("Exactly matching completed fastlane reused", flush=True)
        return
    started = time.monotonic()
    deadline = started + args.budget_seconds
    report = dict(state="preparing", started_utc=now(), input_key=key, input_sha256=hashes, budget_seconds=args.budget_seconds,
                  session="v2d-t3-fastlane-reference", object_assistance=False, observations_promoted=False)
    save_json(report_path, report)
    model = mujoco.MjModel.from_xml_path(str(paths["scene"]))
    model.opt.timestep = .001
    qref = load_npz(paths["source"])["qpos"]
    assert qref.shape == (137, model.nq) and np.isfinite(qref).all()
    shutil.copyfile(paths["scene"], out / "scene_cpu.xml")
    # The compiled model freezes the explicitly overridden 1ms option.
    mujoco.mj_saveModel(model, str(out / "native_model_1ms.mjb"))
    report["reference"] = prepare_reference(model, qref, paths, out)
    report.update(state="reference_ready", reference_ready_utc=now(), variants={})
    save_json(report_path, report)
    print(json.dumps(report["reference"]), flush=True)
    if args.prepare_only:
        return
    for label in ("baseline", "allhinge_safety"):
        p = out / label / "report.json"
        previous = json.loads(p.read_text()) if p.exists() else {}
        result = previous if previous.get("state") == "completed" else run_variant(model, qref, out, label, deadline)
        report["variants"][label] = result
        save_json(report_path, report)
    report.update(state="completed" if all(r["state"] == "completed" for r in report["variants"].values()) else "partial_or_failed",
                  completed_utc=now(), elapsed_s=time.monotonic() - started,
                  physical_steps=sum(r["completed_steps"] for r in report["variants"].values()),
                  independent_replay_steps=sum(r.get("independent_action_replay", {}).get("steps", 0) for r in report["variants"].values()),
                  training_ready=False, quality_gate_is_diagnostic=True)
    save_json(report_path, report)
    print(json.dumps(dict(state=report["state"], elapsed_s=report["elapsed_s"], output=str(out))), flush=True)
    if report["state"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
