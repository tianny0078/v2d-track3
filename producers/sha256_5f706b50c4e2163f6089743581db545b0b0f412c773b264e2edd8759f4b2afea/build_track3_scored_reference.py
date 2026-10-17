"""Bounded, generic scored-video reference adapter; no training or GPU work.

Position IK reuses the project's original initializer, not NVIDIA PINK.
Missing observations remain missing; finite robot plans are separately marked.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import time
import traceback
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

import initialize_sharpa_retarget as ik
from run_track3_fastlane_reference import urdf_frames, save_json, save_npz, now, digest

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "vendor/video_to_data/robotic_grounding/flash_chord/src/flash_chord/assets/urdfs/sharpawave"
SIDES = ("left", "right")
CONTACT_TERMS = ("contact_wrench_support", "missed_contact", "unintended_contact", "contact_force_l2", "force_closure")
TEMPLATE = ROOT / "outputs/track3/episode_000011/downstream_training_contact_off_v1/config.yaml"


def npz_dict(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def absolute(path, parent):
    p = Path(path)
    return p.resolve() if p.is_absolute() else (parent / p).resolve()


def read_inputs(args):
    manifest_path = args.object_manifest.resolve()
    obj = json.loads(manifest_path.read_text())
    h = npz_dict(args.hands)
    frames = np.asarray(h["frame_index"], dtype=np.int64)
    T = len(frames)
    assert T >= 2 and np.all(np.diff(frames) == 1), "Require contiguous source frame indices"
    present = np.asarray(h["present"], dtype=bool)
    joints = np.asarray(h["joints_world"], dtype=np.float64)
    assert present.shape == (T, 2) and joints.shape == (T, 2, 21, 3)
    vertices = np.asarray(h.get("vertices_world", np.full((T, 2, 778, 3), np.nan)), dtype=np.float64)
    assert vertices.shape == (T, 2, 778, 3)
    if "frame_index" in obj:
        np.testing.assert_array_equal(frames, np.asarray(obj["frame_index"]))
    if "episode_id" in obj:
        assert int(obj["episode_id"]) == args.episode_id
    fps = float(obj["fps"] if args.fps is None else args.fps)
    assert fps > 0 and np.isclose(fps, float(obj.get("fps", fps))), "Do not silently retime"
    names = list(obj["object_names"])
    assert names and len(set(names)) == len(names)
    B = len(names)
    records = {x["name"]: x for x in obj["objects"]}
    assert set(records) == set(names)
    records = [records[name] for name in names]
    paths = {}
    for i, item in enumerate(records):
        for key in ("mesh_path", "urdf_path"):
            item[key] = str(absolute(item[key], manifest_path.parent))
            assert Path(item[key]).is_file(), item[key]
            paths[f"object_{i}_{key}"] = Path(item[key])
    poses = npz_dict(absolute(obj["poses_path"], manifest_path.parent)) if "poses_path" in obj else obj
    if "poses_path" in obj:
        paths["object_poses"] = absolute(obj["poses_path"], manifest_path.parent)
    if "world_T_canonical" in poses:
        P = np.asarray(poses["world_T_canonical"], dtype=np.float64)
    elif "object_pose_wxyz" in poses:
        p = np.asarray(poses["object_pose_wxyz"], dtype=np.float64)
        assert p.shape == (T, B, 7)
        P = np.broadcast_to(np.eye(4), (T, B, 4, 4)).copy()
        P[..., :3, 3] = p[..., :3]
        P[..., :3, :3] = Rotation.from_quat(p[..., [4, 5, 6, 3]].reshape(-1, 4)).as_matrix().reshape(T, B, 3, 3)
    else:
        raise ValueError("Object manifest needs finite link-origin poses")
    assert P.shape == (T, B, 4, 4) and np.isfinite(P).all()
    np.testing.assert_allclose(P[..., 3, :], np.broadcast_to([0, 0, 0, 1], (T, B, 4)), atol=1e-7)
    np.testing.assert_allclose(P[..., :3, :3].swapaxes(-1, -2) @ P[..., :3, :3], np.broadcast_to(np.eye(3), (T, B, 3, 3)), atol=1e-5)
    assert np.all(np.linalg.det(P[..., :3, :3]) > .999)
    assert obj.get("pose_semantics") == "link_origin"
    A = np.asarray(obj["table_T_world"], dtype=np.float64)
    assert A.shape == (4, 4) and np.isfinite(A).all()
    np.testing.assert_allclose(A[3], [0, 0, 0, 1], atol=1e-7)
    np.testing.assert_allclose(A[:3, :3].T @ A[:3, :3], np.eye(3), atol=1e-6)
    assert np.linalg.det(A[:3, :3]) > .999
    coordinate_frame = obj.get("coordinate_frame", obj.get("coord_frame", "world"))
    assert coordinate_frame in ("world", "table", "table_Z_up_meters")
    table_poses = A @ P if coordinate_frame == "world" else P.copy()
    com = np.asarray([x["canonical_COM"] for x in records], dtype=np.float64)
    radius = np.asarray([x["bounding_radius_m"] for x in records], dtype=np.float64)
    assert com.shape == (B, 3) and np.isfinite(com).all() and radius.shape == (B,) and np.isfinite(radius).all() and (radius > 0).all()
    return dict(T=T, B=B, frames=frames, present=present, joints=joints, vertices=vertices,
                fps=fps, names=names, records=records, poses=poses, table_poses=table_poses,
                table_T_world=A, canonical_COM=com, radius=radius, manifest=obj, input_paths=paths)


def kinematics(side, out):
    # Build our own mesh-free derivative in this episode's output, leaving the
    # initializer's old OUT and every official asset untouched.
    tree = ET.parse(ik.ASSETS / f"{side}_sharpawave.xml")
    root = tree.getroot()
    for parent in root.iter():
        for child in list(parent):
            if child.tag in {"asset", "geom", "actuator", "contact"}:
                parent.remove(child)
    root.find("compiler").attrib.pop("meshdir", None)
    path = out / f"{side}_kinematics.xml"
    tree.write(path, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    ids = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, side + "_" + site) for site in ik.SITES])
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
    assert model.nq == model.nv == 22 and model.jnt_limited.all() and (ids >= 0).all()
    return model, data, ids, names


def build_robot(data, out, budget):
    T = data["T"]
    models = [kinematics(side, out) for side in SIDES]
    q = np.full((T, 2, 22), np.nan)
    wrist = np.full((T, 2, 7), np.nan)
    done = np.zeros((T, 2), dtype=bool)
    eligible = data["present"] & np.isfinite(data["joints"]).all(axis=(-1, -2))
    rows = []
    start = time.monotonic()
    # Cover the complete sequence before spending the remaining budget densely.
    anchors = np.unique(np.linspace(0, T - 1, min(T, 16)).round().astype(int)).tolist()
    order = anchors + [i for i in range(T) if i not in set(anchors)]
    def checkpoint():
        save_npz(out / "ik_checkpoint.npz", observed_finger_q=q, observed_wrist_world=wrist, done=done,
                 observed_hand=data["present"], ik_eligible=eligible, frame_index=data["frames"])
        save_json(out / "ik-report.json", dict(solved=int(done.sum()), eligible=int(eligible.sum()), elapsed_s=time.monotonic() - start, rows=rows))
    for f in order:
        if time.monotonic() - start >= budget:
            break
        for hi, side in enumerate(SIDES):
            if not eligible[f, hi]:
                continue
            m, d, ids, _ = models[hi]
            solved = np.flatnonzero(done[:, hi])
            nearest = int(solved[np.argmin(abs(solved - f))]) if len(solved) else None
            previous = q[nearest, hi] if nearest is not None else None
            try:
                result = ik.solve(m, d, ids, data["joints"][f, hi], previous)
                assert np.isfinite(result["qpos"]).all() and np.isfinite(result["wrist_pose"]).all()
                q[f, hi], wrist[f, hi] = result["qpos"], result["wrist_pose"]
                done[f, hi] = True
                rows.append(dict(frame=int(data["frames"][f]), side=side, **result["row"]))
            except (ValueError, AssertionError, np.linalg.LinAlgError) as error:
                rows.append(dict(frame=int(data["frames"][f]), side=side, ik_failed=str(error)))
            if len(rows) % 25 == 0:
                checkpoint()
    checkpoint()
    planned_q, planned_w = q.copy(), wrist.copy()
    source = np.full((T, 2), -1, dtype=np.int64)
    A = data["table_T_world"]
    retreat = []
    for hi, side in enumerate(SIDES):
        good = np.flatnonzero(done[:, hi])
        if len(good):
            nearest = good[np.argmin(abs(np.arange(T)[:, None] - good), axis=1)]
            planned_q[:, hi], planned_w[:, hi] = q[nearest, hi], wrist[nearest, hi]
            source[:, hi] = data["frames"][nearest]
            planned_w[:, hi, :3] = planned_w[:, hi, :3] @ A[:3, :3].T + A[:3, 3]
            R = Rotation.from_quat(planned_w[:, hi, [4, 5, 6, 3]]).as_matrix()
            planned_w[:, hi, 3:] = Rotation.from_matrix(A[:3, :3] @ R).as_quat(scalar_first=True)
        else:
            m = models[hi][0]
            planned_q[:, hi] = np.clip(np.full(22, .2), m.jnt_range[:, 0] + .01, m.jnt_range[:, 1] - .01)
            object_p = data["table_poses"][..., :3, 3]
            edge = float(object_p[..., 0].min() - data["radius"].max() - .25) if hi == 0 else float(object_p[..., 0].max() + data["radius"].max() + .25)
            fallback = np.array([edge, float(np.median(object_p[..., 1])), max(.35, float(object_p[..., 2].max() + .15)), 1., 0., 0., 0.])
            planned_w[:, hi] = fallback
            retreat.append(side)
    return planned_q, planned_w, source, done, models, dict(solved=int(done.sum()), eligible=int(eligible.sum()),
        declared_present_invalid_joints=int((data["present"] & ~eligible).sum()), budget_seconds=budget,
        elapsed_s=time.monotonic() - start, explicit_retreat_sides=retreat, morphology_scale=ik.MORPHOLOGY_SCALE,
        morphology_scale_is_not_scene_metric_scale=True, position_only=True, contact_optimized=False)


def build_reference(args, data, out):
    q, wrist, sources, done, models, stats = build_robot(data, out, args.ik_budget_seconds)
    T, B = data["T"], data["B"]
    P = data["table_poses"]
    objq = Rotation.from_matrix(P[..., :3, :3].reshape(-1, 3, 3)).as_quat(scalar_first=True).reshape(T, B, 4)
    zeros = np.zeros((T, 2, 16), dtype=np.int32)
    row = dict(fps=data["fps"], source_dataset=data["manifest"].get("source_dataset", "v2d_track3_scored_video_baseline"),
        sequence_id=f"episode_{args.episode_id:06d}", robot_name="sharpa_wave", coord_frame="table_Z_up_meters",
        object_name="__".join(data["names"]), object_body_names=data["names"],
        object_mesh_paths=[x["mesh_path"] for x in data["records"]], object_urdf_paths=[x["urdf_path"] for x in data["records"]],
        object_mesh_radius=data["radius"].tolist(), object_body_position=P[..., :3, 3].tolist(), object_body_wxyz=objq.tolist(),
        object_articulation=np.zeros((T, 1)).tolist(), observed_hand=data["present"].tolist(), planned_reference=(~done).tolist(),
        contact_ground_truth_known=zeros.tolist(), geometry_proxy_known=np.zeros((T, 2, B), bool).tolist(),
        interface_only_not_training_ready=False, quality_gate_pass=False, downstream_diagnostic_authorized=True,
        force_reference_is_measured=False, contact_reference_provenance="All contacts unknown. Zero-valued fields are interface placeholders; all contact rewards disabled.")
    fk = {}
    for hi, side in enumerate(SIDES):
        path = ASSETS / f"{side}_sharpawave.urdf"
        names = [x.get("name") for x in ET.parse(path).getroot().findall("link")]
        joints = models[hi][3]
        frames = urdf_frames(path, names, joints, q[:, hi], wrist[:, hi])
        fk[side + "_frames"] = frames
        fk[side + "_frame_names"] = np.asarray(names)
        row.update({f"robot_{side}_wrist_position": wrist[:, hi, :3].tolist(), f"robot_{side}_wrist_wxyz": wrist[:, hi, 3:].tolist(),
            f"robot_{side}_finger_joints": q[:, hi].tolist(), f"{side}_robot_finger_joint_names": joints,
            f"robot_{side}_frames": frames.tolist(), f"{side}_robot_frame_names": names,
            f"mano_{side}_object_contact_positions": np.zeros((T, 16, 3)).tolist(),
            f"mano_{side}_object_contact_normals": np.zeros((T, 16, 3)).tolist(),
            f"mano_{side}_object_contact_part_ids": zeros[:, hi].tolist()})
    parquet = out / "episode_reference.parquet"
    pq.write_table(pa.Table.from_pylist([row]), parquet)
    object_masks = {}
    for name in ("observed", "accepted", "confidence", "planned_reference"):
        if name in data["poses"]:
            values = np.asarray(data["poses"][name])
            assert values.shape == (T, B), f"Invalid object {name} mask"
            object_masks["object_" + name] = values
    save_npz(out / "reference_checkpoint.npz", frame_index=data["frames"], side_names=np.asarray(SIDES),
        object_names=np.asarray(data["names"]), table_T_world=data["table_T_world"], world_T_table=np.linalg.inv(data["table_T_world"]),
        observed_hand=data["present"], robot_ik_solved=done, planned_reference=~done, planned_source_frame=sources,
        planned_wrist_table=wrist, planned_finger_q=q, canonical_COM=data["canonical_COM"],
        object_bounding_radius_COM=data["radius"], object_link_pose_table=P,
        object_COM_table=P[..., :3, 3] + np.einsum("tbij,bj->tbi", P[..., :3, :3], data["canonical_COM"]),
        geometry_proxy_known=np.zeros((T, 2, B), bool), contact_ground_truth_known=zeros.astype(bool),
        contact_position_table_unknown_nan=np.full((T, 2, 16, 3), np.nan),
        contact_normal_table_unknown_nan=np.full((T, 2, 16, 3), np.nan), **object_masks, **fk)
    human_j, human_v = data["joints"].copy(), data["vertices"].copy()
    human_j[~data["present"]], human_v[~data["present"]] = np.nan, np.nan
    save_npz(out / "human_observations.npz", frame_index=data["frames"], present=data["present"], joints_world=human_j, vertices_world=human_v)
    return parquet, stats


def build_config(data, out, template, worlds):
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(template), resolve=True)
    cfg["task"].update(parquet=str(out / "episode_reference.parquet"), motion_speed=1., source_frame_playback=False, motion_start_frame=0, motion_end_frame=-1)
    for sim in (cfg["sim"], cfg["env"]["sim"]):
        sim["fps"] = data["fps"]
    cfg["scene"].update(world_count=worlds, object_scale_min=1., object_scale_max=1.)
    cfg["training"].update(world_count=worlds, total_environment_steps=50000, resume=False, checkpoint_path=None)
    cfg["training"]["checkpoint"].update(save_interval_environment_steps=50000, save_replay=True, load_replay_path=None)
    cfg["output_dir"] = str(out.parent / "training/checkpoints")
    cfg["logging"].update(mode="disabled", upload_checkpoints=False, upload_training_state=False)
    objective = cfg["env"]["objective"]
    objective.update(reference_known=np.zeros((data["T"], 2, data["B"]), np.int32).tolist(),
                     provenance="Scored-video baseline: all GT contacts unknown; contact objectives disabled; no measured force.",
                     body_com_local=data["canonical_COM"].tolist(), com_radius=data["radius"].tolist())
    cfg["objective"] = json.loads(json.dumps(objective["config"]))
    def disable(value):
        if isinstance(value, dict):
            for k, v in value.items():
                if k in CONTACT_TERMS:
                    if isinstance(v, dict):
                        v["weight"] = 0.
                        v["enabled"] = False
                    elif isinstance(v, (int, float)):
                        value[k] = 0.
                if k in ("voc_scale", "voc_scale_override", "max_force", "max_torque"):
                    value[k] = 0.
                disable(v)
        elif isinstance(value, list):
            for v in value:
                disable(v)
    disable(cfg)
    cfg["voc_scale_override"] = 0.
    OmegaConf.save(OmegaConf.create(cfg), out / "config.yaml")
    python = ROOT / ".deps/flash-chord-venv/bin/python"
    common = [str(python), str(ROOT / "scripts/train_track3_masked.py"), "--config", str(out / "config.yaml"),
              "--reference-mode", "ground_truth_unknown", "--reference-checkpoint", str(out / "reference_checkpoint.npz")]
    commands = dict(train_50000=shlex.join(common + ["--total-environment-steps", "50000"]),
                    resume_60000=shlex.join(common + ["--resume", "--resume-step", "50000", "--total-environment-steps", "60000"]))
    save_json(out / "training-commands.json", commands)
    return commands


def check_scene(out, data):
    # Import the official scene stack only when explicitly requested. It runs
    # on CPU; this is a hard interface check, never a policy rollout.
    import warp as wp
    import newton
    from omegaconf import OmegaConf
    from flash_chord.data.loader import load_reference
    from flash_chord.embodiments.sharpa_hands import SharpaHands
    from flash_chord.scene.builder import build_scene
    from flash_chord.scene.collision import CollisionPolicy
    from track3_reference_alignment import align_known
    ref = load_reference(out / "episode_reference.parquet")
    cp = npz_dict(out / "reference_checkpoint.npz")
    cfg = OmegaConf.load(out / "config.yaml")
    _, frames, known, _ = align_known(cfg, cp["contact_ground_truth_known"])
    assert ref.num_frames == data["T"] and len(ref.object_body_names()) == data["B"] and not known.any()
    wp.init()
    with wp.ScopedDevice("cpu"):
        scene = build_scene(SharpaHands(), ref, world_count=1, decompose_objects=False, contact_friction=.5,
                            collision=CollisionPolicy(robot_support_collision=False))
        model = scene.model
        s0, s1 = model.state(), model.state()
        q = model.joint_q.numpy().copy()
        q[:scene.layout.num_joint_q] = scene.robot_reference.joint_q[0]
        for i, binding in enumerate(scene.objects):
            ids = list(binding.root.free_q_ids)
            q[ids[:3]] = ref.object_body_pos_w()[0, i]
            q[ids[3:]] = ref.object_body_quat_w()[0, i][[1, 2, 3, 0]]
        s0.joint_q.assign(q)
        s0.joint_qd.zero_()
        newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
        bodyq = s0.body_q.numpy()
        coms = model.body_com.numpy()
        checks = []
        for i, binding in enumerate(scene.objects):
            bi = binding.root.body_id
            actual_com = bodyq[bi, :3] + Rotation.from_quat(bodyq[bi, 3:]).as_matrix() @ coms[bi]
            checks.append(dict(name=binding.name, link_position_error_m=float(np.linalg.norm(bodyq[bi, :3] - ref.object_body_pos_w()[0, i])),
                com_error_m=float(np.linalg.norm(actual_com - cp["object_COM_table"][0, i])),
                local_com_error_m=float(np.linalg.norm(coms[bi] - data["canonical_COM"][i]))))
        assert all(x["link_position_error_m"] < 1e-5 and x["com_error_m"] < 1e-5 and x["local_com_error_m"] < 1e-5 for x in checks), checks
        solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True, use_mujoco_contacts=True,
                    nconmax=4096, njmax=12000, save_to_mjcf=str(out / "scene_cpu.xml"))
        control = model.control()
        target = np.zeros(model.joint_dof_count, np.float32)
        target[:scene.layout.num_joint_dof] = scene.robot_reference.joint_target[0]
        control.joint_target_pos.assign(target)
        contacts = model.contacts()
        initialq = s0.joint_q.numpy().copy()
        for _ in range(5):
            s0.clear_forces()
            solver.step(s0, s1, control, contacts, .001)
            s0, s1 = s1, s0
        finalq, finalv = s0.joint_q.numpy(), s0.joint_qd.numpy()
        assert np.isfinite(finalq).all() and np.isfinite(finalv).all()
        save_npz(out / "scene_check_checkpoint.npz", initial_joint_q=initialq, final_joint_q=finalq, final_joint_qd=finalv,
                 robot_joint_q=scene.robot_reference.joint_q, robot_joint_target=scene.robot_reference.joint_target,
                 body_com=coms, mask_aligned=known, frames=frames)
        mujoco.mj_saveModel(solver.mj_model, str(out / "scene_cpu.mjb"))
    result = dict(state="completed", loader_pass=True, mask_alignment_pass=True, scene_compile_pass=True,
                  frames=ref.num_frames, fps=ref.fps, objects=ref.object_body_names(), device="cpu", physics_steps=5,
                  object_checks=checks, finite=True, object_assistance=False, policy_rollout_performed=False,
                  success_or_collision_quality_asserted=False)
    save_json(out / "scene-check.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--hands", type=Path, required=True)
    parser.add_argument("--object-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--ik-budget-seconds", type=float, default=120.)
    parser.add_argument("--world-count", type=int, default=8)
    parser.add_argument("--config-template", type=Path, default=TEMPLATE)
    parser.add_argument("--scene-check", action="store_true")
    args = parser.parse_args()
    assert args.ik_budget_seconds > 0 and args.world_count > 0
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".reference.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    data = read_inputs(args)
    source_paths = dict(script=Path(__file__), ik=Path(ik.__file__), fk=ROOT / "scripts/run_track3_fastlane_reference.py",
                        hands=args.hands.resolve(), manifest=args.object_manifest.resolve(), template=args.config_template.resolve(), **data["input_paths"])
    for side in SIDES:
        source_paths[f"mjcf_{side}"] = ik.ASSETS / f"{side}_sharpawave.xml"
        source_paths[f"urdf_{side}"] = ASSETS / f"{side}_sharpawave.urdf"
    if args.calibration:
        source_paths["calibration"] = args.calibration.resolve()
    hashes = {k: dict(path=str(p), sha256=digest(p)) for k, p in source_paths.items()}
    settings = dict(episode_id=args.episode_id, fps=data["fps"], ik_budget_seconds=args.ik_budget_seconds, world_count=args.world_count)
    key = hashlib.sha256(json.dumps(dict(hashes=hashes, settings=settings), sort_keys=True).encode()).hexdigest()
    rp = out / "report.json"
    previous = json.loads(rp.read_text()) if rp.exists() else {}
    if previous and previous.get("input_key") != key:
        raise RuntimeError("Existing reference inputs/code differ; choose a new output directory")
    if previous.get("state") == "completed" and (not args.scene_check or previous.get("scene_check", {}).get("state") == "completed"):
        print("Exact completed reference reused", flush=True)
        return
    report = dict(state="building", started_utc=now(), input_key=key, input_sha256=hashes, settings=settings,
                  episode_id=args.episode_id, frames=data["T"], object_names=data["names"], source_frames=data["frames"].tolist(),
                  fps=data["fps"], object_assistance=False, contact_ground_truth_known=0, geometry_proxy_known=0,
                  policy_rollout_performed=False, training_environment_steps=0, training_ready=False, quality_gate_pass=False,
                  reference_mode="ground_truth_unknown", checkpoint_interval=50000, total_environment_steps_plan=[50000, 60000])
    save_json(rp, report)
    try:
        parquet, stats = build_reference(args, data, out)
        report["ik"] = stats
        report["training_commands"] = build_config(data, out, args.config_template, args.world_count)
        report.update(state="reference_ready", parquet=str(parquet), elapsed_s=time.monotonic() - started)
        save_json(rp, report)
        if args.scene_check:
            report["scene_check"] = check_scene(out, data)
        report.update(state="completed", hard_format_ready=bool(args.scene_check), completed_utc=now(), elapsed_s=time.monotonic() - started)
    except Exception:
        report.update(state="failed", error=traceback.format_exc(), elapsed_s=time.monotonic() - started)
        save_json(rp, report)
        raise
    save_json(rp, report)
    artifacts = {name: dict(path=str(out / name), sha256=digest(out / name)) for name in
                 ("episode_reference.parquet", "reference_checkpoint.npz", "config.yaml", "human_observations.npz", "report.json")}
    save_json(out / "artifact-manifest.json", dict(state="completed", artifact_kind="reference_only", episode_id=args.episode_id,
        frames=data["T"], fps=data["fps"], object_names=data["names"], artifacts=artifacts,
        policy_rollout_performed=False, training_environment_steps=0, object_assistance=False,
        producer_source_sha256={str(p.relative_to(ROOT)): digest(p) for p in source_paths.values() if p.is_relative_to(ROOT) and p.suffix == ".py"}))
    print(json.dumps(dict(state=report["state"], episode_id=args.episode_id, frames=data["T"], objects=data["B"], output=str(out), elapsed_s=report["elapsed_s"])), flush=True)


if __name__ == "__main__":
    main()
