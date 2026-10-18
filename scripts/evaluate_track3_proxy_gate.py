#!/usr/bin/env python3
"""Evaluate real Track 3 rollouts against their own reconstructed references.

This is an offline proxy, never a Kaggle/GT evaluator. The fixed-initial-pose
comparator exists only in memory: this program does not export pose trajectories.
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np
import pyarrow.parquet as pq
import trimesh

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COHORT = ROOT / "outputs/track3/scored_baseline_20261017/unified-gpu-repair-v1"
DEFAULT_OUTPUT = ROOT / "outputs/track3/advice_restart_20261017/proxy-baseline"
SCORERS = ROOT / "toolkit/8d2bce64/v2d_submission_kit/metric_code/track_3"
EPISODES = [1, 3, 5, 6, 8, 10, 14, 15, 16, 19, 20, 25, 28, 30, 32, 33, 35, 36, 37, 38]
METRICS = ["add_auc", "spider_sr", "maniptrans_sr", "rpe_cm", "mppe_cm"]
SHARED = ["_register_object_body", "_convert_object_body", "_align_registered_scene",
          "_registered_pose_metric", "_symmetry_point_errors", "_symmetry_orientation_errors",
          "_symmetry_relative_position_cm", "_add_auc"]


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def load_scorer():
    spec = importlib.util.spec_from_file_location("official_track3_auc_proxy", SCORERS / "AUC.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_audit():
    signatures = {}
    for name in ["AUC", "SP-SR", "MP-SR", "RPE", "MPPE"]:
        tree = ast.parse((SCORERS / (name + ".py")).read_text())
        signatures[name] = {f.name: hashlib.sha256(ast.dump(f, include_attributes=False).encode()).hexdigest()
                            for f in tree.body if isinstance(f, ast.FunctionDef) and f.name in SHARED}
    assert all(signatures[n] == signatures["AUC"] for n in signatures), "official metric cores differ"
    assert len(signatures["AUC"]) == len(SHARED)
    return dict(five_scorer_shared_functions_ast_identical=True,
                shared_function_ast_sha256=signatures["AUC"],
                official_source_sha256={n.name: sha(n) for n in sorted(SCORERS.glob("*.py"))})


def self_test(m):
    from scipy.spatial.transform import Rotation
    ref = np.zeros((4, 1, 2, 7)); ref[..., 6] = 1
    ref[:, 0, 1, 0] = .2
    points = [np.eye(3) * .03] * 2
    axes, orders = [np.zeros(3)] * 2, [1, 1]
    expected = dict(add_auc=1., spider_sr=1., maniptrans_sr=1., rpe_cm=0., mppe_cm=0.)
    for name in METRICS:
        assert abs(m._registered_pose_metric(name, ref, ref, points, axes, orders) - expected[name]) < 1e-9
    q = Rotation.from_rotvec([.2, -.3, .4]).as_quat(); r = Rotation.from_quat(q).as_matrix()
    moved = ref.copy(); moved[..., :3] = ref[..., :3] @ r.T + [.3, -.2, .5]
    moved[..., 3:] = m._quaternion_multiply(q, ref[..., 3:])
    aligned = m._align_registered_scene(moved, ref, axes)
    for name in METRICS:
        assert abs(m._registered_pose_metric(name, aligned, ref, points, axes, orders) - expected[name]) < 1e-8
    displaced = ref.copy(); displaced[:, :, 1, 0] += .031
    assert m._registered_pose_metric("maniptrans_sr", displaced, ref, points, axes, orders) == 0
    return dict(identity_metrics_pass=True, rigid_scene_alignment_invariance_pass=True,
                maniptrans_per_object_threshold_pass=True)


def poses(frame, prefix, T, B):
    cols = [prefix + "position_" + x for x in "xyz"] + [prefix + "quaternion_" + x for x in "xyzw"]
    result = frame[cols].to_numpy(dtype=np.float64).reshape(T, 1, B, 7)
    assert np.isfinite(result).all(), "non-finite pose"
    assert np.max(np.abs(np.linalg.norm(result[..., 3:], axis=-1) - 1)) < 1e-4, "non-unit quaternion"
    result[..., 3:] /= np.linalg.norm(result[..., 3:], axis=-1, keepdims=True)
    return result


def mesh_stats(path, packed_mesh, episode, name, slot, m):
    mesh = trimesh.load(path, force="mesh", process=False)
    assert len(mesh.vertices) and len(mesh.faces) and np.isfinite(mesh.vertices).all()
    extents = np.asarray(mesh.extents)
    v, f = packed_mesh
    used = np.unique(f[np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1) > 1e-12])
    center, radius = m._surface_moments(v, f)
    return dict(episode=episode, object_slot=slot, object_name=name, path=str(path), sha256=sha(path),
                raw_extents_m=extents.tolist(), max_extent_m=float(extents.max()),
                exceeds_half_meter=bool(extents.max() > .5), vertices=len(mesh.vertices), faces=len(mesh.faces),
                watertight=bool(mesh.is_watertight), winding_consistent=bool(mesh.is_winding_consistent),
                packed_used_extents_m=np.ptp(v[used], axis=0).tolist(),
                packed_storage_vertices=len(v), packed_storage_faces=len(f), packed_used_vertices=len(used),
                proxy_surface_centroid_m=center.tolist(), proxy_surface_rms_radius_m=radius)


def drift_stats(achieved, reference, fps, names):
    """Raw link-origin motion; no registration/alignment can hide initial bursts."""
    rows = []
    elapsed = (np.arange(len(achieved)) + 1) / fps
    early = elapsed <= .15 + 1e-10
    review_frame = min(int(round(.15 * fps)), len(achieved) - 1)
    for b, name in enumerate(names):
        a, r = achieved[:, 0, b, :3], reference[:, 0, b, :3]
        d = np.linalg.norm(a - r[0], axis=-1)
        from_first = np.linalg.norm(a - a[0], axis=-1)
        tracking = np.linalg.norm(a - r, axis=-1)
        idx = int(np.argmax(d))
        rows.append(dict(object_name=name, object_slot=b,
                         max_displacement_from_reference_initial_first_015s_m=float(d[early].max()),
                         displacement_at_015s_from_reference_initial_m=float(d[np.flatnonzero(early)[-1]]),
                         first_export_to_label015s_displacement_m=float(from_first[review_frame]),
                         max_displacement_from_reference_initial_m=float(d[idx]),
                         max_displacement_recorded_frame=idx, max_displacement_physics_elapsed_s=float(elapsed[idx]),
                         max_displacement_from_first_export_m=float(from_first.max()),
                         mean_position_tracking_error_m=float(tracking.mean()),
                         mean_reference_displacement_from_initial_m=float(np.linalg.norm(r-r[0], axis=-1).mean()),
                         max_reference_displacement_from_initial_m=float(np.linalg.norm(r-r[0], axis=-1).max()),
                         first_frame_exceeding_10cm_from_initial=(int(np.flatnonzero(d > .1)[0]) if np.any(d > .1) else None)))
    return rows


def metric_set(m, achieved, reference, points, axes, orders):
    aligned = m._align_registered_scene(achieved, reference, axes)
    result = {k: m._registered_pose_metric(k, aligned, reference, points, axes, orders) for k in METRICS}
    result["cd_o_cm"] = None
    return result


def evaluate_episode(args, m, packed, contract, manifest, episode, fingerprint):
    entry = manifest["episodes"][str(episode)]
    assert entry["policy_rollout_performed"] is True and entry["object_assistance"] is False
    path = args.cohort / entry["trajectory"]
    table = pq.read_table(path); frame = table.to_pandas().sort_values(["step", "environment_id", "object_body_id"])
    c = contract["episodes"][str(episode)]
    names = c["objects_in_slot_order"]; T, B = c["scored_frames"], len(names)
    assert len(frame) == c["rows"] == T * B
    assert frame["environment_id"].unique().tolist() == [0]
    assert frame["environment_valid"].all()
    assert not frame.duplicated(["step", "environment_id", "object_body_id"]).any()
    assert sorted(frame.step.unique()) == list(range(T))
    assert sorted(frame.reference_frame.unique()) == list(range(T))
    assert frame.object_body_name.to_numpy().reshape(T, B)[0].tolist() == names
    assert (frame.object_body_name.to_numpy().reshape(T, B) == np.array(names)).all()
    meta = table.schema.metadata
    fps = float(meta[b"flash_chord.control_fps"])
    np.testing.assert_allclose(frame.time_s.to_numpy().reshape(T, B)[:, 0], np.arange(T) / fps, atol=1e-9)
    achieved, reference = poses(frame, "achieved_", T, B), poses(frame, "reference_", T, B)
    raw_achieved = frame[["achieved_position_"+x for x in "xyz"] + ["achieved_quaternion_"+x for x in "xyzw"]].to_numpy()
    pg = packed[packed.row_id.str.startswith(f"track_3/e{episode:06d}/")].copy()
    indices = pg.row_id.str.extract(r"/w(\d+)/f(\d+)/o(\d+)$").astype(int)
    pg["world"], pg["frame"], pg["slot"] = indices[0].to_numpy(), indices[1].to_numpy(), indices[2].to_numpy()
    pg = pg.sort_values(["frame", "world", "slot"])
    assert len(pg) == len(frame)
    np.testing.assert_array_equal(pg[["pos_"+x for x in "xyz"] + ["quat_"+x for x in "xyzw"]].to_numpy(), raw_achieved)
    # No resampling or time shift: use exactly the scored row grid.
    registered = achieved.copy(); target = reference.copy()
    points, axes, orders, meshes, registrations = [], [], [], [], []
    for slot, name in enumerate(names):
        payload = pg[pg.slot == slot].object_mesh_b64.fillna("").astype(str)
        assert (payload.str.len() > 0).sum() == 1 and len(payload.iloc[0]) > 0
        candidate = m._decode_object_mesh(payload.iloc[0])
        anchor, _ = m._surface_moments(*candidate)
        seed = m._object_seed(episode, slot)
        ca, rotation = m._register_object_body(candidate, candidate, anchor, seed)
        assert np.linalg.norm(ca - anchor) < 1e-7 and np.linalg.norm(rotation - np.eye(3)) < 1e-7, "self registration is not identity"
        registered[:, :, slot] = m._convert_object_body(achieved[:, :, slot], ca, rotation)
        target[:, :, slot, :3] += np.einsum("...ij,j->...i", m._rotation_matrix(reference[:, :, slot, 3:]), anchor)
        # The confidential host's points/anchor/axis are unavailable. These are proxy values.
        points.append(m._sample_mesh_surface(*candidate, args.surface_samples, seed + 71) - anchor)
        order = 360 if name in ("blue_cup", "beige_cup") else 2 if name == "white_pot_lid" else 1
        axis = m._rotation_matrix(reference[0, 0, slot, 3:]).T @ np.array([0., 0., 1.]) if order > 1 else np.zeros(3)
        axes.append(axis); orders.append(order)
        mesh_path = args.cohort / entry["meshes"][name]["path"]
        meshes.append(mesh_stats(mesh_path, candidate, episode, name, slot, m))
        registrations.append(dict(object_name=name, candidate_anchor_m=ca.tolist(), body_rotation=rotation.tolist(),
                                  anchor_source="self packed mesh surface centroid, not host anchor",
                                  symmetry_axis_canonical=axis.tolist(), symmetry_order=order,
                                  symmetry_axis_source="initial reconstructed pose inverse maps table +Z; assumes upright initial cup/lid" if order > 1 else "asymmetric declared object",
                                  registration_is_self_identity=True))
    # Analysis only. Never save these pose arrays, including as parquet/npz.
    stationary = np.broadcast_to(target[:1], target.shape).copy()
    policy = metric_set(m, registered, target, points, axes, orders)
    fixed = metric_set(m, stationary, target, points, axes, orders)
    no_axes, no_orders = [np.zeros(3) for _ in names], [1] * B
    policy_no_sym = metric_set(m, registered, target, points, no_axes, no_orders)
    fixed_no_sym = metric_set(m, stationary, target, points, no_axes, no_orders)
    old_report_path = path.with_suffix(".json")
    old = json.loads(old_report_path.read_text()) if old_report_path.exists() else {}
    drift = drift_stats(achieved, reference, fps, names)
    return dict(episode=episode, fingerprint=fingerprint, completed_utc=utc(), frames=T, rows=T*B,
                objects=names, control_fps=fps, actual_rollout_path=str(path), actual_rollout_sha256=sha(path),
                policy_rollout_performed=True, assistance=False, packed_pose_bytes_match_real_rollout=True,
                metrics_proxy=policy, stationary_initial_pose_analysis=fixed,
                metrics_no_symmetry_sensitivity=policy_no_sym,
                stationary_no_symmetry_sensitivity=fixed_no_sym,
                raw_link_origin_motion=drift, mesh_statistics=meshes, proxy_registration=registrations,
                old_flash_chord_metrics=old.get("metrics"),
                gate=dict(beats_stationary_mppe=policy["mppe_cm"] < fixed["mppe_cm"],
                          beats_stationary_auc=policy["add_auc"] > fixed["add_auc"],
                          initial_burst_over_10cm=any(x["max_displacement_from_reference_initial_first_015s_m"] > .1 for x in drift),
                          oversized_mesh_diagnostic=any(x["exceeds_half_meter"] for x in meshes)))


def write_summary(args, report):
    episodes = report["episodes"]
    if not episodes:
        return
    groups = ["metrics_proxy", "stationary_initial_pose_analysis", "metrics_no_symmetry_sensitivity", "stationary_no_symmetry_sensitivity"]
    report["means_equal_weight_per_episode"] = {g: {**{k: float(np.mean([e[g][k] for e in episodes])) for k in METRICS}, "cd_o_cm": None} for g in groups}
    meshes = [m for e in episodes for m in e["mesh_statistics"]]
    report["mesh_summary"] = dict(count=len(meshes), unique_object_names=len(set(m["object_name"] for m in meshes)),
                                 over_half_meter_count=sum(m["exceeds_half_meter"] for m in meshes),
                                 non_watertight_count=sum(not m["watertight"] for m in meshes),
                                 over_half_meter=[m for m in meshes if m["exceeds_half_meter"]])
    means = report["means_equal_weight_per_episode"]
    burst_episodes = [e["episode"] for e in episodes if any(
        x["max_displacement_from_reference_initial_first_015s_m"] > args.max_initial_burst_m
        for x in e["raw_link_origin_motion"])]
    checks = dict(all_20_real_rollouts_checked=sorted(e["episode"] for e in episodes) == EPISODES,
                  policy_mean_mppe_beats_fixed_initial_pose=means["metrics_proxy"]["mppe_cm"] < means["stationary_initial_pose_analysis"]["mppe_cm"],
                  policy_mean_auc_beats_fixed_initial_pose=means["metrics_proxy"]["add_auc"] > means["stationary_initial_pose_analysis"]["add_auc"],
                  no_initial_burst_above_threshold=not burst_episodes)
    report["gate"] = dict(passed=all(checks.values()), checks=checks,
                          blocker_reasons=[k for k, value in checks.items() if not value],
                          initial_burst_threshold_m=args.max_initial_burst_m,
                          approved_to_submit=False, reason="proxy gate is a necessary diagnostic only; full submission approval also needs geometry and physical stability review",
                          policy_mppe_beats_stationary_episodes=sum(e["gate"]["beats_stationary_mppe"] for e in episodes),
                          policy_auc_beats_stationary_episodes=sum(e["gate"]["beats_stationary_auc"] for e in episodes),
                          early_burst_above_threshold_episodes=burst_episodes)
    save(args.output / "report.json", report)
    with (args.output / "episode-metrics.csv").open("w", newline="") as f:
        keys = ["episode", "frames", "rows"] + [g+"."+k for g in groups for k in METRICS]
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader()
        for e in episodes:
            writer.writerow({**{k: e[k] for k in keys[:3]}, **{g+"."+k:e[g][k] for g in groups for k in METRICS}})
    with (args.output / "object-motion.csv").open("w", newline="") as f:
        rows = [dict(episode=e["episode"], **row) for e in episodes for row in e["raw_link_origin_motion"]]
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    save(args.output / "mesh-statistics.json", report["mesh_summary"] | {"all_meshes": meshes})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    p.add_argument("--packed", type=Path)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--surface-samples", type=int, default=512)
    p.add_argument("--wall-budget-seconds", type=int, default=1800)
    p.add_argument("--episode-ids", type=int, nargs="*", default=EPISODES)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--max-initial-burst-m", type=float, default=.1,
                   help="Diagnostic maximum motion from reference initial pose in physical first 0.15s")
    p.add_argument("--fail-on-proxy-gate", action="store_true", help="Exit 2 after completed evaluation when diagnostic gate fails")
    args = p.parse_args()
    m = load_scorer(); audit = source_audit(); tests = self_test(m)
    if args.self_test:
        print(json.dumps(dict(audit=audit, tests=tests), indent=2)); return 0
    assert args.surface_samples > 0 and args.wall_budget_seconds > 0 and args.max_initial_burst_m > 0
    args.cohort, args.output = args.cohort.resolve(), args.output.resolve()
    if args.packed is None:
        args.packed = args.cohort / "submission/attempt_6ab91580f45209a0/track3_baseline.parquet"
    args.packed = args.packed.resolve()
    manifest_path = args.cohort / "artifact-manifest.json"
    contract_path = ROOT / "notes/track3-submission-contract.json"
    manifest, contract = json.loads(manifest_path.read_text()), json.loads(contract_path.read_text())
    packed = pq.read_table(args.packed).to_pandas()
    inputs = {str(x): sha(x) for x in [Path(__file__), manifest_path, contract_path, args.packed]}
    fingerprint = hashlib.sha256(json.dumps(dict(inputs=inputs, audit=audit, samples=args.surface_samples), sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(schema="track3_self_reference_proxy_gate_v1", state="running", started_utc=utc(),
                  pid=os.getpid(), cohort=str(args.cohort), packed_submission=str(args.packed), input_sha256=inputs,
                  source_audit=audit, self_tests=tests, fingerprint=fingerprint,
                  surface_samples_per_object=args.surface_samples, episodes=[], wall_budget_seconds=args.wall_budget_seconds,
                  is_kaggle_score=False, uses_ground_truth=False, stationary_pose_artifacts_exported=False,
                  limitations=["Self-reference proxy: reconstruction errors shared with rollout are invisible.",
                    "Official host reference geometry is unavailable. Self mesh surface centroid, sampled points and inferred symmetry axes replace its hidden values.",
                    "Cup/lid canonical axes assume upright frame0 relative to reconstructed table; no-symmetry sensitivity is reported separately.",
                    "CD-O is not evaluated: a self-versus-self mesh comparison has no geometric accuracy value.",
                    "Fixed-initial-pose comparator is analysis, not a realizable controller nor a mathematical bound on all policies.",
                    "Mean metrics give each episode equal weight; RPE single-object episodes contribute zero.",
                    "First recorded pose is post one control step although time_s starts at zero. Early window uses physics elapsed=(step+1)/fps.",
                    "Observed rollout displacement does not establish whether collision, actuation, gravity or reset is its cause."],
                  timing_definition=dict(early_window_s=.15, physical_elapsed="(recorded_step+1)/control_fps",
                                         review_compatible="first exported pose to pose labeled time_s=0.15; separate diagnostic"))
    save(args.output / "report.json", report)
    save(args.output / "job.json", dict(pid=os.getpid(), started_utc=utc(), argv=sys.argv,
                                       resume_argv=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]] + ([] if args.resume else ["--resume"])))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("CPU wall budget reached; rerun --resume")))
    signal.alarm(args.wall_budget_seconds)
    start = time.monotonic()
    try:
        for episode in args.episode_ids:
            cache = args.output / f"episodes/episode_{episode:06d}.json"
            cached = json.loads(cache.read_text()) if args.resume and cache.exists() else None
            if (cached and cached.get("fingerprint") == fingerprint
                    and cached.get("actual_rollout_sha256") == sha(args.cohort / manifest["episodes"][str(episode)]["trajectory"])
                    and all(sha(x["path"]) == x["sha256"] for x in cached.get("mesh_statistics", []))):
                result = cached
                print(f"episode {episode}: verified cache", flush=True)
            else:
                print(f"episode {episode}: computing official formula proxy", flush=True)
                result = evaluate_episode(args, m, packed, contract, manifest, episode, fingerprint)
                save(cache, result)
            report["episodes"].append(result)
            report["elapsed_seconds"] = time.monotonic() - start
            write_summary(args, report)
            print(json.dumps(dict(episode=episode, policy=result["metrics_proxy"], stationary=result["stationary_initial_pose_analysis"], elapsed_s=report["elapsed_seconds"])), flush=True)
        report.update(state="completed", completed_utc=utc(), all_20_episodes=(sorted(args.episode_ids) == EPISODES))
        write_summary(args, report)
        return 2 if args.fail_on_proxy_gate and not report["gate"]["passed"] else 0
    except Exception as error:
        report.update(state="budget_exhausted" if isinstance(error, TimeoutError) else "failed", error=str(error), traceback=traceback.format_exc(), stopped_utc=utc())
        save(args.output / "report.json", report)
        traceback.print_exc()
        return 3
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    raise SystemExit(main())
