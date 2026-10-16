"""Bounded robot-only retreat for the initially unobserved right hand.

Human observations/unknown labels and all observed robot frames are immutable.
This is a candidate plan under an unvalidated finite-table model, not recovered
human motion or a repaired complete training reference.
"""
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import time
import traceback
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/track3/episode_000011'
PRE = BASE / 'flashchord_reference_preflight'
FINITE = BASE / 'scene_contact_attribution/finite_table_candidate'
OUT = BASE / 'initial_unobserved_robot_plan'


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def right_contacts(model, data):
    values = {k: 0. for k in ['table', 'object', 'other_hand', 'self', 'other']}
    pairs = []
    for c in data.contact[:data.ncon]:
        if c.dist >= 0:
            continue
        b = [int(model.geom_bodyid[g]) for g in [c.geom1, c.geom2]]
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or 'world' for i in b]
        sides = ['right' in name for name in names]
        if not any(sides):
            continue
        other = names[1 if sides[0] else 0]
        category = ('self' if all(sides) else 'other_hand' if 'left' in other else
                    'object' if 'cup' in other else 'table' if 0 in b else 'other')
        depth = float(-c.dist)
        values[category] = max(values[category], depth)
        pairs.append({'bodies': names, 'category': category, 'penetration_m': depth})
    return {'max_penetration_m': values,
            'target_max_m': max(values[k] for k in ['table', 'object', 'other_hand', 'other']),
            'pairs': sorted(pairs, key=lambda x: -x['penetration_m'])[:12]}


def main():
    OUT.mkdir(exist_ok=True)
    started = time.monotonic()
    sources = [Path(__file__), PRE / 'reference_checkpoint.npz',
               BASE / 'sharpa_keypoint_initializer_v2/report.json',
               BASE / 'agent_cup_pose_supported/candidate_poses.npz', FINITE / 'finite_scene.xml']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    key = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    reportfile = OUT / 'report.json'
    if reportfile.exists():
        old = json.loads(reportfile.read_text())
        if old.get('state') == 'completed' and old.get('source_hash') == key:
            print('Reused completed matching robot-only plan.'); return
        save(OUT / ('previous-' + str(time.time_ns()) + '.json'), old)
    report = {'state': 'running', 'started_utc': now(), 'source_hash': key,
              'source_sha256': hashes, 'budget_seconds': 180, 'training_ready': False,
              'automatic_promotion': False, 'human_motion_inferred': False,
              'scope': 'Only robot right-wrist translation in unobserved frames before first observation',
              'table_model_validated': False, 'trials': []}
    save(reportfile, report)
    try:
        cp = np.load(PRE / 'reference_checkpoint.npz')
        observed = cp['observed_hand']
        first = int(np.flatnonzero(observed[:, 1])[0])
        assert first == 5 and not observed[:first, 1].any()
        wrist = cp['planned_wrist_table'].copy()
        fingers = cp['planned_finger_q']
        names = json.loads((BASE / 'sharpa_keypoint_initializer_v2/report.json').read_text())['joint_names']
        poses = cp['table_T_world'] @ np.load(BASE / 'agent_cup_pose_supported/candidate_poses.npz')['world_T_canonical_refined']
        tree = ET.parse(FINITE / 'finite_scene.xml')
        for element in tree.iter():
            filename = element.get('file')
            if filename and not Path(filename).is_absolute():
                element.set('file', str((FINITE / filename).resolve()))
        table = tree.find('.//geom[@name="ground_plane_240"]')
        assert table is not None
        size = np.fromstring(table.get('size'), sep=' ')
        models = []
        for delta in [-.020, 0., .020]:
            table.set('size', f'{size[0]+delta:.12g} {size[1]:.12g}')
            path = OUT / f'finite_radius_delta_{int(delta*1000):+d}mm.xml'
            tree.write(path)
            model = mujoco.MjModel.from_xml_path(str(path))
            model.opt.timestep = .001
            models.append((delta, model, mujoco.MjData(model)))

        def joint_id(model, name):
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j < 0:
                raise ValueError('Missing scene joint: ' + name)
            return j

        model = models[0][1]
        qframes = np.tile(model.qpos0, (len(wrist), 1))
        right_xyz = [int(model.jnt_qposadr[joint_id(model, 'right_wrist_' + axis)]) for axis in 'xyz']
        for hi, side in enumerate(['left', 'right']):
            for ai, axis in enumerate('xyz'):
                qframes[:, model.jnt_qposadr[joint_id(model, side + '_wrist_' + axis)]] = wrist[:, hi, ai]
            address = model.jnt_qposadr[joint_id(model, side + '_wrist_rotvec')]
            qframes[:, address:address+4] = wrist[:, hi, 3:]
            for fi, name in enumerate(names[side]):
                address = model.jnt_qposadr[joint_id(model, side + '_sharpa_wave_' + name)]
                qframes[:, address] = fingers[:, hi, fi]
        for oi, name in enumerate(['blue_cup', 'beige_cup']):
            address = model.jnt_qposadr[joint_id(model, name + '_floating_base')]
            qframes[:, address:address+3] = poses[:, oi, :3, 3]
            qframes[:, address+3:address+7] = Rotation.from_matrix(poses[:, oi, :3, :3]).as_quat(scalar_first=True)
        # Zero derivative at the first observed frame; observation itself is immutable.
        t = 1. - np.arange(first) / first
        weights = 10*t**3 - 15*t**4 + 6*t**5

        def check(offset, details=False, frames=None):
            maximum = 0.; rows = []
            for delta, m, d in models:
                for frame in (range(first) if frames is None else frames):
                    mujoco.mj_resetData(m, d)
                    d.qpos[:] = qframes[frame]
                    if frame < first:
                        d.qpos[right_xyz] += offset * weights[frame]
                    mujoco.mj_forward(m, d)
                    contacts = right_contacts(m, d)
                    maximum = max(maximum, contacts['target_max_m'])
                    if details:
                        rows.append({'frame': frame, 'radius_delta_m': delta, **contacts})
            return maximum, rows

        baseline, baseline_rows = check(np.zeros(3), True)
        report['baseline'] = {'target_max_m': baseline, 'rows': baseline_rows}
        _, boundary_rows = check(np.zeros(3), True, [first])
        report['first_observed_frame_unchanged_contacts'] = boundary_rows
        candidates = [np.array(v) for v in itertools.product(
            [0, -.01, .01, -.02, .02, -.04, .04, -.06, .06],
            [0, -.01, .01, -.02, .02, -.04, .04], [0, .01, .02, .03, .04, .06, .08])]
        candidates.sort(key=lambda v: (float(v @ v), float(v[2])))
        schedules = [('quintic_five_frame_taper', weights.copy()),
                     ('linear_last_two_intervals', np.minimum(1., (first-np.arange(first))/2)),
                     ('constant_initial_retreat', np.ones(first))]
        selected = np.zeros(3); best = baseline; feasible = False
        selected_weights = weights.copy(); selected_schedule = schedules[0][0]
        for schedule, schedule_weights in schedules:
            weights = schedule_weights
            for offset in candidates:
                if time.monotonic() - started > 160:
                    break
                proposed = wrist[:first+1, 1, :3].copy()
                proposed[:first] += weights[:, None] * offset
                if np.linalg.norm(np.diff(proposed, axis=0), axis=1).max()*20 > 1.:
                    continue
                maximum, _ = check(offset)
                report['trials'].append({'schedule': schedule, 'offset_m': offset.tolist(), 'target_max_m': maximum})
                if maximum < best:
                    selected, best = offset, maximum
                    selected_weights, selected_schedule = weights.copy(), schedule
                if maximum <= .001:
                    selected, best, feasible = offset, maximum, True
                    selected_weights, selected_schedule = weights.copy(), schedule
                    break
                if len(report['trials']) % 32 == 0:
                    save(reportfile, report)
            if feasible or time.monotonic()-started > 160:
                break
        weights = selected_weights
        candidate = wrist.copy()
        candidate[:first, 1, :3] += weights[:, None] * selected
        chosen, chosen_rows = check(selected, True)
        speed = np.linalg.norm(np.diff(candidate[:, 1, :3], axis=0), axis=1) * 20.
        oldspeed = np.linalg.norm(np.diff(wrist[:, 1, :3], axis=0), axis=1) * 20.
        physics = []
        for delta, m, d in models:
            for frame in range(first):
                mujoco.mj_resetData(m, d); d.qpos[:] = qframes[frame]
                d.qpos[right_xyz] += weights[frame] * selected
                mujoco.mj_forward(m, d)
                # Hold each native servo at its exact initial transmission length.
                d.ctrl[:] = np.where(np.abs(m.actuator_biasprm[:, 1]) > 0, d.actuator_length, 0.)
                peak = 0.
                for _ in range(20):
                    mujoco.mj_step(m, d)
                    peak = max(peak, right_contacts(m, d)['target_max_m'])
                physics.append({'frame': frame, 'radius_delta_m': delta,
                                'finite_state': bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()),
                                'peak_target_penetration_over_20ms_m': peak})
        np.savez_compressed(OUT / 'checkpoint.npz', planned_wrist_table=candidate,
                            planned_finger_q=fingers, observed_hand=observed,
                            original_observed_wrist_world=cp['observed_wrist_world'],
                            contact_ground_truth_known=cp['contact_ground_truth_known'],
                            translation_offset_m=selected, taper_weights=weights,
                            source_hash=np.array(key))
        report.update(state='completed', completed_utc=now(), elapsed_seconds=time.monotonic()-started,
                      first_observed_frame=first, changed_frames=list(range(first)),
                      selected_schedule=selected_schedule, selected_offset_m=selected.tolist(), selected_static_max_m=chosen,
                      selected_rows=chosen_rows, physics_smoke=physics,
                      actual_native_physics_steps=len(physics)*20,
                      initial_segment_max_speed_mps=float(speed[:first].max()),
                      baseline_initial_segment_max_speed_mps=float(oldspeed[:first].max()),
                      full_sequence_max_speed_mps=float(speed.max()),
                      baseline_full_sequence_max_speed_mps=float(oldspeed.max()),
                      gates={'static_target_penetration_under_1mm_all_radius_variants': feasible,
                             'native_smoke_finite': all(v['finite_state'] for v in physics),
                             'native_smoke_target_penetration_under_1mm': all(v['peak_target_penetration_over_20ms_m']<=.001 for v in physics),
                             'observed_robot_wrist_unchanged': bool(np.array_equal(candidate[observed], wrist[observed])),
                             'other_robot_frames_unchanged': bool(np.array_equal(candidate[first:], wrist[first:])),
                             'initial_speed_under_assumed_1mps': bool(speed[:first].max()<=1.),
                             'complete_geometry_reference_pass': False, 'training_ready': False})
        save(reportfile, report)
        (OUT / 'README.md').write_text(
            '# Initial unobserved right-hand robot plan\n\n'
            'Only robot wrist translation at frames 0–4 is changed; all human arrays, '
            'missing NaNs, unknown contact labels and observed robot frames are preserved. '
            'This is an explicit robot plan, not inferred human inactivity/motion.\n\n'
            f'Selected offset: {selected.tolist()} m; schedule {selected_schedule}, zero at frame 5. '
            f'Static target penetration across radius ±20 mm: {baseline*1000:.3f} → {chosen*1000:.3f} mm. '
            f'Native MuJoCo steps: {len(physics)*20}; see report for dynamic residuals.\n\n'
            'The finite table is still unvalidated. Observed-hand penetration and cup-cup '
            'intersection remain separate failed gates. No production reference or policy '
            'has been replaced. The 1 m/s planning bound is an engineering assumption. '
            'All 137-frame baseline velocity outliers are retained and reported, not silently repaired.\n')
        print(json.dumps({k: report[k] for k in ['state','elapsed_seconds','selected_offset_m','selected_static_max_m','gates']}, indent=2), flush=True)
    except Exception:
        report.update(state='failed', error=traceback.format_exc(), completed_utc=now())
        save(reportfile, report)
        raise
    finally:
        jobfile = ROOT / 'notes/track3-initial-retreat-job.json'
        if jobfile.exists():
            job = json.loads(jobfile.read_text())
            job.update(state=report['state'], updated_utc=now(), training_ready=False)
            save(jobfile, job)
        livefile = ROOT / 'notes/track3-live-progress.json'
        live = json.loads(livefile.read_text())
        live['initial_unobserved_robot_plan'] = {
            'state': report['state'], 'report': str(reportfile.relative_to(ROOT)),
            'production_integrated': False, 'training_ready': False,
        }
        save(livefile, live)
        save(ROOT / 'notes/track3-publication-request.json', {
            'requested_utc': now(), 'reason': 'Robot-only initial unknown-hand plan result',
            'report': str(reportfile.relative_to(ROOT)),
        })


if __name__ == '__main__':
    main()
