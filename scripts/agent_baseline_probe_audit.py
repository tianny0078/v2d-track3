"""Offline audit of real checkpoints and policy trajectories from the baseline probe."""
from pathlib import Path
import hashlib
import json
import shlex
import datetime as dt
import xml.etree.ElementTree as ET

import numpy as np
from safetensors import safe_open
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/track3/episode_000011/baseline_probe'
JOB = ROOT / 'notes/track3-baseline-probe-job.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint(path):
    with safe_open(str(path), framework='numpy') as handle:
        metadata = handle.metadata() or {}
        arrays = {key: handle.get_tensor(key) for key in handle.keys()}
    user = {key.removeprefix('user.'): value for key, value in metadata.items() if key.startswith('user.')}
    keys = ['algorithm', 'checkpoint_kind', 'environment_steps', 'global_update_step',
            'checkpoint_set_id', 'voc_scale', 'replay_included', 'curriculum_stage']
    summary = {'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size,
               'array_count': len(arrays), 'all_arrays_finite': all(np.isfinite(a).all() for a in arrays.values()),
               'metadata': {key: user[key] for key in keys if key in user}}
    return summary, arrays


def main():
    report = json.loads((OUT / 'report.json').read_text())
    if report['state'] == 'running':
        raise SystemExit('Worker is still running; do not race report writes.')
    audit = {'updated_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
             'scope': 'offline verification of diagnostic baseline artifacts', 'checkpoints': [],
             'source_sha256': {str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))}}
    arrays_by_name = {}
    for path in sorted((OUT / 'checkpoints').glob('*.safetensors')):
        summary, arrays = checkpoint(path)
        audit['checkpoints'].append(summary)
        if path.name.startswith('policy_'):
            arrays_by_name[path.stem] = arrays
    audit['actor_state_changes'] = []
    for a, b in [('policy_16', 'policy_64'), ('policy_64', 'policy_80')]:
        if a not in arrays_by_name or b not in arrays_by_name:
            continue
        left, right = arrays_by_name[a], arrays_by_name[b]
        squared = [np.square(left[k].astype(np.float64) - right[k].astype(np.float64)).sum() for k in left]
        audit['actor_state_changes'].append({'from': a, 'to': b,
                                             'changed_tensor_count': sum(not np.array_equal(left[k], right[k]) for k in left),
                                             'state_l2_difference': float(np.sqrt(sum(squared)))})
    for step in (64, 80):
        records = [r for r in audit['checkpoints'] if r['metadata'].get('environment_steps') == str(step)]
        audit[f'checkpoint_set_{step}_matched'] = len(records) == 3 and len({r['metadata']['checkpoint_set_id'] for r in records}) == 1
    actor = OUT / 'checkpoints/policy_80.safetensors'
    if actor.exists():
        with safe_open(str(actor), framework='numpy') as handle:
            config = json.loads(handle.metadata()['user.resolved_config_json'])
        xml = ROOT / 'outputs/track3/episode_000011/flashchord_reference_preflight/official_scene_export.xml'
        exported = ET.parse(xml).getroot()
        audit['physics_configuration_evidence'] = {
            'checkpoint_config': {key: config[key] for key in ['scene', 'collision', 'sim', 'reset']},
            'env_voc': {key: config['env'][key] for key in ['voc', 'voc_scale', 'auto_reset']},
            'runtime_voc_override': config['voc_scale_override'],
            'ground_plane_code': 'scene/builder.py:506-507: if collision.ground, builder.add_ground_plane(...mu=contact_friction). scene.support controls only optional support USDA.',
            'ground_collision_filter_code': 'scene/collision.py:171-172 filters robot-support pairs only within the optional support span. The ground is appended afterwards and object-ground collisions remain active.',
            'gravity_source': 'newton/_src/sim/builder.py:795: ModelBuilder(up_axis=Axis.Z, gravity=-9.81)',
            'independent_preflight_xml': str(xml),
            'independent_preflight_xml_sha256': digest(xml),
            'independent_preflight_gravity': exported.find('option').get('gravity'),
            'independent_preflight_planes': [g.attrib for g in exported.findall('.//geom') if g.get('type') == 'plane'],
            'conclusion': 'The executed GPU recipe includes a z=0 ground collision plane and gravity [0,0,-9.81]. support=false omits an optional scanned tabletop asset; it does not remove the ground plane. GPU runtime model/contact buffers were not separately exported.',
        }
    semantics = ROOT / 'outputs/track3/episode_000011/wrench_origin_semantics/report.json'
    if semantics.exists():
        known = json.loads(semantics.read_text())
        audit['independent_initialization_warning'] = {
            'report': str(semantics), 'sha256': digest(semantics),
            'source': 'Existing Astra CPU semantics probe; reused without rerunning.',
            'frame_zero_probe': [row for row in known.get('physics', []) if row.get('frame') == 0],
            'limitation': 'Independent exact-reference 1ms CPU contact probe, not this GPU training reset or an all-step penetration bound. That probe assumed friction 0.3; this baseline assumes 0.5.',
            'production_adapters_integrated_into_this_baseline': False,
        }
    if (OUT / 'evaluation.json').exists():
        evaluation = json.loads((OUT / 'evaluation.json').read_text())
        audit['evaluation'] = evaluation
    trajectory = OUT / 'episode_000011.parquet'
    if trajectory.exists():
        table = pq.read_table(trajectory)
        data = table.to_pydict()
        achieved = np.stack([data['achieved_position_' + axis] for axis in 'xyz'], axis=-1)
        reference = np.stack([data['reference_position_' + axis] for axis in 'xyz'], axis=-1)
        errors = np.linalg.norm(achieved - reference, axis=-1)
        aq = np.stack([data['achieved_quaternion_' + axis] for axis in 'xyzw'], axis=-1)
        rq = np.stack([data['reference_quaternion_' + axis] for axis in 'xyzw'], axis=-1)
        aq /= np.linalg.norm(aq, axis=1, keepdims=True)
        rq /= np.linalg.norm(rq, axis=1, keepdims=True)
        angles = 2 * np.arccos(np.clip(np.abs(np.sum(aq * rq, axis=1)), 0, 1))
        def axis_z(q):
            x,y,z,w = q.T
            return np.stack([2*(x*z+y*w), 2*(y*z-x*w), 1-2*(x*x+y*y)], axis=-1)
        tilt = np.arccos(np.clip(np.sum(axis_z(aq)*axis_z(rq), axis=1), -1, 1))
        names = np.asarray(data['object_body_name'])
        valid = np.asarray(data['environment_valid'])
        audit['trajectory'] = {'path': str(trajectory), 'sha256': digest(trajectory),
                                'rows': table.num_rows, 'columns': table.column_names,
                                'metadata': {k.decode(): v.decode() for k, v in table.schema.metadata.items()},
                                'valid_rows': int(valid.sum()),
                                'all_achieved_positions_finite': bool(np.isfinite(achieved).all()),
                                'nonzero_simulated_tracking_error': bool(np.any(errors > 1e-6)),
                                'per_object': []}
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        for name in sorted(set(names.tolist())):
            mask = names == name
            t = np.asarray(data['time_s'])[mask]
            error = errors[mask]
            def first_crossing(values, threshold):
                hits = np.flatnonzero(values > threshold)
                if not len(hits):
                    return None
                index = int(hits[0])
                rows = np.flatnonzero(mask)
                return {'step': int(data['step'][rows[index]]), 'reference_frame': int(data['reference_frame'][rows[index]]),
                        'exported_time_s': float(t[index]), 'value': float(values[index])}
            audit['trajectory']['per_object'].append({'name': name, 'mean_position_error_m': float(np.mean(error)),
                                                       'max_position_error_m': float(np.max(error)),
                                                       'simulated_z_range_m': [float(achieved[mask, 2].min()), float(achieved[mask, 2].max())],
                                                       'first_position_error_over_10cm': first_crossing(error, 0.1),
                                                       'first_full_quaternion_error_over_0_5rad': first_crossing(angles[mask], 0.5),
                                                       'first_axis_tilt_error_over_0_5rad': first_crossing(tilt[mask], 0.5),
                                                       'axis_tilt_error_mean_rad': float(np.mean(tilt[mask])),
                                                       'first_achieved_link_position_m': achieved[mask][0].tolist(),
                                                       'first_reference_link_position_m': reference[mask][0].tolist(),
                                                       'last_achieved_link_position_m': achieved[mask][-1].tolist()})
            line, = axes[0].plot(t, 100 * error, label=name)
            axes[1].plot(t, achieved[mask, 2], color=line.get_color(), label=name + ' policy')
            axes[1].plot(t, reference[mask, 2], '--', color=line.get_color(), alpha=.7, label=name + ' reference')
        axes[0].set_ylabel('Object position error (cm)')
        axes[1].set_ylabel('Object link height (m)')
        axes[1].set_xlabel('Simulation time (s)')
        for axis in axes:
            axis.grid(alpha=.25)
            axis.legend(fontsize=8)
        fig.suptitle('Episode 11: 80-step diagnostic policy, no object assistance\nKnown geometry/contact/COM gates remain failed', fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / 'policy_rollout_diagnostic.png', dpi=150)
        fig.savefig(OUT / 'policy_rollout_diagnostic.pdf')
        plt.close(fig)
        audit['trajectory']['failure_timing_scope'] = 'Object threshold crossings reconstructed offline from exported poses. Wrist timing and GPU contact penetration/forces were not exported. The official evaluator disables early failure termination and completes 272 steps; its recorded object/wrist failure fractions are 1.0. Export row step 0 is after the first control step, while exported time_s starts at zero.'
        audit['trajectory']['interpretation'] = 'Both objects remain near the z=0 plane instead of following the provisional reference; tilt and position tracking fail. This 80-step runtime smoke is too short to assess algorithm capability. No further training or physics experiment was started.'
    preflight = json.loads((ROOT / 'outputs/track3/episode_000011/flashchord_reference_preflight/report.json').read_text())
    audit['input_asset_sha256'] = {}
    for relative in preflight.get('source_sha256', {}):
        if relative.endswith(('.urdf', '.xml', '.obj', '.npz')):
            path = ROOT / relative
            audit['input_asset_sha256'][relative] = digest(path)
    report['offline_audit'] = audit
    report['training_ready'] = False
    report['task_success_claimed'] = False
    report['submission_ready'] = False
    if 'actual_resume_command' in report:
        cwd = ROOT / 'vendor/video_to_data/robotic_grounding/flash_chord'
        report['resume_shell_command'] = 'cd ' + shlex.quote(str(cwd)) + ' && env CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 ' + report['actual_resume_command']
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    job = json.loads(JOB.read_text())
    job['audit'] = str(OUT / 'audit.json')
    job['resume_shell_command'] = report.get('resume_shell_command')
    job['completed_utc'] = report.get('completed_utc')
    exit_file = Path(job['exit_file'])
    job['exit_code'] = int(exit_file.read_text().strip()) if exit_file.exists() else None
    job['initial_total_wall_seconds'] = report.get('total_initial_budget_elapsed_seconds')
    job['gradient_updates_at_80'] = next((int(c['metadata']['global_update_step']) for c in audit['checkpoints']
                                         if c['metadata'].get('checkpoint_kind') == 'training_state'
                                         and c['metadata'].get('environment_steps') == '80'), None)
    job['policy_rollout_pass'] = report.get('policy_rollout_pass', False)
    (JOB).write_text(json.dumps(job, indent=2) + '\n')
    (OUT / 'README.md').write_text('''# Diagnostic official Sharpa baseline: episode 11

This is a bounded runtime/checkpoint experiment, not a training-ready result or
task-success claim. The source video episode is not in the scored 20-episode roster.
The interface-only parquet retains failed geometry/trajectory, unknown-contact,
and COM-wrench gates. Official unknown placeholders are not ground-truth negatives.

The unmodified official Sharpa FlashSAC CLI uses one GPU world, 64 environment
steps, then loads complete learner state plus matching replay and continues to 80.
The canonical 256/512 actor/critic architecture remains unchanged; the replay and
step budgets are reduced. Actor, learner and replay checkpoints are independent.
Training resets always start at reference frame zero, with the official random
finger scale Uniform[0,0.7] retained. Evaluation uses exact reference fingers.
Initial geometry remains unverified. The independent CPU semantics probe already
found 83mm initial penetration at frame zero; its different settings and 1ms
duration are recorded separately rather than presented as this GPU run's bound.
The executed checkpoint retains collision.ground=true, so scene.support=false
still adds a z=0 collision plane. Gravity is [0,0,-9.81]. Ground support is present;
the rollout does not follow the provisional object motion. Offline threshold
crossings, plane source evidence and the unrecorded wrist/contact-time limitation
are in audit.json. An 80-step smoke cannot assess the algorithm's eventual quality.
Official resume restores learning state and resets physics; this is not an exact
continuation of the simulator state.

Object assistance is disabled with zero VOC scale, max force/torque, reset decay,
and runtime curriculum override. Wrench rewards retain the upstream body/link
origin convention, with no COM conclusion or project-side semantic patch.

The submission kit evaluator and its three configs are byte-identical to the
installed vendor files despite different pinned revisions. The actual kit CLI is
composed and run against the produced actor. The exported parquet is real physics
under policy inference. Official evaluation disables early failure termination,
continues to reference end, and scores failures afterwards; completion does not
imply successful control. Metrics are against this provisional reconstruction,
not hidden competition ground truth.

See report.json, audit.json, individual stage logs/exits, composed_training.yaml,
and notes/track3-baseline-probe-job.json for hashes, results and exact commands.
Re-run the launcher to recover from the newest saved learner/replay checkpoint.
''')
    print(json.dumps({'state': report['state'], 'audit': str(OUT / 'audit.json'),
                      'checkpoint_count': len(audit['checkpoints']), 'trajectory': audit.get('trajectory'),
                      'actor_state_changes': audit['actor_state_changes']}, indent=2))


if __name__ == '__main__':
    main()
