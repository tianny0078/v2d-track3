"""Real full-length saved-policy rollout with a new canonical-object reference.

The actor and its observation/action schema remain the checkpoint's. No training,
reference-pose substitution, object assistance, or old-trajectory copying occurs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def metadata(checkpoint):
    with Path(checkpoint).open('rb') as f:
        length = struct.unpack('<Q', f.read(8))[0]
        if length > 32*1024**2:
            raise ValueError('Invalid checkpoint header size')
        return json.loads(f.read(length))['__metadata__']


def absolute(path, parent):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (parent/path).resolve()


def prepare_reference(checkpoint, objects_path, episode, output):
    """CPU only. Preserve every original human/robot/contact field exactly."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scipy.spatial.transform import Rotation
    checkpoint, objects_path, output = Path(checkpoint).resolve(), Path(objects_path).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    saved = json.loads(metadata(checkpoint)['user.resolved_config_json'])
    source = Path(saved['task']['parquet']).resolve()
    table = pq.read_table(source)
    if table.num_rows != 1:
        raise ValueError('Expected one-row full-sequence reference')
    original = table.to_pylist()[0]
    obj = json.loads(objects_path.read_text())
    names = list(obj['object_names'])
    assert names == original['object_body_names']
    frames = len(original['object_body_position']); bodies = len(names)
    assert int(obj.get('episode_id', episode)) == episode
    assert float(obj['fps']) == float(original['fps'])
    if 'frame_index' in obj:
        np.testing.assert_array_equal(obj['frame_index'], np.arange(frames))
    assert obj['pose_semantics'] == 'link_origin'
    objects = {r['name']:r for r in obj['objects']}
    records = [objects[name] for name in names]
    pose_path = absolute(obj['poses_path'], objects_path.parent) if 'poses_path' in obj else None
    if pose_path is not None:
        with np.load(pose_path, allow_pickle=False) as z:
            poses = {key:z[key].copy() for key in z.files}
    else:
        poses = obj
    if 'world_T_canonical' in poses:
        transforms = np.asarray(poses['world_T_canonical'], float)
    else:
        q = np.asarray(poses['object_pose_wxyz'], float)
        assert q.shape == (frames, bodies, 7) and np.isfinite(q).all()
        np.testing.assert_allclose(np.linalg.norm(q[..., 3:], axis=-1), 1, atol=1e-5)
        transforms = np.broadcast_to(np.eye(4), (frames,bodies,4,4)).copy()
        transforms[..., :3, 3] = q[..., :3]
        transforms[..., :3, :3] = Rotation.from_quat(q[..., [4,5,6,3]].reshape(-1,4)).as_matrix().reshape(frames,bodies,3,3)
    assert transforms.shape == (frames,bodies,4,4) and np.isfinite(transforms).all()
    frame = obj.get('coordinate_frame', obj.get('coord_frame'))
    if frame == 'world':
        transforms = np.asarray(obj['table_T_world'], float) @ transforms
    elif frame not in ('table', 'table_Z_up_meters'):
        raise ValueError('Explicit world/table coordinate frame required')
    np.testing.assert_allclose(transforms[...,3,:], np.broadcast_to([0,0,0,1], (frames,bodies,4)), atol=1e-6)
    rotation = transforms[...,:3,:3]
    np.testing.assert_allclose(rotation.swapaxes(-1,-2)@rotation, np.broadcast_to(np.eye(3), rotation.shape), atol=1e-5)
    assert (np.linalg.det(rotation) > .999).all()
    quat = Rotation.from_matrix(rotation.reshape(-1,3,3)).as_quat().reshape(frames,bodies,4)[...,[3,0,1,2]]
    com = np.asarray([r['canonical_COM'] for r in records], float)
    radius = np.asarray([r['bounding_radius_m'] for r in records], float)
    assert com.shape == (bodies,3) and np.isfinite(com).all()
    assert radius.shape == (bodies,) and np.isfinite(radius).all() and (radius > 0).all()
    meshes = [absolute(r['mesh_path'], objects_path.parent) for r in records]
    urdfs = [absolute(r['urdf_path'], objects_path.parent) for r in records]
    assets = {}
    for path in meshes+urdfs:
        assets[str(path)] = sha(path)
    for urdf in urdfs:
        for mesh in ET.parse(urdf).getroot().iter('mesh'):
            path = absolute(mesh.attrib['filename'], urdf.parent)
            assets[str(path)] = sha(path)
    replacement = dict(object_mesh_paths=[str(p) for p in meshes], object_urdf_paths=[str(p) for p in urdfs],
        object_mesh_radius=radius.tolist(), object_body_position=transforms[...,:3,3].tolist(), object_body_wxyz=quat.tolist())
    binding = dict(checkpoint_sha256=sha(checkpoint), source_reference_sha256=sha(source),
                   objects_manifest_sha256=sha(objects_path), poses_sha256=sha(pose_path) if pose_path else None,
                   producer_sha256=sha(Path(__file__)), asset_sha256=assets)
    report_path = output/'reference-report.json'
    path = output/'episode_reference.parquet'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        assert report['binding'] == binding and report['reference_sha256'] == sha(path), 'Existing reference changed; use new output'
        return report
    for name, value in replacement.items():
        index = table.schema.get_field_index(name)
        table = table.set_column(index, table.schema.field(index), pa.array([value], type=table.schema.field(index).type))
    temporary = path.with_suffix('.tmp')
    pq.write_table(table, temporary); temporary.replace(path)
    reread = pq.read_table(path)
    source_table = pq.read_table(source)
    for name in source_table.column_names:
        if name not in replacement:
            assert reread[name].equals(source_table[name]), name
    np.savez_compressed(output/'reference_checkpoint.npz', object_link_pose_table=transforms,
         object_COM_table=transforms[...,:3,3]+np.einsum('tbij,bj->tbi',rotation,com),
         object_COM_local=com, object_bounding_radius_COM=radius, frame_index=np.arange(frames),
         contact_ground_truth_known=np.zeros((frames,2,bodies),np.int32))
    report = dict(state='prepared_reference_only', episode=episode, frames=frames, objects=names, fps=float(original['fps']),
        reference_parquet=str(path), reference_sha256=sha(path), source_reference=str(source),
        object_manifest=str(objects_path), binding=binding, body_com_local=com.tolist(), com_radius=radius.tolist(),
        replaced_columns=list(replacement), preserved_other_columns=True, policy_rollout_performed=False,
        coordinate_frame='table_Z_up_meters', quality_pass=False)
    save(report_path, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--object-manifest', type=Path, required=True)
    p.add_argument('--episode', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--substeps', type=int, default=20)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    assert a.substeps > 0
    a.output = a.output.resolve(); a.output.mkdir(parents=True,exist_ok=True)
    prepared = prepare_reference(a.checkpoint, a.object_manifest, a.episode, a.output/'reference')
    if a.prepare_only:
        print(json.dumps(prepared)); return
    snapshot = a.checkpoint.resolve().parent.parent/'producer-source'
    assert snapshot.is_dir(), 'Missing original checkpoint producer snapshot'
    sys.path.insert(0,str(snapshot))
    import numpy as np
    import warp as wp
    import track3_evaluation_harness as h
    wp.init(); wp.set_device(a.device)
    base_config = h.checkpoint_config
    effective = {}
    def config(meta):
        c = base_config(meta)
        effective['original_reference'] = c['task']['parquet']
        effective['original_substeps'] = c['env']['sim']['substeps']
        c['task']['parquet'] = prepared['reference_parquet']
        c['task'].update(motion_speed=1., motion_start_frame=0, motion_end_frame=-1)
        c['training']['device'] = a.device
        c['env']['sim']['substeps'] = a.substeps
        c['scene'].update(object_scale_min=1., object_scale_max=1., decompose_objects=False)
        objective = c['env']['objective']
        if objective.get('_target_','').endswith('MaskedObjectiveSpec'):
            objective.update(body_com_local=prepared['body_com_local'], com_radius=prepared['com_radius'],
                reference_known=np.zeros((prepared['frames'],2,len(prepared['objects'])),np.int32).tolist(),
                provenance='Canonical scene replay; GT contact unknown; saved actor reused without retraining.')
        effective.update(device=a.device, substeps=a.substeps, reference_parquet=c['task']['parquet'],
                         scale=1., motion_speed=1., frozen_actor=True)
        save(a.output/'resolved-evaluation-config.json',c)
        return c
    h.checkpoint_config = config
    original_record = h.record_object_pose_rollout
    runtime = {}
    def record(env, jax_env, policy_action, actor_state, key, observation, **kwargs):
        assert env.state_0.body_q.device == env.device, 'Scene/runtime device mismatch'
        assert str(env.device) == a.device, 'Unexpected physics device'
        assert kwargs['start_frame'] == 0 and kwargs['step_count'] == prepared['frames']
        initial = {k:getattr(env.state_0,k).numpy() for k in ('joint_q','joint_qd','body_q','body_qd')}
        np.savez_compressed(a.output/'initial-state.npz', **initial)
        actions = []
        def capture_action(*args):
            action, next_key = policy_action(*args)
            value = np.asarray(action)
            if not np.isfinite(value).all():
                raise FloatingPointError('Policy emitted nonfinite action')
            actions.append(value.copy())
            return action, next_key
        r = original_record(env, jax_env, capture_action, actor_state, key, observation, **kwargs)
        np.savez_compressed(a.output/'rollout_checkpoint.npz', achieved_pose_w=r.achieved_pose_w,
              reference_pose_w=r.reference_pose_w, tracking_error=r.tracking_error,
              invalid_worlds=np.asarray(r.non_finite_worlds), policy_actions=np.asarray(actions),
              final_joint_q=env.state_0.joint_q.numpy(), final_joint_qd=env.state_0.joint_qd.numpy())
        runtime.update(physics_device=str(env.device), actual_policy_calls=len(actions),
                       observed_dimensions=env.observation_dim, action_dimensions=env.action.action_dim)
        if r.non_finite_worlds:
            raise FloatingPointError(f'Invalid policy rollout worlds {r.non_finite_worlds}; no sanitized trajectory accepted')
        assert len(actions) == prepared['frames']
        return r
    h.record_object_pose_rollout = record
    start = time.monotonic()
    with wp.ScopedDevice(a.device):
        result = h.evaluate_checkpoint(a.checkpoint, world_count=1,start_frame=0,reset_mode='explicit',
             object_trajectories_output=a.output/f'episode_{a.episode:06d}.parquet',eval_script=Path(__file__))
    h.write_evaluation_report(a.output/'evaluation.json',result)
    trajectory = a.output/f'episode_{a.episode:06d}.parquet'
    report = dict(state='completed',episode=a.episode,completed_utc=datetime.now(timezone.utc).isoformat(),
         elapsed_seconds=time.monotonic()-start, checkpoint=str(a.checkpoint.resolve()),
         checkpoint_sha256=sha(a.checkpoint), evaluator_sha256=sha(Path(__file__)),
         harness_sha256=sha(h.__file__), eval_code_sha256=h.evaluation_code_sha256(),
         reference_report=str(a.output/'reference/reference-report.json'), reference_sha256=prepared['reference_sha256'],
         trajectory=str(trajectory),trajectory_sha256=sha(trajectory), physics=effective,runtime=runtime,
         metrics=h.evaluation_scalars(result), training_environment_steps=result.environment_steps,
         frames=prepared['frames'],objects=prepared['objects'],policy_rollout_performed=True,object_assistance=False,
         original_checkpoint_unchanged=True, training_repeated=False,quality_pass=False)
    save(a.output/'replay-report.json',report)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
