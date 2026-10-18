"""Sequential canonical-scene saved-policy replays, honest candidate assembly and official dry-run.

Run inside root's server tmux. Does not launch tmux, train, upload or publish code.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from evaluate_track3_canonical_checkpoint import absolute, metadata, save, sha

ROOT = Path('/home/tianyuan/projects/v2d')
BASE = ROOT/'outputs/track3/scored_baseline_20261017/unified-gpu-repair-v1'
KIT = ROOT/'toolkit/8d2bce64/v2d_submission_kit'
CONTRACT = ROOT/'notes/track3-submission-contract.json'
# Text sources used by the frozen canonical builder, including the raster library's
# source rather than its compiled .so. Checked before replay and at assembly.
SCENE_HELPER_SOURCES = {
    'scripts/track3_canonical9_metric_batch.py': 'deaa85e95ea9b33e390c752e16e0ffbc1150339d17a878e5d9304ad8c69b5891',
    'scripts/build_track3_coarse_objects.py': 'e170ec7e9d640ac15fde9360113a6dc8b323e474d64243a67086bb55ed215936',
    'outputs/track3/advice_restart_20261017/canonical9_metric_v1/raster-source.cpp': '8b84a0c47b4a9f81448eb43567821e0f49c37c649f5f62220055999822cad870',
}


def scene_entries(path):
    manifest = json.loads(path.read_text())
    entries = manifest['episodes']
    if isinstance(entries, dict):
        entries = [dict(value, episode=key) for key,value in entries.items()]
    result = {}
    for row in entries:
        episode = str(int(str(row.get('episode',row.get('episode_id'))).replace('episode_','')))
        source = row.get('object_manifest',row.get('objects_manifest',row.get('objects_json')))
        if source is None:
            source = str(path.parent/f'episode_{int(episode):06d}/objects/objects.json')
        result[episode] = dict(row, object_manifest=str(absolute(source,path.parent)))
    return manifest,result


def validate_rollout(target, checkpoint, episode, contract, evaluator_sha):
    import numpy as np
    import pyarrow.parquet as pq
    report = json.loads((target/'replay-report.json').read_text())
    trajectory = target/f'episode_{int(episode):06d}.parquet'
    assert report['state'] == 'completed' and report['policy_rollout_performed'] and not report['object_assistance']
    assert report['checkpoint_sha256'] == sha(checkpoint)
    assert report['evaluator_sha256'] == evaluator_sha and report['trajectory_sha256'] == sha(trajectory)
    c = contract['episodes'][episode]
    table = pq.read_table(trajectory); data = table.to_pydict()
    names = c['objects_in_slot_order']
    assert table.num_rows == c['rows']
    grid = set(zip(data['environment_id'],data['reference_frame'],data['object_body_id']))
    assert grid == {(0,t,b) for t in range(c['scored_frames']) for b in range(len(names))} and len(grid)==table.num_rows
    assert all(data['environment_valid'])
    pose = np.array([[data['achieved_position_'+x][i] for x in 'xyz']+
                     [data['achieved_quaternion_'+x][i] for x in 'xyzw'] for i in range(table.num_rows)])
    assert np.isfinite(pose).all()
    np.testing.assert_allclose(np.linalg.norm(pose[:,3:],axis=1),1,atol=1e-3)
    assert all(name==names[slot] for name,slot in zip(data['object_body_name'],data['object_body_id']))
    meta = {k.decode():v.decode() for k,v in table.schema.metadata.items()}
    assert meta['flash_chord.checkpoint_sha256']==sha(checkpoint)
    assert meta['flash_chord.eval_script_sha256']==evaluator_sha
    assert meta['flash_chord.position_unit']=='meter' and meta['flash_chord.quaternion_order']=='xyzw'
    assert report['runtime']['actual_policy_calls']==c['scored_frames']
    assert report['training_environment_steps']==int(metadata(checkpoint)['user.environment_steps'])
    prepared=json.loads(Path(report['reference_report']).read_text())
    assert sha(prepared['reference_parquet'])==prepared['reference_sha256']==report['reference_sha256']
    assert sha(prepared['object_manifest'])==prepared['binding']['objects_manifest_sha256']
    for path,digest in prepared['binding']['asset_sha256'].items():
        assert sha(path)==digest, f'Scene asset changed since execution: {path}'
    return report


def copy_same(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        assert sha(source)==sha(dest), f'Frozen candidate differs: {dest}'
    else:
        shutil.copyfile(source,dest)


def source_export(output,path):
    digest=sha(path)
    key=f'producers/sha256_{digest}/{path.name}'
    copy_same(path,output/'code-export'/key)
    return key,digest


def mesh_provenance(output, scene_root, episode, name, obj, object_path, record, mesh, previous, entry):
    """Keep shared shape generation distinct from target-video rigid registration."""
    source_inputs=[str(absolute(p,ROOT)) for p in record.get('source_inputs',obj.get('source_inputs',[]))]
    reconstruction=absolute(record.get('reconstruction_source',str(object_path)),object_path.parent)
    if not reconstruction.is_file():
        reconstruction=object_path
    if entry.get('canonical_replacement') is False:
        old=previous['meshes'][name]
        assert sha(absolute(old['path'],BASE))==sha(mesh), 'Fallback mesh must equal its actual prior self-built asset'
        source_inputs=list(dict.fromkeys(source_inputs+[str(absolute(p,BASE)) for p in old['source_inputs']]))
        return dict(reconstruction_source=str(reconstruction),source_inputs=source_inputs,
                    source_usage='Exact prior self-built asset; no new canonical registration claimed.',
                    canonical_shared_shape=None,instance_registration=None)

    ep=f'episode_{int(episode):06d}'
    episode_root=ROOT/'outputs/track3'/ep
    calibration=episode_root/'calibrated_sequence/manifest.json'
    video=absolute(json.loads(calibration.read_text())['source_video'],ROOT)
    assert video.is_file() and video.name==ep+'.mp4'
    assert video.is_relative_to(ROOT/'data/track3_scored/track_3/evaluation/videos')
    registration_path=scene_root/'episodes'/ep/'registrations'/name/'report.json'
    registration=json.loads(registration_path.read_text())
    assert name in obj['registrations'] and registration['state']=='completed_candidate'
    assert obj['registrations'][name]['cache_key']==registration['cache_key']
    config_path=episode_root/'coarse-input.json'
    config=json.loads(config_path.read_text())
    rgb=absolute(config['rgb_pattern'].format(frame=int(registration['fit_frame'])),ROOT)
    assert rgb.is_file(), 'Actual registration RGB must exist'
    asset_report_path=scene_root/'assets'/name/'report.json'
    asset_report=json.loads(asset_report_path.read_text())
    parent_mesh=absolute(asset_report['source_mesh'],ROOT)
    assert sha(parent_mesh)==asset_report['source_mesh_sha256']
    registry=ROOT/'outputs/track3/advice_restart_20261017/all20-registration-inputs-v1/input-manifest.json'
    registry_row=next(r for r in json.loads(registry.read_text())['rows'] if r['episode']==ep and r['name']==name)
    shape=dict(role='canonical_shared_shape',parent_mesh=str(parent_mesh),parent_mesh_sha256=sha(parent_mesh),
               asset_report=str(asset_report_path),asset_report_sha256=sha(asset_report_path),
               source_registry=str(registry),source_registry_sha256=sha(registry),
               source_frame_metadata=registry_row.get('asset_frame'),
               current_episode_shape_generation_claimed=False)
    instance=dict(role='instance_registration',method=registration['method'],source_video=str(video),
                  calibration_manifest=str(calibration),calibration_manifest_sha256=sha(calibration),
                  input_config=str(config_path),input_config_sha256=sha(config_path),
                  original_rgb=str(rgb),original_rgb_sha256=sha(rgb),fit_frame=registration['fit_frame'],
                  registration_report=str(registration_path),registration_report_sha256=sha(registration_path),
                  shape_and_scale_frozen=True,depth_is_ground_truth=False)
    source_inputs=list(dict.fromkeys(source_inputs+[str(parent_mesh),str(video),str(calibration),str(config_path),str(rgb)]))
    lineage=output/'provenance'/ep/(name+'.json')
    save(lineage,dict(schema='track3_shared_shape_instance_lineage_v1',mesh_sha256=sha(mesh),
                     actual_object_manifest=str(object_path),actual_object_manifest_sha256=sha(object_path),
                     canonical_shared_shape=shape,instance_registration=instance,source_inputs=source_inputs,
                     production_quality_pass=False))
    return dict(reconstruction_source=str(lineage),source_inputs=source_inputs,
                source_usage='Shared reconstructed shape; current episode RGB/stereo used for rigid instance registration, not claimed as shared-shape generation.',
                canonical_shared_shape=shape,instance_registration=instance)


def assemble(args,entries,checkpoints,contract,evaluator):
    source_manifest=json.loads((BASE/'artifact-manifest.json').read_text())
    artifacts={};reports=[]
    executed={}
    for p in (evaluator,Path(__file__).resolve()):
        key,digest=source_export(args.output,p);executed[key]=digest
    for name,digest in SCENE_HELPER_SOURCES.items():
        path=ROOT/name
        assert sha(path)==digest, f'Frozen canonical helper changed: {path}'
        key,_=source_export(args.output,path);executed[key]=digest
    scene_doc=json.loads(args.scene_manifest.read_text())
    # The scene producer records the actual executed hash. Never bind a guessed current file.
    scene_sources=scene_doc.get('producer_source_sha256',{})
    for name,digest in scene_sources.items():
        path=absolute(name,ROOT)
        assert sha(path)==digest
        key,_=source_export(args.output,path);executed[key]=digest
    scene_digest=scene_doc.get('producer_sha256')
    if not scene_sources and not scene_digest:
        scene_protocol=args.scene_manifest.parent/'protocol.json'
        if scene_protocol.exists():
            scene_digest=json.loads(scene_protocol.read_text()).get('source_sha256')
    if not scene_sources and scene_digest:
        path=ROOT/'scripts/run_track3_canonical_all20_scenes.py'
        assert sha(path)==scene_digest
        key,digest=source_export(args.output,path);executed[key]=digest
    if not scene_sources and not scene_digest:
        raise ValueError('Scene producer source hash missing from manifest/protocol')
    for episode in contract['episodes']:
        target=args.output/'episodes'/f'episode_{int(episode):06d}'
        checkpoint=Path(checkpoints[episode])
        report=validate_rollout(target,checkpoint,episode,contract,sha(evaluator))
        object_path=Path(entries[episode]['object_manifest'])
        obj=json.loads(object_path.read_text());records={r['name']:r for r in obj['objects']}
        names=contract['episodes'][episode]['objects_in_slot_order'];assert obj['object_names']==names
        trajectory=target/f'episode_{int(episode):06d}.parquet'
        destination=args.output/'results'/trajectory.name;copy_same(trajectory,destination)
        meshes={}
        previous=source_manifest['episodes'][episode]
        for name in names:
            record=records[name]
            mesh=absolute(record['mesh_path'],object_path.parent)
            assert mesh.suffix.lower() in ('.obj','.glb')
            dest=args.output/'meshes'/f'episode_{int(episode):06d}'/(name+mesh.suffix.lower())
            copy_same(mesh,dest)
            provenance=mesh_provenance(args.output,args.scene_manifest.parent,episode,name,obj,object_path,record,mesh,previous,entries[episode])
            meshes[name]=dict(path=str(dest.relative_to(args.output)),**provenance)
        producers=dict(previous['producer_source_sha256'],**executed)
        artifacts[episode]=dict(state='completed',trajectory=str(destination.relative_to(args.output)),checkpoint=str(checkpoint),
             training_environment_steps=report['training_environment_steps'],policy_rollout_performed=True,object_assistance=False,
             object_names=names,meshes=meshes,producer_source_sha256=producers,quality_pass=False,
             acceptance_report=str(target/'acceptance.json'),evaluation_report=str(target/'evaluation.json'),
             reference_report=report['reference_report'],evaluation_physics=report['physics'],
             reused_policy_new_canonical_scene=entries[episode].get('canonical_replacement',True),
             scene_fallback=entries[episode].get('fallback'),training_repeated=False)
        reports.append(report)
    save(args.output/'checkpoints.json',checkpoints)
    save(args.output/'artifact-manifest.json',dict(schema='track3_cohort_artifacts_v1',state='completed',episodes=artifacts))
    from run_track3_submission_watch import inspect_artifacts
    audit=inspect_artifacts(args.output)
    save(args.output/'assembly-inspect.json',audit)
    if not audit['ready']:
        raise ValueError('Candidate formal inspection failed; see assembly-inspect.json')
    assert len({r['evaluator_sha256'] for r in reports})==1
    assert len({r['eval_code_sha256'] for r in reports})==1, 'Official pack requires common actual evaluator code'
    save(args.output/'assembly-report.json',dict(state='completed',ready_episodes=audit['ready_episodes'],
         ready_meshes=audit['ready_meshes'],rows=audit['rows'],policy_calls=sum(r['runtime']['actual_policy_calls'] for r in reports),
         quality_pass=False,uploads_performed=0,producer_source_sha256=audit['producer_source_sha256']))
    return audit


def official_checks(args):
    if not args.code_commit:
        raise ValueError('--code-commit must identify the actual newly published producing source before packing')
    python=ROOT/'.deps/v2d-submit-venv/bin/python'
    package=args.output/'official-package';package.mkdir(exist_ok=True)
    packed=package/'canonical_checkpoint.parquet'
    commands=[('pack',[str(python),str(KIT/'tools/pack_submission.py'),'--track','track_3','--results',str(args.output/'results'),
                  '--meshes',str(args.output/'meshes'),'--code-commit',args.code_commit,'-o',str(packed)]),
              ('preflight_offline',[str(python),str(KIT/'tools/preflight.py'),'--track','track_3','--results',str(args.output/'results'),'--skip-kaggle'])]
    if args.online_checks:
        commands += [('preflight_online',[str(python),str(KIT/'tools/preflight.py'),'--track','track_3','--results',str(args.output/'results')]),
                     ('dry_run',[str(python),str(KIT/'submit/v2d_submit.py'),str(packed),'--track','track_3','--dry-run',
                                  '-m','Canonical scene saved-policy real rollout candidate; no upload'])]
    from run_track3_submission_watch import redact
    results={}
    for name,cmd in commands:
        try:
            result=subprocess.run(cmd,capture_output=True,text=True,timeout=300)
            results[name]=result.returncode
            log=result.stdout+result.stderr
        except subprocess.TimeoutExpired:
            results[name]=124;log='Stage timeout\n'
        (package/(name+'.log')).write_text(redact(log))
        save(package/'checkpoint.json',dict(stages=results,uploads_performed=0))
        if results[name] and name=='pack':
            raise ValueError('Official pack failed; see redacted log')
    save(package/'report.json',dict(state='completed_checks',stages=results,packed=str(packed),
         packed_sha256=sha(packed),code_commit=args.code_commit,uploads_performed=0,quality_pass=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint-map',type=Path,default=BASE/'checkpoints.json')
    p.add_argument('--python',type=Path,default=ROOT/'.deps/flash-chord-venv/bin/python')
    p.add_argument('--episode-ids',type=int,nargs='*')
    p.add_argument('--substeps',type=int,default=20)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--episode-timeout',type=int,default=900)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--assemble-only',action='store_true')
    p.add_argument('--pack',action='store_true')
    p.add_argument('--code-commit')
    p.add_argument('--online-checks',action='store_true')
    a=p.parse_args();a.output=a.output.resolve();a.scene_manifest=a.scene_manifest.resolve()
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'batch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        doc,entries=scene_entries(a.scene_manifest)
        checkpoints=json.loads(a.checkpoint_map.read_text());contract=json.loads(CONTRACT.read_text())
        requested=[str(x) for x in a.episode_ids] if a.episode_ids else list(contract['episodes'])
        assert set(requested)<=set(contract['episodes']) and set(checkpoints)==set(contract['episodes'])
        runtime=a.output/'canonical-producer-source';runtime.mkdir(exist_ok=True)
        evaluator=runtime/'evaluate_track3_canonical_checkpoint.py'
        copy_same(ROOT/'scripts/evaluate_track3_canonical_checkpoint.py',evaluator)
        binding=dict(evaluator_sha256=sha(evaluator),batch_sha256=sha(Path(__file__)),
             checkpoint_map_sha256=sha(a.checkpoint_map),scene_manifest_sha256=sha(a.scene_manifest),substeps=a.substeps,device=a.device,
             scene_helper_source_sha256=SCENE_HELPER_SOURCES)
        for name,digest in SCENE_HELPER_SOURCES.items():
            assert sha(ROOT/name)==digest, f'Frozen canonical helper changed: {name}'
        if (a.output/'protocol.json').exists():
            assert json.loads((a.output/'protocol.json').read_text())==binding,'Frozen job inputs changed; use new output'
        else:save(a.output/'protocol.json',binding)
        state=dict(state='running',started_utc=datetime.now(timezone.utc).isoformat(),completed=[],failed={},expected_episodes=requested)
        env=dict(os.environ,XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',WANDB_MODE='disabled')
        if not a.assemble_only:
            for episode in requested:
                target=a.output/'episodes'/f'episode_{int(episode):06d}';target.mkdir(parents=True,exist_ok=True)
                try:
                    assert episode in entries,'Canonical scene entry missing'
                    checkpoint=Path(checkpoints[episode])
                    if (target/'replay-report.json').exists() and not a.prepare_only:
                        report=validate_rollout(target,checkpoint,episode,contract,sha(evaluator))
                    else:
                        command=[str(a.python),'-u',str(evaluator),'--checkpoint',str(checkpoint),'--episode',episode,
                             '--object-manifest',entries[episode]['object_manifest'],'--output',str(target),
                             '--substeps',str(a.substeps),'--device',a.device]
                        if a.prepare_only:command+=['--prepare-only']
                        with (target/'run.log').open('a') as log:
                            try:
                                process=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=a.episode_timeout)
                                code=process.returncode
                            except subprocess.TimeoutExpired:code=124
                        (target/'run.exit').write_text(str(code)+'\n')
                        if code:raise RuntimeError(f'Child exit {code}')
                        report=validate_rollout(target,checkpoint,episode,contract,sha(evaluator)) if not a.prepare_only else None
                    if report:save(target/'acceptance.json',dict(report,hard_format_pass=True,quality_pass=False))
                    state['completed'].append(episode)
                except Exception as exc:
                    state['failed'][episode]=dict(error_type=type(exc).__name__,error=str(exc),log=str(target/'run.log'))
                state.update(current_episode=episode,updated_utc=datetime.now(timezone.utc).isoformat())
                save(a.output/'batch-checkpoint.json',state)
                print(json.dumps(dict(episode=episode,completed=len(state['completed']),failure=state['failed'].get(episode))),flush=True)
        all20=set(requested)==set(contract['episodes'])
        if not a.prepare_only and all20 and not state['failed']:
            assemble(a,entries,checkpoints,contract,evaluator)
            if a.pack:official_checks(a)
        state['state']='completed_with_failures' if state['failed'] else 'prepared_references' if a.prepare_only else 'completed'
        state.update(uploads_performed=0,quality_pass=False)
        save(a.output/'batch-report.json',state)
        if state['failed']:sys.exit(2)


if __name__=='__main__':
    main()
