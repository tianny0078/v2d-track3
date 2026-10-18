"""Read-only v1/recovery selection and bounded affected-scene derivation.
Never edits v1 evidence, reruns shape/scale models, or evaluates a policy.
"""
from pathlib import Path
import argparse,datetime,fcntl,hashlib,json,os,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/track3/advice_restart_20261017'
FROZEN_SCENE=ROOT/'scripts/run_track3_canonical_all20_scenes.py'
FROZEN_SCENE_SHA='d97f0d8caf20e09a795bafd5d950c31fd78655a4d138b961b08b3636b7786fa8'
SELF=Path(__file__).resolve()

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def safe_read(p):
 try:return read(p)
 except (OSError,ValueError):return {}
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def absolute(p):
 p=Path(p);return p if p.is_absolute()else ROOT/p

def verify_outputs(report):
 for path,digest in report.get('output_sha256',{}).items():
  if sha(absolute(path))!=digest:raise ValueError('Bound output changed: '+str(path))

def terminal_inputs(source,recoveries):
 manifest=safe_read(source/'manifest.json');job=safe_read(source/'job.json')
 source_done=(manifest.get('state')=='completed_bounded_batch' and job.get('state')=='completed_bounded_batch' and len(manifest.get('episodes',[]))==20)
 results=[]
 for folder in recoveries:
  report=safe_read(folder/'report.json');failure=safe_read(folder/'failure.json')
  results.append(dict(folder=str(folder),terminal=report.get('state')=='completed' or bool(failure),state=report.get('state') or failure.get('state') or 'pending'))
 return source_done and all(x['terminal']for x in results),dict(source_terminal=source_done,recovery=results)

def bind_inputs(source,recoveries):
 assert sha(FROZEN_SCENE)==FROZEN_SCENE_SHA,'Original scene producer changed; do not silently change frozen experiment'
 input_hashes={};producer={str(SELF.relative_to(ROOT)):sha(SELF),str(FROZEN_SCENE.relative_to(ROOT)):sha(FROZEN_SCENE)}
 def bind(p):
  p=Path(p).resolve();input_hashes[str(p)]=sha(p);return read(p)
 manifest=bind(source/'manifest.json');bind(source/'report.json');protocol=bind(source/'protocol.json')
 assert protocol['source_sha256']==FROZEN_SCENE_SHA
 original_rows=bind(BASE/'all20-registration-inputs-v1/input-manifest.json')['rows'];ready={};failures=[]
 for folder in recoveries:
  report_path=folder/'report.json'
  if not report_path.exists():
   failures.append(dict(folder=str(folder),failure=bind(folder/'failure.json')));continue
  report=bind(report_path);assert report['state']=='completed';verify_outputs(report)
  name=report['name'];a=report['asset'];assert a['name']==name
  assert report.get('visual_mesh_unchanged') is True and report.get('scale_applied_again') is False
  for required in ['mesh_path','urdf_path','canonical_COM','bounding_radius_m','source_inputs']:assert required in a,(name,required)
  # The recovery's own visual-array equality check is source-bound; no extra fit.
  assert sha(report['source_mesh'])==report['source_mesh_sha256']
  allowed_producers={'scripts/recover_track3_canonical_collision.py','scripts/recover_track3_canonical_collision_qem.py','scripts/recover_track3_canonical_collision_qem_extra.py','scripts/recover_track3_canonical_collision_postqem.py'}
  declared=report.get('producer_path','scripts/recover_track3_canonical_collision.py')
  assert declared in allowed_producers,'Unrecognized recovery producer'
  code=ROOT/declared;helper=ROOT/'scripts/build_track3_coarse_objects.py'
  assert sha(code)==report['source_sha256'] and sha(helper)==report['helper_sha256']
  producer[declared]=report['source_sha256'];producer[str(helper.relative_to(ROOT))]=report['helper_sha256']
  allowed_helpers=allowed_producers|{'scripts/build_track3_coarse_objects.py'}
  for path,digest in report.get('producer_source_sha256',{}).items():
   assert path in allowed_helpers and sha(ROOT/path)==digest,'Changed or unrecognized recovery helper'
   producer[path]=digest
  ready[name]=dict(folder=folder,report=report,report_path=report_path)
 # Bind all successfully reused asset reports and their exact output hashes.
 for p in sorted((source/'assets').glob('*/report.json')):
  d=bind(p);verify_outputs(d)
 return manifest,original_rows,ready,failures,input_hashes,producer

def link_once(source,dest):
 source=Path(source).resolve();dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists()or dest.is_symlink():
  assert dest.is_symlink()and dest.resolve()==source,'Conflicting derived link: '+str(dest)
 else:dest.symlink_to(source,target_is_directory=source.is_dir())

def run_affected(ep,source,out,timeout_seconds):
 dest=out/'episodes'/ep;dest.mkdir(parents=True,exist_ok=True);failure=dest/'failure.json'
 if failure.exists():return dict(episode=ep,state='preserved_failure',failure=read(failure))
 # These are complete shape/pose registration caches. Never link partial caches:
 # the frozen producer may then write them, which would modify v1.
 reused=[]
 for registration in (source/'episodes'/ep/'registrations').glob('*'):
  rp=registration/'report.json'
  if not rp.exists():continue
  report=read(rp)
  if report.get('state')!='completed_candidate'or not(registration/'registration.npz').exists():continue
  verify_outputs(report);link_once(registration,dest/'registrations'/registration.name);reused.append(registration.name)
 command=[sys.executable,'-u',str(FROZEN_SCENE),'--stage','episode','--key',ep,'--output',str(out)]
 start=time.monotonic()
 with (dest/'selection-run.log').open('a')as log:
  try:code=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=timeout_seconds).returncode
  except subprocess.TimeoutExpired:code=124
 (dest/'selection-run.exit').write_text(str(code)+'\n')
 if code:
  save(failure,dict(state='timeout'if code==124 else'failed',exit_code=code,wall_cap_seconds=timeout_seconds,source_v1_unchanged=True,checkpoint_preserved=True,automatic_retry=False))
 return dict(episode=ep,state='completed'if code==0 else'failed',exit_code=code,reused_registration_names=reused,elapsed_seconds=time.monotonic()-start,command=command)

def selected_entry(ep,old_entry,source,out,derived_ok):
 candidate_report=out/'episodes'/ep/'report.json'
 if derived_ok and candidate_report.exists():
  report=read(candidate_report);verify_outputs(report)
  assert report['state']=='completed'
  path=absolute(report['recommended_objects_json']);kind='recovered_collision_derived_scene';scene_root=candidate_report.parent
  entry=dict(episode=ep,objects_json=str(path),object_manifest=str(path),original_objects_json=report['objects_json'],settled_objects_json=report['settled_objects_json'],scene_xml=report['scene_xml'],reference_parquet=None)
 else:
  entry=dict(old_entry);path=absolute(entry.get('object_manifest')or entry['objects_json']);kind='v1_completed_scene'if (source/'episodes'/ep/'report.json').exists()else'v1_previous_self_built_fallback';scene_root=source/'episodes'/ep
  candidate_report=scene_root/'report.json';report=safe_read(candidate_report)
 objects=read(path);assert int(str(objects.get('episode_id',ep)).replace('episode_',''))==int(ep.split('_')[1])
 # Count the actual chosen arm, not ORs from unselected candidate alternatives.
 arm='settled_initial'if path.parent.name=='settled_objects'else'original_initial'
 physics_path=scene_root/arm/'report.json';physics=safe_read(physics_path)if report and kind!='v1_previous_self_built_fallback'else{}
 if physics:verify_outputs(physics)
 passed=physics.get('engineering_pass')is True
 entry.update(episode=ep,objects_json=str(path),object_manifest=str(path),selection_kind=kind,selected_static_arm=arm if physics else None,selected_static_report=str(physics_path)if physics else None,selected_static_completed=physics.get('state')=='completed',engineering_pass=passed,production_pass=False,quality_pass=False,objects_manifest_sha256=sha(path),selected_scene_report=str(candidate_report)if report else None)
 if kind=='v1_previous_self_built_fallback':entry['fallback']='Previous self-built source retained after bounded asset/scene failure; new canonical size claims do not apply'
 return entry,objects,physics

def measure_selected(entries):
 import numpy as np
 import trimesh
 rows=[]
 for entry,objects,_ in entries:
  names=objects['object_names'];assets=objects['objects'];assert len(names)==len(assets)
  for name,a in zip(names,assets):
   assert name==a['name'];path=absolute(a['mesh_path']);mesh=trimesh.load(path,force='mesh',process=False);finite=bool(np.isfinite(mesh.vertices).all())
   rows.append(dict(episode=entry['episode'],name=name,selection_kind=entry['selection_kind'],mesh_path=str(path),mesh_sha256=sha(path),finite=finite,vertices=len(mesh.vertices),faces=len(mesh.faces),extents_m=mesh.extents.tolist()if finite else None,oversized_max_axis_gt_0_5m=bool(mesh.extents.max()>.5)if finite else None,objects_manifest=entry['object_manifest'],objects_manifest_sha256=entry['objects_manifest_sha256']))
 return rows

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=BASE/'canonical_all20_scenes_v1');p.add_argument('--recovery',type=Path,action='append');p.add_argument('--output',type=Path,default=BASE/'canonical_evening_scene_selection_v1');p.add_argument('--wait-seconds',type=float,default=7200);p.add_argument('--episode-timeout',type=int,default=360);p.add_argument('--inspect',action='store_true');a=p.parse_args()
 source=a.source.resolve();out=a.output.resolve();recoveries=[(BASE/'canonical_collision_recovery_v1'/x).resolve()if len(x.parts)==1 else x.resolve()for x in (a.recovery or [Path('plastic_dish_rack'),Path('mini_dust_pan')])]
 assert out!=source and not source.is_relative_to(out)and not out.is_relative_to(source),'Independent sibling output required'
 assert 1<=a.episode_timeout<=360 and 0<=a.wait_seconds<=10800
 ready,wait=terminal_inputs(source,recoveries)
 if a.inspect:print(json.dumps(dict(ready=ready,**wait),indent=2));return
 out.mkdir(parents=True,exist_ok=True)
 with (out/'selector.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.monotonic()
  while not ready:
   save(out/'job.json',dict(state='waiting_for_existing_jobs',heartbeat_utc=now(),wait=wait,source=str(source),recovery=[str(x)for x in recoveries],no_new_model_jobs=True))
   if time.monotonic()-started>=a.wait_seconds:raise TimeoutError('Existing v1/recovery jobs not terminal within bounded wait')
   time.sleep(min(15,max(0,a.wait_seconds-(time.monotonic()-started))));ready,wait=terminal_inputs(source,recoveries)
  manifest,rows,recovery_ok,recovery_failed,bindings,producer=bind_inputs(source,recoveries)
  episodes=sorted(set(r['episode']for r in rows));assert len(episodes)==20 and len(rows)==36
  old={r['episode']:r for r in manifest['episodes']};assert set(old)==set(episodes)
  affected=sorted(set(r['episode']for r in rows if r['name']in recovery_ok))
  protocol=dict(source_v1=str(source),source_manifest_sha256=sha(source/'manifest.json'),recovery_names=sorted(recovery_ok),affected_episodes=affected,episode_timeout_seconds=a.episode_timeout,producer_source_sha256=producer,input_sha256=bindings,selection_rule='Completed collision-recovery derived scene; execution failure retains original v1 selection. Engineering failures reported, not silently promoted.',GT_read=False,model_generation=False,scale_fit=False,policy_rollout=False)
  if (out/'protocol.json').exists():assert read(out/'protocol.json')==protocol,'Selection inputs/code changed; use independent output'
  else:save(out/'protocol.json',protocol)
  if (out/'report.json').exists():
   report=read(out/'report.json');verify_outputs(report);print('Completed frozen selection retained');return
  available=set()
  for name in sorted(set(r['name']for r in rows)):
   src=recovery_ok[name]['folder']if name in recovery_ok else source/'assets'/name
   if (src/'report.json').exists():link_once(src,out/'assets'/name);available.add(name)
  tasks=[];derived=set()
  # Read-only evidence links let the official consumer locate unchanged instance
  # registrations under the selected manifest root without copying or rewriting v1.
  for ep in episodes:
   original=source/'episodes'/ep
   if ep not in affected and original.is_dir():link_once(original,out/'episodes'/ep)
  for ep in affected:
   missing=[r['name']for r in rows if r['episode']==ep and r['name']not in available]
   if missing:task=dict(episode=ep,state='other_asset_still_missing',missing=missing)
   else:
    save(out/'job.json',dict(state='running_affected_scene',current_episode=ep,affected_episodes=affected,heartbeat_utc=now(),completed_tasks=tasks))
    task=run_affected(ep,source,out,a.episode_timeout)
    if task['state']=='completed':derived.add(ep)
   tasks.append(task);save(out/'checkpoint.json',dict(state='running',tasks=tasks,updated_utc=now()))
  selected=[selected_entry(ep,old[ep],source,out,ep in derived)for ep in episodes];sizes=measure_selected(selected)
  assert len(sizes)==36
  for path,digest in bindings.items():assert sha(path)==digest,'Original input changed during selection: '+path
  # Include actual selected mesh/URDF bytes in the auditable derivative manifest.
  selected_assets={}
  for entry,objects,physics in selected:
   verify_outputs(safe_read(entry['selected_scene_report']))if entry.get('selected_scene_report')else None
   for obj in objects['objects']:
    for key in ['mesh_path','urdf_path']:
     path=absolute(obj[key]);selected_assets[str(path)]=sha(path)
  result_manifest=dict(state='completed_bounded_scene_selection',episodes=[x[0]for x in selected],producer_source_sha256=producer,source_v1_manifest=str(source/'manifest.json'),source_v1_manifest_sha256=sha(source/'manifest.json'),recovery_report_sha256={name:sha(value['report_path'])for name,value in recovery_ok.items()},selected_asset_sha256=selected_assets,GT_read=False,production_pass=False)
  save(out/'manifest.json',result_manifest);save(out/'selected-mesh-inventory.json',dict(rows=sizes,object_instances=36,oversized_instances=sum(x['oversized_max_axis_gt_0_5m']is True for x in sizes),unknown_size_instances=sum(x['oversized_max_axis_gt_0_5m']is None for x in sizes),source='Actual selected objects.json mesh paths including any legacy fallback; not unselected canonical inventory'))
  report=dict(state='completed_bounded_scene_selection',completed_utc=now(),episodes=20,object_instances=36,recovered_asset_names=sorted(recovery_ok),recovery_failures=recovery_failed,affected_episodes=affected,derived_completed=len(derived),selected_original_v1=20-len(derived),selected_legacy_fallback_count=sum(x[0]['selection_kind']=='v1_previous_self_built_fallback'for x in selected),selected_full_static_completed=sum(x[0]['selected_static_completed']for x in selected),selected_engineering_pass_count=sum(x[0]['engineering_pass']for x in selected),engineering_pass_count=sum(x[0]['engineering_pass']for x in selected),oversize_checked_instances=sum(x['oversized_max_axis_gt_0_5m']is not None for x in sizes),oversized_instances=sum(x['oversized_max_axis_gt_0_5m']is True for x in sizes),oversized_selected_instances=sum(x['oversized_max_axis_gt_0_5m']is True for x in sizes),unknown_size_instances=sum(x['oversized_max_axis_gt_0_5m']is None for x in sizes),tasks=tasks,producer_source_sha256=producer,source_input_sha256=bindings,production_pass=False,policy_rollout=False,output_sha256={str(out/'manifest.json'):sha(out/'manifest.json'),str(out/'selected-mesh-inventory.json'):sha(out/'selected-mesh-inventory.json'),**selected_assets})
  save(out/'report.json',report);save(out/'job.json',dict(state='completed_bounded_scene_selection',completed_utc=now(),report=str(out/'report.json')));print(json.dumps({k:report[k]for k in ['state','derived_completed','selected_full_static_completed','selected_engineering_pass_count','oversized_selected_instances']}),flush=True)
if __name__=='__main__':main()
