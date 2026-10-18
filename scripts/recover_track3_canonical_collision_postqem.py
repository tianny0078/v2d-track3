"""One fixed CoACD restart from completed QEM cache; no QEM/model regeneration.
Only resource budget changes: OpenMP16, BLAS1, real worker deadline1200s.
This restarts CoACD from the saved mesh, not from a C++ search checkpoint.
"""
from pathlib import Path
import argparse,datetime,fcntl,hashlib,importlib.metadata,json,os,signal,subprocess,sys,time
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/track3/advice_restart_20261017'
SELF=Path(__file__).resolve();HELPER=ROOT/'scripts/build_track3_coarse_objects.py';DIAGNOSTIC=ROOT/'scripts/recover_track3_canonical_collision.py'
PARAMS=dict(threshold=.001,real_metric=True,max_convex_hull=128,preprocess_mode='off',resolution=200,mcts_nodes=4,mcts_iterations=4,mcts_max_depth=2,pca=False,merge=True,decimate=True,max_ch_vertex=64,extrude=False,seed=18102026)
ALLOWED_QEM={'scripts/recover_track3_canonical_collision_qem.py','scripts/recover_track3_canonical_collision_qem_extra.py'}

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def bindings(source):
 # The parent writes terminal job only after wait()/kill/wait(). This prevents
 # consuming a cache while its original CoACD worker is still active.
 job=read(source/'job.json')
 assert job.get('state')in {'completed','failed'} and (source/'run.exit').exists(),'Original QEM/CoACD worker not terminal; do not run concurrently'
 proto=read(source/'protocol.json');qem=read(source/'qem-report.json');assert qem['state']=='completed'
 for key in ['watertight','winding_consistent','component_count_equal']:assert qem['topology'][key] is True
 fixes=[m for m in qem['matched_components']if not m['volume_orientation_equal']]
 assert all(m['euler_equal']for m in qem['matched_components'])
 if fixes:
  assert proto['name']=='mini_dust_pan' and {m['source_component']for m in fixes}=={26,46},'Only the two explicitly diagnosed tiny dustpan shells may transfer original winding'
  for m in fixes:
   before=qem['components_before'][m['source_component']];after=qem['components_after'][m['qem_component']]
   assert before['volume_m3']<0<after['volume_m3'] and after['faces']==4
 else:assert qem['topology']['matched_euler_and_orientation_preserved'] is True
 producer_path=proto['producer_path'];assert producer_path in ALLOWED_QEM
 assert sha(ROOT/producer_path)==proto['source_sha256'] and sha(HELPER)==proto['helper_sha256']
 producers={str(SELF.relative_to(ROOT)):sha(SELF),str(HELPER.relative_to(ROOT)):sha(HELPER),str(DIAGNOSTIC.relative_to(ROOT)):sha(DIAGNOSTIC)}
 for path,digest in proto['producer_source_sha256'].items():assert sha(ROOT/path)==digest;producers[path]=digest
 source_mesh=Path(proto['source_mesh']);assert sha(source_mesh)==proto['source_mesh_sha256']
 ck=read(source/'checkpoint.json');prior=read(source/'report.json')if(source/'report.json').exists()else{}
 previous_parameters=prior.get('coacd_parameters')or (ck.get('parameters')if ck.get('state')=='whole_oriented_mesh_CoACD'else None)or PARAMS
 assert previous_parameters==PARAMS,'No algorithm/threshold change permitted'
 paths=[source/'protocol.json',source/'qem-report.json',source/'collision_input_QEM.ply',source_mesh]
 return dict(source=str(source),source_mesh=str(source_mesh),name=proto['name'],mass_kg=proto['mass_kg'],source_producer_path=producer_path,source_producer_sha256=proto['source_sha256'],producer_source_sha256=producers,input_sha256={str(p):sha(p)for p in paths},coacd_parameters=PARAMS,qem_parameters=qem['parameters'],qem_topology=qem['topology'],orientation_transfers=fixes,source_completed_asset=bool(prior.get('state')=='completed'))

def worker(a):
 import numpy as np,trimesh,coacd
 from scipy.spatial import ConvexHull,QhullError
 from build_track3_coarse_objects import export_asset
 from recover_track3_canonical_collision import section_diagnostic,components
 D=a.output;start=time.monotonic();binding=read(D/'protocol.json')['binding'];source=Path(binding['source']);mesh=trimesh.load(binding['source_mesh'],force='mesh',process=False);simple=trimesh.load(source/'collision_input_QEM.ply',force='mesh',process=False)
 for path,digest in binding['input_sha256'].items():assert sha(path)==digest
 expected_faces=read(source/'qem-report.json')['collision_input']['faces'];assert len(simple.faces)==expected_faces
 assert np.isfinite(simple.vertices).all()and simple.is_watertight and simple.is_winding_consistent
 if binding['source_completed_asset']:
  previous=read(source/'report.json')
  for path,digest in previous['output_sha256'].items():assert sha(path)==digest
  result=dict(previous,producer_path=str(SELF.relative_to(ROOT)),source_sha256=sha(SELF),producer_source_sha256=binding['producer_source_sha256'],method='reuse_already_completed_QEM_recovery_no_new_CoACD',reused_completed_source=str(source/'report.json'),source_QEM_binding=binding,CoACD_restarted=False,QEM_recomputed=False,elapsed_seconds=time.monotonic()-start)
  save(D/'report.json',result);save(D/'checkpoint.json',dict(state='completed_reused_source',collision_parts=result['collision_parts']));return
 # Preserve the cached mesh byte-for-byte; transfer only explicitly matched
 # negative-shell orientation into a separate collision input. No vertex moves.
 orientation=[];qem=read(source/'qem-report.json');groups=components(simple);original_vertices=simple.vertices.copy();original_faces=simple.faces.copy();faces=original_faces.copy()
 for m in binding['orientation_transfers']:
  idx=m['qem_component'];g=groups[idx];sub=simple.submesh([g],append=True,repair=False);expected=qem['components_after'][idx]
  assert len(g)==expected['faces']==4 and sub.volume>0
  np.testing.assert_allclose(sub.centroid,expected['centroid'],atol=1e-7,rtol=0)
  faces[g]=faces[g][:,::-1]
  orientation.append(dict(**m,source_volume_m3=qem['components_before'][m['source_component']]['volume_m3'],cached_QEM_volume_m3=float(sub.volume),transferred_volume_m3=-float(sub.volume),changed_face_indices=g.tolist(),vertices_changed=False,scale_changed=False))
 if orientation:simple.faces=faces
 np.testing.assert_array_equal(simple.vertices,original_vertices)
 assert simple.is_watertight and simple.is_winding_consistent and len(components(simple))==len(groups)
 assert int(simple.euler_number)==qem['collision_input']['euler_number']
 for m in qem['matched_components']:
  sub=simple.submesh([groups[m['qem_component']]],append=True,repair=False)
  assert np.sign(sub.volume)==np.sign(qem['components_before'][m['source_component']]['volume_m3'])
 actual_input=D/'collision_input_oriented.ply';simple.export(actual_input)
 orientation_report=D/'orientation-transfer.json';save(orientation_report,dict(transfers=orientation,source_cache_unchanged=True,vertices_unchanged=True,component_count_unchanged=True,euler_unchanged=True,original_shell_signs_restored=True,known_cavity_shape_preserved_validated=False,input_sha256=sha(actual_input)))
 raw_cache=D/'coacd_parts.npz';raw_meta=D/'raw-parts.json'
 if raw_cache.exists()and raw_meta.exists():
  meta=read(raw_meta);assert sha(raw_cache)==meta['sha256'];assert meta['input_sha256']==binding['input_sha256'];assert meta['oriented_input_sha256']==sha(actual_input);z=np.load(raw_cache);raw=[(z[f'vertices_{i:04d}'],z[f'faces_{i:04d}'])for i in range(meta['parts'])];reused_raw=True
 else:
  save(D/'checkpoint.json',dict(state='CoACD_from_existing_QEM',input_faces=len(simple.faces),parameters=PARAMS,OMP_NUM_THREADS=16,elapsed_seconds=time.monotonic()-start))
  coacd.set_log_level('info');print('START fixed CoACD from completed QEM cache, OMP16,1200s worker cap',flush=True)
  raw=coacd.run_coacd(coacd.Mesh(np.asarray(simple.vertices),np.asarray(simple.faces)),**PARAMS)
  arrays={}
  for i,(v,f)in enumerate(raw):arrays[f'vertices_{i:04d}']=np.asarray(v);arrays[f'faces_{i:04d}']=np.asarray(f)
  temp=D/'coacd_parts.tmp.npz';np.savez_compressed(temp,**arrays);temp.replace(raw_cache)
  save(raw_meta,dict(state='CoACD_returned',parts=len(raw),sha256=sha(raw_cache),oriented_input_sha256=sha(actual_input),input_sha256=binding['input_sha256'],parameters=PARAMS,elapsed_seconds=time.monotonic()-start));reused_raw=False
 if len(raw)>128:raise RuntimeError('Actual CoACD parts exceed128; cached evidence retained, no URDF export')
 parts=[];discarded=[]
 for i,(v,f)in enumerate(raw):
  v=np.unique(np.asarray(v,dtype=float),axis=0)
  if not np.isfinite(v).all():raise RuntimeError('Nonfinite CoACD convex vertices')
  if len(v)<4 or np.linalg.matrix_rank(v-v.mean(0),tol=1e-12)<3:discarded.append(dict(index=i,vertices=len(v),reason='fewer_than4_or_affine_rank_below3'));continue
  try:ch=ConvexHull(v);part=trimesh.convex.convex_hull(v)
  except QhullError:discarded.append(dict(index=i,vertices=len(v),reason='Qhull_cannot_form_finite_solid_without_joggle'));continue
  if ch.volume<=0:discarded.append(dict(index=i,vertices=len(v),reason='nonpositive_convex_volume'));continue
  if part.volume<0:part.invert() # convex output hull only, never source negative shells
  assert part.is_watertight and part.volume>0 and np.isfinite(part.vertices).all();part.export(D/f'partial_collision_{len(parts):03d}.obj');parts.append(part)
 save(D/'collision-budget.json',dict(raw_parts=len(raw),actual_parts=len(parts),hard_global_budget=128,degenerate_omissions=discarded))
 assert 1<len(parts)<=128,'No single whole-object hull fallback; no geometry-count overflow'
 save(D/'checkpoint.json',dict(state='exporting',collision_parts=len(parts),elapsed_seconds=time.monotonic()-start))
 asset=export_asset(D/'assets',a.name,mesh,parts,binding['mass_kg']);u=ET.parse(asset['urdf_path']);mat=ET.SubElement(u.getroot().find('link/visual'),'material',name='canonical_uniform');ET.SubElement(mat,'color',rgba='0.65 0.62 0.55 1');u.write(asset['urdf_path'])
 visual=trimesh.load(asset['mesh_path'],force='mesh',process=False);np.testing.assert_array_equal(visual.faces,mesh.faces);np.testing.assert_allclose(visual.vertices,mesh.vertices,rtol=0,atol=5.1e-9)
 sections=section_diagnostic(mesh,parts)
 asset.update(reconstruction_source=str(D/'report.json'),source_inputs=[binding['source_mesh']],coordinate_origin='unchanged metric canonical link origin',scale_applied_again=False)
 files=[Path(asset['mesh_path']),Path(asset['urdf_path']),*(D/'assets').glob('collision_*.obj'),raw_cache,raw_meta,actual_input,orientation_report]
 for path,digest in binding['input_sha256'].items():assert sha(path)==digest
 report=dict(state='completed',name=a.name,created_utc=now(),producer_path=str(SELF.relative_to(ROOT)),source_sha256=sha(SELF),helper_sha256=sha(HELPER),producer_source_sha256=binding['producer_source_sha256'],source_mesh=binding['source_mesh'],source_mesh_sha256=sha(binding['source_mesh']),asset=asset,visual_mesh_unchanged=True,scale_applied_again=False,method='fixed_CoACD_restart_from_completed_whole_mesh_QEM_cache_with_larger_CPU_budget',source_QEM_binding=binding,qem_parameters=binding['qem_parameters'],coacd_parameters=PARAMS,input_faces=len(mesh.faces),collision_input_faces=len(simple.faces),collision_parts=len(parts),global_collision_part_budget=128,resource_budget_pass=True,input_negative_shells_preserved=True,whole_mesh_CoACD_not_component_union=True,topology=binding['qem_topology'],orientation_transfer=read(orientation_report),original_shell_signs_restored=True,collision_section_diagnostic=sections,degenerate_output_omissions=discarded,cavity_preservation_validated=False,production_pass=False,QEM_recomputed=False,model_recomputed=False,CPP_search_resumed=False,raw_return_cache_reused=reused_raw,OMP_NUM_THREADS=16,BLAS_threads=1,worker_cap_seconds=1200,elapsed_seconds=time.monotonic()-start,output_sha256={str(p):sha(p)for p in files})
 save(D/'report.json',report);save(D/'checkpoint.json',dict(state='completed',collision_parts=len(parts),elapsed_seconds=time.monotonic()-start));print(json.dumps(dict(state='completed',parts=len(parts),elapsed_seconds=time.monotonic()-start)),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--name',choices=['plastic_dish_rack','mini_dust_pan','white_pot_lid'],required=True);p.add_argument('--source',type=Path);p.add_argument('--output',type=Path);p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
 source_area='canonical_collision_qem_extra_v1'if a.name=='white_pot_lid'else'canonical_collision_qem_v1';a.source=(a.source or BASE/source_area/a.name).resolve();a.output=(a.output or BASE/'canonical_collision_postqem_v1'/a.name).resolve();assert a.output!=a.source
 a.output.mkdir(parents=True,exist_ok=True)
 if a.worker:worker(a);return
 with (a.output/'job.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  # Refuse concurrently active input producers; root queues only terminal jobs.
  binding=bindings(a.source);assert binding['name']==a.name
  protocol=dict(name=a.name,binding=binding,worker_hard_cap_seconds=1200,OMP_NUM_THREADS=16,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,OMP_MAX_ACTIVE_LEVELS=1,coacd_parameters_unchanged=True,QEM_recomputed=False,CPP_search_resumed=False)
  if(a.output/'protocol.json').exists():assert read(a.output/'protocol.json')==protocol,'Frozen post-QEM inputs changed'
  else:save(a.output/'protocol.json',protocol)
  if(a.output/'report.json').exists():
   d=read(a.output/'report.json')
   for path,h in d['output_sha256'].items():assert sha(path)==h
   print('Existing completed post-QEM result retained');return
  if(a.output/'failure.json').exists():print('Existing terminal post-QEM failure retained',file=sys.stderr);sys.exit(2)
  cmd=[sys.executable,'-u',str(SELF),'--worker','--name',a.name,'--source',str(a.source),'--output',str(a.output)];env=dict(os.environ,OMP_NUM_THREADS='16',OMP_DYNAMIC='FALSE',OMP_MAX_ACTIVE_LEVELS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');started=time.monotonic()
  with(a.output/'worker.log').open('a')as log:
   child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save(a.output/'job.json',dict(state='running',started_utc=now(),pid=child.pid,command=cmd,max_seconds=1200,OMP_NUM_THREADS=16))
   try:code=child.wait(timeout=1200)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait();code=124
   except BaseException:os.killpg(child.pid,signal.SIGKILL);child.wait();raise
  (a.output/'run.exit').write_text(str(code)+'\n')
  if code:save(a.output/'failure.json',dict(state='timeout'if code==124 else'failed',exit_code=code,elapsed_seconds=time.monotonic()-started,automatic_retry=False,partial_evidence_preserved=True))
  save(a.output/'job.json',dict(state='completed'if code==0 else'failed',exit_code=code,completed_utc=now(),report=str(a.output/'report.json')))
  if code:sys.exit(code)
if __name__=='__main__':main()
