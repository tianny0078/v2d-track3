"""One whole-oriented-mesh QEM + globally bounded CoACD recovery.
The original visual/scale/origin is immutable. Inner negative shells stay together
with the positive outer shell; they are never decomposed as additive solids.
"""
from pathlib import Path
import argparse,datetime,fcntl,hashlib,importlib.metadata,json,os,signal,subprocess,sys,time
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/track3/advice_restart_20261017'
SELF=Path(__file__).resolve()
HELPER=ROOT/'scripts/build_track3_coarse_objects.py'
DIAGNOSTIC=ROOT/'scripts/recover_track3_canonical_collision.py'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
def producer_map():return {str(p.relative_to(ROOT)):sha(p)for p in [SELF,HELPER,DIAGNOSTIC]}
def versions():
 return {p:importlib.metadata.version(p)for p in ['pymeshlab','coacd','trimesh','numpy','scipy']}

def worker(a):
 import numpy as np,trimesh,coacd,pymeshlab
 from scipy.optimize import linear_sum_assignment
 from scipy.spatial import ConvexHull,QhullError
 from recover_track3_canonical_collision import components,stats,section_diagnostic
 from build_track3_coarse_objects import export_asset
 D=a.output;start=time.monotonic();mesh=trimesh.load(a.mesh,force='mesh',process=False)
 assert np.isfinite(mesh.vertices).all()and mesh.is_watertight and mesh.is_winding_consistent
 def component_info(m):
  out=[]
  for f in components(m):
   sub=m.submesh([f],append=True,repair=False)
   out.append(dict(faces=len(sub.faces),euler=int(sub.euler_number),volume_m3=float(sub.volume),centroid=sub.centroid.tolist(),extents_m=sub.extents.tolist()))
  return out
 before=component_info(mesh)
 save(D/'input-inventory.json',dict(source_mesh=str(a.mesh),source_mesh_sha256=sha(a.mesh),mesh=stats(mesh),components=before,negative_shell_count=sum(x['volume_m3']<0 for x in before),input_orientation_unchanged=True))
 qem=dict(targetfacenum=20000,targetperc=0.0,qualitythr=.3,preserveboundary=True,boundaryweight=1.,preservenormal=True,preservetopology=True,optimalplacement=False,planarquadric=True,planarweight=.001,qualityweight=False,autoclean=True,selected=False)
 save(D/'checkpoint.json',dict(state='whole_mesh_QEM',parameters=qem,source_faces=len(mesh.faces),elapsed_seconds=time.monotonic()-start))
 ms=pymeshlab.MeshSet();ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(mesh.vertices),face_matrix=np.asarray(mesh.faces,dtype=np.int32)))
 if len(mesh.faces)>20000:ms.apply_filter('meshing_decimation_quadric_edge_collapse',**qem)
 simple=trimesh.Trimesh(np.asarray(ms.current_mesh().vertex_matrix()),np.asarray(ms.current_mesh().face_matrix()),process=False)
 simple.export(D/'collision_input_QEM.ply');after=component_info(simple)
 topology=dict(watertight=bool(simple.is_watertight),winding_consistent=bool(simple.is_winding_consistent),component_count_equal=len(before)==len(after),negative_shell_count_before=sum(x['volume_m3']<0 for x in before),negative_shell_count_after=sum(x['volume_m3']<0 for x in after))
 pairs=[]
 if len(before)==len(after):
  x=np.array([v['centroid']for v in before]);y=np.array([v['centroid']for v in after]);cost=np.linalg.norm(x[:,None]-y[None],axis=-1);ii,jj=linear_sum_assignment(cost)
  for i,j in zip(ii,jj):pairs.append(dict(source_component=int(i),qem_component=int(j),centroid_shift_m=float(cost[i,j]),euler_equal=before[i]['euler']==after[j]['euler'],volume_orientation_equal=bool(np.sign(before[i]['volume_m3'])==np.sign(after[j]['volume_m3']))))
 topology['matched_euler_and_orientation_preserved']=bool(pairs and all(p['euler_equal']and p['volume_orientation_equal']for p in pairs))
 save(D/'qem-report.json',dict(state='completed',parameters=qem,source=stats(mesh),collision_input=stats(simple),components_before=before,components_after=after,matched_components=pairs,topology=topology,visual_changed=False,scale_changed=False,elapsed_seconds=time.monotonic()-start))
 if not all(topology[k]for k in ['watertight','winding_consistent','component_count_equal','matched_euler_and_orientation_preserved']):raise RuntimeError('Single QEM changed closed topology/orientation; preserve evidence, no repair loop or original-large-mesh retry')
 params=dict(threshold=.001,real_metric=True,max_convex_hull=128,preprocess_mode='off',resolution=200,mcts_nodes=4,mcts_iterations=4,mcts_max_depth=2,pca=False,merge=True,decimate=True,max_ch_vertex=64,extrude=False,seed=18102026)
 save(D/'checkpoint.json',dict(state='whole_oriented_mesh_CoACD',input_faces=len(simple.faces),parameters=params,elapsed_seconds=time.monotonic()-start))
 coacd.set_log_level('warn');raw=coacd.run_coacd(coacd.Mesh(np.asarray(simple.vertices),np.asarray(simple.faces)),**params)
 save(D/'raw-coacd-count.json',dict(returned_parts=len(raw),hard_global_part_budget=128,returned_within_budget=len(raw)<=128,whole_mesh_material_boundary=True,elapsed_seconds=time.monotonic()-start))
 if len(raw)>128:raise RuntimeError('Actual CoACD returned more than global128 runtime budget; no URDF export')
 parts=[];discarded=[]
 for i,(v,f)in enumerate(raw):
  v=np.unique(np.asarray(v,dtype=float),axis=0)
  if not np.isfinite(v).all():raise RuntimeError('Nonfinite CoACD convex vertices')
  # CoACD occasionally emits zero-volume planar remnants. They cannot represent
  # physical solid hulls. Rebuild faces from the returned convex vertices without
  # perturbing/expanding them; record only mathematically degenerate omissions.
  if len(v)<4 or np.linalg.matrix_rank(v-v.mean(0),tol=1e-12)<3:
   discarded.append(dict(index=i,vertices=len(v),reason='fewer_than4_or_affine_rank_below3'));continue
  try:
   ch=ConvexHull(v);part=trimesh.convex.convex_hull(v)
  except QhullError:
   discarded.append(dict(index=i,vertices=len(v),reason='Qhull_cannot_form_finite_solid_without_joggle'));continue
  if ch.volume<=0:
   discarded.append(dict(index=i,vertices=len(v),reason='nonpositive_convex_volume'));continue
  if part.volume<0:part.invert() # output convex solid only; never original negative input shells
  assert part.is_watertight and part.volume>0 and np.isfinite(part.vertices).all()
  part.export(D/f'partial_collision_{len(parts):03d}.obj');parts.append(part)
 save(D/'collision-budget.json',dict(raw_parts=len(raw),actual_parts=len(parts),hard_global_budget=128,degenerate_omissions=discarded))
 assert 1<len(parts)<=128,'No global single-hull fallback, no runtime geom-count overflow'
 save(D/'checkpoint.json',dict(state='exporting',collision_parts=len(parts),elapsed_seconds=time.monotonic()-start))
 asset=export_asset(D/'assets',a.name,mesh,parts,a.mass)
 u=ET.parse(asset['urdf_path']);mat=ET.SubElement(u.getroot().find('link/visual'),'material',name='canonical_uniform');ET.SubElement(mat,'color',rgba='0.65 0.62 0.55 1');u.write(asset['urdf_path'])
 visual=trimesh.load(asset['mesh_path'],force='mesh',process=False);np.testing.assert_array_equal(visual.faces,mesh.faces);np.testing.assert_allclose(visual.vertices,mesh.vertices,rtol=0,atol=5.1e-9)
 protocol=json.loads((D/'protocol.json').read_text());assert sha(a.mesh)==protocol['source_mesh_sha256']
 sections=section_diagnostic(mesh,parts)
 asset.update(reconstruction_source=str(D/'report.json'),source_inputs=[str(a.mesh)],coordinate_origin='unchanged metric canonical link origin',scale_applied_again=False)
 files=[Path(asset['mesh_path']),Path(asset['urdf_path']),*(D/'assets').glob('collision_*.obj')]
 report=dict(state='completed',name=a.name,created_utc=now(),producer_path=str(SELF.relative_to(ROOT)),source_sha256=sha(SELF),helper_sha256=sha(HELPER),producer_source_sha256=producer_map(),source_mesh=str(a.mesh),source_mesh_sha256=sha(a.mesh),asset=asset,visual_mesh_unchanged=True,scale_applied_again=False,method='one_whole_oriented_mesh_topology_preserving_QEM_then_global_merged_CoACD',qem_parameters=qem,coacd_parameters=params,input_faces=len(mesh.faces),collision_input_faces=len(simple.faces),collision_parts=len(parts),global_collision_part_budget=128,resource_budget_pass=True,input_negative_shells_preserved=True,whole_mesh_CoACD_not_component_union=True,topology=topology,collision_section_diagnostic=sections,degenerate_output_omissions=discarded,cavity_preservation_validated=False,production_pass=False,versions=versions(),elapsed_seconds=time.monotonic()-start,output_sha256={str(p):sha(p)for p in files})
 save(D/'report.json',report);save(D/'checkpoint.json',dict(state='completed',collision_parts=len(parts),elapsed_seconds=time.monotonic()-start));print(json.dumps(dict(state='completed',parts=len(parts),input_faces=len(simple.faces),elapsed_seconds=time.monotonic()-start)),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--name',choices=['plastic_dish_rack','mini_dust_pan','white_pot_lid','wooden_piece_1','wooden_piece_2'],required=True);p.add_argument('--mesh',type=Path);p.add_argument('--output',type=Path);p.add_argument('--mass',type=float);p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
 a.mesh=(a.mesh or BASE/'canonical9_metric_v1/objects'/a.name/'metric_canonical.ply').resolve();a.output=(a.output or BASE/'canonical_collision_qem_extra_v1'/a.name).resolve();a.output.mkdir(parents=True,exist_ok=True)
 if a.mass is None:
  rows=json.loads((BASE/'all20-registration-inputs-v1/input-manifest.json').read_text())['rows'];row=next(x for x in rows if x['name']==a.name);objects=json.loads((ROOT/'outputs/track3'/row['episode']/'objects/objects.json').read_text())['objects'];a.mass=float(next(x for x in objects if x['name']==a.name).get('mass_kg',.15))
 if a.worker:worker(a);return
 with (a.output/'job.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  protocol=dict(name=a.name,source_mesh=str(a.mesh),source_mesh_sha256=sha(a.mesh),producer_path=str(SELF.relative_to(ROOT)),source_sha256=sha(SELF),helper_sha256=sha(HELPER),producer_source_sha256=producer_map(),mass_kg=a.mass,target_faces=20000,worker_hard_cap_seconds=285,global_collision_part_budget=128,versions=versions(),GT_read=False,visual_scale_origin_frozen=True)
  if (a.output/'protocol.json').exists():assert json.loads((a.output/'protocol.json').read_text())==protocol,'Frozen recovery changed; use separate output'
  else:save(a.output/'protocol.json',protocol)
  if (a.output/'report.json').exists():
   d=json.loads((a.output/'report.json').read_text())
   for path,h in d['output_sha256'].items():assert sha(path)==h
   print('Existing matching completed QEM recovery retained');return
  if (a.output/'failure.json').exists():print('Existing terminal QEM failure retained',file=sys.stderr);sys.exit(2)
  cmd=[sys.executable,'-u',str(SELF),'--worker','--name',a.name,'--mesh',str(a.mesh),'--output',str(a.output),'--mass',str(a.mass)];env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2');started=time.monotonic()
  with (a.output/'worker.log').open('a')as log:
   child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);save(a.output/'job.json',dict(state='running',started_utc=now(),pid=child.pid,command=cmd,max_seconds=285))
   try:code=child.wait(timeout=285)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait();code=124
   except BaseException:os.killpg(child.pid,signal.SIGKILL);child.wait();raise
  (a.output/'run.exit').write_text(str(code)+'\n')
  if code:save(a.output/'failure.json',dict(state='timeout'if code==124 else'failed',exit_code=code,elapsed_seconds=time.monotonic()-started,automatic_retry=False,partial_evidence_preserved=True))
  save(a.output/'job.json',dict(state='completed'if code==0 else'failed',exit_code=code,completed_utc=now(),report=str(a.output/'report.json')))
  if code:sys.exit(code)
if __name__=='__main__':main()
