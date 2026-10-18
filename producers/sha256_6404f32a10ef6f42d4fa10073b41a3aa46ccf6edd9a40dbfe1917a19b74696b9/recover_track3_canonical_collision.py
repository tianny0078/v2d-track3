"""One bounded collision-only recovery; never alters frozen canonical visuals/poses.

Root runs this entry in server tmux. The internal worker is killed at the declared
wall limit; failed output is terminal and requires an explicitly new directory.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path('/home/tianyuan/projects/v2d')
BASE = ROOT/'outputs/track3/advice_restart_20261017'
EXPORTER = ROOT/'scripts/build_track3_coarse_objects.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def components(mesh):
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    faces = mesh.faces
    edges = np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    graph = coo_matrix((np.ones(len(edges),np.int8),(edges[:,0],edges[:,1])),shape=(len(mesh.vertices),)*2)
    count,labels = connected_components(graph,directed=False)
    groups = [np.flatnonzero(labels[faces[:,0]]==i) for i in range(count)]
    return sorted([g for g in groups if len(g)],key=len,reverse=True)


def stats(mesh):
    return dict(vertices=len(mesh.vertices),faces=len(mesh.faces),watertight=bool(mesh.is_watertight),
                winding_consistent=bool(mesh.is_winding_consistent),euler_number=int(mesh.euler_number),
                extents_m=mesh.extents.tolist(),volume_m3=float(mesh.volume))


def cluster_once(mesh, cell):
    """One local vertex clustering pass; use only if closed topology is retained."""
    import numpy as np
    import trimesh
    cells=np.floor(np.asarray(mesh.vertices)/cell+.5).astype(np.int64)
    _,inverse=np.unique(cells,axis=0,return_inverse=True)
    counts=np.bincount(inverse)
    vertices=np.stack([np.bincount(inverse,weights=mesh.vertices[:,i])/counts for i in range(3)],axis=1)
    faces=inverse[mesh.faces]
    nondegenerate=(faces[:,0]!=faces[:,1])&(faces[:,1]!=faces[:,2])&(faces[:,0]!=faces[:,2])
    faces=faces[nondegenerate]
    _,unique=np.unique(np.sort(faces,axis=1),axis=0,return_index=True)
    faces=faces[np.sort(unique)]
    candidate=trimesh.Trimesh(vertices,faces,process=False)
    candidate.remove_unreferenced_vertices()
    equivalent=(candidate.is_watertight and candidate.is_winding_consistent and
                candidate.euler_number==mesh.euler_number and len(components(candidate))==1)
    displacement=float(np.linalg.norm(mesh.vertices-vertices[inverse],axis=1).max())
    return (candidate if equivalent else mesh),dict(method='single_vertex_clustering',cell_size_m=cell,
        applied=bool(equivalent),topology_matches=bool(equivalent),source=stats(mesh),candidate=stats(candidate),
        source_vertex_max_displacement_m=displacement,
        fallback_reason=None if equivalent else 'Retain this original component because clustering changed closed topology')


def section_diagnostic(mesh, parts):
    """Deterministic occupancy evidence, not a declaration of all-hole preservation."""
    import numpy as np
    from scipy.spatial import ConvexHull
    equations=[ConvexHull(p.vertices).equations for p in parts]
    rng=np.random.default_rng(18102026);triangles=mesh.triangles;reports=[]
    for axis in range(3):
        other=[x for x in range(3) if x!=axis];lo,hi=mesh.bounds
        xy=rng.uniform(lo[other],hi[other],(12,2));add=miss=inside_count=empty_count=0
        for point in xy:
            u=triangles[:,:,other[0]];v=triangles[:,:,other[1]]
            denominator=(v[:,1]-v[:,2])*(u[:,0]-u[:,2])+(u[:,2]-u[:,1])*(v[:,0]-v[:,2])
            a=np.divide((v[:,1]-v[:,2])*(point[0]-u[:,2])+(u[:,2]-u[:,1])*(point[1]-v[:,2]),denominator,out=np.full(len(triangles),np.nan),where=abs(denominator)>1e-15)
            b=np.divide((v[:,2]-v[:,0])*(point[0]-u[:,2])+(u[:,0]-u[:,2])*(point[1]-v[:,2]),denominator,out=np.full(len(triangles),np.nan),where=abs(denominator)>1e-15)
            c=1-a-b;valid=(a>=0)&(b>=0)&(c>=0)
            hits=np.sort((a*triangles[:,0,axis]+b*triangles[:,1,axis]+c*triangles[:,2,axis])[valid])
            # Deduplicate shared-edge hits at numerical precision.
            hits=hits[np.r_[True,np.diff(hits)>1e-9]] if len(hits) else hits
            x=np.linspace(lo[axis],hi[axis],128);original=np.searchsorted(hits,x)%2==1
            p=np.zeros((len(x),3));p[:,axis]=x;p[:,other]=point
            collision=np.zeros(len(x),bool)
            for eq in equations:collision|=(p@eq[:,:3].T+eq[:,3]<=1e-9).all(axis=1)
            add+=int((collision&~original).sum());miss+=int((original&~collision).sum())
            inside_count+=int(original.sum());empty_count+=int((~original).sum())
        reports.append(dict(axis=axis,rays=12,samples_per_ray=128,original_solid_samples=inside_count,
            original_empty_samples=empty_count,original_empty_filled_samples=add,original_solid_missed_samples=miss,
            sampled_empty_filled_fraction=add/max(empty_count,1),sampled_solid_missed_fraction=miss/max(inside_count,1)))
    return reports


def worker(args):
    import numpy as np
    import trimesh
    import coacd
    from build_track3_coarse_objects import export_asset
    started=time.monotonic();out=args.output
    source_mesh=trimesh.load(args.mesh,force='mesh',process=False)
    assert np.isfinite(source_mesh.vertices).all() and source_mesh.is_watertight and len(source_mesh.faces)>0
    groups=components(source_mesh)
    save(out/'inventory.json',dict(source_mesh=str(args.mesh),source_mesh_sha256=sha(args.mesh),
        **stats(source_mesh),connected_components=len(groups),component_face_counts=[len(x) for x in groups]))
    params=dict(threshold=.001,real_metric=True,max_convex_hull=128,preprocess_mode='off',resolution=200,
        mcts_nodes=4,mcts_iterations=4,mcts_max_depth=2,pca=False,merge=False,decimate=False,seed=18102026)
    parts=[];simplification=[];component_results=[]
    save(out/'checkpoint.json',dict(state='preprocessing',started_utc=now(),component_count=len(groups),parameters=params))
    coacd.set_log_level('warn')
    for index,faces in enumerate(groups):
        component=source_mesh.submesh([faces],append=True,repair=False)
        collision_input,simplified=cluster_once(component,args.cell_size)
        simplification.append(simplified)
        collision_input.export(out/f'collision_input_component_{index:03d}.ply')
        save(out/'checkpoint.json',dict(state='decomposing',current_component=index,completed_components=len(component_results),
            component_count=len(groups),collision_parts=len(parts),elapsed_seconds=time.monotonic()-started,
            simplification=simplification,component_results=component_results,parameters=params))
        component_start=time.monotonic()
        raw=coacd.run_coacd(coacd.Mesh(collision_input.vertices,collision_input.faces),**params)
        component_parts=[]
        for vertices,triangles in raw:
            part=trimesh.Trimesh(vertices,triangles,process=False)
            if part.volume<0:part.invert()
            assert part.is_watertight and part.volume>0 and np.isfinite(part.vertices).all()
            part.export(out/f'partial_collision_{len(parts)+len(component_parts):03d}.obj')
            component_parts.append(part)
        assert component_parts
        parts.extend(component_parts)
        component_results.append(dict(component=index,input_faces=len(collision_input.faces),parts=len(component_parts),
                                      elapsed_seconds=time.monotonic()-component_start))
    assert len(parts)>1, 'Never substitute one whole-object convex hull'
    if len(groups[0])>.5*len(source_mesh.faces):
        assert component_results[0]['parts']>1, 'Main connected rack structure cannot become one hull'
    save(out/'checkpoint.json',dict(state='exporting',collision_parts=len(parts),component_results=component_results,
                                   simplification=simplification,elapsed_seconds=time.monotonic()-started))
    asset=export_asset(out/'assets',args.name,source_mesh,parts,args.mass)
    tree=ET.parse(asset['urdf_path']);visual=tree.getroot().find('link/visual')
    material=ET.SubElement(visual,'material',name='canonical_uniform')
    ET.SubElement(material,'color',rgba='0.65 0.62 0.55 1');tree.write(asset['urdf_path'])
    # Verify exporter retained the original metric visual array/order within OBJ precision.
    visual_mesh=trimesh.load(asset['mesh_path'],force='mesh',process=False)
    np.testing.assert_array_equal(visual_mesh.faces,source_mesh.faces)
    np.testing.assert_allclose(visual_mesh.vertices,source_mesh.vertices,rtol=0,atol=5.1e-9)
    assert sha(args.mesh)==json.loads((out/'protocol.json').read_text())['source_mesh_sha256']
    sections=section_diagnostic(source_mesh,parts)
    volume=sum(abs(p.volume) for p in parts)
    asset.update(reconstruction_source=str(out/'report.json'),source_inputs=[str(args.mesh)],
                 coordinate_origin='unchanged metric canonical link origin',scale_applied_again=False)
    files=[Path(asset['mesh_path']),Path(asset['urdf_path']),*(out/'assets').glob('collision_*.obj')]
    report=dict(state='completed',created_utc=now(),name=args.name,asset=asset,source_mesh=str(args.mesh),
        source_mesh_sha256=sha(args.mesh),source_sha256=sha(__file__),helper_sha256=sha(EXPORTER),
        method='component_separated_single_clustering_then_low_search_CoACD',parameters=params,
        collision_input_cell_size_m=args.cell_size,simplification=simplification,component_results=component_results,
        connected_components=len(groups),collision_parts=len(parts),visual_mesh_unchanged=True,scale_applied_again=False,
        collision_volume_sum_m3=volume,visual_volume_m3=float(source_mesh.volume),
        collision_volume_relative_difference=volume/abs(source_mesh.volume)-1,
        collision_section_diagnostic=sections,cavity_preservation_validated=False,production_pass=False,
        elapsed_seconds=time.monotonic()-started,output_sha256={str(p):sha(p) for p in files})
    save(out/'report.json',report)
    save(out/'checkpoint.json',dict(state='completed',report=str(out/'report.json'),elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(state='completed',collision_parts=len(parts),elapsed_seconds=report['elapsed_seconds'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',default='plastic_dish_rack')
    p.add_argument('--mesh',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--mass',type=float)
    p.add_argument('--cell-size',type=float,default=.0005)
    p.add_argument('--max-seconds',type=float,default=285)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    args.mesh=(args.mesh or BASE/'canonical9_metric_v1/objects'/args.name/'metric_canonical.ply').resolve()
    args.output=(args.output or BASE/'canonical_collision_recovery_v1'/args.name).resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    assert 0<args.cell_size<=.001 and 1<=args.max_seconds<=290
    if args.mass is None:
        rows=json.loads((BASE/'all20-registration-inputs-v1/input-manifest.json').read_text())['rows']
        row=next(r for r in rows if r['name']==args.name)
        objects=json.loads((ROOT/'outputs/track3'/row['episode']/'objects/objects.json').read_text())['objects']
        args.mass=float(next(o for o in objects if o['name']==args.name).get('mass_kg',.15))
    if args.worker:
        worker(args);return
    with (args.output/'job.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol=dict(source_mesh=str(args.mesh),source_mesh_sha256=sha(args.mesh),source_sha256=sha(__file__),
            helper_sha256=sha(EXPORTER),name=args.name,mass_kg=args.mass,cell_size_m=args.cell_size,
            worker_wall_cap_seconds=args.max_seconds,unchanged_visual=True,GT_read=False,production_pass=False)
        if (args.output/'protocol.json').exists():
            assert json.loads((args.output/'protocol.json').read_text())==protocol, 'Frozen inputs changed; choose new output'
        else:save(args.output/'protocol.json',protocol)
        if (args.output/'report.json').exists():
            report=json.loads((args.output/'report.json').read_text())
            for path,digest in report['output_sha256'].items():assert sha(path)==digest
            print('Completed matching recovery reused; no computation repeated');return
        if (args.output/'failure.json').exists():
            print('Existing terminal failure retained; no automatic retry',file=sys.stderr);sys.exit(2)
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker','--name',args.name,'--mesh',str(args.mesh),
            '--output',str(args.output),'--mass',str(args.mass),'--cell-size',str(args.cell_size),'--max-seconds',str(args.max_seconds)]
        env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        started=time.monotonic()
        with (args.output/'worker.log').open('a') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(args.output/'job.json',dict(state='running',pid=process.pid,started_utc=now(),command=command,
                 output=str(args.output),log=str(args.output/'worker.log'),max_seconds=args.max_seconds))
            try:code=process.wait(timeout=args.max_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait();code=124
            except BaseException:
                os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=5);raise
        (args.output/'run.exit').write_text(str(code)+'\n')
        if code:
            save(args.output/'failure.json',dict(state='timeout' if code==124 else 'failed',exit_code=code,
                 elapsed_seconds=time.monotonic()-started,partial_checkpoint=str(args.output/'checkpoint.json'),
                 automatic_retry=False,visual_or_original_reports_modified=False))
        save(args.output/'job.json',dict(state='completed' if code==0 else 'failed',exit_code=code,
            finished_utc=now(),elapsed_seconds=time.monotonic()-started,production_pass=False))
        sys.exit(code)


if __name__=='__main__':
    main()
