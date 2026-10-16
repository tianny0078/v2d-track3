"""Provisional geometry-only contact/wrench uncertainty using official CHORD GPU kernels.

No force measurements or inferred ground-truth contact labels. Keep missing hand
observations explicitly unknown, and report sensitivity before contact fitting.
"""
from datetime import datetime,timezone
import hashlib,json,time
from pathlib import Path
import numpy as np,trimesh,warp as wp
from scipy.spatial import cKDTree
from flash_chord.objectives.wrench import sample_wrench_basis,friction_cone_angles,wrench_support

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/track3/episode_000011';OUT=BASE/'contact_wrench_probe'
SOURCE=BASE/'hand_depth_tradeoff/alpha_050/hands_world.npz';GEO=BASE/'agent_cup_reconstruction'
NAMES=['blue_cup','beige_cup'];THRESHOLDS=[.005,.01,.02];MUS=[.1,.3,.6]
LINKS=[[0,1,5,9,13,17],[1,2],[2,3],[3,4],[5,6],[6,7],[7,8],[9,10],[10,11],[11,12],[13,14],[14,15],[15,16],[17,18],[18,19],[19,20]]

def save(path,r):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def main():
    OUT.mkdir(exist_ok=True);start=time.monotonic();wp.init();device='cuda:0'
    hands=np.load(SOURCE);present=hands['present'].astype(bool);verts=hands['vertices_world'];joints=hands['joints_world'];poses=np.load(GEO/'candidate_poses.npz')['world_T_canonical_initial']
    p=np.zeros((137,2,2,16,3),np.float32);n=np.zeros_like(p);dist=np.full((137,2,2,16),np.nan,np.float32)
    rc=[];centers=[];bases=[];sources={str(SOURCE.relative_to(ROOT)):hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
    for oi,name in enumerate(NAMES):
        file=GEO/name/'wall_03mm_3000um.ply';mesh=trimesh.load(file,process=False);sources[str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
        com=np.array(mesh.center_mass);radius=float(np.linalg.norm(mesh.vertices-com,axis=1).max());centers.append(com);rc.append(radius)
        # Radius normalization is performed once inside the official kernel.
        bases.append(sample_wrench_basis(512,1.,seed=oi));np.random.seed(400+oi)
        surface,face=trimesh.sample.sample_surface(mesh,60000);normal=-mesh.face_normals[face];tree=cKDTree(surface)
        for f in range(137):
            T=poses[f,oi]
            for hi in range(2):
                if not present[f,hi]:continue
                local=(verts[f,hi]-T[:3,3])@T[:3,:3];distance,index=tree.query(local,workers=4)
                centers_world=np.stack([joints[f,hi,link].mean(0) for link in LINKS]);assignment=cKDTree(centers_world).query(verts[f,hi])[1]
                for li in range(16):
                    candidates=np.flatnonzero(assignment==li)
                    if not len(candidates):continue
                    vi=candidates[np.argmin(distance[candidates])];si=index[vi]
                    p[f,hi,oi,li]=surface[si]-com;n[f,hi,oi,li]=normal[si];dist[f,hi,oi,li]=distance[vi]
    output=np.full((len(THRESHOLDS),len(MUS),137,2,2,512),np.nan,np.float32);counts=[]
    cos,sin=friction_cone_angles(8);cos_d=wp.array(cos,dtype=wp.float32,device=device);sin_d=wp.array(sin,dtype=wp.float32,device=device)
    enabled=wp.ones(16,dtype=wp.int32,device=device)
    for ti,threshold in enumerate(THRESHOLDS):
        active=np.isfinite(dist)&(dist<threshold);count=active.sum(-1)
        counts.append({'threshold_m':threshold,'active_link_count':int(active.sum()),'present_hand_object_cells_with_proximity':int(((count>0)&present[:,:,None]).sum()),
                       'present_hand_object_cells':int(present.sum()*2),'unknown_missing_hand_object_cells':int((~present).sum()*2),
                       'frames_with_any_proximity':int((count>0).any(axis=(1,2)).sum())})
        for hi in range(2):
            for oi in range(2):
                pp=wp.array(p[:,hi,oi].reshape(-1,3),dtype=wp.vec3,device=device)
                nn=wp.array((n[:,hi,oi]*active[:,hi,oi,:,None]).reshape(-1,3),dtype=wp.vec3,device=device)
                basis=wp.array(np.tile(bases[oi],(137,1)),dtype=wp.spatial_vector,device=device);radius=wp.array(np.full(137,rc[oi],np.float32),dtype=wp.float32,device=device)
                for mi,mu in enumerate(MUS):
                    support=wp.zeros(137*512,dtype=wp.float32,device=device)
                    wp.launch(wrench_support,dim=137*512,inputs=[pp,nn,enabled,basis,cos_d,sin_d,mu,radius,16,8,512],outputs=[support],device=device)
                    value=support.numpy().reshape(137,512);value[~present[:,hi]]=np.nan;output[ti,mi,:,hi,oi]=value
    finite=output[np.isfinite(output)];assert np.all(finite>=0)
    # The same cone phase angles are reused. Each ray rotates with mu, so a finite
    # cone approximation need not make every support coordinate monotone in mu.
    np.savez_compressed(OUT/'provisional_reference.npz',contact_pos_object_COM=p,contact_normal_inward=n,nearest_link_distance_m=dist,
                        observed_hand=present,thresholds_m=THRESHOLDS,friction_coefficients=MUS,support=output,shared_basis=np.array(bases),
                        bounding_radius_m=rc,canonical_COM_m=centers,object_names=np.array(NAMES),frame_index=np.arange(137))
    low=np.isfinite(dist)&(dist<THRESHOLDS[0]);high=np.isfinite(dist)&(dist<THRESHOLDS[-1]);added=int(high.sum()-low.sum())
    report={'state':'completed','completed_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start,
            'source_sha256':sources,'device':device,'official_kernel':'flash_chord.objectives.wrench.wrench_support',
            'contact_method':'Nearest sampled object surface pair per nearest-joint MANO link; provisional geometric proximity only',
            'pose_source':'v3 initial fixed-shape candidate; known geometric quality failures retained','threshold_sensitivity':counts,
            'additional_active_links_5_to_20mm':added,'friction_coefficients':MUS,'basis_directions_per_object':512,'cone_edges_plus_normal':9,
            'radius_normalization':'Basis sampled with rc=1; cross-product moment divided by COM bounding radius exactly once inside official kernel',
            'normal_convention':'Force directed into object material: negative mesh outward normal, including cavity wall',
            'uniform_material_COM_assumption':True,'missing_hand_support_is_nan':bool(np.isnan(output[:,:,~present]).all()),
            'support_finite_nonnegative':bool(np.all(finite>=0)),'geometry_quality_pass':False,'contact_ready':False,
            'limitations':['Unit contact wrench support is a geometric capability proxy, not measured force or net force/torque reconstruction.',
                           'Object shape/poses and alpha_050 hands have unresolved uncertainty. No reference automatically promoted.',
                           'Each link uses one nearest sampled pair; contact persistence/normal agreement not yet fitted.',
                           'Friction coefficient and 3mm wall/bottom thickness are assumed; thresholds sweep 5/10/20mm.',
                           'Human hand absence is unknown, not a negative contact label. Present zero-support only means no proximity under this candidate/threshold.']}
    save(OUT/'report.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
