"""Metric, observed-surface TSDF fusion from the diagnostic rigid tracks."""
import json,time
from pathlib import Path
import cv2
import numpy as np
from scipy import ndimage
from skimage.measure import marching_cubes
import trimesh
from run_object_surface_tracks import cloud
from download_track3_sample import digest,now,save

ROOT=Path('/work');BASE=ROOT/'outputs/track3/episode_000011';OUT=BASE/'object_surface_fusion'
RESOLUTION=192;TRUNCATION=.008


def main():
    cv2.setNumThreads(4);OUT.mkdir(exist_ok=True);started=time.perf_counter()
    sources=[BASE/'object_surface_tracks/report.json',BASE/'object_surface_tracks/poses.npz',BASE/'foundation_stereo_sequence/report.json',BASE/'hand_masks_sequence/report.json']
    hashes=[digest(p) for p in sources]
    assert json.loads(sources[0].read_text())['state']=='completed'
    tracks=np.load(sources[1],allow_pickle=False);camera=np.load(BASE/'stereo_sequence/camera_poses.npz',allow_pickle=False)['world_T_rgb']
    k=np.load(BASE/'moge_sample_calibrated/frame_000000/undistorted_input_K.npy',allow_pickle=False)
    report={'state':'running','started_utc':now(),'training':False,'network':'disabled','source_sha256':hashes,'objects':{},
            'resolution_per_axis':RESOLUTION,'truncation_m':TRUNCATION,
            'reference':'Per-object initial visible-surface frame, not canonical object origin',
            'limitations':['Stereo and object motion are predictions, without GT pose/depth validation.',
                           'Only observed surfaces are extracted; unknown backsides are not filled.',
                           'Symmetry/ICP drift can blur fusion; output is a diagnostic partial mesh, not submission geometry.',
                           'No texture, canonical mesh registration, face-budget simplification or collision validation.']}
    if (OUT/'report.json').exists() and json.loads((OUT/'report.json').read_text()).get('source_sha256')!=hashes:raise RuntimeError('Fusion sources changed; refusing mixed volume checkpoints')
    save(OUT/'report.json',report)
    for oi,name in enumerate(tracks['object_names'].tolist()):
        dest=OUT/name;dest.mkdir(exist_ok=True);source=tracks['world_T_reference'][:,oi];accepted=tracks['accepted'][:,oi]
        samples=[]
        for frame in range(137):
            if not accepted[frame]:continue
            world=cloud(frame,oi,k,camera[frame]);ref=(world-source[frame,:3,3])@source[frame,:3,:3];samples.append(ref)
        points=np.concatenate(samples);np.save(dest/'fused_visible_points.npy',points,allow_pickle=False)
        lower,upper=np.percentile(points,[.5,99.5],axis=0);lower-=.02;upper+=.02
        spacing=(upper-lower)/(RESOLUTION-1)
        axes=[np.linspace(a,b,RESOLUTION,dtype=np.float32) for a,b in zip(lower,upper)]
        grid=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
        shape=(RESOLUTION,)*3;total=np.zeros(len(grid),np.float32);weight=np.zeros(len(grid),np.uint16);first=0
        checkpoint=dest/'volume_checkpoint.npz';checkpoint_info=dest/'checkpoint.json'
        if checkpoint.exists():
            info=json.loads(checkpoint_info.read_text())
            if info['source_sha256']!=hashes:raise RuntimeError('Volume checkpoint sources changed')
            with np.load(checkpoint,allow_pickle=False) as old:
                assert np.array_equal(old['lower'],lower) and np.array_equal(old['upper'],upper)
                total=old['tsdf_sum'].reshape(-1);weight=old['weight'].reshape(-1);first=int(old['next_frame'])
        item={'state':'running','first_resumed_frame':first,'bounds_m':[lower.tolist(),upper.tolist()],
              'voxel_spacing_m':spacing.tolist(),'frames_integrated':int(accepted[:first].sum()),'frames':[]}
        report['objects'][name]=item;save(OUT/'report.json',report)
        for frame in range(first,137):
            if accepted[frame]:
                depth=np.load(BASE/f'foundation_stereo_sequence/frame_{frame:06d}/prediction.npz',allow_pickle=False)['depth_rgb']
                regions=np.load(BASE/f'object_masks_temporal/masks/{frame:06d}.npz',allow_pickle=False)['masks']
                hands=np.load(BASE/f'hand_masks_sequence/frame_{frame:06d}/masks.npz',allow_pickle=False)['masks'].any(axis=0)
                region=cv2.erode(regions[oi].astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)&~hands&np.isfinite(depth)
                z=depth[region];med=np.median(z);mad=np.median(np.abs(z-med));region&=np.abs(depth-med)<max(.02,4*mad)
                cam_T_ref=np.linalg.inv(camera[frame])@source[frame]
                cam=grid@cam_T_ref[:3,:3].T+cam_T_ref[:3,3];proj=cam@k.T;uv=np.rint(proj[:,:2]/proj[:,2:3]).astype(np.int32)
                inside=(cam[:,2]>.1)&(uv[:,0]>=0)&(uv[:,0]<1024)&(uv[:,1]>=0)&(uv[:,1]<767)
                index=np.flatnonzero(inside);pixels=uv[index];index=index[region[pixels[:,1],pixels[:,0]]];pixels=uv[index]
                sdf=depth[pixels[:,1],pixels[:,0]]-cam[index,2];good=sdf>=-TRUNCATION;index=index[good];sdf=sdf[good]
                total[index]+=np.clip(sdf/TRUNCATION,-1,1).astype(np.float32);weight[index]+=1
                item['frames_integrated']+=1;item['frames'].append({'frame_index':frame,'updated_voxels':len(index)})
            if frame%10==0 or frame==136:
                np.savez_compressed(dest/'volume_checkpoint.tmp.npz',tsdf_sum=total.reshape(shape),weight=weight.reshape(shape),lower=lower,upper=upper,next_frame=np.asarray(frame+1))
                (dest/'volume_checkpoint.tmp.npz').replace(checkpoint)
                save(checkpoint_info,{'source_sha256':hashes,'next_frame':frame+1,'updated_utc':now()})
                save(OUT/'report.json',report);print(name,'frame',frame,'observed voxels',int((weight>=5).sum()),flush=True)
        tsdf=np.ones(len(grid),np.float32);np.divide(total,weight,out=tsdf,where=weight>0);tsdf=tsdf.reshape(shape)
        known=ndimage.binary_erosion((weight.reshape(shape)>=5),structure=np.ones((3,3,3),bool))
        item['known_surface_extraction_voxels']=int(known.sum())
        try:
            verts,faces,_,_=marching_cubes(tsdf,level=0,spacing=tuple(spacing),mask=known,allow_degenerate=False)
            mesh=trimesh.Trimesh(vertices=verts+lower,faces=faces,process=False);mesh.export(dest/'observed_surface.ply')
            item.update(mesh_generated=True,vertices=len(verts),faces=len(faces),watertight=bool(mesh.is_watertight),mesh_sha256=digest(dest/'observed_surface.ply'))
        except (ValueError,RuntimeError) as exc:item.update(mesh_generated=False,extraction_error_type=type(exc).__name__)
        item.update(state='completed',completed_utc=now());save(OUT/'report.json',report)
        del grid,total,weight,tsdf,known
    report.update(state='completed',completed_utc=now(),elapsed_seconds=time.perf_counter()-started);save(OUT/'report.json',report)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if (OUT/'report.json').exists():
            r=json.loads((OUT/'report.json').read_text());r.update(state='failed',error_type=type(exc).__name__,updated_utc=now());save(OUT/'report.json',r)
        raise
