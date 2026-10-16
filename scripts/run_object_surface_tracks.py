"""Conservative masked RGB-D rigid tracking; object frame is an initial visible surface."""
import json,time
from pathlib import Path
import cv2
import numpy as np
import scipy
from scipy.spatial import cKDTree
from download_track3_sample import digest,now,save

ROOT=Path('/work');BASE=ROOT/'outputs/track3/episode_000011';OUT=BASE/'object_surface_tracks'


def rigid_fit(a,b):
    ca,cb=a.mean(axis=0),b.mean(axis=0);u,_,vt=np.linalg.svd((a-ca).T@(b-cb));r=vt.T@u.T
    if np.linalg.det(r)<0:vt[-1]*=-1;r=vt.T@u.T
    t=cb-r@ca;out=np.eye(4);out[:3,:3]=r;out[:3,3]=t;return out


def cloud(frame,object_index,k,pose):
    depth=np.load(BASE/f'foundation_stereo_sequence/frame_{frame:06d}/prediction.npz',allow_pickle=False)['depth_rgb']
    regions=np.load(BASE/f'object_masks_temporal/masks/{frame:06d}.npz',allow_pickle=False)['masks']
    hands=np.load(BASE/f'hand_masks_sequence/frame_{frame:06d}/masks.npz',allow_pickle=False)['masks'].any(axis=0)
    region=cv2.erode(regions[object_index].astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)&~hands&np.isfinite(depth)
    yy,xx=np.where(region);zz=depth[yy,xx]
    if len(zz)<100:return np.empty((0,3))
    med=np.median(zz);mad=np.median(np.abs(zz-med));good=np.abs(zz-med)<max(.02,4*mad)
    yy,xx,zz=yy[good],xx[good],zz[good]
    xyz=np.stack([(xx-k[0,2])*zz/k[0,0],(yy-k[1,2])*zz/k[1,1],zz],axis=1)
    xyz=xyz@pose[:3,:3].T+pose[:3,3]
    _,indices=np.unique(np.floor(xyz/.003).astype(np.int64),axis=0,return_index=True);xyz=xyz[np.sort(indices)]
    if len(xyz)>1800:xyz=xyz[np.linspace(0,len(xyz)-1,1800).astype(int)]
    return xyz


def register(template,observed,initial):
    if len(observed)<100:return initial,{'accepted':False,'reason':'Too little visible depth','observed_points':len(observed)}
    tree=cKDTree(observed);pose=initial.copy();previous=np.inf
    for iteration in range(40):
        moved=template@pose[:3,:3].T+pose[:3,3];distance,index=tree.query(moved,k=1,workers=2)
        limit=max(.012,.035*(.96**iteration));good=distance<limit
        if good.sum()<100:break
        cutoff=np.percentile(distance[good],75);good&=distance<=cutoff
        if good.sum()<75:break
        delta=rigid_fit(moved[good],observed[index[good]]);pose=delta@pose
        error=float(np.median(distance[good]))
        if abs(previous-error)<1e-6:break
        previous=error
    moved=template@pose[:3,:3].T+pose[:3,3];distance,_=tree.query(moved,k=1,workers=2);good=distance<.02
    delta=pose@np.linalg.inv(initial);angle=float(np.arccos(np.clip((np.trace(delta[:3,:3])-1)/2,-1,1)))
    # Bound the movement of the reference centre, independently of camera-origin rotation.
    translation=float(np.linalg.norm(pose[:3,3]-initial[:3,3]));count=int(good.sum())
    residual=np.percentile(distance[good],[50,95]).tolist() if count else None
    accepted=count>=100 and count/len(template)>=.2 and residual[0]<.010 and residual[1]<.020 and angle<np.deg2rad(25) and translation<.12
    return pose,{'accepted':bool(accepted),'observed_points':len(observed),'matched_template_points':count,
                 'template_match_fraction':count/len(template),'correspondence_residual_m_p50_p95':residual,
                 'relative_rotation_degrees':float(np.rad2deg(angle)),'reference_centre_step_m':translation,
                 'reason':'Accepted geometric fit' if accepted else 'ICP support/residual/motion gate'}


def main():
    cv2.setNumThreads(4);OUT.mkdir(exist_ok=True)
    source_paths=[BASE/'foundation_stereo_sequence/report.json',BASE/'hand_masks_sequence/report.json',BASE/'stereo_sequence/camera_poses.npz']
    hashes=[digest(p) for p in source_paths]
    for p in source_paths[:2]:assert json.loads(p.read_text())['state']=='completed'
    camera=np.load(source_paths[-1],allow_pickle=False)['world_T_rgb']
    k=np.load(BASE/'moge_sample_calibrated/frame_000000/undistorted_input_K.npy',allow_pickle=False)
    names=np.load(BASE/'object_masks_temporal/masks/000000.npz',allow_pickle=False)['object_names'].tolist()
    old=json.loads((OUT/'report.json').read_text()) if (OUT/'report.json').exists() else {}
    if old and old.get('source_sha256')!=hashes:raise RuntimeError('Tracking inputs changed; refusing mixed pose checkpoints')
    report={'state':'running','started_utc':now(),'training':False,'network':'disabled','source_sha256':hashes,'frames':old.get('frames',[]),'scipy':scipy.__version__,
            'world_frame':'rectified ego_cam_c at frame 0','object_reference':'Frame-0 robust visible-surface centroid and initial world axes, not canonical mesh/physical origin',
            'method':'Hand-excluded eroded object masks; median/MAD depth filtering; 3mm voxel samples; trimmed point-to-point ICP against fixed initial template',
            'limitations':['ICP residual and acceptance are internal consistency, not GT 6D pose accuracy.',
                           'Cup symmetry and partial visibility make rotation ambiguous; this is a geometry initialization.',
                           'Failed frames hold the last accepted estimate and mark accepted=false; no interpolated poses.',
                           'Generated complete meshes are not registered yet; these tracks cannot be submitted as canonical object poses.']}
    templates=[];current=[]
    for i,name in enumerate(names):
        points=cloud(0,i,k,camera[0]);assert len(points)>=100
        centre=points.mean(axis=0);template=points-centre;initial=np.eye(4);initial[:3,3]=centre
        templates.append(template);current.append(initial);np.save(OUT/(name+'_initial_visible_template.npy'),template,allow_pickle=False)
    poses=np.full((137,2,4,4),np.nan);accepted=np.zeros((137,2),bool)
    for f in report['frames']:
        frame=f['frame_index']
        for i,name in enumerate(names):
            row=f['objects'][name];poses[frame,i]=np.asarray(row['world_T_reference']);accepted[frame,i]=row['accepted']
            if row['accepted']:current[i]=poses[frame,i].copy()
    start=time.perf_counter();save(OUT/'report.json',report)
    for frame in range(len(report['frames']),137):
        item={'frame_index':frame,'objects':{}}
        for i,name in enumerate(names):
            observed=cloud(frame,i,k,camera[frame])
            if frame==0:proposal=current[i];stats={'accepted':True,'reason':'Initial reference frame','observed_points':len(observed)}
            else:proposal,stats=register(templates[i],observed,current[i])
            if stats['accepted']:current[i]=proposal.copy()
            poses[frame,i]=current[i];accepted[frame,i]=stats['accepted'];item['objects'][name]=dict(stats,world_T_reference=current[i].tolist(),held_previous_estimate=not stats['accepted'])
        report['frames'].append(item);save(OUT/'report.json',report)
        np.savez_compressed(OUT/'poses.tmp.npz',frame_index=np.arange(137),object_names=np.asarray(names),world_T_reference=poses,accepted=accepted)
        (OUT/'poses.tmp.npz').replace(OUT/'poses.npz')
        if frame%10==0 or frame==136:print('frame',frame,{name:item['objects'][name]['accepted'] for name in names},flush=True)
    report.update(state='completed',completed_utc=now(),elapsed_seconds=time.perf_counter()-start,accepted_counts={name:int(accepted[:,i].sum()) for i,name in enumerate(names)})
    save(OUT/'report.json',report)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if (OUT/'report.json').exists():
            r=json.loads((OUT/'report.json').read_text());r.update(state='failed',updated_utc=now(),error_type=type(exc).__name__);save(OUT/'report.json',r)
        raise
