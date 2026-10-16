"""Inspect actual SAM3D geometry and its saved OpenCV metric transform, offline."""
import os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import cv2
import trimesh
import pyrender
from scipy.spatial.transform import Rotation

ROOT=Path('/work');BASE=ROOT/'outputs/track3/episode_000011';OUT=BASE/'object_meshes_sam3d_local'


def camera_pose(eye,center,up):
    back=eye-center;back/=np.linalg.norm(back);right=np.cross(up,back);right/=np.linalg.norm(right)
    pose=np.eye(4);pose[:3,:3]=np.column_stack([right,np.cross(back,right),back]);pose[:3,3]=eye;return pose


def main():
    renderer=pyrender.OffscreenRenderer(512,512);canvas=np.full((1120,1536,3),245,np.uint8)
    report={'state':'running','frames':[],'transform_convention':'Saved transform already maps canonical mesh to OpenCV; no second SAM3D axis flip.'}
    original=cv2.imread(str(BASE/'calibrated_sequence/frames/000000.jpg'));overlay=original.copy()
    k=np.load(BASE/'moge_sample_calibrated/frame_000000/undistorted_input_K.npy')
    masks=np.load(BASE/'object_masks_temporal/masks/000000.npz')['masks']
    observed=np.load(BASE/'foundation_stereo_sequence/frame_000000/prediction.npz')['depth_rgb']
    for row,name in enumerate(['blue_cup','beige_cup']):
        mesh=trimesh.load(OUT/name/'mesh.glb',force='mesh',process=False)
        center=mesh.bounds.mean(0);distance=float(mesh.extents.max())*2.5
        scene=pyrender.Scene(bg_color=[.96,.96,.96,1],ambient_light=[.5,.5,.5]);scene.add(pyrender.Mesh.from_trimesh(mesh,smooth=True))
        node=scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(35),znear=.01,zfar=100),pose=np.eye(4))
        light=scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2.5),pose=np.eye(4))
        for col,azimuth in enumerate([0,120,240]):
            angle=np.deg2rad(azimuth);eye=center+distance*np.array([np.cos(angle),.6,np.sin(angle)])
            pose=camera_pose(eye,center,np.array([0.,1.,0.]));scene.set_pose(node,pose);scene.set_pose(light,pose)
            rgba,_=renderer.render(scene);image=cv2.cvtColor(rgba[:,:,:3],cv2.COLOR_RGB2BGR);y=row*560+36;x=col*512
            canvas[y:y+512,x:x+512]=image;cv2.putText(canvas,f'{name}, canonical view {azimuth}',(x+10,y-10),cv2.FONT_HERSHEY_SIMPLEX,.48,(30,30,30),1,cv2.LINE_AA)
        transform=json.loads((OUT/name/'transform.json').read_text());mat=np.eye(4)
        mat[:3,:3]=Rotation.from_quat(transform['rotation'],scalar_first=True).as_matrix()@np.diag(transform['scale']);mat[:3,3]=transform['translation']
        metric=mesh.copy();metric.apply_transform(mat)
        renderer.viewport_width=1024;renderer.viewport_height=767
        scene=pyrender.Scene(bg_color=[0,0,0,0],ambient_light=[.65,.65,.65]);scene.add(pyrender.Mesh.from_trimesh(metric,smooth=True))
        view=np.diag([1.,-1.,-1.,1.]);scene.add(pyrender.IntrinsicsCamera(fx=k[0,0],fy=k[1,1],cx=k[0,2],cy=k[1,2],znear=.05,zfar=3),pose=view)
        scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2),pose=view)
        rgba,depth=renderer.render(scene);pred=depth>0;target=masks[row].astype(bool)
        region=pred&target&np.isfinite(observed);residual=abs(depth[region]-observed[region])
        overlay[pred]=(overlay[pred]*.35+cv2.cvtColor(rgba[:,:,:3],cv2.COLOR_RGB2BGR)[pred]*.65).astype(np.uint8)
        report['frames'].append({'object':name,'frame_index':0,'mask_iou':float((pred&target).sum()/max((pred|target).sum(),1)),
                                 'depth_residual_m_p50_p95':np.percentile(residual,[50,95]).tolist() if len(residual) else None,
                                 'depth_compared_pixels':int(region.sum()),'predicted_transform_only':True,
                                 'metric_camera_extents_m':metric.extents.tolist(),'watertight':bool(mesh.is_watertight)})
        renderer.viewport_width=512;renderer.viewport_height=512
    renderer.delete();cv2.imwrite(str(OUT/'mesh_preview.jpg'),canvas);cv2.imwrite(str(OUT/'frame0_metric_overlay.jpg'),overlay)
    report.update(state='completed',completed_utc=datetime.now(timezone.utc).isoformat(),
                  limitations=['Only the generation frame is measured here; not held-out accuracy.',
                               'Metric pose/scale predicted by SAM3D still needs registration.',
                               'Watertightness alone does not establish correct cup cavity or collision behavior.'])
    (OUT/'render_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
