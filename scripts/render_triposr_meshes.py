"""Render generated meshes from three directions for geometry inspection."""
import json,os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
from pathlib import Path
import cv2
import numpy as np
import pyrender
import trimesh

BASE=Path('/work/outputs/track3/episode_000011/object_meshes_triposr')


def camera_pose(eye,center):
    back=eye-center;back/=np.linalg.norm(back)
    right=np.cross(np.array([0.,0.,1.]),back);right/=np.linalg.norm(right)
    up=np.cross(back,right)
    pose=np.eye(4);pose[:3,:3]=np.stack([right,up,back],axis=1);pose[:3,3]=eye
    return pose


def main():
    cv2.setNumThreads(4)
    renderer=pyrender.OffscreenRenderer(512,512)
    canvas=np.full((1120,1536,3),245,np.uint8)
    meta={}
    for row,name in enumerate(['blue_cup','beige_cup']):
        mesh=trimesh.load(BASE/(name+'.glb'),force='mesh')
        center=mesh.bounds.mean(axis=0);distance=float(mesh.extents.max())*2.5
        scene=pyrender.Scene(bg_color=[.96,.96,.96,1],ambient_light=[.45,.45,.45])
        scene.add(pyrender.Mesh.from_trimesh(mesh,smooth=True))
        camera=pyrender.PerspectiveCamera(yfov=np.deg2rad(35),znear=.01,zfar=100)
        node=scene.add(camera,pose=np.eye(4))
        light_node=scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2.5),pose=np.eye(4))
        for col,azimuth in enumerate([0,120,240]):
            angle=np.deg2rad(azimuth)
            eye=center+distance*np.array([np.cos(angle),np.sin(angle),.55])
            pose=camera_pose(eye,center);scene.set_pose(node,pose);scene.set_pose(light_node,pose)
            rgba,depth=renderer.render(scene)
            image=cv2.cvtColor(rgba[:,:,:3],cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(BASE/f'{name}_view_{azimuth:03d}.png'),image)
            y=row*560+36;x=col*512
            canvas[y:y+512,x:x+512]=image
            cv2.putText(canvas,f'{name}, view {azimuth}',(x+12,y-10),cv2.FONT_HERSHEY_SIMPLEX,.55,(30,30,30),1,cv2.LINE_AA)
        meta[name]={'vertices':len(mesh.vertices),'faces':len(mesh.faces),'watertight':bool(mesh.is_watertight),'canonical_extents':mesh.extents.tolist()}
    renderer.delete()
    cv2.imwrite(str(BASE/'mesh_preview.jpg'),canvas)
    (BASE/'render_report.json').write_text(json.dumps({'state':'completed','meshes':meta,'scale':'canonical, not yet metric'},indent=2)+'\n')
    print('Rendered two generated meshes from three views each',flush=True)


if __name__=='__main__':main()
