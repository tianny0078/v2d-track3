"""Offline SAM3D comparison using existing local weights and our metric stereo pointmap."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image

ROOT=Path(os.environ.get('V2D_TASK_ROOT',Path(__file__).resolve().parents[1]))
BASE=ROOT/'outputs/track3/episode_000011';OUT=BASE/'object_meshes_sam3d_local'
WEIGHTS=Path('/mnt/nas3/v2d/reconstruction/data/weights/sam3d')
IMAGE='sha256:7ada1f16b351db919fed82a8f28e1bdb3593d0be05c169a94f3a5b5c81b88727'


def save(report):
    tmp=OUT/'report.tmp.json';tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');tmp.replace(OUT/'report.json')


def inside():
    import torch
    import trimesh
    from v2d.sam3d.lib.image_to_mesh import image_to_mesh
    report=json.loads((OUT/'report.json').read_text());started=time.monotonic()
    report['runtime']={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)};save(report)
    torch.manual_seed(42)
    for name in ['blue_cup','beige_cup']:
        dest=OUT/name;dest.mkdir(exist_ok=True)
        if name in report['objects'] and report['objects'][name].get('state')=='completed':continue
        print('SAM3D starting',name,flush=True);begin=time.monotonic()
        image_to_mesh(image_path=str(BASE/'calibrated_sequence/frames/000000.jpg'),
                      mask_path=str(OUT/'inputs'/f'{name}.png'),mesh_path=str(dest/'mesh.glb'),
                      transform_path=str(dest/'transform.json'),intrinsics_path=str(dest/'intrinsics.json'),
                      weights_dir='/weights',seed=42,with_mesh_postprocess=False,
                      with_texture_baking=False,with_layout_postprocess=False,
                      pointmap_path=str(OUT/'inputs/stereo_pointmap.npy'),
                      pointmap_intrinsics_path=str(OUT/'inputs/intrinsics.json'))
        mesh=trimesh.load(dest/'mesh.glb',force='mesh',process=False)
        mesh.export(dest/'mesh.ply');mesh.export(dest/'mesh.obj')
        report['objects'][name]={'state':'completed','vertices':len(mesh.vertices),'faces':len(mesh.faces),
                                 'watertight':bool(mesh.is_watertight),'winding_consistent':bool(mesh.is_winding_consistent),
                                 'canonical_extents':mesh.extents.tolist(),'elapsed_seconds':time.monotonic()-begin,
                                 'transform':json.loads((dest/'transform.json').read_text()),
                                 'mesh_sha256':hashlib.sha256((dest/'mesh.glb').read_bytes()).hexdigest()}
        save(report);print(json.dumps(report['objects'][name],indent=2),flush=True)
    report.update(state='completed',completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started);save(report)


def main():
    OUT.mkdir(exist_ok=True);inputs=OUT/'inputs';inputs.mkdir(exist_ok=True)
    old=json.loads((OUT/'report.json').read_text()) if (OUT/'report.json').exists() else {}
    report={'state':'running','started_utc':datetime.now(timezone.utc).isoformat(),'objects':old.get('objects',{}),
            'image_id':IMAGE,'source_frame':0,'training':False,'network':'disabled',
            'method':'Official SAM3D wrapper with existing shared weights and calibrated stereo pointmap; no GT mesh/pose/scale.',
            'limitations':['A learned complete-shape prior, not an accepted collision asset.',
                           'Predicted transform/scale and cup cavity must be checked against observations.',
                           'No hole closing is applied solely to claim watertightness.']}
    checkpoint=WEIGHTS/'hf-download/checkpoints'
    required=['ss_generator.ckpt','slat_generator.ckpt','ss_decoder.ckpt','slat_decoder_mesh.ckpt','pipeline.yaml']
    report['weights']=[{'name':name,'bytes':(checkpoint/name).stat().st_size} for name in required]
    report['pipeline_sha256']=hashlib.sha256((checkpoint/'pipeline.yaml').read_bytes()).hexdigest();save(report)
    masks=np.load(BASE/'object_masks_temporal/masks/000000.npz',allow_pickle=False)
    for name,mask in zip(masks['object_names'],masks['masks']):Image.fromarray(mask.astype(np.uint8)*255).save(inputs/f'{name}.png')
    depth=np.load(BASE/'foundation_stereo_sequence/frame_000000/prediction.npz',allow_pickle=False)['depth_rgb']
    k=np.load(BASE/'moge_sample_calibrated/frame_000000/undistorted_input_K.npy');y,x=np.indices(depth.shape)
    points=np.stack([(x-k[0,2])*depth/k[0,0],(y-k[1,2])*depth/k[1,1],depth],axis=-1).astype(np.float32)
    points[~np.isfinite(depth)|(depth<=0)]=np.nan;np.save(inputs/'stereo_pointmap.npy',points)
    (inputs/'intrinsics.json').write_text(json.dumps(dict(fx=float(k[0,0]),fy=float(k[1,1]),cx=float(k[0,2]),cy=float(k[1,2]),width=1024,height=767),indent=2)+'\n')
    cmd=['docker','run','--rm','--pull=never','--name','v2d-t3-sam3d-local','--gpus','device=2','--network=none',
         '--cpus','12','--memory','80g','--read-only','--tmpfs','/tmp:rw,size=12g','--shm-size','8g',
         '--user',f'{os.getuid()}:{os.getgid()}',
         '--mount',f'type=bind,src={ROOT}/scripts,dst=/work/scripts,readonly',
         '--mount',f'type=bind,src={BASE},dst=/work/outputs/track3/episode_000011,readonly',
         '--mount',f'type=bind,src={OUT},dst=/work/outputs/track3/episode_000011/object_meshes_sam3d_local',
         '--mount',f'type=bind,src={WEIGHTS},dst=/weights,readonly',
         '--mount',f'type=bind,src={ROOT}/vendor/video_to_data/reconstruction/modules/v2d_sam3d/lib,dst=/workspace/v2d_sam3d/lib,readonly',
         '--mount',f'type=bind,src={ROOT}/vendor/video_to_data/reconstruction/modules/v2d_common,dst=/workspace/v2d_common,readonly',
         '-e','V2D_TASK_ROOT=/work','-e','V2D_SAM3D_INSIDE=1','-e','PYTHONDONTWRITEBYTECODE=1',
         '-e','OMP_NUM_THREADS=8','-e','OPENBLAS_NUM_THREADS=8','-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1',
         '-e','HF_HOME=/weights/hf_home','-e','TORCH_HOME=/weights/torch_home','-e','HOME=/tmp',
         '--entrypoint','python',IMAGE,'-u','/work/scripts/run_sam3d_local_sample.py']
    report['docker_argv']=cmd;save(report)
    try:subprocess.run(cmd,check=True)
    except Exception as exc:
        report=json.loads((OUT/'report.json').read_text());report.update(state='failed',error_type=type(exc).__name__,error=str(exc),completed_utc=datetime.now(timezone.utc).isoformat());save(report);raise


if __name__=='__main__':
    if os.environ.get('V2D_SAM3D_INSIDE')=='1':inside()
    else:main()
