"""Offline two-object mesh reconstruction from our own masked public video frame."""
import hashlib,json,sys,time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path('/work')
sys.path.insert(0,str(ROOT/'vendor/TripoSR'))
sys.path.insert(0,str(ROOT/'runtime_deps'))
import numpy as np
import torch
import transformers
from PIL import Image
from omegaconf import OmegaConf
from tsr.system import TSR

OUT=ROOT/'outputs/track3/episode_000011/object_meshes_triposr'


def save(report):
    temp=OUT/'report.json.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(OUT/'report.json')


def main():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    cfg=OmegaConf.load(ROOT/'weights/triposr/config.yaml')
    cfg.image_tokenizer.pretrained_model_name_or_path='/work/weights/triposr/dino-config'
    OmegaConf.resolve(cfg)
    report={'state':'running','started_utc':datetime.now(timezone.utc).isoformat(),'training':False,'network':'disabled',
            'transformers':transformers.__version__,
            'method':'TripoSR, public MIT model; no public object scans used','objects':{},'mesh_grid_resolution':128,
            'metric_scale_known':False,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'source_changes':['optional rembg; reuse SAM2 foreground','local DINO architecture config','skimage CPU marching cubes fallback'],
            'limitations':['Meshes are generated in canonical unitless coordinates, not yet registered to tracking/body coordinates.',
                           'Single-frame shape completion has unobserved-surface uncertainty; no GT mesh score measured.']}
    save(report);start=time.perf_counter()
    model=TSR(cfg)
    ckpt=torch.load(ROOT/'weights/triposr/model.ckpt',map_location='cpu',weights_only=True)
    model.load_state_dict(ckpt,strict=True)
    del ckpt
    model.eval().to('cuda');model.renderer.set_chunk_size(8192)
    torch.cuda.synchronize();report['model_load_seconds']=time.perf_counter()-start
    for name in ['blue_cup','beige_cup']:
        image_path=ROOT/f'outputs/track3/episode_000011/mesh_inputs/{name}.png'
        image=Image.open(image_path).convert('RGB')
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():
            scene_codes=model([image],device='cuda')
            torch.cuda.synchronize();encoding=time.perf_counter()-start
            torch.save(scene_codes.detach().cpu(),OUT/(name+'_scene_code.pt'))
            mesh=model.extract_mesh(scene_codes,has_vertex_color=True,resolution=128)[0]
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        if not np.isfinite(mesh.vertices).all() or not len(mesh.faces) or not np.isfinite(mesh.area) or mesh.area<=0:
            raise RuntimeError('Invalid generated mesh: '+name)
        mesh.fix_normals()
        mesh.export(OUT/(name+'.glb'));mesh.export(OUT/(name+'.ply'))
        info={'vertices':len(mesh.vertices),'faces':len(mesh.faces),'area_canonical_units2':float(mesh.area),
              'bounds_canonical':mesh.bounds.tolist(),'watertight':bool(mesh.is_watertight),
              'encoding_seconds':encoding,'encoding_and_extraction_seconds':elapsed,
              'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
              'input_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest(),
              'mesh_sha256':hashlib.sha256((OUT/(name+'.glb')).read_bytes()).hexdigest()}
        report['objects'][name]=info;save(report);print(name,json.dumps(info),flush=True)
    report.update(state='completed',completed_utc=datetime.now(timezone.utc).isoformat());save(report)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        if (OUT/'report.json').exists():
            failed=json.loads((OUT/'report.json').read_text())
            failed.update(state='failed',error_type=type(exc).__name__,error_message=str(exc).split('\n')[0])
            save(failed)
        raise
