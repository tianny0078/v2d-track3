from pathlib import Path
import datetime, hashlib, json
import numpy as np
import warp as wp
from omegaconf import OmegaConf
from hydra.utils import instantiate
from track3_reference_alignment import align_known
from flash_chord.scene.setup import setup_scene
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/track3/episode_000011/reference_alignment'

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 cfg=OmegaConf.load(ROOT/'outputs/track3/episode_000011/masked_physics_integration/composed_training.yaml')
 z=np.load(ROOT/'outputs/track3/episode_000011/flashchord_reference_preflight/reference_checkpoint.npz')
 source,frames,known,side_indices=align_known(cfg,z['contact_ground_truth_known'])
 cfg.env.objective.reference_known=known.tolist()
 cfg.env.objective.config.sides=list(instantiate(cfg.env.objective.config).sides)
 with wp.ScopedDevice('cpu'):
  setup=setup_scene(parquet=cfg.task.parquet,control_fps=cfg.env.sim.fps,motion_speed=cfg.task.motion_speed,embodiment=instantiate(cfg.embodiment),collision=instantiate(cfg.collision),world_count=2,include_support=cfg.scene.support,decompose_objects=cfg.scene.decompose_objects,contact_friction=cfg.scene.contact_friction,source_frame_playback=cfg.task.source_frame_playback,motion_start_frame=cfg.task.motion_start_frame,motion_end_frame=cfg.task.motion_end_frame)
 ref=setup.reference
 errors={}
 for name in ('object_body_pos_w','object_body_quat_w'):
  a=getattr(ref,name)(); b=getattr(source,name)()[frames]
  np.testing.assert_array_equal(a,b);errors[name]=float(np.max(np.abs(a-b)))
 assert ref.num_frames==len(known) and ref.fps==source.fps
 assert tuple(cfg.env.objective.config.sides)==tuple(instantiate(cfg.env.contact).sides)
 rejected=[]
 for name,value in [('speed',.5),('fps',40),('scale',1.1),('known_slot',1)]:
  c=OmegaConf.create(OmegaConf.to_container(cfg));mask=z['contact_ground_truth_known'].copy()
  if name=='speed':c.task.motion_speed=value
  elif name=='fps':c.env.sim.fps=value
  elif name=='scale':c.scene.object_scale_max=value
  else:mask[0,0,0]=value
  try:align_known(c,mask)
  except ValueError:rejected.append(name)
  else:raise AssertionError(name+' silently accepted')
 # Crop semantics are half-open and preserve original indices.
 crop=OmegaConf.create(OmegaConf.to_container(cfg));crop.task.motion_start_frame=12;crop.task.motion_end_frame=100
 _,cropped,cm,_=align_known(crop,z['contact_ground_truth_known']);np.testing.assert_array_equal(cropped,np.arange(12,100));assert cm.shape[0]==88
 OmegaConf.save(cfg,OUT/'learner_config.yaml')
 np.savez_compressed(OUT/'checkpoint.npz',source_frames=frames,known=known,source_side_indices=side_indices,cropped_source_frames=cropped)
 report=dict(state='completed',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),official_setup_scene_called=True,world_count=2,frames=ref.num_frames,fps=ref.fps,observation_contact_sides=list(cfg.env.observation.contact_sides),tracker_sides=list(instantiate(cfg.env.contact).sides),source_sides=['left','right'],objective_sides=list(cfg.env.objective.config.sides),source_side_indices=side_indices,exact_pose_errors=errors,rejected_unsafe_cases=rejected,crop_pass=True,production_integrated=False,learner_exercised=False,training_ready=False,limitations=['CPU scene build only; no new physics steps or learner updates','all source contacts unknown; known-slot mapping deliberately rejected'],source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'scripts/track3_reference_alignment.py']})
 (OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
