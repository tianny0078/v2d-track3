"""Fail-closed mask alignment for the pinned Track3 learner configuration."""
import numpy as np
from hydra.utils import instantiate
from flash_chord.data import load_reference
from flash_chord.data.reference import reference_frame_slice

def align_known(cfg, source_known, source_sides=('left','right')):
    source = load_reference(cfg.task.parquet)
    target_sides = tuple(instantiate(cfg.env.objective.config).sides)
    if len(set(target_sides)) != len(target_sides):
        raise ValueError('objective side names must be unique')
    if len(set(source_sides)) != len(source_sides) or set(source_sides) != set(target_sides):
        raise ValueError('mask side names must match objective sides exactly')
    # This checkpoint contains per-contact-slot validity, not per-object labels.
    # Only the fully unknown case has an unambiguous object-level reduction.
    values = np.asarray(source_known)
    if values.shape[:2] != (source.num_frames, len(source_sides)) or values.ndim != 3:
        raise ValueError('source validity must have frame, named-side, contact-slot axes')
    if not np.isin(values, [0,1]).all() or values.any():
        raise ValueError('known contact slots require an explicit slot-to-object annotation mapping')
    if float(cfg.task.motion_speed) != 1.0 or float(cfg.env.sim.fps) != float(source.fps) or cfg.task.source_frame_playback:
        raise ValueError('mask retiming unsupported: require original FPS, speed=1 and no source playback')
    if float(cfg.scene.object_scale_min) != 1.0 or float(cfg.scene.object_scale_max) != 1.0:
        raise ValueError('COM offsets/radii require unscaled objects')
    selection = reference_frame_slice(source.num_frames, int(cfg.task.motion_start_frame), int(cfg.task.motion_end_frame))
    frames = np.arange(source.num_frames)[selection]
    known = np.zeros((len(frames),len(target_sides),len(source.object_body_names())),np.int32)
    return source, frames, known, [source_sides.index(s) for s in target_sides]
