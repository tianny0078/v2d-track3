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


def align_geometry_proxy(cfg, checkpoint, source_object_names):
    """Align explicitly object-labelled proxy validity; never infer it from GT slots."""
    source = load_reference(cfg.task.parquet)
    sides = tuple(str(s) for s in checkpoint['side_names'])
    # Reuse the fail-closed retiming/scale/side checks on a dummy unknown mask.
    _, frames, _, side_indices = align_known(
        cfg, np.zeros((source.num_frames, len(sides), 1), np.int32), sides)
    names = tuple(source_object_names)
    target_names = tuple(source.object_body_names())
    if len(set(names)) != len(names) or set(names) != set(target_names):
        raise ValueError('proxy object names must match reference bodies exactly')
    values = np.asarray(checkpoint['geometry_proxy_known'])
    if values.shape != (source.num_frames, len(sides), len(names)) or not np.isin(values, [0, 1]).all():
        raise ValueError('geometry proxy requires binary frame/side/object validity')
    if np.asarray(checkpoint['contact_ground_truth_known']).any():
        raise ValueError('this proxy-only entry requires genuine GT to remain unknown')
    observed = np.asarray(checkpoint['observed_hand'])
    if observed.shape != values.shape[:2] or np.any(values.astype(bool) & ~observed.astype(bool)[..., None]):
        raise ValueError('unobserved human hand cannot supply known geometry proxy')
    object_indices = [names.index(n) for n in target_names]
    known = values[frames][:, side_indices][:, :, object_indices].astype(np.int32)
    com = np.asarray(checkpoint['canonical_COM'])[object_indices]
    radius = np.asarray(checkpoint['object_bounding_radius_COM'])[object_indices]
    if com.shape != (len(names), 3) or radius.shape != (len(names),) or not np.isfinite(com).all() or not np.isfinite(radius).all() or (radius <= 0).any():
        raise ValueError('invalid COM offsets or normalization radii')
    return source, frames, known, side_indices, com, radius
