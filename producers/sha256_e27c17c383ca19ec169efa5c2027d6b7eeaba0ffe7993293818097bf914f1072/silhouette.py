"""Multi-view silhouette agreement: project a mesh with the per-frame FoundationPose v4 camera_T_link (the poses the
learning trees are built from) through rgb_K and compare with the SAM2 object masks (object_masks_v3).

Used (1) to validate new vs current geometry on the scored episodes (IoU / area ratio / contour distance per frame) and
(2) to estimate parameters the canonical mesh cannot pin down (e.g. the sweeper's bristle fan): masks only, no scans.
"""
import json
from pathlib import Path
import numpy as np, cv2, trimesh
from common import T3

H, W = 767, 1024


def episode_data(ep, name):
    E = T3/f'episode_{ep:06d}'; fp = np.load(E/'foundationpose_v4/poses_table.npz')
    names = list(map(str, fp['object_names'])); K = np.load(E/'calibrated_sequence/rgb_K.npy')
    src = json.loads((E/'foundationpose_v4/SOURCE.json').read_text())[name]
    return dict(K=K, camT=fp['camera_T_link'][:, names.index(name)], scale=float(src.get('scale', 1.0)), E=E, name=name)


def mask(d, f):
    z = np.load(d['E']/f'object_masks_v3/masks/{f:06d}.npz'); nm = list(map(str, z['object_names']))
    return z['masks'][nm.index(d['name'])] if d['name'] in nm else None


def render_mask(mesh, T, K, scale=1.0):
    v = mesh.vertices*scale @ T[:3, :3].T + T[:3, 3]
    if (v[:, 2] <= 0.01).any(): return None
    uv = v[:, :2]/v[:, 2:3] @ K[:2, :2].T + K[:2, 2]
    tri = np.round(uv[mesh.faces]*8).astype(np.int32)  # 3 bits subpixel
    img = np.zeros((H, W), np.uint8)
    # one convex fill per triangle (a multi-polygon fillPoly is even-odd: overlapping triangles would cancel)
    for t in tri: cv2.fillConvexPoly(img, t, 1, lineType=cv2.LINE_8, shift=3)
    return img.astype(bool)


def frame_metrics(rm, mm):
    inter = (rm & mm).sum(); uni = (rm | mm).sum()
    return dict(iou=float(inter/max(uni, 1)), area_ratio=float(rm.sum()/max(mm.sum(), 1)))


def compare(ep, name, meshes, frames=None, step=5, min_area=1500):
    """meshes: {label: trimesh (in the base canonical frame, unscaled)} -> per-label list of per-frame metrics."""
    d = episode_data(ep, name); n = len(d['camT'])
    frames = range(0, n, step) if frames is None else frames
    out = {k: [] for k in meshes}; used = []
    for f in frames:
        T = d['camT'][f]
        if not np.isfinite(T).all(): continue
        mm = mask(d, f)
        if mm is None or mm.sum() < min_area: continue
        rs = {k: render_mask(m, T, d['K'], d['scale']) for k, m in meshes.items()}
        if any(r is None for r in rs.values()): continue
        used.append(f)
        for k, r in rs.items(): out[k].append(frame_metrics(r, mm))
    return used, out
