"""Collision-body quality, current vs procedural (Monte Carlo, 200k points in the visual's bounding box):
  coverage        = P(point inside some collision part | point inside the visual solid)        (want ~1)
  concavity_fill  = P(point inside some collision part | inside convex hull(visual), outside visual) (want ~0: cavities kept)
  excess_cm3      = volume inside collision but outside the visual
The current visual meshes are watertight (trimesh) so contains() is defined for both.
Writes outputs/track3/procedural_geometry_v1/eval/collision.json.
"""
import json, sys
from pathlib import Path
import numpy as np, trimesh
sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, BASE, load_base, base_collision


def inside_any(parts, P):
    hit = np.zeros(len(P), bool)
    for h in parts:
        h = h.convex_hull; n = h.face_normals; d = np.einsum('ij,ij->i', n, h.triangles[:, 0])
        lo, hi = h.bounds; k = ~hit & (P >= lo).all(1) & (P <= hi).all(1)
        if k.any(): hit[np.flatnonzero(k)[(P[k] @ n.T - d <= 1e-9).all(1)]] = True
    return hit


def stats(vis, parts, n=200000, seed=0):
    lo, hi = vis.bounds; P = np.random.default_rng(seed).uniform(lo, hi, (n, 3)); box = np.prod(hi - lo)
    inv = vis.contains(P); inh = inside_any([vis.convex_hull], P); inc = inside_any(parts, P)
    cav = inh & ~inv
    return dict(coverage=float((inc & inv).sum()/max(inv.sum(), 1)), concavity_fill=float((inc & cav).sum()/max(cav.sum(), 1)),
                excess_cm3=float((inc & ~inv).sum()/n*box*1e6), cavity_cm3=float(cav.sum()/n*box*1e6), parts=len(parts))


def main():
    rep = {}
    for name in BASE:
        old = stats(load_base(name), base_collision(name))
        D = OUT/name; import xml.etree.ElementTree as ET
        cols = [trimesh.load(D/c.find('geometry/mesh').attrib['filename'], force='mesh') for c in ET.parse(D/'object.urdf').getroot().iter('collision')]
        new = stats(trimesh.load(D/'visual.obj', force='mesh'), cols)
        rep[name] = dict(current=old, procedural=new); print(name, json.dumps(rep[name]), flush=True)
    (OUT/'eval').mkdir(exist_ok=True); (OUT/'eval/collision.json').write_text(json.dumps(rep, indent=1))


if __name__ == '__main__': main()
