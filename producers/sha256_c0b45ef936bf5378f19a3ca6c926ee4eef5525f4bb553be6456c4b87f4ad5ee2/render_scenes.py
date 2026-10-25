"""Frame-0 scene renders (table frame), learning_v6 (current geometry) vs learning_v7_geom (procedural geometry):
full view + cut-away (half of every object removed by a vertical plane through the first object's COM) so nested
configurations (lid in/on pot, sweeper in rack) are visible. Writes outputs/track3/procedural_geometry_v1/eval/scene_epX.png.
Usage: .deps/geom-venv/bin/python scripts/procedural_geometry/render_scenes.py 1 30 ...
"""
import json, sys
from pathlib import Path
import numpy as np, trimesh, cv2
sys.path.insert(0, str(Path(__file__).parent))
from common import T3, OUT, render

COLS = [(220, 120, 90), (90, 150, 220), (120, 200, 120)]


def scene(tree, ep, which='visual'):
    E = T3/tree/f'episode_{ep:06d}/objects'; man = json.loads((E/'objects.json').read_text()); z = np.load(man['poses_path'])
    A = np.asarray(man['table_T_world']); items = {o['name']: o for o in man['objects']}; out = []
    for b, n in enumerate(map(str, z['object_names'])):
        m = trimesh.load(items[n]['mesh_path'], force='mesh')
        if len(m.faces) > 60000:
            import fast_simplification
            m = trimesh.Trimesh(*fast_simplification.simplify(np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32), target_reduction=1 - 60000/len(m.faces)))
        m.apply_transform(A @ z['world_T_canonical'][0, b]); out.append((n, m))
    return out


def cut(ms, origin, normal):
    res = []
    for n, m in ms:
        try: c = m.slice_plane(origin, normal, cap=True)
        except Exception: c = m.slice_plane(origin, normal)
        if c is not None and len(c.faces): res.append((n, c))
    return res


def main(ep):
    rows = []
    for tree in ['learning_v6', 'learning_v7_geom']:
        ms = scene(tree, ep); c0 = ms[0][1].center_mass if ms[0][1].is_volume else ms[0][1].bounds.mean(0)
        allm = trimesh.util.concatenate([m for _, m in ms]); ctr = allm.bounds.mean(0); sp = np.linalg.norm(allm.extents)*0.55
        ground = trimesh.creation.box([sp*2.2, sp*2.2, 0.002]); ground.apply_translation([ctr[0], ctr[1], -0.001])
        cutm = cut(ms, c0, [0, -1, 0])
        tiles = []
        for meshes, view in [(ms, (1, 1, -0.6)), (ms, (0, 1, -0.05)), (cutm, (0, 1, -0.05)), (cutm, (0.4, 1, -0.5))]:
            L = [m for _, m in meshes] + [ground]; C = [COLS[i % 3] for i in range(len(meshes))] + [(235, 225, 210)]
            tiles.append(render(L, C, view, (0, 0, 1), 300, sp, ctr))
        row = np.hstack(tiles); cv2.putText(row, f'ep{ep} f0 {tree}', (5, 18), 0, 0.55, (0, 0, 200), 1); rows.append(row)
    hdr = np.full((22, rows[0].shape[1], 3), 255, np.uint8)
    for i, t in enumerate(['iso', 'side', 'cut-away side', 'cut-away iso']): cv2.putText(hdr, t, (i*300 + 5, 16), 0, 0.5, (0, 0, 0), 1)
    (OUT/'eval').mkdir(exist_ok=True); cv2.imwrite(str(OUT/f'eval/scene_ep{ep}.png'), cv2.cvtColor(np.vstack([hdr] + rows), cv2.COLOR_RGB2BGR))
    print(ep, 'ok', flush=True)


if __name__ == '__main__':
    for e in map(int, sys.argv[1:]): main(e)
