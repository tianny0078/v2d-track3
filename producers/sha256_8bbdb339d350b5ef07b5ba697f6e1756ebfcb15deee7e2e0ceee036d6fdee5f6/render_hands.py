"""Comparison video: source hands vs corrected hands, with object meshes.
Left: RGB frame + projected object mesh (edges, grey) + source skeleton (red) + corrected skeleton (green) + MediaPipe (blue dots).
Right: two orthographic views in the table frame (front x-z, side y-z; table z=0): object surface points + both skeletons.
Usage: render_hands.py --tree hands_contact_v1 8 [--frames 40:120] ; writes NAS mp4 + symlink under outputs/<tree>/<E>/.
"""
import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np
import cv2
import trimesh
sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa
import evidence as EV

BONES = [(PARENT[j], j) for j in range(1, 21)]


def draw_skel(img, uv, col, th=2):
    if not np.isfinite(uv).all(): return
    for a, b in BONES:
        cv2.line(img, tuple(np.round(uv[a]).astype(int)), tuple(np.round(uv[b]).astype(int)), col, th, cv2.LINE_AA)
    for j in TIPS: cv2.circle(img, tuple(np.round(uv[j]).astype(int)), 4, col, -1, cv2.LINE_AA)


def ortho_panel(Xs, cols, objpts, held, W=480, H=380, axes=(0, 2), center=None, span=.45, title=''):
    img = np.full((H, W, 3), 255, np.uint8)
    s = min(W, H)/span

    def P(x): return np.stack([W/2+(x[..., axes[0]]-center[axes[0]])*s, H/2-(x[..., axes[1]]-center[axes[1]])*s], -1)
    if axes[1] == 2: cv2.line(img, (0, int(H/2+center[2]*s)), (W, int(H/2+center[2]*s)), (150, 150, 150), 1)
    for pts, hb in objpts:
        for p in P(pts).astype(int): cv2.circle(img, tuple(p), 1, (0, 160, 230) if hb else (150, 150, 150), -1)
    for X, c in zip(Xs, cols):
        for h in range(2):
            if np.isfinite(X[h]).all(): draw_skel(img, P(X[h]), c, 2)
    cv2.putText(img, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def main(ep, tree, frames=None, fps=10):
    d0 = load_episode(ep); d1 = load_episode(ep, hands_root=tree); ev = EV.build(ep); E = d0['E']; T = d0['T']
    meshes = [trimesh.load(m, force='mesh', process=False) for m in d0['mesh']]
    core = moving_mask(d0['A'] @ d0['P'])
    edges = [m.vertices[m.edges_unique] for m in meshes]
    surf = [trimesh.sample.sample_surface(m, 1500, seed=0)[0] for m in meshes]
    uv0, _ = project(d0['J'], d0['world_T_rgb'], d0['K']); uv1, _ = project(d1['J'], d1['world_T_rgb'], d1['K'])
    J0t = to_table(d0['J'], d0['A']); J1t = to_table(d1['J'], d1['A']); Pt = d0['A'] @ d0['P']
    rng = range(T) if frames is None else range(*frames)
    out = NAS/E/f'hands_{tree.replace("/", "_")}_compare.mp4'; out.parent.mkdir(parents=True, exist_ok=True)
    tmp = WORK/'render'/E; tmp.mkdir(parents=True, exist_ok=True)
    for old in tmp.glob('*.jpg'): old.rename(old.with_suffix('.old'))
    for k, f in enumerate(rng):
        img = cv2.imread(str(T3/E/f'calibrated_sequence/frames/{f:06d}.jpg'))
        for b, e in enumerate(edges):
            X = e.reshape(-1, 3) @ d0['P'][f, b, :3, :3].T+d0['P'][f, b, :3, 3]
            u, z = project(X[None], d0['world_T_rgb'][f:f+1], d0['K']); u = u[0].reshape(-1, 2, 2)
            for a, c in u[::2]:
                if np.isfinite(a).all() and np.isfinite(c).all() and abs(a).max() < 5000 and abs(c).max() < 5000:
                    cv2.line(img, tuple(a.astype(int)), tuple(c.astype(int)), (200, 200, 200) if not core[f, b] else (0, 200, 255), 1)
        for h in range(2):
            draw_skel(img, uv0[f, h], (0, 0, 255), 2); draw_skel(img, uv1[f, h], (0, 200, 0), 2)
            if np.isfinite(ev['uv_mp'][f, h]).all():
                for p in ev['uv_mp'][f, h]: cv2.circle(img, tuple(np.round(p).astype(int)), 3, (255, 120, 0), -1)
        cv2.putText(img, f'ep{ep} f{f}  red=source green={tree} blue=MediaPipe(held-out) yellow=moving obj', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
        objpts = [(surf[b] @ Pt[f, b, :3, :3].T+Pt[f, b, :3, 3], core[f, b]) for b in range(len(meshes))]
        ctr = np.nanmean(np.concatenate([J1t[f].reshape(-1, 3), *[o for o, _ in objpts]]), 0) if np.isfinite(J1t[f]).any() else Pt[f, 0, :3, 3]
        ctr = np.where(np.isfinite(ctr), ctr, Pt[f, 0, :3, 3])
        p1 = ortho_panel([J0t[f], J1t[f]], [(0, 0, 255), (0, 170, 0)], objpts, None, axes=(0, 2), center=ctr, title='table x-z (front)')
        p2 = ortho_panel([J0t[f], J1t[f]], [(0, 0, 255), (0, 170, 0)], objpts, None, axes=(1, 2), center=ctr, title='table y-z (side)')
        right = np.concatenate([p1, p2], 0)
        img = cv2.resize(img, (int(img.shape[1]*760/img.shape[0]), 760))
        cv2.imwrite(str(tmp/f'{k:05d}.jpg'), np.concatenate([img, right], 1))
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', str(tmp/'%05d.jpg'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', str(out)], check=True)
    link = T3/tree/E/out.name
    if not link.exists(): link.symlink_to(out)
    print(out, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('episodes', type=int, nargs='+'); ap.add_argument('--tree', default='hands_contact_v1')
    ap.add_argument('--frames'); a = ap.parse_args()
    for e in a.episodes: main(e, a.tree, tuple(map(int, a.frames.split(':'))) if a.frames else None)
