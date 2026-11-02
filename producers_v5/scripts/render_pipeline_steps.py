"""Per-episode contact sheets of every current Track3 processing step (no new estimation, only reads existing outputs).

Per episode, 4 frames (first, 1/3, 2/3, last), one 1280x240 strip per step:
  rgb    calibrated_sequence/frames (undistorted ego_cam_a, rgb_K)
  depth  foundation_stereo_v1/depth_rgb (turbo, 0.3-2.0 m, black = invalid)
  masks  object_masks_v3 (SAM2.1) filled + outlined per object
  objpose (scored only) learning_v7_geom world_T_canonical x submitted 4096-face mesh, camera stereo_sequence_orb_objmask
  hands  hands_v6 joints (MANO21 skeleton, left orange / right green) projected with rgb_K
plus camera.png: ORB-SLAM3 camera translation (cm, relative to frame 0) and rotation angle vs frame.
Outputs web/assets/pipeline/<scored|public>_ep<NN>_<step>.jpg + pipeline_index.json.
Usage: python scripts/render_pipeline_steps.py
"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import cv2, numpy as np, trimesh
from scipy.spatial.transform import Rotation

R = Path(__file__).resolve().parents[1]; T3 = R/'outputs/track3'; PUB = R/'outputs/track3_public'; OUT = R/'web/assets/pipeline'
sys.path.insert(0, str(R/'scripts/hand_contact_v1'))
from common import PARENT  # noqa: E402
SC = [1, 3, 5, 6, 8, 10, 14, 15, 16, 19, 20, 25, 28, 30, 32, 33, 35, 36, 37, 38]
PU = [0, 2, 11, 12, 13, 18, 21, 22, 23, 24, 26, 27, 31, 39, 40, 41, 42, 43, 44, 45]
W, H = 320, 240; S = W/1024
OBJ_COL = [(0, 170, 255), (255, 60, 200), (60, 220, 255)]
HAND_COL = [(40, 140, 255), (60, 230, 60)]


def label(img, txt):
    cv2.rectangle(img, (0, 0), (W, 16), (0, 0, 0), -1); cv2.putText(img, txt, (4, 12), 0, .38, (255, 255, 255), 1, cv2.LINE_AA); return img


def rgb(base, f):
    return cv2.resize(cv2.imread(str(base/f'calibrated_sequence/frames/{f:06d}.jpg')), (W, H), interpolation=cv2.INTER_AREA)


def depth(base, f):
    Z = np.load(base/f'foundation_stereo_v1/depth_rgb/{f:06d}.npz')['depth'].astype(np.float32)
    ok = np.isfinite(Z); v = np.clip((np.nan_to_num(Z) - .3)/1.7, 0, 1)
    c = cv2.applyColorMap((v*255).astype(np.uint8), cv2.COLORMAP_TURBO); c[~ok] = 0
    return cv2.resize(c, (W, H), interpolation=cv2.INTER_NEAREST)


def masks(base, f):
    img = rgb(base, f); p = base/f'object_masks_v3/masks/{f:06d}.npz'
    if not p.exists(): return img
    z = np.load(p)
    for b, m in enumerate(z['masks']):
        m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        img[m] = (.5*img[m] + .5*np.array(OBJ_COL[b % 3])).astype(np.uint8)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(img, cs, -1, OBJ_COL[b % 3], 1, cv2.LINE_AA)
    return img


def proj(Xc, K):
    return (Xc @ K.T)[:, :2]/Xc[:, 2:3]*S


def objpose(base, f, ctx):
    img = rgb(base, f); C = ctx['cam'][f]
    if not np.isfinite(C).all(): return label(img, 'no camera pose')
    Ci = np.linalg.inv(C)
    for b, (v, fc) in enumerate(ctx['meshes']):
        Tc = Ci @ ctx['P'][f, b]; X = v @ Tc[:3, :3].T + Tc[:3, 3]
        if (X[:, 2] < .05).any(): continue
        uv = (proj(X, ctx['K'])*16).astype(np.int32); m = np.zeros((H, W), np.uint8)
        for tri in uv[fc]: cv2.fillConvexPoly(m, tri, 1, lineType=cv2.LINE_8, shift=4)
        m = m.astype(bool); img[m] = (.55*img[m] + .45*np.array(OBJ_COL[b % 3])).astype(np.uint8)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(img, cs, -1, OBJ_COL[b % 3], 1, cv2.LINE_AA)
    return img


def hands(base, f, ctx):
    img = rgb(base, f); J = ctx['J'][f]
    for h in range(2):
        if not np.isfinite(J[h]).all() or (J[h][:, 2] < .05).any(): continue
        p = np.round(proj(J[h], ctx['K'])).astype(int)
        for j in range(1, 21): cv2.line(img, tuple(p[PARENT[j]]), tuple(p[j]), HAND_COL[h], 1, cv2.LINE_AA)
        cv2.circle(img, tuple(p[0]), 3, (0, 0, 255), -1)
    return img


def camera_plot(cam, path):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ok = np.isfinite(cam).all((1, 2)); t = (cam[:, :3, 3] - cam[0, :3, 3])*100
    ang = np.degrees(np.linalg.norm(Rotation.from_matrix(np.einsum('ji,fjk->fik', cam[0, :3, :3], np.nan_to_num(cam[:, :3, :3]) + ~ok[:, None, None]*np.eye(3))).as_rotvec(), axis=-1))
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.2), dpi=100)
    for i, (n, c) in enumerate(zip('xyz', ['#c0392b', '#27ae60', '#2471a3'])): ax[0].plot(np.where(ok, t[:, i], np.nan), c, lw=1.2, label=n)
    ax[0].set_title('camera translation (cm, vs frame 0)', fontsize=8); ax[0].legend(fontsize=7, ncol=3, frameon=False)
    ax[1].plot(np.where(ok, ang, np.nan), '#7d3c98', lw=1.2); ax[1].set_title('camera rotation (deg, vs frame 0)', fontsize=8)
    for a in ax: a.tick_params(labelsize=7); a.set_xlabel('frame', fontsize=7); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, facecolor='white'); plt.close(fig)
    return dict(frames=int(len(cam)), pose_ok=int(ok.sum()), max_translation_cm=round(float(np.nanmax(np.linalg.norm(t[ok], axis=-1))), 1), max_rotation_deg=round(float(np.nanmax(ang[ok])), 1))


def run(job):
    split, ep = job; E = f'episode_{ep:06d}'; base = (T3 if split == 'scored' else PUB)/E; tag = f'{split}_ep{ep:02d}'
    n = len(list((base/'calibrated_sequence/frames').glob('*.jpg'))); fr = [0, n//3, 2*n//3, n-1]
    K = np.load(base/'calibrated_sequence/rgb_K.npy')
    cam = np.load(base/'stereo_sequence_orb_objmask/camera_poses.npz')['world_T_rgb']
    hz = np.load(T3/'hands_v6/episodes'/(f'{E}.npz' if split == 'scored' else f'public_{E}.npz'))['joints_camera_m']
    ctx = dict(K=K, cam=cam, J=hz)
    steps = dict(rgb=lambda f: rgb(base, f), depth=lambda f: depth(base, f), masks=lambda f: masks(base, f), hands=lambda f: hands(base, f, ctx))
    info = dict(split=split, episode=ep, frames=n, sample_frames=fr)
    if split == 'scored':
        man = json.loads((T3/'learning_v7_geom'/E/'objects/objects.json').read_text()); z = np.load(man['poses_path'], allow_pickle=True)
        items = {o['name']: o for o in man['objects']}; names = [str(x) for x in z['object_names']]
        ms = []
        for nm in names:
            m = trimesh.load(items[nm].get('mesh_path_4096') or items[nm]['mesh_path'], force='mesh'); ms.append((np.asarray(m.vertices), np.asarray(m.faces)))
        ctx.update(P=np.asarray(z['world_T_canonical'], np.float64), meshes=ms); steps['objpose'] = lambda f: objpose(base, f, ctx); info['objects'] = names
    else:
        import pandas as pd
        d = pd.read_parquet(R/f'data/track3_public/track_3/public/data/chunk-000/{E}.parquet', columns=['observation.objects'])
        info['objects'] = [o['name'] for o in d.iloc[0]['observation.objects']]
    for k, fn in steps.items():
        strip = np.concatenate([label(fn(f), f'{tag}  {k}  f{f}') for f in fr], 1)
        cv2.imwrite(str(OUT/f'{tag}_{k}.jpg'), strip, [cv2.IMWRITE_JPEG_QUALITY, 78])
    info['camera'] = camera_plot(cam, OUT/f'{tag}_camera.png'); info['steps'] = list(steps) + ['camera']
    print(tag, 'ok', flush=True); return info


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [('scored', e) for e in SC] + [('public', e) for e in PU]
    with Pool(16) as p: rows = p.map(run, jobs)
    (OUT/'pipeline_index.json').write_text(json.dumps(rows, indent=1)); print('done', len(rows))
