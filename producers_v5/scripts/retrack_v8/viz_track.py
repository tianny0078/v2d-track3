"""Visual check of a retrack_v8 result: contact sheet (v7 vs new, 24 frames) + mp4 (every frame, mask contour red, new
mesh cyan, v7 mesh outline yellow, per-frame IoU / flags). Usage: viz_track.py <obj> <ep> [tag]"""
import os, sys, json, subprocess
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, cv2
import common as C

obj, ep = sys.argv[1], int(sys.argv[2]); tag = sys.argv[3] if len(sys.argv) > 3 else ''
cx = C.Cx(); od = C.OUTR/f'episode_{ep:06d}'
Z = np.load(od/f'{obj}_poses{tag}.npz', allow_pickle=True); J = json.loads((od/f'{obj}_track{tag}.json').read_text())
mesh = C.Mesh(cx, str(Z['mesh_path'])); s = float(Z['mesh_scale']); e = C.Episode(cx, ep, obj)
v7m = C.Mesh(cx, str(C.T3/f'procedural_geometry_v1/{obj}/visual_4096.obj'))
rows = J['rows']; n = e.n
# ---- contact sheet
fr = np.linspace(0, n - 1, 24).round().astype(int)
top, bot = [], []
for f in fr:
    r = rows[f]
    top.append(C.overlay_tile(cx, v7m, e, int(f), e.camT_v7[f], e.scale_v7, 200, f'v7 f{f} {r["v7_iou"]:.2f}', col=(0, 255, 255)))
    bot.append(C.overlay_tile(cx, mesh, e, int(f), Z['camera_T_object'][f], s, 200, f'new {r["iou"]:.2f}{" ?" if r["uncertain"] else ""}'))
sheet = []
for i in range(0, 24, 8): sheet += [np.hstack(top[i:i+8]), np.hstack(bot[i:i+8])]
cv2.imwrite(str(od/f'{obj}_contact_sheet{tag}.jpg'), np.vstack(sheet), [cv2.IMWRITE_JPEG_QUALITY, 85])
# ---- mp4
import torch
outp = od/f'{obj}_overlay{tag}.mp4'
ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1024x384', '-r', '10', '-i', '-',
                       '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '26', str(outp)], stdin=subprocess.PIPE)


def rend(m, T, sc):
    R = cx.T(np.asarray(T)[None, :3, :3]); tt = cx.T(np.asarray(T)[None, :3, 3]); A = float(sc)*R
    p = tt + torch.einsum('bij,j->bi', A, cx.T(m.cc))
    with torch.no_grad(): h, _, _ = cx.render(m, A, p, cx.T(e.Kv[None]), (C.W, 768))
    return h[0, :C.H].cpu().numpy()


for f in range(n):
    rgb = cv2.imread(str(e.E/f'calibrated_sequence/frames/{f:06d}.jpg')); M = e.obs[f, 0].cpu().numpy() > 0.5
    a7 = rgb.copy(); an = rgb.copy()
    for im, m, T, sc, col in [(a7, v7m, e.camT_v7[f], e.scale_v7, (0, 255, 255)), (an, mesh, Z['camera_T_object'][f], s, (255, 255, 0))]:
        Rh = rend(m, T, sc); im[Rh] = (0.6*im[Rh] + 0.4*np.array(col)).astype(np.uint8)
        cs, _ = cv2.findContours(M.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(im, cs, -1, (0, 0, 255), 2)
        cr, _ = cv2.findContours(Rh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(im, cr, -1, col, 2)
    r = rows[f]
    cv2.putText(a7, f'v7  f{f}  IoU {r["v7_iou"]:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(an, f'retrack_v8  IoU {r["iou"]:.2f} {"static" if r["static_run"] >= 0 else ""} {",".join(r["uncertain"])}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    fr_ = np.hstack([cv2.resize(a7, (512, 384)), cv2.resize(an, (512, 384))])
    ff.stdin.write(fr_.tobytes())
ff.stdin.close(); ff.wait()
print(od/f'{obj}_contact_sheet{tag}.jpg', outp)
