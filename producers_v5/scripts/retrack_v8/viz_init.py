"""Contact sheet of the global-init candidates: per candidate v7 pose (top) vs global init (bottom). Usage: viz_init.py <obj> <ep>"""
import os, sys, json
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, cv2
import common as C
obj, ep = sys.argv[1], int(sys.argv[2])
cx = C.Cx(); od = C.OUTR/f'episode_{ep:06d}'
J = json.loads((od/f'{obj}_global_init.json').read_text()); Z = np.load(od/f'{obj}_global_init.npz')
mesh = C.Mesh(cx, str(Z['mesh_path'])); e = C.Episode(cx, ep, obj)
top, bot = [], []
for j, f, T in zip(J['candidates'], Z['frames'], Z['camera_T_object']):
    top.append(C.overlay_tile(cx, mesh, e, int(f), e.camT_v7[f], e.scale_v7, 200, f'v7 f{f} {j["v7_iou"]:.2f}', col=(0, 255, 255)))
    bot.append(C.overlay_tile(cx, mesh, e, int(f), T, float(Z['scale']), 200, f'init {j["init_iou"]:.2f}{" AMB" if j["ambiguous"] else ""}'))
rows = []
for i in range(0, len(top), 8):
    a, b = top[i:i+8], bot[i:i+8]; pad = [np.zeros_like(top[0])]*(8 - len(a)); rows += [np.hstack(a + pad), np.hstack(b + pad)]
cv2.imwrite(str(od/f'{obj}_global_init_sheet.jpg'), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85]); print(od/f'{obj}_global_init_sheet.jpg')
