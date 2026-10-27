"""How does the human pick up the flat spoon? (reference objects + corrected human hands hands_contact_v1; no GT)
Per spoon episode: first core segment of the spoon; at core start s and s+10: spoon long-axis pitch vs table, min z of
the spoon mesh (table frame), position along the long axis (0 = bowl end .. 1 = handle end) of the thumb/index tips of
the assigned hand, tip heights above the table, and whether any fingertip is below the spoon's top surface height there."""
import json, sys
from pathlib import Path
import numpy as np, trimesh
sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa
EPS_SPOON = [3, 5, 6, 15, 16, 32, 33, 35, 36, 37]
res = {}
for ep in EPS_SPOON:
    d = load_episode(ep, 'hands_contact_v1'); b = d['names'].index('wooden_spoon'); Pt = d['A'] @ d['P']
    m = trimesh.load(d['mesh'][b], force='mesh', process=False); V = np.asarray(m.vertices)
    ax = np.argmax(m.extents); core = moving_mask(Pt)[:, b]; s = int(np.flatnonzero(core)[0]); e = s+np.flatnonzero(~core[s:])[0] if (~core[s:]).any() else len(core)
    z = np.load(T3/'hands_contact_v1'/d['E']/'hands_world.npz', allow_pickle=True); held = z['hc_held'][:, :, b]
    h = int(np.argmax(held[s:e].sum(0))) if held[s:e].any() else None
    Jt = to_table(d['J'], d['A'])
    # which end is the bowl: the end with larger cross-section
    lo, hi = V[:, ax].min(), V[:, ax].max(); n1 = np.ptp(V[V[:, ax] < lo+.04][:, [i for i in range(3) if i != ax]], 0).prod(); n2 = np.ptp(V[V[:, ax] > hi-.04][:, [i for i in range(3) if i != ax]], 0).prod()
    sign = 1 if n1 > n2 else -1  # u = 0 at the bowl end
    def u_of(Xt, f):
        L = (Xt-Pt[f, b, :3, 3]) @ Pt[f, b, :3, :3]; uu = (L[..., ax]-lo)/(hi-lo); return uu if sign == 1 else 1-uu
    rows = []
    for f in [max(0, s-6), s, min(s+10, len(core)-1), min(s+20, len(core)-1)]:
        W = V @ Pt[f, b, :3, :3].T+Pt[f, b, :3, 3]; axis_w = Pt[f, b, :3, ax]
        pitch = float(np.degrees(np.arcsin(abs(axis_w[2]))))
        r = dict(frame=f, spoon_min_z_cm=round(float(W[:, 2].min())*100, 1), spoon_max_z_cm=round(float(W[:, 2].max())*100, 1), pitch_deg=round(pitch, 1))
        if h is not None and np.isfinite(Jt[f, h]).all():
            tips = Jt[f, h, TIPS]; r.update(hand=['left', 'right'][h], tip_u=np.round(u_of(tips, f), 2).tolist(), tip_z_cm=np.round(tips[:, 2]*100, 1).tolist())
        rows.append(r)
    # lift shape: z of the bowl end vs handle end over the segment
    ends = []
    for f in range(s, e):
        W = V @ Pt[f, b, :3, :3].T+Pt[f, b, :3, 3]; uu = (V[:, ax]-lo)/(hi-lo); uu = uu if sign == 1 else 1-uu
        ends.append((W[uu < .15, 2].mean(), W[uu > .85, 2].mean()))
    ends = np.array(ends)
    res[ep] = dict(core=[int(s), int(e)], hand=None if h is None else ['left', 'right'][h], frames=rows,
                   bowl_end_rise_cm_first10=round(float(ends[min(10, len(ends)-1), 0]-ends[0, 0])*100, 1),
                   handle_end_rise_cm_first10=round(float(ends[min(10, len(ends)-1), 1]-ends[0, 1])*100, 1),
                   spoon_thickness_cm=round(float(sorted(m.extents)[0])*100, 1))
    print(ep, json.dumps(res[ep]), flush=True)
save_json(T3/'hands_contact_v1'/'spoon_pickup.json', res)
