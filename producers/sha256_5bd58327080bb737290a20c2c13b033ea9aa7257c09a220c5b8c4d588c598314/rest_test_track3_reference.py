"""Hands-off rest test of a learning tree's frame-0 object state in plain MuJoCo (proxy for the FlashCHORD scene).

Scene = ground plane z=0 (table frame) + every object as a free body with its URDF collision hulls and URDF inertial.
Objects are placed at the object reference pose of frame 0 and simulated SECONDS with no hands. Reports per object the
initial penetration (-min contact distance), COM drift and rotation after SECONDS. Pass = pen <= 1 mm, drift <= 5 mm,
rot <= 3 deg. Usage: .deps/flash-chord-venv/bin/python scripts/rest_test_track3_reference.py <tree> [episodes...]
Writes outputs/track3/<tree>/rest_test.json.
"""
import json, sys, xml.etree.ElementTree as ET
from pathlib import Path
import mujoco, numpy as np
from scipy.spatial.transform import Rotation

R = Path(__file__).resolve().parents[1]; SECONDS = 2.0
EPS = [1, 3, 5, 6, 8, 10, 14, 15, 16, 19, 20, 25, 28, 30, 32, 33, 35, 36, 37, 38]

def mjcf(objs, free=None, damping=0.):
    """objs: (name, urdf, table_T_link); free: indices of free bodies (default all), the rest are static."""
    assets, bodies = [], []
    for k, (name, urdf, T0) in enumerate(objs):
        u = Path(urdf); root = ET.parse(u).getroot(); inr = root.find('link/inertial')
        com = inr.find('origin').attrib['xyz']; mass = inr.find('mass').attrib['value']; I = inr.find('inertia').attrib
        geoms = []
        for i, c in enumerate(root.iter('collision')):
            f = (u.parent/c.find('geometry/mesh').attrib['filename']).resolve(); assets.append(f'<mesh name="m{k}_{i}" file="{f}"/>')
            geoms.append(f'<geom type="mesh" mesh="m{k}_{i}" contype="1" conaffinity="1" friction="0.8 0.01 0.001"/>')
        q = Rotation.from_matrix(T0[:3, :3]).as_quat(scalar_first=True); p = T0[:3, 3]
        fj = (f'<joint type="free" damping="{damping}"/>' if damping else '<freejoint/>') if free is None or k in free else ''
        bodies.append(f'<body name="{name}" pos="{p[0]} {p[1]} {p[2]}" quat="{q[0]} {q[1]} {q[2]} {q[3]}">{fj}'
                      f'<inertial pos="{com}" mass="{mass}" fullinertia="{I["ixx"]} {I["iyy"]} {I["izz"]} {I["ixy"]} {I["ixz"]} {I["iyz"]}"/>'
                      + ''.join(geoms) + '</body>')
    return (f'<mujoco><compiler inertiafromgeom="false"/><option timestep="0.002" integrator="implicitfast"/>'
            f'<asset>{"".join(assets)}</asset><worldbody><geom type="plane" size="5 5 0.1" friction="0.8 0.01 0.001"/>{"".join(bodies)}</worldbody></mujoco>')

def run(tree, ep):
    E = R/'outputs/track3'/tree/f'episode_{ep:06d}/objects'; man = json.loads((E/'objects.json').read_text())
    z = np.load(man['poses_path']); A = np.asarray(man['table_T_world'], float); items = {o['name']: o for o in man['objects']}
    names = [str(x) for x in z['object_names']]; objs = [(n, items[n]['urdf_path'], A @ z['world_T_canonical'][0, b]) for b, n in enumerate(names)]
    m = mujoco.MjModel.from_xml_string(mjcf(objs)); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    pen = {n: 0. for n in names}
    for i in range(d.ncon):
        c = d.contact[i]
        for g in (c.geom1, c.geom2):
            b = m.body(m.geom_bodyid[g]).name
            if b in pen: pen[b] = max(pen[b], -1000*c.dist)
    x0 = {n: d.body(n).xipos.copy() for n in names}; r0 = {n: d.body(n).xmat.reshape(3, 3).copy() for n in names}
    for _ in range(int(SECONDS/m.opt.timestep)): mujoco.mj_step(m, d)
    out = {}
    for n in names:
        dr = float(np.linalg.norm(d.body(n).xipos - x0[n])*1000); rot = float(np.degrees(Rotation.from_matrix(r0[n].T @ d.body(n).xmat.reshape(3, 3)).magnitude()))
        ok = bool(np.isfinite(dr) and pen[n] <= 1.0 and dr <= 5.0 and rot <= 3.0)
        out[n] = dict(init_pen_mm=round(pen[n], 2), drift_mm=round(dr, 2), rot_deg=round(rot, 2), passed=ok)
    return out

if __name__ == '__main__':
    tree = sys.argv[1]; eps = list(map(int, sys.argv[2:])) or EPS; p = R/'outputs/track3'/tree/'rest_test.json'
    rep = json.loads(p.read_text()) if p.exists() else {}
    for e in eps:
        rep[str(e)] = run(tree, e); print(e, json.dumps(rep[str(e)]), flush=True)
    p.write_text(json.dumps(rep, indent=1))
    allo = [o for e in rep.values() for o in e.values()]
    print(f'{tree}: objects passed {sum(o["passed"] for o in allo)}/{len(allo)}; median init pen {np.median([o["init_pen_mm"] for o in allo]):.1f} mm, median drift {np.median([o["drift_mm"] for o in allo]):.1f} mm')
