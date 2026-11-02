"""Export parametric-stage arms of multiep_recon (blue_cup) as <=4096-face watertight meshes (no public data).

arm_P{i}.obj            : mesh_4096.obj of param-only ablation run P{i} (same export path as recon/export.py)
arm_P1_cad4096.obj      : P1 profile deformation applied directly to procedural visual_4096.obj (no re-simplification)
arm_param_nomarkers.obj : P1 profile deformation applied to a marker-free revolved cup body (cup.py profile)
Usage: export_arms.py --runs <dir_with_P1..P4> --out outputs/track3/multiep_recon_v1/blue_cup
"""
import argparse, json, shutil
from pathlib import Path
import numpy as np, trimesh, torch

ROOT = Path(__file__).resolve().parents[2]
PG = ROOT/'outputs/track3/procedural_geometry_v1/blue_cup'


def deform(V, F, pp, st, sbar):
    """numpy copy of recon.py param-stage deformation (V already x sbar)."""
    A = np.array(pp['axis'], float); A /= np.linalg.norm(A); C = np.array(pp['centre'])*sbar
    d = V - C; h = d @ A; rv = d - h[:, None]*A; rho = np.linalg.norm(rv, axis=1); u = rv/np.maximum(rho, 1e-9)[:, None]
    h0, h1 = pp['h0']*sbar, pp['h1']*sbar
    vn = trimesh.Trimesh(V, F, process=False).vertex_normals
    inner = ((vn*u).sum(1) < -0.5) & (rho > 0.5*pp['r_bottom_out']*sbar)
    prof = np.log(np.array(st['prof'])); kp = len(prof); dh = np.array(st['dh']); dt = st['dt']
    x = np.clip((h - h0)/(h1 - h0), 0, 1)*(kp-1); i0 = np.clip(np.floor(x), 0, kp-2).astype(int); w = x - i0
    ls = prof[i0]*(1-w) + prof[i0+1]*w
    hn = h0 + dh[0] + (h - h0)*((h1 + dh[1]) - (h0 + dh[0]))/(h1 - h0)
    rn = rho*np.exp(ls) - inner*dt
    return C + hn[:, None]*A + rn[:, None]*u


def revolved_body(pp, n=64):
    """Marker-free cup body: closed cup.py profile (bottom + chamfer + outer wall + rim + inner wall + floor) revolved."""
    R = lambda hh: pp['r_a'] + pp['r_b']*hh
    h0, h1, t, f = pp['h0'], pp['h1'], pp['t_wall'], pp['chamfer_bottom']; tb = max(pp['t_bottom'], f + 0.0005)
    prof = [(0.0, h0)]
    prof += [((R(h0)-f)*k/4, h0) for k in range(1, 4)] + [(R(h0)-f, h0), (R(h0+f), h0+f)]
    prof += [(R(hh), hh) for hh in np.linspace(h0+f, h1, 11)[1:]]
    prof += [(R(hh)-t, hh) for hh in np.linspace(h1, h0+tb, 11)]
    prof += [((R(h0+tb)-t)*k/4, h0+tb) for k in range(3, 0, -1)] + [(0.0, h0+tb)]
    prof = np.array(prof)
    A = np.array(pp['axis'], float); A /= np.linalg.norm(A); C = np.array(pp['centre'])
    e1 = np.array(pp['x_hint'], float); e1 -= (e1 @ A)*A; e1 /= np.linalg.norm(e1); e2 = np.cross(A, e1)
    rings = prof[1:-1]; K = len(rings); th = np.linspace(0, 2*np.pi, n, endpoint=False)
    V = [C + prof[0, 1]*A]
    for r, hh in rings:
        V += list(C + hh*A + r*(np.cos(th)[:, None]*e1 + np.sin(th)[:, None]*e2))
    V.append(C + prof[-1, 1]*A); V = np.array(V); top = len(V)-1
    idx = lambda k, j: 1 + k*n + (j % n)
    F = []
    for j in range(n): F.append([0, idx(0, j+1), idx(0, j)])
    for k in range(K-1):
        for j in range(n):
            a, b, c, d = idx(k, j), idx(k, j+1), idx(k+1, j), idx(k+1, j+1)
            F += [[a, b, d], [a, d, c]]
    for j in range(n): F.append([top, idx(K-1, j), idx(K-1, j+1)])
    m = trimesh.Trimesh(V, np.array(F), process=True)
    if m.volume < 0: m.invert()
    return m


def check(m):
    w = trimesh.Trimesh(m.vertices, m.faces, process=True)
    return dict(faces=len(m.faces), vertices=len(m.vertices), watertight=bool(w.is_watertight), is_volume=bool(w.is_volume),
                euler=int(w.euler_number), volume_cm3=round(float(w.volume*1e6), 2), extents_mm=(w.extents*1000).round(1).tolist())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--runs', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); out = Path(a.out); runs = Path(a.runs)
    pp = json.loads((PG/'params.json').read_text())
    info = {}
    for i in (1, 2, 3, 4):
        src = runs/f'P{i}/blue_cup/mesh_4096.obj'
        dst = out/f'arm_P{i}.obj'
        if not dst.exists(): shutil.copyfile(src, dst)
        assert dst.read_bytes() == src.read_bytes(), dst; info[dst.name] = check(trimesh.load(dst, process=False, force='mesh'))
    st = torch.load(runs/'P1/blue_cup/stage_param.pt', weights_only=False)
    init = np.load(runs/'P1/blue_cup/init.npz'); sbar = float(init['sbar'])
    # sanity: numpy deformation reproduces the P1 stage mesh
    Vchk = deform(init['V'], init['F'], pp, st, sbar)
    e = np.linalg.norm(Vchk - st['V'], axis=1); err = e.max()
    print(f'deform reproduction: max err {err*1000:.3f} mm, vertices >1e-5 m: {(e > 1e-5).sum()}/{len(e)} (inner-wall normal classification at rim; dt={st["dt"]*1000:.2f} mm)')
    assert err < st['dt'] + 1e-4
    cad = trimesh.load(PG/'visual_4096.obj', process=False, force='mesh')
    m = trimesh.Trimesh(deform(cad.vertices*sbar, cad.faces, pp, st, sbar), cad.faces, process=False)
    m.export(out/'arm_P1_cad4096.obj'); info['arm_P1_cad4096.obj'] = check(m)
    body = revolved_body(pp)
    mb = trimesh.Trimesh(deform(body.vertices*sbar, body.faces, pp, st, sbar), body.faces, process=False)
    mb.export(out/'arm_param_nomarkers.obj'); info['arm_param_nomarkers.obj'] = check(mb)
    (out/'arms.json').write_text(json.dumps(info, indent=1)); print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
