"""Driver: fit parameters on the current canonical mesh, build the procedural object (high-res + <=4096-face), export
visual / collision / URDF / params / report / comparison render into outputs/track3/procedural_geometry_v1/<name>/.

Usage: .deps/geom-venv/bin/python scripts/procedural_geometry/build.py <name> [<name> ...] [--no-render]
"""
import importlib, json, sys, time
from pathlib import Path
import numpy as np, trimesh
sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, load_base, base_collision, union, difference, mass_props, write_urdf, chamfer, iou, icp, sample, contact_sheet, decimate
import manifold3d as m3d

FACE_BUDGET = 4096


def assemble(parts):
    """visual = manifold union of all parts; mass from density groups (later groups minus earlier ones: no double count)."""
    vis = union([p.mesh for p in parts])
    groups = {}
    for p in parts: groups.setdefault(p.density, []).append(p.mesh)
    done = None; dens = []
    for rho, ms in sorted(groups.items(), key=lambda kv: -sum(m.volume for m in kv[1])):
        g = union(ms)
        if done is not None: g = difference(g, [done])
        if g.volume > 1e-12: dens.append((g, rho))
        done = g if done is None else union([done, g])
    mass, com, I = mass_props(dens)
    col = [c for p in parts for c in p.collision]
    return vis, col, mass, com, I


def rot_deg(M):
    from scipy.spatial.transform import Rotation
    return float(np.degrees(Rotation.from_matrix(M[:3, :3]/np.cbrt(np.linalg.det(M[:3, :3]))).magnitude()))


def main(name, render=True):
    t0 = time.time(); mod = importlib.import_module(name); base = load_base(name)
    params = mod.fit(base)
    hi = mod.build(params, 'high'); lo = mod.build(params, 'low')
    vis, col, mass, com, I = assemble(hi)
    vis_lo, col_lo, mass_lo, _, _ = assemble(lo)
    dec = False
    if len(vis_lo.faces) > FACE_BUDGET:
        d = decimate(vis_lo, FACE_BUDGET); dec = True
        if d is None: raise RuntimeError(f'{name}: low-res {len(vis_lo.faces)} faces and decimation failed')
        vis_lo = d
    D = OUT/name; D.mkdir(parents=True, exist_ok=True)
    for f in D.glob('collision_*.obj'): f.unlink()  # our own previous build output only
    vis.export(D/'visual.obj'); vis_lo.export(D/'visual_4096.obj')
    names = []
    for i, c in enumerate(col):
        fn = f'collision_{i:03d}.obj'; c.export(D/fn); names.append(fn)
    write_urdf(D/'object.urdf', name, mass, com, I, 'visual.obj', names, getattr(mod, 'RGBA', '0.8 0.8 0.8 1'))
    # ---- checks and metrics vs the current mesh (same canonical frame, no re-alignment applied)
    chk = dict(visual=dict(faces=len(vis.faces), watertight=bool(vis.is_watertight), is_volume=bool(vis.is_volume), volume_cm3=float(vis.volume*1e6),
                           euler=int(vis.euler_number), bodies=len(vis.split(only_watertight=False))),
               visual_4096=dict(faces=len(vis_lo.faces), watertight=bool(vis_lo.is_watertight), is_volume=bool(vis_lo.is_volume),
                                volume_cm3=float(vis_lo.volume*1e6), decimated=dec, bodies=len(vis_lo.split(only_watertight=False))),
               collision=dict(parts=len(col), all_convex=bool(all(c.is_convex for c in col)), volume_sum_cm3=float(sum(c.volume for c in col)*1e6)))
    for k, fn in [('visual', 'visual.obj'), ('visual_4096', 'visual_4096.obj')]:  # what a consumer reloading the OBJ sees
        r = trimesh.load(D/fn, force='mesh')
        chk[k]['reload_watertight'] = bool(r.is_watertight); chk[k]['reload_is_volume'] = bool(r.is_volume)
    m = dict(chamfer_new_vs_old=chamfer(vis, base), chamfer_4096_vs_old=chamfer(vis_lo, base))
    try: m['iou_new_vs_old'] = iou(vis, base)
    except Exception as e: m['iou_new_vs_old'] = f'failed: {e}'
    M, rms, med = icp(sample(vis, 20000), base, iters=30)
    m['icp_new_to_old'] = dict(rot_deg=rot_deg(M), trans_mm=float(np.linalg.norm(M[:3, 3])*1000), rms_mm=rms, median_mm=med,
                               note='residual rigid transform that would further align new->old; NOT applied (parameters are fitted in the canonical frame)')
    old_col = base_collision(name)
    phys = dict(mass_kg=float(mass), com=com.tolist(), inertia=I.tolist(), inertia_eig=np.linalg.eigvalsh(I).tolist(),
                inertia_eig_ratio_mean_over_min=float(np.mean(np.linalg.eigvalsh(I))/np.linalg.eigvalsh(I).min()),
                densities={p.label: p.density for p in hi}, old_collision_parts=len(old_col))
    rep = dict(name=name, params=params, checks=chk, metrics=m, physics=phys, seconds=round(time.time()-t0, 1))
    (D/'params.json').write_text(json.dumps(params, indent=1, default=float))
    (D/'report.json').write_text(json.dumps(rep, indent=1, default=float))
    if render:
        cols = [tuple(np.random.default_rng(i).integers(60, 230, 3)) for i in range(len(col))]
        contact_sheet([([base], [(200, 190, 170)]), ([vis], [(120, 170, 220)]), ([vis_lo], [(120, 200, 150)]), (col, cols)],
                      ['current (SAM3D)', f'procedural {len(vis.faces)}f', f'procedural <=4096: {len(vis_lo.faces)}f', f'collision {len(col)} convex'],
                      D/'compare.png', res=220)
    print(json.dumps(dict(name=name, checks=chk, chamfer=m['chamfer_new_vs_old'], iou=m['iou_new_vs_old'], icp=m['icp_new_to_old'],
                          mass=mass, sec=rep['seconds']), default=float), flush=True)
    return rep


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    for n in args: main(n, '--no-render' not in sys.argv)
