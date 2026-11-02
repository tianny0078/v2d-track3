"""procedural_geometry_v8: simulator assets for the adopted multiep_recon_v1 meshes.

Per object (deform_spec.json from export_v8_specs.py = the multiep_recon param-stage spatial map of the canonical frame):
  visual.obj       = deform(procedural high-res visual.obj x sbar)          (same map that produced final_4096.obj)
  visual_4096.obj  = byte copy of multiep_recon_v1/<obj>/final_4096.obj     (what gets submitted)
  collision_*.obj  = convex hull of deform(procedural convex part x sbar)   (parts rebuilt from the generator params;
                     for v1 objects the rebuilt parts are checked against v1's collision_*.obj)
  object.urdf      = mass / COM / inertia from the generator's density groups, each group deformed (same as build.assemble)
Checks -> report.json: watertight / is_volume (after OBJ reload), collision coverage / concavity fill (eval_collision.stats),
Chamfer(visual.obj, visual_4096.obj), centroid shift / volume / mass vs procedural_geometry_v1.
Usage: .deps/geom-venv/bin/python scripts/procedural_geometry/build_v8.py [obj ...]
"""
import importlib, json, shutil, sys, time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, trimesh
sys.path.insert(0, str(Path(__file__).parent))
from common import T3, union, difference, mass_props, write_urdf, chamfer, sample
from eval_collision import stats as col_stats
from v8_deform import DeformNP

V1 = T3/'procedural_geometry_v1'; OUT = T3/'procedural_geometry_v8'
OBJS = ['blue_cup', 'plastic_dish_rack', 'white_pot', 'beige_cup', 'white_pot_lid', 'wooden_piece_1', 'wooden_piece_2', 'wooden_spoon']


def groups_by_density(parts):
    """build.assemble's density grouping (later groups minus earlier ones), returned as [(mesh, rho)]."""
    groups = {}
    for p in parts: groups.setdefault(p.density, []).append(p.mesh)
    done = None; dens = []
    for rho, ms in sorted(groups.items(), key=lambda kv: -sum(m.volume for m in kv[1])):
        g = union(ms)
        if done is not None: g = difference(g, [done])
        if g.volume > 1e-12: dens.append((g, rho))
        done = g if done is None else union([done, g])
    return dens


def urdf_collisions(u):
    return [u.parent/c.find('geometry/mesh').attrib['filename'] for c in ET.parse(u).getroot().iter('collision')]


def surf_dist(a, b, n=20000):
    """exact symmetric point-to-surface distance (mm): n area-uniform samples on each mesh -> closest point on the other."""
    da = np.abs(trimesh.proximity.closest_point(b, sample(a, n))[1]); db = np.abs(trimesh.proximity.closest_point(a, sample(b, n, ))[1])
    return dict(mean_mm=float((da.mean() + db.mean())/2*1000), p95_mm=float(np.percentile(np.r_[da, db], 95)*1000), max_mm=float(max(da.max(), db.max())*1000))


def mesh_check(path):
    r = trimesh.load(path, force='mesh')
    return dict(faces=len(r.faces), vertices=len(r.vertices), watertight=bool(r.is_watertight), is_volume=bool(r.is_volume),
                euler=int(r.euler_number), volume_cm3=float(r.volume*1e6), extents_mm=(r.extents*1000).round(2).tolist())


def build(obj):
    t0 = time.time(); D = OUT/obj; sp = json.loads((D/'deform_spec.json').read_text())
    dfm0 = DeformNP.from_spec(sp['spec']); s = float(sp['spec']['sbar'])
    PR = np.asarray(sp['spec']['post_rigid'], float) if sp['spec'].get('post_rigid') is not None else None
    def dfm(V, F):   # deformation (+ optional rigid re-framing into the v1 canonical frame, white_pot 2026-10-31)
        X = dfm0(V, F)
        return X if PR is None else X @ PR[:3, :3].T + PR[:3, 3]
    proc = Path(sp['proc_mesh']); pp = json.loads(Path(sp['proc_params']).read_text())
    mod = importlib.import_module(obj)
    hi = mod.build(pp, 'high')
    rep = dict(object=obj, deform_spec=str(D/'deform_spec.json'), sbar=s, proc_mesh=str(proc), proc_params=sp['proc_params'])
    # ---- rebuild sanity: generator(params) reproduces the procedural source assets
    src_vis = trimesh.load(proc, force='mesh', process=False)
    vis_rebuilt = union([p.mesh for p in hi])
    rep['rebuild_vs_source_visual_chamfer'] = chamfer(vis_rebuilt, src_vis)
    col_src = [c for p in hi for c in p.collision]
    src_urdf = proc.parent/'object.urdf'
    if src_urdf.exists():
        v1c = [trimesh.load(f, force='mesh', process=False) for f in urdf_collisions(src_urdf)]
        rep['rebuild_vs_v1_collision'] = dict(n_rebuilt=len(col_src), n_v1=len(v1c),
                                              max_vertex_dev_mm=float(max(np.abs(np.sort(a.vertices, 0) - np.sort(b.vertices, 0)).max() for a, b in zip(col_src, v1c))*1000)
                                              if len(v1c) == len(col_src) and all(len(a.vertices) == len(b.vertices) for a, b in zip(col_src, v1c)) else None)
        if len(v1c) == len(col_src): col_src = v1c   # use the exact v1 files when they correspond one-to-one
    # ---- visual (high-res) and the submitted 4096 mesh
    Vh = dfm(src_vis.vertices*s, src_vis.faces)
    vis = trimesh.Trimesh(Vh, src_vis.faces, process=False)
    if vis.volume < 0: vis.invert()
    for f in D.glob('collision_*.obj'): f.unlink()   # our own previous build output only
    vis.export(D/'visual.obj')
    if PR is None:
        shutil.copyfile(sp['final_4096'], D/'visual_4096.obj')
        assert (D/'visual_4096.obj').read_bytes() == Path(sp['final_4096']).read_bytes()
    else:   # submitted mesh = adopted final_4096 rigidly re-framed (same faces, vertices x PR); final_4096 itself untouched
        f4 = trimesh.load(sp['final_4096'], process=False, force='mesh')
        m4 = trimesh.Trimesh(f4.vertices @ PR[:3, :3].T + PR[:3, 3], f4.faces, process=False); m4.export(D/'visual_4096.obj')
        rep['visual_4096_from'] = dict(final_4096=sp['final_4096'], post_rigid=PR.tolist(), faces_identical=True)
    # ---- collision: deform each convex part, re-take the hull
    names = []; cols = []
    for i, c in enumerate(col_src):
        cv = dfm(c.vertices*s, c.faces); h = trimesh.Trimesh(cv, process=False).convex_hull
        fn = f'collision_{i:03d}.obj'; h.export(D/fn); names.append(fn); cols.append(h)
    # ---- mass properties: density groups deformed
    dens = groups_by_density(hi)
    m0, c0, I0 = mass_props([(trimesh.Trimesh(g.vertices*s, g.faces, process=False), rho) for g, rho in dens])
    dd = []
    for g, rho in dens:
        gm = trimesh.Trimesh(dfm(g.vertices*s, g.faces), g.faces, process=False)
        if gm.volume < 0: gm.invert()
        dd.append((gm, rho))
    mass, com, I = mass_props(dd)
    write_urdf(D/'object.urdf', obj, mass, com, I, 'visual.obj', names, getattr(mod, 'RGBA', '0.8 0.8 0.8 1'))
    # ---- checks
    vis_r = trimesh.load(D/'visual.obj', force='mesh'); v4_r = trimesh.load(D/'visual_4096.obj', force='mesh')
    v1_vis = trimesh.load(V1/obj/'visual.obj', force='mesh')
    rep['checks'] = dict(visual=mesh_check(D/'visual.obj'), visual_4096=mesh_check(D/'visual_4096.obj'),
                         visual_4096_bytes_equal_final=PR is None,
                         collision=dict(parts=len(cols), all_convex=bool(all(c.is_convex for c in cols)),
                                        volume_sum_cm3=float(sum(c.volume for c in cols)*1e6)))
    rep['chamfer_visual_vs_visual4096'] = chamfer(vis_r, v4_r)
    rep['surface_dist_visual_vs_visual4096'] = surf_dist(vis_r, v4_r)
    rep['collision_quality'] = dict(v8=col_stats(vis_r, cols),
                                    v1=col_stats(v1_vis, [trimesh.load(f, force='mesh') for f in urdf_collisions(V1/obj/'object.urdf')]))
    pts8 = sample(vis_r, 100000); pts1 = sample(v1_vis, 100000)
    rep['vs_v1'] = dict(surface_centroid_shift_mm=float(np.linalg.norm(pts8.mean(0) - pts1.mean(0))*1000),
                        volume_ratio=float(vis_r.volume/v1_vis.volume), chamfer_visual=chamfer(vis_r, v1_vis),
                        note='v1 unscaled; blue_cup v7 used per-episode copies x 0.911-0.949, v8 bakes the shared sbar into the mesh')
    v1rep = json.loads((V1/obj/'report.json').read_text())['physics'] if (V1/obj/'report.json').exists() else {}
    rep['physics'] = dict(mass_kg=float(mass), com=com.tolist(), inertia=I.tolist(), inertia_eig=np.linalg.eigvalsh(I).tolist(),
                          inertia_eig_ratio_mean_over_min=float(np.mean(np.linalg.eigvalsh(I))/np.linalg.eigvalsh(I).min()),
                          undeformed_groups_mass_kg=float(m0), v1_mass_kg=v1rep.get('mass_kg'), mass_ratio_vs_undeformed=float(mass/m0),
                          densities={p.label: p.density for p in hi}, mass_from='generator density groups, each deformed by the adopted map')
    rep['seconds'] = round(time.time() - t0, 1)
    (D/'report.json').write_text(json.dumps(rep, indent=1, default=float))
    c = rep['checks']; q = rep['collision_quality']
    print(json.dumps(dict(obj=obj, vis=[c['visual']['watertight'], c['visual']['is_volume'], c['visual']['faces']],
                          v4096=[c['visual_4096']['watertight'], c['visual_4096']['is_volume'], c['visual_4096']['faces']],
                          cham_vis_4096_mm=round(rep['chamfer_visual_vs_visual4096']['chamfer_mean_mm'], 3),
                          surf_vis_4096_mm=[round(rep['surface_dist_visual_vs_visual4096'][k], 3) for k in ('mean_mm', 'p95_mm', 'max_mm')],
                          rebuild_cham_mm=round(rep['rebuild_vs_source_visual_chamfer']['chamfer_mean_mm'], 4),
                          col=[len(cols), c['collision']['all_convex']], cov=[round(q['v1']['coverage'], 3), round(q['v8']['coverage'], 3)],
                          fill=[round(q['v1']['concavity_fill'], 3), round(q['v8']['concavity_fill'], 3)],
                          mass=[v1rep.get('mass_kg'), round(m0, 4), round(mass, 4)], centroid_mm=round(rep['vs_v1']['surface_centroid_shift_mm'], 2),
                          sec=rep['seconds'])), flush=True)


if __name__ == '__main__':
    for o in (sys.argv[1:] or OBJS): build(o)
