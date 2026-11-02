"""VALIDATION ONLY: does the new multi-episode mesh land on the same registration branch as the old (submission #4) mesh
when the host registers submitted meshes to its GT scans?

Host path (toolkit/8d2bce64 metric_code/track_3/MPPE.py::_score_registered_track3):
    candidate = _decode_object_mesh(payload)   # payload = packer's budget_mesh(path, 4096, 4096), float64 round-trip
    candidate_anchor, R = _register_object_body(candidate, (v_i, f_i), anchor_i, 1_000_003*episode + 101*slot)
    achieved = _convert_object_body(achieved, candidate_anchor, R)   # orientation -> R_pred @ R.T
so swapping old->new mesh changes every achieved orientation by A_new = A_old @ M, M = R_old @ R_new.T (reference body
frame), and every position by R_pred @ (c_new - c_old).  The metric's symmetry (config/tracks.json + README): blue_cup,
beige_cup axial (any spin about the axis free, axis direction NOT free); white_pot_lid half-turn (order 2); all other
objects (incl. white_pot) full orientation scored.

Proxy for the confidential host reference: the public scan (data/track3_public/.../mesh/<name>), anchor = scan surface
centroid (also reported: anchor = scan-file origin). Symmetry axis in the scan frame = exact triangle-area covariance PCA
(max eigenvector for cups, min for lid), as in scripts/evaluate_track3_public_diagnostic.py::reference_axis.
Scans are only used here as an evaluation reference; nothing is written back into any reconstruction.

Output: outputs/track3/procedural_geometry_v8/eval/registration_branch.{json,md}
Usage: REG_PROCS=24 python3 scripts/multiep_recon_eval/check_registration_branch.py all|<objects...>  (CPU only; subset -> *_<objs>.json)
       REG_PROCS=24 python3 scripts/multiep_recon_eval/check_registration_branch.py --sweep 40 wooden_piece_1 white_pot_lid ...
"""
import importlib.util, json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

R = Path(__file__).resolve().parents[2]; KIT = R/'toolkit/8d2bce64/v2d_submission_kit'
spec = importlib.util.spec_from_file_location('mppe', KIT/'metric_code/track_3/MPPE.py')
MP = importlib.util.module_from_spec(spec); spec.loader.exec_module(MP)
sys.path.insert(0, str(KIT)); from v2dlb.mesh_budget import budget_mesh  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent)); from eval_public_scans import scan  # noqa: E402

PUB = R/'data/track3_public/track_3/public'; OLD = R/'outputs/track3/procedural_geometry_v1'
NEW = R/'outputs/track3/multiep_recon_v1'; READY = R/'outputs/track3/retrack_v8/READY.json'
OUT = R/'outputs/track3/procedural_geometry_v8/eval'
OBJECTS = ['blue_cup', 'plastic_dish_rack', 'white_pot', 'beige_cup', 'white_pot_lid', 'wooden_piece_1', 'wooden_piece_2']
AXIAL, HALF_TURN = {'blue_cup', 'beige_cup'}, {'white_pot_lid'}
EXTRA_SEEDS = [7919 * k for k in range(1, 6)]   # offsets added to the official seed for the robustness check
SAME_DEG, DIFF_DEG = 10.0, 30.0                    # symmetry-aware angle: <10 same branch, >30 different branch


def mesh_paths():
    paths = {o: (OLD/o/'visual_4096.obj', NEW/o/'final_4096.obj') for o in OBJECTS}
    skipped = {'mini_dust_pan': 'keeps the old mesh (multiep variant not adopted)'}
    ready = json.loads(READY.read_text()) if READY.is_file() else {}
    for o in ('mini_sweeper', 'water_pitcher'):
        p = ready.get(o) if isinstance(ready.get(o), str) else (ready.get(o) or {}).get('mesh') if isinstance(ready.get(o), dict) else None
        if p and (R/p).is_file(): paths[o] = (OLD/o/'visual_4096.obj', R/p)
        else: skipped[o] = f'no adopted mesh ({READY.relative_to(R)} {"missing" if not READY.is_file() else "does not name one"})'
    return paths, skipped


def occurrences():
    occ = {}
    for f in sorted((PUB/'data/chunk-000').glob('episode_*.parquet')):
        ep = int(f.stem.split('_')[1]); d = pd.read_parquet(f, columns=['observation.objects'])
        for slot, o in enumerate(d.iloc[0]['observation.objects']): occ.setdefault(o['name'], []).append((ep, slot))
    return occ


def packed(path):
    """Exactly what the host decodes: budget_mesh -> savez payload -> _decode_object_mesh."""
    v, f = budget_mesh(path, 4096, 4096)
    return MP._decode_object_mesh(MP._encode_archive(vertices=v, faces=f))


def align_all_starts(reference, candidate):
    """Line-for-line copy of MP._align_rigid that also keeps every start's result (official picks min score)."""
    tree = cKDTree(reference); out = []
    for start in MP._initial_rotations(reference, candidate, MP._ALIGNMENT_STARTS):
        rotation = start.copy(); translation = reference.mean(axis=0) - rotation @ candidate.mean(axis=0); previous = math.inf
        for _ in range(MP._ICP_ITERATIONS):
            moved = candidate @ rotation.T + translation
            distances, indices = tree.query(moved, k=1)
            keep = max(3, int(math.ceil(len(distances) * MP._TRIM_FRACTION)))
            chosen = np.argpartition(distances, keep - 1)[:keep] if keep < len(distances) else np.arange(len(distances))
            sr, st = MP._kabsch(moved[chosen], reference[indices[chosen]])
            rotation = sr @ rotation; translation = sr @ translation + st
            error = float(np.mean(distances[chosen]))
            if math.isfinite(previous) and abs(previous - error) <= 1e-8 * max(previous, 1.0): break
            previous = error
        out.append((rotation, translation, MP._symmetric_mean_distance(reference, candidate @ rotation.T + translation)))
    return out


def ang(Rm):
    return float(np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1))))


def vang(a, b):
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))))


def sym_angle(R1, R2, name, axis):
    """Metric-equivalent orientation difference between body rotations R1, R2 (cand->ref): M = R1 R2^T acts on the
    reference body frame; mirrors MP._symmetry_orientation_errors with the reference at identity."""
    M = R1 @ R2.T
    if name in AXIAL: return vang(M @ axis, axis)
    if name in HALF_TURN: return min(ang(M), ang(M @ Rotation.from_rotvec(np.pi * axis).as_matrix().T))
    return ang(M)


def keypoint_shift_cm(Ro, co, Rn, cn, name, axis):
    """MPPE-style keypoint error (official _symmetry_point_errors, 6 pts @ 5 cm, declared symmetry) between the
    converted poses from old vs new registration for the same predicted pose (before the initial-scene transform)."""
    M = Ro @ Rn.T; t = Ro @ (cn - co)
    q = Rotation.from_matrix(M).as_quat()
    ach = np.concatenate([t, q])[None, None]; ref = np.array([0, 0, 0, 0, 0, 0, 1.0])[None, None]
    ax = axis if name in AXIAL | HALF_TURN else np.zeros(3); order = 2 if name in HALF_TURN else 360
    pts = MP._MPPE_KEYPOINT_DIRECTIONS * MP._MPPE_KEYPOINT_RADIUS_M
    return float(MP._symmetry_point_errors(ach, ref, pts, ax, order)[0, 0] * 100)


def scan_axis(v, f, name):
    if name not in AXIAL | HALF_TURN: return None
    c, _ = MP._surface_moments(v, f); tri = v[f] - c
    areas = .5 * np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1); s = tri.sum(axis=1)
    cov = np.einsum('t,tij->ij', areas, (np.einsum('tki,tkj->tij', tri, tri) + np.einsum('ti,tj->tij', s, s)) / 12) / areas.sum()
    w, vec = np.linalg.eigh(cov); a = vec[:, 0 if name in HALF_TURN else -1]
    return (a if a[np.argmax(np.abs(a))] > 0 else -a), w.tolist()


def cand_axis(v, f, seed=0):
    """Candidate-frame symmetry axis: which of link x/y/z (through the surface centroid) the mesh is most invariant to
    under a 90-degree spin (revolve meshes) - reported only as a cross-check of the scan PCA axis."""
    p = MP._sample_mesh_surface(v, f, 4000, seed); c = p.mean(0); tree = cKDTree(p); res = []
    for k in range(3):
        Rk = Rotation.from_rotvec(np.eye(3)[k] * np.pi / 2).as_matrix(); q = (p - c) @ Rk.T + c
        res.append(0.5 * (tree.query(q)[0].mean() + cKDTree(q).query(p)[0].mean()))
    k = int(np.argmin(res)); return np.eye(3)[k], dict(link_axis='xyz'[k], spin90_chamfer_mm=[round(x * 1000, 2) for x in res])


def register(cand, ref, seed, name, axis):
    cv, cf = cand; rv, rf = ref
    cc, cr = MP._surface_moments(cv, cf); rc, rr = MP._surface_moments(rv, rf); scale = rr / cr
    cp = (MP._sample_mesh_surface(cv, cf, 3000, seed) - cc) * scale; rp = MP._sample_mesh_surface(rv, rf, 3000, seed) - rc
    starts = align_all_starts(rp, cp); best = min(range(len(starts)), key=lambda i: starts[i][2]); Rb, tb, sb = starts[best]
    ca_off, R_off = MP._register_object_body(cand, ref, rc, seed)          # the official call, anchor = scan centroid
    assert np.allclose(R_off, Rb, atol=1e-9), 'copied _align_rigid diverged from official'
    ca_org, _ = MP._register_object_body(cand, ref, np.zeros(3), seed)    # anchor = scan-file origin
    others = [s for (Rs, _, s) in starts if sym_angle(Rs, Rb, name, axis) > DIFF_DEG]
    return dict(R=Rb, anchor_c=ca_off, anchor_o=ca_org, resid_mm=sb * 1000, start=best, scale=scale,
                alt_resid_mm=(min(others) * 1000 if others else None))


_CACHE = {}


def setup(name, paths):
    if name not in _CACHE:
        ref = scan(name); old, new = packed(paths[name][0]), packed(paths[name][1])
        sa = scan_axis(*ref, name); c, _ = MP._surface_moments(*ref); tri = ref[0][ref[1]] - c
        pca = MP._pca_basis(tri.reshape(-1, 3))   # columns: major, mid, minor (vertex-of-triangle PCA, for reporting only)
        ax = dict(old=cand_axis(*old), new=cand_axis(*new)) if name in AXIAL | HALF_TURN else None
        _CACHE[name] = dict(ref=ref, old=old, new=new, axis=sa[0] if sa else None, eig=sa[1] if sa else None, pca=pca, cax=ax)
    return _CACHE[name]


def job(task):
    name, which, seed, paths = task; S = setup(name, paths)
    r = register(S[which], S['ref'], seed, name, S['axis'])
    return (name, which, seed), r


def compare(name, S, o, n):
    axis = S['axis']; M = o['R'] @ n['R'].T; rv = Rotation.from_matrix(M).as_rotvec(); th = np.linalg.norm(rv)
    row = dict(rot_diff_raw_deg=ang(M), rot_diff_sym_deg=sym_angle(o['R'], n['R'], name, axis),
               trans_diff_mm=float(np.linalg.norm(n['anchor_c'] - o['anchor_c']) * 1000),
               trans_diff_mm_origin_anchor=float(np.linalg.norm(n['anchor_o'] - o['anchor_o']) * 1000),
               keypoint_shift_cm=keypoint_shift_cm(o['R'], o['anchor_c'], n['R'], n['anchor_c'], name, axis),
               diff_rotation_axis_abs_cos_vs_scan_pca_major_mid_minor=(np.abs(S['pca'].T @ (rv / th)).round(3).tolist() if th > 1e-9 else None),
               resid_old_mm=o['resid_mm'], resid_new_mm=n['resid_mm'],
               alt_branch_resid_old_mm=o['alt_resid_mm'], alt_branch_resid_new_mm=n['alt_resid_mm'],
               start_old=o['start'], start_new=n['start'], R_old=o['R'].round(6).tolist(), R_new=n['R'].round(6).tolist(),
               cand_anchor_old=o['anchor_c'].round(6).tolist(), cand_anchor_new=n['anchor_c'].round(6).tolist())
    if axis is not None:
        row['axis_tilt_old_vs_new_deg'] = vang(o['R'].T @ axis, n['R'].T @ axis)
        row['cand_axis_vs_scan_axis_old_deg'] = vang(o['R'] @ S['cax']['old'][0], axis)
        row['cand_axis_vs_scan_axis_new_deg'] = vang(n['R'] @ S['cax']['new'][0], axis)
    d = row['rot_diff_sym_deg']; row['same_branch'] = 'yes' if d < SAME_DEG else 'NO' if d > DIFF_DEG else 'ambiguous'
    return row


def main(names):
    from multiprocessing import Pool
    t0 = time.time(); paths, skipped = mesh_paths(); occ = occurrences(); OUT.mkdir(parents=True, exist_ok=True)
    roster = json.loads((KIT/'data/track_3_evaluation_objects.json').read_text())['episodes']
    scored = {}
    for ep, objs in roster.items():
        for slot, o in enumerate(objs): scored.setdefault(o, []).append((int(ep), slot))
    names = [n for n in (list(paths) if names == ['all'] else names) if n in paths]
    tasks = set()
    for name in names:
        for ep, slot in occ.get(name, []):
            for off in [0] + EXTRA_SEEDS: tasks |= {(name, w, MP._object_seed(ep, slot) + off) for w in ('old', 'new')}
        for ep, slot in scored.get(name, []): tasks |= {(name, w, MP._object_seed(ep, slot)) for w in ('old', 'new')}
    tasks = sorted(tasks); print(f'{len(tasks)} registrations', flush=True)
    with Pool(int(sys_env('REG_PROCS', 24))) as pool:
        res = {}
        for i, (k, r) in enumerate(pool.imap_unordered(job, [t + (paths,) for t in tasks])):
            res[k] = r
            if i % 20 == 0: print(f'{i+1}/{len(tasks)} [{time.time()-t0:.0f}s]', flush=True)
    report = dict(script=str(Path(__file__).relative_to(R)), reference='public scan (proxy for host confidential scan)',
                  thresholds_deg=dict(same=SAME_DEG, different=DIFF_DEG), extra_seed_offsets=EXTRA_SEEDS,
                  skipped=skipped, objects={})
    for name in names:
        S = setup(name, paths); axis = S['axis']
        obj = dict(old=str(paths[name][0].relative_to(R)), new=str(paths[name][1].relative_to(R)),
                   symmetry='axial' if name in AXIAL else 'half_turn' if name in HALF_TURN else 'none', public=[], scored=[])
        if axis is not None: obj['scan_axis'] = dict(axis=axis.round(5).tolist(), eigenvalues_m2=S['eig'])
        if S['cax']: obj['cand_axis_old'] = S['cax']['old'][1]; obj['cand_axis_new'] = S['cax']['new'][1]
        for ep, slot in occ.get(name, []):
            seed0 = MP._object_seed(ep, slot); o, n = res[(name, 'old', seed0)], res[(name, 'new', seed0)]
            row = dict(episode=ep, slot=slot, seed=seed0, **compare(name, S, o, n)); seeds = []
            for off in EXTRA_SEEDS:
                ok, nk = res[(name, 'old', seed0 + off)], res[(name, 'new', seed0 + off)]
                e = dict(offset=off, new_vs_new0_sym_deg=sym_angle(nk['R'], n['R'], name, axis),
                         old_vs_old0_sym_deg=sym_angle(ok['R'], o['R'], name, axis),
                         old_vs_new_sym_deg=sym_angle(ok['R'], nk['R'], name, axis),
                         new_vs_new0_raw_deg=ang(nk['R'] @ n['R'].T), old_vs_old0_raw_deg=ang(ok['R'] @ o['R'].T),
                         resid_old_mm=ok['resid_mm'], resid_new_mm=nk['resid_mm'])
                if axis is not None:
                    e['new_vs_new0_tilt_deg'] = vang(nk['R'].T @ axis, n['R'].T @ axis)
                    e['old_vs_old0_tilt_deg'] = vang(ok['R'].T @ axis, o['R'].T @ axis)
                seeds.append(e)
            row['seeds'] = seeds
            row['new_seed_max_dev_deg'] = max(s['new_vs_new0_sym_deg'] for s in seeds)
            row['old_seed_max_dev_deg'] = max(s['old_vs_old0_sym_deg'] for s in seeds)
            row['old_vs_new_all_seeds_max_deg'] = max([row['rot_diff_sym_deg']] + [s['old_vs_new_sym_deg'] for s in seeds])
            row['new_seed_stable'] = row['new_seed_max_dev_deg'] < SAME_DEG
            obj['public'].append(row)
        for ep, slot in scored.get(name, []):
            seed0 = MP._object_seed(ep, slot)
            obj['scored'].append(dict(episode=ep, slot=slot, seed=seed0,
                                      **compare(name, S, res[(name, 'old', seed0)], res[(name, 'new', seed0)])))
        allv = [r['rot_diff_sym_deg'] for r in obj['public'] + obj['scored']] + [s['old_vs_new_sym_deg'] for r in obj['public'] for s in r['seeds']]
        obj['summary'] = dict(n_registration_pairs=len(allv), old_vs_new_sym_deg_median=float(np.median(allv)),
                              old_vs_new_sym_deg_max=float(np.max(allv)), n_pairs_different_branch=int(sum(v > DIFF_DEG for v in allv)),
                              new_seed_max_dev_deg=max(r['new_seed_max_dev_deg'] for r in obj['public']) if obj['public'] else None,
                              old_seed_max_dev_deg=max(r['old_seed_max_dev_deg'] for r in obj['public']) if obj['public'] else None)
        report['objects'][name] = obj
        print(name, json.dumps(obj['summary']), flush=True)
    tag = '' if set(names) == set(paths) else '_' + '_'.join(names)
    (OUT/f'registration_branch{tag}.json').write_text(json.dumps(report, indent=1))
    write_md(report, OUT/f'registration_branch{tag}.md')


def sys_env(k, default):
    import os; return os.environ.get(k, default)


def write_md(rep, path):
    L = ['# Registration branch check: old (procedural_geometry_v1 visual_4096) vs new (multiep_recon_v1 final_4096)', '',
         'VALIDATION ONLY. Official `_register_object_body` (MPPE.py) on host-packed meshes (`budget_mesh` 4096/4096 -> payload '
         'round-trip) against the **public scan** as a proxy of the confidential host scan; anchor = scan surface centroid; '
         'seed = 1,000,003*ep + 101*slot (`_object_seed`, identical to the host call). sym = metric-equivalent orientation change '
         '(cups: tilt of the declared axis; lid: min over 0/180 deg; others incl. white_pot: full geodesic). kp = MPPE-style '
         '6-keypoint shift (cm, official `_symmetry_point_errors`) between old/new converted poses for the same predicted pose, '
         'before the initial-scene transform. res = registration symmetric mean distance (mm, candidate scaled to scan); '
         f'alt = best residual among the 25 ICP starts on a branch >{DIFF_DEG:.0f} deg (sym) from the chosen one. '
         'Seeds: 5 extra seeds per public occurrence (official + 7919k).', '',
         '## Public occurrences', '',
         '| object | ep/slot | sym deg | raw deg | dt mm | kp cm | res old/new mm | alt old/new mm | same branch | seed max dev old/new deg | old-vs-new max over 6 seeds deg |',
         '|---|---|---:|---:|---:|---:|---|---|---|---|---:|']
    f = lambda x: '-' if x is None else f'{x:.2f}'
    for name, o in rep['objects'].items():
        for r in o['public']:
            L.append(f"| {name} | {r['episode']}/{r['slot']} | {r['rot_diff_sym_deg']:.2f} | {r['rot_diff_raw_deg']:.2f} | "
                     f"{r['trans_diff_mm']:.1f} | {r['keypoint_shift_cm']:.2f} | {r['resid_old_mm']:.2f}/{r['resid_new_mm']:.2f} | "
                     f"{f(r['alt_branch_resid_old_mm'])}/{f(r['alt_branch_resid_new_mm'])} | {r['same_branch']} | "
                     f"{r['old_seed_max_dev_deg']:.2f}/{r['new_seed_max_dev_deg']:.2f} | {r['old_vs_new_all_seeds_max_deg']:.2f} |")
    L += ['', '## Scored-episode seeds (roster from toolkit data/track_3_evaluation_objects.json), still against the public scan', '',
          '| object | ep/slot | sym deg | raw deg | dt mm | kp cm | res old/new mm | alt old/new mm | same branch |', '|---|---|---:|---:|---:|---:|---|---|---|']
    for name, o in rep['objects'].items():
        for r in o['scored']:
            L.append(f"| {name} | {r['episode']}/{r['slot']} | {r['rot_diff_sym_deg']:.2f} | {r['rot_diff_raw_deg']:.2f} | "
                     f"{r['trans_diff_mm']:.1f} | {r['keypoint_shift_cm']:.2f} | {r['resid_old_mm']:.2f}/{r['resid_new_mm']:.2f} | "
                     f"{f(r['alt_branch_resid_old_mm'])}/{f(r['alt_branch_resid_new_mm'])} | {r['same_branch']} |")
    L += ['', '## Per-object summary (all old/new pairs: public x 6 seeds + scored seeds)', '',
          '| object | pairs | median sym deg | max sym deg | pairs on different branch (>30 deg) | seed max dev old/new deg |', '|---|---:|---:|---:|---:|---|']
    for name, o in rep['objects'].items():
        s = o['summary']
        L.append(f"| {name} | {s['n_registration_pairs']} | {s['old_vs_new_sym_deg_median']:.2f} | {s['old_vs_new_sym_deg_max']:.2f} | "
                 f"{s['n_pairs_different_branch']} | {f(s['old_seed_max_dev_deg'])}/{f(s['new_seed_max_dev_deg'])} |")
    L += ['', 'Skipped: ' + '; '.join(f'{k}: {v}' for k, v in rep['skipped'].items())]
    path.write_text('\n'.join(L) + '\n')


def sweep(names, n):
    """Branch-rate sweep: n extra seeds (official scored seed + 7919k) per scored occurrence; branch measured against
    the OLD mesh registered at the official scored seed."""
    from multiprocessing import Pool
    paths, _ = mesh_paths(); roster = json.loads((KIT/'data/track_3_evaluation_objects.json').read_text())['episodes']
    tasks = []
    for name in names:
        for ep, objs in roster.items():
            for slot, o in enumerate(objs):
                if o == name: tasks += [(name, w, MP._object_seed(int(ep), slot) + 7919 * k, paths) for k in range(n + 1) for w in ('old', 'new')]
    with Pool(int(sys_env('REG_PROCS', 24))) as pool: res = dict(pool.map(job, tasks))
    out = {}
    for name in names:
        S = setup(name, paths); axis = S['axis']; base = {}
        for ep, objs in roster.items():
            for slot, o in enumerate(objs):
                if o != name: continue
                s0 = MP._object_seed(int(ep), slot); R0 = res[(name, 'old', s0)]['R']; rows = []
                for k in range(n + 1):
                    ro, rn = res[(name, 'old', s0 + 7919 * k)], res[(name, 'new', s0 + 7919 * k)]
                    rows.append(dict(k=k, old_vs_old_official_deg=sym_angle(ro['R'], R0, name, axis),
                                     new_vs_old_official_deg=sym_angle(rn['R'], R0, name, axis),
                                     old_vs_new_deg=sym_angle(ro['R'], rn['R'], name, axis),
                                     resid_old_mm=ro['resid_mm'], resid_new_mm=rn['resid_mm']))
                frac = lambda key: float(np.mean([r[key] > DIFF_DEG for r in rows]))
                base[f'{ep}/{slot}'] = dict(n_seeds=n + 1, old_off_branch_rate=frac('old_vs_old_official_deg'),
                                            new_off_branch_rate=frac('new_vs_old_official_deg'), old_new_disagree_rate=frac('old_vs_new_deg'),
                                            old_dev_deg_quantiles=np.percentile([r['old_vs_old_official_deg'] for r in rows], [50, 90, 100]).round(2).tolist(),
                                            new_dev_deg_quantiles=np.percentile([r['new_vs_old_official_deg'] for r in rows], [50, 90, 100]).round(2).tolist(),
                                            rows=rows)
                print(name, ep, slot, {k: v for k, v in base[f'{ep}/{slot}'].items() if k != 'rows'}, flush=True)
        out[name] = base
    (OUT/'registration_branch_sweep.json').write_text(json.dumps(dict(seed_rule='official scored seed + 7919k, k=0..n',
        branch_threshold_deg=DIFF_DEG, reference='old mesh at official scored seed', objects=out), indent=1))
    sweep_md()


def sweep_md():
    d = json.loads((OUT/'registration_branch_sweep.json').read_text())
    L = ['# Seed sweep at scored-episode seeds (public scan as reference)', '',
         f"Seeds: {d['seed_rule']} (k=0 is the host's seed). Branch = sym angle > {d['branch_threshold_deg']:.0f} deg from "
         'the OLD mesh registered at the host seed. dev quantiles = p50/p90/max of that angle.', '',
         '| object | ep/slot | seeds | old off-branch | new off-branch | old/new disagree | old dev p50/p90/max | new dev p50/p90/max | k=0 old-vs-new deg |',
         '|---|---|---:|---:|---:|---:|---|---|---:|']
    for name, occ in d['objects'].items():
        for key, r in occ.items():
            L.append(f"| {name} | {key} | {r['n_seeds']} | {r['old_off_branch_rate']:.2f} | {r['new_off_branch_rate']:.2f} | "
                     f"{r['old_new_disagree_rate']:.2f} | {'/'.join(map(str, r['old_dev_deg_quantiles']))} | "
                     f"{'/'.join(map(str, r['new_dev_deg_quantiles']))} | {r['rows'][0]['old_vs_new_deg']:.2f} |")
    (OUT/'registration_branch_sweep.md').write_text('\n'.join(L) + '\n')


if __name__ == '__main__':
    if sys.argv[1:2] == ['--sweep']: sweep(sys.argv[3:], int(sys.argv[2]))
    elif sys.argv[1:2] == ['--sweep-md']: sweep_md()
    else: main(sys.argv[1:])
