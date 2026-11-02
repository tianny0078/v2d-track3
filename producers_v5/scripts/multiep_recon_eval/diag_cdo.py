"""VALIDATION ONLY: where does CD-O come from? Re-runs the official _object_chamfer_cm steps (radius normalisation, sampling,
_align_rigid) for current vs new mesh against the public scan and splits the two directed terms
(scan->ours = missing/misplaced surface, ours->scan = extra surface) by region of the scan/candidate, plus an error-coloured
point render. Writes outputs/track3/multiep_recon_v1/eval/diag_<obj>.{json,png}.
Usage: python3 scripts/multiep_recon_eval/diag_cdo.py blue_cup [episode slot] [--final  (current vs final_4096.obj)]
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

H = Path(__file__).parent
s = importlib.util.spec_from_file_location('e', H/'eval_public_scans.py'); E = importlib.util.module_from_spec(s); s.loader.exec_module(E)
MP = E.MP


def aligned(ref, cand, seed):
    rv, rf = ref; cv, cf = cand
    _, rr = MP._surface_moments(rv, rf); cc, cr = MP._surface_moments(cv, cf)
    cv = (cv - cc)/(cr/rr) + cc
    rm = MP._sample_mesh_surface(rv, rf, MP._METRIC_SAMPLES, seed); cm = MP._sample_mesh_surface(cv, cf, MP._METRIC_SAMPLES, seed)
    R, t = MP._align_rigid(MP._sample_mesh_surface(rv, rf, MP._ALIGNMENT_SAMPLES, seed + 1), MP._sample_mesh_surface(cv, cf, MP._ALIGNMENT_SAMPLES, seed + 1))
    cm = cm @ R.T + t
    d_rc = cKDTree(cm).query(rm)[0]; d_cr = cKDTree(rm).query(cm)[0]
    return rm, cm, d_rc, d_cr, cr/rr


def main(name, ep=11, slot=0):
    ref = E.scan(name); seed = MP._object_seed(ep, slot)
    arms = {'current': E.CUR/name/'visual_4096.obj'}
    if '--final' in sys.argv: arms['final'] = E.NEW/name/'final_4096.obj'
    else:
        arms['new'] = E.NEW/name/'mesh_4096.obj'
        for extra in sorted((E.NEW/name).glob('arm_*.obj')): arms[extra.stem] = extra
    # scan frame: PCA axis 0 = longest; height bins along it
    rv = ref[0]; c0 = rv.mean(0); B = MP._pca_basis(rv)
    rep = {}; import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(arms), 2, figsize=(9, 4.2*len(arms)), dpi=90, squeeze=False)
    for i, (k, p) in enumerate(arms.items()):
        rm, cm, d_rc, d_cr, ratio = aligned(ref, E.budget_mesh(Path(p), 4096, 4096), seed)
        h_r = (rm - c0) @ B[:, 0]; h_c = (cm - c0) @ B[:, 0]; lo, hi = h_r.min(), h_r.max(); bins = np.linspace(lo, hi, 6)
        def prof(h, d): return [round(float(d[(h >= a) & (h < b + 1e-9)].mean()*100), 3) if ((h >= a) & (h < b + 1e-9)).any() else None for a, b in zip(bins[:-1], bins[1:])]
        rep[k] = dict(cdo=round(50*(d_rc.mean() + d_cr.mean()), 4), scan_to_ours_cm=round(float(d_rc.mean()*100), 4), ours_to_scan_cm=round(float(d_cr.mean()*100), 4),
                      p90_scan_to_ours_cm=round(float(np.percentile(d_rc, 90)*100), 3), p90_ours_to_scan_cm=round(float(np.percentile(d_cr, 90)*100), 3),
                      frac_scan_pts_gt3mm=round(float((d_rc > .003).mean()), 3), frac_our_pts_gt3mm=round(float((d_cr > .003).mean()), 3),
                      height_profile_scan_to_ours_cm=prof(h_r, d_rc), height_profile_ours_to_scan_cm=prof(h_c, d_cr), radius_ratio=round(float(ratio), 4))
        for j, (pts, d, ttl) in enumerate([(rm, d_rc, 'scan pts: dist to ours'), (cm, d_cr, 'our pts: dist to scan')]):
            q = (pts - c0) @ B; a = ax[i, j]; o = np.argsort(d)
            sc = a.scatter(q[o, 1], q[o, 0], c=np.clip(d[o]*1000, 0, 8), s=1.2, cmap='inferno'); a.set_aspect('equal'); a.set_title(f'{k}: {ttl} (mm, side view)', fontsize=8)
        print(k, json.dumps(rep[k]), flush=True)
    fig.colorbar(sc, ax=ax.ravel().tolist(), shrink=.5); fig.savefig(E.NEW/f"eval/diag_{name}{'_final' if '--final' in sys.argv else ''}.png"); (E.NEW/f"eval/diag_{name}{'_final' if '--final' in sys.argv else ''}.json").write_text(json.dumps(rep, indent=1))


if __name__ == '__main__': main(sys.argv[1], *map(int, [a for a in sys.argv[2:] if a != '--final']))
