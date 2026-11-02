"""Per-object driver: keyframes -> param-only variants (cad path) -> pick by scored held-out IoU -> final_4096.obj + STATUS line.

Variants (same keyframes / init for both):
  A = --col-geom 1 --w-gauge 1   (blue_cup P1 settings)
  B = --col-geom 0 --w-gauge 0   (blue_cup P3 settings)
Layout: outputs/track3/multiep_recon_v1/<obj>/variants/<A|B>/<obj>/...  ; chosen files copied to outputs/track3/multiep_recon_v1/<obj>/
Usage: run_object.py <obj> [--gpu 0] [--extra "<args for recon.py>"]
"""
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/track3/multiep_recon_v1'
LOGD = ROOT/'logs/multiep_recon'
VARS = {'A': ['--col-geom', '1', '--w-gauge', '1'], 'B': ['--col-geom', '0', '--w-gauge', '0']}


def run(cmd, log):
    print(time.strftime('%H:%M:%S'), ' '.join(cmd), flush=True)
    with open(log, 'a') as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode: raise SystemExit(f'failed ({r.returncode}): see {log}')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('obj'); ap.add_argument('--gpu', default='0'); ap.add_argument('--extra', default='')
    ap.add_argument('--init-poses', default=None, help='passed to recon.py --init-poses (npz format path with {ep} {obj}, e.g. retrack_v8); '
                    'if the refit is not adopted, final_4096.obj = procedural visual_4096 x the init mesh_scale')
    a = ap.parse_args(); obj = a.obj
    od = OUT/obj; od.mkdir(parents=True, exist_ok=True)
    extra = a.extra.split() if a.extra else []
    if a.init_poses: extra += ['--init-poses', a.init_poses]
    base = [sys.executable, str(Path(__file__).parent/'recon.py'), '--object', obj, '--gpu', a.gpu, '--cad', '1', '--edge-mm', '0'] + extra
    vA = od/'variants/A'; kj = vA/obj/'keyframes.json'
    notes = []
    if not kj.exists():
        run(base + VARS['A'] + ['--out-root', str(vA), '--stop-after', 'keyframes'], LOGD/f'{obj}-A.log')
    sel = json.loads(kj.read_text())['selection']; nk = sum(len(v['keyframes']) for v in sel.values())
    if nk < 12 and json.loads(kj.read_text())['selector']['min_vis'] > 0.7:
        notes.append(f'only {nk} keyframes at min_vis 0.80 -> relaxed to 0.70')
        for f in ['keyframes.json', 'keyframes_contact_sheet.jpg']:
            if (vA/obj/f).exists(): (vA/obj/f).rename(vA/obj/f.replace('keyframes', 'keyframes.minvis080'))
        extra += ['--min-vis', '0.7']; base += ['--min-vis', '0.7']
        run(base + VARS['A'] + ['--out-root', str(vA), '--stop-after', 'keyframes'], LOGD/f'{obj}-A.log')
        sel = json.loads(kj.read_text())['selection']; nk = sum(len(v['keyframes']) for v in sel.values())
    res = {}
    for v, flags in VARS.items():
        vd = od/'variants'/v
        if v != 'A':
            (vd/obj).mkdir(parents=True, exist_ok=True)
            for f in ['init.npz', 'keyframes.json', 'keyframes_contact_sheet.jpg']:
                if not (vd/obj/f).exists() and (vA/obj/f).exists(): shutil.copyfile(vA/obj/f, vd/obj/f)
        run(base + flags + ['--out-root', str(vd)], LOGD/f'{obj}-{v}.log')
        r = json.loads((vd/obj/'report.json').read_text())
        ms = r['metrics_summary']
        res[v] = dict(iou_refit_after=ms['hold_after_refit']['all_episodes']['iou']['median'],
                      iou_refit_before=ms['hold_before_refit']['all_episodes']['iou']['median'],
                      iou_init_after=ms['hold_after_initpose']['all_episodes']['iou']['median'],
                      iou_init_before=ms['hold_before_initpose']['all_episodes']['iou']['median'],
                      dep_after=ms['hold_after_refit']['all_episodes']['depth_mae_mm']['median'],
                      dep_before=ms['hold_before_refit']['all_episodes']['depth_mae_mm']['median'],
                      col_after=ms['hold_after_refit']['all_episodes']['color_mae']['median'],
                      col_before=ms['hold_before_refit']['all_episodes']['color_mae']['median'],
                      mesh=r['mesh_checks'].get('final_4096.obj'), deform=r['scale']['param_stage'])
        print(v, json.dumps(res[v]), flush=True)
    best = max(res, key=lambda k: res[k]['iou_refit_after'])
    bd = od/'variants'/best/obj
    kept = res[best]['iou_refit_after'] < res[best]['iou_refit_before']
    if kept: notes.append(f'no variant beats the submitted mesh on held-out IoU ({res[best]["iou_refit_after"]:.3f} < {res[best]["iou_refit_before"]:.3f})')
    for f in ['final_4096.obj', 'mesh_full.obj', 'mesh_full.ply', 'report.json', 'turntable_before_after.png', 'keyframes_contact_sheet.jpg',
              'keyframes.json'] + [p.name for p in bd.glob('overlay_ep*.jpg')] + [p.name for p in bd.glob('poses_ep*.npz')]:
        if (bd/f).exists(): shutil.copyfile(bd/f, od/f)
    pub = 'proc-mesh' in ' '.join(extra) and not a.init_poses   # scored-lineage alternative proc mesh (retrack_v8) keeps the adopt rule
    if kept and not pub:
        # keep the submitted procedural <=4096 mesh as final (the refit is not adopted); variant meshes stay under variants/
        shutil.copyfile(od/'final_4096.obj', od/f'final_4096.variant{best}.obj')
        if a.init_poses:
            import numpy as np
            sc = sorted({float(np.load(Path(a.init_poses.format(ep=int(e), obj=obj)))['mesh_scale']) for e in sel})
            src = Path(a.init_poses.format(ep=int(list(sel)[0]), obj=obj)).parent/f'{obj}_visual_4096_x{sc[0]:.4f}.obj'
            shutil.copyfile(src, od/'final_4096.obj')
            notes.append(f'final_4096.obj = procedural visual_4096 x init scale {sc} ({src.name}); best variant kept as final_4096.variant*.obj')
        else:
            shutil.copyfile(ROOT/f'outputs/track3/procedural_geometry_v1/{obj}/visual_4096.obj', od/'final_4096.obj')
            notes.append('final_4096.obj = submitted procedural visual_4096 (unchanged); best variant kept as final_4096.variant*.obj')
    elif kept: notes.append('scored-only re-derivation required (public lineage): final_4096 is the scored-derived variant despite lower IoU')
    (od/'variant_choice.json').write_text(json.dumps(dict(chosen=best, adopted=not (kept and not pub), rule='max scored held-out IoU median (pose re-fit protocol)', variants=res,
                                                         flags=VARS, notes=notes, keyframes=nk, init_poses=a.init_poses or 'learning_v7_geom'), indent=1))
    b = res[best]; m = b['mesh'] or {}
    line = (f'- {obj} | keyframes {nk}{" (min_vis 0.70)" if notes else ""} | held-out IoU {b["iou_refit_before"]:.3f}->{b["iou_refit_after"]:.3f} '
            f'(init-pose {b["iou_init_before"]:.3f}->{b["iou_init_after"]:.3f}), depth {b["dep_before"]:.1f}->{b["dep_after"]:.1f} mm | '
            f'final_4096 {m.get("faces")}F watertight={m.get("watertight")} | variant {best}{" NOT ADOPTED" if kept and not pub else ""} '
            f'(A {res["A"]["iou_refit_after"]:.3f} / B {res["B"]["iou_refit_after"]:.3f}); {b["deform"].get("kind")} deformation; {"; ".join(notes)}'
            f'{"; init poses " + a.init_poses + " (before = procedural x retrack scale)" if a.init_poses else ""}\n')
    with open(OUT/'STATUS.md', 'a') as f: f.write(line)
    print(line, flush=True)


if __name__ == '__main__':
    main()
