"""Re-evaluate existing policy checkpoints inside the learning_v8_geom scenes (new meshes), same evaluator flow as the
training runner (scripts/run_track3_learning_v3.py): the checkpoint's own frozen producer-source evaluator
(evaluate_track3_canonical_checkpoint.py, sha bb8532bc for every tree) with --substeps 25 --record-contacts, 1 world,
VOC 0; only --object-manifest changes (learning_v8_geom/episode_X/objects/objects.json -> object mesh/URDF/pose columns).
Robot targets, friction, etc. stay those of the checkpoint's own training config.

Candidates: outputs/track3/learning_v8_reeval/candidates.json  [{"tree":..., "episode":..., "step":...}, ...]
Output:     outputs/track3/learning_v8_reeval/<tree>/episode_X/step_N/ (replay-report.json + episode_X.parquet)
State:      state/learning_v8_reeval/<tree>_ep<X>_<N>.done | .failed  (resumable: done ones are skipped)
Usage: python3 scripts/run_track3_v8_reeval.py --gpu 1 --jobs 3 [--episodes 1,3,...]
"""
import argparse, json, os, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

R = Path('/home/tianyuan/projects/v2d'); T = R/'outputs/track3'; NAS = Path('/mnt/nas3/tianyuan/v2d-track3')
OUT = T/'learning_v8_reeval'; ST = R/'state/learning_v8_reeval'; LOG = R/'logs/learning_v8/reeval'
PY = R/'.deps/flash-chord-venv/bin/python'


def run_one(c, gpu, geom_tree):
    t, e, s = c['tree'], int(c['episode']), int(c['step']); E = f'episode_{e:06d}'; tag = f'{t}_ep{e}_{s}'
    done, failed = ST/f'{tag}.done', ST/f'{tag}.failed'
    out = OUT/t/E/f'step_{s}'
    if done.exists(): return tag, 'skip'
    ck = NAS/t/E/'training/checkpoints'/f'policy_{s}.safetensors'; snap = ck.parent.parent/'producer-source'
    man = T/geom_tree/E/'objects/objects.json'
    if not ck.exists() or not man.exists(): failed.write_text(f'missing {ck if not ck.exists() else man}\n'); return tag, 'missing'
    if (out/'reference/reference-report.json').exists() and not (out/'replay-report.json').exists():
        # partial earlier attempt: the evaluator refuses a changed reference but a same-input retry is allowed
        pass
    env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=str(gpu), XLA_PYTHON_CLIENT_PREALLOCATE='false', WANDB_MODE='disabled',
                                        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONPATH=str(snap),
                                        JAX_COMPILATION_CACHE_DIR=str(OUT/'jax_cache'/t))
    argv = [str(PY), '-u', str(snap/'evaluate_track3_canonical_checkpoint.py'), '--checkpoint', str(ck), '--object-manifest', str(man),
            '--episode', str(e), '--output', str(out), '--substeps', '25', '--record-contacts']
    out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    with open(LOG/f'{tag}.log', 'a') as log:
        log.write(f'# {time.strftime("%FT%TZ", time.gmtime())} gpu{gpu} {" ".join(argv)}\n'); log.flush()
        try: rc = subprocess.run(argv, cwd=R, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800).returncode
        except subprocess.TimeoutExpired: rc = 'timeout'
    ok = rc == 0 and (out/'replay-report.json').exists() and (out/f'{E}.parquet').exists()
    (done if ok else failed).write_text(json.dumps(dict(rc=rc, seconds=round(time.time()-t0), out=str(out), manifest=str(man), checkpoint=str(ck))) + '\n')
    if ok and failed.exists(): failed.unlink()
    return tag, f'rc={rc} {time.time()-t0:.0f}s'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--gpu', required=True); ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--episodes', default=''); ap.add_argument('--geom-tree', default='learning_v8_geom')
    a = ap.parse_args(); ST.mkdir(parents=True, exist_ok=True); LOG.mkdir(parents=True, exist_ok=True)
    cands = json.loads((OUT/'candidates.json').read_text())
    if a.episodes: cands = [c for c in cands if int(c['episode']) in {int(x) for x in a.episodes.split(',')}]
    cands = [c for c in cands if (T/a.geom_tree/f'episode_{int(c["episode"]):06d}/objects/poses.npz').exists()]
    print(f'{len(cands)} candidates with a ready v8 manifest', flush=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        for tag, msg in ex.map(lambda c: run_one(c, a.gpu, a.geom_tree), cands):
            print(time.strftime('%FT%TZ', time.gmtime()), tag, msg, flush=True)


if __name__ == '__main__':
    main()
