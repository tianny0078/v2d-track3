#!/bin/bash
# Every 30 min: rerun the v8 selection (re-evals + lv8r evaluations); re-pack locally (no upload) only when assembly.json changed.
cd /home/tianyuan/projects/v2d; D=outputs/track3/submission_v5_candidate; L=logs/learning_v8/selection-loop.log
while :; do
  old=$(sha256sum $D/assembly.json 2>/dev/null | cut -c1-16)
  OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 timeout 5400 python3 scripts/select_track3_submission_candidates_v8.py > logs/learning_v8/selection-latest.md 2>&1 || echo "$(date -u +%FT%TZ) selection failed" >> $L
  new=$(sha256sum $D/assembly.json | cut -c1-16)
  echo "$(date -u +%FT%TZ) selection $(tail -2 $D/selection.md | head -1)" >> $L
  if [ "$old" != "$new" ]; then bash scripts/build_track3_v5_candidate_package.sh > logs/learning_v8/package-latest.log 2>&1 && echo "$(date -u +%FT%TZ) repacked (assembly $old -> $new)" >> $L || echo "$(date -u +%FT%TZ) repack FAILED" >> $L; fi
  sleep 2700
done
