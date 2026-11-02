#!/bin/bash
# Local package of the v5 candidate (NO upload): results/ + meshes/ from outputs/track3/submission_v5_candidate/assembly.json,
# then the official packer + preflight. COMMIT_URL must be passed; the default is the #4 commit as a PLACEHOLDER
# (the v8 code is not pushed yet) -> re-pack with the real commit before any upload.
# Usage: build_track3_v5_candidate_package.sh [COMMIT_URL]
set -e
cd /home/tianyuan/projects/v2d
D=outputs/track3/submission_v5_candidate; COMMIT=${1:-https://github.com/tianny0078/v2d-track3/commit/e8d415c}
PY=/home/tianyuan/projects/v2d/.deps/v2d-submit-venv/bin/python
python3 - <<'EOF'
import json, shutil
from pathlib import Path
D = Path('outputs/track3/submission_v5_candidate'); a = json.loads((D/'assembly.json').read_text())
assert len(a) == 20, f'assembly has {len(a)} episodes'
for sub in ('results', 'meshes'):
    p = D/sub
    if p.exists(): p.rename(D/f'{sub}.old')   # keep the previous build aside (no rm -rf)
(D/'results').mkdir(); (D/'meshes').mkdir()
for E, x in sorted(a.items()):
    shutil.copyfile(x['rollout'], D/'results'/f'{E}.parquet')
    (D/'meshes'/E).mkdir()
    for n, m in x['meshes'].items(): shutil.copyfile(m['src'], D/'meshes'/E/f'{n}.obj')
print('assembled', len(a))
EOF
for sub in results meshes; do [ -d $D/$sub.old ] && mv $D/$sub.old $D/.$sub.prev.$(date -u +%Y%m%dT%H%M%S); done
mkdir -p $D/submission
cd toolkit/8d2bce64/v2d_submission_kit
$PY tools/pack_submission.py --track track_3 --results /home/tianyuan/projects/v2d/$D/results --meshes /home/tianyuan/projects/v2d/$D/meshes \
  --code-commit "$COMMIT" -o /home/tianyuan/projects/v2d/$D/submission/track3_v5_candidate.parquet 2>&1 | tee /home/tianyuan/projects/v2d/$D/submission/pack.log
$PY tools/preflight.py --track track_3 --results /home/tianyuan/projects/v2d/$D/results 2>&1 | tee /home/tianyuan/projects/v2d/$D/submission/preflight.log
echo "commit_url_used=$COMMIT (placeholder unless explicitly passed)" >> /home/tianyuan/projects/v2d/$D/submission/pack.log
