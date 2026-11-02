#!/bin/bash
# Prepare-only (CPU: retarget IK + reference + config) for learning_v8r episodes, lv7r recipe env. Usage: prepare_track3_v8r_all.sh EP...
cd /home/tianyuan/projects/v2d
export TRACK3_LEARNING_TREE=learning_v8r TRACK3_HANDS_ROOT=learning_v5/hands TRACK3_CONTACT_FRICTION=1.0 TRACK3_VOC_CURRICULUM=1
for e in "$@"; do
  ( .deps/flash-chord-venv/bin/python -u scripts/run_track3_learning_v3.py --episode $e --gpu 1 --worlds 1024 --milestone-steps 2000000 --milestones 5 --prepare-only >> logs/learning_v8r/prepare_ep$e.log 2>&1; echo "$e rc=$? $(date -u +%FT%TZ)" >> logs/learning_v8r/prepare.exits ) &
done; wait
