#!/bin/bash
# Sequential learning v3 on one Track3 GPU (1 or 2). Resumable: the runner skips completed milestones.
# After an episode completes, recovery state/replay files are deleted on NAS (policies kept).
cd /home/tianyuan/projects/v2d; GPU=$1; shift
TREE=${TRACK3_LEARNING_TREE:-learning_v3}; WORLDS=${WORLDS:-2048}; export TRACK3_LEARNING_TREE=$TREE; mkdir -p logs/$TREE
for e in "$@"; do
  ep=$(printf episode_%06d $e); L=outputs/track3/$TREE/$ep/learning
  for attempt in 1 2 3 4; do  # retry on GPU out-of-memory (shared GPUs); the runner resumes completed milestones
    .deps/flash-chord-venv/bin/python -u scripts/run_track3_learning_v3.py --episode $e --gpu $GPU --worlds $WORLDS --milestone-steps ${MILESTONE_STEPS:-1000000} --milestones ${MILESTONES:-5} >> logs/$TREE/train_ep$e.log 2>&1
    rc=$?; echo "$e $rc attempt$attempt $(date -u +%FT%TZ)" >> logs/$TREE/train-gpu$GPU.exits
    [ $rc = 0 ] && break
    tail -200 logs/$TREE/train_ep$e.log $L/train_*.log 2>/dev/null | grep -qi "out of memory" || break
    sleep 600
  done
  if [ $rc = 0 ] && grep -q '"completed_step_cap"' $L/report.json; then
    rm -f /mnt/nas3/tianyuan/v2d-track3/$TREE/$ep/training/checkpoints/{state,replay}_*.safetensors
  fi
done
echo queue-done >> logs/$TREE/train-gpu$GPU.exits
