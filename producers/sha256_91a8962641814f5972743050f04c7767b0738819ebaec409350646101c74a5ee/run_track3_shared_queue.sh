#!/bin/bash
# Shared episode queue for one learning tree across GPUs: each worker (one per GPU) waits until the GPU has room and
# none of our training runs on it, then claims the next unclaimed episode (atomic mkdir in state/<tree>/claims) and
# trains it with run_track3_learning_v3_queue.sh (resumable, OOM retry). Usage: run_track3_shared_queue.sh GPU EP...
cd /home/tianyuan/projects/v2d; GPU=$1; shift; TREE=${TRACK3_LEARNING_TREE:?}; C=state/$TREE/claims; mkdir -p $C logs/$TREE
NEED_MIB=${NEED_MIB:-60000}
free() { nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits; }
ours() { nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | xargs -r ps -o user= -p 2>/dev/null | grep -c "^$USER"; }
while :; do
  ok=0; until [ $ok -ge 3 ]; do  # 3 consecutive checks so a milestone gap (train -> evaluate) is not mistaken for idle
    if [ "$(ours)" = 0 ] && [ "$(free)" -ge $NEED_MIB ]; then ok=$((ok+1)); sleep 60; else ok=0; sleep 300; fi; done
  e=; for x in "$@"; do mkdir $C/ep$x 2>/dev/null && { e=$x; break; }; done
  [ -z "$e" ] && break
  echo "$e start gpu$GPU $(date -u +%FT%TZ)" >> logs/$TREE/shared-queue.log
  bash scripts/run_track3_learning_v3_queue.sh $GPU $e
  echo "$e end gpu$GPU $(date -u +%FT%TZ)" >> logs/$TREE/shared-queue.log
done
