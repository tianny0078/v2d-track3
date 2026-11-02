#!/bin/bash
# v8 variant of run_track3_shared_queue.sh (original untouched). Same claim rule: the GPU must have >= NEED_MIB free and
# none of "our" processes on it for 3 consecutive checks; then claim the next unclaimed episode (atomic mkdir in
# state/<tree>/claims) and train it with run_track3_learning_v3_queue.sh (resumable, OOM retry).
# Only difference: OURS_MATCH (optional regex) narrows "our processes" to those whose command line matches it
# (e.g. the training tree name), so short-lived evaluation jobs of other experiments do not block the claim.
# OURS_MATCH unset = original behaviour (any process of $USER blocks). Usage: run_track3_shared_queue_v8.sh GPU EP...
cd /home/tianyuan/projects/v2d; GPU=$1; shift; TREE=${TRACK3_LEARNING_TREE:?}; C=state/$TREE/claims; mkdir -p $C logs/$TREE
NEED_MIB=${NEED_MIB:-60000}
free() { nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits; }
ours() {
  local pids; pids=$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader | xargs -r ps -o user=,pid= -p 2>/dev/null | awk -v u=$USER '$1==u{print $2}')
  if [ -z "$OURS_MATCH" ]; then echo "$pids" | grep -c . ; return; fi
  local n=0; for p in $pids; do tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -Eq "$OURS_MATCH" && n=$((n+1)); done; echo $n
}
while :; do
  ok=0; until [ $ok -ge 3 ]; do  # 3 consecutive checks so a milestone gap (train -> evaluate) is not mistaken for idle
    if [ "$(ours)" = 0 ] && [ "$(free)" -ge $NEED_MIB ]; then ok=$((ok+1)); sleep 60; else ok=0; sleep 300; fi; done
  e=; for x in "$@"; do mkdir $C/ep$x 2>/dev/null && { e=$x; break; }; done
  [ -z "$e" ] && break
  echo "$e start gpu$GPU $(date -u +%FT%TZ)" >> logs/$TREE/shared-queue.log
  bash scripts/run_track3_learning_v3_queue.sh $GPU $e
  echo "$e end gpu$GPU $(date -u +%FT%TZ)" >> logs/$TREE/shared-queue.log
done
