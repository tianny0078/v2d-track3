#!/bin/bash
# usage: queue.sh <gpu> <wait_tmux_session> obj1 obj2 ...   (runs run_object.py sequentially after the given session ends)
cd /home/tianyuan/projects/v2d
GPU=$1; WAIT=$2; shift 2
while tmux has-session -t "$WAIT" 2>/dev/null; do sleep 20; done
for o in "$@"; do
  if [ -f outputs/track3/multiep_recon_v1/$o/variant_choice.json ]; then echo "skip $o (done)"; continue; fi
  python3 scripts/multiep_recon/run_object.py $o --gpu $GPU > logs/multiep_recon/driver-$o.log 2>&1 || echo "$o FAILED" >> logs/multiep_recon/queue-failures.log
done
