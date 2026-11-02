#!/bin/bash
# retrack_v8 pipeline: global init (sweeper ep19+30 jointly, pitcher ep20) -> tracking -> visual checks. Resumable per output file.
cd /home/tianyuan/projects/v2d
TAG=${TAG:-}
L=logs/retrack_v8
[ -f outputs/track3/retrack_v8/episode_000020/water_pitcher_global_init.npz ] || python3 scripts/retrack_v8/global_init.py --obj water_pitcher --episodes 20 > $L/global_init_water_pitcher.out 2>&1 &
[ -f outputs/track3/retrack_v8/episode_000030/mini_sweeper_global_init.npz ] || python3 scripts/retrack_v8/global_init.py --obj mini_sweeper --episodes 19,30 > $L/global_init_mini_sweeper.out 2>&1 &
wait
for j in "water_pitcher 20" "mini_sweeper 19" "mini_sweeper 30"; do
  set -- $j
  [ -f outputs/track3/retrack_v8/episode_0000$2/$1_poses$TAG.npz ] || (python3 scripts/retrack_v8/track.py --obj $1 --ep $2 > $L/track_$1_ep$2.out 2>&1 && python3 scripts/retrack_v8/viz_track.py $1 $2 > $L/viz_$1_ep$2.out 2>&1) &
done
wait
for j in "water_pitcher 20" "mini_sweeper 19" "mini_sweeper 30"; do set -- $j; python3 scripts/retrack_v8/viz_init.py $1 $2 >> $L/viz_init.out 2>&1; done
echo ALLDONE
