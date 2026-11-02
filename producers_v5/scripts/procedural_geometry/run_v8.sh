#!/bin/bash
# learning_v8_geom (copy of run_v7.sh flow): manifests (procedural_geometry_v8 assets) -> support fix (fix_support_v8 = copy of v7) -> rest test -> penetration diag
# Usage: run_v8.sh EP... (episodes 19/20/30 only after outputs/track3/retrack_v8/READY.json)
set -e; cd /home/tianyuan/projects/v2d; L=logs/procedural_geometry_v8; mkdir -p $L state/learning_v8_geom
.deps/geom-venv/bin/python scripts/procedural_geometry/make_v8_manifests.py --tree learning_v8_geom "$@" >> $L/make_v8.log 2>&1
.deps/flash-chord-venv/bin/python scripts/procedural_geometry/fix_support_v8.py --tree learning_v8_geom --lift-max 0.06 "$@" >> $L/fix_support_v8.log 2>&1
.deps/flash-chord-venv/bin/python scripts/rest_test_track3_reference.py learning_v8_geom "$@" >> $L/rest_v8.log 2>&1
.deps/flash-chord-venv/bin/python scripts/procedural_geometry/diag_penetration_v7.py learning_v8_geom "$@" >> $L/pen_v8.log 2>&1
for e in "$@"; do [ -f outputs/track3/learning_v8_geom/episode_$(printf %06d $e)/objects/poses.npz ] && touch state/learning_v8_geom/ep$(printf %06d $e).done; done
echo "DONE $*" >> $L/rest_v8.log
