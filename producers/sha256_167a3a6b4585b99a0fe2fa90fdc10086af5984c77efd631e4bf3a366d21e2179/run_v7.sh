#!/bin/bash
# learning_v7_geom: manifests (procedural assets) -> support fix (copy of v6 logic) -> rest test -> penetration diag
cd /home/tianyuan/projects/v2d
.deps/geom-venv/bin/python scripts/procedural_geometry/make_v7_manifests.py --tree learning_v7_geom > logs/procedural_geometry_v1/make_v7.log 2>&1
.deps/flash-chord-venv/bin/python scripts/procedural_geometry/fix_support_v7.py --tree learning_v7_geom --lift-max 0.06 > logs/procedural_geometry_v1/fix_support_v7.log 2>&1
.deps/flash-chord-venv/bin/python scripts/rest_test_track3_reference.py learning_v7_geom > logs/procedural_geometry_v1/rest_v7.log 2>&1
.deps/flash-chord-venv/bin/python scripts/procedural_geometry/diag_penetration_v7.py learning_v7_geom > logs/procedural_geometry_v1/pen_v7.log 2>&1
echo DONE >> logs/procedural_geometry_v1/rest_v7.log
