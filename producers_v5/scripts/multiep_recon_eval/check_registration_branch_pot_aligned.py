"""VALIDATION ONLY: re-run check_registration_branch.py for white_pot with the NEW mesh = procedural_geometry_v8/white_pot/
visual_4096.obj (adopted final_4096 rigidly re-framed into the v1 frame, 2026-10-31). Output: eval/registration_branch_white_pot.{json,md}."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import check_registration_branch as C
_orig = C.mesh_paths
def mesh_paths():
    paths, skipped = _orig(); paths['white_pot'] = (paths['white_pot'][0], C.R/'outputs/track3/procedural_geometry_v8/white_pot/visual_4096.obj')
    return paths, skipped
C.mesh_paths = mesh_paths
if __name__ == '__main__':
    C.main(['white_pot'])
