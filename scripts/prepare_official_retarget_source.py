"""Expand the pinned sparse checkout for the actual upstream retarget implementation, skipping LFS."""
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1];REPO=ROOT/'vendor/video_to_data'
before=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1',GIT_TERMINAL_PROMPT='0')
subprocess.run(['git','-C',str(REPO),'sparse-checkout','add','robotic_grounding/source','robotic_grounding/scripts'],env=env,check=True)
after=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip();assert before==after
out={'state':'completed','code_commit':after,'completed_utc':datetime.now(timezone.utc).isoformat(),
     'lfs_smudge':False,'note':'Existing pinned source only; no GT object meshes or trajectories materialized.'}
(ROOT/'notes/track3-retarget-source.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
