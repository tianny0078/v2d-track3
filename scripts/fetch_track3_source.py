"""Fetch official source without model/LFS payloads; do not run fetched code."""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
target = root / 'vendor' / 'video_to_data'
status = root / 'notes' / 'track3-source-status.json'
env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1', GIT_TERMINAL_PROMPT='0')


def write(data):
    data['updated_utc'] = datetime.now(timezone.utc).isoformat()
    status.write_text(json.dumps(data, indent=2) + '\n')


try:
    write({'state': 'running'})
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--depth', '1', '--filter=blob:none', '--sparse',
                        'https://github.com/nvidia-isaac/video_to_data.git', str(target)], env=env, check=True)
    if not (target / '.git').is_dir():
        raise RuntimeError('Existing target is not a git checkout; leaving it unchanged')
    subprocess.run(['git', 'sparse-checkout', 'set', 'reconstruction', 'robotic_grounding/flash_chord', 'docs'],
                   cwd=target, env=env, check=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=target, text=True).strip()
    write({'state': 'completed', 'path': str(target), 'commit': commit,
           'lfs_payloads_downloaded': False, 'executed_upstream_code': False})
    print('Pinned official source:', commit, flush=True)
except Exception as exc:
    write({'state': 'failed', 'error_type': type(exc).__name__})
    raise
