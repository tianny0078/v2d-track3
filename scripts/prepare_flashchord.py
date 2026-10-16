"""Prepare the pinned official FlashCHORD lane in a project-local environment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
LANE = ROOT / 'vendor/video_to_data/robotic_grounding/flash_chord'
DEPS = ROOT / '.deps'
ENV = DEPS / 'flash-chord-venv'
REPORT = ROOT / 'notes/track3-flashchord-environment.json'


def main():
    config = os.environ.copy()
    config.update(UV_CACHE_DIR=str(DEPS/'flash-chord-cache'),
                  UV_PYTHON_INSTALL_DIR=str(DEPS/'flash-chord-python'),
                  UV_PROJECT_ENVIRONMENT=str(ENV), CUDA_VISIBLE_DEVICES='0',
                  XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='8',
                  OPENBLAS_NUM_THREADS='8', WANDB_MODE='disabled',
                  WP_CACHE_DIR=str(DEPS/'flash-chord-warp-cache'))
    report = {'state': 'preparing', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'code_commit': subprocess.check_output(['git', '-C', str(ROOT/'vendor/video_to_data'), 'rev-parse', 'HEAD'], text=True).strip(),
              'lock_sha256': hashlib.sha256((LANE/'uv.lock').read_bytes()).hexdigest(),
              'environment': str(ENV), 'training_started': False}
    REPORT.write_text(json.dumps(report, indent=2)+'\n')
    uv = DEPS/'uvtool/bin/uv'
    try:
        if not uv.exists():
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                            '--target', str(DEPS/'uvtool'), '--report', str(DEPS/'uvtool-install.json'),
                            'uv==0.11.21'], check=True, env=config)
        subprocess.run([str(uv), 'sync', '--project', str(LANE), '--python', '3.11.15',
                        '--frozen', '--extra', 'train', '--extra', 'dev'], check=True, env=config)
        subprocess.run([str(ENV/'bin/python'), '-u', str(ROOT/'scripts/smoke_flashchord.py')],
                       check=True, env=config, cwd=LANE)
        report.update(state='completed', completed_utc=datetime.now(timezone.utc).isoformat(),
                      smoke_report=str(ROOT/'notes/track3-flashchord-smoke.json'))
    except Exception as exc:
        report.update(state='failed', error=str(exc), completed_utc=datetime.now(timezone.utc).isoformat())
        REPORT.write_text(json.dumps(report, indent=2)+'\n')
        raise
    REPORT.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
