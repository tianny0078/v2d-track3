"""Isolated submission client setup and read-only Track 3 account/rules check."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT/'.deps/v2d-submit-venv'
REQUIREMENTS = ROOT/'toolkit/8d2bce64/v2d_submission_kit/requirements.txt'
REPORT = ROOT/'notes/track3-kaggle-environment.json'


def main():
    report = {'state': 'preparing', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'environment': str(ENV), 'requirements_sha256': hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest(),
              'submission': False, 'existing_credentials_reused': True}
    REPORT.write_text(json.dumps(report, indent=2)+'\n')
    config = dict(os.environ, UV_CACHE_DIR=str(ROOT/'.deps/kaggle-uv-cache'), UV_PYTHON_DOWNLOADS='never')
    uv = ROOT/'.deps/uvtool/bin/uv'; python = ENV/'bin/python'
    try:
        if not python.exists():
            subprocess.run([str(uv), 'venv', '--python', str(ROOT/'.deps/flash-chord-venv/bin/python'), str(ENV)], check=True, env=config)
        subprocess.run([str(uv), 'pip', 'install', '--python', str(python), '-r', str(REQUIREMENTS)], check=True, env=config)
        frozen = subprocess.check_output([str(uv), 'pip', 'freeze', '--python', str(python)], text=True, env=config)
        (ROOT/'notes/track3-kaggle-packages.txt').write_text(frozen)
        result = subprocess.run([str(python), '-u', str(ROOT/'scripts/check_kaggle_track3_entry.py')], env=config, timeout=180)
        report.update(state='completed', entry_check_exit_code=result.returncode,
                      entry_report='notes/track3-kaggle-entry.json', completed_utc=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        report.update(state='failed', error_type=type(exc).__name__, completed_utc=datetime.now(timezone.utc).isoformat())
        REPORT.write_text(json.dumps(report, indent=2)+'\n')
        raise
    REPORT.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
