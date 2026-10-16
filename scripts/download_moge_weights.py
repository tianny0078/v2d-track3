"""Download the public MIT-licensed MoGe checkpoint referenced by V2D."""
import subprocess
import traceback
from pathlib import Path
from download_track3_sample import digest, save, now

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'weights' / 'moge'
META = {
    'repository': 'Ruicheng/moge-2-vitl-normal',
    'revision': 'cb0e8bbd6b1e243589717c78e750b1ba4c093acf',
    'license_metadata': 'mit',
    'file': 'model.pt',
    'expected_bytes': 1323815904,
    'expected_sha256': '280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01',
}
STATUS = ROOT / 'notes' / 'track3-moge-download-status.json'


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/{META["repository"]}/resolve/{META["revision"]}/model.pt'
    save(STATUS, {'state': 'running', 'started_utc': now()})
    target = DEST / 'model.pt'
    if target.exists():
        if target.stat().st_size != META['expected_bytes'] or digest(target) != META['expected_sha256']:
            raise RuntimeError('Existing checkpoint differs; preserving it and stopping')
    else:
        partial = DEST / 'model.pt.part'
        print(now(), 'Downloading pinned public checkpoint', META['repository'], flush=True)
        subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                        '--proto', '=https', '--proto-redir', '=https',
                        '--retry', '3', '--retry-delay', '2', '--connect-timeout', '20',
                        '--max-time', '900', '--max-filesize', str(META['expected_bytes']),
                        '--continue-at', '-', '--output', str(partial), url], check=True)
        if partial.stat().st_size != META['expected_bytes'] or digest(partial) != META['expected_sha256']:
            raise RuntimeError('Checkpoint size/SHA256 mismatch; incomplete file preserved')
        partial.replace(target)
    save(DEST / 'source.json', dict(META, url=url, verified_utc=now()))
    save(STATUS, {'state': 'completed', 'completed_utc': now(), 'bytes': target.stat().st_size,
                  'path': str(target), 'sha256': META['expected_sha256']})
    print(now(), 'Checkpoint verified', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save(STATUS, {'state': 'failed', 'updated_utc': now(), 'error_type': type(exc).__name__})
        traceback.print_exc()
        raise SystemExit(1)
