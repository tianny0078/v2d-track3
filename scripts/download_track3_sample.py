"""Fetch the public Track 3 episode 11 with pinned revision and resumable files."""
import hashlib
import json
import subprocess
import sys
import traceback
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'track3_sample'
STATUS = ROOT / 'notes' / 'track3-download-status.json'
MANIFEST = OUT / 'download_manifest.json'
REPO = 'nvidia/video_to_data_challenge'


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    save(STATUS, {'state': 'running', 'started_utc': now(), 'pid': __import__('os').getpid()})
    pin = OUT / 'dataset_revision.json'
    if pin.exists():
        metadata = json.loads(pin.read_text())
    else:
        with urllib.request.urlopen(f'https://huggingface.co/api/datasets/{REPO}?blobs=true', timeout=30) as response:
            metadata = json.load(response)
        save(pin, metadata)
    revision = metadata['sha']
    print(now(), 'dataset revision', revision, flush=True)
    selected = []
    for entry in metadata['siblings']:
        path = entry['rfilename']
        if (path in ['README.md', 'track_3/README.md', 'track_3/v2d-challenge-track3.md']
                or path.startswith('track_3/public/meta/')
                or (path.startswith('track_3/public/') and ('/episode_000011.' in path))):
            selected.append(entry)
    if not selected or not any(e['rfilename'].endswith('episode_000011.parquet') for e in selected):
        raise RuntimeError('Episode 11 files not present; refusing an unscoped download')
    planned = sum(e.get('size', 0) for e in selected)
    if planned > 200 * 1024 * 1024:
        raise RuntimeError(f'Sample exceeds 200 MiB cap: {planned}')
    files = []
    for entry in sorted(selected, key=lambda e: e['rfilename']):
        path = entry['rfilename']
        files.append({
            'path': path, 'url': f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{urllib.parse.quote(path)}',
            'expected_size': entry.get('size'), 'expected_sha256': entry.get('lfs', {}).get('sha256'),
        })
    files.append({
        'path': 'toolkit/v2d_submission_kit.zip',
        'url': 'https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_submission_kit.zip',
        'expected_size': None, 'expected_sha256': None,
    })
    manifest = {'repository': REPO, 'revision': revision, 'started_utc': now(), 'files': []}
    previous = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    old = {f['path']: f for f in previous.get('files', [])}
    for entry in files:
        target = OUT / entry['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        expected_hash = entry['expected_sha256'] or old.get(entry['path'], {}).get('sha256')
        valid = target.exists() and (entry['expected_size'] is None or target.stat().st_size == entry['expected_size'])
        if valid and expected_hash:
            valid = digest(target) == expected_hash
        elif valid:
            valid = False
        if not valid:
            if target.exists():
                raise RuntimeError(f'Existing completed file failed validation: {entry["path"]}')
            partial = target.with_name(target.name + '.part')
            print(now(), 'download', entry['path'], flush=True)
            subprocess.run([
                'curl', '--fail', '--location', '--silent', '--show-error',
                '--proto', '=https', '--proto-redir', '=https',
                '--retry', '3', '--retry-delay', '2', '--connect-timeout', '20',
                '--max-time', '300', '--max-filesize', str(64 * 1024 * 1024),
                '--continue-at', '-', '--output', str(partial), entry['url'],
            ], check=True)
            if entry['expected_size'] is not None and partial.stat().st_size != entry['expected_size']:
                raise RuntimeError(f'Size mismatch: {entry["path"]}')
            if expected_hash and digest(partial) != expected_hash:
                raise RuntimeError(f'SHA256 mismatch: {entry["path"]}')
            partial.replace(target)
        result = dict(entry, size_bytes=target.stat().st_size, sha256=digest(target), verified_utc=now())
        manifest['files'].append(result)
        save(MANIFEST, manifest)
        print(now(), 'verified', entry['path'], result['size_bytes'], flush=True)
    manifest.update(completed_utc=now(), total_bytes=sum(f['size_bytes'] for f in manifest['files']))
    save(MANIFEST, manifest)
    save(STATUS, {'state': 'completed', 'completed_utc': now(), 'files': len(files),
                  'bytes': manifest['total_bytes'], 'revision': revision, 'manifest': str(MANIFEST)})
    print(now(), 'completed', len(files), 'files', manifest['total_bytes'], 'bytes', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save(STATUS, {'state': 'failed', 'updated_utc': now(), 'error_type': type(exc).__name__})
        traceback.print_exc()
        sys.exit(1)
