"""Resumable, versioned downloads for the Track 3 three-frame mask diagnostic."""
import json
import subprocess
import traceback
from pathlib import Path

from download_track3_sample import digest, now, save

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / 'notes/track3-segmentation-download-status.json'
MANIFEST = ROOT / 'weights/segmentation_sources.json'


def main():
    plans = []
    for repo, tag, names in [
        ('google-bert/bert-base-uncased', 'bert', ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt']),
        ('facebook/sam2.1-hiera-large', 'sam2', ['sam2.1_hiera_large.pt']),
    ]:
        meta = json.loads((ROOT / 'weights/model_catalog' / (tag + '.json')).read_text())
        if meta.get('gated') or meta.get('cardData', {}).get('license') != 'apache-2.0':
            raise RuntimeError('Unexpected gated/licensing metadata; stopping')
        for name in names:
            e = next(e for e in meta['siblings'] if e['rfilename'] == name)
            plans.append({'path': tag + '/' + name, 'repository': repo, 'revision': meta['sha'],
                          'url': f'https://huggingface.co/{repo}/resolve/{meta["sha"]}/{name}',
                          'expected_size': e['size'], 'expected_sha256': e.get('lfs', {}).get('sha256')})
    release = json.loads((ROOT / 'weights/model_catalog/groundingdino-release.json').read_text())
    e = next(e for e in release['assets'] if e['name'] == 'groundingdino_swint_ogc.pth')
    plans.append({'path': 'grounding_dino/' + e['name'], 'repository': 'IDEA-Research/GroundingDINO',
                  'release_id': release['id'], 'release_tag': release['tag_name'], 'asset_id': e['id'],
                  'url': e['browser_download_url'], 'expected_size': e['size'],
                  'expected_sha256': e.get('digest', '').removeprefix('sha256:') if e.get('digest') else None})
    if sum(p['expected_size'] for p in plans) > 3 * 1024**3:
        raise RuntimeError('Download plan exceeds 3 GiB diagnostic cap')
    previous = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    old = {f['path']: f for f in previous.get('files', [])}
    manifest = {'started_utc': now(), 'files': []}
    save(STATUS, {'state': 'running', 'started_utc': now(), 'file_count': len(plans)})
    for plan in plans:
        target = ROOT / 'weights' / plan['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = plan['expected_sha256'] or old.get(plan['path'], {}).get('sha256')
        if target.exists():
            if target.stat().st_size != plan['expected_size'] or (expected and digest(target) != expected):
                raise RuntimeError('Existing model file differs: ' + plan['path'])
        else:
            part = target.with_name(target.name + '.part')
            print(now(), 'download', plan['path'], plan['expected_size'], flush=True)
            subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                            '--proto', '=https', '--proto-redir', '=https',
                            '--retry', '3', '--retry-delay', '2', '--connect-timeout', '20',
                            '--max-time', '900', '--max-filesize', str(max(plan['expected_size'], 1024 * 1024)),
                            '--continue-at', '-', '--output', str(part), plan['url']], check=True)
            if part.stat().st_size != plan['expected_size'] or (expected and digest(part) != expected):
                raise RuntimeError('Model file validation failed: ' + plan['path'])
            part.replace(target)
        entry = dict(plan, size_bytes=target.stat().st_size, sha256=digest(target), verified_utc=now())
        manifest['files'].append(entry)
        save(MANIFEST, manifest)
        print(now(), 'verified', plan['path'], flush=True)
    manifest.update(completed_utc=now(), total_bytes=sum(f['size_bytes'] for f in manifest['files']))
    save(MANIFEST, manifest)
    save(STATUS, {'state': 'completed', 'completed_utc': now(), 'files': len(plans),
                  'bytes': manifest['total_bytes'], 'manifest': str(MANIFEST)})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save(STATUS, {'state': 'failed', 'updated_utc': now(), 'error_type': type(exc).__name__})
        traceback.print_exc()
        raise SystemExit(1)
