"""Fetch public author-hosted hand assets and an ungated mesh-reconstruction fallback."""
import json
import os
import subprocess
import traceback
import urllib.request
from pathlib import Path

from download_track3_sample import digest, now, save

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'weights/reconstruction_sources.json'
STATUS = ROOT/'notes/track3-reconstruction-download-status.json'


def main():
    save(STATUS, {'state': 'running', 'started_utc': now()})
    plans = []
    for kind, repo, tag, mapping in [
        ('spaces', 'rolpotamias/WiLoR', 'wilor-author-space', {
            'mano_data/mano/MANO_RIGHT.pkl': 'wilor/pretrained_models/MANO_RIGHT.pkl',
            'mano_data/mano_mean_params.npz': 'wilor/pretrained_models/mano_mean_params.npz',
            'pretrained_models/wilor_final.ckpt': 'wilor/pretrained_models/wilor_final.ckpt',
            'pretrained_models/detector.pt': 'wilor/pretrained_models/detector.pt',
        }),
        ('models', 'stabilityai/TripoSR', 'triposr', {'config.yaml': 'triposr/config.yaml', 'model.ckpt': 'triposr/model.ckpt'}),
    ]:
        metadata_path = ROOT/'weights/model_catalog'/(tag+'.json')
        meta = json.loads(metadata_path.read_text())
        if meta.get('gated'):
            raise RuntimeError('Refusing to fetch a gated artifact through this public-only downloader')
        for name, local in mapping.items():
            e = next(e for e in meta['siblings'] if e['rfilename'] == name)
            prefix = 'spaces/' if kind == 'spaces' else ''
            plans.append({'path': local, 'upstream_path': name, 'repository': repo, 'repository_type': kind,
                          'revision': meta['sha'], 'url': f'https://huggingface.co/{prefix}{repo}/resolve/{meta["sha"]}/{name}',
                          'expected_size': e['size'], 'expected_sha256': e.get('lfs', {}).get('sha256')})
    if sum(e['expected_size'] for e in plans) > 6*1024**3:
        raise RuntimeError('Public diagnostic download plan exceeds 6 GiB')
    old_manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    old = {e['path']: e for e in old_manifest.get('files', [])}
    manifest = {'started_utc': now(), 'files': [], 'gated_resources_downloaded': False}
    for e in plans:
        dest = ROOT/'weights'/e['path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        expected = e['expected_sha256'] or old.get(e['path'], {}).get('sha256')
        if dest.exists():
            if dest.stat().st_size != e['expected_size'] or (expected and digest(dest) != expected):
                raise RuntimeError('Existing artifact differs: '+e['path'])
        else:
            partial = dest.with_name(dest.name+'.part')
            print(now(), 'download', e['path'], e['expected_size'], flush=True)
            subprocess.run(['curl','--fail','--location','--silent','--show-error','--proto','=https','--proto-redir','=https',
                            '--retry','3','--retry-delay','2','--connect-timeout','20','--max-time','1200',
                            '--max-filesize',str(max(e['expected_size'],1024*1024)),
                            '--continue-at','-','--output',str(partial),e['url']],check=True)
            if partial.stat().st_size != e['expected_size'] or (expected and digest(partial) != expected):
                raise RuntimeError('Artifact validation failed: '+e['path'])
            partial.replace(dest)
        manifest['files'].append(dict(e, size_bytes=dest.stat().st_size, sha256=digest(dest), verified_utc=now()))
        save(MANIFEST, manifest)
        print(now(), 'verified', e['path'], flush=True)
    source = ROOT/'vendor/TripoSR'
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1', GIT_TERMINAL_PROMPT='0')
    if not source.exists():
        subprocess.run(['git','clone','--depth','1','https://github.com/VAST-AI-Research/TripoSR.git',str(source)],env=env,check=True)
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    manifest.update(completed_utc=now(), total_bytes=sum(e['size_bytes'] for e in manifest['files']), triposr_source_commit=commit)
    save(MANIFEST, manifest)
    save(STATUS, {'state':'completed','completed_utc':now(),'files':len(plans),'bytes':manifest['total_bytes'],
                  'triposr_source_commit':commit,'manifest':str(MANIFEST)})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save(STATUS, {'state':'failed','updated_utc':now(),'error_type':type(exc).__name__})
        traceback.print_exc()
        raise SystemExit(1)
