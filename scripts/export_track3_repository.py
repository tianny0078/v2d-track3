"""Export an explicit source allowlist to a separate Git checkout, never runtime data.

This copies source only and does not commit, push, delete files, or alter experiments.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
RULES = {
    'private_key': r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
    'github_token': r'\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})',
    'aws_access_key': r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'literal_secret': r'''(?i)(?:token|password|secret|api_key)\s*[:=]\s*["'][A-Za-z0-9_-]{24,}["']''',
    'signed_url': r'(?i)[?&](?:X-Amz-Signature|access_token|AWSAccessKeyId)=',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    source = args.source.resolve(); destination = args.destination.resolve()
    if source == destination:
        raise SystemExit('Use a separate source checkout and publication checkout.')
    files = sorted((source/'scripts').glob('*.py'))
    site = Path('sites/track3-results-20261015/dist')
    files.extend(source/site/name for name in ['index.html', 'styles.css', 'app.js', 'progress.js'])
    manifest = []; findings = []
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise SystemExit('Only regular allowlisted source files may be exported.')
        relative = path.relative_to(source)
        text = path.read_text()
        for name, pattern in RULES.items():
            if re.search(pattern, text):
                findings.append({'path': str(relative), 'rule': name})
        if path.stat().st_size > 512*1024:
            raise SystemExit('Unexpected large source file; review the allowlist.')
        manifest.append({'path': str(relative), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size})
    if findings:
        print(json.dumps({'findings_without_values': findings}, indent=2))
        raise SystemExit('Source export stopped for review; no files copied.')
    if not args.dry_run:
        if not (destination/'.git').exists():
            raise SystemExit('Destination must be the selected Git checkout.')
        for item in manifest:
            target = destination/item['path']; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source/item['path'], target)
        target = destination/'config/source-manifest.json'; target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps({'exported_utc': datetime.now(timezone.utc).isoformat(),
                                     'scope': 'Own Python source and four static page templates only',
                                     'files': manifest}, indent=2)+'\n')
    print(json.dumps({'mode': 'dry-run' if args.dry_run else 'exported',
                      'files': len(manifest), 'bytes': sum(f['bytes'] for f in manifest),
                      'secret_pattern_findings': len(findings)}, indent=2))


if __name__ == '__main__':
    main()
