"""Create runtime directories and optionally fetch pinned source, without model data.

Default invocation prints the plan. Put --fetch-sources in server tmux.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = '193382a184308f6800331f3db829f76dc4294cfe'
TRIPOSR_COMMIT = '107cefdc244c39106fa830359024f6a2f1c78871'
REPOS = [
    ('video_to_data', 'https://github.com/nvidia-isaac/video_to_data.git', SOURCE_COMMIT,
     ['reconstruction', 'robotic_grounding/flash_chord', 'robotic_grounding/source', 'robotic_grounding/scripts', 'docs']),
    ('TripoSR', 'https://github.com/VAST-AI-Research/TripoSR.git', TRIPOSR_COMMIT, None),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--init-runtime-dirs', action='store_true')
    p.add_argument('--fetch-sources', action='store_true')
    args = p.parse_args()
    runtime = ['notes', 'logs', 'data', 'weights', 'outputs', 'vendor', 'toolkit', '.deps']
    print(json.dumps({'workspace': str(ROOT), 'runtime_directories': runtime,
                      'sources': [{'name': n, 'url': u, 'commit': s} for n,u,s,_ in REPOS],
                      'models_or_data_downloaded': False}, indent=2))
    if args.init_runtime_dirs or args.fetch_sources:
        for folder in runtime: (ROOT/folder).mkdir(exist_ok=True)
    if not args.fetch_sources: return
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1', GIT_TERMINAL_PROMPT='0')
    for name, url, commit, sparse in REPOS:
        path = ROOT/'vendor'/name
        if path.exists():
            if not (path/'.git').exists():
                raise SystemExit(f'{name} is not a Git checkout; leaving it unchanged.')
            probe = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True, capture_output=True)
            actual = probe.stdout.strip() if probe.returncode == 0 else None
            if actual is not None and actual != commit:
                raise SystemExit(f'{name} has a different revision; leaving existing checkout unchanged.')
            if actual == commit:
                print('Source ready:',name,commit)
                continue
        else:
            path.mkdir()
            subprocess.run(['git','init',str(path)],check=True,env=env)
            subprocess.run(['git','-C',str(path),'remote','add','origin',url],check=True,env=env)
        subprocess.run(['git','-C',str(path),'fetch','--depth=1','--filter=blob:none','origin',commit],check=True,env=env)
        if sparse:
            subprocess.run(['git','-C',str(path),'sparse-checkout','init','--cone'],check=True,env=env)
            subprocess.run(['git','-C',str(path),'sparse-checkout','set',*sparse],check=True,env=env)
        subprocess.run(['git','-C',str(path),'checkout','--detach',commit],check=True,env=env)
        print('Source ready:',name,commit)


if __name__ == '__main__':
    main()
