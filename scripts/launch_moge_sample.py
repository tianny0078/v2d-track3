"""Run the recorded offline three-frame experiment in an independent tmux session."""
import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--centered-intrinsics', action='store_true')
options = parser.parse_args()
tag = 'moge-sample-calibrated' if options.centered_intrinsics else 'moge-sample'
subdir = 'moge_sample_calibrated' if options.centered_intrinsics else 'moge_sample'
out = root / 'outputs/track3/episode_000011' / subdir
report = out / 'report.json'
if report.exists() and json.loads(report.read_text()).get('state') == 'completed':
    print('Completed results already exist:', report)
    raise SystemExit(0)
out.mkdir(parents=True, exist_ok=True)
mounts = [(root/'scripts', '/work/scripts', True),
          (root/'weights/moge', '/work/weights/moge', True),
          (root/'data/track3_sample/track_3/public/meta', '/work/data/track3_sample/track_3/public/meta', True),
          (root/'outputs/track3/episode_000011/validation', '/work/outputs/track3/episode_000011/validation', True),
          (out, '/work/outputs/track3/episode_000011/' + subdir, False)]
image = 'sha256:80b015d849c7b8d00222d87da1a9d6c0b427da2561121205dab4b435d5fec202'
args = ['docker', 'run', '--rm', '--pull=never', '--name', 'v2d-t3-' + tag,
        '--network=none', '--read-only', '--tmpfs', '/tmp:rw,exec,size=4g', '--gpus', 'device=0',
        '--user', f'{os.getuid()}:{os.getgid()}', '-e', 'TORCH_HOME=/tmp/torch',
        '-e', 'CUDA_CACHE_PATH=/tmp/cuda-cache']
for src, dst, ro in mounts:
    args += ['--mount', f'type=bind,src={src},dst={dst}' + (',readonly' if ro else '')]
args += ['--entrypoint', 'python', image, '-u', '/work/scripts/run_moge_sample.py']
if options.centered_intrinsics:
    args += ['--centered-intrinsics']
stamp = datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d-%H%M%S')
record = {'session': 'v2d-t3-' + tag + '-' + stamp,
          'log': str(root / 'logs' / f'track3-{tag}.log'),
          'exit_file': str(root / 'logs' / f'track3-{tag}.exit'),
          'argv': args, 'image_id': image, 'checkpoint': str(root / 'weights/moge/model.pt'),
          'training': False}
record['attach'] = 'tmux attach -t ' + record['session']
record['resume'] = 'cd ' + str(root) + ' && python3 scripts/launch_moge_sample.py' + (' --centered-intrinsics' if options.centered_intrinsics else '')
command = shlex.join(args) + ' > ' + shlex.quote(record['log']) + ' 2>&1; task_rc=$?; printf "%s\\n" "$task_rc" > ' + shlex.quote(record['exit_file']) + '; exit "$task_rc"'
(root / 'notes' / f'track3-{tag}-job.json').write_text(json.dumps(record, indent=2) + '\n')
subprocess.run(['tmux', 'new-session', '-d', '-s', record['session'], '-c', str(root), command], check=True)
print(json.dumps(record, indent=2))
