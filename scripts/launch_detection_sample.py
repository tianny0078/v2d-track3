"""Launch/resume the three-frame offline detector in tmux without replacing jobs."""
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
parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
args = parser.parse_args()
out = root / 'outputs/track3/episode_000011/object_detection'
report = out / 'report.json'
if report.exists() and json.loads(report.read_text()).get('state') == 'completed':
    print('Reusing completed result:', report)
    raise SystemExit(0)
status = json.loads((root / 'notes/track3-segmentation-download-status.json').read_text())
if status.get('state') != 'completed':
    raise RuntimeError('Required model download has not completed')
out.mkdir(parents=True, exist_ok=True)
image = 'sha256:6b1b94dd1fa0418b1aad2b3e164b35745624e9d9d42c6404dd5754c990d5e05c'
name = 'v2d-t3-object-detection'
active = subprocess.check_output(['docker', 'ps', '--filter', 'name=^/' + name + '$', '--format', '{{.Names}}'], text=True).strip()
if active:
    print('Existing task remains active:', active)
    raise SystemExit(0)
cmd = ['docker', 'run', '--rm', '--pull=never', '--name', name, '--network=none', '--read-only',
       '--tmpfs', '/tmp:rw,exec,size=4g', '--user', f'{os.getuid()}:{os.getgid()}',
       '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1', '-e', 'HF_HOME=/tmp/hf',
       '-e', 'TORCH_HOME=/tmp/torch', '-e', 'CUDA_CACHE_PATH=/tmp/cuda-cache']
if args.device == 'cuda':
    cmd += ['--gpus', 'device=0']
for src, dst, ro in [(root/'scripts', '/work/scripts', True), (root/'weights/bert', '/work/weights/bert', True),
                     (root/'weights/grounding_dino', '/work/weights/grounding_dino', True),
                     (root/'outputs/track3/episode_000011/moge_sample_calibrated', '/work/outputs/track3/episode_000011/moge_sample_calibrated', True),
                     (out, '/work/outputs/track3/episode_000011/object_detection', False)]:
    cmd += ['--mount', f'type=bind,src={src},dst={dst}' + (',readonly' if ro else '')]
cmd += ['--entrypoint', 'python', image, '-u', '/work/scripts/run_grounding_dino_sample.py', '--device', args.device]
stamp = datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d-%H%M%S')
record = {'session': 'v2d-t3-detection-' + stamp, 'argv': cmd, 'image_id': image,
          'log': str(root/'logs/track3-object-detection.log'), 'exit_file': str(root/'logs/track3-object-detection.exit'),
          'training': False}
record['attach'] = 'tmux attach -t ' + record['session']
record['resume'] = f'cd {root} && python3 scripts/launch_detection_sample.py --device {args.device}'
shell_cmd = shlex.join(cmd) + ' > ' + shlex.quote(record['log']) + ' 2>&1; task_rc=$?; printf "%s\\n" "$task_rc" > ' + shlex.quote(record['exit_file']) + '; exit "$task_rc"'
(root/'notes/track3-detection-job.json').write_text(json.dumps(record, indent=2) + '\n')
subprocess.run(['tmux', 'new-session', '-d', '-s', record['session'], '-c', str(root), shell_cmd], check=True)
print('Started', record['session'])
