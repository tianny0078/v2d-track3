"""Launch one temporal-mask diagnostic in tmux, preserving completed experiments."""
import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

root = Path(__file__).resolve().parents[1]
out = root/'outputs/track3/episode_000011/object_masks_temporal'
if (out/'report.json').exists() and json.loads((out/'report.json').read_text()).get('state') == 'completed':
    print('Reusing completed temporal masks')
    raise SystemExit(0)
if not (root/'outputs/track3/episode_000011/calibrated_sequence/manifest.json').exists():
    raise RuntimeError('Calibrated sequence has not been prepared')
name = 'v2d-t3-temporal-masks'
if subprocess.check_output(['docker', 'ps', '--filter', 'name=^/' + name + '$', '--format', '{{.Names}}'], text=True).strip():
    print('Temporal task already running; no duplicate started')
    raise SystemExit(0)
out.mkdir(parents=True, exist_ok=True)
image = 'sha256:bc01a90c3cf71e0336e2baa312dda34f983778ba12472902ce6b4eb5ef1b4d52'
cmd = ['docker', 'run', '--rm', '--pull=never', '--name', name, '--network=none', '--read-only',
       '--tmpfs', '/tmp:rw,exec,size=4g', '--gpus', 'device=0', '--user', f'{os.getuid()}:{os.getgid()}',
       '-e', 'TORCH_HOME=/tmp/torch', '-e', 'CUDA_CACHE_PATH=/tmp/cuda-cache']
for src, dst, ro in [(root/'scripts', '/work/scripts', True), (root/'weights/sam2', '/work/weights/sam2', True),
                     (root/'outputs/track3/episode_000011/calibrated_sequence', '/work/outputs/track3/episode_000011/calibrated_sequence', True),
                     (root/'outputs/track3/episode_000011/object_masks/frame_000000', '/work/outputs/track3/episode_000011/object_masks/frame_000000', True),
                     (out, '/work/outputs/track3/episode_000011/object_masks_temporal', False)]:
    cmd += ['--mount', f'type=bind,src={src},dst={dst}' + (',readonly' if ro else '')]
cmd += ['--entrypoint', 'python', image, '-u', '/work/scripts/run_sam2_temporal.py']
stamp = datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d-%H%M%S')
record = {'session': 'v2d-t3-temporal-masks-' + stamp, 'argv': cmd, 'image_id': image,
          'log': str(root/'logs/track3-temporal-masks.log'), 'exit_file': str(root/'logs/track3-temporal-masks.exit'),
          'training': False, 'resume': f'cd {root} && python3 scripts/launch_sam2_temporal.py'}
record['attach'] = 'tmux attach -t '+record['session']
shell_cmd = shlex.join(cmd)+' > '+shlex.quote(record['log'])+' 2>&1; task_rc=$?; printf "%s\\n" "$task_rc" > '+shlex.quote(record['exit_file'])+'; exit "$task_rc"'
(root/'notes/track3-temporal-masks-job.json').write_text(json.dumps(record,indent=2)+'\n')
subprocess.run(['tmux','new-session','-d','-s',record['session'],'-c',str(root),shell_cmd],check=True)
print('Started',record['session'])
