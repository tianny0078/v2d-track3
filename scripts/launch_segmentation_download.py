"""Launch a single resumable public-weight download, preserving earlier logs."""
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

root = Path(__file__).resolve().parents[1]
status = root/'notes/track3-segmentation-download-status.json'
if status.exists() and json.loads(status.read_text()).get('state') == 'completed':
    print('Weight download already completed; no job started')
    raise SystemExit(0)
active = subprocess.run(['tmux', 'list-sessions', '-F', '#{session_name}'], text=True, capture_output=True)
if any(s.startswith('v2d-t3-segmentation-weights-') for s in active.stdout.splitlines()):
    print('Existing weight-download session is still active; no duplicate started')
    raise SystemExit(0)
stamp = datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d-%H%M%S')
record = {'session': 'v2d-t3-segmentation-weights-' + stamp,
          'log': str(root/'logs/track3-segmentation-download.log'),
          'exit_file': str(root/'logs/track3-segmentation-download.exit'),
          'resume': f'cd {root} && python3 scripts/launch_segmentation_download.py', 'training': False}
record['attach'] = 'tmux attach -t ' + record['session']
cmd = 'python3 -u scripts/download_segmentation_weights.py >> ' + shlex.quote(record['log']) + ' 2>&1; task_rc=$?; printf "%s\\n" "$task_rc" > ' + shlex.quote(record['exit_file']) + '; exit "$task_rc"'
record['shell_command'] = cmd
(root/'notes/track3-segmentation-download-job.json').write_text(json.dumps(record, indent=2) + '\n')
subprocess.run(['tmux', 'new-session', '-d', '-s', record['session'], '-c', str(root), cmd], check=True)
print('Started/resumed', record['session'])
