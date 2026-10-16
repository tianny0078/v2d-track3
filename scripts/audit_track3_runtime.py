"""Read-only runtime inventory; save status without reading process arguments or credentials."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
queries = {
    'tmux': ['tmux', 'list-panes', '-a', '-F', '#{session_name} command=#{pane_current_command} cwd=#{pane_current_path}'],
    'docker': ['docker', 'ps', '--format', '{{.Names}} | {{.Image}} | {{.Status}}'],
    'gpu': ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.free', '--format=csv'],
}
report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'read_only': True, 'runtime': {}, 'exit_codes': {}, 'outputs': {}}
for name, cmd in queries.items():
    result = subprocess.run(cmd, capture_output=True, text=True)
    report['runtime'][name] = {'exit_code': result.returncode, 'output': result.stdout.strip(), 'error': result.stderr.strip()}
for path in sorted((root/'logs').glob('*.exit')):
    report['exit_codes'][path.name] = path.read_text().strip()
for path in sorted((root/'outputs/track3').rglob('report.json')):
    info = json.loads(path.read_text())
    report['outputs'][str(path.relative_to(root))] = {'state': info.get('state', 'validation'), 'recorded_frames': (info.get('frames', 0) if isinstance(info.get('frames'), int) else len(info.get('frames', [])))}
dest = root/'notes/track3-runtime-audit.json'
dest.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
