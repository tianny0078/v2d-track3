"""Launch a named, resumable Track 3 stage in tmux; never duplicate active stages."""
import argparse
import json
import shlex
import subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
STEPS={
    'flashchord-assets': ('prepare_flashchord_assets.py','notes/track3-flashchord-assets.json'),
    'flashchord-interfaces': ('check_flashchord_interfaces.py','notes/track3-flashchord-interfaces.json'),
    'retarget-source': ('prepare_official_retarget_source.py','notes/track3-retarget-source.json'),
    'hourly-probe': ('probe_hourly_agent.py','notes/track3-hourly-capability.json'),
    'hot3d-hand-pilot': ('run_hot3d_hand_pilot.py','outputs/track3/episode_000011/hot3d_hand_tracking/report.json'),
    'sam3d-local-sample': ('run_sam3d_local_sample.py','outputs/track3/episode_000011/object_meshes_sam3d_local/report.json'),
    'sharpa-ik-initializer': ('initialize_sharpa_retarget.py','outputs/track3/episode_000011/sharpa_keypoint_initializer/report.json'),
    'sharpa-ik-v2': ('initialize_sharpa_retarget.py','outputs/track3/episode_000011/sharpa_keypoint_initializer_v2/report.json'),
    'contact-wrench-probe': ('probe_contact_wrench.py','outputs/track3/episode_000011/contact_wrench_probe/report.json'),
}
parser=argparse.ArgumentParser();parser.add_argument('step',choices=STEPS);args=parser.parse_args()
script,report=STEPS[args.step];prefix='v2d-t3-'+args.step
sessions=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True)
active=[s for s in sessions.stdout.splitlines() if s.startswith(prefix+'-')]
if active:print('Already running:',','.join(active));raise SystemExit(0)
report_path=ROOT/report
if report_path.exists() and json.loads(report_path.read_text()).get('state')=='completed':
    print('Completed stage exists; reusing',report);raise SystemExit(0)
stamp=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d-%H%M%S')
log=ROOT/f'logs/track3-{args.step}-{stamp}.log';exitfile=log.with_suffix('.exit');session=prefix+'-'+stamp
argv=[str(ROOT/'.deps/flash-chord-venv/bin/python'),'-u',str(ROOT/'scripts'/script)]
env=['env','CUDA_VISIBLE_DEVICES=0','XLA_PYTHON_CLIENT_PREALLOCATE=false',
     'OPENBLAS_NUM_THREADS=8','OMP_NUM_THREADS=8','WANDB_MODE=disabled',
     'WP_CACHE_DIR='+str(ROOT/'.deps/flash-chord-warp-cache')]
shell=shlex.join(env+argv)+' >> '+shlex.quote(str(log))+' 2>&1; task_rc=$?; printf "%s\\n" "$task_rc" > '+shlex.quote(str(exitfile))+'; exit "$task_rc"'
job={'session':session,'argv':env+argv,'log':str(log),'exit_file':str(exitfile),'report':report,
     'hostname':subprocess.check_output(['hostname'],text=True).strip(),
     'attach':'tmux attach -t '+session,'tail':'tail -n 50 '+str(log),
     'resume':'cd '+str(ROOT)+' && python3 scripts/launch_track3_step.py '+args.step}
(ROOT/f'notes/track3-{args.step}-job.json').write_text(json.dumps(job,indent=2)+'\n')
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),shell],check=True)
print(json.dumps(job,indent=2))
