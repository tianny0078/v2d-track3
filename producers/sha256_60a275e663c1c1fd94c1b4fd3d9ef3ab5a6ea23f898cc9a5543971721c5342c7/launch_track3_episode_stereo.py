"""Prepare or launch a bounded, episode-specific stereo preprocessing job."""
import argparse,datetime,hashlib,json,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--episode',type=int,required=True)
p.add_argument('--input-root',type=Path,required=True)
p.add_argument('--calibration',type=Path,required=True)
p.add_argument('--output-root',type=Path,required=True)
p.add_argument('--launch',action='store_true')
a=p.parse_args()
for name in ['input_root','calibration','output_root']:setattr(a,name,getattr(a,name).resolve())
contract=json.loads((ROOT/'notes/track3-submission-contract.json').read_text())
if str(a.episode) not in contract['episodes'] and a.episode!=11:p.error('Episode outside Track3 contract')
session=f'v2d-t3-stereo-ep{a.episode:06d}'
log=a.output_root/'stereo-job.log';exit_file=a.output_root/'stereo-job.exit'
argv=['timeout','600','python3','-u',str(ROOT/'scripts/run_stereo_sequence.py'),'--episode',str(a.episode),'--input-root',str(a.input_root),'--calibration',str(a.calibration),'--output-root',str(a.output_root),'--allow-missing-masks']
resume=shlex.join(['python3',str(Path(__file__).resolve()),'--episode',str(a.episode),'--input-root',str(a.input_root),'--calibration',str(a.calibration),'--output-root',str(a.output_root),'--launch'])
a.output_root.mkdir(parents=True,exist_ok=True)
inputs=[a.calibration]+[a.input_root/f'videos/chunk-000/observation.images.ego_cam_{c}/episode_{a.episode:06d}.mp4' for c in ['b','c']]
missing=[str(i) for i in inputs if not i.is_file()]
job=dict(state='prepared_missing_inputs' if missing else 'prepared',episode=a.episode,session=session,argv=argv,missing_inputs=missing,log=str(log),exit_file=str(exit_file),checkpoint=str(a.output_root/'stereo_sequence/camera_poses.npz'),checkpoint_semantics='Per-frame depths and camera poses retained; interrupted VO restarts frame zero. Completed output is protected.',resume_command=resume,timeout_seconds=600,expected_scored_frames=contract['episodes'].get(str(a.episode),{}).get('scored_frames'),source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),ROOT/'scripts/run_stereo_sequence.py']},training=False,object_pose_inference=False)
jobpath=a.output_root/'stereo-job.json'
def save():jobpath.write_text(json.dumps(job,indent=2)+'\n')
save()
if a.launch:
 if missing:raise SystemExit('Required inputs missing; job configuration saved')
 result=subprocess.run(['tmux','has-session','-t',session],capture_output=True)
 if result.returncode==0:raise SystemExit('Session exists; inspect existing job')
 report=a.output_root/'stereo_sequence/report.json'
 if report.exists() and json.loads(report.read_text()).get('state')=='completed':raise SystemExit('Completed output exists; no duplicate launched')
 command=shlex.join(argv)+' > '+shlex.quote(str(log))+' 2>&1; task_rc=$?; printf "%s\\n" "$task_rc" > '+shlex.quote(str(exit_file))+'; exit "$task_rc"'
 subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),command],check=True)
 job.update(state='launched',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
print(json.dumps(job,indent=2))
