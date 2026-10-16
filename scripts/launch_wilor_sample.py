"""Launch one offline three-frame hand experiment with persistent logs."""
import argparse,json,os,shlex,subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--low-confidence',action='store_true')
parser.add_argument('--mediapipe-boxes',action='store_true')
parser.add_argument('--sequence',action='store_true')
parser.add_argument('--identity-repair',action='store_true')
options=parser.parse_args()
tag='hand_sample_low_threshold' if options.low_confidence else 'hand_sample'
if options.mediapipe_boxes:tag='hand_sample_mediapipe_guided'
if options.sequence:
    if not options.mediapipe_boxes:raise ValueError('--sequence requires --mediapipe-boxes')
    tag='hand_sequence_hybrid'
if options.identity_repair:
    if not options.sequence:raise ValueError('--identity-repair requires --sequence')
    tag='hand_sequence_identity_repair'
root=Path(__file__).resolve().parents[1];out=root/'outputs/track3/episode_000011'/tag
if (out/'report.json').exists() and json.loads((out/'report.json').read_text()).get('state')=='completed':
    print('Completed hand experiment already exists');raise SystemExit(0)
for file in ['MANO_RIGHT.pkl','mano_mean_params.npz','wilor_final.ckpt','detector.pt']:
    if not (root/'weights/wilor/pretrained_models'/file).exists():
        print('Hand weights not ready; no task started:',file);raise SystemExit(0)
name='v2d-t3-'+tag.replace('_','-')
if subprocess.check_output(['docker','ps','--filter','name=^/'+name+'$','--format','{{.Names}}'],text=True).strip():
    print('Existing hand task active; no duplicate started');raise SystemExit(0)
out.mkdir(parents=True,exist_ok=True)
image='sha256:367f7ab879e29469b7d222635f5797f962906ef0afd01e2586556850cc9aec32'
cmd=['docker','run','--rm','--pull=never','--name',name,'--network=none','--read-only','--tmpfs','/tmp:rw,exec,size=4g',
     '--gpus','device=0','--user',f'{os.getuid()}:{os.getgid()}','-e','HF_HUB_OFFLINE=1','-e','HF_HOME=/tmp/hf',
     '-e','TORCH_HOME=/tmp/torch','-e','CUDA_CACHE_PATH=/tmp/cuda-cache','-e','YOLO_CONFIG_DIR=/tmp/yolo',
     '-e','PYTORCH_KERNEL_CACHE_PATH=/tmp/torch-kernels','-e','MPLCONFIGDIR=/tmp/matplotlib']
for src,dst,ro in [(root/'scripts','/work/scripts',True),(root/'weights/wilor','/work/weights/wilor',True),
                   (root/'outputs/track3/episode_000011/moge_sample_calibrated','/work/outputs/track3/episode_000011/moge_sample_calibrated',True),
                   (out,'/work/outputs/track3/episode_000011/'+tag,False)]:
    cmd+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
cmd+=['--entrypoint','python',image,'-u','/work/scripts/run_wilor_sample.py']
if options.low_confidence:cmd+=['--low-confidence']
if options.mediapipe_boxes:
    # Insert a read-only model-independent detector result before the image argument.
    image_index=cmd.index('--entrypoint')
    mp_tag='hand_detector_mediapipe_sequence' if options.sequence else 'hand_detector_mediapipe'
    cmd[image_index:image_index]=['--mount',f'type=bind,src={root}/outputs/track3/episode_000011/{mp_tag},dst=/work/outputs/track3/episode_000011/{mp_tag},readonly']
    cmd+=['--mediapipe-boxes']
if options.sequence:
    image_index=cmd.index('--entrypoint')
    cmd[image_index:image_index]=['--mount',f'type=bind,src={root}/outputs/track3/episode_000011/calibrated_sequence,dst=/work/outputs/track3/episode_000011/calibrated_sequence,readonly']
    cmd+=['--sequence']
if options.identity_repair:
    image_index=cmd.index('--entrypoint')
    cmd[image_index:image_index]=['--mount',f'type=bind,src={root}/outputs/track3/episode_000011/hand_sequence_hybrid,dst=/work/outputs/track3/episode_000011/hand_sequence_hybrid,readonly']
    cmd+=['--identity-repair']
stamp=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%d-%H%M%S')
r={'session':'v2d-t3-'+tag.replace('_','-')+'-'+stamp,'argv':cmd,'image_id':image,'log':str(root/'logs'/('track3-'+tag+'.log')),
   'exit_file':str(root/'logs'/('track3-'+tag+'.exit')),'resume':f'cd {root} && python3 scripts/launch_wilor_sample.py'+(' --low-confidence' if options.low_confidence else ''),'training':False}
if options.mediapipe_boxes:r['resume']+=' --mediapipe-boxes'
if options.sequence:r['resume']+=' --sequence'
if options.identity_repair:r['resume']+=' --identity-repair'
r['attach']='tmux attach -t '+r['session']
shell=shlex.join(cmd)+' >> '+shlex.quote(r['log'])+' 2>&1; task_rc=$?; printf "%s\\n" "$task_rc" > '+shlex.quote(r['exit_file'])+'; exit "$task_rc"'
(root/'notes'/('track3-'+tag+'-job.json')).write_text(json.dumps(r,indent=2)+'\n')
subprocess.run(['tmux','new-session','-d','-s',r['session'],'-c',str(root),shell],check=True);print('Started',r['session'])
