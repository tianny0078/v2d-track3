"""Reuse the user's existing validated HOT3D hand pipeline on this calibrated episode."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/track3/episode_000011'
OUT=BASE/'hot3d_hand_tracking'
SOURCE=Path('/mnt/nas3/hot3d/repo_handpose/mai_annotation_handpose')
IMAGE='sha256:094dc71cf1c0694345c4aca6a789762186a5cbc8dcdff489a3dfb7ad3b48d1f1'


def main():
    OUT.mkdir(exist_ok=True)
    report={'state':'running','started_utc':datetime.now(timezone.utc).isoformat(),'frames':[],
            'image_id':IMAGE,'method':'Existing HOT3D pinhole perspective crop + HaMeR + 3D detection-by-tracking; unmodified algorithms.',
            'source_sha256':{str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             [SOURCE/'scripts/run_video.py',SOURCE/'handpose/core.py',SOURCE/'deliver/dbt/dbt_run.py']},
            'limitations':['New method comparison on this episode, not GT accuracy.',
                           'No E2/HiiCam calibration or per-camera k is transferred.',
                           'Tracked-only outputs retain frames_since_detection and are not automatically accepted as observed hands.']}
    save=lambda:(OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    save()
    k=np.load(BASE/'moge_sample_calibrated/frame_000000/undistorted_input_K.npy')
    assert abs(float(k[0,0]-k[1,1]))<1e-6
    camera={'model':'pinhole','f':float(k[0,0]),'cx':float(k[0,2]),'cy':float(k[1,2]),'width':1024,'height':767}
    (OUT/'camera.json').write_text(json.dumps(camera,indent=2)+'\n')
    video=OUT/'calibrated_rgb.avi'
    if not video.exists():
        partial=OUT/'calibrated_rgb.partial.avi'
        subprocess.run(['ffmpeg','-v','error','-y','-framerate','20','-i',str(BASE/'calibrated_sequence/frames/%06d.jpg'),
                        '-frames:v','137','-c:v','ffv1','-level','3','-pix_fmt','bgr0',str(partial)],check=True)
        partial.replace(video)
    cmd=['docker','run','--rm','--pull=never','--name','v2d-t3-hot3d-hand','--gpus','device=1','--network=none',
         '--cpus','8','--memory','32g','--read-only','--tmpfs','/tmp:rw,size=8g','--shm-size','2g',
         '--user',f'{os.getuid()}:{os.getgid()}',
         '--mount',f'type=bind,src={SOURCE},dst=/opt/handpose,readonly',
         '--mount','type=bind,src=/mnt/nas3/hot3d,dst=/mnt/nas3/hot3d,readonly',
         '--mount','type=bind,src=/mnt/nas3/simfoundry-deps/hamer/_DATA/hamer_ckpts,dst=/opt/hamer/_DATA/hamer_ckpts,readonly',
         '--mount','type=bind,src=/mnt/nas3/simfoundry-deps/hamer/_DATA/data/mano_mean_params.npz,dst=/opt/hamer/_DATA/data/mano_mean_params.npz,readonly',
         '--mount','type=bind,src=/mnt/nas3/hot3d/mano_v1_2/models,dst=/opt/hamer/_DATA/data/mano,readonly',
         '--mount','type=bind,src=/mnt/nas3/v2d/reconstruction/data/weights/wilor/pretrained_models/detector.pt,dst=/models/wilor/detector.pt,readonly',
         '--mount',f'type=bind,src={OUT},dst={OUT}',
         '-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=8','-e','OPENBLAS_NUM_THREADS=8',
         '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','--entrypoint','python',IMAGE,
         '/opt/handpose/scripts/run_video.py','--video',str(video),'--camera',str(OUT/'camera.json'),
         '--out',str(OUT/'pipeline'),'--frames','0:136','--gpu','0','--name','track3_episode_000011','--no-render','--detector','native']
    report['docker_argv']=cmd;save()
    try:
        subprocess.run(cmd,check=True)
        rows=[json.loads(line) for line in (OUT/'pipeline/annotations.jsonl').read_text().splitlines()]
        perframe={int(r['frame_index']):r['hands'] for r in rows}
        report['frames']=[{'frame_index':i,'hands':len(perframe.get(i,[])),
                           'tracked_only':sum(h.get('frames_since_detection',0)>0 for h in perframe.get(i,[]))} for i in range(137)]
        report.update(state='completed',completed_utc=datetime.now(timezone.utc).isoformat(),
                      total_hands=sum(r['hands'] for r in report['frames']),
                      two_hand_frames=sum(r['hands']==2 for r in report['frames']),
                      frames_without_hands=sum(r['hands']==0 for r in report['frames']))
        save();print(json.dumps({k:v for k,v in report.items() if k not in ['frames','docker_argv']},ensure_ascii=False,indent=2),flush=True)
    except Exception as exc:
        report.update(state='failed',error_type=type(exc).__name__,error=str(exc),completed_utc=datetime.now(timezone.utc).isoformat());save();raise


if __name__=='__main__':main()
