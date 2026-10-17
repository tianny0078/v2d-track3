"""Prepare calibrated RGB for any Track3 episode; frame caches are source-bound."""
import argparse, hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode',type=int,required=True)
    p.add_argument('--input-root',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args();cv2.setNumThreads(4)
    video=a.input_root/f'videos/chunk-000/observation.images.ego_cam_a/episode_{a.episode:06d}.mp4'
    source={str(f.resolve()):digest(f) for f in [video,a.calibration,Path(__file__)]}
    out=a.output_root/'calibrated_sequence';frames=out/'frames';frames.mkdir(parents=True,exist_ok=True)
    manifest=out/'manifest.json';previous=json.loads(manifest.read_text()) if manifest.exists() else {}
    if previous and previous.get('source_sha256')!=source:raise RuntimeError('Source changed; use a new output root')
    camera=json.loads(a.calibration.read_text())['cameras']['ego_cam_a'];w,h=1024,767
    k=np.asarray(camera['intrinsic_matrix'],dtype=float);rw,rh=camera['resolution'];k[0]*=w/rw;k[1]*=h/rh
    distortion=np.asarray(camera['distortion_coefficients']);target,_=cv2.getOptimalNewCameraMatrix(k,distortion,(w,h),0,(w,h))
    maps=cv2.initUndistortRectifyMap(k,distortion,None,target,(w,h),cv2.CV_32FC1)
    cap=cv2.VideoCapture(str(video));assert cap.isOpened(),'Video unavailable'
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));assert count>0
    report=dict(state='running',episode=a.episode,source_sha256=source,source_video=str(video),source_frame_count=count,resolution=[w,h],target_K=target.tolist(),frames=[],frame_alignment='frame_index; no temporal resampling',pipeline='source JPEG encode/decode -> area resize -> calibrated undistort -> JPEG',started_utc=datetime.now(timezone.utc).isoformat(),limitations=['Model inputs only; no hand identity, object pose or GT claim.'])
    cached={r['frame_index']:r for r in previous.get('frames',[])};start=time.perf_counter();reused=0
    def save():
        tmp=manifest.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(manifest)
    for i in range(count):
        ok,image=cap.read()
        if not ok:raise RuntimeError(f'Video ended at {i}/{count}')
        dest=frames/f'{i:06d}.jpg';row=cached.get(i)
        if row and dest.exists() and digest(dest)==row['sha256']:reused+=1
        else:
            ok,encoded=cv2.imencode('.jpg',image);assert ok
            image=cv2.imdecode(encoded,cv2.IMREAD_COLOR);image=cv2.resize(image,(w,h),interpolation=cv2.INTER_AREA)
            image=cv2.remap(image,*maps,cv2.INTER_LINEAR);assert cv2.imwrite(str(dest),image)
            row=dict(frame_index=i,filename=dest.name,sha256=digest(dest))
        report['frames'].append(row)
        if i%20==0:save();print('frame',i,flush=True)
    cap.release();report.update(state='completed',elapsed_seconds=time.perf_counter()-start,reused_frames=reused,completed_utc=datetime.now(timezone.utc).isoformat());save();np.save(out/'rgb_K.npy',target)
    print('completed',count,flush=True)
if __name__=='__main__':main()
