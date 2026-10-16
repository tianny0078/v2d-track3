"""Cache one public NVIDIA stereo model and isolated ONNX dependencies."""
import json,os,subprocess,traceback
from pathlib import Path
from download_track3_sample import digest,now,save

ROOT=Path(__file__).resolve().parents[1];STATUS=ROOT/'notes/track3-foundation-stereo-prepare-status.json'


def main():
    save(STATUS,{'state':'running','started_utc':now()})
    meta=json.loads((ROOT/'weights/model_catalog/c-foundationstereo-s.json').read_text())
    if meta.get('gated'):raise RuntimeError('This downloader only uses ungated public resources')
    name='onnx/deployable_foundationstereo_small_576x960_v2.0.onnx'
    item=next(e for e in meta['siblings'] if e['rfilename']==name)
    expected=item['lfs']['sha256'];size=item['size']
    if size>1024**3:raise RuntimeError('Model exceeds this diagnostic download budget')
    out=ROOT/'weights/foundation_stereo';out.mkdir(exist_ok=True)
    path=out/Path(name).name;url=f'https://huggingface.co/nvidia/c-foundationstereo-s/resolve/{meta["sha"]}/{name}'
    if not path.exists():
        part=path.with_name(path.name+'.part')
        print(now(),'download',name,size,flush=True)
        subprocess.run(['curl','--fail','--location','--silent','--show-error','--proto','=https','--proto-redir','=https',
                        '--retry','3','--connect-timeout','20','--max-time','900','--max-filesize',str(size),
                        '--continue-at','-','--output',str(part),url],check=True)
        if part.stat().st_size!=size or digest(part)!=expected:raise RuntimeError('Public model validation failed')
        part.replace(path)
    if path.stat().st_size!=size or digest(path)!=expected:raise RuntimeError('Cached model validation failed')
    save(out/'source.json',{'repository':'nvidia/c-foundationstereo-s','revision':meta['sha'],'url':url,
                            'sha256':expected,'size_bytes':size,'license':'nvidia-open-model-license','verified_utc':now()})
    print(now(),'model SHA256 verified',flush=True)
    deps=ROOT/'.deps/stereo-py311';deps.mkdir(exist_ok=True)
    image='sha256:367f7ab879e29469b7d222635f5797f962906ef0afd01e2586556850cc9aec32'
    common=['docker','run','--rm','--pull=never','--read-only','--tmpfs','/tmp:rw,exec,size=2g','--user',f'{os.getuid()}:{os.getgid()}',
            '-e','PYTHONPATH=/deps','-e','PIP_DISABLE_PIP_VERSION_CHECK=1','--mount',f'type=bind,src={deps},dst=/deps','--entrypoint','python',image]
    requirements=['onnxruntime-gpu==1.23.2','onnx==1.19.1','flatbuffers==25.9.23','coloredlogs==15.0.1','humanfriendly==10.0','ml-dtypes==0.5.3','protobuf==5.29.5']
    if not (deps/'install-report.json').exists():
        subprocess.run(common+['-m','pip','install','--no-cache-dir','--no-deps','--target','/deps','--report','/deps/install-report.json']+requirements,check=True)
    if not (deps/'google/protobuf').is_dir():
        subprocess.run(common+['-m','pip','install','--no-cache-dir','--no-deps','--target','/deps',
                               '--report','/deps/protobuf-install-report.json','protobuf==5.29.5'],check=True)
    check="import json,onnx,onnxruntime; print(json.dumps({'onnx':onnx.__version__,'ort':onnxruntime.__version__,'providers':onnxruntime.get_available_providers()}))"
    subprocess.run(common+['-c',check],check=True)
    save(STATUS,{'state':'completed','completed_utc':now(),'model':str(path),'model_sha256':expected,
                 'requirements':requirements,'image_id':image,'dependency_directory':str(deps),'training':False})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        save(STATUS,{'state':'failed','updated_utc':now(),'error_type':type(exc).__name__});traceback.print_exc();raise SystemExit(1)
