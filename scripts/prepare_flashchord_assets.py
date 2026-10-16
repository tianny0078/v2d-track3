"""Materialize only the official floating-hand mesh closure and reference paper."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'vendor/video_to_data'
LANE=REPO/'robotic_grounding/flash_chord'
ASSETS=LANE/'src/flash_chord/assets'
COMMIT=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()


def fetch(path):
    # Read the committed LFS pointer, not a possibly already materialized worktree file.
    rel=path.relative_to(REPO).as_posix()
    committed=subprocess.check_output(['git','-C',str(REPO),'show',COMMIT+':'+rel])
    if not committed.startswith(b'version https://git-lfs.github.com/spec/v1'):
        return {'path':rel,'state':'already_tracked','bytes':path.stat().st_size}
    fields=dict(line.split(' ',1) for line in committed.decode().splitlines()[1:])
    expected=fields['oid'].split(':',1)[1];size=int(fields['size'])
    if path.exists() and path.stat().st_size==size and hashlib.sha256(path.read_bytes()).hexdigest()==expected:
        return {'path':rel,'state':'cached','sha256':expected,'bytes':size}
    url='https://media.githubusercontent.com/media/nvidia-isaac/video_to_data/'+COMMIT+'/'+rel
    partial=path.with_name(path.name+'.download')
    with urllib.request.urlopen(url,timeout=120) as response,partial.open('wb') as out:
        while chunk:=response.read(1024*1024):out.write(chunk)
    assert partial.stat().st_size==size,(rel,partial.stat().st_size,size)
    assert hashlib.sha256(partial.read_bytes()).hexdigest()==expected,rel
    partial.replace(path)
    print('Verified',rel,size,flush=True)
    return {'path':rel,'state':'downloaded','source':url,'sha256':expected,'bytes':size}


def main():
    paths=set()
    for side in ['left','right']:
        urdf=ASSETS/f'urdfs/sharpawave/{side}_sharpawave.urdf'
        for mesh in ET.parse(urdf).findall('.//mesh'):
            path=(urdf.parent/mesh.attrib['filename']).resolve()
            assert path.is_relative_to(ASSETS)
            paths.add(path)
    report={'state':'running','code_commit':COMMIT,'files':[],'training':False}
    target=ROOT/'notes/track3-flashchord-assets.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        report['files']=list(pool.map(fetch,sorted(paths)))
    report.update(state='completed',completed_utc=datetime.now(timezone.utc).isoformat(),
                  total_bytes=sum(f['bytes'] for f in report['files']))
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2),flush=True)


if __name__=='__main__':main()
