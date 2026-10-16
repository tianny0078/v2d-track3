"""Install a checksum-verified Node binary locally for the Sites helpers."""
import hashlib,json,subprocess,tarfile
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'.deps/sites-runtime';OUT.mkdir(exist_ok=True)
VERSION='22.17.0';NAME=f'node-v{VERSION}-linux-x64';node=OUT/NAME/'bin/node'
if node.exists():
    subprocess.run([str(node),'--version'],check=True);raise SystemExit(0)
base=f'https://nodejs.org/dist/v{VERSION}'
subprocess.run(['curl','--fail','--silent','--show-error','--location','--max-time','60',base+'/SHASUMS256.txt','--output',str(OUT/'SHASUMS256.txt')],check=True)
archive=OUT/(NAME+'.tar.xz');expected=next(line.split()[0] for line in (OUT/'SHASUMS256.txt').read_text().splitlines() if line.split()[-1]==archive.name)
subprocess.run(['curl','--fail','--silent','--show-error','--location','--retry','2','--max-time','300','--max-filesize','50000000','--continue-at','-',base+'/'+archive.name,'--output',str(archive)],check=True)
actual=hashlib.sha256(archive.read_bytes()).hexdigest();assert actual==expected,'Node archive checksum mismatch'
with tarfile.open(archive,'r:xz') as tar:
    member=tar.getmember(NAME+'/bin/node');assert member.isfile()
    node.parent.mkdir(parents=True,exist_ok=True);node.write_bytes(tar.extractfile(member).read());node.chmod(0o755)
(OUT/'source.json').write_text(json.dumps({'version':VERSION,'url':base+'/'+archive.name,'sha256':actual,'verified_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
subprocess.run([str(node),'--version'],check=True)
