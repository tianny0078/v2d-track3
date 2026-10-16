"""Create the offline result page and check every local reference and data buffer."""
import base64,json,re,zipfile
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote

ROOT=Path(__file__).resolve().parents[1];DIST=ROOT/'sites/track3-results-20261015/dist'
archive=DIST/'downloads/track3-results-offline.zip'
readme='V2D Track 3 实验结果快照\n\n解压整个文件夹后，用现代浏览器打开 index.html。\n图像、摘要、PLY 下载和三维网格使用随包资源，无需启动服务器。\n三维预览需要浏览器支持 WebGL 2；原始 PLY 也可用三维软件打开。\n这是公开 episode 11 的实验快照；局部网格未封闭，未完成评测提交或策略训练。\n'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for path in sorted(DIST.rglob('*')):
        if path.is_file() and path!=archive:z.write(path,str(path.relative_to(DIST)))
    z.writestr('README.zh.txt',readme)
assert archive.stat().st_size<25*1024**2,'Offline ZIP exceeds static hosting file budget'
class References(HTMLParser):
    refs=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ['src','href'] and value:self.refs.append(value)
p=References();p.feed((DIST/'index.html').read_text())
for ref in p.refs:
    if ref.startswith(('#','https:','data:')):continue
    local_ref=unquote(urlsplit(ref).path)
    assert (DIST/local_ref).is_file(),f'Missing local asset: {ref}'
data=json.loads((DIST/'assets/data.js').read_text()[len('window.V2D_RESULTS='):-2])
for frame in data['keyframes']:
    for kind in ['rgb','sgbm','learned','cups','hands']:assert (DIST/f'assets/{frame["prefix"]}-{kind}.jpg').exists()
models=json.loads((DIST/'assets/mesh-data.js').read_text()[len('window.V2D_MESHES='):-2])
for name,m in models.items():
    assert len(base64.b64decode(m['positions']))==m['vertexCount']*3*4
    assert len(base64.b64decode(m['normals']))==m['vertexCount']*3*4
    assert len(base64.b64decode(m['indices']))==m['indexCount']*4
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert 'index.html' in z.namelist() and 'assets/mesh-data.js' in z.namelist()
    assert not any('.git/' in name or '.openai/' in name for name in z.namelist())
    assert 'fetch(' not in z.read('app.js').decode('utf8'),'Offline page should not depend on fetch/file origin access'
print(json.dumps({'checked_html_references':len(p.refs),'models':len(models),'offline_zip_bytes':archive.stat().st_size,'static_files':sum(f.is_file() for f in DIST.rglob('*'))},indent=2))
