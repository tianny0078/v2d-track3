"""Check unattended local-agent and Sites access without persisting raw tool output."""
import json
import re
from pathlib import Path
import subprocess
import tempfile
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
schema={'type':'object','properties':{'local_notes_readable':{'type':'boolean'},
        'sites_tool_available':{'type':'boolean'},'site_verified':{'type':'boolean'},
        'reason':{'type':'string'}},'required':['local_notes_readable','sites_tool_available','site_verified','reason'],
        'additionalProperties':False}
schemafile=ROOT/'notes/track3-hourly-capability.schema.json'
schemafile.write_text(json.dumps(schema)+'\n')
prompt='''只做无人值守能力检查，不执行实验，不修改任何源码、网页或笔记，不读取任何凭证文件。
读取 /home/tianyuan/projects/v2d/notes/track3-live-progress.json 确认本地文件可访问。
检查工具清单是否有 Sites get_site。若有，仅调用 get_site(project_id="appgprj_6ad1b84732148191b89784e3cdd08ef4") 验证能访问现有网页，不输出工具完整结果，不输出个人资料、token或credential。
返回结构化JSON，说明local_notes_readable、sites_tool_available、site_verified和简短reason。没有工具则如实返回false，不尝试安装或获取凭证。'''
with tempfile.TemporaryDirectory(prefix='v2d-hourly-probe-') as temp:
    final=Path(temp)/'result.json'
    argv=['codex','exec','--approve-for-me','--skip-git-repo-check',
          '--ephemeral','--cd',str(ROOT),'--output-schema',str(schemafile),'--output-last-message',str(final),'-']
    try:
        result=subprocess.run(argv,input=prompt,text=True,capture_output=True,timeout=180)
        data=json.loads(final.read_text()) if final.exists() else {'reason':'Local agent produced no structured result.'}
        data['exit_code']=result.returncode
        if result.returncode:
            errors=[line for line in result.stderr.splitlines() if any(w in line.lower() for w in ['error:', 'unexpected argument', 'usage:', 'cannot', 'conflict'])]
            data['diagnostic']=re.sub(r'(sk-|token[=: ]+)[A-Za-z0-9_\-]{8,}', '[REDACTED]', '\n'.join(errors)[:1000], flags=re.IGNORECASE)
    except subprocess.TimeoutExpired:
        data={'reason':'Local agent capability check exceeded 180 seconds.','exit_code':124}
    except Exception as exc:
        data={'reason':type(exc).__name__,'exit_code':1}
data['checked_utc']=datetime.now(timezone.utc).isoformat()
(ROOT/'notes/track3-hourly-capability.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(data,ensure_ascii=False,indent=2),flush=True)
