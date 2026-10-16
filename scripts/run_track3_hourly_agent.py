"""Server-local hourly agent, retaining sandbox/auto-review and no raw credential logs."""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'notes/track3-hourly-agent-state.json'
STOP=ROOT/'notes/track3-hourly-agent.stop'
parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');args=parser.parse_args()
lock=(ROOT/'notes/.track3-hourly-agent.lock').open('a')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit('Hourly agent already active; no duplicate.')
schema={'type':'object','properties':{'notes_updated':{'type':'boolean'},'site_published':{'type':'boolean'},
        'deployment_id':{'type':['string','null']},'summary':{'type':'string'},'needs_attention':{'type':'boolean'}},
        'required':['notes_updated','site_published','deployment_id','summary','needs_attention'],'additionalProperties':False}
schemafile=ROOT/'notes/track3-hourly-result.schema.json';schemafile.write_text(json.dumps(schema)+'\n')
prompt='执行 /home/tianyuan/projects/v2d/notes/track3-hourly-agent-instructions.md 中的每小时维护任务。先读取该文件，按它核查真实状态、更新notes并发布到既有私有进度网页。不要启动实验，不更改模型或安全设置，不输出任何凭证。'
while not STOP.exists():
    started=datetime.now(timezone.utc);stamp=started.strftime('%Y%m%dT%H%M%SZ')
    next_run=started+timedelta(hours=1)
    state={'state':'running','started_utc':started.isoformat(),'next_run_utc':next_run.isoformat(),
           'interval_seconds':3600,'mechanism':'server tmux + Codex local agent with auto-review',
           'policy':'workspace-write; no sandbox bypass; no raw tool transcripts retained'}
    STATE.write_text(json.dumps(state,indent=2)+'\n')
    print('Hourly run started',stamp,flush=True)
    with tempfile.TemporaryDirectory(prefix='v2d-hourly-') as temp:
        final=Path(temp)/'result.json'
        argv=['codex','exec','--approve-for-me','--skip-git-repo-check','--ephemeral','--cd',str(ROOT),
              '--output-schema',str(schemafile),'--output-last-message',str(final),'-']
        try:
            result=subprocess.run(argv,input=prompt,text=True,capture_output=True,timeout=1800)
            summary=json.loads(final.read_text()) if final.exists() else {'summary':'Agent produced no structured result.','needs_attention':True}
            summary['exit_code']=result.returncode
        except subprocess.TimeoutExpired:
            summary={'summary':'Hourly agent exceeded 30 minutes; next run remains scheduled.','needs_attention':True,'exit_code':124}
        except Exception as exc:
            summary={'summary':type(exc).__name__,'needs_attention':True,'exit_code':1}
    summary.update(started_utc=started.isoformat(),completed_utc=datetime.now(timezone.utc).isoformat())
    dest=ROOT/f'notes/hourly/agent-result-{stamp}.json';dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    state.update(state='waiting',last_result=summary,last_result_file=str(dest))
    STATE.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    if args.once:break
    while not STOP.exists() and datetime.now(timezone.utc)<next_run:
        time.sleep(min(60,max(1,(next_run-datetime.now(timezone.utc)).total_seconds())))
if STOP.exists():
    state.update(state='stopped');STATE.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
