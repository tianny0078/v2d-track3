"""Post-exit capacity and artifact audit; updates only this task's notes fields."""
from pathlib import Path
import json,datetime,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'outputs/track3/episode_000011/masked_physics_integration'
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
def load(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
r=load(out/'report.json');log=ROOT/'logs/track3-masked-physics.log'
assert (ROOT/'logs/track3-masked-physics.exit').read_text().strip()=='0'
assert r['state']=='completed' and 'overflow' not in log.read_text().lower()
z=np.load(out/'checkpoint.npz');assert np.isfinite(z['joint_q']).all() and np.isfinite(z['joint_qd']).all()
assert r['world_physics_steps']==30 and all(any(abs(v)>1e-8 for v in row['reward']) for row in r['rows'])
a=dict(checked_utc=now,exit_code=0,capacity_overflow_count=0,njmax=4096,nconmax=2048,actual_world_physics_steps_accepted=30,rejected_attempt_world_physics_steps=30,intermediate_corrected_attempt_world_physics_steps=30,total_world_physics_steps_this_cycle=90,checkpoint_sha256=hashlib.sha256((out/'checkpoint.npz').read_bytes()).hexdigest(),runtime_validation_pass=True,learner_training_entry_exercised=False,training_ready=False,production_integrated=False)
save(out/'audit.json',a)
r['post_exit_audit']=a;save(out/'report.json',r)
p=ROOT/'notes/track3-masked-physics-job.json';j=load(p);j.update(state='completed',exit_code=0,completed_utc=now,audit=str((out/'audit.json').relative_to(ROOT)),runtime_validation_pass=True);save(p,j)
summary='真实双world GPU RLEnv调用MaskedObjectiveSpec通过：最终30 world物理步、非零姿态reward、unknown四项归零、独立reset与跨帧mask通过。首轮接触容量溢出和全零权重已具体修复；最终日志无overflow。仍未learner训练接入/几何合格。'
p=ROOT/'notes/track3-live-progress.json';live=load(p);live['masked_contact_integration']=dict(state='native_gpu_rl_environment_verified',detail=summary,report=str((out/'report.json').relative_to(ROOT)),job='notes/track3-masked-physics-job.json',checked_at=now,next='校验官方setup_scene原速与mask帧/side顺序，再通过项目训练入口选择配置；随后接手hand/table联合拟合和杯间相交修复。',production_integrated=False,training_ready=False);live['updated_at']=now;save(p,live)
text=f'\n## 续研实做 {now}\n\n{summary}\n\n产物：outputs/track3/episode_000011/masked_physics_integration/{{report.json,audit.json,checkpoint.npz,composed_training.yaml}}；失败首轮保留attempt_00_capacity_overflow。最终30 world步，本轮三次共90 world步。checkpoint为诊断快照，非精确solver恢复。\n\n恢复 `python3 scripts/launch_track3_masked_physics.py`；完成后 `python3 scripts/audit_track3_masked_physics.py`。日志logs/track3-masked-physics.log，job见notes/track3-masked-physics-job.json。配置motion_speed=1保持源20Hz；其他重采样/裁剪尚未支持mask重映射，不得直接使用。全部实际GT contact仍unknown；模型接触反力不是人类实测force。下一步核查setup_scene对齐并接项目learner入口，保持几何门槛false。\n'
p=ROOT/'notes/track3-current-handoff.md';p.write_text(text+'\n'+p.read_text())
p=ROOT/'notes/track3-progress.md';p.write_text(p.read_text()+text)
save(ROOT/'notes/track3-publication-request.json',dict(requested_utc=now,result_paths=[str((out/'report.json').relative_to(ROOT)),str((out/'audit.json').relative_to(ROOT))],reason=summary))
print(json.dumps(a,indent=2))
