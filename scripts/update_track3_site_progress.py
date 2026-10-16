"""Export only explicitly selected, non-secret progress to the fixed result Site."""
import json
import re
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT/'sites/track3-results-20261015/dist'
data = json.loads((ROOT/'notes/track3-live-progress.json').read_text())
data['updated'] = datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M:%S PDT')
envfile = ROOT/'notes/track3-flashchord-environment.json'
if envfile.exists():
    env = json.loads(envfile.read_text())
    data['environment_state'] = env['state']
    env_stage = next((s for s in data['milestones'] if s.get('id') == 'environment'), None)
    if env['state'] == 'completed':
        if env_stage: env_stage.update(state='completed', detail='官方固定依赖安装及 JAX / Warp / Newton GPU 基础检查均通过。')
    elif env['state'] == 'failed':
        if env_stage: env_stage.update(state='failed', detail='环境检查未通过，正在根据日志修复；尚未开始策略训练。')
safe = {k: data[k] for k in ['updated', 'current_stage', 'state', 'detail', 'strategy', 'milestones', 'limitations']}
experiments=[]
for title,relative in [('CHORD GPU接口验证','notes/track3-flashchord-interfaces.json'),
                       ('HOT3D手跟踪对照','outputs/track3/episode_000011/hot3d_hand_tracking/report.json'),
                       ('代码生成空心杯','outputs/track3/episode_000011/agent_cup_reconstruction/report.json'),
                       ('SAM3D度量深度候选','outputs/track3/episode_000011/object_meshes_sam3d_local/report.json'),
                       ('SAM3D双目注册','outputs/track3/episode_000011/sam3d_metric_registration/report.json'),
                       ('杯体位姿修正审计','outputs/track3/episode_000011/agent_cup_pose_refinement/report.json'),
                       ('桌面与时间窗口约束','outputs/track3/episode_000011/agent_cup_pose_supported/report.json'),
                       ('机器人手IK初始化','outputs/track3/episode_000011/sharpa_keypoint_initializer_v2/report.json'),
                       ('CHORD接触与wrench敏感性','outputs/track3/episode_000011/contact_wrench_probe/report.json'),
                       ('保留杯腔的碰撞体物理检查','outputs/track3/episode_000011/agent_cup_collision_smoke/report.json')]:
    path=ROOT/relative
    if not path.exists():continue
    r=json.loads(path.read_text());entry={'name':title,'state':r.get('state','unknown')}
    if title=='CHORD GPU接口验证':entry['detail']=str(r.get('counts',{}).get('tests',0))+'项实际GPU检查通过；不等于策略成功。'
    elif title=='HOT3D手跟踪对照':entry['detail']=f"{len(r.get('frames',[]))}帧，{r.get('total_hands',0)}手实例，{r.get('two_hand_frames',0)}帧双手。保留tracked-only标记，覆盖率不代表准确率。"
    elif title=='代码生成空心杯':
        evaluation=next((e for e in r.get('evaluations',[]) if e.get('variant')=='initial' and abs(e.get('wall_thickness_m',0)-.003)<1e-7),{})
        entry['detail']='六个空心网格均封闭；3mm壁厚候选留出IoU '+format(evaluation.get('median_heldout_iou',0),'.3f')+'。位姿与嵌套检查未通过，正在修正；壁厚为假设。'
    elif title=='SAM3D度量深度候选':entry['detail']=str(sum(o.get('state')=='completed' for o in r.get('objects',{}).values()))+' / 2杯候选已生成；复用本机权重与标定双目点图。渲染可见杯腔和标记点，仍需注册与碰撞验证。'
    elif title=='SAM3D双目注册':
        values=[o['selected']['silhouette_iou_occlusion_aware'] for o in r.get('objects',{}).values() if 'selected' in o]
        entry['detail']='首帧拟合IoU '+(' / '.join(format(x,'.3f') for x in values) or '计算中')+'；蓝杯深度大尾部、米杯悬空未解决，不自动采用。'
    elif title=='杯体位姿修正审计':
        ha=r.get('summaries',{}).get('heldout_interpolated',{}).get('after',{})
        entry.update(state='failed',detail='留出IoU '+format(ha.get('silhouette_iou_median',0),'.3f')+'，但桌面穿透最坏14.5mm及杯体交叉未通过；保留失败版，正在补桌面约束。')
    elif title=='桌面与时间窗口约束':
        entry['detail']='137帧桌面硬约束通过，24/25时间窗口接受；留出IoU 0.936。杯间材料交叉仍未通过，尚未作为物理参考。'
    elif title=='机器人手IK初始化':
        e=r.get('fingertip_error_p50_p95_m',[0,0]);entry['detail']=f"{r.get('completed_hand_instances',0)}手实例，指尖拟合中位/P95 {e[0]*1000:.1f}/{e[1]*1000:.1f}mm，关节越界0。仅位置IK，接触优化与执行尚未验证。"
    elif title=='CHORD接触与wrench敏感性':
        counts=r.get('threshold_sensitivity',[]);entry['detail']='官方GPU wrench内核运行完成；5/10/20mm距离阈值对应 '+ '/'.join(str(v['active_link_count']) for v in counts)+' 个临近关节对。对齐误差敏感，属于几何接触能力参考，尚未训练。'
    else:
        newton=r.get('newton',{});entry['detail']='每杯64个凸壁楔加杯底；三种假设壁厚静置/嵌套检查完成。同轴最大采样插入164/144/118mm。Newton CPU实际'+str(newton.get('actual_newton_solver_steps',0))+'步，静置检查'+('通过' if newton.get('rest_gate') else '待通过')+'；尚未验证整段轨迹或GPU批量。'
    experiments.append(entry)
safe['experiments']=experiments
if (ROOT/'notes/track3-hourly-agent-job.json').exists():
    safe['hourly_updates']='每小时由服务器agent核查并更新；步骤完成后也会发布。'
    statepath=ROOT/'notes/track3-hourly-agent-state.json'
    if statepath.exists():
        state=json.loads(statepath.read_text());nextrun=state.get('next_run_utc')
        if nextrun:safe['hourly_updates']+=' 下次计划 '+datetime.fromisoformat(nextrun).astimezone(ZoneInfo('America/Los_Angeles')).strftime('%m-%d %H:%M PDT')+'。'
(SITE/'assets/progress-data.js').write_text('window.V2D_PROGRESS='+json.dumps(safe, ensure_ascii=False, separators=(',', ':'))+';\n')
(SITE/'downloads/progress-summary.json').write_text(json.dumps(safe, ensure_ascii=False, indent=2)+'\n')
index = SITE/'index.html'
html = index.read_text()
html = re.sub(r'assets/progress-data\.js(?:\?v=[^"\s]+)?', 'assets/progress-data.js?v='+datetime.now().strftime('%Y%m%d%H%M%S'), html)
index.write_text(html)
print(json.dumps({'updated': safe['updated'], 'state': safe['state'], 'milestones': len(safe['milestones'])}, ensure_ascii=False))
