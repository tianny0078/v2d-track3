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
                       ('保留杯腔的碰撞体物理检查','outputs/track3/episode_000011/agent_cup_collision_smoke/report.json'),
                       ('COM与未知接触语义探针','outputs/track3/episode_000011/wrench_origin_semantics/report.json'),
                       ('官方基线链路诊断','outputs/track3/episode_000011/baseline_probe/report.json'),
                       ('持续推进器','notes/track3-continuation-state.json'),
                       ('Kaggle六指标入赛','notes/track3-kaggle-entry.json')]:
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
        counts=r.get('threshold_sensitivity',[]);entry['detail']='官方GPU wrench内核运行完成；5/10/20mm距离阈值对应 '+ '/'.join(str(v['active_link_count']) for v in counts)+' 个临近关节对。对齐误差敏感；该接触参考尚未集成生产训练。'
    elif title=='保留杯腔的碰撞体物理检查':
        newton=r.get('newton',{});entry['detail']='每杯64个凸壁楔加杯底；三种假设壁厚静置/嵌套检查完成。同轴最大采样插入164/144/118mm。Newton CPU实际'+str(newton.get('actual_newton_solver_steps',0))+'步，静置检查'+('通过' if newton.get('rest_gate') else '待通过')+'；尚未验证整段轨迹或GPU批量。'
    elif title=='COM与未知接触语义探针':
        entry['detail']='参考/runtime两侧同时转COM支持误差小于4.8e-7，刚体不变性检查通过；unknown必须排除归一化分母。生产环境/奖励接入未完成，初态物理质量仍未通过。'
    elif title=='官方基线链路诊断':
        entry['detail']='GPU3小规模64→80步训练、checkpoint、独立恢复与真实policy rollout'+('链路通过' if r.get('policy_rollout_pass') else '执行中')+'。当前短训练成功率0；不能据80步判断算法能力，也不是Kaggle得分。'
    elif title=='持续推进器':
        entry['detail']=f"服务器顺序推进第{r.get('cycle',0)}轮，状态{r.get('state','unknown')}；每轮保存结果后自动接续下一步并触发本页更新。GPU实验是否运行以各job核查为准。最近心跳：{r.get('heartbeat_utc','未知')}。"
    else:
        entered=sum(c.get('entered') is True for c in r.get('competitions',[]))
        entry.update(state='completed' if r.get('all_six_entered') else 'pending',detail=f'{entered} / 6指标已加入。按服务器现有凭证只读核查；'+('六项规则接受已确认，无需重复操作。' if r.get('all_six_entered') else '需本人同一账号在浏览器Join/Accept。')+'官网每队一次报名为独立流程，表单状态未核实。未上传。')
        labels={'add_auc':'AUC','spider_sr':'SP-SR','maniptrans_sr':'MP-SR','rpe_cm':'RPE','mppe_cm':'MPPE','cd_o_cm':'CD-O'}
        entry['links']=[{'label':labels[c['metric']],'url':c['rules_url']} for c in r.get('competitions',[]) if c.get('metric') in labels and c.get('rules_url','').startswith('https://www.kaggle.com/competitions/v2d-challenge-track3-')]
        entry['links'].append({'label':'官网团队报名','url':'https://docs.google.com/forms/d/e/1FAIpQLSdZJYNsEPPGDeIRH2yb_Dui-lWcIxWRF2CON7UOIijzCw8zyA/viewform'})
    experiments.append(entry)
for title, relative, detail in [
    ('手桌坐标与原始双目交叉核查', 'outputs/track3/episode_000011/hand_table_reconcile/report.json', '18帧核查完成；坐标一致性和双目/桌背景交叉检查通过。参考未修复，几何gate未通过；未自动替换手参考。'),
    ('Masked接触奖励内核', 'outputs/track3/episode_000011/masked_contact_kernels/report.json', '四项Warp内核及批量/reset gather验证通过；全known误差0，混合unknown误差2.384e-7。生产入口尚未接线，无新增物理步；training_ready=false。')]:
    path=ROOT/relative
    if path.exists():
        report=json.loads(path.read_text())
        experiments.append({'name':title, 'state':report.get('state','component_verified'), 'detail':detail})
objective_path=ROOT/'outputs/track3/episode_000011/masked_objective_integration/report.json'
if objective_path.exists():
    objective=json.loads(objective_path.read_text())
    experiments.append({'name':'MaskedObjectiveSpec组件验证','state':'component_verified' if objective.get('objective_spec_factory_verified') else 'pending','detail':'工厂及完整evaluator验证通过，全known误差0、unknown/NaN隔离通过；新增物理步0。生产入口未集成，真实多world step/reward/reset待验证；training_ready=false。'})
for title, relative, detail in [
    ('双world真实GPU奖励诊断', 'masked_physics_integration', '2 world、3控制步、15 solver调用/30 world物理步；独立reset、时序mask gather、COM和原contact保留通过。runtime环境接线通过，learner入口未验证；真实GT接触全unknown，production_integrated=false、training_ready=false；不是策略训练。'),
    ('手桌联合拟合候选', 'hand_table_joint_fit', '27训练帧/15留出帧；留出穿桌18.27mm，图像/几何gate失败。未替换参考、未重建机器人或重跑物理；training_ready=false。'),
    ('初始未观测机器人规划', 'initial_unobserved_robot_plan', '仅改首次观测前0至4帧机器人右腕规划候选；300步native物理smoke，穿透gate失败。未推断未观测人手动作，未自动采用；training_ready=false。')]:
    report_path=ROOT/'outputs/track3/episode_000011'/relative/'report.json'
    if report_path.exists():
        report=json.loads(report_path.read_text())
        experiments.append({'name':title, 'state':report.get('state','unknown'), 'detail':detail})
for title, relative, detail in [
    ('参考帧序与左右手映射验证', 'reference_alignment', '官方setup_scene双world CPU构建与137帧对齐通过，位置/旋转误差0；左右手映射[1,0]、88帧裁剪与拒绝不安全转换通过。未运行learner，无新增物理步，training_ready=false。'),
    ('手桌v2候选native物理检查', 'hand_table_joint_fit/v2_keypoint_balanced/physics', '28/86两帧、80步native物理检查完成；刚体FK匹配及有限状态通过，穿透门槛失败。整段候选未采纳，未替换生产参考；不是策略rollout，training_ready=false。')]:
    report_path=ROOT/'outputs/track3/episode_000011'/relative/'report.json'
    if report_path.exists():
        report=json.loads(report_path.read_text())
        state='failed' if report.get('gates',{}).get('all_candidate_penetrations_under_1mm') is False else report.get('state','unknown')
        experiments.append({'name':title,'state':state,'detail':detail})
safe['experiments']=experiments
if (ROOT/'notes/track3-hourly-agent-job.json').exists():
    safe['hourly_updates']='每小时由服务器agent核查并更新；持续推进器每步完成会触发额外发布。'
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
