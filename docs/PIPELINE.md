# Track 3 处理流程

主线为标定双目提供度量几何、手/物体时序重建、接触与 wrench 参考、机器人重定向、仿真训练和真实策略评估。先通过单段完整链路，再扩到全部 Track 3。

| 阶段 | 主要脚本 | 验证重点 |
|---|---|---|
| 数据与标定 | `download_track3_sample.py`, `validate_track3_sample.py`, `prepare_calibrated_sequence.py` | revision/hash、帧索引、相机内参、时间基和坐标约定 |
| 双杯分割 | `run_sam2_temporal.py` | 全段对象身份，保留失败基线 |
| 度量双目与 VO | `run_stereo_sequence.py`, `run_foundation_stereo_sample.py` | 左右一致性、遮挡、有效深度；覆盖率不是准确率 |
| 手跟踪与校正 | `run_wilor_sample.py`, `run_hot3d_hand_pilot.py`, `run_hand_depth_tradeoff.py` | 左右身份、2D/深度/时间的折中、未知可见性 |
| 物体表面与完整几何 | `run_object_surface_tracks.py`, `run_object_surface_fusion.py`, `agent_cup_reconstruct.py`, `run_sam3d_local_sample.py` | 自建度量网格、杯腔、watertight 材料、尺度与注册 |
| 位姿与支撑 | `agent_cup_support_refine.py` | 桌面非穿透、杯间相交、时间连续性与留出诊断 |
| 碰撞体 | `agent_cup_collision_smoke.py`, `agent_cup_collision_urdf.py` | 保留杯腔的复合凸体、惯量、COM、静置与插入 |
| 手部重定向初始化 | `initialize_sharpa_retarget.py` | 官方 Sharpa 运动学、关节范围、拟合与未知手 |
| CHORD 参考 | `probe_contact_wrench.py` | 接触点、向内法向、摩擦锥、共享 512 方向与一次半径归一化 |
| 官方接口 | `agent_reference_preflight.py` | 真实 parquet 加载、frame FK、link/COM、一致坐标与场景编译 |
| 接触语义与 COM 接入 | `track3_masked_contact.py`, `track3_masked_objective.py`, `track3_com_reference.py`, `track3_reference_alignment.py` | 全 known 等价、unknown 屏蔽各分母、左右手与帧序、双 world 真实 GPU 物理/reset；learner 64→80 步真实训练与独立恢复通过，仍未生产就绪 |
| 手桌冲突修复 | `agent_hand_table_reconcile.py`, `agent_hand_table_joint_fit_v2.py`, `agent_hand_table_joint_physics.py` | 原始双目与背景平面交叉核查、TRAIN 选择、重复留出监测、MANO 与机器人碰撞分别检查 |
| 训练、评估、打包 | `agent_baseline_probe.py`，项目 learner 入口继续验证 | 已有 64→80 步诊断、15 个 checkpoint、恢复与真实 272 步策略 rollout；SR0，完整物理参考与评分打包未完成 |

## 坐标与不确定性

原始重建世界系来自首帧相机，不能当作 mocap GT 世界系。仿真前通过拟合桌面建立 Z-up 变换，手腕、物体、接触和机器人 frame 必须同时转换。四元数接口采用显式 `wxyz` 约定。

杯子 URDF link 原点位于杯底 canonical 中心；其 COM 在惯性参数中，偏移约 84 mm。官方当前 wrench 路径在 link/body 原点计算矩，虽然文档称 COM；项目侧 COM 支持视图和 runtime 转换已做对称验证，不能将该视图传给场景/位姿跟踪，也不能只改一侧。mask 对齐当前只允许原始 FPS、speed=1、未缩放物体；不支持的重定时/缩放会明确拒绝。

程序杯的壁厚、杯底、质量和摩擦是假设。网格封闭、碰撞资产可嵌套、观测拟合合格和策略成功分别记录。未知人手或接触保存在 sidecar mask，不作为 GT 负例。

真实 GT 接触当前全部未知，因此安全诊断的接触奖励为零；它验证生产接线而非 force 学习效果。后续几何 proxy 必须单独记录有效性/置信度与来源，不能冒充测得的力或真实接触标签。有限圆桌只是候选：排除了桌外假碰撞，但留出边界仍未合格。局部人手拟合通过不代表形态不同的机器人也无碰撞。

## 方法参考

几何部分参考 AgentSTAR 的代码生成几何与渲染/深度对照；SAM3D 使用现成模型生成候选。重定向与控制参考 CHORD 的接触 wrench 能力，当前未声称完整复现任一论文。
