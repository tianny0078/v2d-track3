# 当前进度（2026-10-16 01:07 PDT）

当前只处理公开 episode 11 的 137 帧，尚未处理评分全量、策略训练或真实策略 rollout。

## 已有结果

- 标定、双杯时序分割、双目深度、背景 VO、手跟踪和表面融合已完成。
- 原手轨迹 233 实例；HOT3D 对照 241 实例。缺失手保持未知。
- 程序空心杯与 SAM3D 候选已生成。桌面非穿透全 137 帧通过，留出轮廓 IoU 中位约 0.936；杯间材料相交仍未通过。
- 复合凸碰撞体每杯 64 壁楔加杯底，三种壁厚静置/嵌套通过；Newton SolverMuJoCo CPU 实际 1000 步通过。双杯 URDF 已导入并编译，质量/COM/惯量一致。
- Sharpa 位置 IK 完成 233 实例，指尖拟合中位/P95 约 1.5/4.9 mm，关节越界 0。接触与执行仍待验证。
- CHORD 官方 GPU wrench 内核运行完成，比较 512 共享方向、三档摩擦与距离阈值。它是几何接触能力参考，不是视频测得的力。
- 官方参考接口加载 137 帧/20Hz parquet，CPU 场景 54 bodies、241 shapes、68 DOF 编译通过；该预检没有运行动力学或策略训练。

## 当前门槛

`asset_smoke_pass=true`，`interface_preflight_pass=true`，`training_ready=false`。

1. 杯间材料相交尚未解决。
2. 官方当前 reference 与 runtime 的接触矩原点为 link/body；本资产 COM 偏移约 84 mm，需要统一物理约定。
3. 官方 loader 忽略 unknown contact sidecar，零占位不能当 GT 接触负例。
4. 尚无训练 checkpoint、恢复验证或真实策略 rollout。

## 运行与网页

服务器每小时维护 agent 已实际成功发布至少两轮；会继续核查 tmux/Docker 与落盘结果，更新服务器 notes 和原有私有网页。GitHub 保存经筛选的源代码与进度快照，不包含原始运行记录和媒体。

[固定结果页（保持原访问范围）](https://v2d-track3-results-20261015.sharp-spice-1328.chatgpt.site/)

机器可读摘要：[status.json](status.json)。流程与下一步见 [PIPELINE](../docs/PIPELINE.md)。
