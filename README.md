# V2D Track 3

NVIDIA Video-to-Data Challenge Track 3 的单轨迹重建、接触参考与机器人接口实验。

当前先完成公开 `episode_000011` 的完整链路，再处理评分数据。执行优先级为 Track 3 → Track 2 → Track 1；本仓库只包含 Track 3 工作。

**已完成短训练、checkpoint 恢复和真实策略 rollout；本地成功率为 0，完整物理参考仍未通过。** 公开 episode 11 不在 20 段评分名单中；80 步诊断不能代表充分训练性能或 Kaggle 得分。最新结果见 [进度摘要](notes/STATUS.md)。

| 已实现 | 当前证据 |
|---|---|
| 数据、标定、分割、双目深度、相机 VO | 同一公开序列 137 帧；记录版本、坐标约定与失败对照 |
| 手部跟踪和校正候选 | WiLoR 233 个手实例；HOT3D 对照 241 个实例；缺失手保留未知 |
| 物体几何 | 多帧 TSDF、程序生成空心杯、SAM3D 候选；完整几何质量仍未通过 |
| 空心碰撞体 | 每杯 64 个凸壁楔与杯底；MuJoCo / Newton CPU 嵌套与静置检查通过 |
| Sharpa 位置 IK | 233 个可见手实例，指尖拟合中位/P95 约 1.5/4.9 mm；尚未验证执行 |
| CHORD 接触参考 | 实际 GPU wrench 内核与距离/摩擦敏感性分析；几何能力不等于测得的真实力 |
| 官方参考/场景接口 | 137 帧 parquet 加载与 CPU 单场景编译通过；训练门槛仍未通过 |

当前在修正手掌穿桌、杯间材料相交并接入项目侧 COM/unknown 奖励。已通过双 world GPU 物理与奖励验证，learner 入口的 64→80 步短训练与独立恢复也已通过。服务器顺序推进器完成一轮后自动接续，步骤交接触发网页发布；每小时仍独立核查。

## 使用

```bash
git clone https://github.com/tianny0078/v2d-track3.git
cd v2d-track3
python3 scripts/bootstrap_track3_workspace.py
python3 scripts/bootstrap_track3_workspace.py --init-runtime-dirs
```

第一条 bootstrap 只打印依赖和目录计划。固定源代码版本见 [依赖记录](config/dependencies.lock.json)；获取源码、下载、安装和实验应在服务器 tmux 中运行，见 [复现步骤](docs/REPRODUCE.md) 与 [运行及恢复](docs/OPERATIONS.md)。

现有推理使用服务器已经安装的 Docker 镜像、模型和部分共享路径。仓库保存的是实际实验脚本；其他服务器需要配置相应资源，不能直接把已有镜像 ID 当作可下载的镜像标签。

## 目录

- `scripts/`：实际使用的下载、校准、重建、几何、接触、IK、场景预检和持久任务脚本。
- `docs/`：处理流程、复现入口、坐标与质量约定、任务恢复方式。
- `config/`：固定第三方版本与源文件校验清单。
- `notes/`：经过筛选的进度摘要；完整运行日志和任务记录保留在服务器。
- `sites/track3-results-20261015/dist/`：结果页 HTML/CSS/JS 模板；数据和图像由服务器产物构建。

服务器实验工作区与 GitHub 提交目录分开维护。使用 `scripts/export_track3_repository.py` 显式导出代码；该脚本不会提交、推送或删除文件。

数据、模型权重、MANO、checkpoint、原始/生成媒体、网站访问凭证与服务器认证配置不随仓库发布。第三方库通过固定版本另行获取；项目侧衍生模块保留原声明，见 [来源](docs/PROVENANCE.md) 和 [第三方说明](THIRD_PARTY_NOTICES.md)。

挑战说明：[NVIDIA V2D Challenge](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/)。
