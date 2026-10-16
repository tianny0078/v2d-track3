# 复现入口与资源

这个仓库包含已运行的实验代码和当前证据摘要。外部权重、数据、许可模型和 Docker 镜像需要在目标服务器准备；部分 HOT3D/SAM3D 入口仍采用当前服务器的共享路径。

## 固定源码与数据

```bash
python3 scripts/bootstrap_track3_workspace.py --init-runtime-dirs
tmux new-session -d -s v2d-t3-source \
  'python3 -u scripts/bootstrap_track3_workspace.py --fetch-sources > logs/source-fetch.log 2>&1'
```

bootstrap 固定官方 commit，并跳过 LFS 模型载荷。已有其他版本时会保留原目录并停止，不重置已有工作。`config/dependencies.lock.json` 同时记录数据 revision 和官方 FlashCHORD lock 哈希。

公开样本下载入口：

```bash
tmux new-session -d -s v2d-t3-data \
  'python3 -u scripts/download_track3_sample.py > logs/data-download.log 2>&1'
```

数据下载器首次运行时读取 Hugging Face 元数据并缓存到 `data/track3_sample/dataset_revision.json`，随后使用该 revision 的固定链接与哈希续传。严格复现本次结果时，需确认缓存的 `sha` 等于依赖记录中的 revision；该历史 revision 不由当前下载器自动回退。

## 环境与已有检查

FlashCHORD 使用项目本地 `.deps/flash-chord-venv`，Python 3.11.15 和官方 `uv.lock`。入口：

```bash
python3 scripts/launch_flashchord_prepare.py
python3 scripts/launch_track3_step.py flashchord-assets
python3 scripts/launch_track3_step.py flashchord-interfaces
```

这些是真实环境准备/GPU基础检查，不是策略训练。已验证环境包括 JAX 0.10.2、Warp 1.14.0、Newton 1.2.0、MuJoCo 3.8.1；硬件为 sm120 NVIDIA Blackwell。

各重建阶段使用不同推理镜像。脚本中的固定 `sha256:` 为已有本机镜像身份，`--pull=never` 不会自动构建它们。运行前需要提供对应镜像，并核对实际挂载路径和模型文件。

HOT3D 和 SAM3D 复用服务器现成资源；不从本仓库分发 MANO 或其他许可模型。原始 HOT3D 仓库保持只读，未原样执行其中删除旧输出的驱动脚本。

## 验证与结果页

接口和质量检查分别保存到 `outputs/track3/episode_000011/`。阶段启动器会读取状态并避免重复已经完成的任务。恢复命令取实际生成的 `notes/*-job.json`，先核查 tmux、Docker 和产物。

完整产物准备好后，结果页构建入口为：

```bash
python3 scripts/build_track3_results_site.py
python3 scripts/update_track3_site_progress.py
python3 scripts/package_track3_offline.py
```

这些命令依赖实际重建产物和服务器进度文件。网站托管身份与凭证不在 GitHub；下载到本地的页面 ZIP 由服务器结果页提供。
