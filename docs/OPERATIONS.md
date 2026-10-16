# 服务器运行、日志与恢复

所有长时间下载、安装、实验和训练放在服务器 tmux 或调度器中。日志、退出码、checkpoint/缓存和恢复入口保存在项目持久目录。

启动前先运行 `python3 scripts/audit_track3_runtime.py`，查看实际 tmux/Docker、退出记录和输出状态。阶段完成后 tmux 会话正常退出，不能据旧笔记宣称任务仍在运行。

## 每小时维护

维护入口为 `scripts/launch_track3_hourly_agent.py`。现有服务器会话是：

```bash
tmux attach -t v2d-t3-hourly-progress-20261015-231826
```

在原实验工作区恢复：

```bash
python3 scripts/launch_track3_hourly_agent.py
```

启动器会拒绝重复活动会话；如果存在主动停止标记，需明确恢复后处理该标记。它每小时启动一次本地 Codex 维护 agent，核查真实状态、更新服务器 notes 和既有私有网页。当前服务器已验证至少两轮发布成功。

每轮输出为 `notes/hourly/` 的摘要；原始工具对话与凭证不写入仓库。`notes/track3-hourly-agent-state.json` 保存下一次时间，`notes/track3-hourly-agent-job.json` 保存会话、日志和恢复命令。这些运行文件不随公共仓库发布。

SSH 断开后 tmux 任务继续；服务器重启后的恢复使用持久缓存/checkpoint 和任务记录。每小时维护只报告和发布，不自动启动实验。

## GitHub 同步

当前实际实验在原项目目录运行，GitHub checkout 独立位于它的 `repos/v2d-track3/`。导出前核查目标仓库和已有更改：

```bash
python3 scripts/export_track3_repository.py --destination repos/v2d-track3 --dry-run
python3 scripts/export_track3_repository.py --destination repos/v2d-track3
```

导出器只复制 Python 源码和四个网页模板，并写源文件哈希。文档与筛选后的进度摘要单独维护；不复制原始 notes、数据、权重或网站运行资产。Git commit/push 由明确的发布步骤执行，每小时维护没有 GitHub 自动推送权限。
