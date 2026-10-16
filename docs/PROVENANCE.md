# 来源与版本

第三方库的完整源码通过 bootstrap 获取。项目侧部分接触/训练衍生模块包含固定版本 NVIDIA 代码，保留其 SPDX 声明，修改与许可文本见 [第三方说明](../THIRD_PARTY_NOTICES.md)。模型和数据不随仓库分发。

| 来源 | 使用方式 |
|---|---|
| [NVIDIA video_to_data](https://github.com/nvidia-isaac/video_to_data) | 固定 commit `193382a184308f6800331f3db829f76dc4294cfe`；重建、Sharpa/接触映射和 FlashCHORD 接口 |
| [公开挑战数据](https://huggingface.co/datasets/nvidia/video_to_data_challenge) | 固定 revision `5f68335f3acc802033d1e80728c1633197521de8` 的 episode 11；数据不打包 |
| [CHORD](https://nvidia-isaac.github.io/video_to_data/chord/) | 接触点、法向、摩擦锥与 wrench 能力参考 |
| [AgentSTAR](https://github.com/makezur/agenticSTAR) | 代码生成几何、位姿与渲染对照思路；并非完整复现 |
| [SAM3D Objects](https://github.com/facebookresearch/sam-3d-objects) | 服务器现成权重与镜像的离线双杯候选；不分发权重 |
| [TripoSR](https://github.com/VAST-AI-Research/TripoSR) | 固定 commit `107cefdc244c39106fa830359024f6a2f1c78871`；保留未采纳的形状基线 |

项目新增实验、包装和修复代码与原作者代码按文件声明区分；导入的官方库、衍生模块中的原代码、借鉴的官方映射和第三方算法归属原作者。部分运行路径依赖服务器的 HOT3D 实现与许可 MANO 资源；这些源码和模型未复制到公共仓库。

`config/source-manifest.json` 记录发布时各源文件的 SHA256。它对应代码快照，不替代数据/模型的运行时下载清单。
