# 结果页模板

`dist/` 中保存 HTML、CSS 和交互 JavaScript。真实图像、几何二进制、下载文件与进度 JS 由服务器构建脚本生成，未发布到 GitHub。

在包含完整实验产物的项目根目录运行 `scripts/build_track3_results_site.py`、`scripts/update_track3_site_progress.py` 和 `scripts/package_track3_offline.py`。生成的离线 ZIP 可直接打开，不依赖 CDN。

现有网页保持原私有访问范围；本目录不包含托管项目身份、Git remote、登录凭证或网站部署授权。
