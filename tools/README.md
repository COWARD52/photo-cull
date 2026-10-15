# macOS 本地版

本目录是 Mac 本地版的代码主副本。桌面上的 `照片初筛工具` 文件夹是运行/分享副本，可通过 `同步到桌面工具.command` 更新。

- `photo_cull.py`：扫描照片、分析与生成初筛页面。
- `photo_cull_server.py`：本地预览服务。
- `photo_cull_vision.swift`：macOS Vision 辅助分析程序源码。
- `照片初筛工具.command`：日常启动入口。
- `首次安装依赖.command`：首次使用时安装依赖。
- `使用说明.txt`、`分享使用说明.txt`：用户说明。

本机编译得到的 `.photo_cull_vision`、Python 缓存和虚拟环境不属于源代码，不进入版本管理。
