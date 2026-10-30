# macOS 本地版

本目录是 Mac 本地版的代码主副本。桌面上的 `照片初筛工具` 文件夹是运行/分享副本，可通过 `同步到桌面工具.command` 更新。

- `photo_cull.py`：扫描照片、分析与生成初筛页面。
- `photo_cull_server.py`：本地预览服务。
- `photo_cull_memory.py`、`photo_cull_memory.js`：跨目录偏好的本机存储、摘要审核与报告页交互。
- `photo_cull_vision.swift`：macOS Vision 辅助分析程序源码。
- `照片初筛工具.command`：日常启动入口。
- `首次安装依赖.command`：首次使用时安装依赖。
- `使用说明.txt`、`分享使用说明.txt`：用户说明。

页面顶部统一使用“配置 AI”选择供应商、模型并测试连接，单张照片的“AI 帮我看看”和对比组的“AI 帮我选”共用当前配置。AI 请求按需触发，单次只发送所选照片的报告预览图（对比组最多 6 张）。可用 OpenAI API Key、已登录的官方 Codex CLI，或支持图片输入的 OpenAI Chat Completions 兼容供应商。可保存多个非密钥配置；配置与照片筛选记录分开，API Key 只在当前页面会话保留，也可在启动本地服务前设置 `OPENAI_API_KEY`。测试连接不会发送照片，但兼容供应商若不支持 `/models` 可能提示失败。AI 建议不修改筛选决定。

“确认筛选并整理”成功后，只在报告目录写入一份代表性案例快照，不自动调用 AI。用户另点“选片偏好 → 总结本次偏好”时，才发送最多 6 张报告预览图；同一快照已有草稿会直接复用，防止重复计费。用户审核、修改或拒绝草稿后，已确认规则保存在 `~/Library/Application Support/拾片/筛选偏好.json`，可在页面编辑或删除。后续手动 AI 选片会引用场景匹配的已确认偏好。当前规则样本量不足以安全调整本地评分或自动淘汰，因此本地初筛不根据 Memory 改写分数；报告页可查看偏好。离线测试：`python3 -m unittest discover -s tools -p 'test_*.py'`。

本机编译得到的 `.photo_cull_vision`、Python 缓存和虚拟环境不属于源代码，不进入版本管理。
