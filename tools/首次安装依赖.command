#!/bin/zsh

set -u

tool_dir="$(cd "$(dirname "$0")" && pwd)"

echo "照片初筛工具 · 首次安装"
echo ""

if ! command -v python3 >/dev/null 2>&1; then
  echo "未检测到 Python 3。"
  echo "请先从 https://www.python.org/downloads/macos/ 安装 Python 3，再重新运行本文件。"
  open "https://www.python.org/downloads/macos/"
  read -k 1 "?按任意键关闭…"
  exit 1
fi

echo "正在创建工具专用的本地运行环境……"
python3 -m venv "$tool_dir/.venv" || {
  echo "创建运行环境失败，请确认 Python 3 安装完整。"
  read -k 1 "?按任意键关闭…"
  exit 1
}

echo "正在安装图片处理组件，需要连接网络……"
"$tool_dir/.venv/bin/python" -m pip install --upgrade pip
"$tool_dir/.venv/bin/python" -m pip install numpy pillow
if [[ $? -ne 0 ]]; then
  echo ""
  echo "依赖安装失败，请检查网络后重试。"
  read -k 1 "?按任意键关闭…"
  exit 1
fi

echo ""
echo "安装完成。现在可以双击“照片初筛工具.command”。"
read -k 1 "?按任意键关闭…"
