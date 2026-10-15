#!/bin/zsh

set -u

tool_dir="$(cd "$(dirname "$0")" && pwd)"

if [[ -x "$tool_dir/.venv/bin/python" ]]; then
  python_bin="$tool_dir/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1 && python3 -c 'import numpy, PIL' >/dev/null 2>&1; then
  python_bin="$(command -v python3)"
else
  echo "首次使用需要先安装本地运行环境。"
  echo "请关闭此窗口，双击同目录中的“首次安装依赖.command”，完成后再启动本工具。"
  read -k 1 "?按任意键关闭…"
  exit 1
fi

if [[ $# -gt 0 && -d "$1" ]]; then
  photo_dir="$(cd "$1" && pwd)"
else
  selected="$(osascript <<'APPLESCRIPT'
try
  set chosenFolder to choose folder with prompt "选择包含照片的文件夹"
  return POSIX path of chosenFolder
on error number -128
  return ""
end try
APPLESCRIPT
)"
  if [[ -z "$selected" ]]; then
    echo "已取消选择。"
    exit 0
  fi
  photo_dir="${selected%/}"
fi

report_dir="$photo_dir/_初筛结果"

selected_mode="$(osascript <<'APPLESCRIPT'
set choices to {"自动识别（推荐）", "通用", "人像", "风景", "夜景", "运动"}
set picked to choose from list choices with prompt "选择这批照片的筛选策略" default items {"自动识别（推荐）"}
if picked is false then return "auto"
set value to item 1 of picked
if value is "人像" then return "portrait"
if value is "风景" then return "landscape"
if value is "夜景" then return "night"
if value is "运动" then return "sports"
if value is "通用" then return "general"
return "auto"
APPLESCRIPT
)"

echo ""
echo "正在分析：$photo_dir"
echo "照片只在本机处理，原文件不会被上传。"
echo ""

"$python_bin" "$tool_dir/photo_cull.py" "$photo_dir" "$report_dir" --mode "$selected_mode"
if [[ $? -ne 0 ]]; then
  echo ""
  echo "分析失败。请检查目录中是否有支持的照片。"
  read -k 1 "?按任意键关闭…"
  exit 1
fi

port="$("$python_bin" - <<'PY'
import socket
s = socket.socket()
s.bind(('127.0.0.1', 0))
print(s.getsockname()[1])
s.close()
PY
)"

"$python_bin" "$report_dir/photo_cull_server.py" --report-dir "$report_dir" --source-dir "$photo_dir" --port "$port" &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null' EXIT INT TERM
sleep 1

report_url="http://localhost:$port/%E6%89%93%E5%BC%80%E6%9F%A5%E7%9C%8B.html"
open "$report_url"

echo ""
echo "初筛页面已打开。"
echo "结果目录：$report_dir"
echo "关闭这个终端窗口即可停止本地工具；筛选结果不会丢失。"
echo ""

wait "$server_pid"
