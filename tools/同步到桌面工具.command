#!/bin/zsh
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST_DIR="$HOME/Desktop/照片初筛工具"

mkdir -p "$DEST_DIR"
for file in \
  photo_cull.py \
  photo_cull_server.py \
  photo_cull_vision.swift \
  使用说明.txt \
  分享使用说明.txt \
  照片初筛工具.command \
  首次安装依赖.command
do
  if [[ -f "$SOURCE_DIR/$file" ]]; then
    cp "$SOURCE_DIR/$file" "$DEST_DIR/$file"
  fi
done

SHORTCUT_PATH="$HOME/Desktop/拾片.command"
if [[ ! -e "$SHORTCUT_PATH" && ! -L "$SHORTCUT_PATH" ]]; then
  ln -s "照片初筛工具/照片初筛工具.command" "$SHORTCUT_PATH"
fi

printf 'Mac 运行副本已更新：%s\n' "$DEST_DIR"
printf '桌面启动入口：%s\n' "$SHORTCUT_PATH"
printf '未删除该目录中的其他文件、编译缓存或照片结果。\n'
