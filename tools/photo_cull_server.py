#!/usr/bin/env python3
"""Serve a photo culling report and safely organize confirmed JPG+RAW pairs."""

from __future__ import annotations

import argparse
import base64
import json
import os
import tempfile
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from itertools import combinations
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from photo_cull_memory import (DRAFT_FILE, SNAPSHOT_FILE, build_snapshot, delete_rule,
                               load_memory, load_snapshot, parse_summary, prompt_preferences,
                               read_json as read_memory_json, save_review, summary_prompt, update_rule,
                               write_json as write_memory_json)


CATEGORY_DIRS = {
    "best": "01_保留",
    "review": "02_待确认",
    "reject": "03_待删除",
}
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
RAW_EXTENSIONS = {".arw", ".cr2", ".cr3", ".nef", ".nrw", ".raf", ".dng", ".rw2", ".orf", ".pef", ".srw", ".raw"}
GROUP_FEEDBACK_FILE = "相似分组反馈.json"


def confirmed_snapshot(report_dir: Path) -> dict:
    snapshot = load_snapshot(report_dir)
    if snapshot:
        return snapshot if snapshot.get("active", True) else {}
    log = read_memory_json(report_dir / "上次整理记录.json", {})
    if not log.get("active"):
        return {}
    report = read_memory_json(report_dir / "初筛数据.json", {})
    names = {item.get("name") for item in report.get("photos", [])}
    reverse = {folder: key for key, folder in CATEGORY_DIRS.items()}
    decisions = {}
    for move in log.get("moves", []):
        target = Path(move.get("to", ""))
        if target.name in names and target.parent.name in reverse:
            decisions[target.name] = reverse[target.parent.name]
    return build_snapshot(report_dir, decisions) if names and set(decisions) == names else {}


def ai_photo_advice(report_dir: Path, names: list[str], supplied_key: str = "", mode: str = "api",
                    model: str = "", base_url: str = "", custom_prompt: str = "", include_memory: bool = True) -> dict:
    """Analyze only explicitly selected report previews; never change decisions."""
    if mode not in {"api", "compatible", "codex"}:
        raise ValueError("不支持的模型接入方式")
    if not isinstance(names, list) or not 1 <= len(names) <= 6 or any(not isinstance(name, str) for name in names):
        raise ValueError("请选择 1–6 张照片进行分析")
    data = json.loads((report_dir / "初筛数据.json").read_text(encoding="utf-8"))
    by_name = {item["name"]: item for item in data.get("photos", [])}
    if len(set(names)) != len(names) or any(name not in by_name for name in names):
        raise ValueError("选择的照片不在当前报告中")
    prompt = custom_prompt or (
        "你是摄影选片助手。请用中文分析这些照片，结合画面内容、构图、光线、主体、情绪和重复度给出筛选建议。"
        "先给明确结论（单张：保留/待确认/淘汰；多张：优先保留哪张，其他如何处理），再给每张最多两点可见证据，"
        "最后说明你看不清或无法判断的地方。不要把已有算法分数当成事实，不要编造画面细节。"
        "建议只供人工参考，禁止代替用户做决定。以下是文件名及本地初筛参考：\n" +
        "\n".join(f"{name}：{by_name[name].get('score', '—')}分；{by_name[name].get('reason', '')}" for name in names)
    )
    if include_memory and not custom_prompt:
        try:
            preferences = prompt_preferences({by_name[name].get("scene", "general") for name in names})
        except (OSError, ValueError, json.JSONDecodeError):
            preferences = ""  # A damaged optional Memory file must not disable photo advice.
        if preferences:
            prompt += "\n以下是用户审核过的历史筛选偏好，仅作参考，不能覆盖眼前画面证据：\n" + preferences
    content = [{"type": "input_text", "text": prompt}]
    previews = []
    for name in names:
        thumb = by_name[name].get("thumb", "")
        preview = (report_dir / thumb).resolve()
        if not thumb.startswith("thumbs/") or not preview.is_relative_to(report_dir.resolve()) or not preview.is_file():
            raise ValueError(f"{name} 的预览图不可用")
        previews.append(preview)
        if mode == "codex":
            continue
        image = base64.b64encode(preview.read_bytes()).decode("ascii")
        content.append({"type": "input_text", "text": f"照片：{name}"})
        content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{image}", "detail": "high"})
    if mode == "codex":
        executable = shutil.which("codex")
        if executable is None:
            local_cli = Path.home() / ".local" / "bin" / "codex"
            executable = str(local_cli) if local_cli.is_file() else None
        if executable is None:
            raise ValueError("未找到 Codex CLI。请先安装并运行 codex login。")
        env = os.environ.copy()
        for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
            env.pop(key, None)
        with tempfile.TemporaryDirectory(prefix="photo-cull-ai-") as workspace:
            command = [executable, "exec", "--sandbox", "read-only", "--skip-git-repo-check",
                       "--ignore-rules", "--ephemeral", "--json", "-C", workspace]
            if model and model != "default":
                command += ["--model", model]
            for preview in previews:
                command += ["--image", str(preview)]
            command += ["--", prompt]
            try:
                result = subprocess.run(command, cwd=workspace, env=env, capture_output=True,
                                        text=True, timeout=120, check=False)
            except subprocess.TimeoutExpired:
                raise ValueError("Codex CLI 分析超时，请检查登录和网络状态") from None
        if result.returncode:
            raise ValueError(f"Codex CLI 分析失败：{(result.stderr or result.stdout)[-300:]}")
        messages = []
        for line in result.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "item.completed" and event.get("item", {}).get("type") == "agent_message":
                messages.append(event["item"].get("text", ""))
        advice = messages[-1].strip() if messages else ""
        if not advice:
            raise ValueError("Codex CLI 没有返回可显示的建议")
        return {"advice": advice, "model": model or "Codex 默认模型", "names": names}
    api_key = (os.environ.get("OPENAI_API_KEY", "") if mode == "api" else "") or supplied_key
    api_key = api_key.strip()
    if not api_key:
        raise ValueError("请填写 OpenAI API Key，或在启动本地服务前设置 OPENAI_API_KEY。")
    model = model.strip() or (os.environ.get("PHOTO_CULL_AI_MODEL", "gpt-6-luna") if mode == "api" else "")
    if not model:
        raise ValueError("请填写支持图片输入的模型 ID")
    if mode == "compatible":
        parsed = urlparse(base_url.rstrip("/"))
        if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}):
            raise ValueError("供应商地址必须是 HTTPS，或本机 HTTP 地址")
        if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("供应商地址无效")
        chat_content = [{"type": "text", "text": part["text"]} if part["type"] == "input_text"
                        else {"type": "image_url", "image_url": {"url": part["image_url"]}}
                        for part in content]
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": chat_content}],
                           "max_tokens": 900 if custom_prompt else 600}, ensure_ascii=False).encode("utf-8")
        endpoint = base_url.rstrip("/") + "/chat/completions"
    else:
        body = json.dumps({"model": model, "input": [{"role": "user", "content": content}],
                           "max_output_tokens": 900 if custom_prompt else 600, "store": False,
                           **({"reasoning": {"effort": "none"}} if model == "gpt-6-luna" else {})}, ensure_ascii=False).encode("utf-8")
        endpoint = "https://api.openai.com/v1/responses"
    request = Request(endpoint, data=body,
                      headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=60) as response:
            result = json.load(response)
    except HTTPError as exc:
        try:
            detail = json.load(exc).get("error", {}).get("message", "")
        except Exception:
            detail = ""
        raise ValueError(f"OpenAI 请求失败（HTTP {exc.code}）：{detail[:180]}") from None
    except URLError as exc:
        raise ValueError(f"连接 OpenAI 失败：{exc.reason}") from None
    if mode == "compatible":
        choices = result.get("choices", [])
        advice = str(choices[0].get("message", {}).get("content", "")).strip() if choices else ""
    else:
        advice = "\n".join(part.get("text", "") for item in result.get("output", [])
                           for part in item.get("content", []) if part.get("type") == "output_text").strip()
    if not advice:
        raise ValueError("模型没有返回可显示的建议，请重试")
    return {"advice": advice, "model": model, "names": names}


def test_ai_connection(mode: str, supplied_key: str = "", base_url: str = "") -> str:
    if mode == "codex":
        executable = shutil.which("codex") or str(Path.home() / ".local" / "bin" / "codex")
        if not Path(executable).is_file():
            raise ValueError("未找到 Codex CLI")
        result = subprocess.run([executable, "login", "status"], capture_output=True, text=True, timeout=15)
        if result.returncode:
            raise ValueError("Codex CLI 尚未登录，请先运行 codex login")
        return result.stdout.strip() or "Codex CLI 已登录"
    if mode not in {"api", "compatible"}:
        raise ValueError("不支持的模型接入方式")
    api_key = ((os.environ.get("OPENAI_API_KEY", "") if mode == "api" else "") or supplied_key).strip()
    if not api_key:
        raise ValueError("请先填写 API Key")
    if mode == "api":
        endpoint = "https://api.openai.com/v1/models"
    else:
        parsed = urlparse(base_url.rstrip("/"))
        if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}):
            raise ValueError("供应商地址必须是 HTTPS，或本机 HTTP 地址")
        if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("供应商地址无效")
        endpoint = base_url.rstrip("/") + "/models"
    request = Request(endpoint, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urlopen(request, timeout=15) as response:
            if response.status != 200:
                raise ValueError(f"连接测试失败（HTTP {response.status}）")
    except HTTPError as exc:
        raise ValueError(f"连接测试失败（HTTP {exc.code}）") from None
    except URLError as exc:
        raise ValueError(f"连接测试失败：{exc.reason}") from None
    return "连接成功。图片模型是否可用，请手动分析一张照片确认。"


def choose_photo_directory() -> Path | None:
    if platform.system() != "Darwin":
        raise RuntimeError("当前仅支持在 macOS 上从页面选择目录")
    script = '''
try
  set chosenFolder to choose folder with prompt "选择要切换到的照片文件夹"
  return POSIX path of chosenFolder
on error number -128
  return ""
end try
'''
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "无法打开文件夹选择器")
    selected = result.stdout.strip().rstrip("/")
    if not selected:
        return None
    path = Path(selected).expanduser().resolve()
    if not path.is_dir():
        raise ValueError("选择的照片目录不存在")
    return path


def existing_report_for(selected: Path) -> tuple[Path, Path] | None:
    report_dir = selected if selected.name == "_初筛结果" else selected / "_初筛结果"
    source_dir = report_dir.parent
    if (report_dir / "打开查看.html").is_file() and ((report_dir / "初筛数据.json").is_file() or (report_dir / "手动调整.json").is_file()):
        return source_dir, report_dir
    return None


def start_report_server(report_dir: Path, source_dir: Path, current_report_dir: Path) -> str:
    server_script = current_report_dir / "photo_cull_server.py"
    if not server_script.exists():
        server_script = report_dir / "photo_cull_server.py"
    if not server_script.exists():
        raise RuntimeError("筛选报告缺少本地服务文件，请用最新版工具重新打开")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    server_process = subprocess.Popen(
        [sys.executable, str(server_script), "--report-dir", str(report_dir), "--source-dir", str(source_dir), "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    time.sleep(0.45)
    if server_process.poll() is not None:
        raise RuntimeError("新目录的本地服务启动失败")
    return f"http://localhost:{port}/{quote('打开查看.html')}"


def launch_photo_directory(source_dir: Path, current_report_dir: Path, update_progress) -> str:
    saved_report = existing_report_for(source_dir)
    if saved_report:
        saved_source, saved_output = saved_report
        update_progress(98, "发现已有筛选结果，正在恢复暂存进度", 0, 0)
        analyzer = Path(__file__).with_name("photo_cull.py")
        if analyzer.exists() and (saved_output / "初筛数据.json").exists():
            result = subprocess.run([sys.executable, str(analyzer), str(saved_source), str(saved_output), "--refresh-only"],
                                    capture_output=True, text=True, timeout=120)
            if result.returncode:
                update_progress(99, "页面更新失败，将打开原有筛选结果", 0, 0)
        return start_report_server(saved_output, saved_source, Path(__file__).resolve().parent)
    analyzer = current_report_dir / "photo_cull.py"
    if not analyzer.exists():
        desktop_analyzer = Path.home() / "Desktop" / "照片初筛工具" / "photo_cull.py"
        analyzer = desktop_analyzer if desktop_analyzer.exists() else analyzer
    if not analyzer.exists():
        raise RuntimeError("缺少照片分析程序，请重新运行最新版照片初筛工具")
    report_dir = source_dir / "_初筛结果"
    process = subprocess.Popen(
        [sys.executable, str(analyzer), str(source_dir), str(report_dir), "--mode", "auto"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    last_line = ""
    update_progress(2, "正在准备照片分析", 0, 0)
    assert process.stdout is not None
    for line in process.stdout:
        last_line = line.strip()
        face_match = re.search(r"人脸分析\s+(\d+)\s+张", last_line)
        processed_match = re.search(r"processed\s+(\d+)/(\d+)", last_line)
        cache_match = re.search(r"cache\s+(\d+)/(\d+)", last_line)
        if face_match:
            update_progress(5, f"已完成人脸与主体检测：{face_match.group(1)} 张", 0, int(face_match.group(1)))
        elif processed_match:
            completed, total = int(processed_match.group(1)), int(processed_match.group(2))
            percent = 8 + round(completed / max(total, 1) * 84)
            update_progress(percent, f"正在分析画质、构图与相似内容：{completed}/{total}", completed, total)
        elif cache_match:
            completed, total = int(cache_match.group(1)), int(cache_match.group(2))
            update_progress(94, f"分析完成，正在生成筛选页面：{completed}/{total}", completed, total)
    return_code = process.wait(timeout=30)
    if return_code != 0:
        raise RuntimeError(last_line or "分析失败")
    update_progress(97, "正在启动新目录的本地服务", 0, 0)
    return start_report_server(report_dir, source_dir, current_report_dir)


def find_pair_files(source: Path, name: str) -> list[Path]:
    safe_name = Path(name).name
    if safe_name != name or Path(name).suffix.lower() not in PHOTO_EXTENSIONS:
        raise ValueError(f"无效文件名：{name}")
    stem = Path(name).stem.lower()
    roots = [source, *(source / "筛选完成" / folder for folder in CATEGORY_DIRS.values())]
    matches: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for item in root.iterdir():
            if item.is_file() and item.stem.lower() == stem and item.suffix.lower() in PHOTO_EXTENSIONS | RAW_EXTENSIONS:
                matches.append(item)
    return matches


def write_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def update_group_feedback(report_dir: Path, action: str, names: list[str], focus: str = "") -> dict:
    clean_names = sorted(set(names))
    if action not in {"confirm", "single_best", "split", "merge_pair", "merge_group", "remove_member", "undo"}:
        raise ValueError("未知的分组反馈")
    if action != "undo" and (not 2 <= len(clean_names) <= 30 or any(Path(name).name != name or Path(name).suffix.lower() not in PHOTO_EXTENSIONS for name in clean_names)):
        raise ValueError("分组照片数据无效")
    path = report_dir / GROUP_FEEDBACK_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        payload = {}
    positive = {tuple(sorted(pair)) for pair in payload.get("positive_pairs", []) if len(pair) == 2}
    negative = {tuple(sorted(pair)) for pair in payload.get("negative_pairs", []) if len(pair) == 2}
    single = {tuple(sorted(pair)) for pair in payload.get("single_pairs", []) if len(pair) == 2}
    history = payload.get("history", []) if isinstance(payload.get("history", []), list) else []
    if action == "undo":
        if not history:
            raise ValueError("没有可撤销的归组操作")
        previous = history.pop()
        result = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "positive_pairs": previous.get("positive_pairs", []),
            "negative_pairs": previous.get("negative_pairs", []),
            "single_pairs": previous.get("single_pairs", []),
            "history": history,
            "undone": previous.get("action", "group_feedback"),
        }
        write_json(path, result)
        return result
    history.append({
        "action": action,
        "at": datetime.now().isoformat(timespec="seconds"),
        "positive_pairs": [list(pair) for pair in sorted(positive)],
        "negative_pairs": [list(pair) for pair in sorted(negative)],
        "single_pairs": [list(pair) for pair in sorted(single)],
    })
    history = history[-30:]
    pairs = set(combinations(clean_names, 2))
    if action in {"confirm", "merge_pair", "merge_group", "single_best"}:
        positive.update(pairs)
        negative.difference_update(pairs)
    if action == "single_best":
        single.update(pairs)
    elif action == "split":
        negative.update(pairs)
        positive.difference_update(pairs)
        single.difference_update(pairs)
    elif action == "remove_member":
        if focus not in clean_names or Path(focus).name != focus:
            raise ValueError("待移出照片无效")
        member_pairs = {tuple(sorted((focus, name))) for name in clean_names if name != focus}
        negative.update(member_pairs)
        positive.difference_update(member_pairs)
        single.difference_update(member_pairs)
    result = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "positive_pairs": [list(pair) for pair in sorted(positive)],
        "negative_pairs": [list(pair) for pair in sorted(negative)],
        "single_pairs": [list(pair) for pair in sorted(single)],
        "history": history,
    }
    write_json(path, result)
    return result


def organize(source: Path, decisions: dict[str, str], report_dir: Path) -> dict[str, int]:
    if not decisions or len(decisions) > 10000:
        raise ValueError("筛选决定为空或数量异常")
    target_root = source / "筛选完成"
    for folder in CATEGORY_DIRS.values():
        (target_root / folder).mkdir(parents=True, exist_ok=True)

    plan: list[tuple[Path, Path]] = []
    counts = {"best": 0, "review": 0, "reject": 0, "files": 0}
    for name, category in decisions.items():
        if category not in CATEGORY_DIRS:
            raise ValueError(f"无效分类：{category}")
        files = find_pair_files(source, name)
        if not files:
            raise FileNotFoundError(f"找不到照片组：{name}")
        destination_dir = target_root / CATEGORY_DIRS[category]
        for item in files:
            destination = destination_dir / item.name
            if item.resolve() == destination.resolve():
                continue
            if destination.exists():
                raise FileExistsError(f"目标已存在：{destination}")
            plan.append((item, destination))
        counts[category] += 1

    moved: list[tuple[Path, Path]] = []
    try:
        for old, new in plan:
            shutil.move(str(old), str(new))
            moved.append((old, new))
    except Exception:
        for old, new in reversed(moved):
            if new.exists() and not old.exists():
                shutil.move(str(new), str(old))
        raise
    counts["files"] = len(moved)
    if moved:
        write_json(report_dir / "上次整理记录.json", {
            "active": True,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "moves": [{"from": str(old), "to": str(new)} for old, new in moved],
        })
    return counts


def undo_last_organize(report_dir: Path) -> int:
    log_path = report_dir / "上次整理记录.json"
    if not log_path.exists():
        raise FileNotFoundError("没有可撤销的整理记录")
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    if not payload.get("active"):
        raise ValueError("上次整理已经撤销")
    plan = [(Path(item["from"]), Path(item["to"])) for item in payload.get("moves", [])]
    for old, new in plan:
        if not new.exists():
            raise FileNotFoundError(f"待恢复文件不存在：{new.name}")
        if old.exists():
            raise FileExistsError(f"原位置已有同名文件：{old}")
    restored: list[tuple[Path, Path]] = []
    try:
        for old, new in reversed(plan):
            old.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(new), str(old))
            restored.append((old, new))
    except Exception:
        for old, new in reversed(restored):
            if old.exists() and not new.exists():
                new.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old), str(new))
        raise
    payload["active"] = False
    payload["undone_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(log_path, payload)
    return len(restored)


def make_handler(report_dir: Path, source_dir: Path):
    switch_jobs: dict[str, dict] = {}
    switch_lock = threading.Lock()
    memory_lock = threading.Lock()

    def update_switch_job(job_id: str, **changes) -> None:
        with switch_lock:
            if job_id in switch_jobs:
                switch_jobs[job_id].update(changes)

    def run_switch_job(job_id: str, selected: Path) -> None:
        try:
            def report(percent: int, phase: str, processed: int, total: int) -> None:
                update_switch_job(job_id, percent=percent, phase=phase, processed=processed, total=total)

            url = launch_photo_directory(selected, report_dir, report)
            update_switch_job(job_id, status="done", percent=100, phase="目录已就绪，正在打开筛选页面", url=url)
        except Exception as exc:
            update_switch_job(job_id, status="error", phase="切换目录失败", error=str(exc))

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(report_dir), **kwargs)

        def log_message(self, format: str, *args) -> None:
            pass

        def send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_html(self, status: int, markup: str) -> None:
            body = markup.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            route = urlparse(self.path).path
            if route not in {"/api/organize", "/api/undo", "/api/decisions", "/api/group-feedback", "/api/switch-directory", "/api/ai-advice", "/api/ai-test", "/api/memory/summarize", "/api/memory/review", "/api/memory/delete", "/api/memory/update"}:
                self.send_json(404, {"ok": False, "error": "接口不存在"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 2_000_000:
                    raise ValueError("请求大小异常")
                payload = json.loads(self.rfile.read(length))
                if route == "/api/ai-advice":
                    self.send_json(200, {"ok": True, **ai_photo_advice(report_dir, payload.get("names"), payload.get("api_key", ""), payload.get("mode", "api"), payload.get("model", ""), payload.get("base_url", ""))})
                elif route == "/api/ai-test":
                    self.send_json(200, {"ok": True, "message": test_ai_connection(payload.get("mode", "api"), payload.get("api_key", ""), payload.get("base_url", ""))})
                elif route == "/api/memory/summarize":
                    with memory_lock:
                        snapshot = confirmed_snapshot(report_dir)
                        if not snapshot:
                            raise ValueError("请先确认筛选并整理，再总结本次偏好")
                        previous = read_memory_json(report_dir / DRAFT_FILE, {})
                        if previous.get("snapshot_id") == snapshot["id"] and not (previous.get("status") == "parse_error" and payload.get("force") is True):
                            self.send_json(200, {"ok": True, "draft": previous, "reused": True})
                        else:
                            names = [item["name"] for item in snapshot.get("samples", [])]
                            answer = ai_photo_advice(report_dir, names, payload.get("api_key", ""), payload.get("mode", "api"), payload.get("model", ""), payload.get("base_url", ""), summary_prompt(snapshot), False)
                            try:
                                draft = parse_summary(answer["advice"], snapshot)
                            except (ValueError, json.JSONDecodeError):
                                draft = {"snapshot_id": snapshot["id"], "created_at": datetime.now().isoformat(timespec="seconds"),
                                         "status": "parse_error", "rules": [], "raw_response": answer["advice"][:4000]}
                            write_memory_json(report_dir / DRAFT_FILE, draft)
                            self.send_json(200, {"ok": True, "draft": draft, "reused": False})
                elif route == "/api/memory/review":
                    with memory_lock:
                        store = save_review(report_dir, payload.get("snapshot_id", ""), payload.get("rules", []))
                    self.send_json(200, {"ok": True, "memory": store})
                elif route == "/api/memory/delete":
                    with memory_lock:
                        store = delete_rule(payload.get("id", ""))
                    self.send_json(200, {"ok": True, "memory": store})
                elif route == "/api/memory/update":
                    with memory_lock:
                        store = update_rule(payload.get("id", ""), payload.get("text", ""))
                    self.send_json(200, {"ok": True, "memory": store})
                elif route == "/api/switch-directory":
                    selected = choose_photo_directory()
                    if selected is None:
                        self.send_json(200, {"ok": True, "cancelled": True})
                    else:
                        job_id = uuid.uuid4().hex
                        with switch_lock:
                            switch_jobs[job_id] = {"status": "running", "percent": 1, "phase": "正在准备照片分析", "processed": 0, "total": 0, "source": str(selected)}
                        threading.Thread(target=run_switch_job, args=(job_id, selected), daemon=True).start()
                        self.send_json(202, {"ok": True, "job": job_id, "source": str(selected), "url": f"/switch-wait?job={job_id}"})
                elif route == "/api/organize":
                    decisions = payload.get("decisions", {})
                    snapshot = build_snapshot(report_dir, decisions, persist=False)
                    counts = organize(source_dir, decisions, report_dir)
                    memory_error = ""
                    try:
                        write_memory_json(report_dir / SNAPSHOT_FILE, snapshot)
                    except Exception as exc:
                        memory_error = f"文件整理已成功，但偏好候选快照未保存：{exc}"
                    self.send_json(200, {"ok": True, "counts": counts, "target": str(source_dir / "筛选完成"), "memory_candidates": len(snapshot["samples"]) if not memory_error else 0, "memory_error": memory_error})
                elif route == "/api/undo":
                    restored = undo_last_organize(report_dir)
                    snapshot = load_snapshot(report_dir)
                    memory_error = ""
                    if snapshot:
                        snapshot["active"] = False
                        try:
                            write_memory_json(report_dir / SNAPSHOT_FILE, snapshot)
                        except Exception as exc:
                            memory_error = f"照片已恢复，但偏好快照状态未更新：{exc}"
                    self.send_json(200, {"ok": True, "restored": restored, "memory_error": memory_error})
                elif route == "/api/group-feedback":
                    feedback = update_group_feedback(report_dir, payload.get("action", ""), payload.get("names", []), payload.get("focus", ""))
                    self.send_json(200, {"ok": True, "feedback": feedback})
                else:
                    decisions = payload.get("decisions", {})
                    overrides = payload.get("overrides", {})
                    progress = payload.get("progress", {})
                    if not isinstance(progress, dict):
                        raise ValueError("暂存进度无效")
                    if progress:
                        if progress.get("filter") not in {"all", "best", "review", "reject", "group"}:
                            raise ValueError("暂存筛选条件无效")
                        if progress.get("sort") not in {"recommended", "score-desc", "score-asc", "composition-desc", "time-desc", "time-asc", "name", "group"}:
                            raise ValueError("暂存排序条件无效")
                        focus_name = progress.get("focusName", "")
                        if not isinstance(focus_name, str) or (focus_name and Path(focus_name).name != focus_name):
                            raise ValueError("暂存位置无效")
                    if len(decisions) > 10000 or len(overrides) > 10000 or any(Path(name).name != name or value not in CATEGORY_DIRS for name, value in {**decisions, **overrides}.items()):
                        raise ValueError("手动筛选数据无效")
                    write_json(report_dir / "手动调整.json", {"updated_at": datetime.now().isoformat(timespec="seconds"), "decisions": decisions, "overrides": overrides, "progress": progress})
                    self.send_json(200, {"ok": True, "saved": len(decisions), "overrides": len(overrides), "progress": bool(progress)})
            except Exception as exc:
                self.send_json(400, {"ok": False, "error": str(exc)})

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            route = parsed.path
            if route in {"/undefined", "/null"}:
                self.send_response(302)
                self.send_header("Location", f"/{quote('打开查看.html')}")
                self.end_headers()
                return
            if route == "/switch-wait":
                job_id = parse_qs(parsed.query).get("job", [""])[0]
                markup = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>正在切换照片目录</title><style>body{margin:0;background:#090c10;color:#f4f7fb;font:15px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{width:min(680px,calc(100% - 40px));margin:18vh auto;padding:28px;border:1px solid #2b3542;border-radius:20px;background:#11161d}h1{margin:0 0 10px;font-size:24px}p{color:#8f9aaa}.track{overflow:hidden;height:12px;margin:24px 0 12px;border-radius:99px;background:#252c36}.bar{width:1%;height:100%;border-radius:inherit;background:linear-gradient(90deg,#79a7ff,#d7ff39);transition:width .25s ease}.row{display:flex;justify-content:space-between;gap:20px}</style></head><body><main><h1>正在打开照片目录</h1><div class="row"><b id="phase">正在准备</b><span id="percent">1%</span></div><div class="track"><div id="bar" class="bar"></div></div><p id="source">已有筛选结果会直接恢复，不会重新分析。</p></main><script>const job=__JOB__;async function poll(){try{const r=await fetch('/api/switch-status?job='+encodeURIComponent(job),{cache:'no-store'});const s=await r.json();if(!r.ok||!s.ok)throw new Error(s.error||'读取进度失败');const p=Math.max(0,Math.min(100,Math.round(Number(s.percent)||0)));document.querySelector('#phase').textContent=s.phase||'正在处理';document.querySelector('#percent').textContent=p+'%';document.querySelector('#bar').style.width=p+'%';document.querySelector('#source').textContent=(s.source||'')+(s.total?' · '+(s.processed||0)+' / '+s.total+' 张':'');if(s.status==='done'){location.replace(s.url);return}if(s.status==='error')throw new Error(s.error||'处理失败');setTimeout(poll,600)}catch(e){document.querySelector('#phase').textContent='切换失败';document.querySelector('#source').textContent=e.message}}poll();</script></body></html>""".replace("__JOB__", json.dumps(job_id))
                self.send_html(200, markup)
                return
            if route == "/api/switch-status":
                job_id = parse_qs(parsed.query).get("job", [""])[0]
                with switch_lock:
                    payload = dict(switch_jobs.get(job_id, {}))
                if not payload:
                    self.send_json(404, {"ok": False, "error": "切换任务不存在或已过期"})
                else:
                    self.send_json(200, {"ok": True, **payload})
                    if payload.get("status") == "done" and not payload.get("shutdown_scheduled"):
                        update_switch_job(job_id, shutdown_scheduled=True)
                        threading.Timer(3.0, self.server.shutdown).start()
                return
            if route == "/api/group-feedback":
                path = report_dir / GROUP_FEEDBACK_FILE
                try:
                    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"positive_pairs": [], "negative_pairs": [], "single_pairs": [], "history": []}
                    self.send_json(200, {"ok": True, "feedback": payload})
                except Exception as exc:
                    self.send_json(400, {"ok": False, "error": str(exc)})
                return
            if route == "/api/memory":
                try:
                    snapshot = confirmed_snapshot(report_dir)
                    draft = read_memory_json(report_dir / DRAFT_FILE, {})
                    if draft.get("snapshot_id") != snapshot.get("id"):
                        draft = {}
                    self.send_json(200, {"ok": True, "snapshot": snapshot, "draft": draft, "memory": load_memory()})
                except Exception as exc:
                    self.send_json(400, {"ok": False, "error": str(exc)})
                return
            if route == "/api/decisions":
                path = report_dir / "手动调整.json"
                try:
                    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"decisions": {}, "overrides": {}, "progress": {}}
                    self.send_json(200, {"ok": True, "decisions": payload.get("decisions", {}), "overrides": payload.get("overrides", {}), "progress": payload.get("progress", {})})
                except Exception as exc:
                    self.send_json(400, {"ok": False, "error": str(exc)})
                return
            super().do_GET()

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=4174)
    args = parser.parse_args()
    report_dir = args.report_dir.expanduser().resolve()
    source_dir = args.source_dir.expanduser().resolve()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(report_dir, source_dir))
    print(f"照片筛选工具：http://localhost:{args.port}/打开查看.html", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
