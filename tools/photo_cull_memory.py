"""User-reviewed, local-only preferences for the Mac photo culling tool."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path


SNAPSHOT_FILE = "偏好候选快照.json"
DRAFT_FILE = "偏好总结草稿.json"
SCENES = {"general", "portrait", "landscape", "night", "sports"}
DIMENSIONS = {"composition", "sharpness", "expression", "lighting", "variety", "subject", "other"}


def memory_path() -> Path:
    override = os.environ.get("PHOTO_CULL_MEMORY_PATH")
    return Path(override).expanduser() if override else Path.home() / "Library/Application Support/拾片/筛选偏好.json"


def read_json(path: Path, fallback: dict) -> dict:
    if not path.is_file():
        return fallback
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"数据格式无效：{path.name}")
    return value


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=".photo-memory-", delete=False) as handle:
        temp = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temp.replace(path)


def load_memory() -> dict:
    value = read_json(memory_path(), {"version": 1, "rules": []})
    if not isinstance(value.get("rules"), list):
        raise ValueError("本地偏好文件格式无效")
    return value


def build_snapshot(report_dir: Path, decisions: dict[str, str], persist: bool = True) -> dict:
    report = read_json(report_dir / "初筛数据.json", {})
    photos = report.get("photos", [])
    by_name = {p["name"]: p for p in photos if isinstance(p, dict) and isinstance(p.get("name"), str)}
    if not by_name or set(decisions) != set(by_name) or any(v not in {"best", "review", "reject"} for v in decisions.values()):
        raise ValueError("最终决定与当前报告照片不一致，无法建立偏好快照")
    signature = json.dumps([report.get("source"), report.get("generated_at"), sorted(decisions.items())], ensure_ascii=False)
    snapshot_id = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:24]
    changed = [p for p in photos if decisions[p["name"]] != p.get("category")]
    changed.sort(key=lambda p: (decisions[p["name"]] == "review", -abs(int(p.get("score", 50)) - 60), p["name"]))
    selected = [p["name"] for p in changed[:4]]
    groups: dict[str, list[dict]] = {}
    for photo in photos:
        if photo.get("group_id"):
            groups.setdefault(str(photo["group_id"]), []).append(photo)
    pairs = []
    for group_id, members in groups.items():
        kept = sorted((p for p in members if decisions[p["name"]] == "best"), key=lambda p: -int(p.get("score", 0)))
        passed = sorted((p for p in members if decisions[p["name"]] != "best"), key=lambda p: -int(p.get("score", 0)))
        if kept and passed and any(decisions[p["name"]] != p.get("category") for p in members):
            pairs.append({"group": group_id, "keep": kept[0]["name"], "pass": passed[0]["name"]})
    for pair in pairs:
        missing = [name for name in (pair["keep"], pair["pass"]) if name not in selected]
        if len(selected) + len(missing) > 6:
            continue
        selected.extend(missing)
        if len(selected) >= 6:
            break
    samples = [{"name": name, "final": decisions[name], "suggested": by_name[name].get("category", "review"),
                "score": by_name[name].get("score", 0), "scene": by_name[name].get("scene", "general"),
                "reason": str(by_name[name].get("reason", ""))[:160], "group": by_name[name].get("group_id")}
               for name in selected]
    snapshot = {"version": 1, "id": snapshot_id, "active": True, "created_at": datetime.now().isoformat(timespec="seconds"),
                "total": len(photos), "changed": len(changed), "samples": samples,
                "pairs": [p for p in pairs if p["keep"] in selected and p["pass"] in selected]}
    if persist:
        write_json(report_dir / SNAPSHOT_FILE, snapshot)
    return snapshot


def load_snapshot(report_dir: Path) -> dict:
    return read_json(report_dir / SNAPSHOT_FILE, {})


def summary_prompt(snapshot: dict) -> str:
    samples = snapshot.get("samples", [])
    if not samples:
        raise ValueError("这次没有足够有辨别力的案例；不会为了生成泛泛的偏好而付费调用 AI")
    return (
        "你是摄影偏好归纳助手。以下是用户最终确认后挑出的少量代表性案例，不是整批照片。"
        "请观察附图，重点分析用户推翻算法建议的选择，以及同组留下/放弃的差异。"
        "不要把算法分数当成用户喜好，不要猜测图片不可见的细节。只输出 JSON 对象，格式为"
        '{"rules":[{"text":"具体、可迁移的偏好描述","scene":"general|portrait|landscape|night|sports",'
        '"dimension":"composition|sharpness|expression|lighting|variety|subject|other",'
        '"evidence":["照片文件名"]}]}。最多 3 条；证据不足就返回空数组。不要输出 Markdown。\n'
        "案例：\n" + "\n".join(
            f"{s['name']}：算法 {s['suggested']} / 用户最终 {s['final']}；分数 {s['score']}；场景 {s['scene']}；依据 {s['reason']}"
            for s in samples
        ) + "\n同组取舍：\n" + "\n".join(f"保留 {p['keep']}，未保留 {p['pass']}" for p in snapshot.get("pairs", []))
    )


def evidence_count(evidence: list[str], snapshot: dict) -> int:
    """Two photos from one kept/passed pair count as one choice, not two independent votes."""
    names = set(evidence)
    count = len(names)
    for pair in snapshot.get("pairs", []):
        if pair["keep"] in names and pair["pass"] in names:
            count -= 1
    return max(0, count)


def parse_summary(answer: str, snapshot: dict) -> dict:
    start, end = answer.find("{"), answer.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("AI 没有返回可审核的结构化偏好")
    raw = json.loads(answer[start:end + 1])
    if not isinstance(raw, dict) or not isinstance(raw.get("rules"), list):
        raise ValueError("AI 返回的偏好格式无效")
    allowed = {s["name"] for s in snapshot["samples"]}
    rules = []
    for item in raw["rules"][:3]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()[:180]
        evidence = list(dict.fromkeys(name for name in item.get("evidence", []) if isinstance(name, str) and name in allowed)) if isinstance(item.get("evidence"), list) else []
        if not text or not evidence:
            continue
        count = evidence_count(evidence, snapshot)
        rules.append({"text": text, "scene": item.get("scene") if item.get("scene") in SCENES else "general",
                      "dimension": item.get("dimension") if item.get("dimension") in DIMENSIONS else "other",
                      "evidence": evidence[:4], "evidence_count": count, "confidence": "medium" if count >= 3 else "guess"})
    return {"snapshot_id": snapshot["id"], "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "draft", "rules": rules}


def save_review(report_dir: Path, snapshot_id: str, proposed: list[dict]) -> dict:
    snapshot = load_snapshot(report_dir)
    draft = read_json(report_dir / DRAFT_FILE, {})
    if not snapshot or not snapshot.get("active", True) or draft.get("snapshot_id") != snapshot_id or snapshot.get("id") != snapshot_id or draft.get("status") not in {"draft", "reviewed"}:
        raise ValueError("没有可审核的本次偏好草稿")
    if not isinstance(proposed, list) or len(proposed) > 3:
        raise ValueError("偏好条数异常")
    allowed = {s["name"] for s in snapshot["samples"]}
    reviewed = []
    for item in proposed:
        if not isinstance(item, dict):
            raise ValueError("偏好格式无效")
        text = item.get("text", "")
        evidence = item.get("evidence", [])
        if not isinstance(text, str) or not text.strip() or len(text) > 180 or not isinstance(evidence, list) or not evidence or any(name not in allowed for name in evidence):
            raise ValueError("偏好描述或案例无效")
        scene, dimension = item.get("scene"), item.get("dimension")
        if scene not in SCENES or dimension not in DIMENSIONS:
            raise ValueError("偏好分类无效")
        count = evidence_count(evidence, snapshot)
        reviewed.append({"id": uuid.uuid4().hex, "text": text.strip(), "scene": scene, "dimension": dimension,
                         "evidence": list(dict.fromkeys(evidence))[:4], "evidence_count": count,
                         "confidence": "medium" if count >= 3 else "guess",
                         "source_run": snapshot_id, "created_at": datetime.now().isoformat(timespec="seconds")})
    store = load_memory()
    store["rules"] = [rule for rule in store["rules"] if rule.get("source_run") != snapshot_id] + reviewed
    write_json(memory_path(), store)
    draft["status"] = "reviewed"
    draft["reviewed_at"] = datetime.now().isoformat(timespec="seconds")
    draft["rules"] = reviewed
    write_json(report_dir / DRAFT_FILE, draft)
    return store


def delete_rule(rule_id: str) -> dict:
    store = load_memory()
    before = len(store["rules"])
    store["rules"] = [rule for rule in store["rules"] if rule.get("id") != rule_id]
    if len(store["rules"]) == before:
        raise ValueError("找不到这条偏好")
    write_json(memory_path(), store)
    return store


def update_rule(rule_id: str, text: str) -> dict:
    if not isinstance(text, str) or not text.strip() or len(text) > 180:
        raise ValueError("偏好描述应为 1–180 个字符")
    store = load_memory()
    for rule in store["rules"]:
        if rule.get("id") == rule_id:
            rule["text"] = text.strip()
            rule["updated_at"] = datetime.now().isoformat(timespec="seconds")
            write_json(memory_path(), store)
            return store
    raise ValueError("找不到这条偏好")


def prompt_preferences(scenes: set[str]) -> str:
    rules = [r for r in load_memory()["rules"] if r.get("scene") in scenes or r.get("scene") == "general"]
    lines = []
    for rule in rules[-6:]:
        description = re.sub(r"[\r\n]+", " ", str(rule.get("text", "")))[:180]
        lines.append(f"- {description}（{rule.get('confidence', 'guess')}；人工审核）")
    return "\n".join(lines)[:1000]
