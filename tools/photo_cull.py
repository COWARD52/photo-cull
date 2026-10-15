#!/usr/bin/env python3
"""Local-only photo culling report generator. Never modifies source photos."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
RAW_EXTENSIONS = {".arw", ".cr2", ".cr3", ".nef", ".nrw", ".raf", ".dng", ".rw2", ".orf", ".pef", ".srw", ".raw"}
LABELS = {"best": "推荐保留", "review": "待确认", "reject": "建议淘汰"}
MODE_LABELS = {"auto": "自动识别", "general": "通用", "portrait": "人像", "landscape": "风景", "night": "夜景", "sports": "运动"}
CACHE_VERSION = 5


@dataclass
class Photo:
    path: str
    name: str
    taken_at: float
    width: int
    height: int
    sharpness: float
    brightness: float
    contrast: float
    dark_ratio: float
    bright_ratio: float
    dhash: int
    thumb: str
    subject_sharpness: float = 0.0
    face_count: int = 0
    closed_eyes: int = 0
    face_area: float = 0.0
    scene_signature: str = ""
    scene_histogram: list[int] = field(default_factory=list)
    raw_path: str = ""
    raw_name: str = ""
    score: int = 0
    category: str = "review"
    reason: str = "待分析"
    group_id: int | None = None
    group_rank: int = 1
    group_gap: float = 0.0
    group_type: str = ""
    content_similarity: float = 0.0
    confidence: str = "low"
    scene: str = "general"
    global_sharp_score: int = 0
    subject_sharp_score: int = 0
    exposure_score: int = 0
    contrast_score: int = 0
    composition_score: int = 0
    composition_position_score: int = 0
    composition_balance_score: int = 0
    composition_edge_score: int = 0
    composition_tilt: float = 0.0
    composition_notes: list[str] = field(default_factory=list)
    raw_score: int = 0
    group_penalty: int = 0
    score_notes: list[str] = field(default_factory=list)


def timestamp_for(image: Image.Image, path: Path) -> float:
    try:
        raw = image.getexif().get(36867) or image.getexif().get(306)
        if raw:
            return datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S").timestamp()
    except Exception:
        pass
    return path.stat().st_mtime


def dhash(gray: Image.Image) -> int:
    small = gray.resize((17, 16), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.int16)
    bits = (arr[:, :-1] > arr[:, 1:]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def laplacian_variance(gray: np.ndarray) -> float:
    if gray.shape[0] <= 2 or gray.shape[1] <= 2:
        return 0.0
    center = gray[1:-1, 1:-1]
    lap = 4 * center - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    return float(np.var(lap))


def build_vision_helper() -> Path | None:
    if platform.system() != "Darwin" or not shutil.which("swiftc"):
        return None
    source = Path(__file__).with_name("photo_cull_vision.swift")
    binary = Path(__file__).with_name(".photo_cull_vision")
    if not source.exists():
        return None
    try:
        if not binary.exists() or binary.stat().st_mtime_ns < source.stat().st_mtime_ns:
            subprocess.run(
                ["swiftc", str(source), "-o", str(binary)],
                check=True, capture_output=True, text=True, timeout=120,
            )
        return binary
    except Exception as exc:
        print(f"人脸分析已降级：{exc}", flush=True)
        return None


def scan_faces(paths: list[Path]) -> tuple[dict[str, list[dict]], bool]:
    if not paths:
        return {}, build_vision_helper() is not None
    helper = build_vision_helper()
    if helper is None:
        return {}, False
    try:
        result = subprocess.run(
            [str(helper)], input=json.dumps([str(p) for p in paths]),
            capture_output=True, text=True, check=True, timeout=max(120, len(paths) * 4),
        )
        payload = json.loads(result.stdout)
        return {item["path"]: item.get("faces", []) for item in payload}, True
    except Exception as exc:
        print(f"人脸分析已降级：{exc}", flush=True)
        return {}, False


def face_metrics(gray: np.ndarray, faces: list[dict]) -> tuple[float, int, int, float]:
    height, width = gray.shape
    scores: list[float] = []
    closed = 0
    max_area = 0.0
    for face in faces:
        x = max(0, int(float(face.get("x", 0)) * width))
        y = max(0, int((1 - float(face.get("y", 0)) - float(face.get("height", 0))) * height))
        right = min(width, int((float(face.get("x", 0)) + float(face.get("width", 0))) * width))
        bottom = min(height, int((1 - float(face.get("y", 0))) * height))
        area = max(0.0, float(face.get("width", 0)) * float(face.get("height", 0)))
        max_area = max(max_area, area)
        if right - x > 8 and bottom - y > 8:
            scores.append(laplacian_variance(gray[y:bottom, x:right]))
        left_ratio, right_ratio = face.get("leftEyeRatio"), face.get("rightEyeRatio")
        if area >= 0.012 and left_ratio is not None and right_ratio is not None and float(left_ratio) < 0.105 and float(right_ratio) < 0.105:
            closed += 1
    return (max(scores) if scores else 0.0, len(faces), closed, max_area)


def composition_metrics(image: Image.Image, faces: list[dict]) -> tuple[int, int, int, int, float, list[str]]:
    """Estimate composition from subject placement, visual balance, edge pressure and horizontal tilt."""
    sample = image.resize((128, 128), Image.Resampling.LANCZOS)
    gray_image = ImageOps.grayscale(sample)
    gray = np.asarray(gray_image, dtype=np.float32)
    blurred = np.asarray(gray_image.filter(ImageFilter.GaussianBlur(radius=7)), dtype=np.float32)
    local_contrast = np.abs(gray - blurred)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    gradient = np.hypot(gx, gy)
    saliency = local_contrast + gradient * 0.42
    threshold = float(np.percentile(saliency, 68))
    weights = np.maximum(saliency - threshold, 0)
    if float(weights.sum()) < 1e-6:
        weights = np.abs(gray - float(gray.mean())) + 1e-3

    height, width = weights.shape
    yy, xx = np.mgrid[0:height, 0:width]
    total = max(float(weights.sum()), 1e-6)
    saliency_x = float((weights * xx).sum() / total / max(width - 1, 1))
    saliency_y = float((weights * yy).sum() / total / max(height - 1, 1))

    valid_faces = [face for face in faces if float(face.get("width", 0)) * float(face.get("height", 0)) > 0]
    if valid_faces:
        face_weights = [max(float(face.get("width", 0)) * float(face.get("height", 0)), 1e-5) for face in valid_faces]
        face_total = sum(face_weights)
        subject_x = sum((float(face.get("x", 0)) + float(face.get("width", 0)) / 2) * area for face, area in zip(valid_faces, face_weights)) / face_total
        subject_y = sum((1 - float(face.get("y", 0)) - float(face.get("height", 0)) / 2) * area for face, area in zip(valid_faces, face_weights)) / face_total
        anchors = [(0.5, 0.38), (1 / 3, 0.36), (2 / 3, 0.36)]
    else:
        subject_x, subject_y = saliency_x, saliency_y
        anchors = [(0.5, 0.5), (1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3)]
    distance = min(math.hypot((subject_x - ax) / 0.42, (subject_y - ay) / 0.42) for ax, ay in anchors)
    position_score = round(max(20, 100 * math.exp(-1.45 * distance * distance)))

    left, right = float(weights[:, : width // 2].sum()) / total, float(weights[:, width // 2 :].sum()) / total
    top, bottom = float(weights[: height // 2, :].sum()) / total, float(weights[height // 2 :, :].sum()) / total
    balance_score = round(max(20, 100 - 72 * abs(left - right) - 38 * abs(top - bottom)))

    margin = max(4, round(min(width, height) * 0.075))
    border = np.zeros_like(weights, dtype=bool)
    border[:margin, :] = border[-margin:, :] = True
    border[:, :margin] = border[:, -margin:] = True
    border_ratio = float(weights[border].sum()) / total
    edge_score = max(20.0, 100 - max(0.0, border_ratio - 0.13) * 190)
    face_touching = False
    face_close = False
    for face in valid_faces:
        x, y = float(face.get("x", 0)), float(face.get("y", 0))
        face_width, face_height = float(face.get("width", 0)), float(face.get("height", 0))
        clearance = min(x, y, 1 - x - face_width, 1 - y - face_height)
        face_touching = face_touching or clearance < 0.009
        face_close = face_close or clearance < 0.028
    if face_touching:
        edge_score = min(edge_score, 34)
    elif face_close:
        edge_score = min(edge_score, 58)
    edge_score = round(edge_score)

    strong = gradient >= float(np.percentile(gradient, 86))
    line_angles = np.degrees(np.arctan2(gy, gx) + math.pi / 2)
    line_angles = (line_angles + 90) % 180 - 90
    horizontal = strong & (np.abs(line_angles) <= 18)
    horizontal_weight = float(gradient[horizontal].sum())
    strong_weight = max(float(gradient[strong].sum()), 1e-6)
    tilt = 0.0
    if horizontal_weight / strong_weight >= 0.2 and horizontal_weight > 0:
        radians = np.radians(line_angles[horizontal] * 2)
        tilt = math.degrees(0.5 * math.atan2(float((np.sin(radians) * gradient[horizontal]).sum()), float((np.cos(radians) * gradient[horizontal]).sum())))
    tilt_penalty = min(12.0, max(0.0, abs(tilt) - 2.2) * 1.6)

    score = round(max(1, min(100, position_score * 0.45 + balance_score * 0.30 + edge_score * 0.25 - tilt_penalty)))
    horizontal_name = "左侧" if subject_x < 0.42 else ("右侧" if subject_x > 0.58 else "中央")
    vertical_name = "上方" if subject_y < 0.4 else ("下方" if subject_y > 0.64 else "中部")
    subject_name = "人物主体" if valid_faces else "视觉主体"
    notes = [f"{subject_name}约位于画面{horizontal_name}{vertical_name}；主体位置 {position_score} 分"]
    notes.append("画面视觉重心较均衡" if balance_score >= 72 else "画面视觉重心存在偏移")
    if edge_score < 60:
        notes.append("人物或高对比元素靠近画面边缘，建议人工检查裁切")
    else:
        notes.append("主要视觉元素与画面边缘保留了较安全的距离")
    if abs(tilt) >= 3.2:
        notes.append(f"检测到约 {abs(tilt):.1f}° 的水平线倾斜倾向，仅作辅助参考")
    return score, position_score, balance_score, edge_score, round(tilt, 1), notes


def scene_features(image: Image.Image) -> tuple[str, list[int]]:
    width, height = image.size
    vectors: list[np.ndarray] = []
    for scale in (1.0, 0.84, 0.70):
        crop_width, crop_height = max(1, int(width * scale)), max(1, int(height * scale))
        positions = [(0.5, 0.5)] if scale == 1.0 else [(x, y) for y in (0.25, 0.5, 0.75) for x in (0.25, 0.5, 0.75)]
        for px, py in positions:
            left = max(0, min(width - crop_width, int(width * px) - crop_width // 2))
            top = max(0, min(height - crop_height, int(height * py) - crop_height // 2))
            crop = image.crop((left, top, left + crop_width, top + crop_height))
            values = np.asarray(ImageOps.grayscale(crop.resize((16, 16), Image.Resampling.LANCZOS)), dtype=np.float32).reshape(-1)
            values -= float(values.mean())
            norm = float(np.linalg.norm(values))
            if norm:
                values /= norm
            vectors.append(np.clip(np.rint(values * 127), -127, 127).astype(np.int8))
    signature = base64.b64encode(np.stack(vectors).tobytes()).decode("ascii")
    color = np.asarray(image.resize((96, 96), Image.Resampling.LANCZOS), dtype=np.uint8)
    histogram = np.concatenate([np.histogram(color[:, :, channel], bins=12, range=(0, 256))[0] for channel in range(3)]).astype(np.float32)
    histogram /= max(float(np.linalg.norm(histogram)), 1.0)
    return signature, np.clip(np.rint(histogram * 255), 0, 255).astype(np.uint8).tolist()


def content_similarity(first: Photo, second: Photo) -> float:
    if not first.scene_signature or not second.scene_signature or not first.scene_histogram or not second.scene_histogram:
        return 0.0
    try:
        a = np.frombuffer(base64.b64decode(first.scene_signature), dtype=np.int8).astype(np.float32).reshape(-1, 256)
        b = np.frombuffer(base64.b64decode(second.scene_signature), dtype=np.int8).astype(np.float32).reshape(-1, 256)
        a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-6)
        b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-6)
        structure = float(np.max(a @ b.T))
        hist_a = np.asarray(first.scene_histogram, dtype=np.float32)
        hist_b = np.asarray(second.scene_histogram, dtype=np.float32)
        color = float(np.dot(hist_a, hist_b) / max(float(np.linalg.norm(hist_a) * np.linalg.norm(hist_b)), 1e-6))
        return max(0.0, min(1.0, structure * 0.82 + color * 0.18))
    except Exception:
        return 0.0


def analyze(path: Path, source: Path, thumbs: Path, cache_id: str, raw_map: dict[str, Path], faces: list[dict]) -> Photo:
    with Image.open(path) as raw:
        taken = timestamp_for(raw, path)
        full_image = ImageOps.exif_transpose(raw).convert("RGB")
        width, height = full_image.size
        image = full_image.copy()
        image.thumbnail((640, 640), Image.Resampling.LANCZOS)
        gray_img = ImageOps.grayscale(image)
        gray = np.asarray(gray_img, dtype=np.float32)
        sharpness = laplacian_variance(gray)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        dark_ratio = float(np.mean(gray < 18))
        bright_ratio = float(np.mean(gray > 238))
        subject_sharpness, face_count, closed_eyes, face_area = face_metrics(gray, faces)
        composition_score, composition_position_score, composition_balance_score, composition_edge_score, composition_tilt, composition_notes = composition_metrics(image, faces)
        scene_signature, scene_histogram = scene_features(image)

        thumb_name = f"{cache_id}.jpg"
        thumb_path = thumbs / thumb_name
        preview = ImageOps.contain(full_image, (1600, 1200), Image.Resampling.LANCZOS)
        preview.save(thumb_path, "JPEG", quality=91, optimize=True, progressive=True, subsampling=0)

    raw_path = raw_map.get(path.stem.lower())
    return Photo(
        path=str(path.relative_to(source)), name=path.name, taken_at=taken,
        width=width, height=height, sharpness=sharpness, brightness=brightness,
        contrast=contrast, dark_ratio=dark_ratio, bright_ratio=bright_ratio,
        dhash=dhash(gray_img), thumb=f"thumbs/{thumb_name}",
        subject_sharpness=subject_sharpness, face_count=face_count,
        closed_eyes=closed_eyes, face_area=face_area,
        composition_score=composition_score,
        composition_position_score=composition_position_score,
        composition_balance_score=composition_balance_score,
        composition_edge_score=composition_edge_score,
        composition_tilt=composition_tilt,
        composition_notes=composition_notes,
        scene_signature=scene_signature, scene_histogram=scene_histogram,
        raw_path=str(raw_path.relative_to(source)) if raw_path else "",
        raw_name=raw_path.name if raw_path else "",
    )


def percentile(values: list[float], value: float) -> float:
    if len(values) < 2:
        return 0.5
    ordered = sorted(values)
    lo, hi = 0, len(ordered)
    while lo < hi:
        mid = (lo + hi) // 2
        if ordered[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    return lo / len(ordered)


def scene_for(photo: Photo, requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode
    if photo.face_count:
        return "portrait"
    if photo.brightness < 72 and photo.bright_ratio < 0.25:
        return "night"
    return "general"


def quality_details(photo: Photo, sharp_values: list[float], subject_values: list[float], requested_mode: str) -> dict[str, float | str | tuple[float, float, float]]:
    scene = scene_for(photo, requested_mode)
    global_sharp = percentile(sharp_values, photo.sharpness) * 100
    subject_sharp = percentile(subject_values, photo.subject_sharpness) * 100 if photo.subject_sharpness > 0 and subject_values else global_sharp
    sharp = global_sharp
    if photo.face_count:
        sharp = subject_sharp * 0.72 + global_sharp * 0.28

    if scene == "night":
        exposure = 100 - max(0, 28 - photo.brightness) * 1.5 - max(0, photo.brightness - 185) * 1.1
        exposure -= photo.dark_ratio * 38 + photo.bright_ratio * 105
        weights = (0.58, 0.27, 0.15)
    elif scene == "landscape":
        exposure = 100 - max(0, 55 - photo.brightness) * 1.15 - max(0, photo.brightness - 205) * 1.25
        exposure -= photo.dark_ratio * 70 + photo.bright_ratio * 110
        weights = (0.50, 0.28, 0.22)
    elif scene == "sports":
        exposure = 100 - max(0, 48 - photo.brightness) * 1.15 - max(0, photo.brightness - 210) * 1.2
        exposure -= (photo.dark_ratio + photo.bright_ratio) * 80
        weights = (0.70, 0.20, 0.10)
    else:
        exposure = 100 - max(0, 62 - photo.brightness) * 1.35 - max(0, photo.brightness - 198) * 1.35
        exposure -= (photo.dark_ratio + photo.bright_ratio) * 95
        weights = (0.60, 0.27, 0.13) if scene == "portrait" else (0.58, 0.29, 0.13)
    contrast = min(100, max(0, (photo.contrast - 18) * 2.0))
    technical_score = sharp * weights[0] + exposure * weights[1] + contrast * weights[2]
    score = technical_score * 0.88 + photo.composition_score * 0.12
    if photo.closed_eyes:
        score -= 14
    return {
        "scene": scene,
        "global_sharp": max(0, min(100, global_sharp)),
        "subject_sharp": max(0, min(100, subject_sharp)),
        "combined_sharp": max(0, min(100, sharp)),
        "exposure": max(0, min(100, exposure)),
        "contrast": max(0, min(100, contrast)),
        "composition": max(0, min(100, photo.composition_score)),
        "technical": max(0, min(100, technical_score)),
        "weights": weights,
        "total": max(0, min(100, score)),
    }


def quality(photo: Photo, sharp_values: list[float], subject_values: list[float], requested_mode: str) -> float:
    return float(quality_details(photo, sharp_values, subject_values, requested_mode)["total"])


def learn_preferences(out: Path) -> dict[str, float | int]:
    profile: dict[str, float | int] = {"samples": 0, "keep_threshold": 66.0, "reject_gap": 18.0, "rescued_rejects": 0}
    data_path, decisions_path, profile_path = out / "初筛数据.json", out / "手动调整.json", out / "学习偏好.json"
    try:
        if profile_path.exists():
            saved_profile = json.loads(profile_path.read_text(encoding="utf-8"))
            profile.update({key: saved_profile[key] for key in profile if key in saved_profile})
    except Exception:
        pass
    if not data_path.exists() or not decisions_path.exists():
        return profile
    try:
        old = json.loads(data_path.read_text(encoding="utf-8"))
        decision_payload = json.loads(decisions_path.read_text(encoding="utf-8"))
        decisions = decision_payload.get("overrides", {})
        changed = []
        for photo in old.get("photos", []):
            final = decisions.get(photo.get("name"))
            if final and final != photo.get("category"):
                changed.append((photo, final))
        if len(changed) < 3:
            return profile
        profile["samples"] = len(changed)
        promoted = [float(p.get("score", 66)) for p, final in changed if final == "best"]
        demoted = [float(p.get("score", 66)) for p, final in changed if p.get("category") == "best" and final != "best"]
        rescued = sum(1 for p, final in changed if p.get("category") == "reject" and final != "reject")
        candidates = []
        if promoted:
            candidates.append(statistics.median(promoted))
        if demoted:
            candidates.append(statistics.median(demoted) + 2)
        if candidates:
            profile["keep_threshold"] = max(56.0, min(74.0, statistics.mean(candidates)))
        profile["rescued_rejects"] = rescued
        profile["reject_gap"] = 24.0 if rescued >= 2 else 18.0
        temp = profile_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(profile_path)
    except Exception:
        pass
    return profile


def load_group_feedback(out: Path) -> dict[str, list[list[str]]]:
    path = out / "相似分组反馈.json"
    empty = {"positive_pairs": [], "negative_pairs": [], "single_pairs": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return {key: payload.get(key, []) for key in empty}
    except Exception:
        return empty


def learn_similarity_settings(photos: list[Photo], out: Path, feedback: dict[str, list[list[str]]]) -> dict[str, float | int]:
    path = out / "相似学习配置.json"
    settings: dict[str, float | int] = {"group_threshold": 0.93, "duplicate_threshold": 0.94, "pair_samples": 0, "single_samples": 0}
    try:
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            settings.update({key: saved[key] for key in settings if key in saved})
    except Exception:
        pass
    by_name = {photo.name: photo for photo in photos}
    def scores(key: str) -> list[float]:
        return [content_similarity(by_name[a], by_name[b]) for a, b in feedback.get(key, []) if a in by_name and b in by_name]
    positive, negative, single = scores("positive_pairs"), scores("negative_pairs"), scores("single_pairs")
    pair_samples = len(positive) + len(negative)
    if pair_samples >= 20:
        if positive and negative:
            target = (float(np.quantile(positive, 0.20)) + float(np.quantile(negative, 0.80))) / 2
        elif positive:
            target = float(np.quantile(positive, 0.20))
        else:
            target = float(np.quantile(negative, 0.80)) + 0.01
        current = float(settings["group_threshold"])
        settings["group_threshold"] = max(0.88, min(0.96, current + max(-0.005, min(0.005, target - current))))
    if len(single) >= 20:
        target = float(np.quantile(single, 0.20))
        current = float(settings["duplicate_threshold"])
        learned = current + max(-0.005, min(0.005, target - current))
        settings["duplicate_threshold"] = max(float(settings["group_threshold"]) + 0.005, min(0.98, learned))
    settings["pair_samples"] = pair_samples
    settings["single_samples"] = len(single)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)
    return settings


def group_similar(photos: list[Photo], sharp_values: list[float], subject_values: list[float], requested_mode: str, feedback: dict[str, list[list[str]]], settings: dict[str, float | int]) -> None:
    ordered = sorted(photos, key=lambda p: (p.taken_at, p.name))
    parent = list(range(len(ordered)))
    component_members = [{index} for index in range(len(ordered))]
    by_name = {photo.name: index for index, photo in enumerate(ordered)}
    positive = {tuple(sorted(pair)) for pair in feedback.get("positive_pairs", []) if len(pair) == 2}
    negative = {tuple(sorted(pair)) for pair in feedback.get("negative_pairs", []) if len(pair) == 2}
    content_edges: set[tuple[int, int]] = set()

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> bool:
        root_a, root_b = find(first), find(second)
        if root_a == root_b:
            return True
        for member_a in component_members[root_a]:
            for member_b in component_members[root_b]:
                if tuple(sorted((ordered[member_a].name, ordered[member_b].name))) in negative:
                    return False
        parent[root_b] = root_a
        component_members[root_a].update(component_members[root_b])
        component_members[root_b].clear()
        return True

    group_threshold = float(settings.get("group_threshold", 0.93))
    for i, base in enumerate(ordered):
        for j in range(i + 1, min(len(ordered), i + 30)):
            candidate = ordered[j]
            gap = candidate.taken_at - base.taken_at
            if gap > 180:
                break
            pair = tuple(sorted((base.name, candidate.name)))
            if pair in negative:
                continue
            distance = hamming(base.dhash, candidate.dhash)
            similarity = content_similarity(base, candidate)
            hash_match = (gap <= 7 and distance <= 56) or (gap <= 20 and distance <= 32) or distance <= 18
            hybrid_match = gap <= 7 and distance <= 72 and similarity >= 0.90
            content_match = gap <= 120 and similarity >= group_threshold
            if hash_match or hybrid_match or content_match:
                union(i, j)
                if content_match and not (hash_match or hybrid_match):
                    content_edges.add((min(i, j), max(i, j)))

    for first_name, second_name in positive:
        if first_name in by_name and second_name in by_name:
            first, second = by_name[first_name], by_name[second_name]
            union(first, second)
            content_edges.add((min(first, second), max(first, second)))

    components: dict[int, list[int]] = {}
    for index in range(len(ordered)):
        components.setdefault(find(index), []).append(index)
    group_id = 0
    for indices in sorted((items for items in components.values() if len(items) >= 2), key=lambda items: min(items)):
        group_id += 1
        index_set = set(indices)
        group_type = "content" if any(first in index_set and second in index_set for first, second in content_edges) else "burst"
        members = [ordered[index] for index in indices]
        ranked = sorted(members, key=lambda photo: quality(photo, sharp_values, subject_values, requested_mode), reverse=True)
        top, top_score = ranked[0], quality(ranked[0], sharp_values, subject_values, requested_mode)
        for rank, photo in enumerate(ranked, 1):
            photo.group_id = group_id
            photo.group_rank = rank
            photo.group_gap = max(0.0, top_score - quality(photo, sharp_values, subject_values, requested_mode))
            photo.group_type = group_type
            photo.content_similarity = 1.0 if photo is top else content_similarity(top, photo)


def classify(photos: list[Photo], requested_mode: str, profile: dict[str, float | int], feedback: dict[str, list[list[str]]], similarity_settings: dict[str, float | int]) -> None:
    sharp_values = [p.sharpness for p in photos]
    subject_values = [p.subject_sharpness for p in photos if p.subject_sharpness > 0]
    group_similar(photos, sharp_values, subject_values, requested_mode, feedback, similarity_settings)
    keep_threshold = float(profile.get("keep_threshold", 66))
    reject_gap = float(profile.get("reject_gap", 18))
    duplicate_threshold = float(similarity_settings.get("duplicate_threshold", 0.94))
    for photo in photos:
        photo.scene = scene_for(photo, requested_mode)
        details = quality_details(photo, sharp_values, subject_values, requested_mode)
        base_score = float(details["total"])
        penalty = 0
        if photo.group_id and photo.group_rank > 1:
            penalty = min(22, 4 + (photo.group_rank - 1) * 4)
        photo.global_sharp_score = round(float(details["global_sharp"]))
        photo.subject_sharp_score = round(float(details["subject_sharp"]))
        photo.exposure_score = round(float(details["exposure"]))
        photo.contrast_score = round(float(details["contrast"]))
        photo.composition_score = round(float(details["composition"]))
        photo.raw_score = round(base_score)
        photo.group_penalty = penalty
        photo.score = round(max(1, base_score - penalty))

        sharp_pct = percentile(sharp_values, photo.sharpness)
        if photo.closed_eyes:
            photo.category, photo.reason, photo.confidence = "review", "可能闭眼，建议确认", "medium"
        elif photo.scene != "night" and (photo.brightness < 22 or photo.dark_ratio > 0.86):
            photo.category, photo.reason, photo.confidence = "reject", "高置信度：严重欠曝", "high"
        elif photo.brightness > 238 or photo.bright_ratio > 0.82:
            photo.category, photo.reason, photo.confidence = "reject", "高置信度：严重过曝", "high"
        elif sharp_pct <= 0.025 and photo.sharpness < statistics.median(sharp_values) * 0.16:
            photo.category, photo.reason, photo.confidence = "reject", "高置信度：明显失焦", "high"
        elif photo.group_type == "content" and photo.group_rank > 1 and photo.content_similarity >= duplicate_threshold:
            photo.category, photo.reason, photo.confidence = "reject", f"高置信度：与组内照片内容高度相似（{photo.content_similarity * 100:.0f}%）", "high"
        elif photo.group_id and photo.group_rank >= 3 and photo.group_gap >= reject_gap:
            photo.category, photo.reason, photo.confidence = "reject", f"高置信度：组内明显较弱（差 {photo.group_gap:.0f} 分）", "high"
        elif photo.group_id and photo.group_rank == 2:
            photo.category, photo.reason, photo.confidence = "review", "对比组备选", "medium"
        elif photo.group_id and photo.group_rank >= 3:
            photo.category, photo.reason, photo.confidence = "review", f"对比组第 {photo.group_rank} 名，差距不大", "low"
        elif photo.group_id and photo.group_rank == 1:
            if photo.score >= keep_threshold - 5:
                photo.category, photo.reason, photo.confidence = "best", "对比组综合最优", "medium"
            else:
                photo.category, photo.reason, photo.confidence = "review", "组内最优，但质量仍需确认", "low"
        elif photo.score >= keep_threshold:
            subject_text = "主体清晰" if photo.face_count else "画面清晰"
            photo.category, photo.reason, photo.confidence = "best", f"{subject_text}且曝光良好", "medium"
        else:
            photo.category, photo.reason, photo.confidence = "review", "低置信度：建议人工确认", "low"

        weights = details["weights"]
        photo.score_notes = [
            f"综合分由技术质量 88% 与构图 12% 组成；技术质量内部清晰度/曝光/对比度权重为 {weights[0] * 100:.0f}% / {weights[1] * 100:.0f}% / {weights[2] * 100:.0f}%",
            f"全图清晰度 {photo.global_sharp_score} 分，表示在本批照片中的相对位置",
            f"曝光 {photo.exposure_score} 分：平均亮度 {photo.brightness:.0f}，死黑 {photo.dark_ratio * 100:.1f}%，高光溢出 {photo.bright_ratio * 100:.1f}%",
            f"对比度 {photo.contrast_score} 分：原始标准差 {photo.contrast:.1f}",
            f"构图 {photo.composition_score} 分：主体位置 {photo.composition_position_score}，视觉平衡 {photo.composition_balance_score}，边缘安全 {photo.composition_edge_score}",
        ]
        photo.score_notes.extend(photo.composition_notes)
        if photo.face_count:
            photo.score_notes.append(f"检测到 {photo.face_count} 张人脸；主体清晰度 {photo.subject_sharp_score} 分" + (f"；{photo.closed_eyes} 张疑似闭眼" if photo.closed_eyes else "；未发现明显闭眼"))
        else:
            photo.score_notes.append("未检测到人脸，清晰度主要按全图计算")
        if photo.group_id:
            group_label = "对比组"
            group_basis = "内容相似" if photo.group_type == "content" else "连续拍摄"
            similarity_note = f"；内容相似度 {photo.content_similarity * 100:.1f}%" if photo.group_type == "content" else ""
            photo.score_notes.append(f"{group_label} #{photo.group_id} 第 {photo.group_rank} 名；归组依据：{group_basis}；比组内最佳低 {photo.group_gap:.1f} 分；组内扣分 {photo.group_penalty} 分{similarity_note}")
        else:
            photo.score_notes.append("未归入对比组，不进行组内扣分")
        photo.score_notes.append(f"基础质量 {photo.raw_score} 分，最终综合 {photo.score} 分；判断依据：{photo.reason}")


def contact_sheets(photos: list[Photo], out: Path) -> None:
    font = ImageFont.load_default(size=18)
    small_font = ImageFont.load_default(size=14)
    for category in ("best", "review", "reject"):
        selected = [p for p in photos if p.category == category]
        selected.sort(key=lambda p: (-p.score, p.taken_at))
        for page, start in enumerate(range(0, len(selected), 24), 1):
            batch = selected[start:start + 24]
            sheet = Image.new("RGB", (1600, 1200), "#0b0d10")
            draw = ImageDraw.Draw(sheet)
            for n, photo in enumerate(batch):
                col, row = n % 6, n // 6
                x, y = col * 266, row * 300
                with Image.open(out / photo.thumb) as im:
                    tile = ImageOps.pad(im.convert("RGB"), (246, 210), color="#15191e", method=Image.Resampling.LANCZOS)
                    sheet.paste(tile, (x + 10, y + 10))
                draw.text((x + 10, y + 226), photo.name[:29], fill="#f4f5ef", font=small_font)
                draw.text((x + 10, y + 248), f"{photo.score} · {photo.reason}", fill="#dfff54", font=font)
            sheet.save(out / f"总览-{LABELS[category]}-{page}.jpg", "JPEG", quality=86, optimize=True)


def write_csv(photos: list[Photo], out: Path) -> None:
    with (out / "初筛清单.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["JPG文件名", "JPG相对路径", "RAW文件名", "RAW相对路径", "建议", "综合分", "基础质量分", "组内扣分", "原因", "置信度", "识别场景", "全图清晰度分", "主体清晰度分", "曝光分", "对比度分", "构图分", "主体位置分", "视觉平衡分", "边缘安全分", "倾斜倾向", "人脸数", "可能闭眼", "分组类型", "分组编号", "组内排名", "组内差距", "内容相似度", "打分说明"])
        for p in sorted(photos, key=lambda x: (x.category, -x.score, x.taken_at)):
            writer.writerow([p.name, p.path, p.raw_name, p.raw_path, LABELS[p.category], p.score, p.raw_score, p.group_penalty, p.reason, p.confidence, MODE_LABELS[p.scene], p.global_sharp_score, p.subject_sharp_score if p.face_count else "", p.exposure_score, p.contrast_score, p.composition_score, p.composition_position_score, p.composition_balance_score, p.composition_edge_score, p.composition_tilt, p.face_count, p.closed_eyes, ("对比组（内容相似）" if p.group_type == "content" else "对比组（连续拍摄）") if p.group_id else "", p.group_id or "", p.group_rank if p.group_id else "", round(p.group_gap, 1) if p.group_id else "", round(p.content_similarity * 100, 1) if p.group_type == "content" else "", "；".join(p.score_notes)])


def write_html(photos: list[Photo], out: Path, source: Path, requested_mode: str, profile: dict[str, float | int], similarity_settings: dict[str, float | int], vision_enabled: bool) -> None:
    counts = {key: sum(p.category == key for p in photos) for key in LABELS}
    raw_pairs = sum(bool(p.raw_path) for p in photos)
    grouped_count = sum(bool(p.group_id) for p in photos)
    compare_groups = len({p.group_id for p in photos if p.group_id})
    cards = []
    order = {"best": 0, "review": 1, "reject": 2}
    for p in sorted(photos, key=lambda x: (order[x.category], -x.score, x.taken_at)):
        notes_html = "".join(f"<li>{html.escape(note)}</li>" for note in p.score_notes)
        confidence_label = {"high": "高", "medium": "中", "low": "低"}[p.confidence]
        cards.append(f"""
        <article class="card" data-kind="{p.category}" data-original="{p.category}"
          data-name="{html.escape(p.name)}" data-raw="{html.escape(p.raw_name)}"
          data-score="{p.score}" data-composition="{p.composition_score}" data-taken="{p.taken_at}" data-reason="{html.escape(p.reason)}"
          data-group="{p.group_id or ''}" data-rank="{p.group_rank}" data-group-type="{p.group_type}" data-confidence="{p.confidence}">
          <div class="card-overlay"><span data-status>{LABELS[p.category]}</span><b>{p.score}</b></div>
          <img src="{html.escape(p.thumb)}" alt="{html.escape(p.name)}" loading="lazy">
          <div class="meta"><strong title="{html.escape(p.name)}">{html.escape(p.name)}</strong>
          <div class="verdict-line"><span>{html.escape(p.reason)}</span><b>{p.score}</b></div>
          <small class="photo-context">{(('对比组 #' + str(p.group_id) + ' · 第 ' + str(p.group_rank) + ' 名 · 归组依据：' + ('内容相似' if p.group_type == 'content' else '连续拍摄')) if p.group_id else '单张')} · {MODE_LABELS[p.scene]} · {'检测到 ' + str(p.face_count) + ' 张人脸' if p.face_count else '未检测到人脸'} · {'JPG + ' + html.escape(Path(p.raw_name).suffix.upper().lstrip('.')) if p.raw_name else '仅 JPG'}</small>
          <details class="score-details"><summary>查看打分依据 · 置信度{confidence_label}</summary>
          <div class="score-grid"><span>综合<b>{p.score}</b></span><span>构图<b>{p.composition_score}</b></span><span>全图清晰<b>{p.global_sharp_score}</b></span><span>主体清晰<b>{p.subject_sharp_score if p.face_count else '—'}</b></span><span>曝光<b>{p.exposure_score}</b></span><span>对比度<b>{p.contrast_score}</b></span><span>组内扣分<b>-{p.group_penalty}</b></span></div>
          <ul>{notes_html}</ul></details>
          <div class="choices"><button data-set="best">保留</button><button data-set="review">待确认</button><button data-set="reject">淘汰</button></div><div class="group-actions">{'<button data-compare>打开对比组</button>' if p.group_id else ''}<button data-pick-group>{'追加到此组' if p.group_id else '加入对比组'}</button></div></div>
        </article>""")
    css = """
:root{--bg:#0b0d10;--panel:#15191e;--line:#2b3139;--text:#f5f6ef;--muted:#9ba2ac;--lime:#ddff54;--amber:#ffc65a;--coral:#ff746d;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC',sans-serif}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text)}header{position:sticky;top:0;z-index:3;padding:20px 3vw;background:rgba(11,13,16,.94);border-bottom:1px solid var(--line);backdrop-filter:blur(14px)}h1{margin:0 0 5px;font-size:28px}header p{margin:0;color:var(--muted)}header .engine{margin-top:6px;color:#c6cbd2;font-size:13px}.bar,.tools{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}button{border:1px solid var(--line);background:var(--panel);color:var(--text);padding:9px 13px;border-radius:99px;cursor:pointer}button.active{background:var(--lime);color:#111;border-color:var(--lime)}.tools button{border-radius:9px}main{padding:24px 3vw 60px}.summary{display:grid;grid-template-columns:repeat(3,minmax(120px,190px));gap:12px;margin-bottom:18px}.summary div{padding:16px;border:1px solid var(--line);border-radius:14px;background:var(--panel)}.summary b{display:block;font-size:28px}.summary span{color:var(--muted);font-size:14px}.strategy{margin:0 0 18px;padding:14px 16px;border:1px solid var(--line);border-radius:14px;background:#11151a}.strategy summary,.score-details summary{cursor:pointer;color:#dce3eb;font-weight:600}.strategy-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;margin-top:12px}.strategy-grid div{padding:10px;border-radius:9px;background:var(--panel);font-size:13px;color:#c2c8d0}.strategy-grid b{display:block;color:var(--lime);margin-bottom:4px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:14px}.card{overflow:hidden;border:1px solid var(--line);border-radius:15px;background:var(--panel)}.card[data-confidence=high]{border-color:#ff8d87}.card.pair-selected{outline:3px solid #75a7ff;outline-offset:2px}.card img{width:100%;aspect-ratio:4/3;object-fit:contain;background:#070809}.meta{padding:11px}.meta strong{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:14px}.meta>div:not(.choices):not(.score-grid){display:flex;justify-content:space-between;gap:8px;margin-top:7px;color:var(--muted);font-size:13px}.meta b{color:var(--lime)}.meta small{display:block;color:#7f8792;margin-top:5px}.score-details{margin-top:9px;border-top:1px solid var(--line);padding-top:8px;font-size:12px}.score-details summary{font-size:12px;color:#aeb8c4}.score-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-top:9px}.score-grid span{padding:6px;background:#0d1014;border-radius:6px;color:#8f98a4}.score-grid b{display:block;font-size:15px;margin-top:2px}.score-details ul{margin:9px 0 0;padding-left:18px;color:#a8b0ba;line-height:1.5}.choices{display:grid;grid-template-columns:repeat(auto-fit,minmax(52px,1fr));gap:5px;margin-top:10px}.choices button{padding:7px 3px;border-radius:7px;font-size:12px}.card[data-kind=best] .choices [data-set=best]{background:var(--lime);border-color:var(--lime);color:#111}.card[data-kind=review] .choices [data-set=review]{background:var(--amber);border-color:var(--amber);color:#111}.card[data-kind=reject] .choices [data-set=reject]{background:var(--coral);border-color:var(--coral);color:#111}.card.changed{outline:2px solid #75a7ff;outline-offset:1px}.card[hidden]{display:none}dialog{width:min(1200px,96vw);max-height:92vh;border:1px solid var(--line);border-radius:18px;background:#0f1216;color:var(--text);padding:18px}dialog::backdrop{background:rgba(0,0,0,.82)}.compare-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}.compare-head h2{margin:0}.compare-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;overflow:auto}.compare-item{border:1px solid var(--line);border-radius:12px;background:var(--panel);overflow:hidden}.compare-item img{width:100%;height:280px;object-fit:contain;background:#060708}.compare-item .meta{padding:10px}.feedback-tools{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px;padding-top:12px;border-top:1px solid var(--line)}.feedback-tools button{border-radius:8px}.feedback-tools .danger{border-color:#744;color:#ffaaa5}@media(max-width:600px){.summary{grid-template-columns:repeat(3,1fr)}.grid{grid-template-columns:1fr;gap:10px}.meta{padding:9px}.compare-grid{grid-template-columns:1fr}.compare-item img{height:44vh}}
"""
    css += """
:root{--bg:#090b0f;--panel:#12161c;--panel-2:#181d25;--panel-3:#202631;--line:#2a313d;--text:#f6f7f2;--muted:#929ba8;--lime:#d9ff4f;--lime-soft:rgba(217,255,79,.13);--amber:#ffc868;--coral:#ff7d73;--blue:#80a9ff;color-scheme:dark}
html{scroll-behavior:smooth}body{min-height:100vh;background:radial-gradient(circle at 12% -10%,rgba(104,126,176,.13),transparent 34rem),linear-gradient(180deg,#0a0c10,#08090c 60%);font-size:15px;line-height:1.45}
header{position:sticky;top:0;z-index:10;padding:15px clamp(16px,2.6vw,42px) 13px;background:rgba(9,11,15,.88);border-bottom:1px solid rgba(255,255,255,.08);box-shadow:0 12px 38px rgba(0,0,0,.22);backdrop-filter:blur(24px) saturate(130%)}
.header-top{display:flex;align-items:flex-end;justify-content:space-between;gap:24px}.title-block{flex:none}.eyebrow{display:block;margin-bottom:2px;color:var(--lime);font-size:11px;font-weight:750;letter-spacing:.18em}.title-block h1{display:flex;align-items:center;gap:9px;margin:0;font-size:clamp(23px,2.2vw,31px);letter-spacing:-.04em}.title-block h1 em{padding:3px 8px;border:1px solid rgba(217,255,79,.35);border-radius:999px;background:var(--lime-soft);color:var(--lime);font-size:11px;font-style:normal;letter-spacing:.08em}.source-block{min-width:0;text-align:right}.source-block p{color:#d4d8df;font-size:13px}.source-block small{display:block;max-width:min(60vw,760px);overflow:hidden;color:#6f7886;font-size:11px;text-overflow:ellipsis;white-space:nowrap}.engine{margin-top:7px!important;color:#9da6b4!important;font-size:12px!important}.bar{gap:7px;margin-top:10px;overflow-x:auto;scrollbar-width:none}.bar::-webkit-scrollbar{display:none}.bar button{flex:none;padding:7px 12px;border-color:transparent;background:#171b22;color:#aeb6c2;font-size:13px;font-weight:650}.bar button:hover{background:#202630;color:#fff}.bar button.active{background:var(--lime);box-shadow:0 5px 18px rgba(217,255,79,.18);color:#101307}
main{max-width:1900px;margin:0 auto;padding:16px clamp(14px,2.5vw,42px) 72px}.workspace-bar{display:flex;align-items:center;justify-content:space-between;gap:18px;margin-bottom:12px;padding:10px 12px;border:1px solid rgba(255,255,255,.07);border-radius:16px;background:rgba(18,22,28,.72)}.summary{display:flex;grid-template-columns:none;gap:5px;margin:0}.summary div{min-width:78px;padding:6px 11px;border:0;border-right:1px solid var(--line);border-radius:0;background:transparent}.summary div:last-child{border-right:0}.summary b{display:inline;margin-right:6px;font-size:19px;letter-spacing:-.04em}.summary span{font-size:12px}.tools{gap:6px;margin:0}.tools button{padding:8px 11px;border-color:var(--line);background:#171c23;color:#c4cad3;font-size:12px;font-weight:650}.tools button:hover{border-color:#4a5564;background:#202630;color:#fff}.tools .primary-action{border-color:var(--lime);background:var(--lime);color:#11150a}.tools .primary-action:hover{background:#e4ff77;color:#101307}.sort-control{display:flex;align-items:center;gap:7px;padding-left:4px;color:#7f8997;font-size:12px}.sort-control select{height:34px;padding:0 30px 0 10px;border:1px solid var(--line);border-radius:9px;background:#171c23;color:#d4dae2;font:inherit;cursor:pointer}.sort-control select:hover,.sort-control select:focus{border-color:#536071;outline:none}
.strategy{margin:0 0 10px;padding:0;border-color:rgba(255,255,255,.07);border-radius:14px;background:rgba(17,21,27,.76)}.strategy>summary{padding:11px 14px;color:#cbd1da;font-size:13px;list-style:none}.strategy>summary::-webkit-details-marker{display:none}.strategy>summary:before{content:'＋';display:inline-block;width:20px;color:var(--lime)}.strategy[open]>summary:before{content:'−'}.strategy>summary span{margin-left:8px;color:#707988;font-weight:400}.strategy-grid{padding:0 12px 12px}.strategy-grid div{border:1px solid rgba(255,255,255,.04);background:#141820}.notice{min-height:20px;margin:6px 2px 12px;color:#77808e;font-size:12px}
.switch-progress{margin:0 0 12px;padding:13px 14px;border:1px solid rgba(128,169,255,.35);border-radius:14px;background:rgba(18,25,36,.92)}.switch-progress[hidden]{display:none}.switch-progress-head{display:flex;justify-content:space-between;gap:16px;margin-bottom:9px;color:#d9e3f5;font-size:12px}.switch-progress-head b{color:#fff}.switch-progress-track{overflow:hidden;height:9px;border-radius:99px;background:#252c36}.switch-progress-bar{width:0;height:100%;border-radius:inherit;background:linear-gradient(90deg,var(--blue),var(--lime));transition:width .25s ease}.switch-progress small{display:block;overflow:hidden;margin-top:7px;color:#77869a;font-size:11px;text-overflow:ellipsis;white-space:nowrap}
.grid{grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}.card{position:relative;border-color:rgba(255,255,255,.08);border-radius:18px;background:linear-gradient(180deg,#161a21,#12161c);box-shadow:0 8px 30px rgba(0,0,0,.18);transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease}.card:hover{transform:translateY(-2px);border-color:#3a4350;box-shadow:0 16px 42px rgba(0,0,0,.32)}.card[data-confidence=high]{border-color:rgba(255,125,115,.34)}.card img{display:block;width:100%;height:clamp(250px,24vw,390px);aspect-ratio:auto;object-fit:contain;background:radial-gradient(circle at 50% 45%,#11151b,#050608 72%)}.card>img{cursor:zoom-in}.card>img:focus-visible{outline:3px solid var(--lime);outline-offset:-4px}.card-overlay{position:absolute;top:12px;right:12px;left:12px;z-index:2;display:flex;align-items:center;justify-content:space-between;pointer-events:none}.card-overlay span,.card-overlay b{padding:5px 9px;border:1px solid rgba(255,255,255,.12);border-radius:999px;background:rgba(8,10,13,.78);box-shadow:0 4px 16px rgba(0,0,0,.28);backdrop-filter:blur(12px);font-size:11px;font-weight:750}.card-overlay b{color:var(--lime);font-size:15px}.card[data-kind=best] .card-overlay span{border-color:rgba(217,255,79,.36);color:var(--lime)}.card[data-kind=review] .card-overlay span{border-color:rgba(255,200,104,.38);color:var(--amber)}.card[data-kind=reject] .card-overlay span{border-color:rgba(255,125,115,.4);color:var(--coral)}
.meta{padding:14px}.meta>strong{font-size:16px;letter-spacing:.01em}.verdict-line{align-items:flex-start!important;margin-top:7px!important;color:#b1b9c5!important}.verdict-line span{line-height:1.35}.verdict-line b{display:none}.photo-context{margin-top:7px!important;color:#747e8c!important;font-size:11px!important;line-height:1.45}.score-details{margin-top:11px;padding-top:9px}.score-details summary{padding:2px 0;color:#909aa8;font-weight:600}.score-details[open] summary{color:#d7dce3}.score-grid{gap:6px}.score-grid span{background:#0d1015}.score-details ul{color:#9aa3af}.choices{grid-template-columns:repeat(3,1fr);gap:6px;margin-top:12px}.choices button{padding:9px 5px;border-color:#343c48;border-radius:9px;background:#171c23;color:#c4cbd4;font-size:13px;font-weight:700}.choices button:hover{border-color:#697586;color:#fff}.card[data-kind=best] .choices [data-set=best]{box-shadow:0 4px 14px rgba(217,255,79,.16)}.card[data-kind=review] .choices [data-set=review]{box-shadow:0 4px 14px rgba(255,200,104,.13)}.card[data-kind=reject] .choices [data-set=reject]{box-shadow:0 4px 14px rgba(255,125,115,.13)}.stash-button{width:100%;margin-top:7px;padding:8px;border-color:#3b4657;border-radius:8px;background:#111720;color:#91a0b6;font-size:12px;font-weight:700}.stash-button:hover{border-color:#7184a0;color:#dce7f8}.stash-button.active,.compare-stash.active{border-color:var(--blue);background:rgba(128,169,255,.18);color:#c8d8ff;box-shadow:0 4px 16px rgba(128,169,255,.12)}.card[data-stashed=true]{border-color:rgba(128,169,255,.52)}.group-actions{display:flex;gap:6px;margin-top:7px}.group-actions button{flex:1;padding:7px;border:1px solid transparent;border-radius:8px;background:#0e1217;color:#7f8997;font-size:11px}.group-actions button:hover{border-color:#313a46;color:#d5dae1}.group-actions [data-pick-group].active{border-color:rgba(128,169,255,.62);background:rgba(128,169,255,.14);color:#bcd0ff}.card.changed{outline-color:var(--blue)}.card.pair-selected{outline:3px solid var(--blue);outline-offset:2px}.group-selection{position:fixed;right:clamp(14px,2.5vw,42px);bottom:18px;left:clamp(14px,2.5vw,42px);z-index:20;display:flex;align-items:center;gap:12px;padding:12px 14px;border:1px solid rgba(128,169,255,.42);border-radius:15px;background:rgba(19,24,32,.94);box-shadow:0 18px 55px rgba(0,0,0,.55);backdrop-filter:blur(22px)}.group-selection[hidden]{display:none}.group-selection-copy{min-width:0;flex:1}.group-selection-copy b{display:block;color:#e8edf6;font-size:13px}.group-selection-copy small{display:block;overflow:hidden;margin-top:2px;color:#8995a5;font-size:11px;text-overflow:ellipsis;white-space:nowrap}.group-selection button{flex:none;border-radius:9px}.group-selection .create-group{border-color:var(--blue);background:var(--blue);color:#0c1119;font-weight:750}.group-selection .create-group:disabled{border-color:#343c48;background:#252b35;color:#747f8e;cursor:not-allowed}
dialog{width:min(1400px,96vw);padding:18px;border-color:#343c48;border-radius:20px;background:#0d1015;box-shadow:0 30px 100px rgba(0,0,0,.7)}.compare-head{position:sticky;top:-18px;z-index:3;margin:-18px -18px 14px;padding:16px 18px 12px;background:rgba(13,16,21,.94);backdrop-filter:blur(18px)}.compare-head h2{font-size:20px}.compare-head button{padding:7px 12px}.compare-grid{grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}.compare-item{border-color:#2b323d;border-radius:15px;background:#151920}.compare-item img{height:min(52vh,520px)}.compare-current{display:inline-block;margin-top:7px;padding:3px 8px;border:1px solid currentColor;border-radius:999px;font-size:11px;font-weight:750}.compare-item[data-kind=best] .compare-current{color:var(--lime)}.compare-item[data-kind=review] .compare-current{color:var(--amber)}.compare-item[data-kind=reject] .compare-current{color:var(--coral)}.compare-item .choices button.active{color:#101307;box-shadow:0 5px 18px rgba(0,0,0,.26);transform:translateY(-1px)}.compare-item .choices [data-set=best].active{border-color:var(--lime);background:var(--lime)}.compare-item .choices [data-set=review].active{border-color:var(--amber);background:var(--amber)}.compare-item .choices [data-set=reject].active{border-color:var(--coral);background:var(--coral)}.member-tools{display:flex;gap:7px;margin-top:7px}.member-tools button{width:100%;padding:7px;border-color:#49373a;border-radius:8px;background:#191316;color:#d9a2a2;font-size:11px}.member-tools button:hover{border-color:#865158;color:#ffc1be}.member-tools .compare-stash{border-color:#3b4657;background:#111720;color:#91a0b6}.feedback-tools{position:sticky;bottom:-18px;margin:14px -18px -18px;padding:12px 18px;background:rgba(13,16,21,.94);backdrop-filter:blur(18px)}.feedback-tools span{align-self:center;color:var(--muted);font-size:12px}
@media(max-width:800px){header{padding:12px 14px}.header-top{align-items:flex-start}.source-block p{display:none}.source-block small{max-width:42vw}.engine{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.workspace-bar{align-items:stretch;flex-direction:column;gap:8px}.summary{justify-content:space-between}.summary div{flex:1;min-width:0}.tools{display:grid;grid-template-columns:repeat(2,1fr)}.sort-control{grid-column:1/-1;justify-content:space-between;padding:0}.sort-control select{flex:1}.grid{grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}.card img{height:min(68vh,420px)}}
@media(max-width:520px){.source-block{display:none}.title-block h1{font-size:23px}.bar button{padding:7px 10px}.grid{grid-template-columns:1fr}.strategy>summary span{display:none}.card img{height:min(64vh,430px)}.compare-grid{grid-template-columns:1fr}.feedback-tools button{flex:1}.feedback-tools span{width:100%}.group-selection{align-items:stretch;flex-wrap:wrap}.group-selection-copy{width:100%;flex-basis:100%}.group-selection button{flex:1}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}.card{transition:none}.card:hover{transform:none}}
.image-viewer{width:min(96vw,1600px);height:min(94vh,1100px);max-width:none;max-height:none;padding:14px;border:1px solid #343c48;border-radius:20px;background:rgba(10,12,16,.98);color:var(--text);overflow:hidden}.image-viewer::backdrop{background:rgba(0,0,0,.88);backdrop-filter:blur(8px)}.image-viewer-layout{display:grid;grid-template-rows:auto minmax(0,1fr) auto;gap:10px;width:100%;height:100%}.image-viewer-head,.image-viewer-foot{display:flex;align-items:center;justify-content:space-between;gap:12px}.image-viewer-title{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:700}.image-viewer-foot{justify-content:center;color:var(--muted);font-size:13px}.image-viewer-foot button{min-width:100px;border-radius:9px}.image-viewer-image-wrap{display:grid;place-items:center;min-height:0;overflow:auto;background:#050608;border-radius:12px}.image-viewer-image{display:block;max-width:100%;max-height:100%;object-fit:contain;cursor:zoom-in;transition:transform .16s ease;transform-origin:center}.image-viewer-image.zoomed{max-width:none;max-height:none;cursor:zoom-out;transform:scale(2)}
"""
    script = f"""
const key='photo-cull:{html.escape(str(source))}';
const cards=[...document.querySelectorAll('.card')];let current='all';let saved={{}};let savedProgress={{}};
cards.forEach((c,index)=>c.dataset.order=String(index));
const imageViewer=document.createElement('dialog');imageViewer.id='image-viewer';imageViewer.className='image-viewer';imageViewer.innerHTML='<div class="image-viewer-layout"><div class="image-viewer-head"><strong id="image-viewer-title" class="image-viewer-title"></strong><button id="image-viewer-close" type="button">关闭</button></div><div class="image-viewer-image-wrap"><img id="image-viewer-image" class="image-viewer-image" alt=""></div><div class="image-viewer-foot"><button id="image-viewer-prev" type="button">上一张</button><span id="image-viewer-count"></span><button id="image-viewer-next" type="button">下一张</button></div></div>';document.body.append(imageViewer);const viewerImage=document.querySelector('#image-viewer-image');let viewerCards=[];let viewerIndex=0;
function updateViewer(){{const card=viewerCards[viewerIndex];if(!card)return;const image=card.querySelector(':scope > img');viewerImage.src=image.currentSrc||image.src;viewerImage.alt=card.dataset.name;viewerImage.classList.remove('zoomed');document.querySelector('#image-viewer-title').textContent=card.dataset.name+' · '+card.dataset.score+' 分 · '+labels[card.dataset.kind];document.querySelector('#image-viewer-count').textContent=(viewerIndex+1)+' / '+viewerCards.length;document.querySelector('#image-viewer-prev').disabled=viewerIndex===0;document.querySelector('#image-viewer-next').disabled=viewerIndex===viewerCards.length-1}}
function openImage(card){{viewerCards=cards.filter(c=>!c.hidden);viewerIndex=viewerCards.indexOf(card);if(viewerIndex<0){{viewerCards=cards;viewerIndex=cards.indexOf(card)}}if(viewerIndex<0)return;updateViewer();imageViewer.showModal()}}
cards.forEach(card=>{{const image=card.querySelector(':scope > img');image.tabIndex=0;image.setAttribute('role','button');image.setAttribute('aria-label','点击查看大图：'+card.dataset.name);image.onclick=()=>openImage(card);image.onkeydown=event=>{{if(event.key==='Enter'||event.key===' '){{event.preventDefault();openImage(card)}}}}}});
document.querySelector('#image-viewer-close').onclick=()=>imageViewer.close();document.querySelector('#image-viewer-prev').onclick=()=>{{if(viewerIndex>0){{viewerIndex--;updateViewer()}}}};document.querySelector('#image-viewer-next').onclick=()=>{{if(viewerIndex<viewerCards.length-1){{viewerIndex++;updateViewer()}}}};viewerImage.onclick=()=>viewerImage.classList.toggle('zoomed');imageViewer.addEventListener('click',event=>{{if(event.target===imageViewer)imageViewer.close()}});imageViewer.addEventListener('keydown',event=>{{if(event.key==='ArrowLeft'&&viewerIndex>0){{viewerIndex--;updateViewer()}}else if(event.key==='ArrowRight'&&viewerIndex<viewerCards.length-1){{viewerIndex++;updateViewer()}}}});
try{{saved=JSON.parse(localStorage.getItem(key)||'{{}}')}}catch{{}}
try{{savedProgress=JSON.parse(localStorage.getItem(key+':progress')||'{{}}')}}catch{{}}
const labels={{best:'保留',review:'待确认',reject:'淘汰'}};
const grid=document.querySelector('.grid');const collator=new Intl.Collator('zh-CN',{{numeric:true,sensitivity:'base'}});
function sortCards(mode){{const sorted=[...cards];sorted.sort((a,b)=>{{if(mode==='score-desc')return Number(b.dataset.score)-Number(a.dataset.score)||collator.compare(a.dataset.name,b.dataset.name);if(mode==='score-asc')return Number(a.dataset.score)-Number(b.dataset.score)||collator.compare(a.dataset.name,b.dataset.name);if(mode==='composition-desc')return Number(b.dataset.composition)-Number(a.dataset.composition)||Number(b.dataset.score)-Number(a.dataset.score);if(mode==='time-desc')return Number(b.dataset.taken)-Number(a.dataset.taken)||collator.compare(a.dataset.name,b.dataset.name);if(mode==='time-asc')return Number(a.dataset.taken)-Number(b.dataset.taken)||collator.compare(a.dataset.name,b.dataset.name);if(mode==='name')return collator.compare(a.dataset.name,b.dataset.name);if(mode==='group'){{const ag=a.dataset.group,bg=b.dataset.group;if(Boolean(ag)!==Boolean(bg))return ag?-1:1;if(ag&&bg){{const byGroup=collator.compare(ag,bg);if(byGroup)return byGroup;return Number(a.dataset.rank)-Number(b.dataset.rank)}}return collator.compare(a.dataset.name,b.dataset.name)}}return Number(a.dataset.order)-Number(b.dataset.order)}});sorted.forEach(c=>grid.appendChild(c));localStorage.setItem(key+':sort',mode)}}
function counts(){{for(const k of ['best','review','reject']){{const n=cards.filter(c=>c.dataset.kind===k).length;document.querySelector('#n-'+k).textContent=n;document.querySelector('[data-filter='+k+']').textContent=labels[k]+' '+n}}document.querySelector('[data-filter=all]').textContent='全部 '+cards.length;document.querySelector('[data-filter=group]').textContent='对比组 '+cards.filter(c=>c.dataset.group).length}}
function show(){{cards.forEach(c=>c.hidden=current==='group'?!c.dataset.group:current!=='all'&&c.dataset.kind!==current)}}
function decisionMap(){{return Object.fromEntries(cards.map(c=>[c.dataset.name,c.dataset.kind]))}}
let saveTimer=null;
function saveBody(){{return JSON.stringify({{decisions:decisionMap(),overrides:saved,progress:savedProgress}})}}
function scheduleSave(){{clearTimeout(saveTimer);saveTimer=setTimeout(async()=>{{if(location.protocol==='file:')return;try{{await fetch('/api/decisions',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:saveBody()}})}}catch{{}}}},250)}}
function setCard(c,k,persist=true){{c.dataset.kind=k;c.querySelector('[data-status]').textContent=labels[k];c.classList.toggle('changed',k!==c.dataset.original);if(persist){{if(k===c.dataset.original)delete saved[c.dataset.name];else saved[c.dataset.name]=k;localStorage.setItem(key,JSON.stringify(saved));scheduleSave()}}counts();show()}}
function focusCardName(){{const visible=cards.filter(c=>!c.hidden);if(!visible.length)return'';const target=window.innerHeight*.32;return visible.reduce((best,c)=>Math.abs(c.getBoundingClientRect().top-target)<Math.abs(best.getBoundingClientRect().top-target)?c:best,visible[0]).dataset.name}}
function captureProgress(){{return{{filter:current,sort:document.querySelector('#sort').value,focusName:focusCardName(),scrollY:Math.round(window.scrollY),savedAt:new Date().toISOString()}}}}
function restoreProgress(progress){{if(!progress||!progress.savedAt)return;const sort=document.querySelector('#sort');if(progress.sort){{sort.value=progress.sort;sortCards(progress.sort)}}current=progress.filter||'all';document.querySelectorAll('[data-filter]').forEach(x=>x.classList.toggle('active',x.dataset.filter===current));show();requestAnimationFrame(()=>{{const focus=cards.find(c=>c.dataset.name===progress.focusName&&!c.hidden);if(focus){{focus.scrollIntoView({{block:'start'}});window.scrollBy(0,-145)}}else window.scrollTo(0,Number(progress.scrollY)||0)}});const when=new Date(progress.savedAt).toLocaleString('zh-CN',{{hour12:false}});document.querySelector('#notice').textContent='已恢复 '+when+' 暂存的筛选进度。'}}
async function hydrate(){{if(location.protocol==='file:')return;try{{const r=await fetch('/api/decisions');const data=await r.json();if(data.ok){{saved={{}};savedProgress=data.progress||savedProgress;cards.forEach(c=>setCard(c,c.dataset.original,false));cards.forEach(c=>{{const desired=(data.decisions||{{}})[c.dataset.name];if(desired&&labels[desired]){{setCard(c,desired,false);if(desired!==c.dataset.original)saved[c.dataset.name]=desired}}}});localStorage.setItem(key,JSON.stringify(saved));localStorage.setItem(key+':progress',JSON.stringify(savedProgress));counts();restoreProgress(savedProgress);scheduleSave()}}}}catch{{}}}}
async function sendGroupFeedback(action,names,extra={{}}){{if(location.protocol==='file:'){{alert('请通过照片初筛工具打开页面后保存反馈。');return null}}try{{const r=await fetch('/api/group-feedback',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{action,names,...extra}})}});const data=await r.json();if(!r.ok||!data.ok)throw new Error(data.error||'保存失败');return data}}catch(error){{alert('保存反馈失败：'+error.message);return null}}}}
const groupUiHistory=[];
function captureGroupState(){{return Object.fromEntries(cards.map(c=>[c.dataset.name,{{group:c.dataset.group,groupType:c.dataset.groupType,rank:c.dataset.rank,context:c.querySelector('.photo-context').textContent}}]))}}
function contextTail(c){{const parts=c.querySelector('.photo-context').textContent.split(' · ');return (c.dataset.group?parts.slice(3):parts.slice(1)).join(' · ')}}
function ensureCompareButton(c){{const actions=c.querySelector('.group-actions');let button=actions.querySelector('[data-compare]');if(!c.dataset.group){{if(button)button.remove();return}}if(!button){{button=document.createElement('button');button.dataset.compare='';actions.prepend(button)}}button.textContent='打开对比组';button.onclick=()=>openCompare(c.dataset.group)}}
function setGroupState(c,group,rank,groupType='manual'){{const tail=contextTail(c);const basis=groupType==='content'?'内容相似':(groupType==='burst'?'连续拍摄':'手动归组');c.dataset.group=group;c.dataset.groupType=groupType;c.dataset.rank=String(rank);c.querySelector('.photo-context').textContent=(group?'对比组 #'+group+' · 第 '+rank+' 名 · 归组依据：'+basis:'单张')+(tail?' · '+tail:'');ensureCompareButton(c)}}
function applyManualGroup(names){{const members=cards.filter(c=>names.includes(c.dataset.name)).sort((a,b)=>Number(b.dataset.score)-Number(a.dataset.score));const group='手动'+Date.now().toString().slice(-6);members.forEach((c,i)=>setGroupState(c,group,i+1,'manual'));counts();show();return group}}
function restoreGroupState(state){{cards.forEach(c=>{{const savedState=state[c.dataset.name];if(!savedState)return;c.dataset.group=savedState.group;c.dataset.groupType=savedState.groupType;c.dataset.rank=savedState.rank;c.querySelector('.photo-context').textContent=savedState.context;ensureCompareButton(c)}});counts();show()}}
const groupSelection=new Set();
const selectionBar=document.querySelector('#group-selection');
function selectionMembers(c){{return c.dataset.group?cards.filter(x=>x.dataset.group===c.dataset.group&&x.dataset.groupType===c.dataset.groupType):[c]}}
function updateGroupSelection(){{const selected=cards.filter(c=>groupSelection.has(c.dataset.name));const existingGroups=new Set(selected.filter(c=>c.dataset.group).map(c=>c.dataset.groupType+':'+c.dataset.group));cards.forEach(c=>{{const active=groupSelection.has(c.dataset.name);c.classList.toggle('pair-selected',active);const button=c.querySelector('[data-pick-group]');button.classList.toggle('active',active);button.textContent=active?'已选择 ✓':(c.dataset.group?'追加到此组':'加入对比组')}});document.querySelector('#group-selection-count').textContent='已选 '+selected.length+' 张照片';document.querySelector('#group-selection-names').textContent=selected.map(c=>c.dataset.name).join('、')||'请选择至少两张需要放在一起对比的照片';const create=document.querySelector('#create-group');create.textContent=existingGroups.size>1?'合并对比组':(existingGroups.size===1?'追加到对比组':'创建对比组');create.disabled=selected.length<2;selectionBar.hidden=selected.length===0}}
function clearGroupSelection(){{groupSelection.clear();updateGroupSelection()}}
cards.forEach(c=>{{if(saved[c.dataset.name])setCard(c,saved[c.dataset.name],false);c.querySelectorAll('[data-set]').forEach(b=>b.onclick=()=>setCard(c,b.dataset.set));c.querySelector('[data-pick-group]').onclick=()=>{{const members=selectionMembers(c);const remove=members.every(x=>groupSelection.has(x.dataset.name));const additions=members.filter(x=>!groupSelection.has(x.dataset.name));if(!remove&&groupSelection.size+additions.length>30){{alert('一个手动对比组最多选择 30 张照片。');return}}members.forEach(x=>remove?groupSelection.delete(x.dataset.name):groupSelection.add(x.dataset.name));updateGroupSelection()}}}});counts();
document.querySelector('#clear-group-selection').onclick=clearGroupSelection;
document.querySelector('#create-group').onclick=async()=>{{const names=[...groupSelection];if(names.length<2)return;const selected=cards.filter(c=>groupSelection.has(c.dataset.name));const existingGroups=new Set(selected.filter(c=>c.dataset.group).map(c=>c.dataset.groupType+':'+c.dataset.group));if(existingGroups.size>1&&!confirm('所选照片来自多个已有对比组，确认合并这些组？'))return;const hasExisting=existingGroups.size>0;const before=captureGroupState();const button=document.querySelector('#create-group');button.disabled=true;button.textContent='正在保存…';const result=await sendGroupFeedback('merge_group',names);if(result){{groupUiHistory.push(before);const group=applyManualGroup(names);document.querySelector('#notice').textContent=hasExisting?'已立即追加到对比组 #'+group+'，下次分析仍会保留此结果。':'已立即创建对比组 #'+group+'，下次分析仍会保留此结果。';clearGroupSelection()}}else updateGroupSelection()}};
document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{{document.querySelectorAll('[data-filter]').forEach(x=>x.classList.remove('active'));b.classList.add('active');current=b.dataset.filter;show()}});
const sortSelect=document.querySelector('#sort');sortSelect.value=localStorage.getItem(key+':sort')||'recommended';sortSelect.onchange=()=>sortCards(sortSelect.value);sortCards(sortSelect.value);
document.querySelector('#save-progress').onclick=async()=>{{
  if(location.protocol==='file:'){{alert('请通过照片初筛工具打开页面后暂存进度。');return}}
  const button=document.querySelector('#save-progress');savedProgress=captureProgress();localStorage.setItem(key+':progress',JSON.stringify(savedProgress));button.disabled=true;button.textContent='正在暂存…';
  try{{const response=await fetch('/api/decisions',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:saveBody()}});const result=await response.json();if(!response.ok||!result.ok)throw new Error(result.error||'暂存失败');const when=new Date(savedProgress.savedAt).toLocaleString('zh-CN',{{hour12:false}});document.querySelector('#notice').textContent='筛选进度已暂存（'+when+'），下次打开会回到 '+(savedProgress.focusName||'当前位置')+'。';button.textContent='已暂存 ✓'}}catch(error){{alert('暂存失败：'+error.message);button.textContent='暂存进度'}}finally{{button.disabled=false}}
}};
async function waitForSwitchJob(jobId){{
  const panel=document.querySelector('#switch-progress');const phase=document.querySelector('#switch-phase');const percentLabel=document.querySelector('#switch-percent');const bar=document.querySelector('#switch-progress-bar');const track=panel.querySelector('[role="progressbar"]');const source=document.querySelector('#switch-source');
  panel.hidden=false;
  while(true){{
    const response=await fetch('/api/switch-status?job='+encodeURIComponent(jobId),{{cache:'no-store'}});const state=await response.json();
    if(!response.ok||!state.ok)throw new Error(state.error||'读取分析进度失败');
    const percent=Math.max(0,Math.min(100,Math.round(Number(state.percent)||0)));
    phase.textContent=state.phase||'正在分析照片';percentLabel.textContent=percent+'%';bar.style.width=percent+'%';track.setAttribute('aria-valuenow',String(percent));
    source.textContent=(state.source||'')+(state.total?' · '+(state.processed||0)+' / '+state.total+' 张':'');
    if(state.status==='done'){{location.href=state.url;return}}
    if(state.status==='error')throw new Error(state.error||'分析失败');
    await new Promise(resolve=>setTimeout(resolve,600));
  }}
}}
document.querySelector('#switch-directory').onclick=async()=>{{
  if(location.protocol==='file:'){{alert('请通过照片初筛工具打开页面后切换目录。');return}}
  const button=document.querySelector('#switch-directory');const notice=document.querySelector('#notice');
  savedProgress=captureProgress();localStorage.setItem(key+':progress',JSON.stringify(savedProgress));button.disabled=true;button.textContent='请选择文件夹…';notice.textContent='正在保存当前进度；选择新目录后会在本机完成分析并自动打开。';
  try{{
    const savedResponse=await fetch('/api/decisions',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:saveBody()}});const savedResult=await savedResponse.json();if(!savedResponse.ok||!savedResult.ok)throw new Error(savedResult.error||'当前进度保存失败');
    const response=await fetch('/api/switch-directory',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:'{{}}'}});const result=await response.json();if(!response.ok||!result.ok)throw new Error(result.error||'切换目录失败');
    if(result.cancelled){{notice.textContent='已取消切换目录，当前筛选进度已保存。';button.textContent='切换目录';return}}
    if(!result.job){{if(result.url){{location.href=result.url;return}}throw new Error('服务未返回切换任务')}}
    notice.textContent='已选择目录，正在检查已有结果或分析照片。';button.textContent='正在处理…';document.querySelector('#switch-source').textContent=result.source||'';await waitForSwitchJob(result.job);
  }}catch(error){{alert('切换目录失败：'+error.message);notice.textContent='切换目录失败，当前目录仍可继续筛选。';button.textContent='切换目录';const phase=document.querySelector('#switch-phase');if(!document.querySelector('#switch-progress').hidden)phase.textContent='处理失败：'+error.message}}finally{{if(button.textContent!=='正在处理…')button.disabled=false}}
}};
document.querySelector('#reset').onclick=()=>{{if(!confirm('重置全部手动调整和暂存进度？'))return;localStorage.removeItem(key);localStorage.removeItem(key+':progress');saved={{}};savedProgress={{}};cards.forEach(c=>setCard(c,c.dataset.original,false));scheduleSave()}};
if(savedProgress.savedAt)restoreProgress(savedProgress);hydrate();
const compare=document.querySelector('#compare');const compareGrid=document.querySelector('#compare-grid');
async function removeFromGroup(c,group){{const members=cards.filter(x=>x.dataset.group===group);const names=members.map(x=>x.dataset.name);const groupType=members[0]?.dataset.groupType||'manual';if(names.length<2||!confirm('将 '+c.dataset.name+' 从此对比组移出？'))return;const before=captureGroupState();const result=await sendGroupFeedback('remove_member',names,{{focus:c.dataset.name}});if(!result)return;groupUiHistory.push(before);const remaining=members.filter(x=>x!==c).sort((a,b)=>Number(b.dataset.score)-Number(a.dataset.score));setGroupState(c,'',0,'');if(remaining.length<2)remaining.forEach(x=>setGroupState(x,'',0,''));else remaining.forEach((x,i)=>setGroupState(x,group,i+1,groupType));counts();show();document.querySelector('#notice').textContent='已将 '+c.dataset.name+' 移出对比组，可用“撤销归组”恢复。';if(remaining.length>=2)openCompare(group);else compare.close()}}
function openCompare(group){{const members=cards.filter(c=>c.dataset.group===group).sort((a,b)=>Number(a.dataset.rank)-Number(b.dataset.rank));if(!members.length)return;compare.dataset.group=group;document.querySelector('#compare-title').textContent='对比组 #'+group+' · '+members.length+' 张';compareGrid.replaceChildren(...members.map(c=>{{const item=document.createElement('article');item.className='compare-item';item.dataset.kind=c.dataset.kind;const image=c.querySelector('img').cloneNode();image.loading='eager';const meta=document.createElement('div');meta.className='meta';const title=document.createElement('strong');title.textContent=c.dataset.name+' · '+c.dataset.score+' 分';const currentState=document.createElement('span');currentState.className='compare-current';currentState.textContent='当前：'+labels[c.dataset.kind];const info=document.createElement('small');const basis=c.dataset.groupType==='content'?'内容相似':(c.dataset.groupType==='burst'?'连续拍摄':'手动归组');info.textContent='第 '+c.dataset.rank+' 名 · '+basis+' · '+c.dataset.reason;const details=c.querySelector('.score-details').cloneNode(true);const choices=document.createElement('div');choices.className='choices';for(const k of ['best','review','reject']){{const b=document.createElement('button');const active=c.dataset.kind===k;b.dataset.set=k;b.textContent=(active?'✓ ':'')+labels[k];b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));b.onclick=()=>{{setCard(c,k);openCompare(group)}};choices.appendChild(b)}}const memberTools=document.createElement('div');memberTools.className='member-tools';const remove=document.createElement('button');remove.textContent='从本组移除';remove.onclick=()=>removeFromGroup(c,group);memberTools.append(remove);meta.append(title,currentState,info,details,choices,memberTools);item.append(image,meta);return item}}));if(!compare.open)compare.showModal()}}
cards.forEach(c=>{{const button=c.querySelector('[data-compare]');if(button)button.onclick=()=>openCompare(c.dataset.group)}});document.querySelector('#compare-close').onclick=()=>compare.close();
function currentGroupNames(){{return cards.filter(c=>c.dataset.group===compare.dataset.group).map(c=>c.dataset.name)}}
document.querySelector('#group-confirm').onclick=async()=>{{if(await sendGroupFeedback('confirm',currentGroupNames()))compare.close()}};
document.querySelector('#group-single').onclick=async()=>{{if(await sendGroupFeedback('single_best',currentGroupNames()))compare.close()}};
document.querySelector('#group-add').onclick=()=>{{const names=currentGroupNames();clearGroupSelection();names.forEach(name=>groupSelection.add(name));updateGroupSelection();compare.close();document.querySelector('#notice').textContent='当前对比组已选中，请继续选择要追加的照片。'}};
document.querySelector('#group-split').onclick=async()=>{{const names=currentGroupNames();if(!confirm('确认拆分这个对比组？页面会立即更新。'))return;const before=captureGroupState();if(await sendGroupFeedback('split',names)){{groupUiHistory.push(before);names.forEach(name=>{{const c=cards.find(x=>x.dataset.name===name);if(c)setGroupState(c,'',0,'')}});counts();show();compare.close();document.querySelector('#notice').textContent='对比组已拆分，可用“撤销归组”恢复。'}}}};
document.querySelector('#undo-group').onclick=async()=>{{const result=await sendGroupFeedback('undo',[]);if(!result)return;const previous=groupUiHistory.pop();if(previous){{restoreGroupState(previous);document.querySelector('#notice').textContent='已撤销上一次归组操作。'}}else document.querySelector('#notice').textContent='已撤销保存的归组反馈；重新运行分析后页面会同步。'}};
document.querySelector('#organize').onclick=async()=>{{
  if(location.protocol==='file:'){{alert('请先双击报告目录中的「启动筛选工具.command」，再使用确认整理功能。');return}}
  const totals={{best:cards.filter(c=>c.dataset.kind==='best').length,review:cards.filter(c=>c.dataset.kind==='review').length,reject:cards.filter(c=>c.dataset.kind==='reject').length}};
  const message=`确认整理文件？\n\n保留 ${{totals.best}} 组\n待确认 ${{totals.review}} 组\n待删除 ${{totals.reject}} 组\n\nJPG 与 RAW 会一起移动到「筛选完成」下的三个目录。本操作不会删除文件。`;
  if(!confirm(message))return;
  const button=document.querySelector('#organize');button.disabled=true;button.textContent='正在整理…';
  try{{const decisions=Object.fromEntries(cards.map(c=>[c.dataset.name,c.dataset.kind]));const response=await fetch('/api/organize',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{decisions}})}});const result=await response.json();if(!response.ok||!result.ok)throw new Error(result.error||'整理失败');savedProgress={{}};localStorage.removeItem(key+':progress');scheduleSave();document.querySelector('#notice').textContent=`整理完成：移动了 ${{result.counts.files}} 个文件，位置：${{result.target}}`;button.textContent='重新确认并同步目录'}}catch(error){{alert('整理失败：'+error.message);button.textContent='确认筛选并整理文件'}}finally{{button.disabled=false}}
}};
document.querySelector('#undo').onclick=async()=>{{if(location.protocol==='file:'){{alert('请通过「照片初筛工具.command」打开页面后再撤销。');return}}if(!confirm('撤销上次整理，把文件恢复到整理前的位置？'))return;const button=document.querySelector('#undo');button.disabled=true;try{{const response=await fetch('/api/undo',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:'{{}}'}});const result=await response.json();if(!response.ok||!result.ok)throw new Error(result.error||'撤销失败');document.querySelector('#notice').textContent=`已撤销上次整理，恢复 ${{result.restored}} 个文件。`}}catch(error){{alert('撤销失败：'+error.message)}}finally{{button.disabled=false}}}};
document.querySelector('#export').onclick=()=>{{const rows=[['JPG文件名','RAW文件名','最终决定','算法建议','综合分','构图分','原因']];cards.forEach(c=>rows.push([c.dataset.name,c.dataset.raw,labels[c.dataset.kind],labels[c.dataset.original],c.dataset.score,c.dataset.composition,c.dataset.reason]));const csv='\\ufeff'+rows.map(r=>r.map(v=>'"'+String(v).replaceAll('"','""')+'"').join(',')).join('\\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv;charset=utf-8'}}));a.download='调整后初筛清单.csv';a.click();URL.revokeObjectURL(a.href)}};
"""
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>照片初筛结果</title><style>{css}</style></head><body>
<header><div class="header-top"><div class="title-block"><span class="eyebrow">本机选片台</span><h1>照片初筛 <em>V2</em></h1></div><div class="source-block"><p>{len(photos)} 组照片 · {raw_pairs} 组 JPG+RAW · 原图未修改</p><small title="{html.escape(str(source))}">{html.escape(str(source))}</small></div></div><p class="engine">{MODE_LABELS[requested_mode]}策略 · {'人脸与眼部分析已启用' if vision_enabled else '通用分析'} · 构图分析已启用 · {compare_groups} 个对比组 · 分组学习 {int(similarity_settings.get('pair_samples', 0))}/20</p><nav class="bar" aria-label="照片筛选"><button class="active" data-filter="all">全部 {len(photos)}</button><button data-filter="best">保留 {counts['best']}</button><button data-filter="review">待确认 {counts['review']}</button><button data-filter="reject">淘汰 {counts['reject']}</button><button data-filter="group">对比组 {grouped_count}</button></nav></header>
<main><div class="workspace-bar"><section class="summary"><div><b id="n-best">{counts['best']}</b><span>保留</span></div><div><b id="n-review">{counts['review']}</b><span>待确认</span></div><div><b id="n-reject">{counts['reject']}</b><span>淘汰</span></div></section><div class="tools"><label class="sort-control"><span>排序</span><select id="sort" aria-label="照片排序"><option value="recommended">推荐顺序</option><option value="score-desc">分数：高到低</option><option value="score-asc">分数：低到高</option><option value="composition-desc">构图：高到低</option><option value="time-desc">时间：新到旧</option><option value="time-asc">时间：旧到新</option><option value="name">文件名</option><option value="group">按对比组</option></select></label><button id="save-progress">暂存进度</button><button id="switch-directory">切换目录</button><button id="organize" class="primary-action">确认筛选并整理</button><button id="undo-group">撤销归组</button><button id="undo">撤销整理</button><button id="export">导出清单</button><button id="reset">重置调整</button></div></div>
<section id="switch-progress" class="switch-progress" hidden aria-live="polite"><div class="switch-progress-head"><b id="switch-phase">正在准备照片分析</b><span id="switch-percent">0%</span></div><div class="switch-progress-track" role="progressbar" aria-label="切换目录分析进度" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><div id="switch-progress-bar" class="switch-progress-bar"></div></div><small id="switch-source"></small></section>
<details class="strategy"><summary>筛选策略与分数说明 <span>查看阈值、权重和学习状态</span></summary><div class="strategy-grid">
<div><b>综合分如何计算</b>技术质量占 88%，构图占 12%；技术质量包含清晰度、曝光和对比度，人像优先采用主体清晰度。所有分数范围为 0–100。</div>
<div><b>构图分如何计算</b>综合主体位置、视觉重心、边缘裁切风险与水平线倾斜倾向。人像优先依据人脸位置，其余照片使用本地视觉显著区域估计；构图不会单独触发淘汰。</div>
<div><b>推荐保留</b>单张综合分达到 {float(profile.get('keep_threshold', 66)):.0f}；连拍最佳达到 {max(1, float(profile.get('keep_threshold', 66)) - 5):.0f}。这只是推荐，仍可手动调整。</div>
<div><b>高置信度淘汰</b>仅限严重欠曝、严重过曝、处于本批最模糊的约 2.5%，或连拍比最佳低至少 {float(profile.get('reject_gap', 18)):.0f} 分。</div>
<div><b>待确认</b>算法证据不足、疑似闭眼、连拍差距较小或技术分处于中间区域时，不自动淘汰。</div>
<div><b>对比组如何形成</b>连续拍摄或画面内容相似的照片统一进入“对比组”。内容归组门槛 {float(similarity_settings.get('group_threshold', 0.93)) * 100:.1f}%，只留最佳门槛 {float(similarity_settings.get('duplicate_threshold', 0.94)) * 100:.1f}%。</div>
<div><b>如何学习</b>“分组正确/拆分”累计 20 对后开始调整内容识别门槛；“这组只留一张”累计 20 对后调整去重门槛，每次最多变化 0.5%。</div>
<div><b>相对分的含义</b>清晰度是在本批照片内比较，不代表跨批次绝对画质；曝光和对比度依据当前场景策略计算。</div>
<div><b>个性化状态</b>已参考 {int(profile.get('samples', 0))} 次手动修正；当前策略优先避免误删。蓝色外框表示本次手动改动。</div>
</div></details>
<p id="notice" class="notice">JPG 与 RAW 始终同步处理；连续拍摄和内容相似的照片都统一放入“对比组”。</p><section class="grid">{''.join(cards)}</section></main><aside id="group-selection" class="group-selection" hidden><div class="group-selection-copy"><b id="group-selection-count">已选 0 张照片</b><small id="group-selection-names">请选择至少两张需要放在一起对比的照片</small></div><button id="clear-group-selection">清空</button><button id="create-group" class="create-group" disabled>创建对比组</button></aside><dialog id="compare"><div class="compare-head"><h2 id="compare-title">对比组</h2><button id="compare-close">关闭</button></div><div id="compare-grid" class="compare-grid"></div><div class="feedback-tools"><button id="group-add">添加更多照片</button><button id="group-confirm">分组正确</button><button id="group-single">这组只留一张</button><button id="group-split" class="danger">拆分此组</button><span>操作会立即更新页面并保存在本机</span></div></dialog><script>{script}</script></body></html>"""
    (out / "打开查看.html").write_text(page, encoding="utf-8")


def write_local_launcher(out: Path) -> None:
    tool_dir = Path(__file__).resolve().parent
    for name in ("photo_cull.py", "photo_cull_server.py", "photo_cull_vision.swift"):
        source_file = tool_dir / name
        target_file = out / name
        if source_file.exists() and source_file.resolve() != target_file.resolve():
            shutil.copy2(source_file, target_file)
    launcher = """#!/bin/zsh
cd "$(dirname "$0")"
python3 photo_cull_server.py --report-dir . --source-dir .. --port 4174 &
server_pid=$!
sleep 1
open 'http://localhost:4174/打开查看.html'
wait $server_pid
"""
    launcher_path = out / "启动筛选工具.command"
    launcher_path.write_text(launcher, encoding="utf-8")
    launcher_path.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=MODE_LABELS, default="auto")
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    out = args.output.expanduser().resolve()
    if source == out or source in out.parents and out.name == source.name:
        raise SystemExit("输出目录不能覆盖照片目录")
    out.mkdir(parents=True, exist_ok=True)
    thumbs = out / "thumbs"
    thumbs.mkdir(exist_ok=True)

    cache_path = out / "分析缓存.json"
    try:
        cache_payload = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        old_cache = cache_payload.get("items", {}) if cache_payload.get("version") == CACHE_VERSION else {}
    except Exception:
        old_cache = {}

    photos_paths = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED and out not in p.parents)
    raw_map = {p.stem.lower(): p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in RAW_EXTENSIONS and out not in p.parents}
    uncached_paths = []
    for path in photos_paths:
        stat = path.stat()
        key = f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}"
        cached = old_cache.get(key)
        if not cached or not (out / cached.get("thumb", "")).exists():
            uncached_paths.append(path)
    face_map, vision_enabled = scan_faces(uncached_paths)
    if vision_enabled:
        print(f"人脸分析 {len(uncached_paths)} 张", flush=True)
    photos: list[Photo] = []
    errors: list[dict[str, str]] = []
    new_cache: dict[str, dict] = {}
    cached_count = 0
    for index, path in enumerate(photos_paths, 1):
        try:
            stat = path.stat()
            cache_key = f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}"
            cache_id = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:18]
            raw_path = raw_map.get(path.stem.lower())
            cached = old_cache.get(cache_key)
            if cached and (out / cached.get("thumb", "")).exists():
                photo = Photo(
                    path=str(path.relative_to(source)), name=path.name,
                    taken_at=cached["taken_at"], width=cached["width"], height=cached["height"],
                    sharpness=cached["sharpness"], brightness=cached["brightness"], contrast=cached["contrast"],
                    dark_ratio=cached["dark_ratio"], bright_ratio=cached["bright_ratio"],
                    dhash=int(cached["dhash"]), thumb=cached["thumb"],
                    subject_sharpness=cached.get("subject_sharpness", 0.0),
                    face_count=cached.get("face_count", 0), closed_eyes=cached.get("closed_eyes", 0),
                    face_area=cached.get("face_area", 0.0),
                    composition_score=cached.get("composition_score", 0),
                    composition_position_score=cached.get("composition_position_score", 0),
                    composition_balance_score=cached.get("composition_balance_score", 0),
                    composition_edge_score=cached.get("composition_edge_score", 0),
                    composition_tilt=cached.get("composition_tilt", 0.0),
                    composition_notes=cached.get("composition_notes", []),
                    scene_signature=cached.get("scene_signature", ""), scene_histogram=cached.get("scene_histogram", []),
                    raw_path=str(raw_path.relative_to(source)) if raw_path else "",
                    raw_name=raw_path.name if raw_path else "",
                )
                cached_count += 1
            else:
                photo = analyze(path, source, thumbs, cache_id, raw_map, face_map.get(str(path), []))
            photos.append(photo)
            new_cache[cache_key] = {
                key: getattr(photo, key) for key in (
                    "taken_at", "width", "height", "sharpness", "brightness", "contrast",
                    "dark_ratio", "bright_ratio", "dhash", "thumb", "subject_sharpness",
                    "face_count", "closed_eyes", "face_area", "composition_score",
                    "composition_position_score", "composition_balance_score", "composition_edge_score",
                    "composition_tilt", "composition_notes", "scene_signature", "scene_histogram"
                )
            }
        except Exception as exc:
            errors.append({"file": str(path), "error": str(exc)})
        if index % 20 == 0 or index == len(photos_paths):
            print(f"processed {index}/{len(photos_paths)}", flush=True)
    if not photos:
        raise SystemExit("没有成功读取任何照片")

    cache_temp = cache_path.with_suffix(".json.tmp")
    cache_temp.write_text(json.dumps({"version": CACHE_VERSION, "items": new_cache}, ensure_ascii=False), encoding="utf-8")
    cache_temp.replace(cache_path)
    print(f"cache {cached_count}/{len(photos)}", flush=True)

    profile = learn_preferences(out)
    group_feedback = load_group_feedback(out)
    similarity_settings = learn_similarity_settings(photos, out, group_feedback)
    classify(photos, args.mode, profile, group_feedback, similarity_settings)
    write_csv(photos, out)
    write_html(photos, out, source, args.mode, profile, similarity_settings, vision_enabled)
    write_local_launcher(out)
    contact_sheets(photos, out)
    payload = {"source": str(source), "generated_at": datetime.now().isoformat(timespec="seconds"), "mode": args.mode, "vision_enabled": vision_enabled, "learning_profile": profile, "similarity_settings": similarity_settings, "counts": {key: sum(p.category == key for p in photos) for key in LABELS}, "errors": errors, "photos": [asdict(p) for p in photos]}
    (out / "初筛数据.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"photos": len(photos), "counts": payload["counts"], "errors": len(errors), "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
