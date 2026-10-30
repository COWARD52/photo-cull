"""Local HTTP workflow tests; no external AI call or user photos."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request, urlopen

from photo_cull_memory import SNAPSHOT_FILE, build_snapshot, write_json
from photo_cull_server import make_handler


class ServerMemoryTests(unittest.TestCase):
    def test_unparseable_paid_response_is_saved_and_not_repeated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); report = root / "_初筛结果"; report.mkdir()
            write_json(report / "初筛数据.json", {"source": str(root), "generated_at": "run-1", "photos": [
                {"name": "a.jpg", "category": "reject", "score": 40, "scene": "general", "reason": "测试", "thumb": "thumbs/a.jpg"}]})
            build_snapshot(report, {"a.jpg": "best"})
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(report, root))
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/api/memory/summarize"
                request = Request(url, data=b'{"mode":"codex"}', headers={"Content-Type": "application/json"})
                with patch("photo_cull_server.ai_photo_advice", return_value={"advice": "not structured JSON"}) as ai:
                    with urlopen(request, timeout=5) as response:
                        first = json.load(response)
                    with urlopen(request, timeout=5) as response:
                        second = json.load(response)
                    self.assertEqual(first["draft"]["status"], "parse_error")
                    self.assertTrue(second["reused"])
                    ai.assert_called_once()
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=5)

    def test_organize_does_not_call_ai_and_summary_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "photos"; source.mkdir()
            report = source / "_初筛结果"; report.mkdir()
            thumbs = report / "thumbs"; thumbs.mkdir()
            photos = []
            for name, category, score in [("a.jpg", "reject", 45), ("b.jpg", "best", 85)]:
                (source / name).write_bytes(b"test photo")
                (thumbs / name).write_bytes(b"test preview")
                photos.append({"name": name, "category": category, "score": score, "scene": "portrait",
                               "reason": "测试", "group_id": 1, "thumb": "thumbs/" + name})
            write_json(report / "初筛数据.json", {"source": str(source), "generated_at": "2026-10-31T10:00:00", "photos": photos})
            with patch.dict(os.environ, {"PHOTO_CULL_MEMORY_PATH": str(root / "memory.json")}):
                server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(report, source))
                thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
                base = f"http://127.0.0.1:{server.server_port}"
                def call(path: str, payload: dict | None = None) -> dict:
                    request = Request(base + path, data=json.dumps(payload).encode() if payload is not None else None,
                                      headers={"Content-Type": "application/json"} if payload is not None else {})
                    with urlopen(request, timeout=5) as response:
                        return json.load(response)
                try:
                    with patch("photo_cull_server.ai_photo_advice") as ai:
                        organized = call("/api/organize", {"decisions": {"a.jpg": "best", "b.jpg": "reject"}})
                        self.assertTrue(organized["ok"])
                        self.assertEqual(organized["memory_candidates"], 2)
                        ai.assert_not_called()
                        self.assertFalse((root / "memory.json").exists())
                        (report / SNAPSHOT_FILE).unlink()  # Simulate a report organized by the older app version.
                        state = call("/api/memory")
                        self.assertEqual(len(state["snapshot"]["samples"]), 2)
                        ai.return_value = {"advice": json.dumps({"rules": [{"text": "偏好自然表情", "scene": "portrait",
                            "dimension": "expression", "evidence": ["a.jpg", "b.jpg"]}]})}
                        first = call("/api/memory/summarize", {"mode": "codex"})
                        second = call("/api/memory/summarize", {"mode": "codex"})
                        self.assertFalse(first["reused"])
                        self.assertTrue(second["reused"])
                        ai.assert_called_once()
                        rules = first["draft"]["rules"]
                        reviewed = call("/api/memory/review", {"snapshot_id": state["snapshot"]["id"], "rules": rules})
                        self.assertEqual(len(reviewed["memory"]["rules"]), 1)
                        third = call("/api/memory/summarize", {"mode": "codex"})
                        self.assertTrue(third["reused"])
                        ai.assert_called_once()
                        self.assertEqual(call("/api/undo", {})["restored"], 2)
                        self.assertFalse(call("/api/memory")["snapshot"])
                finally:
                    server.shutdown(); server.server_close(); thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
