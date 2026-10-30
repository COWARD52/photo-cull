"""Offline tests for the user-reviewed preference workflow."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from photo_cull_memory import (DRAFT_FILE, build_snapshot, delete_rule, load_memory,
                               parse_summary, prompt_preferences, save_review, summary_prompt,
                               update_rule, write_json)


class MemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.report = self.base / "_初筛结果"
        self.report.mkdir()
        self.memory = self.base / "memory.json"
        self.env = patch.dict(os.environ, {"PHOTO_CULL_MEMORY_PATH": str(self.memory)})
        self.env.start()
        self.addCleanup(self.env.stop)
        write_json(self.report / "初筛数据.json", {
            "source": str(self.base), "generated_at": "2026-10-31T10:00:00",
            "photos": [
                {"name": "a.jpg", "category": "reject", "score": 45, "scene": "portrait", "reason": "较暗", "group_id": 1},
                {"name": "b.jpg", "category": "best", "score": 85, "scene": "portrait", "reason": "清晰", "group_id": 1},
                {"name": "c.jpg", "category": "best", "score": 80, "scene": "landscape", "reason": "清晰", "group_id": None},
            ],
        })

    def test_snapshot_review_and_prompt(self) -> None:
        final = {"a.jpg": "best", "b.jpg": "reject", "c.jpg": "best"}
        snapshot = build_snapshot(self.report, final)
        self.assertEqual(snapshot["changed"], 2)
        self.assertEqual(snapshot["pairs"][0]["keep"], "a.jpg")
        self.assertIn("a.jpg", summary_prompt(snapshot))
        draft = parse_summary(json.dumps({"rules": [{"text": "人像更喜欢自然表情", "scene": "portrait",
            "dimension": "expression", "evidence": ["a.jpg", "b.jpg"]}]}), snapshot)
        write_json(self.report / DRAFT_FILE, draft)
        self.assertEqual(draft["rules"][0]["confidence"], "guess")
        self.assertEqual(draft["rules"][0]["evidence_count"], 1)
        store = save_review(self.report, snapshot["id"], draft["rules"])
        self.assertEqual(len(store["rules"]), 1)
        self.assertIn("自然表情", prompt_preferences({"portrait"}))
        self.assertEqual(prompt_preferences({"landscape"}), "")
        rule_id = store["rules"][0]["id"]
        update_rule(rule_id, "人像偏好自然表情")
        self.assertIn("偏好自然表情", load_memory()["rules"][0]["text"])
        delete_rule(rule_id)
        self.assertEqual(load_memory()["rules"], [])

    def test_rejects_unverified_evidence(self) -> None:
        snapshot = build_snapshot(self.report, {"a.jpg": "best", "b.jpg": "reject", "c.jpg": "best"})
        draft = parse_summary('{"rules":[{"text":"猜测","scene":"portrait","dimension":"other","evidence":["not-in-report.jpg"]}]}', snapshot)
        self.assertEqual(draft["rules"], [])
        write_json(self.report / DRAFT_FILE, draft)
        with self.assertRaises(ValueError):
            save_review(self.report, snapshot["id"], [{"text": "伪造", "scene": "portrait", "dimension": "other", "evidence": ["not-in-report.jpg"]}])
        self.assertFalse(self.memory.exists())

    def test_no_diagnostic_samples_means_no_paid_summary(self) -> None:
        snapshot = build_snapshot(self.report, {"a.jpg": "reject", "b.jpg": "best", "c.jpg": "best"})
        self.assertEqual(snapshot["samples"], [])
        with self.assertRaises(ValueError):
            summary_prompt(snapshot)


if __name__ == "__main__":
    unittest.main()
