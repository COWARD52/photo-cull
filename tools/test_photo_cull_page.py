"""Smoke-test the JavaScript embedded in a generated Mac report."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from photo_cull import Photo, write_html, write_local_launcher


class ReportPageTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is needed for JavaScript syntax checks")
    def test_generated_scripts_parse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            photo = Photo(path=str(root / "a.jpg"), name="a.jpg", taken_at=0, width=100, height=100,
                          sharpness=1, brightness=100, contrast=20, dark_ratio=0, bright_ratio=0,
                          dhash=0, thumb="thumbs/a.jpg", category="review", reason="待确认")
            write_html([photo], root, root, "general", {}, {}, False)
            page = (root / "打开查看.html").read_text(encoding="utf-8")
            self.assertIn('id="memory-open"', page)
            scripts = re.findall(r"<script>(.*?)</script>", page, re.S)
            self.assertEqual(len(scripts), 2)
            for script in scripts:
                result = subprocess.run(["node", "--check"], input=script, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            write_local_launcher(root)
            self.assertTrue((root / "photo_cull_memory.py").is_file())
            self.assertTrue((root / "photo_cull_memory.js").is_file())


if __name__ == "__main__":
    unittest.main()
