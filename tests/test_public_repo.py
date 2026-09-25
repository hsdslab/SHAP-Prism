"""Ensure local-only material cannot enter public manifests or releases."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import build_public_manifest, validate_public_repo


class TestPublicRepository(unittest.TestCase):
    def test_public_scanners_skip_nested_local_only_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "private" / "nested").mkdir(parents=True)
            (root / "private" / "nested" / "source.pdf").write_bytes(b"local")
            (root / "docs").mkdir()
            public = root / "docs" / "overview.png"
            public.write_bytes(b"public")
            for module in (build_public_manifest, validate_public_repo):
                with self.subTest(module=module.__name__), patch.object(module, "ROOT", root):
                    self.assertEqual(module._files(), [public])

    def test_manifest_contains_only_public_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "private").mkdir()
            (root / "private" / "source.pdf").write_bytes(b"local")
            (root / "README.md").write_text("public", encoding="utf-8")
            output = root / "PUBLIC_SOURCE_SHA256SUMS"
            with patch.object(build_public_manifest, "ROOT", root), patch.object(
                build_public_manifest, "OUTPUT", output
            ), redirect_stdout(io.StringIO()):
                build_public_manifest.main()
            entries = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(entries), 1)
            self.assertTrue(entries[0].endswith("  README.md"))

    def test_release_validation_rejects_forced_tracking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "private").mkdir()
            (root / "private" / "source.txt").write_text("local", encoding="utf-8")
            (root / ".gitignore").write_text("/private/\n", encoding="utf-8")
            (root / "pyproject.toml").write_text(
                '[project]\nversion = "0.4.5"\n', encoding="utf-8"
            )
            package = root / "src" / "shap_prism"
            package.mkdir(parents=True)
            (package / "_version.py").write_text(
                '__version__ = "0.4.5"\n', encoding="utf-8"
            )
            for arguments in (["init"], ["add", "-f", "private/source.txt"]):
                subprocess.run(
                    ["git", "-C", str(root), *arguments],
                    capture_output=True, check=True,
                )
            with patch.object(validate_public_repo, "ROOT", root):
                with self.assertRaises(SystemExit) as raised:
                    validate_public_repo.validate()
            report = json.loads(str(raised.exception))
            self.assertIn("local-only folder is tracked by Git", report["errors"])


if __name__ == "__main__":
    unittest.main()
