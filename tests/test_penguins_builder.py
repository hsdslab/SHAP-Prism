from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from examples.palmer_penguins_global_categories import (
    _atomic_publish,
    _render,
    _scientific_payload,
    _write_json_atomic,
)


class TestPenguinsBuilder(unittest.TestCase):
    def test_scientific_payload_separates_environment_and_artifacts(self) -> None:
        expected = {
            "analysis_id": "case",
            "validation": {"rmse": 1.25},
            "software": {"python": "3.12"},
            "artifacts": {"table": {"sha256": "old"}},
        }
        observed = {
            "analysis_id": "case",
            "validation": {"rmse": 1.25},
            "software": {"python": "3.13"},
            "artifacts": {"table": {"sha256": "new"}},
            "verification": {"environment_match": False},
        }
        self.assertEqual(
            _scientific_payload(observed),
            _scientific_payload(expected),
        )

    def test_atomic_publish_replaces_complete_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.txt"
            destination = root / "published" / "result.txt"
            source.write_text("complete\n", encoding="utf-8")
            destination.parent.mkdir()
            destination.write_text("old\n", encoding="utf-8")

            _atomic_publish(source, destination)

            self.assertEqual(destination.read_text(encoding="utf-8"), "complete\n")
            self.assertEqual(
                list(destination.parent.glob(f".{destination.name}.*.tmp")),
                [],
            )

    def test_atomic_json_writer_emits_valid_document(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "run.json"
            _write_json_atomic(target, {"verified": True, "rows": 333})
            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8")),
                {"verified": True, "rows": 333},
            )

    def test_figure_five_uses_right_stacked_category_keys(self) -> None:
        display = pd.DataFrame({"category": ["A", "B"]})
        phi = pd.DataFrame({"category": [-0.1, 0.1]})
        target = "examples.palmer_penguins_global_categories.plot_summary"
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch(
                target,
                return_value=SimpleNamespace(summary={}),
            ) as mocked_plot:
                _render(display, phi, Path(temporary_directory))

        self.assertEqual(
            mocked_plot.call_args.kwargs["category_key_placement"],
            "right_stacked",
        )


if __name__ == "__main__":
    unittest.main()
