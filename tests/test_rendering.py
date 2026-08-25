from __future__ import annotations

import errno
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pypdf import PdfReader

import shap_prism._artifacts as artifact_module
from shap_prism import plot_prism
from shap_prism._artifacts import (
    ArtifactValidationError,
    atomic_save_figure,
    validate_pdf,
)
from tests._fixtures import make_case


class TestRendering(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_png_pdf_and_pixel_smoke(self) -> None:
        phi, frame, groups = make_case(180, 8, 3, categorical=True, unequal=True)
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / "prism"
            result = plot_prism(
                phi,
                frame,
                groups,
                "feature_1",
                title="Categorical rendering smoke test",
                max_points=75,
                output=stem,
                dpi=180,
            )
            png, pdf = stem.with_suffix(".png"), stem.with_suffix(".pdf")
            self.assertEqual(result.summary["output_files"], [str(png), str(pdf)])
            self.assertTrue(png.is_file())
            self.assertTrue(pdf.is_file())
            self.assertGreater(png.stat().st_size, 20_000)
            self.assertGreater(pdf.stat().st_size, 10_000)
            reader = PdfReader(str(pdf), strict=True)
            self.assertGreaterEqual(len(reader.pages), 1)
            image = np.asarray(Image.open(png).convert("RGB"), dtype=np.uint8)
            self.assertGreaterEqual(image.shape[1], 1200)
            self.assertGreaterEqual(image.shape[0], 850)
            nonwhite = np.mean(np.any(image < 248, axis=2))
            self.assertGreater(nonwhite, 0.01)
            self.assertLess(nonwhite, 0.50)
            colored = np.mean((np.max(image, axis=2) - np.min(image, axis=2)) > 18)
            self.assertGreater(colored, 0.002)

    def test_truncated_pdf_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.pdf"
            path.write_bytes(b"%PDF-1.4\n1 0 obj\n")
            with self.assertRaises(ArtifactValidationError):
                validate_pdf(path)

    def test_failed_atomic_export_preserves_existing_target(self) -> None:
        class BrokenFigure:
            @staticmethod
            def savefig(path, **_kwargs) -> None:
                Path(path).write_bytes(b"%PDF-1.4\ntruncated")

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "prism.pdf"
            original = b"previous-valid-publication-file"
            target.write_bytes(original)
            with self.assertRaises(ArtifactValidationError):
                atomic_save_figure(BrokenFigure(), [target], png_dpi=180)
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual(list(Path(directory).glob(".*")), [])

    def test_successful_atomic_export_leaves_no_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            figure, axis = plt.subplots()
            axis.plot([0, 1], [0, 1])
            png = Path(directory) / "prism.png"
            pdf = Path(directory) / "prism.pdf"
            atomic_save_figure(figure, [png, pdf], png_dpi=120)
            self.assertTrue(png.is_file())
            self.assertTrue(pdf.is_file())
            self.assertEqual(list(Path(directory).glob(".*")), [])

    def test_cross_device_publish_uses_validated_local_staging(self) -> None:
        real_replace = os.replace
        raised = False

        def replace_once(source, target) -> None:
            nonlocal raised
            if not raised:
                raised = True
                raise OSError(errno.EXDEV, "simulated cross-device rename")
            real_replace(source, target)

        with tempfile.TemporaryDirectory() as directory:
            figure, axis = plt.subplots()
            axis.plot([0, 1], [1, 0])
            target = Path(directory) / "prism.png"
            with patch.object(artifact_module.os, "replace", side_effect=replace_once):
                atomic_save_figure(figure, [target], png_dpi=120)
            self.assertTrue(target.is_file())
            self.assertEqual(list(Path(directory).glob(".*")), [])

    def test_deterministic_rerun_has_same_summary_and_near_identical_pixels(self) -> None:
        phi, frame, groups = make_case(2000, 8, 4, unequal=True, seed=91)
        first = plot_prism(phi, frame, groups, "feature_1", max_points=80, random_state=17)
        second = plot_prism(phi, frame, groups, "feature_1", max_points=80, random_state=17)
        self.assertEqual(first.summary, second.summary)
        first.figure.canvas.draw()
        second.figure.canvas.draw()
        image_a = np.asarray(first.figure.canvas.buffer_rgba(), dtype=np.int16)
        image_b = np.asarray(second.figure.canvas.buffer_rgba(), dtype=np.int16)
        self.assertEqual(image_a.shape, image_b.shape)
        self.assertLess(float(np.mean(np.abs(image_a - image_b))), 0.01)


if __name__ == "__main__":
    unittest.main()
