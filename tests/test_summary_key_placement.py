from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shap_prism import plot_summary


def _placement_case(
    *,
    levels: tuple[int, ...] = (2, 3, 4),
    numeric_features: int = 2,
    n: int = 240,
    seed: int = 904,
) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    columns: dict[str, object] = {}
    shap_columns: list[np.ndarray] = []
    total_features = len(levels) + numeric_features
    for index, level_count in enumerate(levels):
        labels = tuple(f"L{index + 1}-{level + 1}" for level in range(level_count))
        values = np.resize(np.asarray(labels, dtype=object), n)
        columns[f"category_{index + 1}"] = pd.Categorical(
            values,
            categories=list(labels),
            ordered=True,
        )
        codes = pd.Categorical(values, categories=labels).codes.astype(float)
        codes -= np.mean(codes)
        codes /= max(float(np.mean(np.abs(codes))), 1.0)
        shap_columns.append(
            (1.0 - 0.07 * index) * codes + rng.normal(0, 0.025, n)
        )
    for index in range(numeric_features):
        values = rng.normal(size=n)
        columns[f"numeric_{index + 1}"] = values
        rank = len(levels) + index
        shap_columns.append(
            (1.0 - 0.07 * rank) * values + rng.normal(0, 0.025, n)
        )
    phi = np.column_stack(shap_columns)
    if phi.shape[1] != total_features:
        raise RuntimeError("fixture construction failed")
    return phi, pd.DataFrame(columns)


def _category_encoding(result: object) -> dict[str, tuple[tuple[str, ...], ...]]:
    return {
        feature: (
            tuple(key["levels"]),
            tuple(key["colors"]),
            tuple(key["markers"]),
        )
        for feature, key in result.summary["global_category_keys"].items()
    }


class TestSummaryCategoryKeyPlacement(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_default_is_row_aligned_and_matches_explicit_call(self) -> None:
        parameter = inspect.signature(plot_summary).parameters[
            "category_key_placement"
        ]
        self.assertEqual(parameter.default, "row_aligned")
        phi, frame = _placement_case()
        implicit = plot_summary(phi, frame, max_features=5, random_state=17)
        explicit = plot_summary(
            phi,
            frame,
            max_features=5,
            random_state=17,
            category_key_placement="row_aligned",
        )
        self.assertEqual(
            implicit.summary["category_key_layout"],
            explicit.summary["category_key_layout"],
        )
        np.testing.assert_array_equal(
            implicit.figure.get_size_inches(),
            explicit.figure.get_size_inches(),
        )
        np.testing.assert_array_equal(
            implicit.axes["global"].get_position().bounds,
            explicit.axes["global"].get_position().bounds,
        )
        for first, second in zip(
            implicit.axes["global"].collections,
            explicit.axes["global"].collections,
            strict=True,
        ):
            np.testing.assert_array_equal(
                first.get_offsets(),
                second.get_offsets(),
            )
            np.testing.assert_array_equal(
                first.get_facecolors(),
                second.get_facecolors(),
            )

    def test_invalid_placement_is_rejected(self) -> None:
        phi, frame = _placement_case()
        with self.assertRaisesRegex(
            ValueError,
            "row_aligned.*bottom.*right_stacked",
        ):
            plot_summary(phi, frame, category_key_placement="side")

    def test_all_placements_preserve_data_encoding_and_feature_order(self) -> None:
        phi, frame = _placement_case()
        results = {
            placement: plot_summary(
                phi,
                frame,
                max_features=5,
                category_key_placement=placement,
                random_state=19,
            )
            for placement in ("row_aligned", "bottom", "right_stacked")
        }
        reference = results["row_aligned"]
        for placement, result in results.items():
            with self.subTest(placement=placement):
                self.assertEqual(
                    result.summary["category_key_placement"],
                    placement,
                )
                self.assertEqual(
                    result.summary["display_parameters"][
                        "category_key_placement"
                    ],
                    placement,
                )
                self.assertEqual(
                    result.summary["displayed_global_features"],
                    reference.summary["displayed_global_features"],
                )
                self.assertEqual(
                    result.summary["global_shap_xlim"],
                    reference.summary["global_shap_xlim"],
                )
                self.assertEqual(
                    _category_encoding(result),
                    _category_encoding(reference),
                )

    def test_bottom_dock_is_below_one_wide_panel_and_packs_adaptively(self) -> None:
        phi, frame = _placement_case(levels=(2, 3, 8), numeric_features=2)
        result = plot_summary(
            phi,
            frame,
            max_features=5,
            category_key_placement="bottom",
        )
        result.figure.canvas.draw()
        global_axis = result.axes["global"]
        dock_axis = result.axes["category_dock"]
        self.assertIs(result.axes["category_keys"], dock_axis)
        self.assertIsNotNone(dock_axis)
        self.assertGreater(global_axis.get_position().width, 0.70)
        self.assertLess(
            dock_axis.get_position().y1,
            global_axis.get_position().y0,
        )
        renderer = result.figure.canvas.get_renderer()
        clearance_px = (
            global_axis.get_tightbbox(renderer=renderer).y0
            - dock_axis.get_tightbbox(renderer=renderer).y1
        )
        self.assertGreaterEqual(clearance_px * 72.0 / result.figure.dpi, 4.0)
        layout = result.summary["category_dock_layout"]
        flattened = [feature for row in layout["rows"] for feature in row]
        self.assertEqual(flattened, result.summary["category_key_features"])
        self.assertTrue(any(len(row) == 2 for row in layout["rows"]))
        self.assertEqual(result.axes["subgroups"], ())

    def test_right_stacked_blocks_start_at_top_and_descend_without_overlap(self) -> None:
        phi, frame = _placement_case(levels=(2, 3, 4, 6), numeric_features=2)
        result = plot_summary(
            phi,
            frame,
            max_features=6,
            category_key_placement="right_stacked",
            display_names={
                "category_1": "Compact plan tier",
                "category_2": "Primary contact channel",
                "category_3": "Service region classification",
                "category_4": "Detailed customer segment",
            },
        )
        result.figure.canvas.draw()
        axis = result.axes["category_keys"]
        renderer = result.figure.canvas.get_renderer()
        blocks = result.summary["category_key_layout"]["blocks"]
        self.assertEqual(
            [block["feature"] for block in blocks],
            result.summary["category_key_features"],
        )
        self.assertTrue(
            all(
                first["top"] > second["top"]
                for first, second in zip(blocks, blocks[1:])
            )
        )
        headings = [
            text
            for text in axis.texts
            if str(text.get_gid()).startswith("category-key-heading:")
        ]
        axis_top = axis.get_window_extent(renderer=renderer).y1
        first_top = headings[0].get_window_extent(renderer=renderer).y1
        self.assertLessEqual(
            (axis_top - first_top) * 72.0 / result.figure.dpi,
            5.0,
        )
        feature_boxes: list[tuple[float, float]] = []
        for feature in result.summary["category_key_features"]:
            artists = [
                text
                for text in axis.texts
                if feature in str(text.get_gid())
            ]
            boxes = [artist.get_window_extent(renderer=renderer) for artist in artists]
            feature_boxes.append(
                (min(box.y0 for box in boxes), max(box.y1 for box in boxes))
            )
        for upper, lower in zip(feature_boxes, feature_boxes[1:]):
            self.assertGreaterEqual(
                (upper[0] - lower[1]) * 72.0 / result.figure.dpi,
                4.0,
            )

    def test_two_through_eight_levels_work_in_every_placement(self) -> None:
        phi, frame = _placement_case(
            levels=(2, 3, 4, 5, 6, 7, 8),
            numeric_features=0,
            n=280,
        )
        reference_encoding = None
        for placement in ("row_aligned", "bottom", "right_stacked"):
            with self.subTest(placement=placement):
                result = plot_summary(
                    phi,
                    frame,
                    max_features=7,
                    category_key_placement=placement,
                )
                result.figure.canvas.draw()
                encoding = _category_encoding(result)
                if reference_encoding is None:
                    reference_encoding = encoding
                else:
                    self.assertEqual(encoding, reference_encoding)
                self.assertEqual(
                    sorted(len(value[0]) for value in encoding.values()),
                    [2, 3, 4, 5, 6, 7, 8],
                )
                self.assertIsNone(result.axes["continuous_key"])
                width, height = result.figure.canvas.get_width_height()
                renderer = result.figure.canvas.get_renderer()
                for text in result.figure.findobj(match=matplotlib.text.Text):
                    if not text.get_visible() or not text.get_text():
                        continue
                    box = text.get_window_extent(renderer=renderer)
                    self.assertGreaterEqual(box.x0, -1.0)
                    self.assertLessEqual(box.x1, width + 1.0)
                    self.assertGreaterEqual(box.y0, -1.0)
                    self.assertLessEqual(box.y1, height + 1.0)

    def test_seven_plus_others_and_missing_work_in_every_placement(self) -> None:
        labels = [f"L{index}" for index in range(9)]
        values = np.asarray(labels * 12 + [None, None], dtype=object)
        frame = pd.DataFrame(
            {
                "category": pd.Categorical(
                    values,
                    categories=labels,
                    ordered=True,
                )
            }
        )
        phi = np.linspace(-1, 1, len(frame))[:, None]
        reference_encoding = None
        for placement in ("row_aligned", "bottom", "right_stacked"):
            with self.subTest(placement=placement):
                result = plot_summary(
                    phi,
                    frame,
                    max_points=len(frame),
                    category_key_placement=placement,
                )
                result.figure.canvas.draw()
                key = result.summary["global_category_keys"]["category"]
                self.assertEqual(
                    key["levels"],
                    [f"L{index}" for index in range(7)] + ["Others"],
                )
                self.assertEqual(key["row_counts"], [12] * 7 + [24])
                self.assertEqual(key["collapsed_levels"], ["L7", "L8"])
                self.assertEqual(key["collapsed_row_count"], 24)
                self.assertTrue(key["missing"])
                self.assertEqual(key["missing_row_count"], 2)
                self.assertEqual(
                    result.summary["category_key_layout"]["placement"],
                    placement,
                )
                encoding = _category_encoding(result)
                if reference_encoding is None:
                    reference_encoding = encoding
                else:
                    self.assertEqual(encoding, reference_encoding)
                gids = {
                    collection.get_gid()
                    for collection in result.axes["global"].collections
                }
                self.assertIn("global-category:category:Others", gids)
                self.assertIn("global-category:category:Missing", gids)
                width, height = result.figure.canvas.get_width_height()
                renderer = result.figure.canvas.get_renderer()
                for text in result.axes["category_keys"].texts:
                    if not text.get_visible() or not text.get_text():
                        continue
                    box = text.get_window_extent(renderer=renderer)
                    self.assertGreaterEqual(box.x0, -1.0)
                    self.assertLessEqual(box.x1, width + 1.0)
                    self.assertGreaterEqual(box.y0, -1.0)
                    self.assertLessEqual(box.y1, height + 1.0)

    def test_right_stacked_long_eight_level_key_with_missing_is_contained(
        self,
    ) -> None:
        phi, frame = _placement_case(levels=(8,), numeric_features=2, n=240)
        labels = [f"Operational category {index + 1}" for index in range(8)]
        values = np.resize(np.asarray(labels, dtype=object), len(frame))
        values[::19] = np.nan
        frame["category_1"] = pd.Categorical(
            values,
            categories=labels,
            ordered=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / "long_stacked"
            result = plot_summary(
                phi,
                frame,
                max_features=3,
                category_key_placement="right_stacked",
                display_names={
                    "category_1": (
                        "Extended operating classification with long wording"
                    )
                },
                output=stem,
                dpi=120,
            )
            result.figure.canvas.draw()
            axis = result.axes["category_keys"]
            renderer = result.figure.canvas.get_renderer()
            width, height = result.figure.canvas.get_width_height()
            key = result.summary["global_category_keys"]["category_1"]
            self.assertEqual(len(key["levels"]), 8)
            self.assertTrue(key["missing"])
            for text in axis.texts:
                if not text.get_visible() or not text.get_text():
                    continue
                box = text.get_window_extent(renderer=renderer)
                self.assertGreaterEqual(box.x0, -1.0)
                self.assertLessEqual(box.x1, width + 1.0)
                self.assertGreaterEqual(box.y0, -1.0)
                self.assertLessEqual(box.y1, height + 1.0)
                gid = str(text.get_gid())
                if ":label:" not in gid:
                    continue
                marker_gid = gid.replace(":label:", ":marker:")
                marker = next(
                    collection
                    for collection in axis.collections
                    if collection.get_gid() == marker_gid
                )
                marker_x = axis.transData.transform(
                    marker.get_offsets()[0]
                )[0]
                self.assertGreaterEqual(box.x0 - marker_x, 2.0)
            self.assertTrue(stem.with_suffix(".png").is_file())
            self.assertTrue(stem.with_suffix(".pdf").is_file())

    def test_continuous_only_input_has_no_category_key_heading(self) -> None:
        phi, frame = _placement_case(levels=(), numeric_features=4)
        for placement in ("row_aligned", "bottom", "right_stacked"):
            with self.subTest(placement=placement):
                result = plot_summary(
                    phi,
                    frame,
                    max_features=4,
                    category_key_placement=placement,
                )
                self.assertEqual(result.summary["category_key_features"], [])
                self.assertIsNone(result.summary["category_dock_layout"])
                all_figure_text = [
                    text.get_text() for text in result.figure.texts
                ]
                self.assertNotIn("CATEGORY KEYS", all_figure_text)
                self.assertIsNotNone(result.axes["continuous_key"])

    def test_bottom_output_dimensions_follow_recorded_figure_size(self) -> None:
        phi, frame = _placement_case(levels=(2, 4), numeric_features=1)
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / "summary"
            result = plot_summary(
                phi,
                frame,
                category_key_placement="bottom",
                output=stem,
                dpi=120,
            )
            width, height = result.summary["display_parameters"][
                "figure_size_inches"
            ]
            image = plt.imread(stem.with_suffix(".png"))
            self.assertLessEqual(abs(image.shape[1] - width * 120), 1.0)
            self.assertLessEqual(abs(image.shape[0] - height * 120), 1.0)
            self.assertTrue(stem.with_suffix(".pdf").is_file())


if __name__ == "__main__":
    unittest.main()
