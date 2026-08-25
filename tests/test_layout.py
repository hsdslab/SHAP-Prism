from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.text import Text

from shap_prism import plot_prism
from tests._fixtures import VISUAL_CASES, make_case


def _relative_luminance(color: str) -> float:
    from matplotlib.colors import to_rgb

    channels = []
    for value in to_rgb(color):
        channels.append(value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast(foreground: str, background: str = "#FFFFFF") -> float:
    first, second = _relative_luminance(foreground), _relative_luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def _top_row_text_boxes(result: object, renderer: object) -> list[object]:
    axes = result.axes["subgroups"][:2]
    texts = [
        artist
        for axis in axes
        for artist in (
            axis.title,
            axis._left_title,
            axis._right_title,
            *axis.texts,
        )
        if artist.get_text()
        and (
            artist in (axis.title, axis._left_title, axis._right_title)
            or artist.get_text().startswith("mean ")
        )
    ]
    return [text.get_window_extent(renderer) for text in texts]


def _dock_case(
    names: list[str],
    levels: list[tuple[str, ...] | None],
    *,
    group_count: int = 3,
    missing_feature: int | None = None,
    seed: int = 320,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 360
    columns: dict[str, object] = {}
    shap_values = np.empty((n, len(names)), dtype=float)
    for index, (name, feature_levels) in enumerate(zip(names, levels, strict=True)):
        if feature_levels is None:
            values = rng.normal(size=n)
            columns[name] = values
            raw = values
        else:
            values = np.asarray(
                [feature_levels[row % len(feature_levels)] for row in range(n)],
                dtype=object,
            )
            if missing_feature == index:
                values[::29] = np.nan
            columns[name] = pd.Categorical(
                values,
                categories=list(feature_levels),
                ordered=True,
            )
            raw = pd.Categorical(values, categories=feature_levels).codes.astype(float)
        raw = raw - float(np.mean(raw))
        raw = raw / float(np.mean(np.abs(raw)))
        shap_values[:, index] = (
            (1.0 - 0.10 * index) * raw
            + rng.normal(0, 0.005, n)
        )
    groups = np.resize(
        np.asarray([f"G{index + 1}" for index in range(group_count)], dtype=object),
        n,
    )
    return shap_values, pd.DataFrame(columns), groups


class TestVisualLayout(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_representative_visual_matrix(self) -> None:
        self.assertEqual(len(VISUAL_CASES), 7)
        for index, case in enumerate(VISUAL_CASES):
            with self.subTest(case=case["name"]):
                phi, frame, groups = make_case(
                    case["n"],
                    case["p"],
                    case["g"],
                    categorical=case["categorical"],
                    unequal=case["unequal"],
                    missing=case["missing"],
                    category_levels=case.get("category_levels", 3),
                    seed=100 + index,
                )
                long_labels = bool(case.get("long_labels", False))
                group_labels = (
                    {
                        group: f"{group} with a long descriptive subgroup name"
                        for group in pd.unique(groups)
                    }
                    if long_labels
                    else None
                )
                result = plot_prism(
                    phi,
                    frame,
                    groups,
                    "feature_1",
                    group_labels=group_labels,
                    title=(
                        "Long-title stress test for eight categorical levels and lengthy subgroup labels"
                        if long_labels
                        else f"Visual check: {case['name']}"
                    ),
                    note=(
                        "Synthetic layout check; not a scientific result"
                        if long_labels
                        else None
                    ),
                    focal_value_label=(
                        "Focal value with a deliberately long categorical legend title"
                        if long_labels
                        else None
                    ),
                    max_global_features=min(8, case["p"]),
                    max_points=75,
                    random_state=77,
                )
                result.figure.canvas.draw()
                self.assertEqual(result.summary["n_groups"], case["g"])
                self.assertEqual(result.summary["n_rows"], case["n"])
                self.assertEqual(result.summary["n_features"], case["p"])
                self.assertEqual(result.summary["focal_kind"], "categorical" if case["categorical"] else "continuous")
                self.assertEqual(result.summary["wrapped_layout"], case["g"] >= 5)
                self.assertLess(
                    result.axes["global"].get_position().x1,
                    result.axes["subgroups"][0].get_position().x0,
                )
                local_limits = [axis.get_xlim() for axis in result.axes["subgroups"]]
                for limits in local_limits[1:]:
                    np.testing.assert_allclose(limits, local_limits[0], rtol=0, atol=1e-12)
                width, height = result.figure.canvas.get_width_height()
                self.assertGreaterEqual(width, 700)
                self.assertGreaterEqual(height, 500)
                renderer = result.figure.canvas.get_renderer()
                for text in result.figure.texts:
                    if not text.get_visible() or not text.get_text():
                        continue
                    box = text.get_window_extent(renderer=renderer)
                    self.assertGreaterEqual(box.x0, -3)
                    self.assertLessEqual(box.x1, width + 3)
                    self.assertGreaterEqual(box.y0, -3)
                    self.assertLessEqual(box.y1, height + 3)

                if long_labels:
                    self.assertEqual(len(result.summary["category_labels"]), 8)
                    self.assertEqual(len(result.summary["category_colors"]), 9)
                    all_text = " ".join(
                        text.get_text() for text in result.figure.findobj(match=Text)
                    )
                    self.assertNotIn("-0.00", all_text)
                    legend = result.figure.legends[0]
                    legend_bottom = legend.get_window_extent(renderer).y0
                    subgroup_top = result.axes["subgroups"][0].get_window_extent(
                        renderer
                    ).y1
                    panel_header = next(
                        text
                        for text in result.figure.texts
                        if text.get_text().startswith("(b)")
                    )
                    note_artist = next(
                        text
                        for text in result.figure.texts
                        if text.get_text()
                        == "Synthetic layout check; not a scientific result"
                    )
                    header_floor = min(
                        panel_header.get_window_extent(renderer).y0,
                        note_artist.get_window_extent(renderer).y0,
                    )
                    legend_box = legend.get_window_extent(renderer)
                    self.assertGreaterEqual(legend_bottom - subgroup_top, 4.0)
                    self.assertGreaterEqual(header_floor - legend_box.y1, 4.0)
                    self.assertGreaterEqual(legend_box.x0, -1.0)
                    self.assertLessEqual(legend_box.x1, width + 1.0)
                    mean_key = next(
                        item
                        for item in result.figure.legends
                        if [text.get_text() for text in item.get_texts()]
                        == ["Subgroup mean"]
                    )
                    mean_box = mean_key.get_window_extent(renderer)
                    focal_title_box = legend.get_title().get_window_extent(
                        renderer
                    )
                    clearance_px = 4.0 * result.figure.dpi / 72.0
                    self.assertGreaterEqual(
                        mean_box.x0 - focal_title_box.x1 + 1e-9,
                        clearance_px,
                    )

    def test_wrapped_categorical_header_clears_titles_and_means(self) -> None:
        phi, frame, groups = make_case(
            120,
            2,
            6,
            categorical=True,
            category_levels=2,
            seed=731,
        )
        result = plot_prism(
            phi,
            frame,
            groups,
            "feature_1",
            focal_kind="categorical",
            focal_value_label="Year value",
            max_global_features=2,
        )
        result.figure.canvas.draw()
        renderer = result.figure.canvas.get_renderer()
        legend = result.axes["focal_category_legend"]
        self.assertIsNotNone(legend)
        legend_box = legend.get_window_extent(renderer)
        panel_header = next(
            text
            for text in result.figure.texts
            if text.get_text().startswith("(b)")
        )
        self.assertGreaterEqual(
            panel_header.get_window_extent(renderer).y0 - legend_box.y1,
            4.0,
        )
        for box in _top_row_text_boxes(result, renderer):
            self.assertGreaterEqual(legend_box.y0 - box.y1, 4.0)

    def test_wrapped_continuous_header_clears_titles_and_means(self) -> None:
        phi, frame, groups = make_case(180, 3, 6, seed=811)
        result = plot_prism(
            phi,
            frame,
            groups,
            "feature_1",
            focal_values=np.geomspace(125, 12_500, len(phi)),
            focal_kind="continuous",
            focal_value_label="Frequency (Hz)",
            max_global_features=3,
        )
        result.figure.canvas.draw()
        renderer = result.figure.canvas.get_renderer()
        key_axis = result.axes["focal_value_key"]
        self.assertIsNotNone(key_axis)
        key_box = key_axis.get_tightbbox(renderer)
        panel_header = next(
            text
            for text in result.figure.texts
            if text.get_text().startswith("(b)")
        )
        self.assertGreaterEqual(
            panel_header.get_window_extent(renderer).y0 - key_box.y1,
            4.0,
        )
        for box in _top_row_text_boxes(result, renderer):
            self.assertGreaterEqual(key_box.y0 - box.y1, 4.0)

    def test_continuous_focal_key_clears_subgroup_mean_key(self) -> None:
        for group_count in (2, 6):
            with self.subTest(group_count=group_count):
                phi, frame, groups = make_case(
                    180,
                    3,
                    group_count,
                    seed=900 + group_count,
                )
                result = plot_prism(
                    phi,
                    frame,
                    groups,
                    "feature_1",
                    focal_kind="continuous",
                    show_mean=True,
                )
                result.figure.canvas.draw()
                renderer = result.figure.canvas.get_renderer()
                key_axis = result.axes["focal_value_key"]
                self.assertIsNotNone(key_axis)
                mean_key = next(
                    legend
                    for legend in result.figure.legends
                    if [text.get_text() for text in legend.get_texts()]
                    == ["Subgroup mean"]
                )
                key_box = key_axis.get_tightbbox(renderer)
                mean_box = mean_key.get_window_extent(renderer)
                self.assertGreaterEqual(mean_box.x0 - key_box.x1, 4.0)

    def test_large_n_sampling_caps_points_without_changing_counts(self) -> None:
        phi, frame, groups = make_case(2000, 20, 4, unequal=True)
        result = plot_prism(phi, frame, groups, "feature_1", max_points=61)
        self.assertEqual(result.summary["global_rendered_rows"], 61)
        self.assertEqual(sum(result.summary["group_counts"].values()), 2000)
        self.assertTrue(all(count <= 61 for count in result.summary["local_rendered_rows"].values()))
        self.assertEqual(result.summary["global_sampling"], "deterministic random sample")
        for group in pd.unique(groups):
            expected = float(np.mean(phi[groups == group, 0]))
            self.assertAlmostEqual(result.summary["group_mean_shap"][str(group)], expected, places=14)

    def test_global_order_is_computed_before_display_sampling(self) -> None:
        phi, frame, groups = make_case(2000, 3, 2, seed=12)
        phi[:] = 0.0
        phi[:, 0] = 0.10
        phi[:, 1] = 0.20
        phi[0, 2] = 1_000.0
        result = plot_prism(phi, frame, groups, "feature_1", max_points=20, random_state=99)
        result.figure.canvas.draw()
        self.assertEqual(result.summary["global_rendered_rows"], 20)
        self.assertEqual(result.summary["global_feature_order"][0], "feature_3")
        self.assertGreaterEqual(result.axes["global"].get_xlim()[1], 1_000.0)
        self.assertEqual(
            list(result.axes["global"].get_xlim()),
            result.summary["global_shap_xlim"],
        )

    def test_categorical_encoding_is_not_color_only(self) -> None:
        phi, frame, groups = make_case(180, 8, 3, categorical=True, missing=True)
        result = plot_prism(phi, frame, groups, "feature_1")
        labels = result.summary["category_labels"]
        markers = result.summary["category_markers"]
        self.assertEqual(len(labels), 3)
        self.assertEqual(len(markers), 3)
        self.assertEqual(len(set(markers)), len(markers))

    def test_all_categorical_global_panel_omits_numeric_colorbar(self) -> None:
        phi, frame, groups = make_case(180, 3, 3, categorical=True)
        for column in frame.columns:
            if pd.api.types.is_numeric_dtype(frame[column]):
                frame[column] = pd.cut(
                    frame[column],
                    bins=3,
                    labels=["Low", "Middle", "High"],
                ).astype(object)
        result = plot_prism(
            phi,
            frame,
            groups,
            "feature_1",
            focal_kind="categorical",
        )
        global_colorbars = [
            axis for axis in result.figure.axes if axis.get_ylabel() == "Feature value"
        ]
        self.assertEqual(global_colorbars, [])

    def test_tiny_shap_scale_uses_distinct_signed_tick_labels(self) -> None:
        phi, frame, groups = make_case(40, 3, 2)
        phi[:, 0] = np.linspace(-1e-4, 1e-4, len(phi))
        result = plot_prism(phi, frame, groups, "feature_1")
        result.figure.canvas.draw()
        labels = [label.get_text() for label in result.axes["subgroups"][0].get_xticklabels()]
        self.assertEqual(len(labels), len(set(labels)))
        self.assertEqual(labels.count("0"), 1)
        self.assertTrue(any(label.startswith("-") and label != "0" for label in labels))
        self.assertTrue(any("e" in label for label in labels))

    def test_tiny_continuous_values_have_distinct_legend_endpoints(self) -> None:
        phi, frame, groups = make_case(40, 3, 2)
        result = plot_prism(
            phi,
            frame,
            groups,
            "feature_1",
            focal_values=np.linspace(1e-8, 2e-8, len(phi)),
            focal_kind="continuous",
        )
        result.figure.canvas.draw()
        gradient_axis = next(
            axis
            for axis in result.figure.axes
            if axis.get_title().startswith("feature_1 value")
        )
        labels = [label.get_text() for label in gradient_axis.get_xticklabels()]
        self.assertEqual(len(labels), 2)
        self.assertNotEqual(labels[0], labels[1])
        self.assertNotIn("0", labels)

    def test_all_zero_focal_shap_has_a_readable_scale(self) -> None:
        phi, frame, groups = make_case(40, 3, 2)
        phi[:] = 0.0
        result = plot_prism(phi, frame, groups, "feature_1")
        result.figure.canvas.draw()
        labels = [label.get_text() for label in result.axes["subgroups"][0].get_xticklabels()]
        self.assertEqual(result.summary["shared_local_shap_limit"], 1.0)
        self.assertEqual(len(labels), len(set(labels)))
        self.assertEqual(labels.count("0"), 1)

    def test_text_colors_meet_normal_contrast_threshold(self) -> None:
        self.assertGreaterEqual(_contrast("#202631"), 7.0)
        self.assertGreaterEqual(_contrast("#667085"), 4.5)

    def test_adaptive_dock_packs_graphical_abstract_keys_in_one_row(self) -> None:
        names = [
            "Plan tier",
            "Context score",
            "Access channel",
            "Exposure",
            "Prior status",
        ]
        levels = [
            ("Basic", "Standard", "Premium"),
            None,
            ("App", "Web", "In person", "Phone"),
            None,
            ("No", "Yes"),
        ]
        phi, frame, groups = _dock_case(names, levels)
        result = plot_prism(
            phi,
            frame,
            groups,
            "Plan tier",
            max_global_features=5,
        )
        layout = result.summary["category_dock_layout"]
        self.assertEqual(
            layout["rows"],
            [["Access channel", "Prior status"]],
        )
        self.assertEqual(layout["packed_row_count"], 1)
        self.assertEqual(
            [feature for row in layout["rows"] for feature in row],
            result.summary["category_dock_features"],
        )
        paired = [block for block in layout["blocks"] if block["row"] == 0]
        self.assertEqual(len(paired), 2)
        self.assertLess(paired[0]["right_edge"], paired[1]["feature_x"])
        self.assertAlmostEqual(
            result.summary["display_parameters"]["figure_size_inches"][1],
            5.25,
            places=2,
        )
        self.assertEqual(
            result.summary["focal_category_legend_locations"],
            ["subgroup_header"],
        )

    def test_short_subgroup_labels_clear_the_panel_divider(self) -> None:
        names = [
            "Plan tier",
            "Context score",
            "Access channel",
            "Exposure",
            "Prior status",
        ]
        levels = [
            ("Basic", "Standard", "Premium"),
            None,
            ("App", "Web", "In person", "Phone"),
            None,
            ("No", "Yes"),
        ]
        phi, frame, _ = _dock_case(names, levels)
        groups = frame["Access channel"].astype(object).to_numpy()
        result = plot_prism(
            phi,
            frame,
            groups,
            "Plan tier",
            max_global_features=5,
        )
        result.figure.canvas.draw()
        renderer = result.figure.canvas.get_renderer()
        canvas_width, _ = result.figure.canvas.get_width_height()
        local_axis = result.axes["subgroups"][0]
        divider = next(
            artist
            for artist in result.figure.artists
            if isinstance(artist, Line2D)
            and len(set(artist.get_xdata())) == 1
        )
        divider_x = float(divider.get_xdata()[0]) * canvas_width
        in_person = next(
            text for text in local_axis.texts if text.get_text().startswith("In person")
        )
        self.assertGreaterEqual(
            in_person.get_window_extent(renderer).x0 - divider_x,
            4.0,
        )
        figure_labels = [text.get_text() for text in result.figure.texts]
        self.assertNotIn("declared/data order", figure_labels)
        self.assertFalse(
            any("all right panels share" in label for label in figure_labels)
        )
        mean_legends = [
            legend
            for legend in result.figure.legends
            if [text.get_text() for text in legend.get_texts()]
            == ["Subgroup mean"]
        ]
        self.assertEqual(len(mean_legends), 1)
        mean_key = mean_legends[0]
        mean_box = mean_key.get_window_extent(renderer)
        focal_key = result.axes["focal_category_legend"]
        self.assertIsNotNone(focal_key)
        focal_title_box = focal_key.get_title().get_window_extent(renderer)
        self.assertAlmostEqual(
            (mean_box.y0 + mean_box.y1) / 2,
            (focal_title_box.y0 + focal_title_box.y1) / 2,
            delta=3.0,
        )
        focal_label = next(
            label
            for label in result.axes["global"].get_yticklabels()
            if label.get_text() == "Plan tier"
        )
        self.assertEqual(focal_label.get_weight(), "bold")
        self.assertIsNone(focal_label.get_bbox_patch())

    def test_adaptive_dock_keeps_wide_key_then_pairs_short_keys(self) -> None:
        names = [
            "target",
            "Detailed category feature with extended descriptive label",
            "B",
            "C",
        ]
        levels = [
            None,
            tuple(f"Level {index + 1} extended" for index in range(8)),
            ("No", "Yes"),
            ("Low", "High"),
        ]
        phi, frame, groups = _dock_case(names, levels)
        result = plot_prism(phi, frame, groups, "target")
        layout = result.summary["category_dock_layout"]
        self.assertEqual(
            layout["rows"],
            [[names[1]], ["B", "C"]],
        )
        self.assertTrue(all(len(row) <= 2 for row in layout["rows"]))

    def test_adaptive_dock_uses_combined_width_for_asymmetric_pair(self) -> None:
        names = [
            "target",
            "Detailed categorical feature with intentionally extended owner label",
            "B",
        ]
        levels = [
            None,
            ("Low", "High"),
            ("No", "Yes"),
        ]
        phi, frame, groups = _dock_case(names, levels)
        result = plot_prism(phi, frame, groups, "target")
        layout = result.summary["category_dock_layout"]
        self.assertEqual(layout["rows"], [[names[1], names[2]]])
        blocks = layout["blocks"]
        self.assertGreater(blocks[0]["right_edge"] - blocks[0]["feature_x"], 0.45)
        self.assertLess(blocks[0]["right_edge"], blocks[1]["feature_x"])

    def test_adaptive_dock_pairs_short_two_to_eight_level_keys(self) -> None:
        for category_count in range(2, 9):
            with self.subTest(category_count=category_count):
                values = tuple(str(index + 1) for index in range(category_count))
                phi, frame, groups = _dock_case(
                    ["target", "A", "B"],
                    [None, values, values],
                    seed=500 + category_count,
                )
                result = plot_prism(phi, frame, groups, "target")
                self.assertEqual(
                    result.summary["category_dock_layout"]["rows"],
                    [["A", "B"]],
                )

    def test_adaptive_dock_dense_wrapped_layout_is_deterministic(self) -> None:
        names = ["target"] + [f"F{index + 1}" for index in range(7)]
        levels = [None] + [
            tuple(
                f"L{value + 1}" + "x" * ((feature_index * 3 + value) % 5)
                for value in range(category_count)
            )
            for feature_index, category_count in enumerate(range(2, 9))
        ]
        phi, frame, groups = _dock_case(
            names,
            levels,
            group_count=6,
            missing_feature=7,
        )
        first = plot_prism(phi, frame, groups, "target", max_global_features=8)
        second = plot_prism(phi, frame, groups, "target", max_global_features=8)
        expected = [["F1", "F2"], ["F3", "F4"], ["F5", "F6"], ["F7"]]
        self.assertEqual(first.summary["category_dock_layout"]["rows"], expected)
        self.assertEqual(
            first.summary["category_dock_layout"],
            second.summary["category_dock_layout"],
        )
        blocks = first.summary["category_dock_layout"]["blocks"]
        for column in (0, 1):
            column_blocks = [
                block for block in blocks if block["column"] == column
            ]
            self.assertTrue(column_blocks)
            for ordinal in range(
                max(len(block["category_tabs"]) for block in column_blocks)
            ):
                ordinal_tabs = [
                    block["category_tabs"][ordinal]
                    for block in column_blocks
                    if len(block["category_tabs"]) > ordinal
                ]
                np.testing.assert_allclose(
                    ordinal_tabs,
                    np.repeat(ordinal_tabs[0], len(ordinal_tabs)),
                    rtol=0,
                    atol=1e-12,
                )
        self.assertTrue(first.summary["wrapped_layout"])
        self.assertTrue(first.summary["global_category_keys"]["F7"]["missing"])


if __name__ == "__main__":
    unittest.main()
