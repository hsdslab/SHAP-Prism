from __future__ import annotations

import json
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shap_prism import __version__, plot_prism, plot_summary
from shap_prism.plotting import BLUE, MISSING, PINK
from tests._fixtures import make_case


class TestCategoricalSummary(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_summary_defaults_to_ten_rows_and_is_public(self) -> None:
        phi, frame, _ = make_case(90, 12, 2)
        result = plot_summary(phi, frame, max_points=40)
        self.assertEqual(len(result.summary["displayed_global_features"]), 10)
        self.assertEqual(result.summary["package_version"], __version__)
        self.assertIn("matplotlib", result.summary["software_versions"])
        self.assertNotIn("shap", result.summary["software_versions"])
        self.assertEqual(result.summary["view"], "global_summary")
        self.assertEqual(result.axes["subgroups"], ())

    def test_two_three_and_four_level_rows_have_row_aligned_keys(self) -> None:
        phi, frame, _ = make_case(120, 4, 2)
        frame["feature_1"] = pd.Categorical(["No", "Yes"] * 60)
        frame["feature_2"] = pd.Categorical(["Low", "Mid", "High"] * 40)
        frame["feature_3"] = pd.Categorical(["A", "B", "C", "D"] * 30)
        result = plot_summary(phi, frame, max_features=4)
        keys = result.summary["global_category_keys"]
        self.assertEqual([len(keys[name]["levels"]) for name in keys], [2, 3, 4])
        self.assertIsNotNone(result.axes["category_keys"])
        self.assertTrue(
            any(
                collection.get_gid() == "global-category:feature_1:No"
                for collection in result.axes["global"].collections
            )
        )

    def test_order_modes_and_severity_direction(self) -> None:
        values = np.asarray(["B", "A", "C"] * 20, dtype=object)
        frame = pd.DataFrame({"band": values})
        phi = np.zeros((60, 1))
        phi[np.asarray(values) == "B", 0] = 0.0
        phi[np.asarray(values) == "A", 0] = -1.0
        phi[np.asarray(values) == "C", 0] = 1.0
        data = plot_summary(phi, frame, category_order_mode="data")
        alphabetical = plot_summary(phi, frame, category_order_mode="alphabetical")
        higher = plot_summary(phi, frame, category_order_mode="severity")
        lower = plot_summary(
            phi,
            frame,
            category_order_mode="severity",
            severity_direction="lower",
        )
        self.assertEqual(
            data.summary["global_category_keys"]["band"]["levels"], ["B", "A", "C"]
        )
        self.assertEqual(
            alphabetical.summary["global_category_keys"]["band"]["levels"],
            ["A", "B", "C"],
        )
        self.assertEqual(
            higher.summary["global_category_keys"]["band"]["levels"], ["A", "B", "C"]
        )
        self.assertEqual(
            higher.summary["global_category_keys"]["band"]["order_source"],
            "mean_signed_shap",
        )
        self.assertEqual(
            lower.summary["global_category_keys"]["band"]["levels"], ["C", "B", "A"]
        )
        self.assertEqual(
            higher.summary["global_category_keys"]["band"]["colors"][0],
            BLUE.casefold(),
        )
        self.assertEqual(
            higher.summary["global_category_keys"]["band"]["colors"][-1],
            PINK.casefold(),
        )

    def test_eight_levels_and_missing_have_redundant_encodings(self) -> None:
        phi, frame, _ = make_case(
            160, 2, 2, categorical=True, missing=True, category_levels=8
        )
        result = plot_summary(phi, frame, max_features=2)
        key = result.summary["global_category_keys"]["feature_1"]
        self.assertEqual(len(key["levels"]), 8)
        self.assertEqual(len(set(key["colors"])), 8)
        self.assertEqual(len(set(key["markers"])), 8)
        self.assertTrue(key["missing"])
        self.assertEqual(key["missing_label"], "Missing")
        missing = next(
            collection
            for collection in result.axes["global"].collections
            if collection.get_gid() == "global-category:feature_1:Missing"
        )
        self.assertEqual(
            matplotlib.colors.to_hex(missing.get_facecolors()[0]), MISSING.casefold()
        )

    def test_literal_missing_category_is_distinct_from_actual_na_globally(
        self,
    ) -> None:
        values = np.asarray(["Missing", "A", None] * 20, dtype=object)
        result = plot_summary(
            np.linspace(-1, 1, len(values))[:, None],
            pd.DataFrame({"category": values}),
        )
        key = result.summary["global_category_keys"]["category"]
        self.assertEqual(key["levels"], ["Missing", "A"])
        self.assertEqual(key["missing_label"], "Missing (NA)")
        gids = {
            collection.get_gid()
            for collection in result.axes["global"].collections
        }
        self.assertIn("global-category:category:Missing", gids)
        self.assertIn("global-category:category:Missing (NA)", gids)
        key_text = [
            text.get_text() for text in result.axes["category_keys"].texts
        ]
        self.assertIn("Missing", key_text)
        self.assertIn("Missing (NA)", key_text)

    def test_literal_missing_category_keeps_focal_style_and_summary_identity(
        self,
    ) -> None:
        values = np.asarray(["Missing", "A", None] * 20, dtype=object)
        groups = np.asarray(["G1", "G2"] * 30, dtype=object)
        result = plot_prism(
            np.linspace(-1, 1, len(values))[:, None],
            pd.DataFrame({"category": values}),
            groups,
            "category",
            focal_kind="categorical",
        )
        self.assertEqual(result.summary["missing_focal_label"], "Missing (NA)")
        self.assertEqual(
            result.summary["category_colors"]["Missing"],
            BLUE.casefold(),
        )
        self.assertEqual(
            result.summary["category_colors"]["Missing (NA)"],
            MISSING,
        )
        self.assertEqual(
            result.summary["resolved_category_markers"]["Missing"], "o"
        )
        self.assertEqual(
            result.summary["resolved_category_markers"]["Missing (NA)"], "X"
        )
        self.assertEqual(
            result.summary["global_category_keys"]["category"]["missing_label"],
            "Missing (NA)",
        )
        legend_labels = {
            text.get_text()
            for legend in result.figure.legends
            for text in legend.get_texts()
        }
        self.assertIn("Missing", legend_labels)
        self.assertIn("Missing (NA)", legend_labels)
        json.dumps(result.summary)

    def test_more_than_eight_levels_are_pooled_as_seven_plus_others(self) -> None:
        phi = np.linspace(-1, 1, 90)[:, None]
        values = np.asarray([f"L{i % 9}" for i in range(90)], dtype=object)
        frame = pd.DataFrame({"category": values})
        result = plot_summary(phi, frame)
        key = result.summary["global_category_keys"]["category"]
        self.assertEqual(key["levels"], [f"L{i}" for i in range(7)] + ["Others"])
        self.assertEqual(key["row_counts"], [10] * 7 + [20])
        self.assertEqual(key["original_level_count"], 9)
        self.assertEqual(key["retained_levels"], [f"L{i}" for i in range(7)])
        self.assertEqual(key["retained_level_row_counts"], [10] * 7)
        self.assertEqual(key["collapsed_levels"], ["L7", "L8"])
        self.assertEqual(key["collapsed_row_count"], 20)
        self.assertIsNotNone(key["high_cardinality_rule"])
        self.assertAlmostEqual(
            key["mean_signed_shap"][-1],
            float(np.mean(phi[np.isin(values, ["L7", "L8"]), 0])),
        )
        self.assertEqual(result.summary["unencoded_high_cardinality_features"], [])
        self.assertEqual(
            result.summary["collapsed_high_cardinality_features"], ["category"]
        )
        self.assertTrue(
            any(
                collection.get_gid() == "global-category:category:Others"
                for collection in result.axes["global"].collections
            )
        )
        self.assertFalse(
            any(
                collection.get_gid() == "global-high-cardinality:category"
                for collection in result.axes["global"].collections
            )
        )

    def test_high_cardinality_focal_feature_uses_same_pooling_contract(self) -> None:
        values = np.asarray(
            [f"L{i % 9}" for i in range(180)] + [None, None],
            dtype=object,
        )
        frame = pd.DataFrame({"category": values})
        phi = np.linspace(-1, 1, len(values))[:, None]
        groups = np.asarray(["A", "B"] * 91, dtype=object)
        explicit = [f"L{i}" for i in range(8, -1, -1)]
        result = plot_prism(
            phi,
            frame,
            groups,
            "category",
            category_order=explicit,
            category_palette={
                "L8": "#22577A",
                "L0": "#C8553D",
                "Others": "#5A189A",
            },
            category_markers={"Others": "h"},
        )
        expected = [f"L{i}" for i in range(8, 1, -1)] + ["Others"]
        self.assertEqual(result.summary["resolved_category_order"], expected)
        self.assertEqual(result.summary["resolved_category_row_counts"], [20] * 7 + [40])
        self.assertEqual(result.summary["category_order_source"], "explicit")
        self.assertEqual(result.summary["focal_missing_count"], 2)
        collapse = result.summary["focal_category_collapse"]
        self.assertEqual(collapse["retained_levels"], expected[:-1])
        self.assertEqual(collapse["retained_level_row_counts"], [20] * 7)
        self.assertEqual(collapse["collapsed_levels"], ["L1", "L0"])
        self.assertEqual(collapse["collapsed_row_count"], 40)
        global_key = result.summary["global_category_keys"]["category"]
        self.assertEqual(global_key["levels"], expected)
        self.assertEqual(global_key["order_source"], "explicit")
        self.assertTrue(global_key["missing"])
        self.assertEqual(global_key["missing_row_count"], 2)
        self.assertTrue(
            any(
                collection.get_gid() == "global-category:category:Missing"
                for collection in result.axes["global"].collections
            )
        )
        self.assertEqual(result.summary["category_colors"]["Others"], "#5a189a")
        self.assertEqual(result.summary["resolved_category_markers"]["Others"], "h")
        json.dumps(result.summary)

    def test_high_cardinality_retention_prefers_frequency_before_order(self) -> None:
        values = np.concatenate(
            [np.repeat(f"L{index}", index + 1) for index in range(9)]
        )
        phi = np.linspace(-1, 1, len(values))[:, None]
        frame = pd.DataFrame({"category": values})
        result = plot_summary(phi, frame)
        sampled = plot_summary(phi, frame, max_points=20, random_state=11)
        key = result.summary["global_category_keys"]["category"]
        sampled_key = sampled.summary["global_category_keys"]["category"]
        self.assertEqual(key["retained_levels"], [f"L{i}" for i in range(2, 9)])
        self.assertEqual(key["retained_level_row_counts"], list(range(3, 10)))
        self.assertEqual(key["collapsed_levels"], ["L0", "L1"])
        self.assertEqual(key["collapsed_row_count"], 3)
        self.assertEqual(sampled_key["retained_levels"], key["retained_levels"])
        self.assertEqual(sampled_key["row_counts"], key["row_counts"])

    def test_real_others_value_is_absorbed_by_the_internal_pool(self) -> None:
        raw_levels = ["Others", "A", "B", "C", "D", "E", "F", "G", "H"]
        values = np.asarray(raw_levels * 10, dtype=object)
        result = plot_summary(
            np.linspace(-1, 1, len(values))[:, None],
            pd.DataFrame({"category": values}),
        )
        key = result.summary["global_category_keys"]["category"]
        self.assertEqual(key["levels"].count("Others"), 1)
        self.assertEqual(key["levels"][-1], "Others")
        self.assertIn("Others", key["collapsed_levels"])
        self.assertEqual(key["collapsed_row_count"], 20)

    def test_pooled_literal_missing_still_disambiguates_actual_na(self) -> None:
        values = np.asarray(
            [value for label in "ABCDEFGH" for value in [label] * 10]
            + ["Missing", None],
            dtype=object,
        )
        phi = np.linspace(-1, 1, len(values))[:, None]
        frame = pd.DataFrame({"category": values})

        global_result = plot_summary(phi, frame)
        global_key = global_result.summary["global_category_keys"]["category"]
        self.assertIn("Missing", global_key["collapsed_levels"])
        self.assertEqual(global_key["missing_label"], "Missing (NA)")
        self.assertTrue(
            any(
                collection.get_gid()
                == "global-category:category:Missing (NA)"
                for collection in global_result.axes["global"].collections
            )
        )

        groups = np.asarray(["G1", "G2"] * (len(values) // 2), dtype=object)
        focal_result = plot_prism(
            phi,
            frame,
            groups,
            "category",
            focal_kind="categorical",
        )
        self.assertEqual(
            focal_result.summary["missing_focal_label"], "Missing (NA)"
        )
        self.assertEqual(
            focal_result.summary["global_category_keys"]["category"][
                "missing_label"
            ],
            "Missing (NA)",
        )
        self.assertEqual(
            focal_result.summary["resolved_category_markers"]["Missing (NA)"],
            "X",
        )
        json.dumps(focal_result.summary)

        nonfocal_frame = pd.DataFrame(
            {
                "focal": np.linspace(-2, 2, len(values)),
                "category": values,
            }
        )
        nonfocal_phi = np.column_stack((phi[:, 0], phi[:, 0] * -0.5))
        nonfocal_result = plot_prism(
            nonfocal_phi,
            nonfocal_frame,
            groups,
            "focal",
        )
        self.assertEqual(
            nonfocal_result.summary["global_category_keys"]["category"][
                "missing_label"
            ],
            "Missing (NA)",
        )
        self.assertIn(
            "Missing (NA)",
            [text.get_text() for text in nonfocal_result.axes["category_dock"].texts],
        )

    def test_prism_nonfocal_global_row_uses_pooled_key_in_dock(self) -> None:
        labels = [f"L{index}" for index in range(9)]
        categories = np.asarray(labels * 12 + [None, None], dtype=object)
        n = len(categories)
        frame = pd.DataFrame(
            {
                "focal": np.linspace(-2, 2, n),
                "many_levels": pd.Categorical(
                    categories,
                    categories=labels,
                    ordered=True,
                ),
            }
        )
        phi = np.column_stack(
            [np.linspace(-1, 1, n), np.linspace(0.8, -0.8, n)]
        )
        groups = np.asarray(["A", "B"] * (n // 2), dtype=object)
        result = plot_prism(phi, frame, groups, "focal")
        key = result.summary["global_category_keys"]["many_levels"]
        self.assertEqual(key["levels"], labels[:7] + ["Others"])
        self.assertEqual(key["row_counts"], [12] * 7 + [24])
        self.assertEqual(key["missing_row_count"], 2)
        self.assertEqual(
            result.summary["collapsed_high_cardinality_features"],
            ["many_levels"],
        )
        self.assertEqual(result.summary["category_dock_features"], ["many_levels"])
        self.assertIsNotNone(result.axes["category_dock"])
        dock_text = [text.get_text() for text in result.axes["category_dock"].texts]
        self.assertIn("Others", dock_text)
        self.assertIn("Missing", dock_text)

    def test_continuous_colors_do_not_depend_on_display_sample_size(self) -> None:
        row_ids = np.arange(200, dtype=float)
        phi = row_ids[:, None]
        frame = pd.DataFrame(
            {
                "signal": np.concatenate(
                    [np.linspace(-2.0, 2.0, 190), np.linspace(20.0, 100.0, 10)]
                )
            }
        )
        sampled = plot_summary(
            phi,
            frame,
            max_points=20,
            random_state=7,
        )
        complete = plot_summary(
            phi,
            frame,
            max_points=200,
            random_state=7,
        )
        sampled.figure.canvas.draw()
        complete.figure.canvas.draw()

        def colors_by_row(result):
            collection = next(
                item
                for item in result.axes["global"].collections
                if item.get_gid() == "global-continuous:signal"
            )
            return {
                int(offset[0]): color
                for offset, color in zip(
                    collection.get_offsets(),
                    collection.get_facecolors(),
                    strict=True,
                )
            }

        sampled_colors = colors_by_row(sampled)
        complete_colors = colors_by_row(complete)
        self.assertEqual(len(sampled_colors), 20)
        for row, color in sampled_colors.items():
            np.testing.assert_allclose(color, complete_colors[row], atol=0, rtol=0)

    def test_prism_keeps_focal_key_local_and_uses_global_category_renderer(self) -> None:
        phi, frame, groups = make_case(120, 4, 3, categorical=True)
        frame["feature_2"] = pd.Categorical(["No", "Yes"] * 60)
        result = plot_prism(
            phi,
            frame,
            groups,
            "feature_1",
            category_order_mode="severity",
        )
        self.assertEqual(
            result.summary["focal_category_legend_locations"],
            ["subgroup_header"],
        )
        self.assertIsNotNone(result.axes["focal_category_legend"])
        self.assertIsNotNone(result.axes["category_dock"])
        self.assertIn("feature_1", result.summary["global_category_keys"])
        self.assertTrue(
            any(
                str(collection.get_gid()).startswith("global-category:feature_1:")
                for collection in result.axes["global"].collections
            )
        )
        layout = result.summary["category_dock_layout"]
        self.assertEqual(layout["alignment"], "left-aligned lane tab stops")
        self.assertEqual(
            layout["packing"],
            "compact adjacent fit; at most two feature blocks per row",
        )
        self.assertEqual(layout["rows"], [["feature_2"]])
        self.assertNotIn("feature_1", result.summary["category_dock_features"])
        self.assertEqual(layout["packed_row_count"], 1)
        self.assertEqual(result.summary["summary_schema_version"], "2.0")
        self.assertNotIn("category_tabs", layout)
        for block in layout["blocks"]:
            self.assertTrue(
                all(
                    first < second
                    for first, second in zip(
                        block["category_tabs"],
                        block["category_tabs"][1:],
                    )
                )
            )


if __name__ == "__main__":
    unittest.main()
