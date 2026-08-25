from __future__ import annotations

import tempfile
import tomllib
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shap_prism import PrismResult, __version__, plot_prism
from shap_prism.plotting import CATEGORY_PALETTES
from tests._fixtures import make_case


class TestPublicAPI(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_pyproject_version_matches_runtime_version(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with (project_root / "pyproject.toml").open("rb") as pyproject_file:
            project_version = tomllib.load(pyproject_file)["project"]["version"]
        self.assertEqual(project_version, __version__)

    def test_dataframe_and_array_inputs(self) -> None:
        phi, frame, groups = make_case(90, 5, 3)
        result = plot_prism(phi, frame, groups, "feature_1", max_points=40)
        self.assertIsInstance(result, PrismResult)
        self.assertEqual(result.summary["view"], "prism")
        self.assertEqual(result.summary["n_features"], 5)
        self.assertEqual(result.summary["layout"], "stacked")
        array_result = plot_prism(
            phi,
            frame.to_numpy(),
            groups,
            0,
            feature_names=list(frame.columns),
            max_points=40,
        )
        self.assertEqual(array_result.summary["focal_feature"], "feature_1")

        mixed = np.empty((60, 2), dtype=object)
        mixed[:, 0] = np.linspace(0.0, 1.0, len(mixed))
        mixed[:, 1] = np.resize(np.asarray(["A", "B"], dtype=object), len(mixed))
        with self.assertRaisesRegex(
            ValueError,
            "mixed numeric/categorical NumPy object arrays are ambiguous",
        ):
            plot_prism(
                phi[:, :2],
                mixed,
                groups,
                0,
                feature_names=["numeric_signal", "category"],
            )

    def test_native_dataframe_columns_are_preserved_and_checked_exactly(self) -> None:
        phi, frame, groups = make_case(60, 3, 2)
        frame.columns = pd.Index([101, 202, 303])
        shap_frame = pd.DataFrame(phi, index=frame.index, columns=frame.columns)
        result = plot_prism(shap_frame, frame, groups, 0)
        self.assertEqual(result.summary["focal_feature"], "101")
        self.assertEqual(
            result.summary["input_alignment"]["shap_to_features"],
            "index and column checked",
        )

        string_columns = pd.DataFrame(
            phi,
            index=frame.index,
            columns=["101", "202", "303"],
        )
        with self.assertRaisesRegex(ValueError, "columns must match exactly"):
            plot_prism(string_columns, frame, groups, 0)
        with self.assertRaisesRegex(ValueError, "feature_names do not match"):
            plot_prism(
                phi,
                frame,
                groups,
                0,
                feature_names=["101", "202", "303"],
            )

    def test_show_mean_can_be_disabled(self) -> None:
        phi, frame, groups = make_case(60, 3, 2)
        result = plot_prism(phi, frame, groups, "feature_1", show_mean=False)
        self.assertFalse(result.summary["show_mean"])
        footer = " ".join(text.get_text() for text in result.figure.texts)
        self.assertNotIn("◇", footer)

    def test_plotting_does_not_change_numpy_global_rng(self) -> None:
        phi, frame, groups = make_case(60, 3, 2)
        np.random.seed(1954)
        state = np.random.get_state()
        plot_prism(phi, frame, groups, "feature_1")
        observed = np.random.random(4)
        np.random.set_state(state)
        expected = np.random.random(4)
        np.testing.assert_array_equal(observed, expected)

    def test_invalid_inputs_have_clear_errors(self) -> None:
        phi, frame, groups = make_case(60, 3, 2)
        cases = [
            (lambda: plot_prism(phi[:-1], frame, groups, "feature_1"), "identical shapes"),
            (lambda: plot_prism(phi, frame, groups[:-1], "feature_1"), "match the number"),
            (lambda: plot_prism(phi, frame, np.repeat("one", 60), "feature_1"), "two through six"),
            (lambda: plot_prism(phi, frame, np.asarray([f"G{i % 7}" for i in range(60)]), "feature_1"), "two through six"),
            (lambda: plot_prism(phi, frame, groups, "absent"), "unknown focal"),
            (lambda: plot_prism(phi, frame, groups, "feature_1", group_order=["Group 1", "other"]), "unknown group"),
            (lambda: plot_prism(phi, frame, groups, "feature_1", max_points=19), "at least 20"),
            (lambda: plot_prism(phi, frame, groups, "feature_1", max_global_features=0), "between 1 and 15"),
        ]
        broken = phi.copy()
        broken[0, 0] = np.nan
        cases.append((lambda: plot_prism(broken, frame, groups, "feature_1"), "finite"))
        missing_groups = groups.astype(object)
        missing_groups[0] = None
        cases.append((lambda: plot_prism(phi, frame, missing_groups, "feature_1"), "cannot contain missing"))
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    call()

    def test_categorical_palette_validation(self) -> None:
        phi, frame, groups = make_case(90, 3, 3, categorical=True)
        with self.assertRaisesRegex(ValueError, "fewer colors"):
            plot_prism(phi, frame, groups, "feature_1", category_palette=["blue"])
        with self.assertRaisesRegex(ValueError, "invalid category color"):
            plot_prism(
                phi,
                frame,
                groups,
                "feature_1",
                category_palette=["blue", "not-a-color", "pink"],
            )
        with self.assertRaisesRegex(ValueError, "fewer markers"):
            plot_prism(
                phi,
                frame,
                groups,
                "feature_1",
                category_markers=["o"],
            )
        with self.assertRaisesRegex(ValueError, "invalid category marker"):
            plot_prism(
                phi,
                frame,
                groups,
                "feature_1",
                category_markers=["o", "not-a-marker", "s"],
            )

    def test_categorical_marker_mapping_is_reflected_in_summary(self) -> None:
        phi, frame, groups = make_case(90, 3, 3, categorical=True)
        observed = list(dict.fromkeys(frame["feature_1"]))
        marker_map = dict(zip(observed, ["o", "^", "D"], strict=True))
        result = plot_prism(
            phi,
            frame,
            groups,
            "feature_1",
            category_markers=marker_map,
        )
        self.assertEqual(result.summary["category_markers"], ["o", "^", "D"])

    def test_ordered_categories_colormap_and_render_metadata(self) -> None:
        phi, frame, groups = make_case(80, 3, 2, categorical=True)
        frame["feature_1"] = frame["feature_1"].cat.reorder_categories(
            ["High", "Middle", "Low"], ordered=True
        )
        result = plot_prism(
            phi,
            frame,
            groups,
            "feature_1",
            continuous_cmap="viridis",
            category_legend_columns=2,
            max_points=30,
            random_state=11,
        )
        self.assertEqual(result.summary["category_labels"], ["High", "Middle", "Low"])
        self.assertEqual(result.summary["package_version"], __version__)
        self.assertTrue(result.summary["rasterized_points"])
        self.assertIn("hybrid", result.summary["pdf_rendering"])
        self.assertIsNotNone(result.summary["sampling_note"])
        self.assertEqual(result.summary["display_parameters"]["category_legend_columns"], 2)

        with self.assertRaisesRegex(ValueError, "every observed level"):
            plot_prism(
                phi,
                frame,
                groups,
                "feature_1",
                category_order=["High", "Middle"],
            )
        with self.assertRaisesRegex(ValueError, "unknown continuous_cmap"):
            plot_prism(phi, frame, groups, "feature_1", continuous_cmap="not-a-cmap")

    def test_continuous_focal_missing_values_are_disclosed(self) -> None:
        phi, frame, groups = make_case(180, 8, 3, categorical=False, missing=True)
        frame.iloc[[0, 12, 24], 0] = np.nan
        result = plot_prism(phi, frame, groups, "feature_1")
        self.assertEqual(result.summary["focal_missing_count"], 3)
        self.assertEqual(sum(result.summary["focal_missing_by_group"].values()), 3)
        self.assertEqual(
            result.summary["missing_focal_encoding"],
            {"color": "#A7B0BC", "marker": "X"},
        )

    def test_pandas_indices_are_checked_when_available(self) -> None:
        phi, frame, groups = make_case(60, 3, 2)
        frame.index = pd.Index([f"row-{index}" for index in range(len(frame))])
        group_series = pd.Series(groups, index=frame.index)
        shap_frame = pd.DataFrame(phi, index=frame.index, columns=frame.columns)
        focal_series = frame["feature_1"].copy()
        result = plot_prism(
            shap_frame,
            frame,
            group_series,
            "feature_1",
            focal_values=focal_series,
        )
        self.assertEqual(
            result.summary["input_alignment"],
            {
                "shap_to_features": "index and column checked",
                "groups_to_features": "index checked",
                "focal_values_to_features": "index checked",
            },
        )

        reversed_index = frame.index[::-1]
        with self.assertRaisesRegex(ValueError, "groups Series index"):
            plot_prism(
                shap_frame,
                frame,
                pd.Series(groups, index=reversed_index),
                "feature_1",
            )
        with self.assertRaisesRegex(ValueError, "focal_values Series index"):
            plot_prism(
                shap_frame,
                frame,
                group_series,
                "feature_1",
                focal_values=pd.Series(focal_series.to_numpy(), index=reversed_index),
            )
        with self.assertRaisesRegex(ValueError, "shap_values and features DataFrame indices"):
            plot_prism(
                shap_frame.set_axis(reversed_index, axis=0),
                frame,
                group_series,
                "feature_1",
            )

    def test_integer_parameters_are_strict_and_bounded(self) -> None:
        phi, frame, groups = make_case(60, 3, 2)
        invalid = [
            {"max_global_features": 2.5},
            {"max_points": 40.5},
            {"random_state": 1.5},
            {"random_state": -1},
            {"random_state": 2**32},
            {"dpi": 99},
            {"dpi": 1201},
            {"category_legend_columns": 1.5},
            {"focal_feature": True},
        ]
        for parameters in invalid:
            focal = parameters.pop("focal_feature", "feature_1")
            with self.subTest(parameters=parameters, focal=focal):
                with self.assertRaises((TypeError, ValueError)):
                    plot_prism(phi, frame, groups, focal, **parameters)

    def test_forced_continuous_rejects_non_numeric_values(self) -> None:
        phi, frame, groups = make_case(60, 3, 2)
        focal_values = pd.Series(["oops", "1"] * 30, dtype=object)
        with self.assertRaisesRegex(ValueError, "non-numeric"):
            plot_prism(
                phi,
                frame,
                groups,
                "feature_1",
                focal_values=focal_values,
                focal_kind="continuous",
            )

    def test_categorical_types_and_default_palette_are_unambiguous(self) -> None:
        phi, frame, groups = make_case(60, 3, 2)
        mixed = pd.Series(([1, 1.0, True] * 20), dtype=object)
        with self.assertRaisesRegex(ValueError, "must not mix Python types"):
            plot_prism(
                phi,
                frame,
                groups,
                "feature_1",
                focal_values=mixed,
                focal_kind="categorical",
            )

        categorical = pd.Series((["A", "B", "C"] * 20), dtype=object)
        result = plot_prism(
            phi,
            frame,
            groups,
            "feature_1",
            focal_values=categorical,
            focal_kind="categorical",
        )
        self.assertEqual(
            list(result.summary["category_colors"].values()),
            [color.casefold() for color in CATEGORY_PALETTES[3]],
        )

    def test_focal_value_override_is_used_by_global_and_local_views(self) -> None:
        phi, frame, groups = make_case(60, 3, 2)
        focal_values = pd.Series(
            ["Low", "High"] * 30,
            index=frame.index,
            dtype=object,
        )
        result = plot_prism(
            phi,
            frame,
            groups,
            "feature_1",
            focal_values=focal_values,
            focal_kind="categorical",
            category_order=["Low", "High"],
        )
        self.assertEqual(result.summary["focal_kind"], "categorical")
        self.assertEqual(
            result.summary["global_category_keys"]["feature_1"]["levels"],
            ["Low", "High"],
        )
        gids = {
            collection.get_gid()
            for collection in result.axes["global"].collections
        }
        self.assertIn("global-category:feature_1:Low", gids)
        self.assertIn("global-category:feature_1:High", gids)
        self.assertNotIn("global-continuous:feature_1", gids)

        categorical_frame = frame.copy()
        categorical_frame["feature_1"] = pd.Categorical(
            ["A", "B"] * 30
        )
        continuous = plot_prism(
            phi,
            categorical_frame,
            groups,
            "feature_1",
            focal_values=np.linspace(-1.0, 1.0, len(frame)),
            focal_kind="continuous",
        )
        self.assertNotIn(
            "feature_1", continuous.summary["global_category_keys"]
        )
        self.assertTrue(
            any(
                collection.get_gid() == "global-continuous:feature_1"
                for collection in continuous.axes["global"].collections
            )
        )

    def test_explicit_category_order_controls_provenance_and_annotation(self) -> None:
        phi, frame, groups = make_case(60, 3, 2)
        levels = np.asarray(["A", "B", "C"] * 20, dtype=object)
        frame["feature_1"] = pd.Categorical(
            levels,
            categories=["A", "B", "C"],
            ordered=True,
        )
        phi[:, 0] = np.select(
            [levels == "A", levels == "B"],
            [-2.0, 0.0],
            default=2.0,
        )
        result = plot_prism(
            phi,
            frame,
            groups,
            "feature_1",
            focal_kind="categorical",
            category_order=["C", "B", "A"],
            category_order_mode="severity",
            severity_direction="higher",
        )
        self.assertEqual(result.summary["category_order_source"], "explicit")
        self.assertEqual(
            result.summary["resolved_category_order"],
            ["C", "B", "A"],
        )
        self.assertEqual(
            result.summary["focal_category_mean_signed_shap"],
            {"C": 2.0, "B": 0.0, "A": -2.0},
        )
        self.assertEqual(
            result.summary["global_category_keys"]["feature_1"]["order_source"],
            "explicit",
        )
        figure_text = " ".join(text.get_text() for text in result.figure.texts)
        self.assertNotIn("mean signed SHAP order", figure_text)

    def test_tuple_valued_categories_are_supported(self) -> None:
        phi, frame, groups = make_case(60, 3, 2)
        tuples = pd.Series(
            [("low", 1), ("high", 2)] * 30,
            index=frame.index,
            dtype=object,
        )
        frame["feature_1"] = tuples
        result = plot_prism(
            phi,
            frame,
            groups,
            "feature_1",
            focal_kind="categorical",
            category_order=[("low", 1), ("high", 2)],
        )
        self.assertEqual(
            result.summary["resolved_category_order"],
            ["('low', 1)", "('high', 2)"],
        )
        self.assertEqual(result.summary["category_order_source"], "explicit")

    def test_continuous_quantile_clipping_is_visible_and_reported(self) -> None:
        phi, frame, groups = make_case(100, 3, 2)
        focal_values = pd.Series(np.linspace(-10.0, 10.0, len(frame)))
        result = plot_prism(
            phi,
            frame,
            groups,
            "feature_1",
            focal_values=focal_values,
            focal_kind="continuous",
        )
        scale = result.summary["continuous_color_scale"]
        self.assertEqual(scale["normalization_quantiles"], [0.05, 0.95])
        self.assertGreater(scale["n_clipped_below"], 0)
        self.assertGreater(scale["n_clipped_above"], 0)
        titles = [axis.get_title() for axis in result.figure.axes]
        self.assertTrue(any("5th–95th pct." in title for title in titles))
        gradient_ticks = [
            label.get_text()
            for axis in result.figure.axes
            for label in axis.get_xticklabels()
        ]
        self.assertTrue(any(label.startswith("≤") for label in gradient_ticks))
        self.assertTrue(any(label.startswith("≥") for label in gradient_ticks))

    def test_blank_group_display_label_is_rejected(self) -> None:
        phi, frame, groups = make_case(40, 3, 2)
        with self.assertRaisesRegex(ValueError, "non-empty strings"):
            plot_prism(
                phi,
                frame,
                groups,
                "feature_1",
                group_labels={"Group 1": "   ", "Group 2": "Second"},
            )

    def test_nullable_categorical_missing_values_are_rendered(self) -> None:
        phi, base_frame, groups = make_case(30, 3, 2)
        nullable_cases = {
            "string": pd.Series(["A", "B", pd.NA] * 10, dtype="string"),
            "boolean": pd.Series([True, False, pd.NA] * 10, dtype="boolean"),
        }
        for dtype_name, focal_values in nullable_cases.items():
            with self.subTest(dtype=dtype_name):
                frame = base_frame.copy()
                frame["feature_1"] = focal_values
                result = plot_prism(
                    phi,
                    frame,
                    groups,
                    "feature_1",
                    show_mean=False,
                )
                result.figure.canvas.draw()
                self.assertEqual(result.summary["focal_kind"], "categorical")
                rendered_points = sum(
                    len(collection.get_offsets())
                    for collection in result.axes["subgroups"][0].collections
                )
                self.assertEqual(rendered_points, len(frame))
                plt.close(result.figure)

    def test_group_identity_is_type_safe_and_supports_tuples(self) -> None:
        phi, frame, _ = make_case(40, 3, 2)
        with self.assertRaisesRegex(ValueError, "unknown group"):
            plot_prism(
                phi,
                frame,
                [1] * 20 + [2] * 20,
                "feature_1",
                group_order=[1, True],
            )

        typed = plot_prism(
            phi,
            frame,
            [1] * 20 + [True] * 20,
            "feature_1",
        )
        self.assertEqual(typed.summary["group_values"], ["1", "True"])
        self.assertEqual(typed.summary["group_counts"], {"1": 20, "True": 20})
        self.assertAlmostEqual(typed.summary["group_mean_shap"]["1"], float(np.mean(phi[:20, 0])))
        self.assertAlmostEqual(typed.summary["group_mean_shap"]["True"], float(np.mean(phi[20:, 0])))

        tuple_groups = [("north", 1)] * 20 + [("south", 2)] * 20
        tuples = plot_prism(
            phi,
            frame,
            tuple_groups,
            "feature_1",
            group_order=[("south", 2), ("north", 1)],
            group_labels={("south", 2): "South", ("north", 1): "North"},
        )
        self.assertEqual(tuples.summary["group_order"], ["South", "North"])
        self.assertEqual(tuples.summary["group_counts"], {"South": 20, "North": 20})

        nested_types = plot_prism(
            phi,
            frame,
            [("cohort", 1)] * 20 + [("cohort", True)] * 20,
            "feature_1",
        )
        self.assertEqual(
            nested_types.summary["group_counts"],
            {"('cohort', 1)": 20, "('cohort', True)": 20},
        )

    def test_display_names_are_validated(self) -> None:
        phi, frame, groups = make_case(30, 3, 2)
        with self.assertRaisesRegex(ValueError, "unknown features"):
            plot_prism(
                phi,
                frame,
                groups,
                "feature_1",
                display_names={"absent": "Unknown"},
            )
        with self.assertRaisesRegex(ValueError, "non-empty strings"):
            plot_prism(
                phi,
                frame,
                groups,
                "feature_1",
                display_names={"feature_1": "  "},
            )
        with self.assertRaisesRegex(ValueError, "non-empty strings"):
            plot_prism(
                phi,
                frame,
                groups,
                "feature_1",
                display_names={"feature_1": 1},  # type: ignore[dict-item]
            )
        with self.assertRaisesRegex(ValueError, "unique label"):
            plot_prism(
                phi,
                frame,
                groups,
                "feature_1",
                display_names={"feature_1": "feature_2"},
            )

    def test_output_suffix_validation(self) -> None:
        phi, frame, groups = make_case(30, 3, 2)
        with tempfile.TemporaryDirectory() as directory:
            invalid_parent = Path(directory) / "must_not_exist"
            with self.assertRaisesRegex(ValueError, "suffix"):
                plot_prism(
                    phi,
                    frame,
                    groups,
                    "feature_1",
                    output=invalid_parent / "prism.svg",
                )
            self.assertFalse(invalid_parent.exists())

    def test_group_display_labels_are_validated_and_reported_consistently(self) -> None:
        phi, frame, groups = make_case(90, 3, 3)
        labels = {"Group 1": "North", "Group 2": "Central", "Group 3": "South"}
        result = plot_prism(
            phi,
            frame,
            groups,
            "feature_1",
            group_order=["Group 1", "Group 2", "Group 3"],
            group_labels=labels,
        )
        self.assertEqual(result.summary["group_order"], ["North", "Central", "South"])
        self.assertEqual(list(result.summary["group_counts"]), ["North", "Central", "South"])
        self.assertEqual(list(result.summary["group_mean_shap"]), ["North", "Central", "South"])
        with self.assertRaisesRegex(ValueError, "unknown groups"):
            plot_prism(phi, frame, groups, "feature_1", group_labels={"absent": "Other"})
        with self.assertRaisesRegex(ValueError, "unique display label"):
            plot_prism(
                phi,
                frame,
                groups,
                "feature_1",
                group_labels={"Group 1": "Same", "Group 2": "Same"},
            )


if __name__ == "__main__":
    unittest.main()
