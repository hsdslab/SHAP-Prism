from __future__ import annotations

import json
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shap_prism import plot_prism, plot_summary
from shap_prism.plotting import (
    MISSING,
    OKABE_ITO_CATEGORY_PALETTE,
)
from tests._fixtures import make_case


class TestColorSchemes(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    @staticmethod
    def _pixels(result) -> np.ndarray:
        result.figure.canvas.draw()
        return np.asarray(result.figure.canvas.buffer_rgba()).copy()

    def test_default_is_pixel_identical_to_explicit_prism(self) -> None:
        phi, frame, _ = make_case(
            120,
            5,
            2,
            categorical=True,
            missing=True,
            seed=17,
        )
        implicit = plot_summary(phi, frame, random_state=91)
        explicit = plot_summary(
            phi,
            frame,
            color_scheme="prism",
            random_state=91,
        )
        np.testing.assert_array_equal(self._pixels(implicit), self._pixels(explicit))
        self.assertEqual(implicit.summary["color_scheme"], "prism")
        self.assertEqual(
            implicit.summary["global_category_keys"],
            explicit.summary["global_category_keys"],
        )

    def test_invalid_scheme_is_rejected_by_both_apis(self) -> None:
        phi, frame, groups = make_case(60, 3, 2)
        for call in (
            lambda: plot_summary(phi, frame, color_scheme="rainbow"),
            lambda: plot_prism(
                phi,
                frame,
                groups,
                "feature_1",
                color_scheme="rainbow",
            ),
        ):
            with self.subTest(call=call):
                with self.assertRaisesRegex(ValueError, "prism.*okabe_ito"):
                    call()

    def test_okabe_ito_exact_colors_for_one_through_eight_levels(self) -> None:
        n = 96
        phi = np.linspace(-1.0, 1.0, n)[:, None]
        for count in range(1, 9):
            values = [f"L{index % count + 1}" for index in range(n)]
            frame = pd.DataFrame(
                {"category": pd.Categorical(values, categories=[f"L{i + 1}" for i in range(count)])}
            )
            result = plot_summary(phi, frame, color_scheme="okabe_ito")
            key = result.summary["global_category_keys"]["category"]
            self.assertEqual(
                key["colors"],
                [color.casefold() for color in OKABE_ITO_CATEGORY_PALETTE[:count]],
            )
            self.assertEqual(len(set(key["markers"])), count)

    def test_okabe_ito_keeps_missing_gray_and_uses_cividis_continuous(self) -> None:
        n = 120
        frame = pd.DataFrame(
            {
                "numeric": np.linspace(-2.0, 3.0, n),
                "category": pd.Categorical(
                    np.asarray(["A", "B", "C", "D"] * 30, dtype=object)
                ),
            }
        )
        frame.loc[0, "category"] = np.nan
        phi = np.column_stack(
            [np.linspace(-0.8, 0.8, n), np.sin(np.linspace(0, 5, n))]
        )
        result = plot_summary(phi, frame, color_scheme="okabe_ito")
        self.assertEqual(result.summary["resolved_continuous_cmap"], "cividis")
        missing = next(
            collection
            for collection in result.axes["global"].collections
            if collection.get_gid() == "global-category:category:Missing"
        )
        self.assertEqual(
            matplotlib.colors.to_hex(missing.get_facecolors()[0]),
            MISSING.casefold(),
        )

    def test_continuous_missing_values_use_the_documented_x_marker(self) -> None:
        phi, frame, groups = make_case(90, 3, 3)
        frame.loc[[0, 11, 22], "feature_1"] = np.nan
        summary = plot_summary(phi, frame)
        global_missing = next(
            collection
            for collection in summary.axes["global"].collections
            if collection.get_gid() == "global-continuous-missing:feature_1"
        )
        self.assertEqual(len(global_missing.get_offsets()), 3)
        prism = plot_prism(phi, frame, groups, "feature_1")
        self.assertTrue(
            any(
                str(collection.get_gid()).endswith(":missing")
                for axis in prism.axes["subgroups"]
                for collection in axis.collections
            )
        )

    def test_custom_category_sequence_applies_to_every_summary_feature(self) -> None:
        n = 120
        phi = np.random.default_rng(8).normal(size=(n, 3))
        frame = pd.DataFrame(
            {
                "two": pd.Categorical(["N", "Y"] * 60),
                "three": pd.Categorical(["L", "M", "H"] * 40),
                "numeric": np.linspace(0, 1, n),
            }
        )
        palette = ["#223344", "#AA3377", "#44AA99"]
        for placement in ("row_aligned", "bottom", "right_stacked"):
            result = plot_summary(
                phi,
                frame,
                category_palette=palette,
                category_key_placement=placement,
            )
            keys = result.summary["global_category_keys"]
            normalized = [color.casefold() for color in palette]
            self.assertEqual(keys["two"]["colors"], normalized[:2])
            self.assertEqual(keys["three"]["colors"], normalized)
            self.assertEqual(
                result.summary["display_parameters"]["custom_category_palette"],
                normalized,
            )

    def test_custom_continuous_color_sequence_and_validation(self) -> None:
        phi, frame, _ = make_case(80, 3, 2)
        result = plot_summary(
            phi,
            frame,
            color_scheme="okabe_ito",
            continuous_cmap=["#102A43", "#F6C85F", "#D1495B"],
        )
        self.assertEqual(result.summary["resolved_continuous_cmap"], "shap_prism_custom")
        self.assertEqual(
            result.summary["continuous_color_endpoints"],
            ["#102a43", "#d1495b"],
        )
        with self.assertRaisesRegex(ValueError, "at least two"):
            plot_summary(phi, frame, continuous_cmap=["#102A43"])
        with self.assertRaisesRegex(ValueError, "invalid continuous color"):
            plot_summary(phi, frame, continuous_cmap=["#102A43", "not-a-color"])
        with self.assertRaisesRegex(ValueError, "must be opaque"):
            plot_summary(phi, frame, continuous_cmap=["#102A43", "#D1495B80"])
        transparent = matplotlib.colors.ListedColormap(
            [(0.0, 0.2, 0.8, 0.0), (0.9, 0.1, 0.2, 0.5)],
            name="shap_prism_transparent_test",
        )
        with self.assertRaisesRegex(ValueError, "must be opaque"):
            plot_summary(phi, frame, continuous_cmap=transparent)
        matplotlib.colormaps.register(transparent, force=True)
        try:
            with self.assertRaisesRegex(ValueError, "must be opaque"):
                plot_summary(
                    phi,
                    frame,
                    continuous_cmap="shap_prism_transparent_test",
                )
        finally:
            matplotlib.colormaps.unregister("shap_prism_transparent_test")

    def test_summary_category_palette_validation(self) -> None:
        phi, frame, _ = make_case(90, 3, 2, categorical=True)
        with self.assertRaisesRegex(TypeError, "sequence"):
            plot_summary(phi, frame, category_palette="blue")
        with self.assertRaisesRegex(ValueError, "fewer colors"):
            plot_summary(phi, frame, category_palette=["#0072B2"])
        with self.assertRaisesRegex(ValueError, "invalid category color"):
            plot_summary(
                phi,
                frame,
                category_palette=["#0072B2", "bad", "#D55E00"],
            )
        with self.assertRaisesRegex(ValueError, "must be opaque"):
            plot_summary(
                phi,
                frame,
                category_palette=["#0072B2", "#E69F0080", "#009E73"],
            )
        with self.assertRaisesRegex(ValueError, "must be opaque"):
            plot_prism(
                phi,
                frame,
                make_case(90, 3, 2, categorical=True)[2],
                "feature_1",
                category_palette=["#0072B2", (0.9, 0.3, 0.1, 0.5), "#009E73"],
            )

    def test_prism_focal_and_global_overrides_have_separate_scope(self) -> None:
        phi, frame, groups = make_case(
            120,
            4,
            3,
            categorical=True,
            seed=44,
        )
        frame["feature_2"] = pd.Categorical(["No", "Yes"] * 60)
        focal = {"Low": "#112233", "Middle": "#445566", "High": "#778899"}
        global_palette = ["#AA4499", "#44AA99"]
        result = plot_prism(
            phi,
            frame,
            groups,
            "feature_1",
            color_scheme="okabe_ito",
            category_palette=focal,
            global_category_palette=global_palette,
        )
        self.assertEqual(
            result.summary["category_colors"],
            {key: value.casefold() for key, value in focal.items()},
        )
        self.assertEqual(
            result.summary["global_category_keys"]["feature_2"]["colors"],
            [color.casefold() for color in global_palette],
        )
        self.assertEqual(
            result.summary["global_category_keys"]["feature_1"]["colors"],
            [value.casefold() for value in focal.values()],
        )

    def test_okabe_ito_prism_is_json_serializable_and_marker_redundant(self) -> None:
        phi, frame, groups = make_case(
            180,
            4,
            4,
            categorical=True,
            missing=True,
            category_levels=8,
        )
        result = plot_prism(
            phi,
            frame,
            groups,
            "feature_1",
            color_scheme="okabe_ito",
        )
        self.assertEqual(result.summary["color_scheme"], "okabe_ito")
        self.assertEqual(result.summary["resolved_continuous_cmap"], "cividis")
        self.assertEqual(len(set(result.summary["category_markers"])), 8)
        json.dumps(result.summary)


if __name__ == "__main__":
    unittest.main()
