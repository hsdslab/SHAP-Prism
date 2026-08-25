"""Minimal categorical-feature example with redundant color and marker cues."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shap_prism import plot_prism


rng = np.random.default_rng(73)
n = 240
groups = np.repeat(["Large opening", "Medium opening", "Small opening"], n // 3)
mask = rng.choice(["A1.5", "A3", "B3", "B6"], size=n)
features = pd.DataFrame(
    {
        "mask": pd.Categorical(mask),
        "panel": rng.normal(size=n),
        "pad": rng.normal(size=n),
        "solder": rng.normal(size=n),
    }
)
shap_values = rng.normal(0, 0.13, size=features.shape)
mask_effect = {"A1.5": -0.45, "A3": -0.25, "B3": 0.18, "B6": 0.62}
opening_scale = {"Large opening": 0.75, "Medium opening": 1.0, "Small opening": 1.45}
shap_values[:, 0] = [mask_effect[value] * opening_scale[group] for value, group in zip(mask, groups, strict=True)]
shap_values[:, 0] += rng.normal(0, 0.05, n)

output = Path(__file__).resolve().parent / "output" / "categorical_prism"
prism = plot_prism(
    shap_values,
    features,
    groups,
    "mask",
    focal_kind="categorical",
    group_order=["Large opening", "Medium opening", "Small opening"],
    title="Synthetic demonstration: mask contribution by opening",
    category_palette=["#1689D8", "#7655C5", "#C44DA5", "#EF4D8B"],
    output=output,
)
print(prism.summary["output_files"])
plt.close(prism.figure)
