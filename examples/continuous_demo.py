"""Minimal continuous-feature example using precomputed SHAP-like values."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shap_prism import plot_prism


rng = np.random.default_rng(42)
n = 360
groups = np.repeat(["North", "Central", "South"], n // 3)
features = pd.DataFrame(
    rng.normal(size=(n, 6)), columns=["temperature", "humidity", "wind", "hour", "load", "trend"]
)
shap_values = 0.20 * features.to_numpy() + rng.normal(0, 0.10, size=features.shape)
group_index = pd.Categorical(groups, ["North", "Central", "South"]).codes
shap_values[:, 0] += (group_index - 1) * (0.30 + 0.12 * features["temperature"].to_numpy())

output = Path(__file__).resolve().parent / "output" / "continuous_prism"
prism = plot_prism(
    shap_values,
    features,
    groups,
    "temperature",
    group_order=["North", "Central", "South"],
    title="Synthetic demonstration: temperature by region",
    note="Precomputed illustrative explanations",
    max_points=180,
    random_state=42,
    output=output,
)
print(prism.summary["output_files"])
plt.close(prism.figure)
