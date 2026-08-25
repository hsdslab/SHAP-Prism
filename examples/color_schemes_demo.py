"""Render the built-in Prism and Okabe-Ito color schemes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shap_prism import plot_prism, plot_summary


rng = np.random.default_rng(204)
n = 240
groups = np.repeat(["Urban", "Suburban", "Rural"], [96, 84, 60])
delivery = np.tile(["Locker", "Home", "Pickup", "In person"], n // 4)
features = pd.DataFrame(
    {
        "delivery_mode": pd.Categorical(
            delivery,
            categories=["Locker", "Home", "Pickup", "In person"],
            ordered=True,
        ),
        "priority": pd.Categorical(np.tile(["Standard", "Express", "Urgent"], n // 3)),
        "weekend": pd.Categorical(np.tile([False, True], n // 2)),
        "route_load": rng.normal(0, 1, n),
    }
)
shap_values = rng.normal(0, 0.10, size=features.shape)
mode_effect = {"Locker": -0.25, "Home": 0.15, "Pickup": -0.05, "In person": 0.32}
group_scale = {"Urban": 0.70, "Suburban": 1.00, "Rural": 1.35}
shap_values[:, 0] = np.asarray(
    [
        mode_effect[mode] * group_scale[group]
        for mode, group in zip(delivery, groups, strict=True)
    ]
) + rng.normal(0, 0.06, n)

output = Path(__file__).resolve().parent / "output"
output.mkdir(parents=True, exist_ok=True)

for scheme in ("prism", "okabe_ito"):
    summary = plot_summary(
        shap_values,
        features,
        color_scheme=scheme,
        category_key_placement="bottom",
        title=f"Standalone summary · {scheme}",
        output=output / f"summary_{scheme}",
    )
    prism = plot_prism(
        shap_values,
        features,
        groups,
        "delivery_mode",
        color_scheme=scheme,
        group_order=["Urban", "Suburban", "Rural"],
        title=f"Subgroup comparison · {scheme}",
        output=output / f"prism_{scheme}",
    )
    plt.close(summary.figure)
    plt.close(prism.figure)
