"""Deterministic synthetic inputs shared by release tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_case(
    n: int,
    p: int,
    group_count: int,
    *,
    categorical: bool = False,
    unequal: bool = False,
    missing: bool = False,
    category_levels: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    names = [f"feature_{index + 1}" for index in range(p)]
    features = pd.DataFrame(rng.normal(size=(n, p)), columns=names)
    if unequal:
        weights = np.arange(1, group_count + 1, dtype=float)
        counts = np.floor(n * weights / weights.sum()).astype(int)
        counts[-1] += n - int(counts.sum())
    else:
        counts = np.full(group_count, n // group_count, dtype=int)
        counts[-1] += n - int(counts.sum())
    groups = np.concatenate(
        [np.repeat(f"Group {index + 1}", count) for index, count in enumerate(counts)]
    )
    permutation = rng.permutation(n)
    groups = groups[permutation]
    features = features.iloc[permutation].reset_index(drop=True)
    shap_values = 0.18 * features.to_numpy() + rng.normal(0, 0.12, size=(n, p))
    codes = pd.Categorical(groups, [f"Group {index + 1}" for index in range(group_count)]).codes
    shap_values[:, 0] += (codes - np.mean(codes)) * (0.18 + 0.08 * features.iloc[:, 0].to_numpy())
    if categorical:
        levels = (
            np.asarray(["Low", "Middle", "High"], dtype=object)
            if category_levels == 3
            else np.asarray(
                [f"Category {index + 1} with long label" for index in range(category_levels)],
                dtype=object,
            )
        )
        category = levels[np.mod(np.arange(n) + codes, len(levels))]
        features[names[0]] = pd.Categorical(category, categories=levels, ordered=True)
        category_effects = np.linspace(-0.30, 0.34, len(levels))
        category_effect = category_effects[np.mod(np.arange(n) + codes, len(levels))]
        shap_values[:, 0] = category_effect * (0.75 + 0.18 * codes) + rng.normal(0, 0.05, n)
    if missing:
        missing_rows = np.arange(0, n, max(3, n // 12))
        features.iloc[missing_rows, min(1, p - 1)] = np.nan
        if categorical:
            features.iloc[missing_rows, 0] = np.nan
    return shap_values, features, groups


VISUAL_CASES = (
    {"name": "n30_p3_g2_continuous", "n": 30, "p": 3, "g": 2, "categorical": False, "unequal": False, "missing": False},
    {"name": "n180_p8_g3_categorical", "n": 180, "p": 8, "g": 3, "categorical": True, "unequal": True, "missing": True},
    {"name": "n2000_p20_g4_continuous", "n": 2000, "p": 20, "g": 4, "categorical": False, "unequal": True, "missing": True},
    {"name": "n180_p8_g5_continuous", "n": 180, "p": 8, "g": 5, "categorical": False, "unequal": False, "missing": False},
    {"name": "n180_p3_g6_categorical", "n": 180, "p": 3, "g": 6, "categorical": True, "unequal": False, "missing": False},
    {"name": "n180_p4_g3_eight_categories", "n": 180, "p": 4, "g": 3, "categorical": True, "unequal": False, "missing": False, "category_levels": 8},
    {"name": "n180_p4_g3_eight_categories_missing_long_labels", "n": 180, "p": 4, "g": 3, "categorical": True, "unequal": False, "missing": True, "category_levels": 8, "long_labels": True},
)
