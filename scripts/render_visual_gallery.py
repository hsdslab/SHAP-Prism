#!/usr/bin/env python3
"""Render the seven-case release gallery and a contact sheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from shap_prism import plot_prism  # noqa: E402
from tests._fixtures import VISUAL_CASES, make_case  # noqa: E402


def build_gallery(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"case_count": len(VISUAL_CASES), "cases": []}
    png_paths: list[Path] = []
    for index, case in enumerate(VISUAL_CASES):
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
        stem = output / case["name"]
        result = plot_prism(
            phi,
            frame,
            groups,
            "feature_1",
            group_order=[f"Group {group + 1}" for group in range(case["g"])],
            group_labels=group_labels,
            title=(
                "Long-title stress test for eight categorical levels and lengthy subgroup labels"
                if long_labels
                else f"Visual validation: {case['name']}"
            ),
            note="Synthetic layout check; not a scientific result",
            focal_value_label=(
                "Focal value with a deliberately long categorical legend title"
                if long_labels
                else None
            ),
            max_global_features=min(8, case["p"]),
            max_points=75,
            random_state=77,
            output=stem,
            dpi=180,
        )
        plt.close(result.figure)
        png_paths.append(stem.with_suffix(".png"))
        manifest["cases"].append({"design": case, "summary": result.summary})

    contact, axes = plt.subplots(4, 2, figsize=(14.7, 20.4), facecolor="white")
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index >= len(png_paths):
            continue
        axis.imshow(plt.imread(png_paths[index]))
        axis.set_title(VISUAL_CASES[index]["name"], loc="left", fontsize=10, weight="bold")
    contact.suptitle("SHAP Prism: release visual validation grid", fontsize=16, weight="bold")
    contact.tight_layout(rect=[0, 0, 1, 0.975])
    gallery = output / "visual_gallery.png"
    contact.savefig(gallery, dpi=150, facecolor="white")
    plt.close(contact)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return gallery


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "visual_checks")
    args = parser.parse_args()
    gallery = build_gallery(args.output)
    print(gallery)


if __name__ == "__main__":
    main()
