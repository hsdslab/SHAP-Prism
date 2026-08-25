#!/usr/bin/env python3
"""Build and execute the fully synthetic SHAP Prism quick-start notebook.

The release environment does not require Jupyter. This builder emits a
standards-compliant notebook and executes its code cells in one Python process
using only the standard library plus the package runtime dependencies.
"""

from __future__ import annotations

import argparse
import ast
import base64
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import traceback
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = Path(__file__).with_name("shap_prism_quickstart.ipynb")


def _source_lines(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def _markdown(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": _source_lines(text)}


def _code(text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source_lines(text),
    }


def _notebook() -> dict[str, Any]:
    cells = [
        _markdown(
            """
# SHAP Prism: fully synthetic quick start

This notebook generates its feature matrix and additive contribution matrix in
memory. It downloads, reads, or redistributes no empirical data. The example
shows both the categorical-aware global summary and the subgroup Prism view.
"""
        ),
        _markdown(
            """
## Build a deterministic synthetic example

The synthetic output is the sum of a baseline and five known feature
contributions. For this additive construction, the contribution matrix is
already aligned and ready for SHAP Prism. The plotting package does not fit a
model or estimate explanations.

`Acquisition channel` deliberately has ten levels. SHAP Prism keeps seven raw
levels under its documented frequency-and-tie rule and pools the remaining
levels into `Others`, preserving all rows while keeping an eight-entry
categorical key.
"""
        ),
        _code(
            """
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ["SOURCE_DATE_EPOCH"] = "1784332800"
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "shap-prism-mpl"),
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src/shap_prism"
        ).is_dir():
            return candidate
    raise FileNotFoundError("Run this notebook from within the release repository.")


ROOT = find_repository_root(Path.cwd().resolve())
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from shap_prism import __version__, plot_prism, plot_summary

RANDOM_STATE = 4701
rng = np.random.default_rng(RANDOM_STATE)
n_rows = 240

segment_values = np.repeat(["North", "Central", "South"], 80)
rng.shuffle(segment_values)

channel_levels = [f"C{index:02d}" for index in range(1, 11)]
channel_counts = [36, 32, 28, 26, 24, 22, 20, 18, 17, 17]
channel_values = np.concatenate(
    [np.repeat(level, count) for level, count in zip(channel_levels, channel_counts)]
)
rng.shuffle(channel_values)

risk_levels = ["Low", "Moderate", "High", "Critical"]
risk_values = np.tile(risk_levels, n_rows // len(risk_levels))
rng.shuffle(risk_values)

exposure = rng.normal(0.0, 1.0, n_rows)
stability = rng.beta(3.0, 2.0, n_rows)

features = pd.DataFrame(
    {
        "Exposure": exposure,
        "Stability": stability,
        "Acquisition channel": pd.Categorical(
            channel_values, categories=channel_levels, ordered=True
        ),
        "Risk band": pd.Categorical(
            risk_values, categories=risk_levels, ordered=True
        ),
        "Segment": pd.Categorical(
            segment_values, categories=["North", "Central", "South"], ordered=True
        ),
    }
)

channel_effect = dict(zip(channel_levels, np.linspace(-0.42, 0.42, 10)))
risk_effect = {"Low": -0.62, "Moderate": -0.16, "High": 0.28, "Critical": 0.72}
segment_effect = {"North": -0.18, "Central": 0.02, "South": 0.22}

shap_values = pd.DataFrame(
    {
        "Exposure": 0.68 * exposure,
        "Stability": -0.82 * (stability - stability.mean()),
        "Acquisition channel": np.array([channel_effect[value] for value in channel_values]),
        "Risk band": np.array([risk_effect[value] for value in risk_values]),
        "Segment": np.array([segment_effect[value] for value in segment_values]),
    }
)

baseline = 1.25
synthetic_output = baseline + shap_values.sum(axis=1)
groups = features["Segment"].copy()

assert features.shape == shap_values.shape == (240, 5)
assert features.index.equals(shap_values.index)
assert features.columns.equals(shap_values.columns)
assert np.isfinite(shap_values.to_numpy(float)).all()
assert np.allclose(baseline + shap_values.sum(axis=1), synthetic_output)

OUTPUT_DIR = Path("notebooks/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"shap-prism {__version__}")
print(f"synthetic feature and contribution matrices: {features.shape}")
print(f"group counts: {groups.value_counts(sort=False).to_dict()}")
print("No external or empirical data were loaded.")
"""
        ),
        _markdown(
            """
## 1. Categorical-aware global summary

The category keys are stacked from the upper right downward. Continuous rows
retain the continuous color map. The ten-level channel row appears as seven
retained levels plus `Others`.
"""
        ),
        _code(
            """
SUMMARY_STEM = OUTPUT_DIR / "synthetic_shap_summary"

summary_result = plot_summary(
    shap_values,
    features,
    category_order_mode="data",
    category_key_placement="right_stacked",
    x_label="Synthetic additive contribution",
    max_features=5,
    random_state=RANDOM_STATE,
    output=SUMMARY_STEM,
)

print("feature order:", summary_result.summary["global_feature_order"])
print("saved:", summary_result.summary["output_files"])
summary_result.figure
"""
        ),
        _markdown(
            """
## 2. SHAP Prism subgroup view

The right panel expands `Risk band` across three analyst-declared segments.
Every subgroup uses one shared contribution scale, and hollow diamonds show
signed means computed from all rows.
"""
        ),
        _code(
            """
PRISM_STEM = OUTPUT_DIR / "synthetic_shap_prism"

prism_result = plot_prism(
    shap_values,
    features,
    groups=groups,
    focal_feature="Risk band",
    focal_kind="categorical",
    group_order=["North", "Central", "South"],
    category_order=risk_levels,
    category_order_mode="data",
    global_x_label="Synthetic additive contribution",
    local_x_label="Risk-band contribution",
    focal_value_label="Risk band",
    max_global_features=5,
    show_mean=True,
    random_state=RANDOM_STATE,
    output=PRISM_STEM,
)

print("group counts:", prism_result.summary["group_counts"])
print("saved:", prism_result.summary["output_files"])
prism_result.figure
"""
        ),
        _markdown(
            """
## 3. Save the machine-readable rendering records

A suffix-free output stem writes both PNG and hybrid PDF. The returned summary
is JSON-safe and records the resolved ordering, category encodings, limits,
sampling, and output paths.
"""
        ),
        _code(
            """
SUMMARY_JSON = OUTPUT_DIR / "synthetic_shap_summary.json"
PRISM_JSON = OUTPUT_DIR / "synthetic_shap_prism_summary.json"

for result, target in (
    (summary_result, SUMMARY_JSON),
    (prism_result, PRISM_JSON),
):
    target.write_text(
        json.dumps(result.summary, indent=2, sort_keys=True) + "\\n",
        encoding="utf-8",
    )

expected_outputs = [
    SUMMARY_STEM.with_suffix(".png"),
    SUMMARY_STEM.with_suffix(".pdf"),
    SUMMARY_JSON,
    PRISM_STEM.with_suffix(".png"),
    PRISM_STEM.with_suffix(".pdf"),
    PRISM_JSON,
]
assert all(path.is_file() and path.stat().st_size > 0 for path in expected_outputs)
summary_record = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
assert summary_record["n_features"] == 5
channel_key = summary_record["global_category_keys"]["Acquisition channel"]
assert channel_key["levels"] == channel_levels[:7] + ["Others"]
assert channel_key["retained_level_row_counts"] == channel_counts[:7]
assert channel_key["collapsed_levels"] == channel_levels[7:]
assert channel_key["collapsed_row_count"] == sum(channel_counts[7:])
assert json.loads(PRISM_JSON.read_text(encoding="utf-8"))["group_counts"] == {
    "North": 80,
    "Central": 80,
    "South": 80,
}

print("verified synthetic outputs:")
for path in expected_outputs:
    print(f"  {path} ({path.stat().st_size:,} bytes)")

plt.close(summary_result.figure)
plt.close(prism_result.figure)
"""
        ),
        _markdown(
            """
## Interpretation boundary

Horizontal position is a synthetic additive contribution. Vertical spread
only prevents overlap; it is not a density estimate. Colors and markers
identify categorical values. The subgroup comparison is descriptive, not
causal, inferential, or a fairness assessment.

Replace the synthetic matrices with aligned explanations from one documented
model output before using SHAP Prism in an empirical analysis.
"""
        ),
    ]
    cell_ids = [
        "title",
        "synthetic-scope",
        "build-synthetic-inputs",
        "summary-intro",
        "plot-summary",
        "prism-intro",
        "plot-prism",
        "export-intro",
        "export-summaries",
        "interpretation",
    ]
    for cell, cell_id in zip(cells, cell_ids, strict=True):
        cell["id"] = cell_id
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
            "shap_prism": {
                "builder": "notebooks/build_shap_prism_quickstart.py",
                "data_scope": "fully synthetic; generated in notebook",
                "random_state": 4701,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _compile_cell(source: str) -> tuple[Any, Any | None]:
    parsed = ast.parse(source, filename=str(NOTEBOOK_PATH), mode="exec")
    expression = None
    if parsed.body and isinstance(parsed.body[-1], ast.Expr):
        final = parsed.body.pop()
        expression = compile(
            ast.Expression(final.value), filename=str(NOTEBOOK_PATH), mode="eval"
        )
    body = compile(parsed, filename=str(NOTEBOOK_PATH), mode="exec")
    return body, expression


def _display_output(value: Any, execution_count: int) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "savefig"):
        buffer = io.BytesIO()
        value.savefig(
            buffer,
            format="png",
            dpi=130,
            facecolor="white",
            bbox_inches="tight",
        )
        return {
            "data": {
                "image/png": base64.b64encode(buffer.getvalue()).decode("ascii"),
                "text/plain": repr(value),
            },
            "metadata": {},
            "output_type": "display_data",
        }
    return {
        "data": {"text/plain": repr(value)},
        "execution_count": execution_count,
        "metadata": {},
        "output_type": "execute_result",
    }


def _execute(notebook: dict[str, Any]) -> None:
    namespace: dict[str, Any] = {"__name__": "__main__"}
    execution_count = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        execution_count += 1
        cell["execution_count"] = execution_count
        stdout = io.StringIO()
        stderr = io.StringIO()
        source = "".join(cell["source"])
        try:
            body, expression = _compile_cell(source)
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exec(body, namespace)
                value = eval(expression, namespace) if expression is not None else None
        except Exception as error:
            if stdout.getvalue():
                cell["outputs"].append(
                    {"name": "stdout", "output_type": "stream", "text": stdout.getvalue()}
                )
            if stderr.getvalue():
                cell["outputs"].append(
                    {"name": "stderr", "output_type": "stream", "text": stderr.getvalue()}
                )
            cell["outputs"].append(
                {
                    "ename": type(error).__name__,
                    "evalue": str(error),
                    "output_type": "error",
                    "traceback": traceback.format_exc().splitlines(),
                }
            )
            raise
        if stdout.getvalue():
            cell["outputs"].append(
                {"name": "stdout", "output_type": "stream", "text": stdout.getvalue()}
            )
        if stderr.getvalue():
            cell["outputs"].append(
                {"name": "stderr", "output_type": "stream", "text": stderr.getvalue()}
            )
        displayed = _display_output(value, execution_count)
        if displayed is not None:
            cell["outputs"].append(displayed)


def _write(notebook: dict[str, Any]) -> None:
    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Build the notebook without executing its code cells.",
    )
    args = parser.parse_args()

    os.chdir(REPOSITORY_ROOT)
    notebook = _notebook()
    if not args.source_only:
        _execute(notebook)
    _write(notebook)
    mode = "source notebook" if args.source_only else "executed notebook"
    print(f"Wrote {mode}: {NOTEBOOK_PATH.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    main()
