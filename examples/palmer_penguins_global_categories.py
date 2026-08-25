"""Build the subgroup-free Palmer Penguins categorical-summary example.

The example predicts flipper length from two numeric measurements and five
categorical source features.  Five-fold out-of-fold TreeSHAP values are
computed for a one-hot random forest, then one-hot contributions are summed
back to the seven displayed source features.  This preserves row-wise
additivity on the model-output scale. Empirical inputs are not bundled; pass
user-acquired files from an external workspace on the command line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


# Fix PDF metadata timestamps and keep Matplotlib's cache in a writable,
# location-independent directory. Exact bytes are only expected in the pinned
# release environment.
os.environ["SOURCE_DATE_EPOCH"] = "1784332800"
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "shap-prism-mpl")
)

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from shap_prism import plot_summary


RAW_DATA: Path
EXPLANATION_DATA: Path
EXPECTED_MANIFEST: Path
RUN_MANIFEST_NAME = "palmer_penguins_run_manifest.json"

TARGET = "flipper_length_mm"
RAW_CATEGORICAL = (
    "sex",
    "species",
    "island",
    "year",
    "body_mass_quartile",
)
RAW_NUMERIC = ("bill_length_mm", "bill_depth_mm")
DISPLAY_NAMES = {
    "sex": "Sex",
    "species": "Species",
    "island": "Island",
    "year": "Year",
    "body_mass_quartile": "Body-mass quartile",
    "bill_length_mm": "Bill length",
    "bill_depth_mm": "Bill depth",
}
CATEGORY_LEVELS = (
    ("female", "male"),
    ("Adelie", "Chinstrap", "Gentoo"),
    ("Biscoe", "Dream", "Torgersen"),
    ("2007", "2008", "2009"),
    ("Q1", "Q2", "Q3", "Q4"),
)
N_SPLITS = 5
RANDOM_STATE = 20_260_728


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _scientific_payload(manifest: dict[str, object]) -> dict[str, object]:
    """Return the scientific record without environment or file identity."""

    return {
        key: value
        for key, value in manifest.items()
        if key not in {"artifacts", "software", "verification"}
    }


def _atomic_publish(source: Path, destination: Path) -> None:
    """Copy one staged artifact into place through an atomic same-dir rename."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    """Write JSON without exposing a partial manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_analysis_frame() -> tuple[pd.DataFrame, np.ndarray]:
    raw = pd.read_csv(RAW_DATA, na_values=["NA"])
    needed = (
        "species",
        "island",
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "sex",
        "year",
    )
    frame = raw.dropna(subset=list(needed)).copy()
    frame.insert(0, "source_row", frame.index.to_numpy(int) + 1)
    frame["year"] = frame["year"].astype(int).astype(str)
    quartile, edges = pd.qcut(
        frame["body_mass_g"],
        q=4,
        labels=CATEGORY_LEVELS[-1],
        retbins=True,
    )
    frame["body_mass_quartile"] = quartile
    for feature, levels in zip(
        RAW_CATEGORICAL,
        CATEGORY_LEVELS,
        strict=True,
    ):
        frame[feature] = pd.Categorical(
            frame[feature],
            categories=levels,
            ordered=(feature == "body_mass_quartile"),
        )
    return frame.reset_index(drop=True), np.asarray(edges, dtype=float)


def _fit_oof(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    import shap
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import OneHotEncoder

    model_features = list(RAW_CATEGORICAL + RAW_NUMERIC)
    x = frame.loc[:, model_features]
    y = frame[TARGET].to_numpy(float)
    n_rows = len(frame)
    oof_prediction = np.empty(n_rows, dtype=float)
    oof_baseline = np.empty(n_rows, dtype=float)
    oof_fold = np.empty(n_rows, dtype=int)
    grouped_phi = np.empty((n_rows, len(model_features)), dtype=float)
    fold_records: list[dict[str, object]] = []

    splitter = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    for fold_index, (train_index, test_index) in enumerate(splitter.split(x), start=1):
        encoder = OneHotEncoder(
            categories=[list(levels) for levels in CATEGORY_LEVELS],
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float64,
        )
        preprocessor = ColumnTransformer(
            (
                ("categorical", encoder, list(RAW_CATEGORICAL)),
                ("numeric", "passthrough", list(RAW_NUMERIC)),
            ),
            verbose_feature_names_out=False,
        )
        train_encoded = np.asarray(
            preprocessor.fit_transform(x.iloc[train_index]),
            dtype=float,
        )
        test_encoded = np.asarray(
            preprocessor.transform(x.iloc[test_index]),
            dtype=float,
        )
        model = RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=RANDOM_STATE + fold_index - 1,
            n_jobs=1,
        )
        model.fit(train_encoded, y[train_index])
        prediction = model.predict(test_encoded)
        explainer = shap.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent",
        )
        encoded_phi = np.asarray(
            explainer.shap_values(test_encoded, check_additivity=True),
            dtype=float,
        )
        baseline = float(np.asarray(explainer.expected_value).reshape(-1)[0])

        offset = 0
        for feature_index, levels in enumerate(CATEGORY_LEVELS):
            width = len(levels)
            grouped_phi[test_index, feature_index] = np.sum(
                encoded_phi[:, offset : offset + width],
                axis=1,
            )
            offset += width
        for numeric_index in range(len(RAW_NUMERIC)):
            grouped_phi[
                test_index,
                len(RAW_CATEGORICAL) + numeric_index,
            ] = encoded_phi[:, offset + numeric_index]

        encoded_error = float(
            np.max(np.abs(baseline + encoded_phi.sum(axis=1) - prediction))
        )
        grouped_error = float(
            np.max(
                np.abs(
                    baseline
                    + grouped_phi[test_index].sum(axis=1)
                    - prediction
                )
            )
        )
        oof_prediction[test_index] = prediction
        oof_baseline[test_index] = baseline
        oof_fold[test_index] = fold_index
        fold_records.append(
            {
                "fold": fold_index,
                "train_rows": int(len(train_index)),
                "test_rows": int(len(test_index)),
                "tree_shap_expected_value_mm": baseline,
                "max_encoded_additivity_error_mm": encoded_error,
                "max_grouped_additivity_error_mm": grouped_error,
            }
        )

    display = frame.loc[:, model_features].rename(columns=DISPLAY_NAMES).copy()
    display["Year"] = pd.Categorical(
        display["Year"],
        categories=CATEGORY_LEVELS[3],
        ordered=True,
    )
    phi = pd.DataFrame(
        grouped_phi,
        columns=[DISPLAY_NAMES[name] for name in model_features],
    )

    released = pd.DataFrame(
        {
            "source_row": frame["source_row"],
            "fold": oof_fold,
            "observed_flipper_length_mm": y,
            "predicted_flipper_length_mm": oof_prediction,
            "fold_baseline_mm": oof_baseline,
        }
    )
    for feature in display.columns:
        released[f"feature__{feature}"] = display[feature].astype(object)
        released[f"shap__{feature}"] = phi[feature]

    metrics = {
        "r2": float(r2_score(y, oof_prediction)),
        "rmse_mm": float(root_mean_squared_error(y, oof_prediction)),
        "mae_mm": float(mean_absolute_error(y, oof_prediction)),
        "mean_only_rmse_mm": float(
            root_mean_squared_error(y, np.full_like(y, np.mean(y)))
        ),
        "max_grouped_additivity_error_mm": float(
            np.max(np.abs(oof_baseline + grouped_phi.sum(axis=1) - oof_prediction))
        ),
    }
    run_record = {"metrics": metrics, "folds": fold_records}
    return released, phi, {"display": display, **run_record}


def _load_external_explanations() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recover aligned display and SHAP matrices without refitting the model."""

    released = pd.read_csv(EXPLANATION_DATA)
    display_columns = [DISPLAY_NAMES[name] for name in RAW_CATEGORICAL + RAW_NUMERIC]
    display = pd.DataFrame(index=released.index)
    phi = pd.DataFrame(index=released.index)
    for index, name in enumerate(display_columns):
        values = released[f"feature__{name}"]
        if index < len(CATEGORY_LEVELS):
            if name == "Year":
                values = values.astype(int).astype(str)
            display[name] = pd.Categorical(
                values,
                categories=CATEGORY_LEVELS[index],
                ordered=name in {"Year", "Body-mass quartile"},
            )
        else:
            display[name] = pd.to_numeric(values, errors="raise")
        phi[name] = pd.to_numeric(released[f"shap__{name}"], errors="raise")
    if not display.index.equals(phi.index) or not display.columns.equals(phi.columns):
        raise RuntimeError("Released feature and SHAP matrices are not aligned")
    if not np.isfinite(phi.to_numpy(float)).all():
        raise RuntimeError("Released SHAP matrix contains nonfinite values")
    return display, phi


def _render(
    display: pd.DataFrame,
    phi: pd.DataFrame,
    output: Path,
    *,
    reported_output: Path | None = None,
) -> tuple[Path, Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "figure_5_penguins"
    result = plot_summary(
        shap_values=phi,
        features=display,
        category_order_mode="severity",
        severity_direction="higher",
        x_label="SHAP value (flipper length, mm)",
        max_features=7,
        max_points=520,
        random_state=RANDOM_STATE,
        category_key_placement="right_stacked",
        output=stem,
    )
    portable_summary = dict(result.summary)
    reported_stem = (reported_output or output) / "figure_5_penguins"
    portable_summary["output_files"] = [
        _portable_path(reported_stem.with_suffix(suffix))
        for suffix in (".png", ".pdf")
    ]
    summary = output / "figure_5_penguins_summary.json"
    summary.write_text(
        json.dumps(portable_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return stem.with_suffix(".png"), stem.with_suffix(".pdf"), summary


def main() -> None:
    global RAW_DATA, EXPLANATION_DATA, EXPECTED_MANIFEST
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=None,
        help="external user-acquired penguins.csv; required unless --render-only",
    )
    parser.add_argument(
        "--explanations",
        type=Path,
        required=True,
        help=(
            "external explanation CSV to read in --render-only mode or write "
            "after model fitting"
        ),
    )
    parser.add_argument(
        "--expected-manifest",
        type=Path,
        required=True,
        help="external scientific reference manifest maintained by the analyst",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="external output directory; must not be inside this repository",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help=(
            "Render the user-supplied external explanation table without "
            "refitting or importing SHAP."
        ),
    )
    parser.add_argument(
        "--strict-environment",
        action="store_true",
        help=(
            "Require exact attribution-generation software versions in addition "
            "to matching the reference scientific record."
        ),
    )
    args = parser.parse_args()
    RAW_DATA = args.raw_data.resolve() if args.raw_data is not None else Path()
    EXPLANATION_DATA = args.explanations.resolve()
    EXPECTED_MANIFEST = args.expected_manifest.resolve()
    output = args.output.resolve()
    try:
        output.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("empirical outputs must be written outside the repository")
    if not EXPECTED_MANIFEST.is_file():
        raise FileNotFoundError(EXPECTED_MANIFEST)
    if not args.render_only and not RAW_DATA.is_file():
        raise FileNotFoundError("--raw-data is required outside --render-only mode")
    if args.render_only and not EXPLANATION_DATA.is_file():
        raise FileNotFoundError(EXPLANATION_DATA)

    if args.render_only:
        display, phi = _load_external_explanations()
        png, pdf, summary = _render(display, phi, output)
        manifest = json.loads(EXPECTED_MANIFEST.read_text(encoding="utf-8"))
        manifest["artifacts"] = {
            "explanations_csv": {
                "path": _portable_path(EXPLANATION_DATA),
                "sha256": _sha256(EXPLANATION_DATA),
            },
            "figure_png": {
                "path": _portable_path(png),
                "sha256": _sha256(png),
            },
            "figure_pdf": {
                "path": _portable_path(pdf),
                "sha256": _sha256(pdf),
            },
            "rendering_summary": {
                "path": _portable_path(summary),
                "sha256": _sha256(summary),
            },
        }
        run_manifest = output / RUN_MANIFEST_NAME
        _write_json_atomic(run_manifest, manifest)
        print(
            json.dumps(
                {
                    "rows": len(display),
                    "mode": "external explanations; render only",
                },
                indent=2,
            )
        )
        return

    import shap
    import sklearn

    frame, mass_edges = _load_analysis_frame()
    released, phi, run_record = _fit_oof(frame)
    display = run_record.pop("display")
    final_output = output

    raw_rows = int(len(pd.read_csv(RAW_DATA, na_values=["NA"])))
    metrics = run_record["metrics"]
    manifest = {
        "schema_version": "1.0",
        "analysis_id": "palmer_penguins_global_categories_v1",
        "purpose": (
            "Subgroup-free visualization example for category-aware global "
            "SHAP summaries; not a causal biological analysis."
        ),
        "dataset": {
            "name": "Palmer Penguins simplified data",
            "source_url": (
                "https://github.com/allisonhorst/palmerpenguins/"
                "blob/v0.1.0/inst/extdata/penguins.csv"
            ),
            "raw_sha256": _sha256(RAW_DATA),
            "license": "CC0-1.0",
            "package_citation_doi": "10.5281/zenodo.3960218",
            "data_article_doi": "10.1371/journal.pone.0090081",
            "raw_rows": raw_rows,
            "complete_analysis_rows": int(len(frame)),
            "excluded_incomplete_rows": raw_rows - int(len(frame)),
        },
        "target": {
            "name": "flipper_length_mm",
            "scale": "random-forest prediction in millimetres",
        },
        "features": {
            "categorical": {
                "Sex": 2,
                "Species": 3,
                "Island": 3,
                "Year": 3,
                "Body-mass quartile": 4,
            },
            "numeric": ["Bill length", "Bill depth"],
            "body_mass_quartile_edges_g": mass_edges.tolist(),
            "body_mass_quartiles": (
                "Edges are defined once on the 333-row complete analysis "
                "cohort before cross-fitting. This target-free preprocessing is "
                "transductive. Body mass enters the model only through this "
                "discretized feature."
            ),
            "raw_body_mass_used_as_separate_feature": False,
        },
        "validation": {
            "scheme": "five-fold shuffled out-of-fold",
            "n_splits": N_SPLITS,
            "split_random_state": RANDOM_STATE,
            "metrics": metrics,
            "folds": run_record["folds"],
        },
        "model": {
            "preprocessing": (
                "Complete cases; explicit one-hot levels for five categorical "
                "features; two numeric features passed through unchanged."
            ),
            "estimator": "sklearn.ensemble.RandomForestRegressor",
            "parameters": {
                "n_estimators": 400,
                "min_samples_leaf": 3,
                "max_features": 0.8,
                "random_state_rule": f"{RANDOM_STATE} + fold - 1",
                "n_jobs": 1,
            },
        },
        "explanations": {
            "method": "TreeSHAP",
            "feature_perturbation": "tree_path_dependent",
            "encoded_to_source_mapping": (
                "One-hot SHAP columns are summed within each source feature. "
                "The sum preserves local additivity but is not claimed as an "
                "Owen value for the unencoded feature."
            ),
            "category_order": (
                "Within each feature, categories are ordered by their "
                "full-sample mean signed out-of-fold SHAP value; color "
                "position encodes rank, not distance."
            ),
            "severity_direction": (
                "higher: the blue-to-pink order runs from lower to higher "
                "predicted flipper length. The API name does not imply "
                "biological severity in this example."
            ),
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "shap": shap.__version__,
        },
    }
    expected = json.loads(EXPECTED_MANIFEST.read_text(encoding="utf-8"))
    if _scientific_payload(manifest) != _scientific_payload(expected):
        raise RuntimeError(
            "Palmer Penguins scientific results differ from the reference "
            "expected manifest"
        )
    environment_match = manifest["software"] == expected.get("software")
    if args.strict_environment and not environment_match:
        raise RuntimeError(
            "Palmer Penguins software versions differ from the reference environment"
        )

    with tempfile.TemporaryDirectory(
        prefix="shap-prism-penguins-"
    ) as temporary_directory:
        staging_root = Path(temporary_directory)
        staged_explanations = staging_root / EXPLANATION_DATA.name
        released.to_csv(
            staged_explanations,
            index=False,
            float_format="%.12g",
        )
        staged_output = staging_root / "rendered"
        png, pdf, summary = _render(
            display,
            phi,
            staged_output,
            reported_output=final_output,
        )
        explanation_hash = _sha256(staged_explanations)
        expected_hash = (
            expected.get("artifacts", {})
            .get("explanations_csv", {})
            .get("sha256")
        )
        if expected_hash is not None and explanation_hash != expected_hash:
            raise RuntimeError(
                "Palmer Penguins explanation table differs from the reference artifact"
            )
        final_png = final_output / png.name
        final_pdf = final_output / pdf.name
        final_summary = final_output / summary.name
        manifest["artifacts"] = {
            "explanations_csv": {
                "path": _portable_path(EXPLANATION_DATA),
                "sha256": explanation_hash,
            },
            "figure_png": {
                "path": _portable_path(final_png),
                "sha256": _sha256(png),
            },
            "figure_pdf": {
                "path": _portable_path(final_pdf),
                "sha256": _sha256(pdf),
            },
            "rendering_summary": {
                "path": _portable_path(final_summary),
                "sha256": _sha256(summary),
            },
        }
        manifest["verification"] = {
            "scientific_record_match": True,
            "explanation_artifact_match": True,
            "environment_match": environment_match,
            "strict_environment": bool(args.strict_environment),
        }

        _atomic_publish(staged_explanations, EXPLANATION_DATA)
        _atomic_publish(png, final_png)
        _atomic_publish(pdf, final_pdf)
        _atomic_publish(summary, final_summary)

    run_manifest = final_output / RUN_MANIFEST_NAME
    _write_json_atomic(run_manifest, manifest)
    print(json.dumps({"rows": len(frame), **metrics}, indent=2))


if __name__ == "__main__":
    main()
