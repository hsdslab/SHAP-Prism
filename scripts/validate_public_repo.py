#!/usr/bin/env python3
"""Validate the public SHAP Prism source repository."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "PUBLIC_SOURCE_SHA256SUMS"

REQUIRED = {
    ".github/workflows/python-ci.yml",
    ".github/dependabot.yml",
    ".gitattributes",
    ".gitignore",
    "CHANGELOG.md",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "DATA_LICENSES.md",
    "DEVELOPMENT.md",
    "LICENSE",
    "LICENSE_SCOPE.md",
    "MANIFEST.in",
    "notebooks/build_shap_prism_quickstart.py",
    "notebooks/shap_prism_quickstart.ipynb",
    "PUBLIC_SOURCE_SHA256SUMS",
    "README.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "ARTICLE_DATA_ACQUISITION.md",
    "pyproject.toml",
    "src/shap_prism/_version.py",
    "src/shap_prism/plotting.py",
}

FORBIDDEN_TOP_LEVEL = {
    "article",
    "dist",
    "final_validation",
    "r",
    "submission",
    "visual_checks",
}

FORBIDDEN_EXACT = {
    "PRE_PUBLICATION_CHECKLIST.md",
    "PUBLIC_RELEASE_CHANGES.md",
    "DELIVERY_NOTE.md",
    "RELEASE_SCOPE.md",
    "REPRODUCIBILITY.md",
    "SUPPLEMENTARY_README.md",
    "SHA256SUMS",
}

FORBIDDEN_DATA = {
    "reproduction/data/airfoil_oof_frequency_explanations.csv",
    "reproduction/data/airfoil_seed42.npz",
    "reproduction/data/barley_oof_explanations.csv",
    "reproduction/data/solder_primary_explanations.csv",
    "reproduction/data/palmer_penguins/penguins.csv",
    "reproduction/data/palmer_penguins_oof_explanations.csv",
    "reproduction/expected/palmer_penguins_run_manifest.json",
}

FORBIDDEN_DATA_SUFFIXES = {
    ".arrow",
    ".csv",
    ".feather",
    ".h5",
    ".hdf5",
    ".json",
    ".npy",
    ".npz",
    ".parquet",
    ".rdata",
    ".rds",
    ".sav",
    ".sas7bdat",
    ".tsv",
    ".xls",
    ".xlsx",
}
FORBIDDEN_EMPIRICAL_DIRS = {
    ("reproduction", "data"),
    ("reproduction", "expected"),
    ("reproduction", "generated"),
}
ALLOWED_NOTEBOOK_FILES = {
    "notebooks/build_shap_prism_quickstart.py",
    "notebooks/shap_prism_quickstart.ipynb",
}

CACHE_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
}

ARCHIVE_SUFFIXES = (".zip", ".whl", ".tar.gz", ".sha256")
UNSUPPORTED_LANGUAGE_SUFFIXES = {".r", ".rd", ".rmd"}
HEX64_PATTERN = re.compile(r"[0-9a-f]{64}")
PHONE_PATTERN = re.compile(r"\+36[\s-]?(?:\d[\s-]?){8,9}")
PLACEHOLDER_PATTERN = re.compile(
    r"<\s*(?:OWNER|REPO|USERNAME|EMAIL|DOI)\s*>"
    r"|github\.com/(?:OWNER|USERNAME)(?:/|$)"
    r"|(?:YOUR|INSERT)[ _-](?:NAME|EMAIL|USERNAME|URL|DOI)",
    re.I,
)
UNFINISHED_MARKER_PATTERN = re.compile(
    r"\b(?:" + "|".join(("TO" + "DO", "FIX" + "ME", "T" + "BD")) + r")\b",
    re.I,
)
INTERNAL_FILENAME_TERMS = (
    "audit",
    "checklist",
    "construction",
    "internal",
    "private",
    "sealed",
    "submission",
    "to" + "do",
    "fix" + "me",
)
INTERNAL_FILENAME_PATTERN = re.compile(
    r"(?:^|[-_. ])(?:"
    + "|".join(re.escape(term) for term in INTERNAL_FILENAME_TERMS)
    + r")(?:$|[-_. ])",
    re.I,
)
INTERNAL_PROVENANCE_PATTERN = re.compile(
    r"\bprivate\s+(?:MethodsX|release|delivery|submission)\b"
    r"|\bsealed\s+(?:delivery|scientific|environment|artifact|record)\b"
    r"|\binternal\s+(?:note|release|validation|document|checklist|audit)\b"
    r"|\bpre[- ]publication\s+checklist\b"
    r"|\brelease\s+construction\b",
    re.I,
)
UNSUPPORTED_LANGUAGE_CLAIMS = (
    "native " + "R",
    "R " + "implementation",
    "r/" + "shapprism",
    "base-" + "R",
    "R " + "CMD",
    "R" + "script",
    "Python and " + "R",
    "Python/" + "R",
    "R " + "workflow",
    "R-" + "workflow",
    "R " + "0.1.1",
    "R version " + "0.1.1",
    "both " + "languages",
    "R-" + "core",
    "Web" + "R",
    "C" + "RAN",
    ".R" + "buildignore",
    "r-" + "cmd-check",
)
UNSUPPORTED_LANGUAGE_CLAIM_PATTERN = re.compile(
    "|".join(re.escape(value) for value in UNSUPPORTED_LANGUAGE_CLAIMS), re.I
)
TEXT_SCAN_EXACT_EXCLUSIONS = {MANIFEST, "LICENSES/MATPLOTLIB.txt"}
TEXT_SCAN_SUFFIX_EXCLUSIONS = {".csv", ".ipynb", ".png", ".svg"}


def _files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(ROOT).parts
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def _validate_synthetic_notebook(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return [f"notebook JSON cannot be parsed: {path.relative_to(ROOT)}"]
    metadata = notebook.get("metadata", {}).get("shap_prism", {})
    if metadata.get("data_scope") != "fully synthetic; generated in notebook":
        errors.append("public notebook does not declare the required synthetic scope")
    cells = notebook.get("cells", [])
    code = "\n".join(
        "".join(cell.get("source", []))
        if isinstance(cell.get("source"), list)
        else str(cell.get("source", ""))
        for cell in cells
        if cell.get("cell_type") == "code"
    )
    forbidden_loaders = (
        "read_csv(",
        "read_excel(",
        "read_parquet(",
        "urlopen(",
        "requests.get(",
    )
    if any(token in code for token in forbidden_loaders):
        errors.append("public notebook contains an external data-loading call")
    source_text = "\n".join(
        "".join(cell.get("source", []))
        if isinstance(cell.get("source"), list)
        else str(cell.get("source", ""))
        for cell in cells
    )
    output_text: list[str] = []
    for cell in cells:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                value = output.get("text", "")
                output_text.append("".join(value) if isinstance(value, list) else str(value))
            plain = output.get("data", {}).get("text/plain")
            if plain is not None:
                output_text.append("".join(plain) if isinstance(plain, list) else str(plain))
    decoded_text = source_text + "\n" + "\n".join(output_text)
    if PLACEHOLDER_PATTERN.search(decoded_text):
        errors.append("public notebook contains an unresolved identity or URL field")
    if UNFINISHED_MARKER_PATTERN.search(decoded_text):
        errors.append("public notebook contains an unfinished-work marker")
    if INTERNAL_PROVENANCE_PATTERN.search(decoded_text):
        errors.append("public notebook contains internal provenance wording")
    if UNSUPPORTED_LANGUAGE_CLAIM_PATTERN.search(decoded_text):
        errors.append("public notebook contains an unsupported language claim")
    if not any(
        cell.get("execution_count") is not None
        for cell in cells
        if cell.get("cell_type") == "code"
    ):
        errors.append("public notebook is not executed")
    return errors


def validate() -> dict[str, object]:
    errors: list[str] = []
    files = _files()
    relative = [path.relative_to(ROOT).as_posix() for path in files]
    relative_set = set(relative)

    missing = sorted(REQUIRED - relative_set)
    errors.extend(f"missing required file: {item}" for item in missing)

    for name in sorted(FORBIDDEN_TOP_LEVEL):
        if (ROOT / name).exists():
            errors.append(f"forbidden top-level path: {name}")
    errors.extend(
        f"forbidden repository file: {item}"
        for item in sorted((FORBIDDEN_EXACT | FORBIDDEN_DATA) & relative_set)
    )

    for item, path in zip(relative, files, strict=True):
        parts = path.relative_to(ROOT).parts
        if any(INTERNAL_FILENAME_PATTERN.search(part) for part in parts):
            errors.append(f"non-public path name present: {item}")
        if any(part in CACHE_PARTS for part in parts):
            errors.append(f"cache path present: {item}")
        if path.suffix.casefold() in UNSUPPORTED_LANGUAGE_SUFFIXES:
            errors.append(f"unsupported language source present: {item}")
        if parts and parts[0] == "notebooks" and item not in ALLOWED_NOTEBOOK_FILES:
            errors.append(f"undeclared notebook artifact present: {item}")
        if path.suffix.casefold() in FORBIDDEN_DATA_SUFFIXES:
            errors.append(f"dataset-like file present in data-free repository: {item}")
        if any(
            tuple(parts[index : index + len(forbidden)]) == forbidden
            for forbidden in FORBIDDEN_EMPIRICAL_DIRS
            for index in range(len(parts) - len(forbidden) + 1)
        ):
            errors.append(f"empirical artifact directory present: {item}")
        if item != MANIFEST and item.lower().endswith(ARCHIVE_SUFFIXES):
            errors.append(f"archive/release artifact present: {item}")
        text = _text(path)
        if text is None:
            continue
        if PHONE_PATTERN.search(text):
            errors.append(f"Hungarian mobile number present: {item}")
        if (
            item not in TEXT_SCAN_EXACT_EXCLUSIONS
            and path.suffix.lower() not in TEXT_SCAN_SUFFIX_EXCLUSIONS
        ):
            if PLACEHOLDER_PATTERN.search(text):
                errors.append(f"unresolved identity or URL field present: {item}")
            if UNFINISHED_MARKER_PATTERN.search(text):
                errors.append(f"unfinished-work marker present: {item}")
            if INTERNAL_PROVENANCE_PATTERN.search(text):
                errors.append(f"internal provenance wording present: {item}")
            if UNSUPPORTED_LANGUAGE_CLAIM_PATTERN.search(text):
                errors.append(f"unsupported language claim present: {item}")
        if item == "notebooks/shap_prism_quickstart.ipynb":
            errors.extend(_validate_synthetic_notebook(path))

    with (ROOT / "pyproject.toml").open("rb") as stream:
        python_version = tomllib.load(stream)["project"]["version"]
    runtime = (ROOT / "src/shap_prism/_version.py").read_text(encoding="utf-8")
    if python_version != "0.4.4" or '__version__ = "0.4.4"' not in runtime:
        errors.append("Python version is not consistently 0.4.4")

    manifest_path = ROOT / MANIFEST
    if manifest_path.is_file():
        declared: dict[str, str] = {}
        manifest_lines = manifest_path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(manifest_lines, 1):
            parts = line.split("  ", 1)
            if len(parts) != 2 or not HEX64_PATTERN.fullmatch(parts[0]):
                errors.append(f"malformed public checksum line: {line_number}")
                continue
            digest, item = parts
            if item in declared:
                errors.append(f"duplicate public checksum target: {item}")
            declared[item] = digest
        expected_files = set(relative) - {MANIFEST}
        if set(declared) != expected_files:
            errors.append("public checksum manifest file list is stale")
        else:
            for item, digest in declared.items():
                if _sha256(ROOT / item) != digest:
                    errors.append(f"public checksum mismatch: {item}")

    report = {
        "status": "passed" if not errors else "failed",
        "files": len(files),
        "python_version": python_version,
        "errors": errors,
    }
    if errors:
        raise SystemExit(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(validate(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
