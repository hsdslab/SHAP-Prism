# Verified compatibility

SHAP Prism 0.4.4 requires Python 3.12 or later. The full public-source test
suite was run on Python 3.12.13 with both of the following environments:

| Component | Pinned publication stack | Latest permitted stack tested on 2026-08-19 |
|---|---:|---:|
| NumPy | 2.3.5 | 2.5.2 |
| pandas | 2.2.3 | 3.0.5 |
| Matplotlib | 3.10.8 | 3.11.1 |
| Pillow | 12.2.0 | 12.3.0 |
| pypdf | 6.10.0 | 6.16.1 |

The pinned versions remain in `requirements-verified.txt`. Continuous
integration also installs unpinned dependencies within `pyproject.toml` so
that newly released dependency versions are tested promptly.

The source test suite contains 86 tests. All 86 passed locally in both Python
environments above.

SHAP and scikit-learn are optional upstream-builder dependencies, not runtime
plotting dependencies. No empirical input is bundled with that builder.
