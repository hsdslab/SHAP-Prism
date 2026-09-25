# Verified compatibility

## 0.4.5

SHAP Prism 0.4.5 requires Python 3.12 or later. All 89 source tests passed on
Windows with Python 3.14.2 on 2026-09-25, using NumPy 2.4.1, pandas 3.0.2,
Matplotlib 3.11.0, Pillow 12.2.0, and pypdf 6.6.2. This includes three new
checks for excluding local-only material from the public repository.

The plotting implementation and runtime dependency requirements are unchanged
from 0.4.4. CI continues to test the pinned publication stack, latest-permitted
dependencies, Python 3.12 through 3.14, and Linux, Windows, and macOS.

## 0.4.4 publication environment

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

The 0.4.4 source test suite contains 86 tests. All 86 passed locally in both Python
environments above.

SHAP and scikit-learn are optional upstream-builder dependencies, not runtime
plotting dependencies. No empirical input is bundled with that builder.
