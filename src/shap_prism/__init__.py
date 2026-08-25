"""Public interface for SHAP Prism."""

from ._version import __version__
from .plotting import (
    PrismResult,
    plot_prism,
    plot_summary,
)

__all__ = [
    "PrismResult",
    "__version__",
    "plot_prism",
    "plot_summary",
]
