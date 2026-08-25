"""Publication plots for categorical-aware global and subgroup SHAP summaries.

Both public functions consume aligned, precomputed SHAP values. ``plot_summary``
draws a categorical-aware global beeswarm; ``plot_prism`` adds a shared-scale
subgroup view for one focal feature. Model fitting and the definition of
scientifically meaningful groups remain with the analyst.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from numbers import Integral, Real
from pathlib import Path
import textwrap
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import FuncFormatter, MaxNLocator
from numpy.typing import ArrayLike

from ._version import __version__
from ._artifacts import atomic_save_figure


TEXT = "#202631"
MUTED = "#667085"
GRID = "#D8DEE7"
ZERO = "#657181"
MEAN = "#111827"
BLUE = "#1689D8"
PURPLE = "#7655C5"
PINK = "#EF4D8B"
MISSING = "#A7B0BC"

SHAP_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "prism_blue_pink", [BLUE, PURPLE, PINK]
)
CATEGORY_PALETTES: dict[int, tuple[str, ...]] = {
    1: (BLUE,),
    2: (BLUE, PINK),
    3: (BLUE, "#9054BB", PINK),
    4: (BLUE, "#6E5FC8", "#B453AB", PINK),
    5: (BLUE, "#626ACC", "#9054BB", "#C452A3", PINK),
    6: (BLUE, "#5971CF", PURPLE, "#A753B2", "#CD519E", PINK),
    7: (BLUE, "#5375D0", "#6E5FC8", "#9054BB", "#B453AB", "#D3509B", PINK),
    8: (BLUE, "#4E78D1", "#6765CA", "#7E55C2", "#A154B5", "#BD52A7", "#D75099", PINK),
}
# Public eight-level form of the default categorical palette.
CATEGORY_PALETTE = CATEGORY_PALETTES[8]
OKABE_ITO_CATEGORY_PALETTE = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#D55E00",
    "#56B4E9",
    "#F0E442",
    "#000000",
)
OKABE_ITO_CATEGORY_PALETTES: dict[int, tuple[str, ...]] = {
    count: OKABE_ITO_CATEGORY_PALETTE[:count] for count in range(1, 9)
}
OKABE_ITO_CMAP = mpl.colormaps["cividis"]
_COLOR_SCHEMES: dict[
    str,
    tuple[Mapping[int, tuple[str, ...]], mpl.colors.Colormap],
] = {
    "prism": (CATEGORY_PALETTES, SHAP_CMAP),
    "okabe_ito": (OKABE_ITO_CATEGORY_PALETTES, OKABE_ITO_CMAP),
}
_MARKERS = ("o", "s", "D", "^", "P", "v", ">", "*")
_CATEGORY_ORDER_MODES = {"data", "alphabetical", "severity"}
_CATEGORY_KEY_PLACEMENTS = {"row_aligned", "bottom", "right_stacked"}
_MAX_DISPLAYED_CATEGORY_LEVELS = 8
_RETAINED_HIGH_CARDINALITY_LEVELS = 7
_OTHERS_LABEL = "Others"
_MISSING_LABEL = "Missing"
_DISAMBIGUATED_MISSING_LABEL = "Missing (NA)"
_HIGH_CARDINALITY_RULE = (
    "seven raw levels retained by decreasing frequency, with a literal raw "
    "Others level reserved for the pooled entry; frequency ties follow the "
    "resolved category order; all remaining levels are pooled as Others"
)
_CategoryOrderSource = Literal[
    "explicit",
    "pandas_ordered",
    "first_appearance",
    "alphabetical",
    "mean_signed_shap",
]


@dataclass(frozen=True)
class _CategoryStyle:
    label: str
    color: str
    marker: str


@dataclass(frozen=True)
class _OtherCategories:
    """Internal identity for raw levels pooled into the displayed Others level."""

    levels: tuple[Hashable, ...]

    def __str__(self) -> str:
        return _OTHERS_LABEL


@dataclass(frozen=True)
class _MissingCategory:
    """Internal identity for true missing cells, not a raw text category."""


_MISSING_CATEGORY = _MissingCategory()


@dataclass(frozen=True)
class _FeatureKey:
    feature: str
    display_feature: str
    levels: tuple[Hashable, ...]
    styles: tuple[_CategoryStyle, ...]
    signed_means: tuple[float, ...]
    order_source: _CategoryOrderSource
    missing: bool
    missing_row_count: int
    original_level_count: int
    collapsed_levels: tuple[Hashable, ...]
    displayed_level_row_counts: tuple[int, ...]


@dataclass(frozen=True)
class _GlobalRender:
    order: tuple[int, ...]
    rendered_rows: int
    xlim: tuple[float, float]
    displayed_features: tuple[str, ...]
    keys: tuple[_FeatureKey, ...]
    collapsed_categorical_features: tuple[str, ...]


@dataclass(frozen=True)
class _DockBlockLayout:
    key_index: int
    feature_x: float
    category_tabs: tuple[float, ...]
    right_edge: float


@dataclass(frozen=True)
class _DockLayout:
    font_size: float
    marker_size: float
    marker_text_gap: float
    rows: tuple[tuple[_DockBlockLayout, ...], ...]
    right_edge: float


@dataclass(frozen=True)
class PrismResult:
    """Figure, named axes, and a JSON-serializable rendering summary."""

    figure: plt.Figure
    axes: Mapping[str, object]
    summary: dict[str, object]


def _nice_limit(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        raise ValueError("focal SHAP values contain no finite values")
    peak = float(np.max(np.abs(finite)))
    if peak == 0:
        return 1.0
    raw = max(peak * 1.035, np.finfo(float).eps)
    exponent = np.floor(np.log10(raw))
    unit = 10.0**exponent
    scaled = raw / unit
    for candidate in (1.0, 1.2, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if scaled <= candidate:
            return float(candidate * unit)
    return float(10.0 * unit)


def _compact_scientific(value: float, decimals: int, *, signed: bool = False) -> str:
    if value == 0:
        return "0"
    text = (
        f"{value:+.{decimals}e}"
        if signed
        else f"{value:.{decimals}e}"
    )
    mantissa, exponent = text.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent):+d}"


def _format_tick(value: float, limit: float) -> str:
    if abs(value) <= limit * 1e-12:
        return "0"
    if limit < 0.01 or limit >= 10_000:
        return _compact_scientific(value, 1)
    if limit >= 10:
        return f"{value:.0f}"
    if limit >= 0.1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _format_value(value: float) -> str:
    magnitude = abs(value)
    if value == 0:
        return "0"
    if magnitude < 0.001:
        return _compact_scientific(value, 2)
    if magnitude >= 10_000:
        return f"{value / 1_000:.0f}k"
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    if magnitude >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _format_mean(value: float, limit: float) -> str:
    if limit < 0.01 or limit >= 10_000:
        text = _compact_scientific(value, 2, signed=True)
    elif limit >= 100:
        text = f"{value:+.0f}"
    elif limit >= 10:
        text = f"{value:+.1f}"
    elif limit >= 1:
        text = f"{value:+.2f}"
    else:
        text = f"{value:+.3f}"
    try:
        if float(text) == 0:
            return "0"
    except ValueError:
        pass
    return text


def _resolve_color_scheme(
    value: str,
) -> tuple[str, Mapping[int, tuple[str, ...]], mpl.colors.Colormap]:
    if not isinstance(value, str):
        raise TypeError("color_scheme must be a string")
    key = value.casefold()
    try:
        palettes, cmap = _COLOR_SCHEMES[key]
    except KeyError as error:
        raise ValueError(
            "color_scheme must be 'prism' or 'okabe_ito'"
        ) from error
    return key, palettes, cmap


def _resolve_cmap(
    value: str | mpl.colors.Colormap | Sequence[str] | None,
    *,
    default: mpl.colors.Colormap,
) -> mpl.colors.Colormap:
    if value is None:
        return default
    if isinstance(value, mpl.colors.Colormap):
        return _require_opaque_cmap(value)
    if isinstance(value, str):
        try:
            return _require_opaque_cmap(mpl.colormaps[value])
        except KeyError as error:
            raise ValueError(f"unknown continuous_cmap: {value!r}") from error
    try:
        colors = list(value)
    except TypeError as error:
        raise TypeError(
            "continuous_cmap must be a Matplotlib colormap, colormap name, "
            "or a sequence of at least two colors"
        ) from error
    if len(colors) < 2:
        raise ValueError("continuous_cmap color sequence must contain at least two colors")
    invalid = [color for color in colors if not mpl.colors.is_color_like(color)]
    if invalid:
        raise ValueError(f"invalid continuous color: {invalid[0]!r}")
    transparent = [
        color for color in colors if mpl.colors.to_rgba(color)[3] < 1.0
    ]
    if transparent:
        raise ValueError("custom continuous colors must be opaque")
    return _require_opaque_cmap(
        mpl.colors.LinearSegmentedColormap.from_list(
            "shap_prism_custom", colors
        )
    )


def _require_opaque_cmap(
    cmap: mpl.colors.Colormap,
) -> mpl.colors.Colormap:
    listed = getattr(cmap, "colors", None)
    if listed is not None:
        rgba = mpl.colors.to_rgba_array(listed)
    else:
        sample_count = max(256, min(int(getattr(cmap, "N", 256)), 4096))
        rgba = np.asarray(cmap(np.linspace(0.0, 1.0, sample_count)))
    if np.any(np.asarray(rgba)[:, 3] < 1.0):
        raise ValueError("continuous_cmap must be opaque")
    return cmap


def _resolve_shared_category_palette(
    value: Sequence[str] | None,
    *,
    name: str,
) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or isinstance(value, Mapping):
        raise TypeError(f"{name} must be a sequence of colors")
    colors = tuple(value)
    if not colors:
        raise ValueError(f"{name} must contain at least one color")
    if len(colors) > 8:
        raise ValueError(f"{name} can contain at most eight colors")
    invalid = [color for color in colors if not mpl.colors.is_color_like(color)]
    if invalid:
        raise ValueError(f"invalid category color: {invalid[0]!r}")
    transparent = [
        color for color in colors if mpl.colors.to_rgba(color)[3] < 1.0
    ]
    if transparent:
        raise ValueError("custom category colors must be opaque")
    return colors


def _wrapped_label(value: object, *, width: int, lines: int = 2) -> str:
    """Wrap display text without changing the value stored in the summary."""

    collapsed = " ".join(str(value).split())
    return textwrap.fill(
        collapsed,
        width=width,
        max_lines=lines,
        placeholder="…",
    )


def _sample(indices: np.ndarray, limit: int, seed: int) -> np.ndarray:
    """Draw a deterministic target-blind display sample."""

    indices = np.asarray(indices, dtype=int)
    if len(indices) <= limit:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=limit, replace=False))


def _derived_seed(seed: int, offset: int) -> int:
    """Return a valid NumPy seed in a stable, isolated display namespace."""

    return (seed + offset) % (2**32)


def _require_integer(
    name: str,
    value: object,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    resolved = int(value)
    if resolved < minimum:
        if maximum is None:
            raise ValueError(f"{name} must be at least {minimum}")
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    if maximum is not None and resolved > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return resolved


def _swarm_offsets(values: np.ndarray, edges: np.ndarray, width: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    bins = np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)
    counts = np.bincount(bins, minlength=len(edges) - 1)
    reference = max(1, int(np.max(counts)))
    offsets = np.zeros(len(values), dtype=float)
    for bin_id in np.flatnonzero(counts):
        positions = np.flatnonzero(bins == bin_id)
        positions = positions[np.argsort(values[positions], kind="stable")]
        if len(positions) < 2:
            continue
        local_width = width * np.sqrt(len(positions) / reference)
        offsets[positions] = np.linspace(-local_width, local_width, len(positions))
    return offsets


def _native_label_key(value: Hashable) -> tuple[type[object], object]:
    """Return a type-sensitive key for a pandas axis label."""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, tuple):
        return type(value), tuple(_native_label_key(part) for part in value)
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return type(value), ("<missing-label>",)
    return type(value), value


def _native_labels_equal(
    left: Sequence[Hashable],
    right: Sequence[Hashable],
) -> bool:
    return len(left) == len(right) and all(
        _native_label_key(a) == _native_label_key(b)
        for a, b in zip(left, right, strict=True)
    )


def _resolve_inputs(
    shap_values: ArrayLike,
    features: ArrayLike | pd.DataFrame,
    feature_names: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str], dict[str, str]]:
    shap_frame = shap_values if isinstance(shap_values, pd.DataFrame) else None
    if shap_frame is not None and not shap_frame.index.is_unique:
        raise ValueError("shap_values DataFrame index must be unique")
    phi = np.asarray(shap_values, dtype=float)
    if phi.ndim != 2:
        raise ValueError("shap_values must be a two-dimensional array")
    if not np.all(np.isfinite(phi)):
        raise ValueError("shap_values must contain only finite numbers")
    if isinstance(features, pd.DataFrame):
        frame = features.copy()
        if not frame.index.is_unique:
            raise ValueError("features DataFrame index must be unique")
        native_columns = list(frame.columns)
        inferred = [str(column) for column in native_columns]
        if feature_names is not None and not _native_labels_equal(
            list(feature_names), native_columns
        ):
            raise ValueError("feature_names do not match the DataFrame columns")
        names = inferred
        if shap_frame is not None:
            if not shap_frame.index.equals(frame.index):
                raise ValueError(
                    "shap_values and features DataFrame indices must match exactly"
                )
            if not _native_labels_equal(
                list(shap_frame.columns), native_columns
            ):
                raise ValueError(
                    "shap_values and features DataFrame columns must match exactly"
                )
    else:
        array = np.asarray(features)
        if array.ndim != 2:
            raise ValueError("features must be a two-dimensional array or DataFrame")
        if feature_names is None:
            raise ValueError("feature_names are required when features is an array")
        if array.dtype.kind == "O":
            column_kinds: set[str] = set()
            for column in array.T:
                observed = [
                    value
                    for value in column
                    if not _category_value_is_missing(value)
                ]
                numeric = bool(observed) and all(
                    isinstance(value, Real)
                    and not isinstance(value, (bool, np.bool_))
                    for value in observed
                )
                column_kinds.add("numeric" if numeric else "categorical")
            if column_kinds == {"numeric", "categorical"}:
                raise ValueError(
                    "mixed numeric/categorical NumPy object arrays are ambiguous; "
                    "use a pandas DataFrame with explicit per-column dtypes"
                )
            if column_kinds == {"numeric"}:
                array = np.asarray(array, dtype=float)
        names = [str(name) for name in feature_names]
        frame = pd.DataFrame(array, columns=names)
    if frame.shape != phi.shape:
        raise ValueError("shap_values and features must have identical shapes")
    if len(set(names)) != len(names):
        raise ValueError("feature names must be unique")
    if phi.shape[0] < 2 or phi.shape[1] < 1:
        raise ValueError("at least two rows and one feature are required")
    alignment = {
        "shap_to_features": (
            "index and column checked"
            if shap_frame is not None and isinstance(features, pd.DataFrame)
            else "positional"
        )
    }
    return phi, frame.to_numpy(), frame, names, alignment


def _resolve_focal(focal_feature: str | int, names: Sequence[str]) -> tuple[int, str]:
    if isinstance(focal_feature, (bool, np.bool_)):
        raise TypeError("focal_feature must be a feature name or integer index")
    if isinstance(focal_feature, (int, np.integer)):
        index = int(focal_feature)
        if not 0 <= index < len(names):
            raise ValueError(f"focal_feature index is out of range: {index}")
        return index, names[index]
    name = str(focal_feature)
    if name not in names:
        raise ValueError(f"unknown focal_feature: {name}")
    return names.index(name), name


def _normalise_group_value(value: Hashable) -> Hashable:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _group_key(value: Hashable) -> tuple[type[object], object]:
    value = _normalise_group_value(value)
    if isinstance(value, tuple):
        return type(value), tuple(_group_key(part) for part in value)
    if not isinstance(value, Hashable):
        raise ValueError("group labels must be hashable")
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        raise ValueError("groups cannot contain missing values")
    return type(value), value


def _group_mask(labels: np.ndarray, group: Hashable) -> np.ndarray:
    target = _group_key(group)
    return np.fromiter(
        (_group_key(label) == target for label in labels),
        dtype=bool,
        count=len(labels),
    )


def _resolve_groups(
    groups: Sequence[Hashable], group_order: Sequence[Hashable] | None, n: int
) -> tuple[np.ndarray, list[Hashable]]:
    raw_labels = list(groups)
    if len(raw_labels) != n:
        raise ValueError("groups must be one-dimensional and match the number of rows")
    labels = np.empty(n, dtype=object)
    observed: list[Hashable] = []
    observed_keys: set[tuple[type[object], object]] = set()
    for index, raw_label in enumerate(raw_labels):
        label = _normalise_group_value(raw_label)
        key = _group_key(label)
        labels[index] = label
        if key not in observed_keys:
            observed.append(label)
            observed_keys.add(key)
    order = [
        _normalise_group_value(item)
        for item in (list(group_order) if group_order is not None else observed)
    ]
    order_keys = [_group_key(item) for item in order]
    if len(set(order_keys)) != len(order_keys):
        raise ValueError("group_order contains duplicate groups")
    if any(key not in observed_keys for key in order_keys):
        raise ValueError("group_order contains an unknown group")
    if len(order) != len(observed) or set(order_keys) != observed_keys:
        raise ValueError("group_order must contain every observed group exactly once")
    if not 2 <= len(order) <= 6:
        raise ValueError("SHAP Prism supports two through six groups")
    return labels, order


def _resolve_display_names(
    display_names: Mapping[str, str] | None,
    names: Sequence[str],
) -> dict[str, str]:
    shown = dict(display_names or {})
    unknown = [key for key in shown if key not in names]
    if unknown:
        raise ValueError(f"display_names contains unknown features: {unknown!r}")
    invalid = [
        key for key, value in shown.items()
        if not isinstance(value, str) or not value.strip()
    ]
    if invalid:
        raise ValueError("display_names values must be non-empty strings")
    resolved = [shown.get(name, name) for name in names]
    if len(set(resolved)) != len(resolved):
        raise ValueError("display_names must give every feature a unique label")
    return shown


def _resolve_group_labels(
    group_labels: Mapping[Hashable, str] | None,
    order: Sequence[Hashable],
) -> tuple[dict[tuple[type[object], object], str], list[str]]:
    order_keys = [_group_key(group) for group in order]
    known = set(order_keys)
    shown: dict[tuple[type[object], object], str] = {}
    unknown: list[Hashable] = []
    for group, label in dict(group_labels or {}).items():
        key = _group_key(group)
        if key not in known:
            unknown.append(group)
        else:
            if not isinstance(label, str) or not label.strip():
                raise ValueError("group_labels values must be non-empty strings")
            shown[key] = label.strip()
    if unknown:
        raise ValueError(f"group_labels contains unknown groups: {unknown!r}")
    displayed = [shown.get(key, str(group)) for key, group in zip(order_keys, order, strict=True)]
    if len(set(displayed)) != len(displayed):
        raise ValueError("group_labels must give every group a unique display label")
    return shown, displayed


def _is_categorical(series: pd.Series) -> bool:
    dtype = series.dtype
    return bool(
        isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
    )


def _category_value_is_missing(value: object) -> bool:
    """Return whether one category cell is missing without expanding tuples."""

    if isinstance(value, (tuple, _OtherCategories)):
        return False
    missing = pd.isna(value)
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _category_key(value: Hashable) -> tuple[type[object], object]:
    """Return a type-sensitive, hashable identity for a category value."""

    value = _normalise_group_value(value)
    if isinstance(value, tuple):
        return type(value), tuple(_category_key(part) for part in value)
    if not isinstance(value, Hashable):
        raise ValueError("categorical values must be hashable")
    if _category_value_is_missing(value):
        raise ValueError("missing values are not category levels")
    return type(value), value


def _category_display_label(
    value: Hashable | _MissingCategory,
    sibling_levels: Sequence[Hashable] = (),
) -> str:
    """Return a public label without conflating NA with a raw category."""

    if not isinstance(value, _MissingCategory):
        return str(value)
    occupied: set[str] = set()
    for level in sibling_levels:
        if isinstance(level, _OtherCategories):
            occupied.update(str(raw_level) for raw_level in level.levels)
        else:
            occupied.add(str(level))
    if _MISSING_LABEL not in occupied:
        return _MISSING_LABEL
    candidate = _DISAMBIGUATED_MISSING_LABEL
    suffix = 2
    while candidate in occupied:
        candidate = f"{_DISAMBIGUATED_MISSING_LABEL} #{suffix}"
        suffix += 1
    return candidate


def _missing_category_label(levels: Sequence[Hashable]) -> str:
    """Label actual missing cells uniquely among the displayed raw levels."""

    return _category_display_label(_MISSING_CATEGORY, levels)


def _category_mask(
    values: Sequence[object],
    level: Hashable,
) -> np.ndarray:
    """Match a category level without pandas' tuple broadcasting semantics."""

    if isinstance(level, _OtherCategories):
        targets = {_category_key(value) for value in level.levels}
        return np.fromiter(
            (
                False
                if _category_value_is_missing(value)
                else _category_key(_normalise_group_value(value)) in targets
                for value in values
            ),
            dtype=bool,
            count=len(values),
        )
    target = _category_key(level)
    return np.fromiter(
        (
            False
            if _category_value_is_missing(value)
            else _category_key(_normalise_group_value(value)) == target
            for value in values
        ),
        dtype=bool,
        count=len(values),
    )


def _observed_categories(values: Sequence[object]) -> list[Hashable]:
    """Return nonmissing levels in declared or stable first-observation order."""

    raw = np.asarray(values, dtype=object)
    nonmissing = [
        _normalise_group_value(value)
        for value in raw
        if not _category_value_is_missing(value)
    ]
    observed_types = {type(value) for value in nonmissing}
    if len(observed_types) > 1:
        readable = ", ".join(sorted(value.__name__ for value in observed_types))
        raise ValueError(
            "categorical values must not mix Python types; "
            f"observed: {readable}"
        )
    observed_keys = {_category_key(value) for value in nonmissing}
    dtype = getattr(values, "dtype", None)
    if isinstance(dtype, pd.CategoricalDtype) and dtype.ordered:
        return [
            _normalise_group_value(value)
            for value in dtype.categories
            if _category_key(value) in observed_keys
        ]
    levels: list[Hashable] = []
    seen: set[tuple[type[object], object]] = set()
    for value in nonmissing:
        key = _category_key(value)
        if key not in seen:
            levels.append(value)
            seen.add(key)
    return levels


def _resolve_category_levels(
    values: Sequence[object],
    shap_values: Sequence[float],
    mode: Literal["data", "alphabetical", "severity"],
    *,
    explicit: Sequence[Hashable] | None = None,
    severity_direction: Literal["higher", "lower"] = "higher",
) -> tuple[
    list[Hashable],
    dict[Hashable, float],
    _CategoryOrderSource,
]:
    """Resolve display order, pooling high-cardinality tails as ``Others``."""

    if mode not in _CATEGORY_ORDER_MODES:
        raise ValueError(
            "category_order_mode must be 'data', 'alphabetical', or 'severity'"
        )
    if severity_direction not in {"higher", "lower"}:
        raise ValueError("severity_direction must be 'higher' or 'lower'")
    levels = _observed_categories(values)
    if not levels:
        raise ValueError("categorical values contain only missing values")
    phi = np.asarray(shap_values, dtype=float)
    means = {
        level: float(np.mean(phi[_category_mask(values, level)]))
        for level in levels
    }
    if explicit is not None:
        ordered = [_normalise_group_value(value) for value in explicit]
        ordered_keys = [_category_key(value) for value in ordered]
        level_keys = {_category_key(value) for value in levels}
        if (
            len(ordered) != len(levels)
            or len(set(ordered_keys)) != len(ordered_keys)
            or set(ordered_keys) != level_keys
        ):
            raise ValueError(
                "category_order must contain every observed level exactly once"
            )
        levels = ordered
        source: _CategoryOrderSource = "explicit"
    else:
        if mode == "alphabetical":
            levels = sorted(levels, key=lambda value: str(value).casefold())
            source = "alphabetical"
        elif mode == "severity":
            direction = 1.0 if severity_direction == "higher" else -1.0
            levels = sorted(
                levels,
                key=lambda value: (
                    direction * means[value],
                    str(value).casefold(),
                ),
            )
            source = "mean_signed_shap"
        else:
            dtype = getattr(values, "dtype", None)
            source = (
                "pandas_ordered"
                if isinstance(dtype, pd.CategoricalDtype) and dtype.ordered
                else "first_appearance"
            )

    if len(levels) > _MAX_DISPLAYED_CATEGORY_LEVELS:
        resolved_positions = {
            _category_key(level): index for index, level in enumerate(levels)
        }
        counts = {
            level: int(np.sum(_category_mask(values, level)))
            for level in levels
        }
        # Reserve the public ``Others`` label for the pooled tail so the key
        # cannot contain two visually indistinguishable entries.
        candidates = [
            level
            for level in levels
            if not (isinstance(level, str) and level == _OTHERS_LABEL)
        ]
        ranked = sorted(
            candidates,
            key=lambda level: (
                -counts[level],
                resolved_positions[_category_key(level)],
            ),
        )
        retained_keys = {
            _category_key(level)
            for level in ranked[:_RETAINED_HIGH_CARDINALITY_LEVELS]
        }
        retained = [
            level for level in levels if _category_key(level) in retained_keys
        ]
        omitted = tuple(
            level for level in levels if _category_key(level) not in retained_keys
        )
        pooled = _OtherCategories(omitted)
        pooled_mask = _category_mask(values, pooled)
        means[pooled] = float(np.mean(phi[pooled_mask]))
        levels = [*retained, pooled]
    return levels, means, source


def _category_order_note(
    order_sources: Sequence[_CategoryOrderSource],
    severity_direction: Literal["higher", "lower"],
    color_scheme: str = "prism",
    *,
    has_pooled_others: bool = False,
) -> str | None:
    """Describe a common resolved order without implying one that was overridden."""

    sources = set(order_sources)
    if sources == {"alphabetical"}:
        return "alphabetical order"
    if sources == {"mean_signed_shap"}:
        if has_pooled_others:
            return "retained levels: mean signed SHAP order  ·  Others = pooled tail"
        direction = (
            "lower → higher"
            if severity_direction == "higher"
            else "higher → lower"
        )
        if color_scheme == "prism":
            return (
                "mean signed SHAP order  ·  blue → pink = "
                f"{direction} model output"
            )
        return f"mean signed SHAP order  ·  {direction} model output"
    return None


def _normalise_style_mapping(
    mapping: Mapping[Hashable, str],
    categories: Sequence[Hashable],
    *,
    name: str,
) -> dict[tuple[type[object], object], str]:
    """Resolve public mapping keys against retained and pooled categories."""

    pooled = next(
        (
            category
            for category in categories
            if isinstance(category, _OtherCategories)
        ),
        None,
    )
    displayed_keys = {_category_key(value) for value in categories}
    omitted_keys = (
        set()
        if pooled is None
        else {_category_key(value) for value in pooled.levels}
    )
    resolved: dict[tuple[type[object], object], str] = {}
    unknown: list[Hashable] = []
    for raw_key, value in mapping.items():
        key = _category_key(raw_key)
        if pooled is not None and isinstance(raw_key, str) and raw_key == _OTHERS_LABEL:
            resolved[_category_key(pooled)] = value
        elif key in displayed_keys:
            resolved[key] = value
        elif key not in omitted_keys:
            unknown.append(raw_key)
    if unknown:
        raise ValueError(f"{name} contains unknown values: {unknown!r}")
    return resolved


def _styles_for_levels(
    categories: Sequence[Hashable],
    palette: Mapping[Hashable, str] | Sequence[str] | None = None,
    markers: Mapping[Hashable, str] | Sequence[str] | None = None,
    *,
    default_palettes: Mapping[int, tuple[str, ...]] = CATEGORY_PALETTES,
) -> dict[Hashable, _CategoryStyle]:
    count = len(categories)
    colors = default_palettes[count]
    supplied_colors: dict[tuple[type[object], object], str] | None = None
    listed_colors: list[str] | None = None
    colors_to_validate: list[str] = []
    supplied_markers: dict[tuple[type[object], object], str] | None = None
    listed_markers: list[str] | None = None
    markers_to_validate: list[str] = []
    if isinstance(palette, Mapping):
        colors_to_validate = list(palette.values())
        supplied_colors = _normalise_style_mapping(
            palette,
            categories,
            name="category_palette",
        )
    elif palette is not None:
        if isinstance(palette, (str, bytes)):
            raise TypeError("category_palette must be a color sequence or mapping")
        listed_colors = list(palette)
        colors_to_validate = listed_colors
        if len(listed_colors) < count:
            raise ValueError("category_palette has fewer colors than focal categories")
    for color in colors_to_validate:
        if not mpl.colors.is_color_like(color):
            raise ValueError(f"invalid category color: {color!r}")
        if mpl.colors.to_rgba(color)[3] < 1.0:
            raise ValueError("custom category colors must be opaque")
    if isinstance(markers, Mapping):
        markers_to_validate = list(markers.values())
        supplied_markers = _normalise_style_mapping(
            markers,
            categories,
            name="category_markers",
        )
    elif markers is not None:
        listed_markers = list(markers)
        markers_to_validate = listed_markers
        if len(listed_markers) < count:
            raise ValueError("category_markers has fewer markers than focal categories")
    for marker in markers_to_validate:
        try:
            MarkerStyle(marker)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid category marker: {marker!r}") from error
    resolved: dict[Hashable, _CategoryStyle] = {}
    for index, value in enumerate(categories):
        key = _category_key(value)
        color = colors[index]
        if supplied_colors is not None and key in supplied_colors:
            color = supplied_colors[key]
        elif listed_colors is not None:
            color = listed_colors[index]
        marker = _MARKERS[index]
        if supplied_markers is not None and key in supplied_markers:
            marker = supplied_markers[key]
        elif listed_markers is not None:
            marker = listed_markers[index]
        resolved[value] = _CategoryStyle(str(value), color, marker)
    return resolved


def _categorical_styles(
    values: np.ndarray,
    palette: Mapping[Hashable, str] | Sequence[str] | None,
    markers: Mapping[Hashable, str] | Sequence[str] | None,
    category_order: Sequence[Hashable] | None,
    *,
    default_palettes: Mapping[int, tuple[str, ...]] = CATEGORY_PALETTES,
) -> tuple[dict[Hashable, _CategoryStyle], bool]:
    missing = bool(np.any(pd.isna(values)))
    categories, _, _ = _resolve_category_levels(
        values,
        np.zeros(len(values), dtype=float),
        "data",
        explicit=category_order,
    )
    return (
        _styles_for_levels(
            categories,
            palette,
            markers,
            default_palettes=default_palettes,
        ),
        missing,
    )


def _draw_global(
    axis: plt.Axes,
    phi: np.ndarray,
    feature_frame: pd.DataFrame,
    names: Sequence[str],
    display_names: Mapping[str, str],
    *,
    focal_name: str | None,
    max_display: int,
    sample_limit: int,
    random_state: int,
    x_label: str,
    cmap: mpl.colors.Colormap,
    category_palettes: Mapping[int, tuple[str, ...]],
    category_palette: Sequence[str] | None,
    category_order_mode: Literal["data", "alphabetical", "severity"],
    severity_direction: Literal["higher", "lower"],
    focal_category_levels: Sequence[Hashable] | None = None,
    focal_category_means: Mapping[Hashable, float] | None = None,
    focal_category_order_source: _CategoryOrderSource | None = None,
    focal_styles: Mapping[Hashable, _CategoryStyle] | None = None,
) -> _GlobalRender:
    """Draw a SHAP summary with native categorical color and marker encoding."""

    importance = np.mean(np.abs(phi), axis=0)
    order = np.argsort(-importance, kind="stable")
    indices = _sample(
        np.arange(len(phi)), sample_limit, _derived_seed(random_state, 1009)
    )
    displayed = order[: min(max_display, len(names))]
    full_display_values = phi[:, displayed].reshape(-1)
    lower = min(0.0, float(np.min(full_display_values)))
    upper = max(0.0, float(np.max(full_display_values)))
    if lower == upper:
        global_xlim = (-1.0, 1.0)
    else:
        padding = max((upper - lower) * 0.035, np.finfo(float).eps)
        global_xlim = (lower - padding, upper + padding)
    positions = np.arange(len(displayed) - 1, -1, -1, dtype=float)
    edges = np.linspace(global_xlim[0], global_xlim[1], 70)
    shown_names: list[str] = []
    keys: list[_FeatureKey] = []
    collapsed: list[str] = []
    for row_y, feature_index in zip(positions, displayed, strict=True):
        name = names[int(feature_index)]
        display_name = display_names.get(name, name)
        shown_names.append(_wrapped_label(display_name, width=18))
        x = phi[indices, feature_index]
        offsets = _swarm_offsets(x, edges, width=0.28)
        full_series = feature_frame.iloc[:, feature_index]
        shown_series = feature_frame.iloc[indices, feature_index]
        if _is_categorical(full_series):
            if (
                name == focal_name
                and focal_styles is not None
                and focal_category_levels is not None
                and focal_category_means is not None
                and focal_category_order_source is not None
            ):
                levels = list(focal_category_levels)
                means = dict(focal_category_means)
                order_source = focal_category_order_source
                styles = dict(focal_styles)
            else:
                levels, means, order_source = _resolve_category_levels(
                    full_series,
                    phi[:, feature_index],
                    category_order_mode,
                    severity_direction=severity_direction,
                )
                styles = _styles_for_levels(
                    levels,
                    category_palette,
                    default_palettes=category_palettes,
                )
            assigned = np.zeros(len(indices), dtype=bool)
            for level in levels:
                mask = _category_mask(shown_series, level)
                assigned |= mask
                if np.any(mask):
                    style = styles[level]
                    axis.scatter(
                        x[mask],
                        row_y + offsets[mask],
                        color=style.color,
                        marker=style.marker,
                        s=10.5,
                        alpha=0.72,
                        edgecolors="none",
                        rasterized=True,
                        zorder=3,
                        gid=f"global-category:{name}:{style.label}",
                    )
            if np.any(~assigned):
                missing_label = _missing_category_label(levels)
                axis.scatter(
                    x[~assigned],
                    row_y + offsets[~assigned],
                    color=MISSING,
                    marker="X",
                    s=10.5,
                    alpha=0.75,
                    edgecolors="none",
                    rasterized=True,
                    zorder=3,
                    gid=f"global-category:{name}:{missing_label}",
                )
            pooled = next(
                (
                    level
                    for level in levels
                    if isinstance(level, _OtherCategories)
                ),
                None,
            )
            collapsed_levels = () if pooled is None else pooled.levels
            if collapsed_levels:
                collapsed.append(name)
            keys.append(
                _FeatureKey(
                    feature=name,
                    display_feature=display_name,
                    levels=tuple(levels),
                    styles=tuple(styles[level] for level in levels),
                    signed_means=tuple(means[level] for level in levels),
                    order_source=order_source,
                    missing=bool(full_series.isna().any()),
                    missing_row_count=int(full_series.isna().sum()),
                    original_level_count=(
                        len(levels)
                        if pooled is None
                        else len(levels) - 1 + len(collapsed_levels)
                    ),
                    collapsed_levels=tuple(collapsed_levels),
                    displayed_level_row_counts=tuple(
                        int(np.sum(_category_mask(full_series, level)))
                        for level in levels
                    ),
                )
            )
        else:
            numeric = pd.to_numeric(shown_series, errors="coerce").to_numpy(float)
            full_numeric = pd.to_numeric(
                full_series, errors="coerce"
            ).to_numpy(float)
            finite = np.isfinite(numeric)
            full_finite = np.isfinite(full_numeric)
            colors = np.repeat(
                np.asarray([mpl.colors.to_rgba(MISSING)]),
                len(numeric),
                axis=0,
            )
            if np.any(finite) and np.any(full_finite):
                low, high = np.quantile(
                    full_numeric[full_finite], [0.05, 0.95]
                )
                if high <= low:
                    low = np.min(full_numeric[full_finite])
                    high = np.max(full_numeric[full_finite])
                normalizer = mpl.colors.Normalize(
                    vmin=float(low), vmax=float(high), clip=True
                )
                colors[finite] = cmap(normalizer(numeric[finite]))
            if np.any(finite):
                axis.scatter(
                    x[finite],
                    row_y + offsets[finite],
                    c=colors[finite],
                    marker="o",
                    s=9.0,
                    alpha=0.68,
                    edgecolors="none",
                    rasterized=True,
                    zorder=3,
                    gid=f"global-continuous:{name}",
                )
            if np.any(~finite):
                axis.scatter(
                    x[~finite],
                    row_y + offsets[~finite],
                    color=MISSING,
                    marker="X",
                    s=10.5,
                    alpha=0.75,
                    edgecolors="none",
                    rasterized=True,
                    zorder=3,
                    gid=f"global-continuous-missing:{name}",
                )

    for row_y in positions:
        axis.hlines(
            row_y,
            *global_xlim,
            color=GRID,
            lw=0.55,
            linestyle=(0, (1.0, 4.0)),
            alpha=0.85,
            zorder=0,
        )
    axis.axvline(0, color=ZERO, lw=1.0, zorder=1)
    axis.set_xlim(global_xlim)
    axis.set_ylim(-0.60, len(displayed) - 0.40)
    axis.set_yticks(positions, shown_names)
    axis.set_xlabel(x_label, fontsize=8.15, labelpad=7)
    axis.tick_params(axis="x", labelsize=7.15, length=3, width=0.7, pad=3)
    axis.tick_params(axis="y", labelsize=7.35, length=0, pad=3)
    global_scale = float(np.max(np.abs(full_display_values)))
    if global_scale >= 1_000:
        axis.xaxis.set_major_locator(MaxNLocator(4))
        axis.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: "0" if abs(value) < 1e-12 else f"{value / 1_000:g}k")
        )
    else:
        axis.xaxis.set_major_locator(MaxNLocator(5))
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    for collection in axis.collections:
        collection.set_rasterized(True)
    if focal_name is not None:
        focal_label = _wrapped_label(
            display_names.get(focal_name, focal_name), width=18
        )
        for label in axis.get_yticklabels():
            if label.get_text() == focal_label:
                label.set_weight("bold")
    return _GlobalRender(
        order=tuple(int(index) for index in order),
        rendered_rows=int(len(indices)),
        xlim=(float(global_xlim[0]), float(global_xlim[1])),
        displayed_features=tuple(names[int(index)] for index in displayed),
        keys=tuple(keys),
        collapsed_categorical_features=tuple(collapsed),
    )


def _point_colors(
    values: np.ndarray,
    normalizer: mpl.colors.Normalize,
    cmap: mpl.colors.Colormap,
) -> np.ndarray:
    colors = np.repeat(np.asarray([mpl.colors.to_rgba(MISSING)]), len(values), axis=0)
    finite = np.isfinite(values)
    colors[finite] = cmap(normalizer(values[finite]))
    return colors


def _draw_points(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    *,
    normalizer: mpl.colors.Normalize | None,
    styles: Mapping[Hashable, _CategoryStyle] | None,
    cmap: mpl.colors.Colormap,
    size: float,
    gid: str,
) -> None:
    if styles is None:
        if normalizer is None:
            raise RuntimeError("continuous focal values require a normalizer")
        numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
        finite = np.isfinite(numeric)
        if np.any(finite):
            axis.scatter(
                x[finite],
                y[finite],
                c=_point_colors(numeric[finite], normalizer, cmap),
                s=size,
                alpha=0.70,
                edgecolors="none",
                rasterized=True,
                zorder=3,
                gid=gid,
            )
        if np.any(~finite):
            axis.scatter(
                x[~finite],
                y[~finite],
                c=MISSING,
                marker="X",
                s=size + 1.5,
                alpha=0.75,
                edgecolors="none",
                rasterized=True,
                zorder=3,
                gid=f"{gid}:missing",
            )
        return
    assigned = np.zeros(len(x), dtype=bool)
    for value, style in styles.items():
        mask = _category_mask(values, value)
        assigned |= mask
        if np.any(mask):
            axis.scatter(
                x[mask],
                y[mask],
                c=style.color,
                marker=style.marker,
                s=size + 1.5,
                alpha=0.72,
                edgecolors="none",
                rasterized=True,
                zorder=3,
                gid=gid,
            )
    if np.any(~assigned):
        axis.scatter(
            x[~assigned],
            y[~assigned],
            c=MISSING,
            marker="X",
            s=size + 1.5,
            alpha=0.75,
            edgecolors="none",
            rasterized=True,
            zorder=3,
            gid=gid,
        )


def _style_local_axis(axis: plt.Axes) -> None:
    axis.set_axisbelow(True)
    axis.grid(axis="x", color=GRID, linewidth=0.70, alpha=0.85, zorder=0)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color("#AEB6C2")
    axis.spines["bottom"].set_linewidth(0.8)
    axis.tick_params(axis="x", labelsize=7.4, length=3, width=0.7, pad=3)
    axis.tick_params(axis="y", length=0)


def _draw_stacked(
    axis: plt.Axes,
    focal_shap: np.ndarray,
    focal_values: np.ndarray,
    labels: np.ndarray,
    order: Sequence[Hashable],
    group_labels: Mapping[Hashable, str],
    *,
    limit: float,
    normalizer: mpl.colors.Normalize | None,
    styles: Mapping[Hashable, _CategoryStyle] | None,
    cmap: mpl.colors.Colormap,
    x_label: str,
    max_points: int,
    random_state: int,
    show_mean: bool,
) -> tuple[dict[str, float], dict[str, int]]:
    edges = np.linspace(-limit, limit, 76)
    positions = np.arange(len(order) - 1, -1, -1, dtype=float)
    resolved_labels = [
        group_labels.get(_group_key(group), str(group)) for group in order
    ]
    long_group_labels = max(map(len, resolved_labels)) > 20
    axis.set_xlim((-1.86 if long_group_labels else -1.42) * limit, 1.42 * limit)
    axis.set_ylim(-0.60, len(order) - 0.40)
    axis.set_xticks(np.linspace(-limit, limit, 5))
    axis.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: _format_tick(value, limit))
    )
    axis.set_yticks([])
    axis.set_xlabel(x_label, fontsize=8.35, labelpad=7)
    axis.axvline(0, color=ZERO, lw=1.05, zorder=1)
    for separator in np.arange(0.5, len(order) - 0.5, 1.0):
        axis.axhline(separator, color=GRID, lw=0.75, zorder=0)
    _style_local_axis(axis)
    if show_mean:
        axis.text(
            1.07 * limit,
            len(order) - 0.52,
            "Group mean",
            ha="left",
            va="bottom",
            fontsize=6.65,
            weight="bold",
            color=MUTED,
        )
    means: dict[str, float] = {}
    rendered: dict[str, int] = {}
    for index, (row_y, group) in enumerate(zip(positions, order, strict=True)):
        available = np.flatnonzero(_group_mask(labels, group))
        selected = _sample(
            available,
            max_points,
            _derived_seed(random_state, 8201 + index * 47),
        )
        x = focal_shap[selected]
        offsets = _swarm_offsets(x, edges, width=0.28)
        _draw_points(
            axis,
            x,
            row_y + offsets,
            focal_values[selected],
            normalizer=normalizer,
            styles=styles,
            cmap=cmap,
            size=11.5,
            gid=f"prism-points:{index}",
        )
        label = resolved_labels[index]
        visual_label = _wrapped_label(label, width=18)
        label_text = (
            f"{visual_label}\n(n={len(available):,})"
            if long_group_labels
            else f"{visual_label}  (n={len(available):,})"
        )
        axis.text(
            (-1.78 if long_group_labels else -1.03) * limit,
            row_y,
            label_text,
            ha="left" if long_group_labels else "right",
            va="center",
            fontsize=7.25 if long_group_labels else 7.7,
            linespacing=1.05,
            color=TEXT,
        )
        mean = float(np.mean(focal_shap[available]))
        means[label] = mean
        rendered[label] = int(len(selected))
        if show_mean:
            axis.scatter(
                [mean], [row_y], marker="D", s=35, facecolor="white",
                edgecolor=MEAN, linewidth=1.15, zorder=5,
                gid=f"prism-mean:{index}",
            )
            axis.text(
                1.07 * limit,
                row_y,
                _format_mean(mean, limit),
                ha="left",
                va="center",
                fontsize=7.35,
                color=TEXT,
            )
    return means, rendered


def _draw_wrapped(
    axes: np.ndarray,
    focal_shap: np.ndarray,
    focal_values: np.ndarray,
    labels: np.ndarray,
    order: Sequence[Hashable],
    group_labels: Mapping[Hashable, str],
    *,
    limit: float,
    normalizer: mpl.colors.Normalize | None,
    styles: Mapping[Hashable, _CategoryStyle] | None,
    cmap: mpl.colors.Colormap,
    x_label: str,
    max_points: int,
    random_state: int,
    show_mean: bool,
) -> tuple[dict[str, float], dict[str, int]]:
    ticks = np.linspace(-limit, limit, 3)
    means: dict[str, float] = {}
    rendered: dict[str, int] = {}
    for panel_index, axis in enumerate(axes.flat):
        if panel_index >= len(order):
            axis.set_visible(False)
            continue
        group = order[panel_index]
        available = np.flatnonzero(_group_mask(labels, group))
        selected = _sample(
            available,
            max_points,
            _derived_seed(random_state, 9101 + panel_index * 47),
        )
        x = focal_shap[selected]
        offsets = np.linspace(-0.25, 0.25, len(x))
        np.random.default_rng(
            _derived_seed(random_state, 12101 + panel_index * 137)
        ).shuffle(offsets)
        _draw_points(
            axis,
            x,
            offsets,
            focal_values[selected],
            normalizer=normalizer,
            styles=styles,
            cmap=cmap,
            size=14.0,
            gid=f"prism-points:{panel_index}",
        )
        axis.axvline(0, color=ZERO, lw=0.95, zorder=1)
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-0.42, 0.42)
        axis.set_xticks(ticks)
        axis.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _position: _format_tick(value, limit))
        )
        axis.set_yticks([])
        _style_local_axis(axis)
        label = group_labels.get(_group_key(group), str(group))
        visual_label = _wrapped_label(label, width=25)
        axis.set_title(
            f"{visual_label}  ·  n={len(available):,}",
            loc="left",
            fontsize=7.45,
            weight="bold",
            pad=4,
        )
        mean = float(np.mean(focal_shap[available]))
        means[label] = mean
        rendered[label] = int(len(selected))
        if show_mean:
            axis.scatter(
                [mean], [0], marker="D", s=31, facecolor="white",
                edgecolor=MEAN, linewidth=1.1, zorder=5,
                gid=f"prism-mean:{panel_index}",
            )
            axis.text(
                0.98, 0.90, f"mean {_format_mean(mean, limit)}",
                transform=axis.transAxes, ha="right", va="top", fontsize=6.1, color=MUTED
            )
        if panel_index // axes.shape[1] == axes.shape[0] - 1:
            axis.set_xlabel(x_label, fontsize=6.9, labelpad=4)
    return means, rendered


def _top_row_artist_top(
    local_axes: np.ndarray,
    renderer: object,
) -> float:
    """Return the rendered top of axes, titles, and annotations in the first row."""

    axes = (
        tuple(local_axes[0, :])
        if local_axes.shape[1] > 1
        else (local_axes[0, 0],)
    )
    tops: list[float] = []
    for axis in axes:
        if not axis.get_visible():
            continue
        tops.append(float(axis.get_tightbbox(renderer=renderer).y1))
    return max(tops)


def _category_entries(
    key: _FeatureKey,
) -> list[tuple[str, str, str]]:
    entries = [
        (style.label, style.color, style.marker)
        for style in key.styles
    ]
    if key.missing:
        entries.append((_missing_category_label(key.levels), MISSING, "X"))
    return entries


def _feature_key_summary(key: _FeatureKey) -> dict[str, object]:
    """Return auditable category-key metadata for one global feature."""

    retained_indices = [
        index
        for index, level in enumerate(key.levels)
        if not isinstance(level, _OtherCategories)
    ]
    pooled_index = next(
        (
            index
            for index, level in enumerate(key.levels)
            if isinstance(level, _OtherCategories)
        ),
        None,
    )
    return {
        "levels": [str(level) for level in key.levels],
        "colors": [mpl.colors.to_hex(style.color) for style in key.styles],
        "markers": [style.marker for style in key.styles],
        "mean_signed_shap": list(key.signed_means),
        "row_counts": list(key.displayed_level_row_counts),
        "order_source": key.order_source,
        "missing": key.missing,
        "missing_label": (
            _missing_category_label(key.levels) if key.missing else None
        ),
        "missing_row_count": key.missing_row_count,
        "original_level_count": key.original_level_count,
        "retained_levels": [str(key.levels[index]) for index in retained_indices],
        "retained_level_row_counts": [
            key.displayed_level_row_counts[index] for index in retained_indices
        ],
        "collapsed_levels": [str(level) for level in key.collapsed_levels],
        "collapsed_row_count": (
            0
            if pooled_index is None
            else key.displayed_level_row_counts[pooled_index]
        ),
        "high_cardinality_rule": (
            _HIGH_CARDINALITY_RULE if key.collapsed_levels else None
        ),
    }


def _draw_key_entries(
    axis: plt.Axes,
    entries: Sequence[tuple[str, str, str]],
    *,
    y: float,
    start: float,
    stop: float,
    columns: int,
    fontsize: float,
    gid_prefix: str | None = None,
) -> None:
    columns = max(1, min(columns, len(entries)))
    rows = int(ceil(len(entries) / columns))
    width = (stop - start) / columns
    offsets = [0.0] if rows == 1 else np.linspace(0.28, -0.28, rows)
    for index, (label, color, marker) in enumerate(entries):
        column, row = index % columns, index // columns
        x = start + column * width
        marker_artist = axis.scatter(
            [x],
            [y + offsets[row]],
            color=color,
            marker=marker,
            s=15,
            edgecolors="none",
            clip_on=False,
            zorder=3,
        )
        label_artist = axis.text(
            x + 0.030,
            y + offsets[row],
            _wrapped_label(label, width=18, lines=1),
            ha="left",
            va="center",
            fontsize=fontsize,
            color=TEXT,
        )
        if gid_prefix is not None:
            marker_artist.set_gid(f"{gid_prefix}:marker:{index}")
            label_artist.set_gid(f"{gid_prefix}:label:{index}")


def _draw_row_keys(
    axis: plt.Axes,
    keys: Sequence[_FeatureKey],
    displayed_features: Sequence[str],
) -> None:
    """Align each categorical key with its SHAP-summary feature row."""

    y_for = {
        feature: float(len(displayed_features) - 1 - index)
        for index, feature in enumerate(displayed_features)
    }
    axis.set_xlim(0, 1)
    axis.set_ylim(-0.60, len(displayed_features) - 0.40)
    axis.axis("off")
    for row_y in range(len(displayed_features)):
        axis.hlines(row_y - 0.5, 0, 1, color=GRID, lw=0.5, zorder=0)
    for key in keys:
        entries = _category_entries(key)
        _draw_key_entries(
            axis,
            entries,
            y=y_for[key.feature],
            start=0.03,
            stop=0.98,
            columns=min(2, len(entries)),
            fontsize=5.75,
            gid_prefix=f"category-key:{key.feature}",
        )


def _stacked_key_required_points(keys: Sequence[_FeatureKey]) -> float:
    """Return the vertical space needed by compact top-aligned key blocks."""

    required = 8.0
    for key in keys:
        feature_label = _wrapped_label(key.display_feature, width=30, lines=2)
        heading_lines = feature_label.count("\n") + 1
        entry_rows = max(1, int(ceil(len(_category_entries(key)) / 2)))
        required += 7.5 * heading_lines + 11.5 * entry_rows + 14.0
    return required


def _draw_stacked_keys(
    axis: plt.Axes,
    keys: Sequence[_FeatureKey],
) -> list[dict[str, object]]:
    """Draw compact feature-key blocks from the upper-right panel downward."""

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    if not keys:
        return []

    figure = axis.get_figure()
    figure.canvas.draw()
    axis_height = axis.get_window_extent(
        renderer=figure.canvas.get_renderer()
    ).height

    def points_to_axis_y(points: float) -> float:
        pixels = points * figure.dpi / 72.0
        return float(pixels / axis_height)

    cursor = 1.0 - points_to_axis_y(4.0)
    blocks: list[dict[str, object]] = []
    for key_index, key in enumerate(keys):
        entries = _category_entries(key)
        entry_rows = max(1, int(ceil(len(entries) / 2)))
        feature_label = _wrapped_label(key.display_feature, width=30, lines=2)
        heading_lines = feature_label.count("\n") + 1
        heading_height = points_to_axis_y(7.5 * heading_lines)
        entry_step = points_to_axis_y(11.5)
        first_entry_y = cursor - heading_height - points_to_axis_y(6.0)

        heading = axis.text(
            0.02,
            cursor,
            feature_label,
            ha="left",
            va="top",
            fontsize=6.0,
            linespacing=1.0,
            weight="bold",
            color=TEXT,
        )
        heading.set_gid(f"category-key-heading:{key.feature}")

        for entry_index, (label, color, marker) in enumerate(entries):
            column = entry_index % 2
            row = entry_index // 2
            x = 0.02 + 0.50 * column
            y = first_entry_y - entry_step * row
            marker_artist = axis.scatter(
                [x],
                [y],
                color=color,
                marker=marker,
                s=5.2**2,
                edgecolors="none",
                clip_on=False,
                zorder=3,
            )
            marker_artist.set_gid(
                f"category-key:{key.feature}:marker:{entry_index}"
            )
            label_artist = axis.text(
                x + _points_to_axis_x(axis, 5.0),
                y,
                _wrapped_label(label, width=17, lines=1),
                ha="left",
                va="center",
                fontsize=5.75,
                color=TEXT,
            )
            label_artist.set_gid(
                f"category-key:{key.feature}:label:{entry_index}"
            )

        block_bottom = (
            first_entry_y
            - entry_step * (entry_rows - 1)
            - points_to_axis_y(5.5)
        )
        axis.hlines(
            block_bottom - points_to_axis_y(1.0),
            0.02,
            0.98,
            color=GRID,
            lw=0.45,
            zorder=0,
        )
        blocks.append(
            {
                "feature": key.feature,
                "display_feature": key.display_feature,
                "index": key_index,
                "top": float(cursor),
                "bottom": float(block_bottom),
                "entry_rows": entry_rows,
            }
        )
        cursor = block_bottom - points_to_axis_y(8.0)
    return blocks


def _measure_text_width(
    axis: plt.Axes,
    text: str,
    *,
    fontsize: float,
    weight: str = "normal",
) -> float:
    figure = axis.get_figure()
    renderer = figure.canvas.get_renderer()
    probe = axis.text(
        0,
        0,
        text,
        fontsize=fontsize,
        weight=weight,
        transform=axis.transAxes,
        alpha=0,
    )
    extent = probe.get_window_extent(renderer=renderer)
    probe.remove()
    return float(extent.width / axis.get_window_extent(renderer=renderer).width)


def _points_to_axis_x(axis: plt.Axes, points: float) -> float:
    pixels = points * axis.get_figure().dpi / 72.0
    return float(pixels / axis.get_window_extent().width)


def _dock_label(label: str, max_levels: int) -> str:
    width = 13 if max_levels >= 7 else 16 if max_levels >= 5 else 22
    return _wrapped_label(label, width=width, lines=2)


def _measure_dock(
    axis: plt.Axes,
    keys: Sequence[_FeatureKey],
) -> tuple[_DockLayout, list[list[tuple[str, str, str]]]]:
    axis.get_figure().canvas.draw()
    raw_entries = [_category_entries(key) for key in keys]
    max_levels = max(map(len, raw_entries), default=0)
    entries = [
        [(_dock_label(label, max_levels), color, marker) for label, color, marker in row]
        for row in raw_entries
    ]
    left = _points_to_axis_x(axis, 6)
    right_limit = 1.0 - left
    base_font = 6.0 if max_levels <= 4 else 5.75 if max_levels <= 6 else 5.5
    for font_size in np.arange(base_font, 5.0, -0.25):
        marker_size = 5.5 if max_levels <= 4 else 5.25 if max_levels <= 6 else 5.0
        marker_gap = _points_to_axis_x(axis, 4.75)
        feature_gap = _points_to_axis_x(axis, 10)
        column_gap = _points_to_axis_x(axis, 8)
        block_gap = _points_to_axis_x(axis, 10)
        lane_safety = _points_to_axis_x(axis, 4)
        available_width = right_limit - left
        feature_widths: list[float] = []
        category_label_widths: list[tuple[float, ...]] = []
        minimum_category_widths: list[float] = []
        for key, row in zip(keys, entries, strict=True):
            feature_width = _measure_text_width(
                axis,
                key.display_feature,
                fontsize=float(font_size),
                weight="bold",
            )
            widths = [
                _measure_text_width(
                    axis,
                    label,
                    fontsize=float(font_size),
                )
                for label, _color, _marker in row
            ]
            feature_widths.append(float(feature_width))
            category_label_widths.append(tuple(float(width) for width in widths))
            minimum_category_widths.append(
                float(
                    marker_gap
                    + widths[-1]
                    + sum(
                        marker_gap + width + column_gap
                        for width in widths[:-1]
                    )
                )
            )

        if any(
            feature_width + feature_gap + category_width > available_width
            for feature_width, category_width in zip(
                feature_widths, minimum_category_widths, strict=True
            )
        ):
            continue

        def partitions(
            index: int,
        ) -> list[tuple[tuple[int, ...], ...]]:
            if index >= len(keys):
                return [()]
            candidates: list[tuple[tuple[int, ...], ...]] = []
            if index + 1 < len(keys):
                candidates.extend(
                    ((index, index + 1),) + tail
                    for tail in partitions(index + 2)
                )
            candidates.extend(
                ((index,),) + tail
                for tail in partitions(index + 1)
            )
            return candidates

        best_rows: tuple[tuple[_DockBlockLayout, ...], ...] | None = None
        best_pair_count = -1
        for candidate in partitions(0):
            lane_feature_widths = [
                max(feature_widths[row[0]] for row in candidate),
                max(
                    (
                        feature_widths[row[1]]
                        for row in candidate
                        if len(row) == 2
                    ),
                    default=0.0,
                ),
            ]
            lane_key_indices = [
                [row[0] for row in candidate],
                [row[1] for row in candidate if len(row) == 2],
            ]
            lane_offsets: list[tuple[float, ...]] = []
            for lane_indices in lane_key_indices:
                if not lane_indices:
                    lane_offsets.append((0.0,))
                    continue
                lane_level_count = max(
                    len(category_label_widths[key_index])
                    for key_index in lane_indices
                )
                offsets = [0.0]
                for ordinal in range(lane_level_count - 1):
                    preceding_width = max(
                        category_label_widths[key_index][ordinal]
                        for key_index in lane_indices
                        if len(category_label_widths[key_index]) > ordinal
                    )
                    offsets.append(
                        offsets[-1]
                        + marker_gap
                        + preceding_width
                        + column_gap
                    )
                lane_offsets.append(tuple(float(offset) for offset in offsets))

            def category_width(key_index: int, column: int) -> float:
                widths = category_label_widths[key_index]
                return float(
                    lane_offsets[column][len(widths) - 1]
                    + marker_gap
                    + widths[-1]
                )

            paired_second_start = left + max(
                (
                    lane_feature_widths[0]
                    + feature_gap
                    + category_width(row[0], 0)
                    for row in candidate
                    if len(row) == 2
                ),
                default=0.0,
            ) + block_gap + lane_safety
            resolved_rows: list[tuple[_DockBlockLayout, ...]] = []
            valid = True
            for row in candidate:
                starts = [left]
                first_width = (
                    lane_feature_widths[0]
                    + feature_gap
                    + category_width(row[0], 0)
                )
                if len(row) == 2:
                    second_width = (
                        lane_feature_widths[1]
                        + feature_gap
                        + category_width(row[1], 1)
                    )
                    starts.append(paired_second_start)
                    if starts[1] + second_width > right_limit:
                        valid = False
                        break
                elif left + first_width > right_limit:
                    valid = False
                    break

                row_blocks: list[_DockBlockLayout] = []
                for column, (key_index, start) in enumerate(
                    zip(row, starts, strict=True)
                ):
                    category_start = (
                        start + lane_feature_widths[column] + feature_gap
                    )
                    tabs = tuple(
                        float(category_start + offset)
                        for offset in lane_offsets[column][
                            : len(category_label_widths[key_index])
                        ]
                    )
                    row_blocks.append(
                        _DockBlockLayout(
                            key_index=key_index,
                            feature_x=float(start),
                            category_tabs=tabs,
                            right_edge=float(
                                category_start
                                + category_width(key_index, column)
                            ),
                        )
                    )
                resolved_rows.append(tuple(row_blocks))

            pair_count = sum(len(row) == 2 for row in candidate)
            if valid and pair_count > best_pair_count:
                best_rows = tuple(resolved_rows)
                best_pair_count = pair_count

        if best_rows is not None:
            right_edge = max(
                block.right_edge
                for row in best_rows
                for block in row
            )
            return (
                _DockLayout(
                    font_size=float(font_size),
                    marker_size=marker_size,
                    marker_text_gap=marker_gap,
                    rows=best_rows,
                    right_edge=float(right_edge),
                ),
                entries,
            )
    raise ValueError(
        "category labels do not fit the publication-width bottom dock; "
        "shorten display_names or category labels"
    )


def _draw_bottom_dock(
    axis: plt.Axes,
    keys: Sequence[_FeatureKey],
    *,
    layout: _DockLayout | None = None,
    all_entries: list[list[tuple[str, str, str]]] | None = None,
) -> _DockLayout:
    """Draw measured feature-key blocks, packing at most two blocks per row."""

    if layout is None or all_entries is None:
        layout, all_entries = _measure_dock(axis, keys)
    rows = len(layout.rows)
    axis.set_xlim(0, 1)
    axis.set_ylim(-0.55, rows - 0.45)
    axis.axis("off")
    for row_index, blocks in enumerate(layout.rows):
        row_y = float(rows - 1 - row_index)
        axis.hlines(row_y - 0.48, 0, 1, color=GRID, lw=0.45, zorder=0)
        if len(blocks) == 2:
            divider_x = (
                blocks[0].right_edge + blocks[1].feature_x
            ) / 2.0
            axis.vlines(
                divider_x,
                row_y - 0.31,
                row_y + 0.31,
                color=GRID,
                lw=0.45,
                zorder=0,
            )
        for block in blocks:
            key = keys[block.key_index]
            entries = all_entries[block.key_index]
            axis.text(
                block.feature_x,
                row_y,
                key.display_feature,
                ha="left",
                va="center",
                fontsize=layout.font_size,
                weight="bold",
                color=TEXT,
                gid=f"category-key-heading:{key.feature}",
            )
            for entry_index, (tab, (label, color, marker)) in enumerate(
                zip(block.category_tabs, entries, strict=True)
            ):
                marker_artist = axis.scatter(
                    [tab],
                    [row_y],
                    color=color,
                    marker=marker,
                    s=layout.marker_size**2,
                    edgecolors="none",
                    clip_on=False,
                    zorder=3,
                )
                marker_artist.set_gid(
                    f"category-key:{key.feature}:marker:{entry_index}"
                )
                label_artist = axis.text(
                    tab + layout.marker_text_gap,
                    row_y,
                    label,
                    ha="left",
                    va="center",
                    fontsize=layout.font_size,
                    linespacing=0.95,
                    color=TEXT,
                )
                label_artist.set_gid(
                    f"category-key:{key.feature}:label:{entry_index}"
                )
    return layout


def _summarize_dock_layout(
    layout: _DockLayout | None,
    keys: Sequence[_FeatureKey],
) -> dict[str, object] | None:
    if layout is None:
        return None
    return {
        "alignment": "left-aligned lane tab stops",
        "packing": "compact adjacent fit; at most two feature blocks per row",
        "font_size": layout.font_size,
        "max_blocks_per_row": 2,
        "minimum_block_clearance_points": 14.0,
        "packed_row_count": len(layout.rows),
        "rows": [
            [keys[block.key_index].feature for block in row]
            for row in layout.rows
        ],
        "blocks": [
            {
                "feature": keys[block.key_index].feature,
                "display_feature": keys[block.key_index].display_feature,
                "row": row_index,
                "column": column_index,
                "feature_x": block.feature_x,
                "category_tabs": list(block.category_tabs),
                "right_edge": block.right_edge,
            }
            for row_index, row in enumerate(layout.rows)
            for column_index, block in enumerate(row)
        ],
    }


def _add_continuous_key(
    figure: plt.Figure,
    global_axis: plt.Axes,
    *,
    cmap: mpl.colors.Colormap,
) -> plt.Axes:
    box = global_axis.get_position()
    axis = figure.add_axes([box.x1 + 0.007, box.y0, 0.010, box.height])
    axis.imshow(np.linspace(0, 1, 160)[:, None], aspect="auto", cmap=cmap, origin="lower")
    axis.set_xticks([])
    axis.set_yticks([0, 159], ["Low", "High"], fontsize=5.9)
    axis.yaxis.tick_right()
    axis.tick_params(length=0, pad=3, colors=MUTED)
    axis.set_ylabel(
        "Continuous feature value",
        rotation=90,
        fontsize=5.9,
        color=MUTED,
        labelpad=6,
    )
    for spine in axis.spines.values():
        spine.set_visible(False)
    return axis


def _save_outputs(figure: plt.Figure, output: str | Path, dpi: int) -> list[str]:
    path = Path(output)
    if path.suffix:
        if path.suffix.lower() not in {".png", ".pdf"}:
            raise ValueError("output suffix must be .png or .pdf")
        targets = [path]
    else:
        targets = [path.with_suffix(".png"), path.with_suffix(".pdf")]
    written = atomic_save_figure(figure, targets, png_dpi=dpi)
    return [str(target) for target in written]


def plot_summary(
    shap_values: ArrayLike,
    features: ArrayLike | pd.DataFrame,
    *,
    feature_names: Sequence[str] | None = None,
    display_names: Mapping[str, str] | None = None,
    category_order_mode: Literal["data", "alphabetical", "severity"] = "data",
    severity_direction: Literal["higher", "lower"] = "higher",
    category_key_placement: Literal[
        "row_aligned", "bottom", "right_stacked"
    ] = "row_aligned",
    color_scheme: Literal["prism", "okabe_ito"] = "prism",
    category_palette: Sequence[str] | None = None,
    continuous_cmap: str | mpl.colors.Colormap | Sequence[str] | None = None,
    title: str | None = None,
    note: str | None = None,
    x_label: str = "SHAP value",
    max_features: int = 10,
    max_points: int = 520,
    random_state: int = 4701,
    output: str | Path | None = None,
    dpi: int = 280,
) -> PrismResult:
    """Plot a categorical-aware global SHAP summary.

    By default, continuous rows use the Prism blue-purple-pink gradient and
    categorical rows use the same ordered hue path plus redundant markers.
    The selected scheme or explicit channel overrides are applied consistently
    to dots and keys. For more than eight observed nonmissing levels, seven raw
    levels retain separate encodings under the documented frequency-and-tie
    rule and the remainder are pooled into a final ``Others`` entry. A literal
    raw ``Others`` level is always included in that pooled entry.

    Parameters
    ----------
    shap_values:
        Finite ``(n_rows, n_features)`` SHAP matrix for one model output.
    features:
        Feature matrix with the same shape. A DataFrame supplies feature names
        and enables index and column checks against pandas SHAP inputs.
    feature_names:
        Feature names required when ``features`` is an array.
    display_names:
        Optional mapping from stored feature names to display labels.
    category_order_mode:
        ``"data"`` keeps declared or first-observed category order,
        ``"alphabetical"`` sorts labels, and ``"severity"`` sorts each
        category row by its category-level mean signed SHAP value.
    severity_direction:
        In severity mode, whether higher or lower explained model output is
        more severe. Under the Prism scheme, the pink endpoint is the more
        severe category for unpooled rows; under Okabe-Ito, the ordered key
        carries severity. For pooled rows, ``Others`` is forced last and its
        color is only an identifier.
    category_key_placement:
        ``"row_aligned"`` keeps each category key beside its feature row,
        ``"bottom"`` places adaptively packed keys below the single panel,
        and ``"right_stacked"`` lists compact feature-key blocks from the
        upper right downward. The default preserves the original layout.
    color_scheme:
        ``"prism"`` keeps the blue-purple-pink house style. ``"okabe_ito"``
        uses the Okabe-Ito qualitative colors for categorical values and the
        color-vision-friendly ``cividis`` map for continuous values. Markers
        remain redundant with categorical color in both schemes.
    category_palette:
        Optional sequence of one through eight colors applied, in resolved
        category order, to every encoded categorical row. It overrides the
        categorical part of ``color_scheme``.
    continuous_cmap:
        Optional Matplotlib colormap, registered colormap name, or sequence of
        at least two colors. It overrides the continuous part of
        ``color_scheme``.
    title:
        Optional figure-level title.
    note:
        Optional short annotation printed below the panel headings.
    x_label:
        Label for the SHAP-value axis.
    max_features:
        Maximum number of global feature rows to display, from 1 through 15.
    max_points:
        Target-blind display cap per row. Feature ranking, axis limits, and
        category summaries still use all rows.
    random_state:
        Seed for display sampling and jitter only.
    output:
        A ``.png`` or ``.pdf`` filename. A suffix-free stem writes both.
    dpi:
        PNG resolution. PDF text and axes remain vector objects.

    Returns
    -------
    PrismResult:
        A small object exposing ``figure``, named ``axes``, and a
        JSON-serializable ``summary``. ``axes["global"]`` contains the
        beeswarm and ``axes["category_keys"]`` contains the active category
        key panel when categorical keys are present.

    Raises
    ------
    TypeError
        If an integer control has the wrong type.
    ValueError
        If shapes, pandas alignment, feature names, parameter ranges, or the
        requested output format violate the plotting contract.

    Notes
    -----
    The plot is descriptive and does not establish causal effects or fairness.
    High-cardinality retention uses all aligned rows, not the display sample.
    Frequency ties follow the resolved category order. The returned summary
    records retained and collapsed levels and their row counts.
    """

    phi, _, feature_frame, names, input_alignment = _resolve_inputs(
        shap_values, features, feature_names
    )
    shown = _resolve_display_names(display_names, names)
    max_features = _require_integer(
        "max_features", max_features, minimum=1, maximum=15
    )
    max_points = _require_integer("max_points", max_points, minimum=20)
    random_state = _require_integer(
        "random_state", random_state, minimum=0, maximum=2**32 - 1
    )
    dpi = _require_integer("dpi", dpi, minimum=100, maximum=1200)
    if category_order_mode not in _CATEGORY_ORDER_MODES:
        raise ValueError(
            "category_order_mode must be 'data', 'alphabetical', or 'severity'"
        )
    if severity_direction not in {"higher", "lower"}:
        raise ValueError("severity_direction must be 'higher' or 'lower'")
    if category_key_placement not in _CATEGORY_KEY_PLACEMENTS:
        raise ValueError(
            "category_key_placement must be 'row_aligned', 'bottom', "
            "or 'right_stacked'"
        )
    if title is not None and (not isinstance(title, str) or not title.strip()):
        raise ValueError("title must be a non-empty string when provided")
    if note is not None and (not isinstance(note, str) or not note.strip()):
        raise ValueError("note must be a non-empty string when provided")
    color_scheme, category_palettes, scheme_cmap = _resolve_color_scheme(
        color_scheme
    )
    shared_category_palette = _resolve_shared_category_palette(
        category_palette,
        name="category_palette",
    )
    cmap = _resolve_cmap(continuous_cmap, default=scheme_cmap)
    displayed_rows = min(max_features, len(names))
    base_height = max(4.85, 1.55 + 0.46 * displayed_rows)
    height = base_height
    title_text = None if title is None else _wrapped_label(title.strip(), width=70)
    plot_top = 0.825 if title_text else 0.865
    header_y = 0.895 if title_text else 0.925
    has_numeric = any(
        not _is_categorical(feature_frame.iloc[:, index])
        for index in np.argsort(-np.mean(np.abs(phi), axis=0), kind="stable")[
            :displayed_rows
        ]
    )
    dock_axis: plt.Axes | None = None
    dock_layout: _DockLayout | None = None
    dock_keys: list[_FeatureKey] = []
    stacked_blocks: list[dict[str, object]] = []

    with mpl.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": TEXT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    ):
        figure = plt.figure(figsize=(7.35, height), facecolor="white")
        if category_key_placement == "bottom":
            outer = figure.add_gridspec(
                1,
                1,
                left=0.155,
                right=0.885,
                top=plot_top,
                bottom=0.135,
            )
            global_axis = figure.add_subplot(outer[0, 0])
            key_axis: plt.Axes | None = None
        else:
            outer = figure.add_gridspec(
                1,
                2,
                width_ratios=(0.56, 0.44),
                left=0.155,
                right=0.965,
                top=plot_top,
                bottom=0.135,
                wspace=0.22,
            )
            global_axis = figure.add_subplot(outer[0, 0])
            key_axis = figure.add_subplot(outer[0, 1])
        rendered = _draw_global(
            global_axis,
            phi,
            feature_frame,
            names,
            shown,
            focal_name=None,
            max_display=max_features,
            sample_limit=max_points,
            random_state=random_state,
            x_label=x_label,
            cmap=cmap,
            category_palettes=category_palettes,
            category_palette=shared_category_palette,
            category_order_mode=category_order_mode,
            severity_direction=severity_direction,
        )
        dock_keys = list(rendered.keys)
        if category_key_placement == "row_aligned":
            if key_axis is None:
                raise RuntimeError("row-aligned keys require a side axis")
            _draw_row_keys(
                key_axis,
                rendered.keys,
                rendered.displayed_features,
            )
            category_key_axis = key_axis
        elif category_key_placement == "right_stacked":
            if key_axis is None:
                raise RuntimeError("right-stacked keys require a side axis")
            figure.canvas.draw()
            required_points = _stacked_key_required_points(rendered.keys)
            available_points = (
                key_axis.get_window_extent(
                    renderer=figure.canvas.get_renderer()
                ).height
                * 72.0
                / figure.dpi
            )
            if rendered.keys and required_points > available_points:
                panel_fraction = plot_top - 0.135
                height += (
                    (required_points - available_points)
                    / 72.0
                    / panel_fraction
                    + 0.06
                )
                figure.set_size_inches(7.35, height, forward=True)
                figure.canvas.draw()
            stacked_blocks = _draw_stacked_keys(key_axis, rendered.keys)
            category_key_axis = key_axis
        else:
            category_key_axis = None
            if dock_keys:
                measurement_axis = figure.add_axes(
                    [0.04, 0.01, 0.92, 0.025]
                )
                measured_layout, measured_entries = _measure_dock(
                    measurement_axis,
                    dock_keys,
                )
                measurement_axis.remove()
                packed_rows = len(measured_layout.rows)
                dock_inches = 0.20 + 0.29 * packed_rows
                dock_bottom = 0.060
                baseline_panel_inches = (plot_top - 0.135) * base_height
                height = max(
                    base_height,
                    (baseline_panel_inches + dock_inches)
                    / (plot_top - dock_bottom - 0.115),
                )
                dock_height = dock_inches / height
                plot_bottom = dock_bottom + dock_height + 0.115
                figure.set_size_inches(7.35, height, forward=True)
                outer.update(bottom=plot_bottom)
                figure.canvas.draw()
                dock_axis = figure.add_axes(
                    [
                        0.04,
                        dock_bottom,
                        0.92,
                        max(dock_height - 0.012, 0.025),
                    ]
                )
                dock_layout = _draw_bottom_dock(
                    dock_axis,
                    dock_keys,
                    layout=measured_layout,
                    all_entries=measured_entries,
                )
                category_key_axis = dock_axis
                figure.text(
                    0.04,
                    dock_bottom + dock_height + 0.005,
                    "CATEGORY KEYS",
                    ha="left",
                    va="bottom",
                    fontsize=6.55,
                    weight="bold",
                    color=MUTED,
                )
                bottom_order_note = _category_order_note(
                    [key.order_source for key in dock_keys],
                    severity_direction,
                    color_scheme,
                    has_pooled_others=any(
                        key.collapsed_levels for key in dock_keys
                    ),
                )
                if bottom_order_note is not None:
                    figure.text(
                        0.925,
                        dock_bottom + dock_height + 0.005,
                        bottom_order_note,
                        ha="right",
                        va="bottom",
                        fontsize=5.75,
                        color=MUTED,
                    )
        if has_numeric:
            continuous_axis = _add_continuous_key(
                figure, global_axis, cmap=cmap
            )
        else:
            continuous_axis = None
        if title_text:
            figure.text(
                0.04, 0.965, title_text, ha="left", va="top",
                fontsize=12.0, weight="bold", color=TEXT
            )
        figure.text(
            0.04, header_y, "GLOBAL SHAP SUMMARY", ha="left", va="center",
            fontsize=7.15, weight="bold", color=MUTED
        )
        if key_axis is not None and rendered.keys:
            key_box = key_axis.get_position()
            figure.text(
                key_box.x0,
                header_y,
                "CATEGORY KEYS",
                ha="left",
                va="center",
                fontsize=7.15,
                weight="bold",
                color=MUTED,
            )
            order_note = _category_order_note(
                [key.order_source for key in rendered.keys],
                severity_direction,
                color_scheme,
                has_pooled_others=any(
                    key.collapsed_levels for key in rendered.keys
                ),
            )
            if order_note is not None:
                figure.text(
                    key_box.x1,
                    header_y - 0.026,
                    order_note,
                    ha="right",
                    va="center",
                    fontsize=5.9,
                    color=MUTED,
                )
        if note:
            figure.text(
                0.965, header_y - 0.052, _wrapped_label(note.strip(), width=75),
                ha="right", va="center", fontsize=6.4, color=MUTED
            )
        output_files = (
            _save_outputs(figure, output, dpi) if output is not None else []
        )

    category_keys = {
        key.feature: _feature_key_summary(key) for key in rendered.keys
    }
    category_dock_layout_summary = _summarize_dock_layout(
        dock_layout,
        dock_keys,
    )
    if category_key_placement == "bottom":
        category_key_layout: dict[str, object] = {
            "placement": "bottom",
            "features": [key.feature for key in rendered.keys],
            "rows": (
                []
                if category_dock_layout_summary is None
                else category_dock_layout_summary["rows"]
            ),
        }
    elif category_key_placement == "right_stacked":
        category_key_layout = {
            "placement": "right_stacked",
            "features": [key.feature for key in rendered.keys],
            "blocks": stacked_blocks,
        }
    else:
        category_key_layout = {
            "placement": "row_aligned",
            "features": [key.feature for key in rendered.keys],
            "rows": [[key.feature] for key in rendered.keys],
        }
    summary: dict[str, object] = {
        "summary_schema_version": "2.0",
        "package_version": __version__,
        "software_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": mpl.__version__,
        },
        "view": "global_summary",
        "color_scheme": color_scheme,
        "resolved_continuous_cmap": cmap.name,
        "continuous_color_endpoints": [
            mpl.colors.to_hex(cmap(0.0)),
            mpl.colors.to_hex(cmap(1.0)),
        ],
        "n_rows": int(len(phi)),
        "n_features": int(len(names)),
        "global_feature_order": [names[index] for index in rendered.order],
        "displayed_global_features": list(rendered.displayed_features),
        "global_rendered_rows": rendered.rendered_rows,
        "global_shap_xlim": list(rendered.xlim),
        "global_sampling": (
            "deterministic random sample" if len(phi) > max_points else "all rows"
        ),
        "category_order_mode": category_order_mode,
        "severity_direction": severity_direction,
        "category_order_sources": {
            key.feature: key.order_source for key in rendered.keys
        },
        "category_key_placement": category_key_placement,
        "category_key_features": [key.feature for key in rendered.keys],
        "category_key_layout": category_key_layout,
        "category_dock_layout": category_dock_layout_summary,
        "global_category_keys": category_keys,
        "unencoded_high_cardinality_features": [],
        "collapsed_high_cardinality_features": list(
            rendered.collapsed_categorical_features
        ),
        "input_alignment": input_alignment,
        "display_parameters": {
            "max_features": max_features,
            "max_points": max_points,
            "random_state": random_state,
            "dpi": dpi,
            "title": title,
            "note": note,
            "x_label": x_label,
            "category_key_placement": category_key_placement,
            "color_scheme": color_scheme,
            "custom_category_palette": (
                None
                if shared_category_palette is None
                else [mpl.colors.to_hex(color) for color in shared_category_palette]
            ),
            "continuous_cmap": cmap.name,
            "figure_size_inches": [
                float(value) for value in figure.get_size_inches()
            ],
        },
        "rasterized_points": True,
        "pdf_rendering": "hybrid: rasterized point clouds with vector text and axes",
        "output_files": output_files,
    }
    axes: dict[str, object] = {
        "global": global_axis,
        "category_keys": category_key_axis,
        "category_dock": dock_axis,
        "continuous_key": continuous_axis,
        "subgroups": (),
    }
    return PrismResult(figure=figure, axes=axes, summary=summary)


def plot_prism(
    shap_values: ArrayLike,
    features: ArrayLike | pd.DataFrame,
    groups: Sequence[Hashable],
    focal_feature: str | int,
    *,
    feature_names: Sequence[str] | None = None,
    focal_values: Sequence[object] | None = None,
    focal_kind: Literal["auto", "continuous", "categorical"] = "auto",
    display_names: Mapping[str, str] | None = None,
    group_order: Sequence[Hashable] | None = None,
    group_labels: Mapping[Hashable, str] | None = None,
    category_order: Sequence[Hashable] | None = None,
    category_order_mode: Literal["data", "alphabetical", "severity"] = "data",
    severity_direction: Literal["higher", "lower"] = "higher",
    color_scheme: Literal["prism", "okabe_ito"] = "prism",
    category_palette: Mapping[Hashable, str] | Sequence[str] | None = None,
    global_category_palette: Sequence[str] | None = None,
    category_markers: Mapping[Hashable, str] | Sequence[str] | None = None,
    category_legend_columns: int | None = None,
    continuous_cmap: str | mpl.colors.Colormap | Sequence[str] | None = None,
    title: str | None = None,
    note: str | None = None,
    global_x_label: str = "SHAP value",
    local_x_label: str | None = None,
    focal_value_label: str | None = None,
    max_global_features: int = 8,
    show_mean: bool = True,
    max_points: int = 520,
    random_state: int = 4701,
    output: str | Path | None = None,
    dpi: int = 280,
) -> PrismResult:
    """Plot a global SHAP summary beside one feature split by subgroup.

    Parameters
    ----------
    shap_values:
        Finite ``(n_rows, n_features)`` SHAP matrix for one model output.
    features:
        Feature matrix. A DataFrame lets the function infer names and enables
        index checks against pandas inputs.
    groups:
        One subgroup label per row. Two through six groups are supported. A
        Series must have the same index as a feature DataFrame.
    focal_feature:
        Name or zero-based index of the feature expanded on the right.
    feature_names:
        Feature names required when ``features`` is an array.
    focal_values:
        Optional display values for the focal feature. This is useful when the
        model matrix contains an encoded version of a human-readable variable.
    focal_kind:
        ``"continuous"`` uses the selected continuous map; ``"categorical"``
        uses distinct colors and markers; ``"auto"`` infers this from dtype.
    display_names, group_labels:
        Optional mappings that change labels without changing stored values.
    group_order:
        Complete display order for subgroups.
    category_order_mode:
        ``"data"`` keeps declared/first-observed order, ``"alphabetical"``
        sorts labels, and ``"severity"`` sorts category-level mean signed SHAP.
    severity_direction:
        In severity mode, whether higher or lower explained model output is
        more severe. Under the Prism scheme, the pink endpoint is the more
        severe category; under Okabe-Ito, the ordered key carries severity
        rather than a hue progression. For pooled rows, ``Others`` is forced
        last and its color is only an identifier.
    color_scheme:
        ``"prism"`` keeps the blue-purple-pink house style. ``"okabe_ito"``
        uses the Okabe-Ito qualitative colors for categorical values and the
        color-vision-friendly ``cividis`` map for continuous values.
    max_global_features:
        Maximum number of feature rows in the conventional left summary.
    show_mean:
        Show a white diamond and signed mean beside every subgroup.
    category_palette:
        Optional sequence of colors or mapping from categorical value to color.
        This controls the focal feature and overrides ``color_scheme`` there.
        Markers remain redundant with color for readability and accessibility.
        In a pooled focal feature, the mapping key ``"Others"`` controls the
        pooled entry; mappings for collapsed raw levels are accepted but not
        displayed separately.
    global_category_palette:
        Optional sequence of one through eight colors applied to nonfocal
        categorical rows in the global summary. It overrides the categorical
        part of ``color_scheme`` without changing the focal mapping.
    category_markers:
        Optional sequence or value-to-marker mapping for categorical focal
        values. Defaults are supplied when this is omitted. The mapping key
        ``"Others"`` controls a pooled entry.
    category_order:
        Optional display order containing every raw observed categorical level.
        Ordered pandas categoricals are respected when this is omitted. When
        more than eight levels are observed, this full order controls display
        order and frequency-tie resolution before the pooled ``Others`` entry
        is placed last.
    category_legend_columns:
        Number of columns, from one through eight, in the categorical
        focal-feature legend. The value is capped at the number of displayed
        levels, including a missing-value entry. When omitted, columns are
        packed automatically. It is ignored for a continuous focal feature
        and does not affect the bottom global-category dock.
    continuous_cmap:
        Optional Matplotlib colormap, registered colormap name, or sequence of
        at least two colors used by the global summary and continuous focal
        values. It overrides the continuous part of ``color_scheme``.
    title:
        Optional figure-level title. Publication figures can omit it and use
        the manuscript caption instead.
    note:
        Optional short annotation printed below the panel headings.
    global_x_label, local_x_label, focal_value_label:
        Axis and legend labels. Label the explained model-output scale.
    max_points:
        Target-blind display cap for the global sample and for each subgroup.
        Means and axis limits still use all rows.
    random_state:
        Seed for display sampling and jitter only.
    output:
        A ``.png`` or ``.pdf`` filename. A suffix-free stem writes both.
    dpi:
        PNG resolution. PDF text and axes remain vector objects.

    Returns
    -------
    PrismResult:
        A small object exposing ``figure``, named ``axes``, and a
        JSON-serializable ``summary``. ``axes["global"]`` is one axis and
        ``axes["subgroups"]`` is a tuple of visible subgroup axes. Counts,
        means, limits, and feature order use all rows; rendered-row counts
        describe the display sample.

    Raises
    ------
    TypeError
        If an integer control or focal-feature index has the wrong type.
    ValueError
        If shapes, pandas indices, labels, encodings, parameter ranges, or
        requested output formats violate the plotting contract.

    Notes
    -----
    The plot is descriptive. It does not establish causal effects or fairness,
    and group differences can reflect different feature distributions. Use
    matched or common-support rows when that distinction matters scientifically.
    Every categorical path uses at most eight nonmissing display entries. For
    more than eight raw levels, seven are encoded separately under the
    documented frequency-and-tie rule and all remaining rows are encoded as a
    final ``Others`` level. Frequency ties follow the resolved category order;
    a literal raw ``Others`` level is always included in the pooled entry.
    The bottom category dock preserves feature order and automatically packs
    at most two adjacent feature keys on a row when their rendered blocks fit.
    """

    phi, _, feature_frame, names, input_alignment = _resolve_inputs(
        shap_values, features, feature_names
    )
    focal_index, focal_name = _resolve_focal(focal_feature, names)
    if isinstance(features, pd.DataFrame) and isinstance(groups, pd.Series):
        if not groups.index.is_unique:
            raise ValueError("groups Series index must be unique")
        if not groups.index.equals(features.index):
            raise ValueError("groups Series index must match features exactly")
        input_alignment["groups_to_features"] = "index checked"
    else:
        input_alignment["groups_to_features"] = "positional"
    labels, order = _resolve_groups(groups, group_order, len(phi))
    if focal_kind not in {"auto", "continuous", "categorical"}:
        raise ValueError("focal_kind must be 'auto', 'continuous', or 'categorical'")
    if category_order_mode not in _CATEGORY_ORDER_MODES:
        raise ValueError(
            "category_order_mode must be 'data', 'alphabetical', or 'severity'"
        )
    if severity_direction not in {"higher", "lower"}:
        raise ValueError("severity_direction must be 'higher' or 'lower'")
    max_global_features = _require_integer(
        "max_global_features", max_global_features, minimum=1, maximum=15
    )
    max_points = _require_integer("max_points", max_points, minimum=20)
    random_state = _require_integer(
        "random_state", random_state, minimum=0, maximum=2**32 - 1
    )
    dpi = _require_integer("dpi", dpi, minimum=100, maximum=1200)
    if category_legend_columns is not None:
        category_legend_columns = _require_integer(
            "category_legend_columns",
            category_legend_columns,
            minimum=1,
            maximum=8,
        )
    if title is not None and (not isinstance(title, str) or not title.strip()):
        raise ValueError("title must be a non-empty string when provided")
    if note is not None and (not isinstance(note, str) or not note.strip()):
        raise ValueError("note must be a non-empty string when provided")
    color_scheme, category_palettes, scheme_cmap = _resolve_color_scheme(
        color_scheme
    )
    shared_global_category_palette = _resolve_shared_category_palette(
        global_category_palette,
        name="global_category_palette",
    )
    cmap = _resolve_cmap(continuous_cmap, default=scheme_cmap)

    if (
        focal_values is not None
        and isinstance(features, pd.DataFrame)
        and isinstance(focal_values, pd.Series)
    ):
        if not focal_values.index.is_unique:
            raise ValueError("focal_values Series index must be unique")
        if not focal_values.index.equals(features.index):
            raise ValueError("focal_values Series index must match features exactly")
        input_alignment["focal_values_to_features"] = "index checked"
    elif focal_values is not None:
        input_alignment["focal_values_to_features"] = "positional"
    else:
        input_alignment["focal_values_to_features"] = "feature column"
    values = (
        feature_frame.iloc[:, focal_index].to_numpy()
        if focal_values is None
        else np.asarray(focal_values)
    )
    if np.asarray(values).ndim != 1 or len(values) != len(phi):
        raise ValueError("focal_values must be one-dimensional and match the rows")

    inferred_categorical = False
    if focal_kind == "auto":
        if category_palette is not None:
            inferred_categorical = True
        elif focal_values is None and isinstance(features, pd.DataFrame):
            dtype = features.iloc[:, focal_index].dtype
            inferred_categorical = bool(
                isinstance(dtype, pd.CategoricalDtype)
                or pd.api.types.is_object_dtype(dtype)
                or pd.api.types.is_bool_dtype(dtype)
                or pd.api.types.is_string_dtype(dtype)
            )
        else:
            inferred_categorical = np.asarray(values).dtype.kind in "OUSb"
    categorical = focal_kind == "categorical" or inferred_categorical

    styles: dict[Hashable, _CategoryStyle] | None = None
    focal_category_means: dict[Hashable, float] = {}
    focal_category_order_source: _CategoryOrderSource | None = None
    has_missing_category = False
    normalizer: mpl.colors.Normalize | None = None
    continuous_color_scale: dict[str, object] | None = None
    if categorical:
        category_source: Sequence[object] = (
            feature_frame.iloc[:, focal_index]
            if focal_values is None
            else focal_values
        )
        levels, focal_category_means, focal_category_order_source = (
            _resolve_category_levels(
                category_source,
                phi[:, focal_index],
                category_order_mode,
                explicit=category_order,
                severity_direction=severity_direction,
            )
        )
        styles = _styles_for_levels(
            levels,
            category_palette,
            category_markers,
            default_palettes=category_palettes,
        )
        has_missing_category = any(
            _category_value_is_missing(value) for value in values
        )
    else:
        raw_values = pd.Series(np.asarray(values, dtype=object), dtype=object)
        numeric_series = pd.to_numeric(raw_values, errors="coerce")
        conversion_failed = ~raw_values.isna() & numeric_series.isna()
        if bool(conversion_failed.any()):
            examples = raw_values.loc[conversion_failed].astype(str).unique()[:3]
            raise ValueError(
                "continuous focal values contain non-numeric values: "
                + ", ".join(map(repr, examples))
            )
        numeric_values = numeric_series.to_numpy(float)
        if np.any(np.isinf(numeric_values)):
            raise ValueError("continuous focal values must be finite or missing")
        finite = numeric_values[np.isfinite(numeric_values)]
        if len(finite) < 2 or np.unique(finite).size < 2:
            raise ValueError("continuous focal values require at least two distinct finite values")
        low, high = np.quantile(finite, [0.05, 0.95])
        normalization_quantiles: list[float] | None = [0.05, 0.95]
        if high <= low:
            low, high = float(np.min(finite)), float(np.max(finite))
            normalization_quantiles = None
        normalizer = mpl.colors.Normalize(vmin=float(low), vmax=float(high), clip=True)
        continuous_color_scale = {
            "cmap": cmap.name,
            "vmin": float(low),
            "vmax": float(high),
            "observed_min": float(np.min(finite)),
            "observed_max": float(np.max(finite)),
            "normalization_quantiles": normalization_quantiles,
            "n_clipped_below": int(np.sum(finite < low)),
            "n_clipped_above": int(np.sum(finite > high)),
            "endpoint_colors_include_outside_values": True,
        }
        values = numeric_values

    missing_mask = (
        np.fromiter(
            (_category_value_is_missing(value) for value in values),
            dtype=bool,
            count=len(values),
        )
        if categorical
        else ~np.isfinite(np.asarray(values, dtype=float))
    )
    missing_count = int(np.sum(missing_mask))
    missing_display_label = (
        _missing_category_label(tuple(styles) if styles is not None else ())
        if missing_count
        else None
    )
    global_feature_frame = feature_frame.copy()
    global_focal_values = pd.Series(
        np.asarray(values, dtype=object if categorical else float),
        index=feature_frame.index,
        dtype=object if categorical else float,
    )
    global_feature_frame.isetitem(focal_index, global_focal_values)

    shown = _resolve_display_names(display_names, names)
    shown_groups, displayed_group_order = _resolve_group_labels(group_labels, order)
    focal_label = shown.get(focal_name, focal_name)
    local_label = local_x_label or f"{focal_label} SHAP value"
    value_label = focal_value_label or f"{focal_label} value"
    focal_shap = phi[:, focal_index]
    limit = _nice_limit(focal_shap)
    wrapped = len(order) >= 5
    displayed_rows = min(max_global_features, len(names))
    preview_order = np.argsort(-np.mean(np.abs(phi), axis=0), kind="stable")[
        :displayed_rows
    ]
    preview_categorical = [
        names[int(index)]
        for index in preview_order
        if _is_categorical(global_feature_frame.iloc[:, int(index)])
    ]
    preview_dock_categorical = [
        name
        for name in preview_categorical
        if not (categorical and name == focal_name)
    ]
    dock_key_rows = len(preview_dock_categorical)
    visual_title = None if title is None else _wrapped_label(title.strip(), width=60)
    two_line_title = bool(visual_title and "\n" in visual_title)
    if visual_title is None:
        plot_top = 0.875
        header_y = 0.925
        note_y = 0.893
    else:
        plot_top = 0.735 if two_line_title else 0.785
        header_y = 0.835 if two_line_title else 0.884
        note_y = 0.792 if two_line_title else 0.837
    category_count = 0 if styles is None else len(styles) + int(has_missing_category)
    if category_count:
        if category_legend_columns is not None:
            legend_columns = min(int(category_legend_columns), category_count)
        else:
            longest_category = max(len(style.label) for style in styles.values())
            legend_columns = min(3 if longest_category > 12 else 4, category_count)
    else:
        legend_columns = 0
    dock_inches = 0.0 if not dock_key_rows else 0.20 + 0.29 * dock_key_rows
    height = max(
        5.25,
        2.25 + 0.45 * displayed_rows + dock_inches,
        6.15 if wrapped else 0.0,
    )
    dock_height = dock_inches / height if dock_key_rows else 0.0
    dock_bottom = 0.060
    plot_bottom = (
        dock_bottom + dock_height + 0.115 if dock_key_rows else 0.145
    )

    with mpl.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": TEXT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    ):
        figure = plt.figure(figsize=(7.35, height), facecolor="white")
        outer = figure.add_gridspec(
            1, 2, width_ratios=[0.29, 0.71], left=0.170, right=0.925,
            top=plot_top, bottom=plot_bottom, wspace=0.50
        )
        global_axis = figure.add_subplot(outer[0, 0])
        if wrapped:
            local_grid = outer[0, 1].subgridspec(3, 2, hspace=0.56, wspace=0.30)
            local_axes = np.asarray(
                [figure.add_subplot(local_grid[row, column]) for row in range(3) for column in range(2)],
                dtype=object,
            ).reshape(3, 2)
        else:
            local_axes = np.asarray([[figure.add_subplot(outer[0, 1])]], dtype=object)

        global_render = _draw_global(
            global_axis,
            phi,
            global_feature_frame,
            names,
            shown,
            focal_name=focal_name,
            max_display=max_global_features,
            sample_limit=max_points,
            random_state=random_state,
            x_label=global_x_label,
            cmap=cmap,
            category_palettes=category_palettes,
            category_palette=shared_global_category_palette,
            category_order_mode=category_order_mode,
            severity_direction=severity_direction,
            focal_category_levels=(
                tuple(styles)
                if categorical and styles is not None
                else None
            ),
            focal_category_means=(
                focal_category_means if categorical else None
            ),
            focal_category_order_source=(
                focal_category_order_source if categorical else None
            ),
            focal_styles=(
                styles if categorical else None
            ),
        )
        displayed_has_numeric = any(
            not _is_categorical(global_feature_frame.iloc[:, int(index)])
            for index in global_render.order[
                : len(global_render.displayed_features)
            ]
        )
        continuous_key_axis: plt.Axes | None = None
        if wrapped:
            means, local_rendered = _draw_wrapped(
                local_axes, focal_shap, np.asarray(values), labels, order, shown_groups,
                limit=limit, normalizer=normalizer, styles=styles, cmap=cmap, x_label=local_label,
                max_points=max_points, random_state=random_state, show_mean=show_mean
            )
        else:
            means, local_rendered = _draw_stacked(
                local_axes[0, 0], focal_shap, np.asarray(values), labels, order, shown_groups,
                limit=limit, normalizer=normalizer, styles=styles, cmap=cmap, x_label=local_label,
                max_points=max_points, random_state=random_state, show_mean=show_mean
            )

        dock_keys = [
            key
            for key in global_render.keys
            if not (styles is not None and key.feature == focal_name)
        ]
        dock_axis: plt.Axes | None = None
        dock_layout: _DockLayout | None = None
        if dock_keys:
            measurement_axis = figure.add_axes(
                [0.04, 0.01, 0.92, 0.025]
            )
            measured_layout, measured_entries = _measure_dock(
                measurement_axis,
                dock_keys,
            )
            measurement_axis.remove()
            packed_rows = len(measured_layout.rows)
            if packed_rows != dock_key_rows:
                dock_key_rows = packed_rows
                dock_inches = 0.20 + 0.29 * dock_key_rows
                height = max(
                    5.25,
                    2.25 + 0.45 * displayed_rows + dock_inches,
                    6.15 if wrapped else 0.0,
                )
                dock_height = dock_inches / height
                plot_bottom = dock_bottom + dock_height + 0.115
                figure.set_size_inches(7.35, height, forward=True)
                outer.update(bottom=plot_bottom)
                figure.canvas.draw()
            dock_axis = figure.add_axes(
                [0.04, dock_bottom, 0.92, max(dock_height - 0.012, 0.025)]
            )
            dock_layout = _draw_bottom_dock(
                dock_axis,
                dock_keys,
                layout=measured_layout,
                all_entries=measured_entries,
            )
            figure.text(
                0.04,
                dock_bottom + dock_height + 0.005,
                "CATEGORY KEYS",
                ha="left",
                va="bottom",
                fontsize=6.55,
                weight="bold",
                color=MUTED,
            )
            order_note = _category_order_note(
                [key.order_source for key in dock_keys],
                severity_direction,
                color_scheme,
                has_pooled_others=any(
                    key.collapsed_levels for key in dock_keys
                ),
            )
            if order_note is not None:
                figure.text(
                    0.925,
                    dock_bottom + dock_height + 0.005,
                    order_note,
                    ha="right",
                    va="bottom",
                    fontsize=5.75,
                    color=MUTED,
                )

        if displayed_has_numeric:
            continuous_key_axis = _add_continuous_key(
                figure,
                global_axis,
                cmap=cmap,
            )

        if visual_title is not None:
            figure.text(0.04, 0.965, visual_title, ha="left", va="top", fontsize=12.0, weight="bold", color=TEXT)
        figure.text(0.04, header_y, "(a)  GLOBAL SHAP SUMMARY", ha="left", va="center", fontsize=7.15, weight="bold", color=MUTED)
        focal_heading = _wrapped_label(focal_label.upper(), width=31, lines=1)
        focal_header = figure.text(
            0.455,
            header_y,
            f"(b)  {focal_heading} BY SUBGROUP",
            ha="left",
            va="center",
            fontsize=7.15,
            weight="bold",
            color=MUTED,
        )
        note_artist: mpl.text.Text | None = None
        if note:
            note_artist = figure.text(
                0.955,
                note_y,
                _wrapped_label(note.strip(), width=60),
                ha="right",
                va="center",
                fontsize=6.7,
                color=MUTED,
            )
        figure.canvas.draw()
        global_box = global_axis.get_position()
        local_box = local_axes.flat[0].get_position()
        renderer = figure.canvas.get_renderer()
        canvas_width, canvas_height = figure.canvas.get_width_height()
        header_floor = focal_header.get_window_extent(renderer=renderer).y0
        if note_artist is not None:
            header_floor = min(
                header_floor,
                note_artist.get_window_extent(renderer=renderer).y0,
            )
        header_padding_px = 5.0
        header_clearance_px = 5.0

        focal_legend: mpl.legend.Legend | None = None
        gradient_axis: plt.Axes | None = None
        if styles is None:
            if normalizer is None:
                raise RuntimeError("continuous legend requires a normalizer")
            gradient_axis = figure.add_axes(
                [local_box.x0, plot_top + 0.007, 0.115, 0.011]
            )
            gradient_axis.imshow(np.linspace(0, 1, 160)[None, :], aspect="auto", cmap=cmap)
            scale_is_quantile = bool(
                continuous_color_scale
                and continuous_color_scale["normalization_quantiles"] is not None
            )
            low_prefix = "≤" if scale_is_quantile else ""
            high_prefix = "≥" if scale_is_quantile else ""
            gradient_axis.set_xticks(
                [0, 159],
                [
                    low_prefix + _format_value(float(normalizer.vmin)),
                    high_prefix + _format_value(float(normalizer.vmax)),
                ],
                fontsize=5.9,
            )
            gradient_axis.set_yticks([])
            scale_suffix = " (5th–95th pct.)" if scale_is_quantile else " (range)"
            gradient_axis.set_title(
                value_label + scale_suffix, fontsize=6.3, pad=2, color=MUTED
            )
            for spine in gradient_axis.spines.values():
                spine.set_visible(False)
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            gradient_box = gradient_axis.get_tightbbox(renderer)
            header_overflow = (
                gradient_box.y1 - (header_floor - header_padding_px)
            )
            if header_overflow > 0:
                gradient_position = gradient_axis.get_position()
                gradient_axis.set_position(
                    [
                        gradient_position.x0,
                        gradient_position.y0 - header_overflow / canvas_height,
                        gradient_position.width,
                        gradient_position.height,
                    ]
                )
                figure.canvas.draw()
            for _attempt in range(2):
                renderer = figure.canvas.get_renderer()
                gradient_box = gradient_axis.get_tightbbox(renderer)
                local_artist_top = _top_row_artist_top(local_axes, renderer)
                overlap = (
                    local_artist_top
                    + header_clearance_px
                    - gradient_box.y0
                )
                if overlap <= 0:
                    break
                plot_top -= overlap / canvas_height
                outer.update(top=plot_top)
                figure.canvas.draw()
            global_box = global_axis.get_position()
            local_box = local_axes.flat[0].get_position()
            if missing_count:
                missing_handle = Line2D(
                    [0], [0], linestyle="none", marker="X",
                    markerfacecolor=MISSING, markeredgecolor="none",
                    markersize=4.8,
                    label=f"{missing_display_label} (n={missing_count:,})",
                )
                figure.legend(
                    handles=[missing_handle], loc="lower left",
                    bbox_to_anchor=(local_box.x0, plot_top + 0.004), frameon=False,
                    fontsize=6.1, handlelength=0.7, handletextpad=0.30,
                )
            category_labels: list[str] = []
        else:
            handles = [
                Line2D([0], [0], linestyle="none", marker=style.marker, markerfacecolor=style.color,
                       markeredgecolor="none", markersize=4.8,
                       label=_wrapped_label(style.label, width=22))
                for style in styles.values()
            ]
            if has_missing_category:
                handles.append(
                    Line2D([0], [0], linestyle="none", marker="X", markerfacecolor=MISSING,
                           markeredgecolor="none", markersize=4.8,
                           label=missing_display_label)
                )
            legend_ceiling = (header_floor - header_padding_px) / canvas_height
            focal_legend = figure.legend(
                handles=handles,
                loc="upper left",
                bbox_to_anchor=(local_box.x0, legend_ceiling),
                bbox_transform=figure.transFigure,
                borderaxespad=0,
                ncol=legend_columns, frameon=False, fontsize=6.15,
                handlelength=0.7, handletextpad=0.30, columnspacing=0.72,
                title=_wrapped_label(value_label, width=38, lines=2),
                title_fontsize=6.15,
            )
            figure.canvas.draw()
            for _attempt in range(2):
                renderer = figure.canvas.get_renderer()
                legend_box = focal_legend.get_window_extent(renderer=renderer)
                local_artist_top = _top_row_artist_top(local_axes, renderer)
                overlap = (
                    local_artist_top
                    + header_clearance_px
                    - legend_box.y0
                )
                if overlap <= 0:
                    break
                plot_top -= overlap / canvas_height
                outer.update(top=plot_top)
                figure.canvas.draw()
            global_box = global_axis.get_position()
            local_box = local_axes.flat[0].get_position()
            category_labels = [style.label for style in styles.values()]

        mean_legend: mpl.legend.Legend | None = None
        if show_mean:
            mean_key_y = header_y
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            if focal_legend is not None:
                focal_title_box = focal_legend.get_title().get_window_extent(
                    renderer=renderer
                )
                mean_key_y = (
                    (focal_title_box.y0 + focal_title_box.y1)
                    / 2
                    / canvas_height
                )
            elif gradient_axis is not None:
                focal_title_box = gradient_axis.title.get_window_extent(
                    renderer=renderer
                )
                mean_key_y = (
                    (focal_title_box.y0 + focal_title_box.y1)
                    / 2
                    / canvas_height
                )
            mean_legend = figure.legend(
                handles=[
                    Line2D(
                        [0],
                        [0],
                        linestyle="none",
                        marker="D",
                        markerfacecolor="white",
                        markeredgecolor=MEAN,
                        markeredgewidth=1.0,
                        markersize=4.7,
                        label="Subgroup mean",
                    )
                ],
                loc="center right",
                bbox_to_anchor=(0.925, mean_key_y),
                bbox_transform=figure.transFigure,
                borderaxespad=0,
                frameon=False,
                fontsize=6.15,
                handlelength=0.75,
                handletextpad=0.35,
            )
            if focal_legend is not None:
                figure.canvas.draw()
                renderer = figure.canvas.get_renderer()
                focal_title_box = focal_legend.get_title().get_window_extent(
                    renderer=renderer
                )
                mean_box = mean_legend.get_window_extent(renderer=renderer)
                clearance_px = 4.0 * figure.dpi / 72.0
                if focal_title_box.x1 + clearance_px > mean_box.x0:
                    focal_legend.get_title().set_fontsize(5.95)
                    for label in mean_legend.get_texts():
                        label.set_fontsize(6.0)
                    figure.canvas.draw()
                    renderer = figure.canvas.get_renderer()
                    focal_title_box = focal_legend.get_title().get_window_extent(
                        renderer=renderer
                    )
                    mean_key_y = (
                        (focal_title_box.y0 + focal_title_box.y1)
                        / 2
                        / canvas_height
                    )
                    mean_legend.set_bbox_to_anchor(
                        (0.925, mean_key_y),
                        transform=figure.transFigure,
                    )
                    figure.canvas.draw()
                    renderer = figure.canvas.get_renderer()
                    focal_title_box = focal_legend.get_title().get_window_extent(
                        renderer=renderer
                    )
                    mean_box = mean_legend.get_window_extent(renderer=renderer)
                overlap = focal_title_box.x1 + clearance_px - mean_box.x0
                if overlap > 0:
                    mean_anchor_x = min(
                        1.0,
                        0.925 + overlap / canvas_width,
                    )
                    mean_legend.set_bbox_to_anchor(
                        (mean_anchor_x, mean_key_y),
                        transform=figure.transFigure,
                    )
                    figure.canvas.draw()

        if continuous_key_axis is not None:
            global_box = global_axis.get_position()
            continuous_key_axis.set_position(
                [
                    global_box.x1 + 0.007,
                    global_box.y0,
                    0.010,
                    global_box.height,
                ]
            )
            figure.canvas.draw()
        global_box = global_axis.get_position()
        local_box = local_axes.flat[0].get_position()
        divider = (global_box.x1 + local_box.x0) / 2 - 0.008
        figure.add_artist(
            Line2D(
                [divider, divider],
                [plot_bottom - 0.025, plot_top + 0.015],
                transform=figure.transFigure,
                color="#DDE3EA",
                linewidth=0.85,
                zorder=0,
            )
        )
        output_files: list[str] = []
        if output is not None:
            output_files = _save_outputs(figure, output, dpi)

    global_capped = len(phi) > max_points
    local_capped = any(
        int(np.sum(_group_mask(labels, group))) > max_points for group in order
    )
    category_colors = (
        {}
        if styles is None
        else {style.label: mpl.colors.to_hex(style.color) for style in styles.values()}
    )
    resolved_markers = (
        {}
        if styles is None
        else {style.label: style.marker for style in styles.values()}
    )
    if has_missing_category:
        if missing_display_label is None:
            raise RuntimeError("categorical missing values require a display label")
        category_colors[missing_display_label] = MISSING
        resolved_markers[missing_display_label] = "X"
    global_category_keys = {
        key.feature: _feature_key_summary(key) for key in global_render.keys
    }
    focal_levels = () if styles is None else tuple(styles)
    focal_level_row_counts = [
        int(np.sum(_category_mask(values, level))) for level in focal_levels
    ]
    focal_pooled_index = next(
        (
            index
            for index, level in enumerate(focal_levels)
            if isinstance(level, _OtherCategories)
        ),
        None,
    )
    focal_pooled = (
        None
        if focal_pooled_index is None
        else focal_levels[focal_pooled_index]
    )
    focal_category_collapse = (
        None
        if not isinstance(focal_pooled, _OtherCategories)
        else {
            "original_level_count": (
                len(focal_levels) - 1 + len(focal_pooled.levels)
            ),
            "retained_levels": [
                str(level)
                for level in focal_levels
                if not isinstance(level, _OtherCategories)
            ],
            "retained_level_row_counts": [
                count
                for level, count in zip(
                    focal_levels,
                    focal_level_row_counts,
                    strict=True,
                )
                if not isinstance(level, _OtherCategories)
            ],
            "collapsed_levels": [str(level) for level in focal_pooled.levels],
            "collapsed_row_count": focal_level_row_counts[focal_pooled_index],
            "high_cardinality_rule": _HIGH_CARDINALITY_RULE,
        }
    )
    category_dock_layout_summary = _summarize_dock_layout(
        dock_layout,
        dock_keys,
    )
    summary: dict[str, object] = {
        "summary_schema_version": "2.0",
        "package_version": __version__,
        "software_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": mpl.__version__,
        },
        "view": "prism",
        "color_scheme": color_scheme,
        "resolved_continuous_cmap": cmap.name,
        "continuous_color_endpoints": [
            mpl.colors.to_hex(cmap(0.0)),
            mpl.colors.to_hex(cmap(1.0)),
        ],
        "focal_feature": focal_name,
        "focal_kind": "categorical" if categorical else "continuous",
        "n_rows": int(len(phi)),
        "n_features": int(len(names)),
        "n_groups": int(len(order)),
        "group_order": displayed_group_order,
        "group_values": [str(group) for group in order],
        "group_identities": [
            {
                "value_repr": repr(group),
                "value_type": f"{type(group).__module__}.{type(group).__qualname__}",
                "display_label": displayed,
            }
            for group, displayed in zip(order, displayed_group_order, strict=True)
        ],
        "group_counts": {
            displayed: int(np.sum(_group_mask(labels, group)))
            for group, displayed in zip(order, displayed_group_order, strict=True)
        },
        "group_mean_shap": means,
        "focal_missing_count": missing_count,
        "missing_focal_label": missing_display_label,
        "focal_missing_by_group": {
            displayed: int(np.sum(missing_mask & _group_mask(labels, group)))
            for group, displayed in zip(order, displayed_group_order, strict=True)
        },
        "missing_focal_encoding": (
            {"color": MISSING, "marker": "X"} if missing_count else None
        ),
        "global_feature_order": [names[index] for index in global_render.order],
        "displayed_global_features": list(global_render.displayed_features),
        "global_rendered_rows": global_render.rendered_rows,
        "global_shap_xlim": list(global_render.xlim),
        "global_category_keys": global_category_keys,
        "unencoded_high_cardinality_features": [],
        "collapsed_high_cardinality_features": list(
            global_render.collapsed_categorical_features
        ),
        "category_order_mode": category_order_mode,
        "severity_direction": severity_direction,
        "category_order_source": focal_category_order_source,
        "resolved_category_order": (
            [] if styles is None else [style.label for style in styles.values()]
        ),
        "resolved_category_row_counts": focal_level_row_counts,
        "focal_category_collapse": focal_category_collapse,
        "focal_category_mean_signed_shap": (
            {}
            if styles is None
            else {
                style.label: float(focal_category_means[level])
                for level, style in styles.items()
            }
        ),
        "category_order_sources": {
            key.feature: key.order_source for key in global_render.keys
        },
        "category_dock_features": [key.feature for key in dock_keys],
        "category_dock_layout": category_dock_layout_summary,
        "focal_category_legend_locations": (
            ["subgroup_header"] if styles is not None else []
        ),
        "local_rendered_rows": local_rendered,
        "global_sampling": "deterministic random sample" if global_capped else "all rows",
        "local_sampling": (
            "capped independently within each group" if local_capped else "all rows"
        ),
        "sampling_note": (
            f"Display points were capped at {max_points:,}; global ranking and axis limits, "
            "subgroup axis limits, counts, and means use all aligned rows."
            if global_capped or local_capped
            else None
        ),
        "shared_local_shap_limit": float(limit),
        "layout": "wrapped" if wrapped else "stacked",
        "wrapped_layout": wrapped,
        "vertical_position_semantics": (
            "deterministic jitter for separation; not a density scale"
            if wrapped
            else "collision-avoidance packing within each subgroup row"
        ),
        "category_labels": category_labels,
        "category_colors": category_colors,
        "category_markers": [] if styles is None else [style.marker for style in styles.values()],
        "resolved_category_markers": resolved_markers,
        "continuous_color_scale": continuous_color_scale,
        "input_alignment": input_alignment,
        "show_mean": bool(show_mean),
        "display_parameters": {
            "max_global_features": max_global_features,
            "max_points": max_points,
            "random_state": random_state,
            "dpi": dpi,
            "title": title,
            "note": note,
            "global_x_label": global_x_label,
            "local_x_label": local_label,
            "focal_value_label": value_label,
            "category_order_mode": category_order_mode,
            "severity_direction": severity_direction,
            "category_legend_columns": legend_columns or None,
            "color_scheme": color_scheme,
            "custom_global_category_palette": (
                None
                if shared_global_category_palette is None
                else [
                    mpl.colors.to_hex(color)
                    for color in shared_global_category_palette
                ]
            ),
            "continuous_cmap": cmap.name,
            "figure_size_inches": [float(value) for value in figure.get_size_inches()],
        },
        "rasterized_points": True,
        "pdf_rendering": "hybrid: rasterized point clouds with vector text and axes",
        "output_files": output_files,
    }
    visible_local_axes = tuple(axis for axis in local_axes.flat if axis.get_visible())
    return PrismResult(
        figure=figure,
        axes={
            "global": global_axis,
            "subgroups": visible_local_axes,
            "category_dock": dock_axis,
            "continuous_key": continuous_key_axis,
            "focal_value_key": gradient_axis,
            "focal_category_legend": focal_legend,
        },
        summary=summary,
    )
