"""CSV loading with validation, error-bar normalization, and audit reports.

This module is pure: it returns a (DataFrame, LoadReport) pair. UI code
translates the report into user-facing warnings/errors (st.warning / shiny
ui.notification / etc.).
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd

from .colors import FALLBACK_HEX, assign_palette_colors, is_valid_hex


REQUIRED_COLUMNS = ["x", "y"]
OPTIONAL_ERROR_COLUMNS = ["y_err_lower", "y_err_upper"]

# A forest-plot CSV describes one horizontal row per series: a point estimate
# (`value`) with a horizontal confidence interval (`value_err_*`) and a
# categorical vertical axis. We normalise it onto the canonical x/y/y_err
# columns so the overlay model, editor, export and dashboard keep working —
# in forest mode `x` holds the value and `y_err_*` bracket `x` (see load_csv).
FOREST_COLUMN_MAP = {
    "value": "x",
    "value_err_lower": "y_err_lower",
    "value_err_upper": "y_err_upper",
}


@dataclass
class LoadReport:
    """Outcome of `load_csv`. UI layer translates messages into user warnings."""
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    n_rows_dropped_missing_xy: int = 0
    n_invalid_series_colors: int = 0
    n_reversed_error_bars: int = 0
    reversed_samples: List[Tuple[str, float]] = field(default_factory=list)
    has_series_color_column: bool = False
    error_bar_type: Optional[str] = None
    is_forest: bool = False


def _looks_like_forest(df: pd.DataFrame) -> bool:
    """A forest CSV carries `value` (the estimate) and no explicit `x` column."""
    return "value" in df.columns and "x" not in df.columns


def _coerce_bool(series: pd.Series) -> pd.Series:
    """Coerce a CSV column of "true"/"false"-ish strings into real booleans."""
    return series.map(
        lambda v: str(v).strip().lower() in ("true", "1", "yes", "t")
    )


def load_csv(csv_source: str) -> Tuple[Optional[pd.DataFrame], LoadReport]:
    """Parse and validate CSV text.

    Returns (df, report). When ``df is None``, ``report.error`` describes why.

    Behaviour matches the legacy `_load_csv` in `app_auto_axis.py`:
    - Required columns: series, x, y
    - Optional error-bar columns: y_err_lower, y_err_upper (filled with NaN
      when absent so downstream `np.isfinite` checks just work).
    - Reversed error bars (lower > y or upper < y) are auto-swapped and
      counted.
    - Invalid series_color hex values are replaced with the fallback grey.
    """
    report = LoadReport()
    try:
        df = pd.read_csv(io.StringIO(csv_source))
    except Exception as e:
        report.error = f"Failed to parse CSV: {e}"
        return None, report

    is_forest = _looks_like_forest(df)
    report.is_forest = is_forest
    if is_forest:
        df = df.rename(columns=FOREST_COLUMN_MAP)

    # Forest CSVs have no `y` column — the vertical axis is a categorical row
    # index synthesised below, so only `x` (the value) is required from disk.
    required = ["x"] if is_forest else REQUIRED_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        report.error = f"CSV is missing required columns: {missing}"
        return None, report

    if "series" not in df.columns:
        df["series"] = "Data"

    has_series_color = "series_color" in df.columns
    report.has_series_color_column = has_series_color
    if not has_series_color:
        # Assign cycling palette colors per unique series so downstream renderers
        # never see pd.NA. `has_series_color_column` stays False so the UI knows
        # the user did not pick these — masking gates on intentional colors only.
        palette = dict(assign_palette_colors(df["series"].astype(str).tolist()))
        df["series_color"] = df["series"].astype(str).map(palette)

    for col in OPTIONAL_ERROR_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y_err_lower"] = pd.to_numeric(df["y_err_lower"], errors="coerce")
    df["y_err_upper"] = pd.to_numeric(df["y_err_upper"], errors="coerce")

    if is_forest:
        # Drop rows with no value, then place each remaining row on the
        # categorical vertical axis: `y` is a row index assigned top→bottom in
        # CSV order (first CSV row sits at the top with the largest index). The
        # even spacing is realised later by the linear y-calibration.
        n_before = len(df)
        df = df.dropna(subset=["x"]).reset_index(drop=True)
        report.n_rows_dropped_missing_xy = n_before - len(df)
        n = len(df)
        df["y"] = [float(n - 1 - i) for i in range(n)]
        if "is_summary" in df.columns:
            df["is_summary"] = _coerce_bool(df["is_summary"])
        else:
            df["is_summary"] = False
        if "status" in df.columns:
            df["status"] = df["status"].fillna("").astype(str)
        else:
            df["status"] = ""
    else:
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        n_before = len(df)
        df = df.dropna(subset=["x", "y"]).reset_index(drop=True)
        report.n_rows_dropped_missing_xy = n_before - len(df)

    if report.n_rows_dropped_missing_xy > 0:
        report.warnings.append(
            f"Dropped {report.n_rows_dropped_missing_xy} row(s) with missing "
            + ("value." if is_forest else "x or y.")
        )

    df["series"] = df["series"].astype(str)

    if has_series_color:
        # Force object dtype: an all-NaN column read from a re-saved CSV comes
        # back as float64, which then rejects the string fallback below.
        df["series_color"] = df["series_color"].astype(object)
        invalid = ~df["series_color"].apply(is_valid_hex)
        report.n_invalid_series_colors = int(invalid.sum())
        if report.n_invalid_series_colors > 0:
            report.warnings.append(
                f"{report.n_invalid_series_colors} row(s) have invalid "
                f"series_color values; using {FALLBACK_HEX} as a fallback."
            )
            df.loc[invalid, "series_color"] = FALLBACK_HEX

    if "error_bar_type" in df.columns:
        non_null = df["error_bar_type"].dropna()
        if len(non_null) > 0:
            raw = str(non_null.iloc[0]).strip()
            report.error_bar_type = raw.title() if raw else None

    # Reversed error bars: the interval must bracket the point estimate. For a
    # forest plot the estimate is `x` (the value); otherwise it is `y`.
    eu = df["y_err_upper"]
    el = df["y_err_lower"]
    point = df["x"] if is_forest else df["y"]
    finite_both = eu.notna() & el.notna() & point.notna()
    reversed_mask = finite_both & ((el > point) | (eu < point))
    report.n_reversed_error_bars = int(reversed_mask.sum())
    if report.n_reversed_error_bars > 0:
        sample = df.loc[reversed_mask, ["series", "x"]].head(5)
        report.reversed_samples = [
            (str(r.series), float(r.x)) for r in sample.itertuples()
        ]
        report.warnings.append(
            f"{report.n_reversed_error_bars} row(s) have reversed error bars "
            "(y_err_lower > y or y_err_upper < y); auto-swapped."
        )
        lower_orig = df.loc[reversed_mask, "y_err_lower"].copy()
        df.loc[reversed_mask, "y_err_lower"] = df.loc[reversed_mask, "y_err_upper"]
        df.loc[reversed_mask, "y_err_upper"] = lower_orig

    return df, report
