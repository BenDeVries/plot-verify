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

from .colors import FALLBACK_HEX, is_valid_hex


REQUIRED_COLUMNS = ["series", "x", "y"]
OPTIONAL_ERROR_COLUMNS = ["y_err_lower", "y_err_upper"]


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

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        report.error = f"CSV is missing required columns: {missing}"
        return None, report

    has_series_color = "series_color" in df.columns
    report.has_series_color_column = has_series_color
    if not has_series_color:
        df["series_color"] = pd.NA

    for col in OPTIONAL_ERROR_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["y_err_lower"] = pd.to_numeric(df["y_err_lower"], errors="coerce")
    df["y_err_upper"] = pd.to_numeric(df["y_err_upper"], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=["x", "y"]).reset_index(drop=True)
    report.n_rows_dropped_missing_xy = n_before - len(df)
    if report.n_rows_dropped_missing_xy > 0:
        report.warnings.append(
            f"Dropped {report.n_rows_dropped_missing_xy} row(s) with missing x or y."
        )

    df["series"] = df["series"].astype(str)

    if has_series_color:
        invalid = ~df["series_color"].apply(is_valid_hex)
        report.n_invalid_series_colors = int(invalid.sum())
        if report.n_invalid_series_colors > 0:
            report.warnings.append(
                f"{report.n_invalid_series_colors} row(s) have invalid "
                f"series_color values; using {FALLBACK_HEX} as a fallback."
            )
            df.loc[invalid, "series_color"] = FALLBACK_HEX

    # Reversed error bars: convention is lower ≤ y ≤ upper.
    eu = df["y_err_upper"]
    el = df["y_err_lower"]
    y = df["y"]
    finite_both = eu.notna() & el.notna() & y.notna()
    reversed_mask = finite_both & ((el > y) | (eu < y))
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
