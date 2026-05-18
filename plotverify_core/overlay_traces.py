"""Build UI-agnostic overlay trace records from a CSV DataFrame.

`build_overlay_traces` is pure: it consumes the data + visibility/colors and
returns a list of `OverlayTrace` records. The Plotly-specific renderer in
``app_auto_axis.py`` builds Plotly figures from the same record list, and a
future Shiny renderer can do the same with shinywidgets / custom JS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .colors import FALLBACK_HEX, hex_complement, is_valid_hex


@dataclass
class OverlayTrace:
    """One series-level trace with optional error bars and ribbon coordinates."""
    series: str
    x: np.ndarray
    y: np.ndarray
    # `has_err` is True where both lower and upper are finite for that index.
    has_err: np.ndarray
    err_array_plus: np.ndarray         # upper - y, zero where !has_err
    err_array_minus: np.ndarray        # y - lower, zero where !has_err
    # Ribbon coordinates (sorted by x, only rows where has_err); empty if none.
    ribbon_x: np.ndarray
    ribbon_y_upper: np.ndarray
    ribbon_y_lower: np.ndarray
    color_hex: str
    overlay_color_hex: str
    visible: bool


def build_overlay_traces(
    df: pd.DataFrame,
    *,
    series_visibility: Optional[Dict[str, bool]] = None,
    series_colors: Optional[Dict[str, str]] = None,
) -> List[OverlayTrace]:
    """Build traces for every distinct series in ``df``.

    ``series_visibility`` defaults to True for every series; ``series_colors``
    overrides the per-series CSV color. Series with no rows are skipped.
    """
    series_visibility = series_visibility or {}
    series_colors = series_colors or {}

    traces: List[OverlayTrace] = []
    for series_name in df["series"].drop_duplicates().tolist():
        sdf = df[df["series"] == series_name]
        if not len(sdf):
            continue

        # Color resolution: explicit override → CSV column → fallback.
        color_hex = series_colors.get(series_name)
        if not color_hex:
            color_hex = sdf["series_color"].iloc[0] if "series_color" in sdf.columns else None
        if not is_valid_hex(color_hex):
            color_hex = FALLBACK_HEX
        overlay_hex = hex_complement(color_hex)

        x = sdf["x"].to_numpy(dtype=float)
        y = sdf["y"].to_numpy(dtype=float)
        eu = sdf["y_err_upper"].to_numpy(dtype=float) if "y_err_upper" in sdf.columns else np.full(len(sdf), np.nan)
        el = sdf["y_err_lower"].to_numpy(dtype=float) if "y_err_lower" in sdf.columns else np.full(len(sdf), np.nan)
        has_err = np.isfinite(eu) & np.isfinite(el)

        err_plus = np.where(has_err, eu - y, 0.0)
        err_minus = np.where(has_err, y - el, 0.0)

        if has_err.any():
            x_rib = x[has_err]
            y_upper = eu[has_err]
            y_lower = el[has_err]
            order = np.argsort(x_rib)
            x_rib = x_rib[order]
            y_upper = y_upper[order]
            y_lower = y_lower[order]
        else:
            x_rib = np.array([], dtype=float)
            y_upper = np.array([], dtype=float)
            y_lower = np.array([], dtype=float)

        traces.append(OverlayTrace(
            series=str(series_name),
            x=x, y=y,
            has_err=has_err,
            err_array_plus=err_plus,
            err_array_minus=err_minus,
            ribbon_x=x_rib,
            ribbon_y_upper=y_upper,
            ribbon_y_lower=y_lower,
            color_hex=color_hex,
            overlay_color_hex=overlay_hex,
            visible=bool(series_visibility.get(series_name, True)),
        ))

    return traces
