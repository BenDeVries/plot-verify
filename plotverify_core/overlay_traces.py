"""Build UI-agnostic overlay trace records from a CSV DataFrame.

`build_overlay_traces` is pure: it consumes the data + visibility/colors and
returns a list of `OverlayTrace` records. The Plotly-specific renderer in
``app_auto_axis.py`` builds Plotly figures from the same record list, and a
future Shiny renderer can do the same with shinywidgets / custom JS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
    marker_color_hex: str
    visible: bool
    # Global point IDs matching EditableOverlay (e.g. "SeriesA#3"). Index is
    # the global DataFrame row position, not a series-local index.
    point_ids: List[str] = field(default_factory=list)
    # Forest-plot metadata (one entry per point). ``is_summary`` selects a
    # diamond marker; ``status`` is surfaced in the hover text. Both are None
    # for time-series / scatter traces.
    is_summary: Optional[np.ndarray] = None
    status: Optional[List[str]] = None


def build_overlay_traces(
    df: pd.DataFrame,
    *,
    series_visibility: Optional[Dict[str, bool]] = None,
    series_colors: Optional[Dict[str, str]] = None,
    plot_type: str = "time_series",
) -> List[OverlayTrace]:
    """Build traces for every distinct series in ``df``.

    ``series_visibility`` defaults to True for every series; ``series_colors``
    overrides the per-series CSV color. Series with no rows are skipped.

    In forest mode (``plot_type == "forest"``) the interval brackets the value
    axis ``x`` rather than ``y``, so the error offsets are measured from ``x``
    and no vertical ribbon is built; ``is_summary``/``status`` are carried
    through for the renderer.
    """
    series_visibility = series_visibility or {}
    series_colors = series_colors or {}
    is_forest = plot_type == "forest"

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
        marker_hex = hex_complement(color_hex)

        # sdf.index holds the global row positions in df (which is produced by
        # EditableOverlay.to_dataframe() and already has a 0-based reset index).
        # These match the indices used to build EditableOverlay pids.
        point_ids = [f"{series_name}#{i}" for i in sdf.index.tolist()]

        x = sdf["x"].to_numpy(dtype=float)
        y = sdf["y"].to_numpy(dtype=float)
        eu = sdf["y_err_upper"].to_numpy(dtype=float) if "y_err_upper" in sdf.columns else np.full(len(sdf), np.nan)
        el = sdf["y_err_lower"].to_numpy(dtype=float) if "y_err_lower" in sdf.columns else np.full(len(sdf), np.nan)
        has_err = np.isfinite(eu) & np.isfinite(el)

        # The interval brackets the value axis: `x` in forest mode, else `y`.
        base = x if is_forest else y
        err_plus = np.where(has_err, eu - base, 0.0)
        err_minus = np.where(has_err, base - el, 0.0)

        # A vertical ribbon fill only makes sense with a numeric y-axis.
        if has_err.any() and not is_forest:
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

        if is_forest:
            is_summary_arr = (
                sdf["is_summary"].to_numpy(dtype=bool)
                if "is_summary" in sdf.columns else np.zeros(len(sdf), dtype=bool)
            )
            status_arr = (
                [str(s) for s in sdf["status"].tolist()]
                if "status" in sdf.columns else ["" for _ in range(len(sdf))]
            )
        else:
            is_summary_arr = None
            status_arr = None

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
            marker_color_hex=marker_hex,
            visible=bool(series_visibility.get(series_name, True)),
            point_ids=point_ids,
            is_summary=is_summary_arr,
            status=status_arr,
        ))

    return traces
