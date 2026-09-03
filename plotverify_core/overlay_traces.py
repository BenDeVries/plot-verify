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


def is_horizontal_layout(plot_type: str, orientation: str = "vertical") -> bool:
    """True when the value axis runs along x (intervals bracket ``x``).

    Forest plots are inherently horizontal; bar and box plots are horizontal
    only when the file's ``orientation`` says so.
    """
    return plot_type == "forest" or (
        orientation == "horizontal" and plot_type in ("bar", "box")
    )


@dataclass
class OverlayTrace:
    """One series-level trace with optional error bars and ribbon coordinates."""
    series: str
    x: np.ndarray
    y: np.ndarray
    # `has_err` is True where at least one of lower/upper is finite (a
    # one-sided interval counts); `has_upper`/`has_lower` say which side.
    has_err: np.ndarray
    has_upper: np.ndarray
    has_lower: np.ndarray
    err_array_plus: np.ndarray         # upper - y, zero where !has_upper
    err_array_minus: np.ndarray        # y - lower, zero where !has_lower
    # Ribbon coordinates (sorted by x, only rows where has_err); empty if none.
    # A missing bound collapses to the point estimate, so a one-sided interval
    # draws a ribbon from the point out to the bound it does have.
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
    # Box-plot quartile arrays (one entry per point). None for non-box traces.
    box_q1: Optional[np.ndarray] = None
    box_median: Optional[np.ndarray] = None
    box_q3: Optional[np.ndarray] = None
    # Kaplan-Meier at-risk count per point. None for non-KM traces.
    at_risk: Optional[np.ndarray] = None


def build_overlay_traces(
    df: pd.DataFrame,
    *,
    series_visibility: Optional[Dict[str, bool]] = None,
    series_colors: Optional[Dict[str, str]] = None,
    plot_type: str = "time_series",
    orientation: str = "vertical",
) -> List[OverlayTrace]:
    """Build traces for every distinct series in ``df``.

    ``series_visibility`` defaults to True for every series; ``series_colors``
    overrides the per-series CSV color. Series with no rows are skipped.

    In horizontal layouts (forest, or bar/box with ``orientation ==
    "horizontal"``) the interval brackets the value axis ``x`` rather than
    ``y``, so the error offsets are measured from ``x`` and no vertical
    ribbon is built; ``is_summary``/``status`` are carried through for the
    renderer.
    """
    series_visibility = series_visibility or {}
    series_colors = series_colors or {}
    is_forest = plot_type == "forest"
    is_horiz = is_horizontal_layout(plot_type, orientation)
    is_bar = plot_type == "bar"
    is_box = plot_type == "box"
    is_km = plot_type == "kaplan_meier"

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
        has_upper = np.isfinite(eu)
        has_lower = np.isfinite(el)
        has_err = has_upper | has_lower

        # The interval brackets the value axis: `x` in horizontal layouts.
        base = x if is_horiz else y
        err_plus = np.where(has_upper, eu - base, 0.0)
        err_minus = np.where(has_lower, base - el, 0.0)

        # A vertical ribbon fill only makes sense for continuous y-axis types.
        if has_err.any() and not is_forest and not is_bar and not is_box:
            x_rib = x[has_err]
            y_upper = np.where(has_upper, eu, y)[has_err]
            y_lower = np.where(has_lower, el, y)[has_err]
            order = np.argsort(x_rib)
            x_rib = x_rib[order]
            y_upper = y_upper[order]
            y_lower = y_lower[order]
        else:
            x_rib = np.array([], dtype=float)
            y_upper = np.array([], dtype=float)
            y_lower = np.array([], dtype=float)

        if is_forest or is_box or is_km:
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

        box_q1_arr = None
        box_median_arr = None
        box_q3_arr = None
        if is_box:
            if "box_q1" in sdf.columns:
                box_q1_arr = sdf["box_q1"].to_numpy(dtype=float)
            if "box_median" in sdf.columns:
                box_median_arr = sdf["box_median"].to_numpy(dtype=float)
            if "box_q3" in sdf.columns:
                box_q3_arr = sdf["box_q3"].to_numpy(dtype=float)

        at_risk_arr = None
        if is_km and "at_risk" in sdf.columns:
            at_risk_arr = sdf["at_risk"].to_numpy(dtype=float)

        traces.append(OverlayTrace(
            series=str(series_name),
            x=x, y=y,
            has_err=has_err,
            has_upper=has_upper,
            has_lower=has_lower,
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
            box_q1=box_q1_arr,
            box_median=box_median_arr,
            box_q3=box_q3_arr,
            at_risk=at_risk_arr,
        ))

    return traces
