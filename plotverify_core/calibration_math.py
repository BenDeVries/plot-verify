"""Three-point manual calibration math and pixel↔data conversions.

The functions in this module are the linear / log10 calibration primitives
used by the Streamlit app's manual override workflow. For the multi-point
calibration produced by the OCR pipeline, see ``axis_pipeline.run_calibration``
and ``axis_pipeline.manual_calibration`` (typed API).

This module remains in plotverify_core because both Streamlit and Shiny need
the same dict-shaped `cal` for backward compatibility with stored sessions.
New code should prefer the typed `CalibrationResult` API directly.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


P1P2_Y_TOLERANCE_PX = 3.0


def log10_or_none(value, log_base):
    """Return log_base(value) when log_base is set and value>0, else value as float.

    Returns None when the requested log transform would be invalid (non-positive
    value), so callers can fail fast rather than producing NaN downstream.
    Supports any positive base > 1 (including math.e for natural log).
    """
    if log_base in (None, 0, 1.0):
        return float(value)
    v = float(value)
    if v <= 0:
        return None
    return float(math.log(v, log_base))


# Module-private alias retained for backwards compatibility with existing
# imports in app_auto_axis.py that referenced the underscore-prefixed name.
_log10_or_none = log10_or_none


def compute_calibration(p1_px_x, p1_px_y, p2_px_x, p2_px_y,
                        p3_px_x, p3_px_y,
                        p1_data_x, p2_data_x, p3_data_y, y_baseline,
                        x_log_base: Optional[float] = None,
                        y_log_base: Optional[float] = None):
    """Compute the pixel↔data transform from the three calibration points.

    Returns a dict with ``x_scale``, ``x_offset``, ``y_scale``, ``y_offset``,
    ``applied``, ``x_log_base``, ``y_log_base`` — or ``None`` if degenerate.

    When ``x_log_base`` / ``y_log_base`` is ``10.0``, the scale/offset are
    fitted in log10 space (so ``data = 10 ** (scale * pixel + offset)``). The
    caller's data values are still in the original linear space.

    P1 and P2 are expected to share the same pixel-y (the x-axis baseline). If
    they differ, the midpoint is used and ``p1p2_y_disagreement_px`` is set so
    the UI can surface a warning when the gap exceeds P1P2_Y_TOLERANCE_PX.
    """
    if abs(p2_px_x - p1_px_x) < 1e-9:
        return None

    p1x_t = log10_or_none(p1_data_x, x_log_base)
    p2x_t = log10_or_none(p2_data_x, x_log_base)
    if p1x_t is None or p2x_t is None:
        return None
    x_scale = (p2x_t - p1x_t) / (p2_px_x - p1_px_x)
    x_offset = p1x_t - x_scale * p1_px_x

    p1p2_y_disagreement = abs(p1_px_y - p2_px_y)
    x_axis_pixel_y = (p1_px_y + p2_px_y) / 2.0
    if abs(p3_px_y - x_axis_pixel_y) < 1e-9:
        return None

    p3y_t = log10_or_none(p3_data_y, y_log_base)
    y_base_t = log10_or_none(y_baseline, y_log_base)
    if p3y_t is None or y_base_t is None:
        return None
    y_scale = (p3y_t - y_base_t) / (p3_px_y - x_axis_pixel_y)
    y_offset = p3y_t - y_scale * p3_px_y

    if not (np.isfinite(x_scale) and np.isfinite(y_scale)):
        return None
    if abs(x_scale) < 1e-12 or abs(y_scale) < 1e-12:
        return None

    return {
        "x_scale": float(x_scale),
        "x_offset": float(x_offset),
        "y_scale": float(y_scale),
        "y_offset": float(y_offset),
        "x_log_base": float(x_log_base) if x_log_base else None,
        "y_log_base": float(y_log_base) if y_log_base else None,
        "applied": True,
        "p1p2_y_disagreement_px": float(p1p2_y_disagreement),
    }


def px_to_data(px_x, px_y, cal):
    """Convert pixel coordinates to data coordinates.

    Honors ``cal["x_log_base"]`` / ``cal["y_log_base"]`` when set.
    """
    x_lin = cal["x_scale"] * px_x + cal["x_offset"]
    y_lin = cal["y_scale"] * px_y + cal["y_offset"]
    x_log = cal.get("x_log_base")
    y_log = cal.get("y_log_base")
    x_val = (x_log ** x_lin) if x_log else x_lin
    y_val = (y_log ** y_lin) if y_log else y_lin
    return (x_val, y_val)


def data_to_px(data_x, data_y, cal):
    """Convert data coordinates to pixel coordinates."""
    x_log = cal.get("x_log_base")
    y_log = cal.get("y_log_base")
    if x_log:
        if data_x is None or float(data_x) <= 0:
            return (float("nan"), float("nan"))
        x_t = math.log(float(data_x), x_log)
    else:
        x_t = float(data_x)
    if y_log:
        if data_y is None or float(data_y) <= 0:
            return (float("nan"), float("nan"))
        y_t = math.log(float(data_y), y_log)
    else:
        y_t = float(data_y)
    return ((x_t - cal["x_offset"]) / cal["x_scale"],
            (y_t - cal["y_offset"]) / cal["y_scale"])
