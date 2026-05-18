"""Backwards-compat shim. The real implementation lives in `axis_pipeline`.

DEPRECATED: This module re-exports dict-shaped legacy functions for the
Streamlit app. New code should import from `axis_pipeline` (typed API).

The Streamlit app silences the underlying `DeprecationWarning` so users do
not see the migration noise; external scripts that import these symbols
WILL see the warning (which is the intended signal).

    auto_detect_axes_and_ticks(img_bgr) -> dict
    build_diagnostic_overlay(img_bgr, detection) -> RGB image

For new code, prefer:
    from axis_pipeline import run_calibration, render_overlay
"""
from axis_pipeline.legacy import (
    auto_detect_axes_and_ticks,
    build_diagnostic_overlay,
)

__all__ = ["auto_detect_axes_and_ticks", "build_diagnostic_overlay"]
