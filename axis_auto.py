"""Backwards-compat shim. The real implementation lives in `axis_pipeline`.

This module used to contain the entire geometry-only axis/tick detector.
It is now a thin wrapper around the multi-phase pipeline package, preserving
the public API the Streamlit app uses:
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
