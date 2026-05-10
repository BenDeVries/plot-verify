"""Backwards-compat shim. The real implementation lives in `axis_pipeline`.

This module used to contain the OCR-assisted detection pipeline. It now
delegates to the multi-phase calibration package while preserving the public
API the Streamlit app uses:
    auto_detect_axes_ticks_ocr(img_bgr, ...) -> dict
    build_ocr_debug_overlay(img_bgr, detection, ...) -> RGB image
    update_detection_from_tick_tables(detection, x_df, y_df) -> dict
    parse_numeric_tick(text) -> (value, cleaned, status, flag)

For new code, prefer:
    from axis_pipeline import run_calibration, render_overlay
"""
from axis_pipeline import parse_numeric_tick
from axis_pipeline.legacy import (
    auto_detect_axes_ticks_ocr,
    build_ocr_debug_overlay,
    update_detection_from_tick_tables,
)

__all__ = [
    "auto_detect_axes_ticks_ocr",
    "build_ocr_debug_overlay",
    "update_detection_from_tick_tables",
    "parse_numeric_tick",
]
