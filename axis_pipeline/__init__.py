"""Axis calibration pipeline.

A multi-phase OCR + geometry pipeline for extracting calibration anchors
from journal-quality scientific plots.

Public surface:
    run_calibration(img_bgr, config=None, ocr_runner=None) -> CalibrationResult
    detect_axis_frame(img_bgr, config=None, ocr_runner=None) -> FramePreview
    render_overlay(img_bgr, result) -> RGB image
    render_band_preview(img_bgr, bbox, y_band, x_band) -> RGB image
    CalibrationConfig, CalibrationResult, AxisCalibration, FramePreview, ...
    parse_numeric_tick(text)  — re-exported for legacy callers
"""

from .calibration import calibrate_axis
from .gridfit import fit_linear_grid
from .ocr import (
    crop_band,
    keep_only_band,
    mask_records,
    parse_numeric_tick,
    run_easyocr,
    x_label_band,
    y_label_band,
)
from .overlay import render_band_preview, render_overlay
from .pipeline import detect_axis_frame, run_calibration
from .types import (
    AxisCalibration,
    AxisFrame,
    CalibrationConfig,
    CalibrationResult,
    FramePreview,
    GridFit,
    OCRPhase,
    OCRRecord,
    PairedTick,
    ScaleType,
)

__all__ = [
    "AxisCalibration",
    "AxisFrame",
    "CalibrationConfig",
    "CalibrationResult",
    "FramePreview",
    "GridFit",
    "OCRPhase",
    "OCRRecord",
    "PairedTick",
    "ScaleType",
    "calibrate_axis",
    "crop_band",
    "detect_axis_frame",
    "fit_linear_grid",
    "keep_only_band",
    "mask_records",
    "parse_numeric_tick",
    "render_band_preview",
    "render_overlay",
    "run_calibration",
    "run_easyocr",
    "x_label_band",
    "y_label_band",
]
