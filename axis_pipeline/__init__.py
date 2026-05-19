"""Axis calibration pipeline.

A multi-phase OCR + geometry pipeline for extracting calibration anchors
from journal-quality scientific plots.

Public surface:
    run_calibration(img_bgr, config=None, ocr_runner=None) -> CalibrationResult
    detect_axis_frame(img_bgr, config=None, ocr_runner=None) -> FramePreview
    manual_calibration(p1_pixel=, p2_pixel=, p3_pixel=, ...) -> CalibrationResult
    ocr_available() -> bool   — runtime check for EasyOCR availability
    render_overlay(img_bgr, result) -> RGB image
    render_band_preview(img_bgr, bbox, y_band, x_band) -> RGB image
    CalibrationConfig, CalibrationResult, AxisCalibration, FramePreview, ...
    parse_numeric_tick(text)  — re-exported for legacy callers
    PIPELINE_VERSION    — bumped manually when run_calibration's output
                          semantics change. Saved sessions compare this to
                          decide whether to trust their saved CalibrationResult
                          or re-run the pipeline on load.
"""

# Bump this constant whenever run_calibration's output semantics change.
# Used by plotverify_core.serialization to decide whether to trust a saved
# CalibrationResult or re-invoke run_calibration on session load.
PIPELINE_VERSION = "0.1"

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
from .pipeline import detect_axis_frame, manual_calibration, ocr_available, run_calibration
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
    "PIPELINE_VERSION",
    "PairedTick",
    "ScaleType",
    "calibrate_axis",
    "crop_band",
    "detect_axis_frame",
    "fit_linear_grid",
    "keep_only_band",
    "manual_calibration",
    "mask_records",
    "ocr_available",
    "parse_numeric_tick",
    "render_band_preview",
    "render_overlay",
    "run_calibration",
    "run_easyocr",
    "x_label_band",
    "y_label_band",
]
