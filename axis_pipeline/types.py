"""Typed records that flow through the axis calibration pipeline.

All public results are dataclasses with `to_dict()` that produce JSON-safe
output suitable for Streamlit session-state and inter-process serialization.

The pipeline is deliberately scale-agnostic: only `LinearScale` is implemented
in v1, but `ScaleType` and the `Calibration.scale_type` field allow log/category
scales to be added later without changing the rest of the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class ScaleType(str, Enum):
    LINEAR = "linear"
    # LOG10 = "log10"           # planned
    # CATEGORICAL = "categorical"  # planned


class OCRPhase(str, Enum):
    FULL = "full"            # phase A — full image discovery scan
    Y_BAND = "y_band"        # phase B — tight band over y-axis tick labels
    X_BAND = "x_band"        # phase C — tight band over x-axis tick labels


@dataclass
class OCRRecord:
    raw_text: str
    cleaned_text: str
    value: Optional[float]
    is_numeric: bool
    confidence: float
    bbox: Tuple[int, int, int, int]   # x0, y0, x1, y1 (image pixels, original frame)
    center: Tuple[float, float]
    parse_status: str
    parse_flag: str
    phase: str = OCRPhase.FULL.value

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        d["center"] = list(self.center)
        # Legacy schema also exposes these as flat fields:
        d["center_x"] = float(self.center[0])
        d["center_y"] = float(self.center[1])
        d["ocr_confidence"] = float(self.confidence)
        d["flag"] = self.parse_flag
        return d


@dataclass
class AxisFrame:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)


@dataclass
class FramePreview:
    """Output of the frame-detection-only pass.

    Used by the manual-band UI: shows what the pipeline would feed to Phase B/C
    without actually running them. The main pipeline reuses the same internal
    routine, so what the user sees here is exactly what `run_calibration` would
    see if invoked next.
    """
    bbox: Optional[AxisFrame]
    phase_a_records: List["OCRRecord"] = field(default_factory=list)
    axis_confidence: float = 0.0
    mode: str = "unknown"
    warnings: List[str] = field(default_factory=list)
    diagnostics: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "bbox": list(self.bbox.as_tuple()) if self.bbox else None,
            "phase_a_records": [r.to_dict() for r in self.phase_a_records],
            "axis_confidence": float(self.axis_confidence),
            "mode": self.mode,
            "warnings": list(self.warnings),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass
class GridFit:
    """Outcome of fitting a regular 1-D grid to detected tick positions."""
    spacing: float                 # modal spacing in pixels
    origin: float                  # position of grid index 0
    fitted_positions: List[float]  # input positions kept (snapped to grid)
    fitted_indices: List[int]      # integer grid index for each kept position
    rejected_positions: List[float]  # positions dropped as non-grid outliers
    grid_residuals: List[float]    # residual (px) for kept positions
    n_grid_cells: int              # how many integer indices the grid spans
    success: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "spacing": float(self.spacing),
            "origin": float(self.origin),
            "fitted_positions": [float(x) for x in self.fitted_positions],
            "fitted_indices": [int(i) for i in self.fitted_indices],
            "rejected_positions": [float(x) for x in self.rejected_positions],
            "grid_residuals": [float(x) for x in self.grid_residuals],
            "n_grid_cells": int(self.n_grid_cells),
            "success": bool(self.success),
        }


@dataclass
class PairedTick:
    """One geometric tick that was matched to one OCR numeric label."""
    pixel_position: float       # along the axis (x for x-axis, y for y-axis)
    fixed_axis_pixel: float     # axis line position, perpendicular to pixel_position
    data_value: float
    pair_distance_px: float
    grid_index: Optional[int]   # integer grid index post grid-fit
    label_bbox: Tuple[int, int, int, int]
    raw_text: str
    cleaned_text: str
    ocr_confidence: float
    parse_status: str
    flag: str
    include: bool = True
    status: str = "paired_to_tick_mark"

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["bbox"] = list(self.label_bbox)
        d.pop("label_bbox", None)
        return d


@dataclass
class AxisCalibration:
    scale: float        # data units per pixel  (slope of data = scale*pixel + offset)
    offset: float       # data value at pixel 0
    n_points: int
    method: str         # "ols" | "two_point"
    rmse_px: float      # residual RMS in pixel coordinates
    rmse_data: float    # residual RMS in data coordinates
    log_likelihood: Optional[float] = None
    slope_se: Optional[float] = None        # standard error of slope
    offset_se: Optional[float] = None
    df_t: Optional[float] = None            # Student-t df, if used

    def data_to_pixel(self, value: float) -> float:
        if abs(self.scale) < 1e-15:
            return float("nan")
        return (value - self.offset) / self.scale

    def pixel_to_data(self, px: float) -> float:
        return self.scale * px + self.offset

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class CalibrationConfig:
    """Tunables for the pipeline.

    Defaults are calibrated for journal-quality scientific plots
    (R/SAS/Python/Excel/SPSS) on light backgrounds.
    """
    # OCR
    min_ocr_confidence: float = 0.20
    use_gpu: bool = False
    ocr_pad_px: int = 4
    # Y-band: how far left of axis to search for y-tick labels.
    # Y-axis labels are right-aligned and grow leftward as digits accumulate
    # (e.g. "100" extends ~3x further left than "0"). 90px is enough for
    # 3-digit integers and short negatives at typical journal-figure DPI;
    # wider for scientific notation or very large fonts via the fallback.
    y_band_extra_px: int = 90
    # X-band: how far below axis to search for x-tick labels.
    # Tight default — captures the tick label row but excludes the
    # x-axis title which typically sits 30-50px below the tick labels.
    # X-axis labels are center-aligned on the tick so 2-3 digit labels
    # don't grow horizontally past the band; this stays narrow.
    x_band_extra_px: int = 28
    # Horizontal padding for x-band (don't bleed into y-axis label area at the left).
    x_band_extra_horizontal_px: int = 0
    # Inward trim for y-band top/bottom: excludes labels at the extreme ends of the y-axis.
    y_band_extra_vertical_px: int = 0
    # Fallback band sizes — used only if the narrow band returns 0 numeric records.
    # Wider bands risk capturing axis titles, but rescue plots where tick labels
    # sit unusually far from the axis (large fonts, multi-line label formats).
    x_band_fallback_extra_px: int = 70
    y_band_fallback_extra_px: int = 140
    # ── EasyOCR tuning for band scans ────────────────────────────
    # Tick labels are typically 8-12px tall in matplotlib defaults, right at
    # EasyOCR's `min_size=10` detection floor. Upsampling the cropped band
    # before recognition makes detection much more reliable for small text.
    band_upsample: float = 2.5
    # Detection-threshold overrides for band scans. Defaults are tuned to
    # detect isolated tick-label digits even when they're small and widely
    # spaced (which defeats EasyOCR's word-oriented defaults). Set values to
    # `None` here to fall back to EasyOCR's library defaults; values below
    # are intentionally more permissive.
    band_text_threshold: float = 0.5    # default 0.7
    band_low_text: float = 0.3          # default 0.4
    band_link_threshold: float = 0.3    # default 0.4
    band_min_size: int = 5              # default 10
    band_mag_ratio: float = 1.5         # default 1.5; effective with upsample
    # Grid fit
    grid_residual_tolerance_frac: float = 0.20  # fraction of modal spacing
    grid_min_ticks: int = 3
    # Pairing
    pair_max_distance_frac_of_spacing: float = 0.55  # accept pair if dist < this * spacing
    pair_max_distance_abs_px: float = 80.0
    # Calibration
    scale_type: str = ScaleType.LINEAR.value
    # Misc
    enable_phase_b_y_band: bool = True
    enable_phase_c_x_band: bool = True


@dataclass
class CalibrationResult:
    """Top-level outcome of the multi-phase pipeline.

    Carries enough information to (a) drive the Streamlit UI, (b) reproduce
    the calibration math, (c) render diagnostic overlays, and (d) round-trip
    edits to the tick tables.
    """
    success: bool
    confidence: float
    mode: str
    bbox: Optional[AxisFrame]
    x_calibration: Optional[AxisCalibration]
    y_calibration: Optional[AxisCalibration]
    x_paired_ticks: List[PairedTick] = field(default_factory=list)
    y_paired_ticks: List[PairedTick] = field(default_factory=list)
    x_geometric_ticks: List[float] = field(default_factory=list)
    y_geometric_ticks: List[float] = field(default_factory=list)
    x_grid_fit: Optional[GridFit] = None
    y_grid_fit: Optional[GridFit] = None
    p1_pixel: Optional[Tuple[float, float]] = None
    p2_pixel: Optional[Tuple[float, float]] = None
    p3_pixel: Optional[Tuple[float, float]] = None
    p1_data_x: Optional[float] = None
    p2_data_x: Optional[float] = None
    p3_data_x: Optional[float] = None    # NEW: derived from x-calibration at p3_pixel
    p1_data_y: Optional[float] = None
    p3_data_y: Optional[float] = None
    ocr_records: List[OCRRecord] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    diagnostics: Dict[str, object] = field(default_factory=dict)
    config: Optional[CalibrationConfig] = None

    def to_legacy_dict(self) -> Dict[str, object]:
        """Render the legacy schema consumed by app_auto_axis.py + ocr_axis helpers."""
        d: Dict[str, object] = {}
        d["success"] = bool(self.success)
        d["confidence"] = float(self.confidence)
        d["mode"] = self.mode
        d["bbox"] = list(self.bbox.as_tuple()) if self.bbox else None
        d["p1"] = list(self.p1_pixel) if self.p1_pixel else None
        d["p2"] = list(self.p2_pixel) if self.p2_pixel else None
        d["p3"] = list(self.p3_pixel) if self.p3_pixel else None
        d["p1_data_x"] = self.p1_data_x
        d["p2_data_x"] = self.p2_data_x
        d["p3_data_x"] = self.p3_data_x
        d["p1_data_y"] = self.p1_data_y
        d["p3_data_y"] = self.p3_data_y

        d["x_ticks"] = [[float(p), float(self.bbox.bottom if self.bbox else 0), 1.0]
                        for p in self.x_geometric_ticks]
        d["y_ticks"] = [[float(self.bbox.left if self.bbox else 0), float(p), 1.0]
                        for p in self.y_geometric_ticks]

        d["x_tick_table"] = [_paired_to_legacy_row(t, "x") for t in self.x_paired_ticks]
        d["y_tick_table"] = [_paired_to_legacy_row(t, "y") for t in self.y_paired_ticks]

        d["x_grid_fit"] = self.x_grid_fit.to_dict() if self.x_grid_fit else None
        d["y_grid_fit"] = self.y_grid_fit.to_dict() if self.y_grid_fit else None
        d["x_calibration"] = self.x_calibration.to_dict() if self.x_calibration else None
        d["y_calibration"] = self.y_calibration.to_dict() if self.y_calibration else None

        d["ocr_records"] = [r.to_dict() for r in self.ocr_records]
        d["ocr_record_count"] = len(self.ocr_records)
        d["ocr_enabled"] = True
        d["warnings"] = list(self.warnings)
        d["diagnostics"] = dict(self.diagnostics)
        return d


def _paired_to_legacy_row(t: PairedTick, axis: str) -> Dict[str, object]:
    return {
        "include": bool(t.include),
        "axis": axis,
        "raw_text": t.raw_text,
        "cleaned_text": t.cleaned_text,
        "value": float(t.data_value) if t.data_value is not None else None,
        "pixel_position": float(t.pixel_position),
        "fixed_axis_pixel": float(t.fixed_axis_pixel),
        "ocr_confidence": float(t.ocr_confidence),
        "pair_distance_px": float(t.pair_distance_px) if np.isfinite(t.pair_distance_px) else None,
        "parse_status": t.parse_status,
        "status": t.status,
        "flag": t.flag,
        "bbox": list(t.label_bbox),
        "grid_index": t.grid_index,
    }
