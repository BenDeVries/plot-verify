"""Multi-phase calibration pipeline.

Top-level flow:

    Phase A: full-image OCR discovery scan
        ↓
    Mask all detected text
        ↓
    Detect axes geometrically on text-masked image
        ↓
    Phase B (y) and Phase C (x): re-OCR a tight band over each axis label
        strip with `allowlist=numeric` for high-precision tick label reading
        ↓
    Detect geometric tick positions on text-masked image
        ↓
    Grid-fit each axis's tick positions (linear-scale modal-spacing snap)
        ↓
    Pair each surviving tick to one OCR numeric label, drop unpaired ticks
        ↓
    Calibrate each axis independently (OLS or Student-t MLE)
        ↓
    Choose anchors P1, P2, P3 and derive P3.data_x from the x-calibration

The whole pipeline is driven by `run_calibration` which returns a
`CalibrationResult`. A legacy adaptor in `legacy.py` produces the dict shape
the existing Streamlit app consumes.
"""
from __future__ import annotations

import logging
import math
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from . import geometry as geom
from . import ocr as ocr_mod
from . import pairing as pair_mod
from .calibration import calibrate_axis
from .gridfit import fit_linear_grid
from .types import (
    AxisCalibration,
    AxisFrame,
    CalibrationConfig,
    CalibrationResult,
    FramePreview,
    OCRPhase,
    OCRRecord,
    PairedTick,
    ScaleType,
)

log = logging.getLogger(__name__)


def ocr_available() -> bool:
    """Return True if EasyOCR can be imported in the current environment.

    Used by the UI to decide whether to enable auto-calibration controls.
    Pure check — does not load any models, so it's cheap to call repeatedly.
    """
    try:
        import easyocr  # noqa: F401
        return True
    except Exception:
        # ImportError on missing package; other errors (e.g. CUDA misconfig
        # raising on import in torch) also count as unavailable.
        return False


def manual_calibration(
    *,
    p1_pixel: Tuple[float, float],
    p2_pixel: Tuple[float, float],
    p3_pixel: Tuple[float, float],
    p1_data_x: float,
    p2_data_x: float,
    p3_data_y: float,
    p1_data_y: float = 0.0,
    bbox: Optional[AxisFrame] = None,
    x_log_base: Optional[float] = None,
    y_log_base: Optional[float] = None,
) -> CalibrationResult:
    """Construct a CalibrationResult from user-supplied anchor points.

    Does not invoke the OCR/geometry pipeline. Intended for the manual-only
    workflow (no EasyOCR / no pytorch) and for users who want to override
    auto-detection entirely. The returned result mirrors the same shape as
    `run_calibration` so the rest of the app (overlay rendering, legacy dict
    adapter, CSV export) is unaffected.

    Pass ``x_log_base=10.0`` / ``y_log_base=10.0`` to fit in log space; data
    values must be strictly positive when a log axis is requested.
    """
    p1_px_x, p1_px_y = float(p1_pixel[0]), float(p1_pixel[1])
    p2_px_x, p2_px_y = float(p2_pixel[0]), float(p2_pixel[1])
    p3_px_x, p3_px_y = float(p3_pixel[0]), float(p3_pixel[1])

    def _fail(msg: str) -> CalibrationResult:
        return CalibrationResult(
            success=False, confidence=0.0, mode="manual_failed",
            bbox=bbox,
            x_calibration=None, y_calibration=None,
            p1_pixel=p1_pixel, p2_pixel=p2_pixel, p3_pixel=p3_pixel,
            p1_data_x=p1_data_x, p2_data_x=p2_data_x, p3_data_x=None,
            p1_data_y=p1_data_y, p3_data_y=p3_data_y,
            warnings=[msg],
            diagnostics={"source": "manual"},
            config=CalibrationConfig(),
        )

    if abs(p2_px_x - p1_px_x) < 1e-9:
        return _fail("Manual calibration: P1 and P2 must differ in pixel X.")

    # Transform data values to log space if requested.
    def _t(value: float, log_base: Optional[float]) -> Optional[float]:
        if not log_base or log_base == 1.0:
            return float(value)
        if log_base != 10.0:
            return float(value)
        v = float(value)
        if v <= 0:
            return None
        return float(math.log10(v))

    x_p1_t = _t(p1_data_x, x_log_base)
    x_p2_t = _t(p2_data_x, x_log_base)
    if x_p1_t is None or x_p2_t is None:
        return _fail("Manual calibration: log10 X-axis requires positive data values.")

    x_scale = (x_p2_t - x_p1_t) / (p2_px_x - p1_px_x)
    x_offset = x_p1_t - x_scale * p1_px_x
    if not math.isfinite(x_scale) or abs(x_scale) < 1e-12:
        return _fail("Manual calibration: degenerate X-axis values.")

    x_axis_pixel_y = (p1_px_y + p2_px_y) / 2.0
    if abs(p3_px_y - x_axis_pixel_y) < 1e-9:
        return _fail("Manual calibration: P3 must differ in pixel Y from the X-axis baseline.")

    y_p3_t = _t(p3_data_y, y_log_base)
    y_base_t = _t(p1_data_y, y_log_base)
    if y_p3_t is None or y_base_t is None:
        return _fail("Manual calibration: log10 Y-axis requires positive data values.")

    y_scale = (y_p3_t - y_base_t) / (p3_px_y - x_axis_pixel_y)
    y_offset = y_p3_t - y_scale * p3_px_y
    if not math.isfinite(y_scale) or abs(y_scale) < 1e-12:
        return _fail("Manual calibration: degenerate Y-axis values.")

    x_cal = AxisCalibration(
        scale=float(x_scale), offset=float(x_offset),
        n_points=2, method="manual_two_point",
        rmse_px=0.0, rmse_data=0.0,
        log_base=float(x_log_base) if x_log_base else None,
    )
    y_cal = AxisCalibration(
        scale=float(y_scale), offset=float(y_offset),
        n_points=2, method="manual_two_point",
        rmse_px=0.0, rmse_data=0.0,
        log_base=float(y_log_base) if y_log_base else None,
    )

    # P3.data_x derived from the x-axis calibration.
    p3_data_x = float(x_cal.pixel_to_data(p3_px_x))

    return CalibrationResult(
        success=True,
        confidence=1.0,
        mode="manual",
        bbox=bbox,
        x_calibration=x_cal,
        y_calibration=y_cal,
        p1_pixel=p1_pixel, p2_pixel=p2_pixel, p3_pixel=p3_pixel,
        p1_data_x=float(p1_data_x), p2_data_x=float(p2_data_x), p3_data_x=p3_data_x,
        p1_data_y=float(p1_data_y), p3_data_y=float(p3_data_y),
        warnings=[],
        diagnostics={"source": "manual"},
        config=CalibrationConfig(),
    )


# Type alias: (image_bgr, allowlist, phase, bbox_offset, kwargs) -> records.
# Injectable so tests can swap in a custom runner instead of EasyOCR.
OCRRunner = Callable[..., List[OCRRecord]]


def _default_ocr_runner(
    img_bgr,
    *,
    gpu: bool = False,
    min_confidence: float = 0.20,
    allowlist: Optional[str] = None,
    phase: str = OCRPhase.FULL.value,
    bbox_offset: Tuple[int, int] = (0, 0),
    upsample: float = 1.0,
    detection_params: Optional[dict] = None,
) -> List[OCRRecord]:
    return ocr_mod.run_easyocr(
        img_bgr,
        gpu=gpu,
        min_confidence=min_confidence,
        allowlist=allowlist,
        phase=phase,
        bbox_offset=bbox_offset,
        upsample=upsample,
        detection_params=detection_params,
    )


def _detect_frame_internal(
    img_bgr,
    cfg: CalibrationConfig,
    ocr: OCRRunner,
) -> Tuple[FramePreview, np.ndarray, np.ndarray]:
    """Phase A OCR + geometric axis detection.

    Returns (preview, masked_img, band_source). The two image byproducts are
    needed by `run_calibration` for the rest of the pipeline:

      * `masked_img` has every Phase A text region whitened — used for
        geometric tick detection (so tick labels can't confuse the projection
        profile).
      * `band_source` has only NON-numeric Phase A text whitened — used as the
        substrate for Phase B/C band crops (numeric labels stay visible so the
        band re-OCR can re-find them at higher confidence; titles, legends,
        and other surrounding text are removed so they can't pollute the band).

    `detect_axis_frame` calls this helper and discards the byproducts;
    `run_calibration` calls it and uses them. This is the only place the
    Phase A + axis-detection logic lives.

    Note: when bbox is None, `band_source` is still returned (for the same
    return shape) but is just `img_bgr.copy()` — callers must check
    `preview.bbox is not None` before using it.
    """
    warnings: List[str] = []
    diagnostics: dict = {}

    # ── Phase A: full-image discovery OCR ──────────────────────────
    try:
        full_records = ocr(
            img_bgr,
            gpu=cfg.use_gpu,
            min_confidence=cfg.min_ocr_confidence,
            allowlist=None,
            phase=OCRPhase.FULL.value,
        )
    except Exception as e:
        warnings.append(f"Phase A OCR failed: {type(e).__name__}: {e}")
        full_records = []
    diagnostics["phase_a_record_count"] = len(full_records)

    # ── Geometric axis detection on text-masked image ─────────────
    # Filter out implausibly large Phase A detections before masking.
    # EasyOCR sometimes merges a whole column of tick labels into one tall
    # region (e.g. "10^0"–"10^30" on a MATLAB log-log plot → one bbox spanning
    # the full y-axis height). That bbox overlaps and whites out the axis line
    # itself, causing geometric detection to fail. Individual text labels are
    # typically < 1% of image area; a merged column is 6–10%.
    _img_area = img_bgr.shape[0] * img_bgr.shape[1]
    _max_mask_area = 0.06 * _img_area
    _records_for_mask = [
        r for r in full_records
        if (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]) <= _max_mask_area
    ]
    diagnostics["phase_a_oversized_filtered"] = len(full_records) - len(_records_for_mask)
    masked = ocr_mod.mask_records(img_bgr, _records_for_mask, pad=cfg.ocr_pad_px)
    bbox, mode, axis_conf, axis_warnings = geom.detect_axes(masked)
    warnings.extend(axis_warnings)
    diagnostics["axis_confidence"] = float(axis_conf)

    if bbox is not None:
        # Substrate for Phase B/C band crops (built lazily — only useful if bbox
        # was found, since without a frame we can't compute band geometry).
        band_source = ocr_mod.mask_non_numeric_records(
            img_bgr, full_records, pad=cfg.ocr_pad_px,
        )
    else:
        band_source = img_bgr.copy()

    preview = FramePreview(
        bbox=bbox,
        phase_a_records=full_records,
        axis_confidence=float(axis_conf),
        mode=mode,
        warnings=warnings,
        diagnostics=diagnostics,
    )
    return preview, masked, band_source


def detect_axis_frame(
    img_bgr,
    *,
    config: Optional[CalibrationConfig] = None,
    ocr_runner: Optional[OCRRunner] = None,
) -> FramePreview:
    """Run Phase A OCR + geometric axis detection only.

    Used by the manual-band UI to show the user the detected bbox and Phase A
    text regions before they commit to a full calibration. Matches the first
    two phases of `run_calibration` exactly — what you see here is what the
    full pipeline would see if invoked next on the same image.

    Phase B/C band scans, grid fit, pairing, and calibration math are skipped.
    """
    cfg = config or CalibrationConfig()
    ocr = ocr_runner or _default_ocr_runner
    preview, _masked, _band_source = _detect_frame_internal(img_bgr, cfg, ocr)
    return preview


def run_calibration(
    img_bgr,
    *,
    config: Optional[CalibrationConfig] = None,
    ocr_runner: Optional[OCRRunner] = None,
) -> CalibrationResult:
    """Run the full multi-phase pipeline on a single plot image."""
    cfg = config or CalibrationConfig()
    ocr = ocr_runner or _default_ocr_runner

    # ── Phase A + axis detection (shared with detect_axis_frame) ──
    preview, masked, band_source = _detect_frame_internal(img_bgr, cfg, ocr)
    full_records = preview.phase_a_records
    bbox = preview.bbox
    mode = preview.mode
    axis_conf = preview.axis_confidence
    # Take ownership of mutable state from the preview — run_calibration extends
    # both lists/dicts as later phases run.
    warnings: List[str] = list(preview.warnings)
    diagnostics: dict = dict(preview.diagnostics)

    if bbox is None:
        return CalibrationResult(
            success=False, confidence=0.0,
            mode="failed", bbox=None,
            x_calibration=None, y_calibration=None,
            ocr_records=full_records,
            warnings=warnings, diagnostics=diagnostics,
            config=cfg,
        )

    # ── Phase B: tight y-axis label band re-OCR ───────────────────
    # `band_source` (returned above) has all non-numeric Phase A records
    # whitened so they can't confuse the recognizer if the band overlaps
    # axis titles or legend text. Numeric Phase A records stay visible so
    # the band re-scan re-finds them at higher confidence.

    y_band_records: List[OCRRecord] = []
    if cfg.enable_phase_b_y_band:
        y_band_records, y_band_used = _band_ocr_with_fallback(
            ocr, band_source, cfg, axis="y", bbox=bbox, warnings=warnings,
        )
        diagnostics["y_band_extra_used"] = y_band_used
    diagnostics["phase_b_record_count"] = len(y_band_records)

    # ── Phase C: tight x-axis label band re-OCR ───────────────────
    x_band_records: List[OCRRecord] = []
    if cfg.enable_phase_c_x_band:
        x_band_records, x_band_used = _band_ocr_with_fallback(
            ocr, band_source, cfg, axis="x", bbox=bbox, warnings=warnings,
        )
        diagnostics["x_band_extra_used"] = x_band_used
    diagnostics["phase_c_record_count"] = len(x_band_records)

    # Combine records (band scans are authoritative for tick labels; full scan
    # is the basis for everything else). De-duplicate by bbox-overlap so we
    # don't double-count the same label.
    combined_records = _merge_record_sources(full_records, y_band_records, x_band_records)
    diagnostics["combined_record_count"] = len(combined_records)

    # ── Geometric tick detection ──────────────────────────────────
    _, dark = geom.prepare_dark_mask(masked)
    x_outward, x_inward, x_tick_diag = geom.detect_x_tick_positions(dark, bbox, config=cfg)
    y_outward, y_inward, y_tick_diag = geom.detect_y_tick_positions(dark, bbox, config=cfg)
    diagnostics.update(x_tick_diag)
    diagnostics.update(y_tick_diag)

    # ── Label filtering (done early so fallback logic can use the records) ──
    x_horiz = cfg.x_band_extra_horizontal_px
    y_vert = cfg.y_band_extra_vertical_px
    x_label_records = pair_mod.filter_x_axis_labels(
        combined_records, bbox,
        x_min=bbox.left + x_horiz if x_horiz > 0 else None,
        x_max=bbox.right - x_horiz if x_horiz > 0 else None,
    )
    y_label_records = pair_mod.filter_y_axis_labels(
        combined_records, bbox,
        y_min=bbox.top + y_vert if y_vert > 0 else None,
        y_max=bbox.bottom - y_vert if y_vert > 0 else None,
    )

    # When the dedicated band scan produced enough records it is authoritative:
    # suppress full-phase (Phase A) records from pairing so stray detections
    # (e.g. a "2" misread from a "2e+90" label in the full image scan) cannot
    # displace or corrupt the band-scan calibration.
    # Guard: if band records have < 2 distinct numeric values (EasyOCR only
    # read the base "10" without exponents on log-scale labels), they are
    # degenerate and cannot calibrate the axis — fall back to Phase A records.
    _y_band_recs = [r for r in y_label_records if r.phase == OCRPhase.Y_BAND.value]
    _y_band_distinct = len({r.value for r in _y_band_recs if r.value is not None})
    if len(_y_band_recs) >= cfg.grid_min_ticks and _y_band_distinct >= 2:
        y_label_records = _y_band_recs
    _x_band_recs = [r for r in x_label_records if r.phase == OCRPhase.X_BAND.value]
    _x_band_distinct = len({r.value for r in _x_band_recs if r.value is not None})
    if len(_x_band_recs) >= cfg.grid_min_ticks and _x_band_distinct >= 2:
        x_label_records = _x_band_recs

    # Fix log-scale label OCR failures before pairing.
    # Pass 1: re-join split records ("10" + "N" → "10^N").
    # Pass 2: correct concatenated reads ("100" → 10^0, "1010" → 10^10) when
    #          most labels on the axis match the "10\d{1,2}" pattern.
    y_label_records = ocr_mod.merge_superscript_fragments(y_label_records)
    y_label_records = ocr_mod.deconcat_log10_labels(y_label_records)
    x_label_records = ocr_mod.merge_superscript_fragments(x_label_records)
    x_label_records = ocr_mod.deconcat_log10_labels(x_label_records)

    diagnostics["x_label_candidates"] = len(x_label_records)
    diagnostics["y_label_candidates"] = len(y_label_records)

    # ── Grid fit with cascade: outward → inward → merged ─────────
    # Try outward-only first to prevent inward false-positives (data markers,
    # gridlines) from poisoning the modal-spacing estimate. Only if outward
    # detection fails do we try inward or the merged candidate set.
    if cfg.scale_type != ScaleType.LINEAR.value:
        warnings.append(f"Scale type {cfg.scale_type!r} is not yet implemented; "
                        "treating as linear.")

    x_tick_positions, x_grid = _cascade_grid_fit(
        "x", x_outward, x_inward, cfg, diagnostics,
    )
    y_tick_positions, y_grid = _cascade_grid_fit(
        "y", y_outward, y_inward, cfg, diagnostics,
    )
    diagnostics["x_tick_raw_count"] = len(x_tick_positions)
    diagnostics["y_tick_raw_count"] = len(y_tick_positions)

    # ── Label-center fallback: last resort if geometric grid failed ───────────
    x_tick_positions, x_grid = _label_center_fallback(
        "x", x_tick_positions, x_grid, x_label_records, cfg, diagnostics, warnings,
    )
    y_tick_positions, y_grid = _label_center_fallback(
        "y", y_tick_positions, y_grid, y_label_records, cfg, diagnostics, warnings,
    )

    diagnostics["x_grid_kept"] = len(x_grid.fitted_positions)
    diagnostics["y_grid_kept"] = len(y_grid.fitted_positions)
    diagnostics["x_grid_rejected"] = len(x_grid.rejected_positions)
    diagnostics["y_grid_rejected"] = len(y_grid.rejected_positions)

    # ── Pairing ───────────────────────────────────────────────────

    x_max_pair_dist = min(
        cfg.pair_max_distance_abs_px,
        cfg.pair_max_distance_frac_of_spacing * x_grid.spacing if x_grid.spacing > 0 else cfg.pair_max_distance_abs_px,
    )
    y_max_pair_dist = min(
        cfg.pair_max_distance_abs_px,
        cfg.pair_max_distance_frac_of_spacing * y_grid.spacing if y_grid.spacing > 0 else cfg.pair_max_distance_abs_px,
    )

    x_paired = pair_mod.pair_x(x_label_records, x_grid, bbox, max_distance=x_max_pair_dist)
    y_paired = pair_mod.pair_y(y_label_records, y_grid, bbox, max_distance=y_max_pair_dist)

    if not any(t.include for t in x_paired):
        warnings.append(_diagnose_pair_failure(
            "x", x_band_records, full_records, x_label_records, x_grid, x_max_pair_dist,
        ))
    if not any(t.include for t in y_paired):
        warnings.append(_diagnose_pair_failure(
            "y", y_band_records, full_records, y_label_records, y_grid, y_max_pair_dist,
        ))

    # ── Calibration ───────────────────────────────────────────────
    x_cal = calibrate_axis(x_paired)
    y_cal = calibrate_axis(y_paired)

    # If calibration came back None despite having ≥2 paired ticks, it's because
    # the values were degenerate (all the same). Surface a specific, actionable
    # warning — the user otherwise sees "calibration failed" with no context.
    if x_cal is None and any(t.include for t in x_paired):
        warnings.append(_diagnose_degenerate_axis("x", x_paired))
    if y_cal is None and any(t.include for t in y_paired):
        warnings.append(_diagnose_degenerate_axis("y", y_paired))

    # ── Anchor selection P1/P2/P3 ─────────────────────────────────
    p1_pixel, p2_pixel, p3_pixel = None, None, None
    p1_data_x = p2_data_x = None
    p1_data_y = p3_data_y = None
    p3_data_x = None

    inc_x = [t for t in x_paired if t.include]
    inc_y = [t for t in y_paired if t.include]
    inc_x.sort(key=lambda t: t.pixel_position)
    inc_y.sort(key=lambda t: t.pixel_position)  # ascending pixel-y = top to bottom

    if inc_x:
        # P1, P2 are the leftmost and rightmost paired x-ticks at the axis baseline.
        p1_data_x = float(inc_x[0].data_value)
        p2_data_x = float(inc_x[-1].data_value)
        p1_pixel = (float(inc_x[0].pixel_position), float(bbox.bottom))
        p2_pixel = (float(inc_x[-1].pixel_position), float(bbox.bottom))

    if inc_y:
        # P3 is the topmost paired y-tick (largest data value, smallest pixel-y).
        topmost = min(inc_y, key=lambda t: t.pixel_position)
        bottommost = max(inc_y, key=lambda t: t.pixel_position)
        p3_pixel_x = float(p1_pixel[0]) if p1_pixel else float(bbox.left)
        p3_pixel = (p3_pixel_x, float(topmost.pixel_position))
        p3_data_y = float(topmost.data_value)
        p1_data_y = float(bottommost.data_value)
        # Anchor P1/P2 vertically at the lowest-paired-y label rather than the
        # axis line, so that calibration extrapolates correctly even when the
        # bottom-most label (e.g. -100) does not sit at the bottom axis pixel.
        if p1_pixel:
            p1_pixel = (p1_pixel[0], float(bottommost.pixel_position))
        if p2_pixel:
            p2_pixel = (p2_pixel[0], float(bottommost.pixel_position))

    # P3.data_x: derive from x-calibration evaluated at p3_pixel.x
    if p3_pixel and x_cal is not None:
        p3_data_x = float(x_cal.pixel_to_data(p3_pixel[0]))

    # ── Confidence aggregation ────────────────────────────────────
    confidence = _compute_confidence(
        axis_conf=axis_conf,
        x_cal=x_cal,
        y_cal=y_cal,
        x_grid=x_grid,
        y_grid=y_grid,
    )
    success = (
        x_cal is not None and y_cal is not None
        and bbox is not None
        and confidence >= 0.35
    )

    if confidence < 0.60 and success:
        warnings.append("Low-confidence calibration. Review the diagnostic overlay before applying.")
    elif confidence < 0.85 and success:
        warnings.append("Medium-confidence calibration. Review the diagnostic overlay.")

    has_band_ocr = (
        any(r.phase != OCRPhase.FULL.value for r in combined_records)
        or len(full_records) > 0
    )
    mode_prefix = "ocr_masked_" if has_band_ocr else "geometry_only_"
    return CalibrationResult(
        success=success,
        confidence=confidence,
        mode=mode_prefix + mode,
        bbox=bbox,
        x_calibration=x_cal,
        y_calibration=y_cal,
        x_paired_ticks=x_paired,
        y_paired_ticks=y_paired,
        x_geometric_ticks=list(x_tick_positions),
        y_geometric_ticks=list(y_tick_positions),
        x_grid_fit=x_grid,
        y_grid_fit=y_grid,
        p1_pixel=p1_pixel, p2_pixel=p2_pixel, p3_pixel=p3_pixel,
        p1_data_x=p1_data_x, p2_data_x=p2_data_x, p3_data_x=p3_data_x,
        p1_data_y=p1_data_y, p3_data_y=p3_data_y,
        ocr_records=combined_records,
        warnings=warnings,
        diagnostics=diagnostics,
        config=cfg,
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _bbox_iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0); ih = max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return float(inter / union) if union > 0 else 0.0


def _band_ocr_with_fallback(
    ocr: OCRRunner,
    img_bgr,
    cfg: CalibrationConfig,
    *,
    axis: str,                # "x" or "y"
    bbox: AxisFrame,
    warnings: List[str],
) -> Tuple[List[OCRRecord], int]:
    """Run a band OCR scan; if it returns 0 numeric records, retry with a wider band.

    Returns (records, extra_px_used). The wider retry catches plots where tick
    labels sit unusually far from the axis (large fonts, R/SAS multi-line
    formats). The narrow default protects against the much more common case
    where the wider band would also pull in axis titles or legend text.

    Both passes use upsampled-and-tuned EasyOCR detection params suited for
    small isolated tick-label digits (see `band_*` settings on
    `CalibrationConfig`).
    """
    if axis == "x":
        narrow_extra = cfg.x_band_extra_px
        wide_extra = cfg.x_band_fallback_extra_px
        horiz = cfg.x_band_extra_horizontal_px
        phase_name = OCRPhase.X_BAND.value
        # Only extend the right side of the band when bbox.right sits close to
        # the image's right edge — that's the configuration where the rightmost
        # tick label can overhang the bbox and get clipped (see x_label_band).
        # When there's ample room past bbox.right, extending pulls in unrelated
        # text and can produce a false-confident calibration on otherwise
        # ambiguous plots.
        img_w = img_bgr.shape[1]
        extend_right = 25 if (img_w - int(bbox.right)) < 30 else 0
        _x_slide = cfg.x_band_y_offset
        def compute_band(extra, _h=horiz, _e=extend_right, _s=_x_slide):
            b = ocr_mod.x_label_band(bbox, extra_below=extra,
                                     extra_horizontal=_h, extend_outward=_e)
            if _s:
                b = (b[0], b[1] + _s, b[2], b[3] + _s)
            return (min(b[0], b[2]), min(b[1], b[3]),
                    max(b[0], b[2]), max(b[1], b[3]))
    elif axis == "y":
        narrow_extra = cfg.y_band_extra_px
        wide_extra = cfg.y_band_fallback_extra_px
        vert = cfg.y_band_extra_vertical_px
        phase_name = OCRPhase.Y_BAND.value
        _y_slide = cfg.y_band_x_offset
        def compute_band(extra, _v=vert, _s=_y_slide):
            b = ocr_mod.y_label_band(bbox, extra_left=extra, extra_vertical=_v)
            if _s:
                b = (b[0] + _s, b[1], b[2] + _s, b[3])
            return (min(b[0], b[2]), min(b[1], b[3]),
                    max(b[0], b[2]), max(b[1], b[3]))
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    band_detection_params = {
        "text_threshold": cfg.band_text_threshold,
        "low_text": cfg.band_low_text,
        "link_threshold": cfg.band_link_threshold,
        "min_size": cfg.band_min_size,
        "mag_ratio": cfg.band_mag_ratio,
    }

    def _run(extra: int) -> List[OCRRecord]:
        band = compute_band(extra)
        crop, offset = ocr_mod.crop_band(img_bgr, band)
        if crop.size == 0:
            return []
        return ocr(
            crop,
            gpu=cfg.use_gpu,
            min_confidence=cfg.min_ocr_confidence,
            allowlist="0123456789.+-eE^x",
            phase=phase_name,
            bbox_offset=offset,
            upsample=cfg.band_upsample,
            detection_params=band_detection_params,
        )

    try:
        records = _run(narrow_extra)
    except Exception as e:
        warnings.append(f"Phase {axis}-band OCR failed: {type(e).__name__}: {e}")
        return [], narrow_extra

    n_numeric = sum(1 for r in records if r.is_numeric)
    if n_numeric > 0 or wide_extra <= narrow_extra:
        # For the y-axis: if all numeric records have the same value, the labels
        # are likely rendered as rotated vertical text (R's default las=0 style,
        # where each character is stacked top-to-bottom). Normal horizontal OCR
        # only detects the topmost digit of each stack and misclassifies it.
        # Re-try on a 90°-CW-rotated copy of the band so text is horizontal.
        if axis == "y" and n_numeric >= 3:
            numerics = [r for r in records if r.is_numeric and r.value is not None]
            distinct_vals = {r.value for r in numerics}
            # Trigger rotation when distinct values are far fewer than records:
            # vertical (rotated) labels produce the same misread for every label.
            if len(distinct_vals) * 2 <= n_numeric:
                try:
                    band = compute_band(narrow_extra)
                    crop, crop_offset = ocr_mod.crop_band(img_bgr, band)
                    if crop.size > 0:
                        rot_records = _run_rotated_band_ocr(
                            ocr, crop, crop_offset,
                            cfg=cfg, phase_name=phase_name,
                            band_detection_params=band_detection_params,
                        )
                        rot_distinct = {r.value for r in rot_records}
                        if len(rot_distinct) > 1:
                            warnings.append(
                                "Y-axis: normal band scan returned all-same values "
                                "(vertical/rotated labels detected, R las=0 style); "
                                "using 90°-CW-rotation pass for y-tick labels."
                            )
                            return rot_records, narrow_extra
                except Exception as e:
                    warnings.append(f"Y-axis rotated band OCR failed: {type(e).__name__}: {e}")
        return records, narrow_extra

    # Narrow band returned no numerics — retry once with the wider fallback band.
    try:
        wide_records = _run(wide_extra)
    except Exception as e:
        warnings.append(f"Phase {axis}-band fallback OCR failed: {type(e).__name__}: {e}")
        return records, narrow_extra

    n_wide_numeric = sum(1 for r in wide_records if r.is_numeric)
    if n_wide_numeric > n_numeric:
        return wide_records, wide_extra
    return records, narrow_extra


def _run_rotated_band_ocr(
    ocr: OCRRunner,
    crop: np.ndarray,
    offset: Tuple[int, int],
    *,
    cfg: CalibrationConfig,
    phase_name: str,
    band_detection_params: dict,
) -> List[OCRRecord]:
    """Run band OCR on a 90°-CW-rotated copy of `crop`; map coordinates back.

    Handles R plots using las=0 (default) where y-axis tick labels are rendered
    as vertical text (one character per row, reading bottom-to-top). Normal
    horizontal OCR only detects the topmost digit of each stacked label and
    typically misclassifies the circular shapes as '8'.

    After rotating so text is horizontal, detected bbox/center coords are
    mapped back to the original crop coordinate system via the inverse of
    cv2.ROTATE_90_CLOCKWISE:
        x_image = rcy + dx          (rotated y-row  → original x-col)
        y_image = H − 1 − rcx + dy  (rotated x-col → original y-row, reversed)
    """
    H, W = crop.shape[:2]
    rotated = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    dx, dy = offset

    raw_records = ocr(
        rotated,
        gpu=cfg.use_gpu,
        min_confidence=cfg.min_ocr_confidence,
        allowlist="0123456789.+-eE^x",
        phase=phase_name,
        bbox_offset=(0, 0),
        upsample=cfg.band_upsample,
        detection_params=band_detection_params,
    )

    out: List[OCRRecord] = []
    for r in raw_records:
        if not r.is_numeric:
            continue
        rx0, ry0, rx1, ry1 = r.bbox
        rcx = (rx0 + rx1) / 2.0
        rcy = (ry0 + ry1) / 2.0
        # Inverse rotation: rotated(x_r, y_r) → original(y_r, H-1-x_r)
        cx = rcy + dx
        cy = H - 1 - rcx + dy
        ox0 = int(round(ry0 + dx))
        ox1 = int(round(ry1 + dx))
        oy0 = int(round(H - 1 - rx1 + dy))
        oy1 = int(round(H - 1 - rx0 + dy))
        out.append(OCRRecord(
            raw_text=r.raw_text,
            cleaned_text=r.cleaned_text,
            value=r.value,
            is_numeric=True,
            confidence=r.confidence,
            bbox=(min(ox0, ox1), min(oy0, oy1), max(ox0, ox1), max(oy0, oy1)),
            center=(cx, cy),
            parse_status=r.parse_status,
            parse_flag=r.parse_flag,
            phase=r.phase,
        ))
    return out


def _cascade_grid_fit(
    axis: str,
    outward: List[float],
    inward: List[float],
    cfg: CalibrationConfig,
    diagnostics: dict,
) -> Tuple[List[float], object]:
    """Try grid fit on outward candidates first, then inward, then merged.

    Cascade prevents inward false-positives (data markers, gridlines) from
    corrupting the modal-spacing estimate when outward detection already succeeds.
    Returns (positions_used, grid).
    """
    tol = cfg.grid_residual_tolerance_frac
    min_t = cfg.grid_min_ticks
    dedup = cfg.tick_dedup_tolerance_px

    # 1. Outward only
    grid = fit_linear_grid(outward, tolerance_frac=tol, min_ticks=min_t)
    if grid.success:
        diagnostics[f"{axis}_tick_detection_mode"] = "outward"
        return outward, grid

    # 2. Inward only
    if inward:
        grid = fit_linear_grid(inward, tolerance_frac=tol, min_ticks=min_t)
        if grid.success:
            diagnostics[f"{axis}_tick_detection_mode"] = "inward"
            return inward, grid

    # 3. Merged (deduplicated union)
    merged = geom._dedup_positions(sorted(set(outward) | set(inward)), dedup)
    grid = fit_linear_grid(merged, tolerance_frac=tol, min_ticks=min_t)
    diagnostics[f"{axis}_tick_detection_mode"] = "merged" if grid.success else "failed"
    return merged, grid


def _label_center_fallback(
    axis: str,
    tick_positions: List[float],
    grid,
    label_records: List[OCRRecord],
    cfg: CalibrationConfig,
    diagnostics: dict,
    warnings: List[str],
) -> Tuple[List[float], object]:
    """If geometric grid fit failed, try fitting a grid to OCR label centers.

    For the x-axis uses label x-centers; for y-axis uses label y-centers.
    Only activates when the geometric grid failed AND >= grid_min_ticks monotonic
    numeric labels are available. Returns (positions, grid) — either unchanged or
    replaced by the fallback.
    """
    if grid.success:
        return tick_positions, grid

    numeric = [r for r in label_records if r.is_numeric and r.value is not None]
    if len(numeric) < cfg.grid_min_ticks:
        return tick_positions, grid

    if axis == "x":
        centers = sorted(r.center[0] for r in numeric)
    else:
        centers = sorted(r.center[1] for r in numeric)

    # Quick monotonicity check on data values at the sorted center positions
    sorted_by_center = sorted(numeric, key=lambda r: r.center[0] if axis == "x" else r.center[1])
    values = [r.value for r in sorted_by_center]
    if axis == "x":
        is_mono = all(values[i] <= values[i + 1] for i in range(len(values) - 1))
    else:
        is_mono = all(values[i] >= values[i + 1] for i in range(len(values) - 1))
    if not is_mono:
        return tick_positions, grid

    fallback_grid = fit_linear_grid(
        centers,
        tolerance_frac=cfg.grid_residual_tolerance_frac,
        min_ticks=cfg.grid_min_ticks,
    )
    if not fallback_grid.success:
        return tick_positions, grid

    warnings.append(
        f"{axis.upper()}-axis: no geometric ticks found; using OCR label centers as "
        "provisional tick positions (label_center_fallback)."
    )
    diagnostics[f"{axis}_tick_detection_mode"] = "label_center_fallback"
    return list(centers), fallback_grid


def _diagnose_degenerate_axis(axis: str, paired: List[PairedTick]) -> str:
    """Return a warning explaining why calibration was refused for `axis`.

    Two failure modes covered:
      1. Only one paired label survived — usually OCR detection issue.
      2. Multiple paired labels but all read the same value — usually the
         band crop bisected multi-digit labels (e.g. "10", "20" both read as
         just "0" because the leading digit fell outside the crop).
    """
    inc = [t for t in paired if t.include and t.data_value is not None]
    distinct = sorted({t.data_value for t in inc})
    n = len(inc)
    cfg_key = "y_band_extra_px" if axis == "y" else "x_band_extra_px"
    default_val = 90 if axis == "y" else 28

    if n < 2:
        sole = distinct[0] if distinct else "<none>"
        return (
            f"{axis.upper()}-axis calibration refused: only {n} label(s) paired "
            f"(value: {sole}). At least 2 distinct labels are needed. The OCR "
            f"backend may have missed labels — check the diagnostic overlay; if "
            f"labels are visible but unread, lower `band_text_threshold` or "
            f"`band_min_size` in CalibrationConfig."
        )
    if len(distinct) <= 1:
        sole = distinct[0]
        return (
            f"{axis.upper()}-axis calibration refused: all {n} paired labels read "
            f"as the same value ({sole}). This usually means the {axis}-band crop "
            f"bisected multi-digit labels (e.g. '10' → '0' when the band's left "
            f"edge falls between the '1' and the '0'). Increase `{cfg_key}` in "
            f"CalibrationConfig (default {default_val}) so the band is wide "
            f"enough to fit the longest label including any minus sign or "
            f"leading digits."
        )
    return (
        f"{axis.upper()}-axis calibration refused: only {len(distinct)} distinct "
        f"data value(s) across {n} paired labels — too degenerate to fit. "
        f"Inspect the diagnostic overlay for misread labels."
    )


def _diagnose_pair_failure(
    axis: str,
    band_records: List[OCRRecord],
    full_records: List[OCRRecord],
    label_candidates: List[OCRRecord],
    grid,
    max_distance: float,
) -> str:
    """Return an actionable warning explaining why no pairs survived for `axis`.

    Distinguishes between (a) OCR found nothing in the band, (b) OCR found things
    but the spatial filter rejected them all, (c) records survived but were too
    far from any geometric tick to pair.
    """
    n_band_numeric = sum(1 for r in band_records if r.is_numeric)
    n_full_numeric = sum(1 for r in full_records if r.is_numeric)
    n_candidates = len(label_candidates)
    n_grid = len(grid.fitted_positions) if grid is not None else 0

    if n_band_numeric == 0 and n_full_numeric == 0:
        return (f"No {axis}-tick labels were paired: OCR found no numeric text anywhere. "
                "Check the OCR backend (EasyOCR models downloaded?) and image contrast.")
    if n_band_numeric == 0:
        return (f"No {axis}-tick labels were paired: the {axis}-axis band scan returned "
                f"no numeric records (full-image scan found {n_full_numeric}). "
                f"Try increasing `{axis}_band_extra_px` in CalibrationConfig if the labels "
                "sit further from the axis than the default band height.")
    if n_candidates == 0:
        return (f"No {axis}-tick labels were paired: {n_band_numeric} numeric record(s) "
                f"in the {axis}-band were rejected by the spatial filter (records may "
                "be outside the detected plot bbox). Inspect the diagnostic overlay.")
    if n_grid == 0:
        return (f"No {axis}-tick labels were paired: {n_candidates} numeric label(s) "
                "survived but no geometric ticks were grid-fitted on this axis. "
                "The axis may have ticks too sparse or irregular for the modal-spacing fit.")
    return (f"No {axis}-tick labels were paired: {n_candidates} numeric label(s) and "
            f"{n_grid} geometric tick(s) found, but no pairs were within "
            f"{max_distance:.0f}px. Tick marks may be misaligned with their labels, "
            "or the labels are unusually offset.")


def _merge_record_sources(
    full_records: List[OCRRecord],
    y_band_records: List[OCRRecord],
    x_band_records: List[OCRRecord],
) -> List[OCRRecord]:
    """Merge OCR records, preferring band-phase records over full-phase ones.

    A band record displaces a full-phase record when their bboxes overlap
    (IoU > 0.3). This keeps the higher-precision numeric reading from the
    band scan and discards the (possibly-misread) full-scan record.
    """
    out: List[OCRRecord] = []
    band_records = list(y_band_records) + list(x_band_records)
    out.extend(band_records)
    for rec in full_records:
        displaced = any(_bbox_iou(rec.bbox, b.bbox) > 0.3 for b in band_records)
        if not displaced:
            out.append(rec)
    return out


def _compute_confidence(
    *,
    axis_conf: float,
    x_cal,
    y_cal,
    x_grid,
    y_grid,
) -> float:
    if x_cal is None or y_cal is None:
        return 0.0
    # Calibration confidence: more anchor points + smaller residuals is better.
    n_score = min(1.0, (x_cal.n_points + y_cal.n_points) / 8.0)
    # Pixel-RMSE relative to grid spacing should be small for a good fit.
    def _rel_residual(cal, grid):
        if cal is None or cal.rmse_px is None:
            return 1.0
        if grid.spacing <= 0:
            return 1.0
        return cal.rmse_px / grid.spacing
    rel_x = _rel_residual(x_cal, x_grid)
    rel_y = _rel_residual(y_cal, y_grid)
    fit_score = float(np.clip(1.0 - 0.5 * (rel_x + rel_y), 0.0, 1.0))
    grid_score = 0.5 * (1.0 if x_grid.success else 0.5) + 0.5 * (1.0 if y_grid.success else 0.5)
    return float(np.clip(0.45 * axis_conf + 0.30 * n_score + 0.15 * fit_score + 0.10 * grid_score,
                         0.0, 0.98))
