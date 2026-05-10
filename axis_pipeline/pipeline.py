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
from typing import Callable, List, Optional, Tuple

import numpy as np

from . import geometry as geom
from . import ocr as ocr_mod
from . import pairing as pair_mod
from .calibration import calibrate_axis
from .gridfit import fit_linear_grid
from .types import (
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


# Type alias: (image_bgr, allowlist, phase, bbox_offset, kwargs) -> records.
# Injectable so tests can swap in a tesseract shim instead of EasyOCR.
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
    masked = ocr_mod.mask_records(img_bgr, full_records, pad=cfg.ocr_pad_px)
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
    x_tick_positions = geom.detect_x_tick_positions(dark, bbox)
    y_tick_positions = geom.detect_y_tick_positions(dark, bbox)
    diagnostics["x_tick_raw_count"] = len(x_tick_positions)
    diagnostics["y_tick_raw_count"] = len(y_tick_positions)

    # ── Grid fit (linear) ─────────────────────────────────────────
    if cfg.scale_type == ScaleType.LINEAR.value:
        x_grid = fit_linear_grid(
            x_tick_positions,
            tolerance_frac=cfg.grid_residual_tolerance_frac,
            min_ticks=cfg.grid_min_ticks,
        )
        y_grid = fit_linear_grid(
            y_tick_positions,
            tolerance_frac=cfg.grid_residual_tolerance_frac,
            min_ticks=cfg.grid_min_ticks,
        )
    else:
        warnings.append(f"Scale type {cfg.scale_type!r} is not yet implemented; "
                        "treating as linear.")
        x_grid = fit_linear_grid(x_tick_positions,
                                 tolerance_frac=cfg.grid_residual_tolerance_frac,
                                 min_ticks=cfg.grid_min_ticks)
        y_grid = fit_linear_grid(y_tick_positions,
                                 tolerance_frac=cfg.grid_residual_tolerance_frac,
                                 min_ticks=cfg.grid_min_ticks)
    diagnostics["x_grid_kept"] = len(x_grid.fitted_positions)
    diagnostics["y_grid_kept"] = len(y_grid.fitted_positions)
    diagnostics["x_grid_rejected"] = len(x_grid.rejected_positions)
    diagnostics["y_grid_rejected"] = len(y_grid.rejected_positions)

    # ── Pairing ───────────────────────────────────────────────────
    x_label_records = pair_mod.filter_x_axis_labels(combined_records, bbox)
    y_label_records = pair_mod.filter_y_axis_labels(combined_records, bbox)
    diagnostics["x_label_candidates"] = len(x_label_records)
    diagnostics["y_label_candidates"] = len(y_label_records)

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
    x_cal = calibrate_axis(
        x_paired,
        use_robust=cfg.use_robust_regression,
        student_t_df=cfg.student_t_df,
    )
    y_cal = calibrate_axis(
        y_paired,
        use_robust=cfg.use_robust_regression,
        student_t_df=cfg.student_t_df,
    )

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
        compute_band = lambda extra: ocr_mod.x_label_band(
            bbox, extra_below=extra, extra_horizontal=horiz,
        )
    elif axis == "y":
        narrow_extra = cfg.y_band_extra_px
        wide_extra = cfg.y_band_fallback_extra_px
        vert = cfg.y_band_extra_vertical_px
        phase_name = OCRPhase.Y_BAND.value
        compute_band = lambda extra: ocr_mod.y_label_band(
            bbox, extra_left=extra, extra_vertical=vert,
        )
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
            allowlist="0123456789.+-eE",
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
