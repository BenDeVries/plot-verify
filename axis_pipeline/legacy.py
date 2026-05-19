"""Legacy-shape adaptor for `app_auto_axis.py`.

DEPRECATED: This module's dict-shaped entry points exist for backward
compatibility with the original Streamlit app and external scripts. New code
should use the typed API in `axis_pipeline` (`run_calibration`,
`manual_calibration`, `CalibrationResult`) and `update_result_from_tick_edits`
from this module instead.

The following functions are scheduled for removal in a future release:
    auto_detect_axes_and_ticks(img_bgr) -> dict
    auto_detect_axes_ticks_ocr(img_bgr, ...) -> dict
    build_diagnostic_overlay(img_bgr, det) -> RGB
    build_ocr_debug_overlay(img_bgr, det, ...) -> RGB
    update_detection_from_tick_tables(det, x_df, y_df)

The typed replacements (`run_calibration`, `manual_calibration`,
`render_overlay`, `update_result_from_tick_edits`) live alongside.
The serialization helpers `rebuild_result_from_detection` and
`CalibrationResult.to_legacy_dict` remain supported indefinitely.
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional

import pandas as pd

from .calibration import calibrate_axis
from .overlay import render_overlay
from .pipeline import run_calibration, _default_ocr_runner
from .types import (
    AxisFrame,
    CalibrationConfig,
    CalibrationResult,
    PairedTick,
)


# ----------------------------------------------------------------------
# Top-level legacy entry points
# ----------------------------------------------------------------------

# Once-per-process latch so a single Streamlit session doesn't emit the same
# warning thousands of times during widget reruns.
_deprecated_warned: set = set()


def _warn_legacy(name: str, replacement: str) -> None:
    if name in _deprecated_warned:
        return
    _deprecated_warned.add(name)
    warnings.warn(
        f"{name}() returns a dict-shaped detection and is deprecated. "
        f"Use {replacement} instead. This will be removed in a future release.",
        DeprecationWarning,
        stacklevel=3,
    )


def auto_detect_axes_and_ticks(img_bgr) -> Dict[str, object]:
    """Geometry-only path (no OCR) that still returns the legacy schema.

    .. deprecated::
        Use `axis_pipeline.run_calibration(img, ocr_runner=lambda *a, **k: [])`
        and consume the typed `CalibrationResult` directly.
    """
    _warn_legacy("auto_detect_axes_and_ticks",
                 "axis_pipeline.run_calibration() with a no-op ocr_runner")
    cfg = CalibrationConfig(
        enable_phase_b_y_band=False,
        enable_phase_c_x_band=False,
    )

    # OCR runner returns no records — equivalent to "OCR disabled".
    def _no_ocr(*args, **kwargs):
        return []

    result = run_calibration(img_bgr, config=cfg, ocr_runner=_no_ocr)
    d = result.to_legacy_dict()
    d["ocr_enabled"] = False
    return d


def auto_detect_axes_ticks_ocr(
    img_bgr,
    *,
    use_ocr: bool = True,
    mask_all_text: bool = True,    # kept for signature compatibility
    gpu: bool = False,
    min_ocr_confidence: float = 0.20,
) -> Dict[str, object]:
    """OCR-assisted path. If `use_ocr=False`, falls back to geometry-only.

    .. deprecated::
        Use `axis_pipeline.run_calibration(img, config=CalibrationConfig(...))`
        and consume the typed `CalibrationResult` directly.
    """
    _warn_legacy("auto_detect_axes_ticks_ocr",
                 "axis_pipeline.run_calibration()")
    if not use_ocr:
        d = auto_detect_axes_and_ticks(img_bgr)
        d["ocr_enabled"] = False
        return d

    cfg = CalibrationConfig(
        min_ocr_confidence=float(min_ocr_confidence),
        use_gpu=bool(gpu),
        enable_phase_b_y_band=True,
        enable_phase_c_x_band=True,
    )
    try:
        result = run_calibration(img_bgr, config=cfg)
        return result.to_legacy_dict()
    except Exception as e:
        # OCR engine missing or model download failed → fall back to geometry only.
        d = auto_detect_axes_and_ticks(img_bgr)
        d.setdefault("warnings", []).append(
            f"OCR pipeline failed; fell back to geometry-only detection. "
            f"Details: {type(e).__name__}: {e}"
        )
        d["ocr_enabled"] = False
        d["ocr_error"] = f"{type(e).__name__}: {e}"
        return d


# ----------------------------------------------------------------------
# Overlay shims (operate on legacy dicts)
# ----------------------------------------------------------------------

def build_diagnostic_overlay(img_bgr, detection: Optional[Dict[str, object]]):
    """Geometry-only diagnostic overlay (no OCR boxes).

    .. deprecated::
        Call `axis_pipeline.render_overlay(img_bgr, result)` with a typed
        `CalibrationResult` instead.
    """
    _warn_legacy("build_diagnostic_overlay", "axis_pipeline.render_overlay")
    if not detection:
        import cv2
        return cv2.cvtColor(img_bgr.copy(), cv2.COLOR_BGR2RGB)
    result = _legacy_dict_to_result(detection)
    return render_overlay(img_bgr, result, show_band_windows=False, show_grid_rejected=True)


def build_ocr_debug_overlay(
    img_bgr,
    detection: Optional[Dict[str, object]],
    *,
    show_mask: bool = True,    # kept for signature compatibility
    show_pairing: bool = True,
):
    """Full OCR + pairing diagnostic overlay.

    .. deprecated::
        Call `axis_pipeline.render_overlay(img_bgr, result, show_band_windows=True)`
        with a typed `CalibrationResult` instead.
    """
    _warn_legacy("build_ocr_debug_overlay", "axis_pipeline.render_overlay")
    if not detection:
        import cv2
        return cv2.cvtColor(img_bgr.copy(), cv2.COLOR_BGR2RGB)
    result = _legacy_dict_to_result(detection)
    return render_overlay(img_bgr, result,
                          show_band_windows=True,
                          show_grid_rejected=True)


# ----------------------------------------------------------------------
# Round-trip from edited Streamlit tables back into a result dict
# ----------------------------------------------------------------------

def update_result_from_tick_edits(
    result: CalibrationResult,
    x_edits,
    y_edits,
) -> CalibrationResult:
    """Apply edited tick tables to a typed CalibrationResult.

    This is the typed version of `update_detection_from_tick_tables`. Both
    functions share the same merge + recalibration logic; this one operates
    on a typed `CalibrationResult` and returns a typed `CalibrationResult`.
    Edits are merged by row index (additions/deletions in the table view are
    treated as edits to existing rows; a new row beyond ``len(orig)`` gets
    default fields and is generally not useful without pixel_position).
    """
    # Round-trip through the legacy dict adapter to reuse the merge logic.
    detection = result.to_legacy_dict()
    updated_dict = update_detection_from_tick_tables(detection, x_edits, y_edits)
    return rebuild_result_from_detection(updated_dict)


def update_detection_from_tick_tables(
    detection: Dict[str, object],
    x_df,
    y_df,
) -> Dict[str, object]:
    """Apply edited Streamlit tick tables to a legacy detection dict.

    Re-runs OLS calibration (with greedy outlier removal) on the edited pairs
    and updates the anchor points P1/P2/P3 + their data values. The input
    `detection` dict is not mutated; nested lists (e.g. p1/p2/p3) are copied
    before any modification.

    .. deprecated::
        Use `update_result_from_tick_edits(result, x_df, y_df)` which operates
        on typed `CalibrationResult` objects.
    """
    _warn_legacy("update_detection_from_tick_tables",
                 "update_result_from_tick_edits")
    out = dict(detection)
    # Copy mutable anchor lists so we never mutate the caller's data.
    for k in ("p1", "p2", "p3"):
        if out.get(k) is not None:
            out[k] = list(out[k])

    def _merge_edited(edited_df, orig_rows: list) -> list:
        """Overlay edited columns onto original rows, preserving hidden fields."""
        if edited_df is None:
            return [dict(r) for r in orig_rows]
        edited = pd.DataFrame(edited_df).to_dict("records")
        result = []
        for i, edits in enumerate(edited):
            base = dict(orig_rows[i]) if i < len(orig_rows) else {}
            base.update(edits)
            result.append(base)
        return result

    x_rows = _merge_edited(x_df, out.get("x_tick_table") or [])
    y_rows = _merge_edited(y_df, out.get("y_tick_table") or [])

    for rows in (x_rows, y_rows):
        for r in rows:
            r["include"] = bool(r.get("include", True))
            for k in ["value", "pixel_position", "fixed_axis_pixel", "ocr_confidence", "pair_distance_px"]:
                v = r.get(k)
                try:
                    r[k] = float(v) if v is not None else None
                except (TypeError, ValueError):
                    r[k] = None if k == "value" else float("nan")

    out["x_tick_table"] = x_rows
    out["y_tick_table"] = y_rows

    # Recompute calibration from edited tables.
    x_paired = [_row_to_paired(r) for r in x_rows]
    y_paired = [_row_to_paired(r) for r in y_rows]

    x_cal = calibrate_axis(x_paired)
    y_cal = calibrate_axis(y_paired)
    out["x_calibration"] = x_cal.to_dict() if x_cal else None
    out["y_calibration"] = y_cal.to_dict() if y_cal else None

    inc_x = [t for t in x_paired if t.include and t.data_value is not None]
    inc_y = [t for t in y_paired if t.include and t.data_value is not None]
    inc_x.sort(key=lambda t: t.pixel_position)
    inc_y.sort(key=lambda t: t.pixel_position)

    bbox_list = out.get("bbox")
    bbox = AxisFrame(*[int(v) for v in bbox_list]) if bbox_list else None

    if inc_x and bbox is not None:
        out["p1"] = [float(inc_x[0].pixel_position), float(bbox.bottom)]
        out["p2"] = [float(inc_x[-1].pixel_position), float(bbox.bottom)]
        out["p1_data_x"] = float(inc_x[0].data_value)
        out["p2_data_x"] = float(inc_x[-1].data_value)

    if inc_y and bbox is not None:
        topmost = min(inc_y, key=lambda t: t.pixel_position)
        bottommost = max(inc_y, key=lambda t: t.pixel_position)
        p3_x = float(out["p1"][0]) if out.get("p1") else float(bbox.left)
        out["p3"] = [p3_x, float(topmost.pixel_position)]
        out["p3_data_y"] = float(topmost.data_value)
        out["p1_data_y"] = float(bottommost.data_value)
        if out.get("p1"):
            out["p1"] = [out["p1"][0], float(bottommost.pixel_position)]
        if out.get("p2"):
            out["p2"] = [out["p2"][0], float(bottommost.pixel_position)]

    if out.get("p3") and x_cal is not None:
        out["p3_data_x"] = float(x_cal.scale * out["p3"][0] + x_cal.offset)

    return out


def rebuild_result_from_detection(
    detection: Dict[str, object],
) -> "CalibrationResult":
    """Reconstruct a typed CalibrationResult from a (possibly edited) legacy dict.

    Used by the Streamlit app after `update_detection_from_tick_tables` so the
    diagnostic overlay (which renders from `auto_axis_result`) reflects the
    user's edits. Carries grid-fit and calibration objects through alongside
    the paired ticks so the rendered overlay shows the same anchors and
    markers as the legacy dict's tables.
    """
    base = _legacy_dict_to_result(detection)

    # _legacy_dict_to_result intentionally leaves x_calibration/y_calibration
    # as None (the overlay renderer doesn't need them). Carry the recomputed
    # ones across so consumers that DO need them (e.g. derived P3.data_x) work.
    x_cal_d = detection.get("x_calibration")
    y_cal_d = detection.get("y_calibration")
    if x_cal_d:
        base.x_calibration = _axis_calibration_from_dict(x_cal_d)
    if y_cal_d:
        base.y_calibration = _axis_calibration_from_dict(y_cal_d)

    # Grid-fit objects, if present in the legacy dict.
    x_grid_d = detection.get("x_grid_fit")
    y_grid_d = detection.get("y_grid_fit")
    if x_grid_d:
        base.x_grid_fit = _grid_fit_from_dict(x_grid_d)
    if y_grid_d:
        base.y_grid_fit = _grid_fit_from_dict(y_grid_d)

    return base


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _row_to_paired(r: dict) -> PairedTick:
    return PairedTick(
        pixel_position=float(r.get("pixel_position", 0.0)),
        fixed_axis_pixel=float(r.get("fixed_axis_pixel", 0.0)),
        data_value=float(r["value"]) if r.get("value") is not None else float("nan"),
        pair_distance_px=float(r.get("pair_distance_px") or float("nan")),
        grid_index=r.get("grid_index"),
        label_bbox=tuple(r.get("bbox") or [0, 0, 0, 0]),
        raw_text=str(r.get("raw_text", "")),
        cleaned_text=str(r.get("cleaned_text", "")),
        ocr_confidence=float(r.get("ocr_confidence") or 0.0),
        parse_status=str(r.get("parse_status", "")),
        flag=str(r.get("flag", "")),
        include=bool(r.get("include", True)),
        status=str(r.get("status", "paired_to_tick_mark")),
    )


def _legacy_dict_to_result(d: Dict[str, object]) -> CalibrationResult:
    """Best-effort reconstruction of CalibrationResult from a legacy detection dict.

    Used by overlay rendering — it doesn't need the full pipeline state, just
    enough geometry + tables + anchors to draw the diagnostic image.
    """
    bbox_list = d.get("bbox")
    bbox = AxisFrame(*[int(v) for v in bbox_list]) if bbox_list else None
    x_paired = [_row_to_paired(r) for r in (d.get("x_tick_table") or [])]
    y_paired = [_row_to_paired(r) for r in (d.get("y_tick_table") or [])]

    from .types import OCRRecord
    ocr_records = []
    for r in (d.get("ocr_records") or []):
        try:
            ocr_records.append(OCRRecord(
                raw_text=r.get("raw_text", ""),
                cleaned_text=r.get("cleaned_text", ""),
                value=r.get("value"),
                is_numeric=bool(r.get("is_numeric", r.get("value") is not None)),
                confidence=float(r.get("ocr_confidence", 0.0) or 0.0),
                bbox=tuple(r.get("bbox") or [0, 0, 0, 0]),
                center=(float(r.get("center_x", 0.0)), float(r.get("center_y", 0.0))),
                parse_status=r.get("parse_status", ""),
                parse_flag=r.get("flag", ""),
                phase=r.get("phase", "full"),
            ))
        except Exception:
            continue

    p1 = tuple(d["p1"]) if d.get("p1") else None
    p2 = tuple(d["p2"]) if d.get("p2") else None
    p3 = tuple(d["p3"]) if d.get("p3") else None

    x_geom = [float(t[0]) for t in (d.get("x_ticks") or []) if len(t) >= 2]
    y_geom = [float(t[1]) for t in (d.get("y_ticks") or []) if len(t) >= 2]

    return CalibrationResult(
        success=bool(d.get("success", True)),
        confidence=float(d.get("confidence", 0.0) or 0.0),
        mode=str(d.get("mode", "unknown")),
        bbox=bbox,
        x_calibration=None,
        y_calibration=None,
        x_paired_ticks=x_paired,
        y_paired_ticks=y_paired,
        x_geometric_ticks=x_geom,
        y_geometric_ticks=y_geom,
        p1_pixel=p1, p2_pixel=p2, p3_pixel=p3,
        p1_data_x=d.get("p1_data_x"),
        p2_data_x=d.get("p2_data_x"),
        p3_data_x=d.get("p3_data_x"),
        p1_data_y=d.get("p1_data_y"),
        p3_data_y=d.get("p3_data_y"),
        ocr_records=ocr_records,
        warnings=list(d.get("warnings") or []),
        diagnostics=dict(d.get("diagnostics") or {}),
        config=CalibrationConfig(),
    )


def _axis_calibration_from_dict(d: Dict[str, object]):
    """Reconstruct an AxisCalibration from its `to_dict()` output."""
    from .types import AxisCalibration
    return AxisCalibration(
        scale=float(d.get("scale", 0.0) or 0.0),
        offset=float(d.get("offset", 0.0) or 0.0),
        n_points=int(d.get("n_points", 0) or 0),
        method=str(d.get("method", "ols")),
        rmse_px=float(d.get("rmse_px", 0.0) or 0.0),
        rmse_data=float(d.get("rmse_data", 0.0) or 0.0),
        log_likelihood=d.get("log_likelihood"),
        slope_se=d.get("slope_se"),
        offset_se=d.get("offset_se"),
        df_t=d.get("df_t"),
        log_base=d.get("log_base"),
    )


def _grid_fit_from_dict(d: Dict[str, object]):
    """Reconstruct a GridFit from its `to_dict()` output."""
    from .types import GridFit
    return GridFit(
        spacing=float(d.get("spacing", 0.0) or 0.0),
        origin=float(d.get("origin", 0.0) or 0.0),
        fitted_positions=[float(x) for x in (d.get("fitted_positions") or [])],
        fitted_indices=[int(i) for i in (d.get("fitted_indices") or [])],
        rejected_positions=[float(x) for x in (d.get("rejected_positions") or [])],
        grid_residuals=[float(x) for x in (d.get("grid_residuals") or [])],
        n_grid_cells=int(d.get("n_grid_cells", 0) or 0),
        success=bool(d.get("success", False)),
    )
