"""Legacy-shape adaptor for `app_auto_axis.py`.

The existing Streamlit app calls into:
    auto_detect_axes_and_ticks(img_bgr) -> dict          # axis_auto.py
    auto_detect_axes_ticks_ocr(img_bgr, ...) -> dict     # ocr_axis.py
    build_diagnostic_overlay(img_bgr, det) -> RGB        # axis_auto.py
    build_ocr_debug_overlay(img_bgr, det, ...) -> RGB    # ocr_axis.py
    update_detection_from_tick_tables(det, x_df, y_df)   # ocr_axis.py
    parse_numeric_tick(text)                             # ocr_axis.py

This module reproduces those entry points, delegating to the new pipeline.
The shim in `axis_auto.py` and `ocr_axis.py` re-exports from here.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
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

def auto_detect_axes_and_ticks(img_bgr) -> Dict[str, object]:
    """Geometry-only path (no OCR) that still returns the legacy schema."""
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
    """OCR-assisted path. If `use_ocr=False`, falls back to geometry-only."""
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
    """Geometry-only diagnostic overlay (no OCR boxes)."""
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
    """Full OCR + pairing diagnostic overlay."""
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

def update_detection_from_tick_tables(
    detection: Dict[str, object],
    x_df,
    y_df,
) -> Dict[str, object]:
    """Apply edited Streamlit tick tables to a legacy detection dict.

    Re-runs OLS calibration (with Cook's distance filtering) on the edited
    pairs and updates the anchor points P1/P2/P3 + their data values.
    """
    out = dict(detection)

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
            out["p1"][1] = float(bottommost.pixel_position)
        if out.get("p2"):
            out["p2"][1] = float(bottommost.pixel_position)

    if out.get("p3") and x_cal is not None:
        out["p3_data_x"] = float(x_cal.scale * out["p3"][0] + x_cal.offset)

    return out


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
