"""Diagnostic overlay rendering for `CalibrationResult`.

Shows everything the pipeline saw and decided:
    - OCR text boxes (numeric vs non-numeric, by phase)
    - Detected plot frame
    - Geometric tick positions before grid-fit
    - Grid-fit kept positions (green) and rejected positions (gray)
    - Paired ticks with their label-to-tick connector lines
    - Calibration anchors P1, P2, P3 (yellow stars)
    - The y-band and x-band OCR re-scan windows (light shading)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .types import AxisFrame, CalibrationResult, OCRPhase, OCRRecord
from .ocr import x_label_band, y_label_band


def render_overlay(
    img_bgr: np.ndarray,
    result: Optional[CalibrationResult],
    *,
    show_band_windows: bool = True,
    show_grid_rejected: bool = True,
) -> np.ndarray:
    overlay = img_bgr.copy()
    if result is None or result.bbox is None:
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    bbox = result.bbox

    # Phase B/C scan windows (light shading).
    if show_band_windows:
        layer = overlay.copy()
        if result.config and result.config.enable_phase_b_y_band:
            yb = y_label_band(bbox, extra_left=result.config.y_band_extra_px)
            cv2.rectangle(layer, (yb[0], yb[1]), (yb[2], yb[3]), (220, 240, 200), -1)
        if result.config and result.config.enable_phase_c_x_band:
            xb = x_label_band(bbox, extra_below=result.config.x_band_extra_px)
            cv2.rectangle(layer, (xb[0], xb[1]), (xb[2], xb[3]), (200, 220, 240), -1)
        overlay = cv2.addWeighted(layer, 0.20, overlay, 0.80, 0)

    # OCR text boxes.
    for rec in result.ocr_records:
        x0, y0, x1, y1 = [int(v) for v in rec.bbox]
        if rec.phase == OCRPhase.Y_BAND.value or rec.phase == OCRPhase.X_BAND.value:
            color = (0, 200, 0) if rec.is_numeric else (140, 140, 140)
        else:
            color = (0, 140, 0) if rec.is_numeric else (170, 170, 170)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 1)

    # Plot frame.
    cv2.rectangle(overlay, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0, 0, 200), 2)

    # Geometric ticks (raw): light blue.
    if show_grid_rejected:
        for x in result.x_geometric_ticks:
            cv2.drawMarker(overlay, (int(round(x)), int(bbox.bottom)),
                           (210, 180, 100), markerType=cv2.MARKER_TILTED_CROSS,
                           markerSize=8, thickness=1)
        for y in result.y_geometric_ticks:
            cv2.drawMarker(overlay, (int(bbox.left), int(round(y))),
                           (210, 180, 100), markerType=cv2.MARKER_TILTED_CROSS,
                           markerSize=8, thickness=1)

    # Grid-fit kept positions: bright green.
    if result.x_grid_fit:
        for x in result.x_grid_fit.fitted_positions:
            cv2.drawMarker(overlay, (int(round(x)), int(bbox.bottom)),
                           (0, 255, 0), markerType=cv2.MARKER_CROSS,
                           markerSize=14, thickness=2)
    if result.y_grid_fit:
        for y in result.y_grid_fit.fitted_positions:
            cv2.drawMarker(overlay, (int(bbox.left), int(round(y))),
                           (0, 255, 0), markerType=cv2.MARKER_CROSS,
                           markerSize=14, thickness=2)

    # Pairings: orange line from label center to tick.
    for t in result.x_paired_ticks:
        if not t.include:
            continue
        bb = t.label_bbox
        cx = int(round((bb[0] + bb[2]) / 2)); cy = int(round((bb[1] + bb[3]) / 2))
        tx = int(round(t.pixel_position)); ty = int(round(t.fixed_axis_pixel))
        cv2.line(overlay, (cx, cy), (tx, ty), (0, 128, 255), 1)
        cv2.drawMarker(overlay, (tx, ty), (0, 128, 255),
                       markerType=cv2.MARKER_DIAMOND, markerSize=10, thickness=2)
    for t in result.y_paired_ticks:
        if not t.include:
            continue
        bb = t.label_bbox
        cx = int(round((bb[0] + bb[2]) / 2)); cy = int(round((bb[1] + bb[3]) / 2))
        tx = int(round(t.fixed_axis_pixel)); ty = int(round(t.pixel_position))
        cv2.line(overlay, (cx, cy), (tx, ty), (255, 128, 0), 1)
        cv2.drawMarker(overlay, (tx, ty), (255, 128, 0),
                       markerType=cv2.MARKER_DIAMOND, markerSize=10, thickness=2)

    # Anchors P1, P2, P3 (magenta stars + labels).
    for label, pt in (("P1", result.p1_pixel), ("P2", result.p2_pixel), ("P3", result.p3_pixel)):
        if pt is None:
            continue
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.drawMarker(overlay, (x, y), (255, 0, 255),
                       markerType=cv2.MARKER_STAR, markerSize=22, thickness=2)
        cv2.putText(overlay, label, (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def render_band_preview(
    img_bgr: np.ndarray,
    bbox: AxisFrame,
    y_band: Tuple[int, int, int, int],
    x_band: Tuple[int, int, int, int],
    *,
    phase_a_records: Optional[List[OCRRecord]] = None,
) -> np.ndarray:
    """Lightweight preview overlay for the manual-band UI.

    Draws only what the user needs to decide whether the bands are right:
    the source image, the bbox outline, the y-band rectangle (left of axis),
    the x-band rectangle (below axis), and — optionally — Phase A OCR record
    outlines color-coded by `is_numeric` so the user can see why the bands
    might miss labels.

    No tick marks, no calibration anchors, no pairing lines. For the full
    diagnostic image after calibration, use `render_overlay`.

    Returns RGB (matches `render_overlay`).
    """
    overlay = img_bgr.copy()
    h, w = overlay.shape[:2]

    # Y-band shading (greenish — same hue render_overlay uses for y_label_band).
    layer = overlay.copy()
    yx0, yy0, yx1, yy1 = (max(0, int(v)) for v in (y_band[0], y_band[1], y_band[2], y_band[3]))
    yx1 = min(w, yx1); yy1 = min(h, yy1)
    if yx1 > yx0 and yy1 > yy0:
        cv2.rectangle(layer, (yx0, yy0), (yx1, yy1), (220, 240, 200), -1)
    # X-band shading (bluish — same hue render_overlay uses for x_label_band).
    xx0, xy0, xx1, xy1 = (max(0, int(v)) for v in (x_band[0], x_band[1], x_band[2], x_band[3]))
    xx1 = min(w, xx1); xy1 = min(h, xy1)
    if xx1 > xx0 and xy1 > xy0:
        cv2.rectangle(layer, (xx0, xy0), (xx1, xy1), (200, 220, 240), -1)
    overlay = cv2.addWeighted(layer, 0.30, overlay, 0.70, 0)

    # Band outlines on top of shading for crisp edges.
    if yx1 > yx0 and yy1 > yy0:
        cv2.rectangle(overlay, (yx0, yy0), (yx1, yy1), (0, 180, 0), 2)
    if xx1 > xx0 and xy1 > xy0:
        cv2.rectangle(overlay, (xx0, xy0), (xx1, xy1), (200, 100, 0), 2)

    # Phase A OCR record outlines — show the user *what* the OCR found.
    # Numeric records get a stronger green; non-numeric records a muted gray
    # so the user's eye is drawn to the labels that matter for calibration.
    if phase_a_records:
        for rec in phase_a_records:
            x0, y0, x1, y1 = (int(v) for v in rec.bbox)
            color = (0, 140, 0) if rec.is_numeric else (170, 170, 170)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 1)

    # Plot frame outline (red, slightly thicker so the user can always find it).
    cv2.rectangle(overlay, (bbox.left, bbox.top), (bbox.right, bbox.bottom),
                  (0, 0, 200), 2)

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
