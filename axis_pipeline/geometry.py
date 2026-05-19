"""Geometric detection of plot axes and tick marks.

The input image should already have OCR text masked out (whitened). With text
gone, the dark structural ink of the plot — axes, tick marks, gridlines, data
series — dominates the projection profile, making axis lines and tick locations
much easier to find by 1-D peak detection.

Two passes find candidate axis lines:
- Row/column projection of a low-saturation low-value mask
- Probabilistic Hough on Canny edges of the same mask
Candidates from both passes are deduplicated and then a "bottom + left" pair
is picked as the canonical x-axis / y-axis. An optional top/right border is
detected for boxed-frame plots.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.signal import find_peaks

from .types import AxisFrame, CalibrationConfig


@dataclass
class _AxisCandidate:
    orientation: str   # "h" or "v"
    pos: float         # row (h) or column (v) of the line
    start: float
    end: float
    length: float
    strength: float
    source: str        # "projection" or "hough"


# ----------------------------------------------------------------------
# Image preparation
# ----------------------------------------------------------------------

def prepare_dark_mask(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (gray, dark_mask). `dark_mask` is uint8 with 1 at dark structural ink.

    Tuned for light-background figures with black/dark-gray axes and ticks.
    Avoids erosion/opening so that 1-pixel matplotlib spines survive intact.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    dark = ((gray < 185) & (sat < 110) & (val < 210)).astype(np.uint8)
    return gray, dark


# ----------------------------------------------------------------------
# Axis line candidate detection
# ----------------------------------------------------------------------

def _longest_run(mask_1d: np.ndarray) -> Tuple[int, int, int]:
    if mask_1d.size == 0 or not np.any(mask_1d):
        return 0, 0, 0
    padded = np.r_[False, mask_1d.astype(bool), False]
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1
    lens = ends - starts + 1
    i = int(np.argmax(lens))
    return int(starts[i]), int(ends[i]), int(lens[i])


def _covered_span(mask_1d: np.ndarray, min_coverage: float = 0.35) -> Tuple[int, int, int]:
    """Return (start, end, span) using full first-to-last dark pixel extent.

    When text masking creates multiple gaps in an axis line the longest
    continuous run can fall below the minimum-length threshold even though the
    axis is clearly present.  If at least `min_coverage` fraction of the
    first-to-last span is dark we report the full span rather than the longest
    continuous run, so masking gaps do not hide the axis.
    """
    if mask_1d.size == 0 or not np.any(mask_1d):
        return 0, 0, 0
    dark_idx = np.where(mask_1d.astype(bool))[0]
    start, end = int(dark_idx[0]), int(dark_idx[-1])
    span = end - start + 1
    coverage = len(dark_idx) / span if span > 0 else 0.0
    if coverage >= min_coverage:
        return start, end, span
    return _longest_run(mask_1d)


def _group_runs(indices: np.ndarray) -> List[np.ndarray]:
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1) + 1
    return [g for g in np.split(indices, breaks) if g.size]


def _dedup_positions(positions: List[float], tolerance: float) -> List[float]:
    """Merge positions within `tolerance` px, keeping the median of each cluster."""
    if not positions:
        return []
    positions = sorted(positions)
    clusters: List[List[float]] = [[positions[0]]]
    for p in positions[1:]:
        if p - clusters[-1][-1] <= tolerance:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [float(np.median(c)) for c in clusters]


def _projection_candidates(dark: np.ndarray) -> Tuple[List[_AxisCandidate], List[_AxisCandidate]]:
    h, w = dark.shape[:2]
    rowc = dark.sum(axis=1)
    colc = dark.sum(axis=0)
    row_thr = max(0.18 * w, np.percentile(rowc, 99) * 0.45)
    col_thr = max(0.18 * h, np.percentile(colc, 99) * 0.45)

    horizontal: List[_AxisCandidate] = []
    for grp in _group_runs(np.flatnonzero(rowc >= row_thr)):
        rep = int(grp[np.argmax(rowc[grp])])
        band = dark[max(0, rep - 1): min(h, rep + 2), :].sum(axis=0) > 0
        x0, x1, length = _covered_span(band)
        if length >= 0.20 * w:
            horizontal.append(_AxisCandidate("h", rep, x0, x1, length,
                                             float(rowc[rep]), "projection"))

    vertical: List[_AxisCandidate] = []
    for grp in _group_runs(np.flatnonzero(colc >= col_thr)):
        rep = int(grp[np.argmax(colc[grp])])
        band = dark[:, max(0, rep - 1): min(w, rep + 2)].sum(axis=1) > 0
        y0, y1, length = _covered_span(band)
        if length >= 0.20 * h:
            vertical.append(_AxisCandidate("v", rep, y0, y1, length,
                                           float(colc[rep]), "projection"))

    return horizontal, vertical


def _hough_candidates(gray: np.ndarray, dark: np.ndarray) -> Tuple[List[_AxisCandidate], List[_AxisCandidate]]:
    h, w = gray.shape[:2]
    edges = cv2.Canny((dark * 255).astype(np.uint8), 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1, theta=np.pi / 180,
        threshold=max(40, int(min(h, w) * 0.08)),
        minLineLength=int(min(h, w) * 0.20),
        maxLineGap=12,
    )
    horizontal: List[_AxisCandidate] = []
    vertical: List[_AxisCandidate] = []
    if lines is None:
        return horizontal, vertical
    for ln in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, ln)
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length < min(h, w) * 0.20:
            continue
        ang = np.degrees(np.arctan2(dy, dx))
        if abs(ang) <= 4 or abs(abs(ang) - 180) <= 4:
            y = (y1 + y2) / 2.0
            horizontal.append(_AxisCandidate("h", y, min(x1, x2), max(x1, x2),
                                             length, length, "hough"))
        elif abs(abs(ang) - 90) <= 4:
            x = (x1 + x2) / 2.0
            vertical.append(_AxisCandidate("v", x, min(y1, y2), max(y1, y2),
                                           length, length, "hough"))
    return horizontal, vertical


def _dedupe(cands: List[_AxisCandidate], tol: float = 4.0) -> List[_AxisCandidate]:
    if not cands:
        return []
    cands = sorted(cands, key=lambda c: c.pos)
    groups: List[List[_AxisCandidate]] = []
    for c in cands:
        if not groups or abs(c.pos - np.mean([g.pos for g in groups[-1]])) > tol:
            groups.append([c])
        else:
            groups[-1].append(c)
    return [max(g, key=lambda c: (c.length, c.strength)) for g in groups]


def _choose_axes(
    horizontal: List[_AxisCandidate],
    vertical: List[_AxisCandidate],
    shape,
) -> Tuple[Optional[AxisFrame], str, float, List[str]]:
    h, w = shape[:2]
    warnings: List[str] = []
    horizontal = [c for c in horizontal if c.length >= 0.25 * w]
    vertical = [c for c in vertical if c.length >= 0.25 * h]
    if not horizontal or not vertical:
        return None, "failed", 0.0, ["Could not find both horizontal and vertical axis candidates."]

    bottom_options = [c for c in horizontal if c.pos > 0.35 * h] or horizontal
    left_options = [c for c in vertical if c.pos < 0.65 * w] or vertical
    bottom = max(bottom_options, key=lambda c: (c.pos, c.length))
    # Drop weak left candidates before applying the leftmost rule: a thin
    # vertical artifact (unmasked rotated y-axis title strokes, legend-frame
    # edges) can sit further left than the real y-axis and would otherwise win
    # `min(pos)` despite being much shorter. Real plot axes are at least ~60%
    # of the longest left-side vertical run.
    max_left_length = max(c.length for c in left_options)
    strong_left = [c for c in left_options if c.length >= 0.6 * max_left_length]
    left = min(strong_left, key=lambda c: (c.pos, -c.length))

    top_options = [c for c in horizontal
                   if c.pos < bottom.pos - 0.15 * h
                   and c.start <= left.pos + 20
                   and c.end >= bottom.end - 40]
    right_options = [c for c in vertical
                     if c.pos > left.pos + 0.15 * w
                     and c.start <= bottom.pos - 0.15 * h
                     and c.end >= bottom.pos - 20]
    top = min(top_options, key=lambda c: c.pos) if top_options else None
    right = max(right_options, key=lambda c: c.pos) if right_options else None

    x_left = int(round(left.pos))
    y_bottom = int(round(bottom.pos))
    x_right = int(round(right.pos)) if right is not None else int(round(bottom.end))
    y_top = int(round(top.pos)) if top is not None else int(round(left.start))

    if right is None:
        warnings.append("Right plot border not confidently detected; using x-axis extent as the right edge.")
    if top is None:
        warnings.append("Top plot border not confidently detected; using y-axis extent as the top edge.")

    x_left = int(np.clip(x_left, 0, w - 2))
    x_right = int(np.clip(x_right, x_left + 5, w - 1))
    y_top = int(np.clip(y_top, 0, h - 2))
    y_bottom = int(np.clip(y_bottom, y_top + 5, h - 1))

    width = x_right - x_left
    height = y_bottom - y_top
    area_frac = (width * height) / float(w * h)
    if area_frac < 0.08 or area_frac > 0.95:
        warnings.append("Detected plot area is unusual; please review.")
    boxed = top is not None and right is not None
    mode = "boxed" if boxed else "open_axes"

    x_len_score = min(1.0, bottom.length / max(1.0, width))
    y_len_score = min(1.0, left.length / max(1.0, height))
    area_score = 1.0 if 0.15 <= area_frac <= 0.85 else 0.70
    box_bonus = 0.12 if boxed else 0.0
    confidence = float(np.clip(0.36 * x_len_score + 0.36 * y_len_score
                               + 0.16 * area_score + box_bonus, 0.0, 0.98))

    return AxisFrame(x_left, y_top, x_right, y_bottom), mode, confidence, warnings


def detect_axes(img_bgr: np.ndarray) -> Tuple[Optional[AxisFrame], str, float, List[str]]:
    """Return (frame, mode, confidence, warnings).

    Pass the image with text already masked out for best results.
    """
    gray, dark = prepare_dark_mask(img_bgr)
    h_proj, v_proj = _projection_candidates(dark)
    h_hough, v_hough = _hough_candidates(gray, dark)
    horizontal = _dedupe(h_proj + h_hough)
    vertical = _dedupe(v_proj + v_hough)
    return _choose_axes(horizontal, vertical, gray.shape)


# ----------------------------------------------------------------------
# Tick mark detection
# ----------------------------------------------------------------------

def detect_x_tick_positions(
    dark: np.ndarray,
    bbox: AxisFrame,
    *,
    config: Optional[CalibrationConfig] = None,
) -> Tuple[List[float], List[float], Dict[str, object]]:
    """Detect x-axis tick positions from outward (below) and inward (above) bands.

    Returns (outward_positions, inward_positions, diagnostics_dict).
    Pipeline uses these separately to try outward first, then inward, then merged.
    """
    h, w = dark.shape[:2]
    bx0 = max(0, bbox.left)
    bx1 = min(w, bbox.right + 1)
    min_dist = max(8, int((bbox.right - bbox.left) * 0.025))

    def _band_peaks(y0: int, y1: int) -> List[float]:
        band = dark[y0:y1, bx0:bx1]
        if band.size == 0:
            return []
        proj = band.sum(axis=0).astype(float)
        signal = np.maximum(0, proj - np.percentile(proj, 55))
        height_thresh = max(1.0, np.max(signal) * 0.25) if np.max(signal) > 0 else 1.0
        peaks, _ = find_peaks(signal, height=height_thresh, distance=min_dist)
        return [float(bx0 + p) for p in peaks if bbox.left - 5 <= bx0 + p <= bbox.right + 5]

    # Outward band: below the axis line
    outward_depth = max(10, int(0.025 * h))
    outward = _band_peaks(max(0, bbox.bottom), min(h, bbox.bottom + outward_depth))

    # Inward band: above the axis line (into the plot interior)
    detect_inward = config is None or getattr(config, "detect_inward_ticks", True)
    inward: List[float] = []
    if detect_inward:
        depth_frac = getattr(config, "inward_tick_depth_frac", 0.025) if config else 0.025
        min_depth = getattr(config, "inward_tick_min_depth_px", 10) if config else 10
        plot_h = max(1, bbox.bottom - bbox.top)
        inward_depth = max(min_depth, int(depth_frac * plot_h))
        iy0 = max(bbox.top, bbox.bottom - inward_depth)
        iy1 = bbox.bottom  # exclusive — axis line row not included
        inward = _band_peaks(iy0, iy1)

    diag: Dict[str, object] = {
        "x_tick_candidates_outward": len(outward),
        "x_tick_candidates_inward": len(inward),
    }
    return outward, inward, diag


def detect_y_tick_positions(
    dark: np.ndarray,
    bbox: AxisFrame,
    *,
    config: Optional[CalibrationConfig] = None,
) -> Tuple[List[float], List[float], Dict[str, object]]:
    """Detect y-axis tick positions from outward (left) and inward (right) bands.

    Returns (outward_positions, inward_positions, diagnostics_dict).
    Pipeline uses these separately to try outward first, then inward, then merged.
    """
    h, w = dark.shape[:2]
    by0 = max(0, bbox.top)
    by1 = min(h, bbox.bottom + 1)
    min_dist = max(8, int((bbox.bottom - bbox.top) * 0.025))

    def _band_peaks(x0: int, x1: int) -> List[float]:
        band = dark[by0:by1, x0:x1]
        if band.size == 0:
            return []
        proj = band.sum(axis=1).astype(float)
        signal = np.maximum(0, proj - np.percentile(proj, 55))
        height_thresh = max(1.0, np.max(signal) * 0.25) if np.max(signal) > 0 else 1.0
        peaks, _ = find_peaks(signal, height=height_thresh, distance=min_dist)
        return [float(by0 + p) for p in peaks if bbox.top - 5 <= by0 + p <= bbox.bottom + 5]

    # Outward band: left of the y-axis line
    outward_depth = max(10, int(0.025 * w))
    outward = _band_peaks(max(0, bbox.left - outward_depth), min(w, bbox.left + 1))

    # Inward band: right of the y-axis line (into the plot interior)
    detect_inward = config is None or getattr(config, "detect_inward_ticks", True)
    inward: List[float] = []
    if detect_inward:
        depth_frac = getattr(config, "inward_tick_depth_frac", 0.025) if config else 0.025
        min_depth = getattr(config, "inward_tick_min_depth_px", 10) if config else 10
        plot_w = max(1, bbox.right - bbox.left)
        inward_depth = max(min_depth, int(depth_frac * plot_w))
        ix0 = bbox.left
        ix1 = min(bbox.right, bbox.left + inward_depth)
        inward = _band_peaks(ix0, ix1)

    diag: Dict[str, object] = {
        "y_tick_candidates_outward": len(outward),
        "y_tick_candidates_inward": len(inward),
    }
    return outward, inward, diag
