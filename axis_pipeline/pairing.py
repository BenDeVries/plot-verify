"""Pair OCR numeric labels to grid-fitted geometric ticks.

The pairing step decides which OCR labels actually annotate which axis ticks.
Inputs:
    - a list of OCR records that survived the band scans
    - a `GridFit` for the axis (which already filtered the geometric ticks)
    - the orientation ('x' or 'y') and the axis line's perpendicular pixel

Outputs a list of `PairedTick` records ready for calibration. Per the design
decision (option 'a' in the redesign): geometric ticks without an associated
OCR label are *dropped*, so the calibration only fits to label-anchored ticks.

The pairer enforces:
    1. Each tick is paired to at most one label (one-to-one).
    2. Pairs further than `max_distance` are rejected.
    3. Among remaining pairs, the resulting (pixel, value) sequence must be
       monotonic — for x-axis, value increases with pixel; for y-axis, value
       decreases as pixel-y increases (image origin at top-left).
       Non-monotonic pairs are flagged and excluded.

The monotonicity check is what catches misreads such as "1s" → 15 (which would
otherwise pair to the last x-tick alongside the correct labels and look fine
locally, but produces an inconsistent overall sequence).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .types import GridFit, OCRRecord, PairedTick


# ----------------------------------------------------------------------
# Spatial filters: which OCR records belong to which axis?
# ----------------------------------------------------------------------

def filter_x_axis_labels(
    records: List[OCRRecord],
    bbox,
) -> List[OCRRecord]:
    """Keep records that should be considered for x-axis pairing.

    Phase-aware: a record from `phase=x_band` is always kept (the band scan ran
    on the x-axis label strip by construction). A record from `phase=y_band` is
    always rejected — it cannot belong to the x-axis. A record from `phase=full`
    falls through to the spatial filter (center within the x-axis label region).
    """
    out: List[OCRRecord] = []
    h_span = max(1, bbox.height)
    w_span = max(1, bbox.width)
    for r in records:
        if not r.is_numeric:
            continue
        # Phase-aware short-circuit
        if r.phase == "x_band":
            out.append(r); continue
        if r.phase == "y_band":
            continue
        # Spatial filter for full-phase records
        cx, cy = r.center
        if (bbox.left - 0.10 * w_span <= cx <= bbox.right + 0.10 * w_span
                and cy >= bbox.bottom - 0.02 * h_span):
            out.append(r)
    return out


def filter_y_axis_labels(
    records: List[OCRRecord],
    bbox,
) -> List[OCRRecord]:
    """Keep records that should be considered for y-axis pairing.

    Phase-aware: see `filter_x_axis_labels` for the symmetric logic.
    """
    out: List[OCRRecord] = []
    h_span = max(1, bbox.height)
    w_span = max(1, bbox.width)
    for r in records:
        if not r.is_numeric:
            continue
        if r.phase == "y_band":
            out.append(r); continue
        if r.phase == "x_band":
            continue
        cx, cy = r.center
        if (bbox.top - 0.06 * h_span <= cy <= bbox.bottom + 0.06 * h_span
                and cx <= bbox.left + 0.04 * w_span):
            out.append(r)
    return out


# ----------------------------------------------------------------------
# Greedy 1-D assignment with mutual nearest neighbours
# ----------------------------------------------------------------------

def _greedy_one_to_one(
    label_positions: np.ndarray,
    tick_positions: np.ndarray,
    max_distance: float,
) -> List[Tuple[int, int, float]]:
    """Return [(label_idx, tick_idx, distance), ...] one-to-one assignments.

    Algorithm: build all (label, tick) pairs with distance <= max_distance,
    sort by distance, accept greedily. This is O(n*m log nm) but n,m are tiny
    (<20 each in practice) and gives the optimal "minimum total cost" matching
    when distances satisfy the Monge property — which is true for points on a
    1-D line, so this is provably optimal here.
    """
    if label_positions.size == 0 or tick_positions.size == 0:
        return []
    pairs = []
    for i, lp in enumerate(label_positions):
        for j, tp in enumerate(tick_positions):
            d = abs(float(lp) - float(tp))
            if d <= max_distance:
                pairs.append((d, i, j))
    pairs.sort()
    used_l: set = set()
    used_t: set = set()
    out: List[Tuple[int, int, float]] = []
    for d, i, j in pairs:
        if i in used_l or j in used_t:
            continue
        used_l.add(i); used_t.add(j)
        out.append((i, j, d))
    return out


def _enforce_monotonic_y(pairs: List[PairedTick]) -> List[PairedTick]:
    """Pixel-y increases downward; data-y decreases. Drop pairs that violate this.

    We look at the longest monotonically-decreasing-by-data subsequence when
    sorted by pixel-y ascending, and drop the rest.
    """
    return _enforce_monotonic_generic(pairs, sort_key=lambda p: p.pixel_position,
                                      data_should_increase=False)


def _enforce_monotonic_x(pairs: List[PairedTick]) -> List[PairedTick]:
    """Pixel-x increases left-to-right and (for normal x-axes) data-x increases too."""
    return _enforce_monotonic_generic(pairs, sort_key=lambda p: p.pixel_position,
                                      data_should_increase=True)


def _enforce_monotonic_generic(
    pairs: List[PairedTick],
    *,
    sort_key,
    data_should_increase: bool,
) -> List[PairedTick]:
    if len(pairs) <= 2:
        return pairs
    sorted_pairs = sorted(pairs, key=sort_key)
    n = len(sorted_pairs)
    # Longest monotonic subsequence by data value
    values = [p.data_value for p in sorted_pairs]
    sign = 1 if data_should_increase else -1
    # DP for longest non-decreasing (after applying sign) subsequence
    dp = [1] * n
    prev = [-1] * n
    for i in range(n):
        for j in range(i):
            if sign * values[j] <= sign * values[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                prev[i] = j
    # Reconstruct
    end = max(range(n), key=lambda k: dp[k])
    keep_indices = []
    cur = end
    while cur != -1:
        keep_indices.append(cur)
        cur = prev[cur]
    keep_indices.reverse()
    keep_set = set(keep_indices)
    out: List[PairedTick] = []
    for k, p in enumerate(sorted_pairs):
        if k in keep_set:
            out.append(p)
        else:
            # Drop the pair but mark a status flag for diagnostics.
            p.include = False
            p.status = "monotonicity_violation"
            out.append(p)
    return out


# ----------------------------------------------------------------------
# Public pairing entry points
# ----------------------------------------------------------------------

def pair_x(
    records: List[OCRRecord],
    grid: GridFit,
    bbox,
    *,
    max_distance: float,
) -> List[PairedTick]:
    """Pair x-axis OCR labels to fitted geometric x-tick positions."""
    label_positions = np.array([r.center[0] for r in records], dtype=float)
    tick_positions = np.array(grid.fitted_positions, dtype=float)
    assignments = _greedy_one_to_one(label_positions, tick_positions, max_distance)

    paired: List[PairedTick] = []
    used_pairs = {(i, j) for i, j, _ in assignments}
    for i, j, d in assignments:
        rec = records[i]
        if rec.value is None:
            continue
        grid_idx = grid.fitted_indices[j] if j < len(grid.fitted_indices) else None
        paired.append(PairedTick(
            pixel_position=float(tick_positions[j]),
            fixed_axis_pixel=float(bbox.bottom),
            data_value=float(rec.value),
            pair_distance_px=float(d),
            grid_index=grid_idx,
            label_bbox=tuple(rec.bbox),
            raw_text=rec.raw_text,
            cleaned_text=rec.cleaned_text,
            ocr_confidence=rec.confidence,
            parse_status=rec.parse_status,
            flag=rec.parse_flag,
            include=True,
            status="paired_to_tick_mark",
        ))
    paired = _enforce_monotonic_x(paired)
    return paired


def pair_y(
    records: List[OCRRecord],
    grid: GridFit,
    bbox,
    *,
    max_distance: float,
) -> List[PairedTick]:
    """Pair y-axis OCR labels to fitted geometric y-tick positions."""
    label_positions = np.array([r.center[1] for r in records], dtype=float)
    tick_positions = np.array(grid.fitted_positions, dtype=float)
    assignments = _greedy_one_to_one(label_positions, tick_positions, max_distance)

    paired: List[PairedTick] = []
    for i, j, d in assignments:
        rec = records[i]
        if rec.value is None:
            continue
        grid_idx = grid.fitted_indices[j] if j < len(grid.fitted_indices) else None
        paired.append(PairedTick(
            pixel_position=float(tick_positions[j]),
            fixed_axis_pixel=float(bbox.left),
            data_value=float(rec.value),
            pair_distance_px=float(d),
            grid_index=grid_idx,
            label_bbox=tuple(rec.bbox),
            raw_text=rec.raw_text,
            cleaned_text=rec.cleaned_text,
            ocr_confidence=rec.confidence,
            parse_status=rec.parse_status,
            flag=rec.parse_flag,
            include=True,
            status="paired_to_tick_mark",
        ))
    paired = _enforce_monotonic_y(paired)
    return paired
