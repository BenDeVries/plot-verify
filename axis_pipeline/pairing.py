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

import math
from dataclasses import replace
from typing import List, Optional, Tuple

import numpy as np

from .types import GridFit, OCRRecord, PairedTick


# ----------------------------------------------------------------------
# Spatial filters: which OCR records belong to which axis?
# ----------------------------------------------------------------------

def filter_x_axis_labels(
    records: List[OCRRecord],
    bbox,
    *,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
) -> List[OCRRecord]:
    """Keep records that should be considered for x-axis pairing.

    Phase-aware: a record from `phase=x_band` is always kept (the band scan ran
    on the x-axis label strip by construction). A record from `phase=y_band` is
    always rejected — it cannot belong to the x-axis. A record from `phase=full`
    falls through to the spatial filter (center within the x-axis label region).

    `x_min` / `x_max` are the effective band boundaries after trimming
    (`extra_horizontal`). When set, full-phase records outside those bounds are
    excluded so the trimming is honoured even for labels Phase A already found.
    Band-phase records are already geometrically within the band crop and need
    no extra check.
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
        if x_min is not None and cx < x_min:
            continue
        if x_max is not None and cx > x_max:
            continue
        if (bbox.left - 0.10 * w_span <= cx <= bbox.right + 0.10 * w_span
                and cy >= bbox.bottom - 0.02 * h_span):
            out.append(r)
    return out


def filter_y_axis_labels(
    records: List[OCRRecord],
    bbox,
    *,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> List[OCRRecord]:
    """Keep records that should be considered for y-axis pairing.

    `y_min` / `y_max` are the effective band boundaries after trimming
    (`extra_vertical`). See `filter_x_axis_labels` for the symmetric logic.
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
        if y_min is not None and cy < y_min:
            continue
        if y_max is not None and cy > y_max:
            continue
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
            # Drop the pair but mark a status flag for diagnostics. Build a
            # fresh PairedTick instead of mutating the caller's object —
            # this function is called multiple times in the pipeline and
            # in-place mutation leaked state between runs.
            out.append(replace(p, include=False,
                                status="monotonicity_violation"))
    return out


# ----------------------------------------------------------------------
# Loss-function based matching extension
# ----------------------------------------------------------------------

def _ols_fit(
    tick_px: np.ndarray,
    values: np.ndarray,
) -> Optional[Tuple[float, float, bool]]:
    """Fit OLS line to (tick_px, values). Auto-detects log10 scale.

    Returns (slope, intercept, use_log) or None if data are insufficient.
    use_log is True when all values are positive and span ≥ 1 order of magnitude.
    """
    if len(tick_px) < 2 or len(values) < 2:
        return None
    use_log = bool(np.all(values > 0) and np.ptp(np.log10(values.astype(float))) >= 1.0)
    y = np.log10(values.astype(float)) if use_log else values.astype(float)
    # Tolerance-based uniqueness: a Python `set` over floats is exact equality
    # and can mis-count near-duplicates produced by log10 rounding.
    if len(np.unique(np.round(y, 9))) < 2:
        return None
    try:
        slope, intercept = np.polyfit(tick_px.astype(float), y, 1)
    except (np.linalg.LinAlgError, ValueError):
        return None
    return float(slope), float(intercept), use_log


def _loss_based_extension(
    initial: List[Tuple[int, int, float]],
    label_positions: np.ndarray,
    label_values: List[float],
    tick_positions: np.ndarray,
    soft_max: float,
) -> List[Tuple[int, int, float]]:
    """Extend initial greedy matching using OLS calibration as a loss function.

    After the hard-distance greedy match, many labels may remain unmatched
    because they fall just beyond max_distance. This pass fits an OLS line to
    the initial pairs, then accepts additional (label, tick) pairs within
    soft_max if their OLS residual is within a tolerance derived from the
    initial pairs.

    Accepts a fallback for empty initial: tries soft_max greedy directly.

    Precondition: `label_values` contains no None entries. Callers must
    filter records before calling this; index `i` in `initial` and in the
    returned tuples references the same `label_positions` / `label_values`
    array.
    """
    used_l = {i for i, _, _ in initial}
    used_t = {j for _, j, _ in initial}

    if not initial:
        # Nothing from the hard pass — try soft threshold directly.
        soft_raw = _greedy_one_to_one(label_positions, tick_positions, soft_max)
        return soft_raw

    rem_l = [i for i in range(len(label_positions)) if i not in used_l]
    rem_t = [j for j in range(len(tick_positions)) if j not in used_t]
    if not rem_l or not rem_t:
        return initial

    init_px = np.array([tick_positions[j] for _, j, _ in initial], dtype=float)
    init_vals = np.array([label_values[i] for i, _, _ in initial], dtype=float)
    fit = _ols_fit(init_px, init_vals)
    if fit is None:
        return initial
    slope, intercept, use_log = fit

    init_residuals = []
    for i, j, _ in initial:
        v = label_values[i]
        if use_log and v <= 0:
            continue
        a = math.log10(v) if use_log else float(v)
        init_residuals.append(abs(a - (slope * float(tick_positions[j]) + intercept)))
    tol = max(max(init_residuals) * 3.0, 0.5) if init_residuals else 0.5

    candidates = []
    for j in rem_t:
        predicted = slope * float(tick_positions[j]) + intercept
        for i in rem_l:
            px_dist = abs(float(label_positions[i]) - float(tick_positions[j]))
            if px_dist > soft_max:
                continue
            v = label_values[i]
            if use_log and v <= 0:
                continue
            a = math.log10(v) if use_log else float(v)
            candidates.append((abs(a - predicted), px_dist, i, j))

    candidates.sort()
    new_used_l = set(used_l)
    new_used_t = set(used_t)
    extended = []
    for ols_res, px_dist, i, j in candidates:
        if i in new_used_l or j in new_used_t:
            continue
        if ols_res > tol:
            break
        new_used_l.add(i)
        new_used_t.add(j)
        extended.append((i, j, px_dist))
    return initial + extended


# ----------------------------------------------------------------------
# Public pairing entry points
# ----------------------------------------------------------------------

def _pair_axis(
    records: List[OCRRecord],
    grid: GridFit,
    *,
    center_index: int,
    fixed_axis_pixel: float,
    max_distance: float,
) -> List[PairedTick]:
    """Shared pairing kernel for x/y. Filters out non-numeric records up
    front so all downstream code (greedy matching, OLS extension) works
    against a clean numeric-only view and index alignment is unambiguous.
    """
    # Keep only records with parsed numeric values. `is_numeric=True` is
    # already implied by upstream filters, but stay defensive: a future
    # caller could pass raw records.
    numeric_records: List[OCRRecord] = [r for r in records if r.value is not None]

    label_positions = np.array(
        [r.center[center_index] for r in numeric_records], dtype=float)
    label_values: List[float] = [float(r.value) for r in numeric_records]
    tick_positions = np.array(grid.fitted_positions, dtype=float)

    assignments = _greedy_one_to_one(label_positions, tick_positions, max_distance)
    soft_max = min(max_distance * 2.0, 80.0)
    assignments = _loss_based_extension(
        assignments, label_positions, label_values, tick_positions, soft_max,
    )

    paired: List[PairedTick] = []
    for i, j, d in assignments:
        rec = numeric_records[i]
        grid_idx = grid.fitted_indices[j] if j < len(grid.fitted_indices) else None
        paired.append(PairedTick(
            pixel_position=float(tick_positions[j]),
            fixed_axis_pixel=float(fixed_axis_pixel),
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
    return paired


def pair_x(
    records: List[OCRRecord],
    grid: GridFit,
    bbox,
    *,
    max_distance: float,
) -> List[PairedTick]:
    """Pair x-axis OCR labels to fitted geometric x-tick positions."""
    paired = _pair_axis(
        records, grid,
        center_index=0,
        fixed_axis_pixel=bbox.bottom,
        max_distance=max_distance,
    )
    return _enforce_monotonic_x(paired)


def pair_y(
    records: List[OCRRecord],
    grid: GridFit,
    bbox,
    *,
    max_distance: float,
) -> List[PairedTick]:
    """Pair y-axis OCR labels to fitted geometric y-tick positions."""
    paired = _pair_axis(
        records, grid,
        center_index=1,
        fixed_axis_pixel=bbox.left,
        max_distance=max_distance,
    )
    return _enforce_monotonic_y(paired)
