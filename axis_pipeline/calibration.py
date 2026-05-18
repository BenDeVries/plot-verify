"""Linear and log-linear axis calibration.

For LINEAR axes:  fit  data = scale * pixel + offset  (OLS).
For LOG10 axes:   fit  log10(data) = scale * pixel + offset  (OLS on log-space).

OLS (ordinary least squares) is used throughout. A greedy search removes one
point at a time while rmse improves, stopping when no removal helps or only 3
points remain. All candidate sets are collected. The 1-SE rule then selects
the candidate with the most points whose rmse ≤ min(rmse) + SE, where SE is
the residual standard error of the min-rmse candidate. (The initial set's
RSE is never used as a floor — that would let a single bad tick inflate the
threshold past the point at which it could exclude itself.)

Log-scale auto-detection: if all included values are positive and span ≥ 2
orders of magnitude, the axis is treated as log10. The returned AxisCalibration
has log_base=10.0 and pixel_to_data / data_to_pixel use 10^(linear) mapping.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats

from .types import AxisCalibration, PairedTick


def _ols(pixels: np.ndarray, values: np.ndarray) -> AxisCalibration:
    n = len(pixels)
    if n < 2:
        raise ValueError("OLS requires at least 2 points")
    if n == 2:
        scale = (values[1] - values[0]) / (pixels[1] - pixels[0])
        offset = values[0] - scale * pixels[0]
        return AxisCalibration(
            scale=float(scale), offset=float(offset),
            n_points=2, method="two_point",
            rmse_px=0.0, rmse_data=0.0,
            slope_se=None, offset_se=None,
        )
    res = stats.linregress(pixels, values)
    scale = float(res.slope)
    offset = float(res.intercept)
    fitted = scale * pixels + offset
    residuals_data = values - fitted
    rmse_data = float(np.sqrt(np.mean(residuals_data ** 2)))
    rmse_px = rmse_data / abs(scale) if abs(scale) > 1e-12 else 0.0
    return AxisCalibration(
        scale=scale, offset=offset, n_points=n, method="ols",
        rmse_px=float(rmse_px), rmse_data=rmse_data,
        slope_se=float(res.stderr) if res.stderr is not None else None,
        offset_se=float(res.intercept_stderr) if res.intercept_stderr is not None else None,
    )


def _is_log10_scale(pts: List[PairedTick]) -> bool:
    """Return True when all values are positive and span ≥ 2 orders of magnitude.

    A 2-decade span (100× range) is enough to distinguish log-scale axes
    (e.g. 1e-14 … 1e-7) from linear axes (e.g. 0.5 … 3.5).
    """
    vals = [t.data_value for t in pts if t.data_value is not None]
    if len(vals) < 2 or any(v <= 0 for v in vals):
        return False
    log_vals = [math.log10(v) for v in vals]
    return (max(log_vals) - min(log_vals)) >= 2.0


def _make_candidate(
    pts: List[PairedTick],
    value_transform=None,
) -> Tuple[List[PairedTick], AxisCalibration, float]:
    """Return (pts, cal, rse) for a set of points.

    rse = sqrt(RSS / (n-2)); 0.0 when n <= 2 (no degrees of freedom).
    value_transform, if provided, is applied element-wise to data values
    before OLS (e.g. math.log10 for log-scale axes).
    """
    pixels = np.array([t.pixel_position for t in pts], dtype=float)
    values = np.array([t.data_value for t in pts], dtype=float)
    if value_transform is not None:
        values = np.array([value_transform(v) for v in values], dtype=float)
    cal = _ols(pixels, values)
    rse = 0.0
    if len(pts) > 2:
        fitted = cal.scale * pixels + cal.offset
        rss = float(np.sum((values - fitted) ** 2))
        rse = float(np.sqrt(rss / (len(pts) - 2)))
    return (list(pts), cal, rse)


def calibrate_axis(
    paired_ticks: List[PairedTick],
) -> Optional[AxisCalibration]:
    """Fit a 1-D OLS calibration with greedy outlier removal and 1-SE selection.

    Auto-detects log10 scale: if all included values are positive and span ≥ 2
    orders of magnitude, the fit is performed on log10(data) vs pixel.  The
    returned AxisCalibration has log_base=10.0 and its pixel_to_data /
    data_to_pixel methods apply the corresponding inverse/forward transform.

    Greedy phase: remove the point whose removal most reduces rmse in the
    (possibly log-transformed) value space, stopping when no removal helps or
    only 3 points remain.

    Selection phase (1-SE rule): find minimum-rmse candidate; compute its SE =
    sqrt(RSS/(n-2)), floored by the initial full-set RSE.  Among all candidates
    with rmse ≤ min_rmse + SE, choose the one with the most points.

    Returns None if fewer than 2 included points are available or data are
    degenerate (all same value or pixel position).
    """
    inc = [t for t in paired_ticks if t.include and t.data_value is not None]
    if len(inc) < 2:
        return None

    if len({round(t.data_value, 9) for t in inc}) < 2:
        return None
    if len({round(t.pixel_position, 6) for t in inc}) < 2:
        return None

    # Detect log10 scale and set up a value transform for the OLS.
    use_log = _is_log10_scale(inc)
    vt = math.log10 if use_log else None   # value_transform shorthand

    active = list(inc)
    candidates = [_make_candidate(active, value_transform=vt)]

    while len(active) > 3:
        pixels = np.array([t.pixel_position for t in active], dtype=float)
        values = np.array([t.data_value for t in active], dtype=float)
        if vt is not None:
            values = np.array([vt(v) for v in values], dtype=float)
        current_rmse = candidates[-1][1].rmse_data

        best_idx = -1
        best_rmse = current_rmse
        for i in range(len(active)):
            cand_px = np.delete(pixels, i)
            cand_vals = np.delete(values, i)
            if len({round(v, 9) for v in cand_vals}) < 2:
                continue
            if len({round(p, 6) for p in cand_px}) < 2:
                continue
            cand_cal = _ols(cand_px, cand_vals)
            if cand_cal.rmse_data < best_rmse:
                best_rmse = cand_cal.rmse_data
                best_idx = i

        if best_idx == -1:
            break

        active.pop(best_idx)
        candidates.append(_make_candidate(active, value_transform=vt))

    # 1-SE rule: locate the minimum-rmse candidate and its SE, then admit
    # any candidate whose rmse_data is within 1 SE of that minimum.
    #
    # We deliberately do NOT floor `se_at_min` with the initial (full-set) RSE.
    # Doing so allows a single bad tick to inflate the threshold by the amount
    # the rule is supposed to detect, defeating the whole point of the rule.
    # If the min-rmse candidate has rmse_data ≈ 0 (e.g. three perfectly
    # collinear points), that is a feature, not a degenerate case: continuous
    # pixel/value coordinates being exactly collinear strongly implies those
    # three measurements are correct, so the resulting tiny threshold should
    # exclude the other (worse-fitting) candidates.
    min_i = min(range(len(candidates)), key=lambda i: candidates[i][1].rmse_data)
    min_rmse = candidates[min_i][1].rmse_data
    se_at_min = candidates[min_i][2]
    threshold = min_rmse + se_at_min

    # Use <= with a small epsilon so the min-rmse candidate is always included
    # even when threshold rounds to exactly min_rmse (se_at_min ≈ 0).
    valid = [c for c in candidates if c[1].rmse_data <= threshold + 1e-9]
    if not valid:
        valid = [candidates[min_i]]

    chosen_pts, chosen_cal, _ = max(valid, key=lambda c: len(c[0]))

    chosen_ids = {id(t) for t in chosen_pts}
    for t in inc:
        t.include = id(t) in chosen_ids

    if use_log:
        chosen_cal = AxisCalibration(
            scale=chosen_cal.scale,
            offset=chosen_cal.offset,
            n_points=chosen_cal.n_points,
            method=chosen_cal.method,
            rmse_px=chosen_cal.rmse_px,
            rmse_data=chosen_cal.rmse_data,
            slope_se=chosen_cal.slope_se,
            offset_se=chosen_cal.offset_se,
            log_base=10.0,
        )

    return chosen_cal
