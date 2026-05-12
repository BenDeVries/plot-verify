"""Linear axis calibration: fit data = scale * pixel + offset.

OLS (ordinary least squares) is used throughout. A greedy search removes one
point at a time while rmse (data units) improves, stopping when no removal
helps or only 3 points remain. All candidate sets visited during the search
are collected. The 1-SE rule then selects the candidate with the most points
whose rmse < min(rmse) + SE, where SE = sqrt(RSS/(n-2)) of the minimum-rmse
candidate. Removed points are marked include=False so the pipeline's anchor
selection also excludes them.

We fit each axis independently. Fitting in pixel space (data ~ pixel)
gives slope=scale, offset=data-intercept directly.
"""
from __future__ import annotations

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


def _make_candidate(
    pts: List[PairedTick],
) -> Tuple[List[PairedTick], AxisCalibration, float]:
    """Return (pts, cal, rse) for a set of points.

    rse = sqrt(RSS / (n-2)); 0.0 when n <= 2 (no degrees of freedom).
    """
    pixels = np.array([t.pixel_position for t in pts], dtype=float)
    values = np.array([t.data_value for t in pts], dtype=float)
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
    """Fit a 1-D linear OLS calibration with greedy outlier removal and 1-SE selection.

    Greedy phase: at each step remove the point whose removal most reduces rmse
    (data units), stopping when no removal improves rmse or only 3 points
    remain. Each candidate set (including the initial full set) is recorded.

    Selection phase (1-SE rule): find the candidate with minimum rmse; compute
    its residual standard error SE = sqrt(RSS/(n-2)). Among all candidates with
    rmse < min_rmse + SE, choose the one with the most points.

    Returns None if fewer than 2 included points are available or the data are
    degenerate (all same value or pixel position).
    """
    inc = [t for t in paired_ticks if t.include and t.data_value is not None]
    if len(inc) < 2:
        return None

    if len({round(t.data_value, 9) for t in inc}) < 2:
        return None
    if len({round(t.pixel_position, 6) for t in inc}) < 2:
        return None

    active = list(inc)
    candidates = [_make_candidate(active)]

    while len(active) > 3:
        pixels = np.array([t.pixel_position for t in active], dtype=float)
        values = np.array([t.data_value for t in active], dtype=float)
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
        candidates.append(_make_candidate(active))

    # 1-SE rule: locate the minimum-rmse candidate and its SE
    min_i = min(range(len(candidates)), key=lambda i: candidates[i][1].rmse_data)
    min_rmse = candidates[min_i][1].rmse_data
    se_at_min = candidates[min_i][2]
    threshold = min_rmse + 3 * se_at_min

    valid = [c for c in candidates if c[1].rmse_data < threshold]
    if not valid:
        valid = candidates

    chosen_pts, chosen_cal, _ = max(valid, key=lambda c: len(c[0]))

    chosen_ids = {id(t) for t in chosen_pts}
    for t in inc:
        t.include = id(t) in chosen_ids

    return chosen_cal
