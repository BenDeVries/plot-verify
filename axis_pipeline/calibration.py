"""Linear axis calibration: fit data = scale * pixel + offset.

OLS (ordinary least squares) is used throughout. After the initial fit,
the residual standard error (RSE = sqrt(RSS / (n-2))) is computed; any
point whose residual magnitude exceeds 2*RSE is removed and OLS is
re-run on the survivors. This repeats until all residuals are within
2*RSE or only 2 points remain. Removed points are marked `include=False`
so the pipeline's anchor selection also excludes them.

We fit each axis independently. Fitting in pixel space (data ~ pixel)
gives slope=scale, offset=data-intercept directly.
"""
from __future__ import annotations

from typing import List, Optional

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


def calibrate_axis(
    paired_ticks: List[PairedTick],
) -> Optional[AxisCalibration]:
    """Fit a 1-D linear OLS calibration with residual-based outlier removal.

    Iteratively removes points whose residual magnitude exceeds 2 * RSE
    (residual standard error = sqrt(RSS / (n-2))) and re-runs OLS until all
    remaining residuals are within the threshold or only 2 points survive.
    Removed points are marked include=False so the pipeline's anchor
    selection also excludes them.

    Returns None if fewer than 2 included points survive, or if the
    remaining points are degenerate (all same data value or pixel position).
    """
    inc = [t for t in paired_ticks if t.include and t.data_value is not None]
    if len(inc) < 2:
        return None

    if len({round(t.data_value, 9) for t in inc}) < 2:
        return None
    if len({round(t.pixel_position, 6) for t in inc}) < 2:
        return None

    active = list(inc)

    while len(active) > 3:
        pixels = np.array([t.pixel_position for t in active], dtype=float)
        values = np.array([t.data_value for t in active], dtype=float)

        cal = _ols(pixels, values)
        if cal.rmse_px <= 0.1:
            break

        # Greedy search: find the point whose removal most reduces rmse_px.
        best_idx = -1
        best_rmse_px = cal.rmse_px
        for i in range(len(active)):
            candidate_px = np.delete(pixels, i)
            candidate_vals = np.delete(values, i)
            if len({round(v, 9) for v in candidate_vals}) < 2:
                continue
            if len({round(p, 6) for p in candidate_px}) < 2:
                continue
            candidate_cal = _ols(candidate_px, candidate_vals)
            if candidate_cal.rmse_px < best_rmse_px:
                best_rmse_px = candidate_cal.rmse_px
                best_idx = i

        if best_idx == -1:
            break  # no removal improves rmse_px

        active[best_idx].include = False
        active.pop(best_idx)

    if len(active) < 2:
        return None

    pixels = np.array([t.pixel_position for t in active], dtype=float)
    values = np.array([t.data_value for t in active], dtype=float)
    return _ols(pixels, values)
