"""Linear axis calibration: fit data = scale * pixel + offset.

Two estimators are available:

OLS (ordinary least squares)
    Closed-form. Optimal under Gaussian residuals. Single bad pair (a misread
    label paired to the wrong tick) can pull the slope substantially. Used as
    the default for 2-point calibration and as a starting point for the
    robust estimator.

Student-t MLE
    Maximum likelihood under a Student-t residual model with degrees-of-
    freedom `df` (default 4). Heavy tails downweight outliers automatically;
    likelihood-based standard errors come from the inverse Hessian. The MLE
    is solved with `scipy.optimize.minimize` starting from the OLS estimate.

Empirical guidance for plot calibration:
    - 2 anchor points → OLS only (degenerate t-MLE).
    - 3+ points → t-MLE if `use_robust=True`; otherwise OLS.

We fit each axis independently. Fitting in pixel space (data ~ pixel) gives
the slope = scale, offset = data-intercept directly, which is what the
existing `compute_calibration` callers expect.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from .types import AxisCalibration, PairedTick


def _ols(pixels: np.ndarray, values: np.ndarray) -> AxisCalibration:
    n = len(pixels)
    if n < 2:
        raise ValueError("OLS requires at least 2 points")
    if n == 2:
        scale = (values[1] - values[0]) / (pixels[1] - pixels[0])
        offset = values[0] - scale * pixels[0]
        residuals_data = np.zeros(2)
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


def _student_t_neg_log_likelihood(params, pixels, values, df):
    slope, offset, log_sigma = params
    sigma = np.exp(log_sigma)
    fitted = slope * pixels + offset
    z = (values - fitted) / sigma
    # Student-t log-pdf (location=fitted, scale=sigma, df=df), summed
    log_pdf = stats.t.logpdf(z, df=df) - log_sigma
    return -float(np.sum(log_pdf))


def _student_t_mle(
    pixels: np.ndarray,
    values: np.ndarray,
    df: float,
) -> AxisCalibration:
    """Robust regression via Student-t maximum likelihood.

    Falls back to OLS if optimization fails.
    """
    n = len(pixels)
    if n < 3:
        return _ols(pixels, values)

    init_ols = _ols(pixels, values)
    sigma_init = max(1e-6, init_ols.rmse_data) if init_ols.rmse_data > 0 else max(1e-6, 0.05 * np.ptp(values))
    x0 = np.array([init_ols.scale, init_ols.offset, np.log(sigma_init)], dtype=float)
    try:
        result = minimize(
            _student_t_neg_log_likelihood,
            x0,
            args=(pixels, values, float(df)),
            method="Nelder-Mead",
            options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 5000, "disp": False},
        )
    except Exception:
        return init_ols
    if not result.success:
        return init_ols
    slope, offset, log_sigma = result.x
    sigma = float(np.exp(log_sigma))
    fitted = slope * pixels + offset
    residuals_data = values - fitted
    rmse_data = float(np.sqrt(np.mean(residuals_data ** 2)))
    rmse_px = rmse_data / abs(slope) if abs(slope) > 1e-12 else 0.0
    log_lik = -float(result.fun)

    # Approximate standard errors from the observed Fisher information,
    # numerically differentiating the negative log-likelihood.
    slope_se: Optional[float] = None
    offset_se: Optional[float] = None
    try:
        eps = np.array([1e-4 * max(1.0, abs(slope)),
                        1e-4 * max(1.0, abs(offset)),
                        1e-3])
        H = np.zeros((3, 3))
        f0 = result.fun
        for i in range(3):
            for j in range(i, 3):
                ei = np.zeros(3); ei[i] = eps[i]
                ej = np.zeros(3); ej[j] = eps[j]
                fpp = _student_t_neg_log_likelihood(result.x + ei + ej, pixels, values, df)
                fpm = _student_t_neg_log_likelihood(result.x + ei - ej, pixels, values, df)
                fmp = _student_t_neg_log_likelihood(result.x - ei + ej, pixels, values, df)
                fmm = _student_t_neg_log_likelihood(result.x - ei - ej, pixels, values, df)
                if i == j:
                    H[i, j] = (fpp - 2 * f0 + fmm) / (eps[i] * eps[j])
                else:
                    H[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps[i] * eps[j])
                    H[j, i] = H[i, j]
        cov = np.linalg.inv(H)
        if cov[0, 0] > 0:
            slope_se = float(np.sqrt(cov[0, 0]))
        if cov[1, 1] > 0:
            offset_se = float(np.sqrt(cov[1, 1]))
    except (np.linalg.LinAlgError, ValueError):
        pass

    return AxisCalibration(
        scale=float(slope), offset=float(offset), n_points=n,
        method="student_t",
        rmse_px=float(rmse_px), rmse_data=rmse_data,
        log_likelihood=log_lik,
        slope_se=slope_se, offset_se=offset_se,
        df_t=float(df),
    )


def calibrate_axis(
    paired_ticks: List[PairedTick],
    *,
    use_robust: bool = True,
    student_t_df: float = 4.0,
) -> Optional[AxisCalibration]:
    """Fit a 1-D linear calibration from paired (pixel, data) points.

    Returns None if there are fewer than 2 included points, or if all included
    points share the same data value (degenerate — would give a horizontal fit
    with scale=0, which is meaningless for axis calibration and a strong
    signal that OCR misread the labels).
    """
    inc = [t for t in paired_ticks if t.include and t.data_value is not None]
    if len(inc) < 2:
        return None

    # Reject degenerate cases where all (or all but one) labels share the same
    # value. A common cause is a band crop that bisects multi-digit labels and
    # OCRs each as just the trailing digit ("10", "20", ... → "0", "0", ...).
    distinct = len({round(t.data_value, 9) for t in inc})
    if distinct < 2:
        return None
    # Also reject if pixel positions collapse (shouldn't happen post-pairing,
    # but defensive — an all-same-pixel fit is also degenerate).
    distinct_px = len({round(t.pixel_position, 6) for t in inc})
    if distinct_px < 2:
        return None

    pixels = np.array([t.pixel_position for t in inc], dtype=float)
    values = np.array([t.data_value for t in inc], dtype=float)

    if len(inc) == 2 or not use_robust:
        return _ols(pixels, values)
    return _student_t_mle(pixels, values, student_t_df)
