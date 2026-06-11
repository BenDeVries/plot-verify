"""Pure-computation helpers for the Overlay tab dashboard.

No Shiny / UI imports — these functions take DataFrames and return DataFrames
or plain dicts so they can be unit-tested in isolation.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

VALID_ERROR_TYPES = ("Confidence", "Prediction", "SE", "SD")


def compute_time_series_stats(
    df: pd.DataFrame,
    error_type: str,
    percent: float = 95.0,
    n_per_series: Optional[Dict[str, Optional[int]]] = None,
    is_log_scale: bool = False,
) -> pd.DataFrame:
    """Wide-format MultiIndex DataFrame of point estimates and SDs.

    Columns are a two-level MultiIndex: level 0 is the series name, level 1
    is the metric label.  For linear scale the metrics are ``("μ", "σ")``;
    for log scale they are ``("μ", "σ_log", "σ")`` where ``σ_log`` is the SD
    on the log scale and ``σ`` is the geometric (back-transformed) SD.

    Returns an empty DataFrame when the input has no rows.
    """
    if n_per_series is None:
        n_per_series = {}

    alpha = 1.0 - percent / 100.0
    series_order = list(dict.fromkeys(df["series"].astype(str)))

    col_frames = []

    for series in series_order:
        sdf = df[df["series"].astype(str) == series].copy()
        y = pd.to_numeric(sdf["y"], errors="coerce").values
        lower = pd.to_numeric(sdf["y_err_lower"], errors="coerce").values
        upper = pd.to_numeric(sdf["y_err_upper"], errors="coerce").values
        x_vals = pd.to_numeric(sdf["x"], errors="coerce").values

        n_val = n_per_series.get(series)

        # Compute half-width in the appropriate space
        if is_log_scale:
            with np.errstate(invalid="ignore", divide="ignore"):
                half_w = np.where(
                    (y > 0) & (lower > 0) & (upper > 0),
                    (np.log(upper) - np.log(lower)) / 2.0,
                    np.nan,
                )
        else:
            half_w = np.where(
                np.isfinite(lower) & np.isfinite(upper),
                (upper - lower) / 2.0,
                np.nan,
            )

        sd = _sd_from_half_width(half_w, error_type, percent, alpha, n_val)

        if is_log_scale:
            sd_orig = np.where(np.isfinite(sd), np.exp(sd), np.nan)
            tuples = [(series, "μ"), (series, "σ_log"), (series, "σ")]
            data = np.column_stack([y, sd, sd_orig])
        else:
            tuples = [(series, "μ"), (series, "σ")]
            data = np.column_stack([y, sd])

        sub = pd.DataFrame(
            data,
            columns=pd.MultiIndex.from_tuples(tuples),
            index=pd.Index(x_vals, name="x"),
        )
        col_frames.append(sub)

    if not col_frames:
        return pd.DataFrame()

    return pd.concat(col_frames, axis=1).sort_index()


def _sd_from_half_width(
    half_w: np.ndarray,
    error_type: str,
    percent: float,
    alpha: float,
    n: Optional[int],
) -> np.ndarray:
    if error_type == "SD":
        return half_w.copy()

    if error_type == "SE":
        if n is None or n < 1:
            return np.full_like(half_w, np.nan)
        return half_w * math.sqrt(n)

    # CI or PI — need n and percent
    if n is None or n < 2:
        return np.full_like(half_w, np.nan)

    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)

    if error_type == "Confidence":
        return half_w * math.sqrt(n) / t_crit

    # Prediction interval: hw = t * sd * sqrt(1 + 1/n)
    return half_w / (t_crit * math.sqrt(1.0 + 1.0 / n))


def build_time_series_display_df(
    df: pd.DataFrame,
    eb_type: str,
    percent: float = 95.0,
    n_per_series: Optional[Dict[str, Optional[int]]] = None,
    is_log_scale: bool = False,
    display_x: str = "None",
) -> pd.DataFrame:
    """Observation-indexed display DataFrame for time series w/ intervals.

    Rows are indexed by observation number (1-based, sorted by x within each
    series).  ``display_x`` controls x-column presence:

    * ``"None"``          — Obs index, no x column
    * ``"Single column"`` — index is the mean x across series per observation
    * ``"Multi column"``  — Obs index + per-series x column before μ/σ columns
    """
    if n_per_series is None:
        n_per_series = {}
    if df.empty:
        return pd.DataFrame()

    alpha = 1.0 - percent / 100.0
    series_order = list(dict.fromkeys(df["series"].astype(str)))

    per_series: dict = {}
    for series in series_order:
        sdf = df[df["series"].astype(str) == series].copy()
        y = pd.to_numeric(sdf["y"], errors="coerce").values
        lower = pd.to_numeric(sdf["y_err_lower"], errors="coerce").values
        upper = pd.to_numeric(sdf["y_err_upper"], errors="coerce").values
        x_vals = pd.to_numeric(sdf["x"], errors="coerce").values

        sort_idx = np.argsort(x_vals, kind="stable")
        x_s = x_vals[sort_idx]
        y_s = y[sort_idx]
        lo_s = lower[sort_idx]
        hi_s = upper[sort_idx]

        n_val = n_per_series.get(series)

        if is_log_scale:
            with np.errstate(invalid="ignore", divide="ignore"):
                half_w = np.where(
                    (y_s > 0) & (lo_s > 0) & (hi_s > 0),
                    (np.log(hi_s) - np.log(lo_s)) / 2.0,
                    np.nan,
                )
        else:
            half_w = np.where(
                np.isfinite(lo_s) & np.isfinite(hi_s),
                (hi_s - lo_s) / 2.0,
                np.nan,
            )

        sd = _sd_from_half_width(half_w, eb_type, percent, alpha, n_val)
        entry: dict = {"x": x_s, "mu": y_s, "sd": sd}
        if is_log_scale:
            entry["sd_log"] = sd.copy()
            entry["sd"] = np.where(np.isfinite(sd), np.exp(sd), np.nan)
        per_series[series] = entry

    max_obs = max(len(v["x"]) for v in per_series.values())
    if max_obs == 0:
        return pd.DataFrame()

    def _pad(arr: np.ndarray, length: int) -> np.ndarray:
        if len(arr) >= length:
            return arr[:length]
        return np.concatenate([arr, np.full(length - len(arr), np.nan)])

    tuples: list = []
    arrays: list = []

    for series, v in per_series.items():
        if display_x == "Multi column":
            tuples.append((series, "x"))
            arrays.append(_pad(v["x"], max_obs))
        tuples.append((series, "μ"))
        arrays.append(_pad(v["mu"], max_obs))
        if is_log_scale:
            tuples.append((series, "σ_log"))
            arrays.append(_pad(v["sd_log"], max_obs))
        tuples.append((series, "σ"))
        arrays.append(_pad(v["sd"], max_obs))

    if not arrays:
        return pd.DataFrame()

    data = np.column_stack(arrays)

    if display_x == "Single column":
        x_stack = np.full((max_obs, len(per_series)), np.nan)
        for col_i, v in enumerate(per_series.values()):
            n = len(v["x"])
            x_stack[:n, col_i] = v["x"]
        x_mean = np.nanmean(x_stack, axis=1)
        index: pd.Index = pd.Index(x_mean, name="x")
    else:
        index = pd.Index(range(1, max_obs + 1), name="Obs")

    return pd.DataFrame(
        data,
        index=index,
        columns=pd.MultiIndex.from_tuples(tuples),
    )


def compute_scatter_stats(df: pd.DataFrame) -> dict:
    """Pearson correlation per series and overall.

    Returns::

        {
            "by_series": {series_name: {"n": int, "r": float, "r2": float}},
            "overall":   {"n": int, "r": float, "r2": float},
        }

    Series or overall with fewer than 2 finite points get r=NaN.
    """
    result: dict = {"by_series": {}, "overall": {}}

    series_order = list(dict.fromkeys(df["series"].astype(str)))
    for series in series_order:
        sdf = df[df["series"].astype(str) == series]
        x = pd.to_numeric(sdf["x"], errors="coerce")
        y = pd.to_numeric(sdf["y"], errors="coerce")
        mask = np.isfinite(x) & np.isfinite(y)
        xv, yv = x[mask].values, y[mask].values
        n = len(xv)
        if n >= 2:
            r, _ = stats.pearsonr(xv, yv)
        else:
            r = float("nan")
        result["by_series"][series] = {"n": n, "r": r, "r2": r ** 2 if math.isfinite(r) else float("nan")}

    x_all = pd.to_numeric(df["x"], errors="coerce")
    y_all = pd.to_numeric(df["y"], errors="coerce")
    mask_all = np.isfinite(x_all) & np.isfinite(y_all)
    xv_all, yv_all = x_all[mask_all].values, y_all[mask_all].values
    n_all = len(xv_all)
    if n_all >= 2:
        r_all, _ = stats.pearsonr(xv_all, yv_all)
    else:
        r_all = float("nan")
    result["overall"] = {
        "n": n_all,
        "r": r_all,
        "r2": r_all ** 2 if math.isfinite(r_all) else float("nan"),
    }

    return result
