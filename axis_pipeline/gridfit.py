"""Reject tick positions that don't fit the modal spacing of a linear axis.

The pipeline produces an unfiltered list of geometric tick positions from a
1-D peak detector. On real journal plots, the peak finder occasionally fires
on:
 - the axis line itself ("tick" at the corner where two axes meet)
 - gridlines inside the plot that visually punch through the axis band
 - anti-aliasing artifacts adjacent to data series
 - the right plot border being misread as a tick

For a linear axis, the *true* major ticks lie on a regular grid. This module
fits a 1-D grid to the candidate positions, then drops candidates that don't
sit on it.

Algorithm
---------
1. Find the modal spacing among consecutive sorted positions, robustly:
    - take all consecutive diffs
    - cluster them with a tolerance of 8 px
    - take the largest cluster's median as the modal spacing
2. Hypothesize: positions = origin + k * spacing, for integer k.
3. For each candidate position, snap to the nearest k. Keep if residual is
   within `tolerance_frac * spacing` (default 20%).
4. Refine `origin` by least-squares: origin = mean(pos - k*spacing) over kept.

The same routine works for x-positions (sorted ascending) and y-positions
(sorted ascending) — direction handling is the caller's responsibility.

For non-linear scales (log, category) this module is bypassed; future
implementations should produce the same `GridFit` shape so the rest of the
pipeline is unaffected.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .types import GridFit


def _modal_spacing(diffs: np.ndarray, cluster_tol_px: float = 8.0) -> Optional[float]:
    """Return the median of the largest cluster of consecutive diffs.

    Diffs from a regular grid cluster tightly; diffs that span an irregular
    gap (a missing tick or a stray detection) are isolated singletons and
    should not influence the modal estimate.

    Clustering compares each new diff against the cluster's current median
    rather than its most recently added element — otherwise a chain of
    small steps could transitively drift one cluster across an arbitrarily
    wide range of values.
    """
    if diffs.size == 0:
        return None
    sd = np.sort(diffs)
    # Greedy clustering: consecutive sorted diffs within cluster_tol_px of
    # the cluster median form a cluster.
    clusters: List[List[float]] = [[float(sd[0])]]
    for d in sd[1:]:
        if abs(d - float(np.median(clusters[-1]))) <= cluster_tol_px:
            clusters[-1].append(float(d))
        else:
            clusters.append([float(d)])
    largest = max(clusters, key=len)
    return float(np.median(largest))


def fit_linear_grid(
    positions: List[float],
    *,
    tolerance_frac: float = 0.20,
    min_ticks: int = 3,
) -> GridFit:
    """Fit a regular grid to `positions`. Drop positions that don't fit.

    Returns an `GridFit` whose `success` reflects whether at least `min_ticks`
    positions snapped to the grid.
    """
    pos = np.array(sorted(positions), dtype=float)
    if pos.size < 2:
        return GridFit(
            spacing=0.0, origin=0.0,
            fitted_positions=list(pos),
            fitted_indices=list(range(len(pos))),
            rejected_positions=[],
            grid_residuals=[0.0] * len(pos),
            n_grid_cells=len(pos),
            success=len(pos) >= min_ticks,
        )

    diffs = np.diff(pos)
    spacing = _modal_spacing(diffs)
    if spacing is None or spacing <= 0:
        return GridFit(
            spacing=0.0, origin=float(pos[0]),
            fitted_positions=list(pos),
            fitted_indices=list(range(len(pos))),
            rejected_positions=[],
            grid_residuals=[0.0] * len(pos),
            n_grid_cells=len(pos),
            success=len(pos) >= min_ticks,
        )

    # Provisional origin: the smallest position. Snap each pos to nearest k.
    origin = float(pos[0])
    ks = np.round((pos - origin) / spacing).astype(int)
    snapped = origin + ks * spacing
    residuals = pos - snapped
    tol = tolerance_frac * spacing
    keep = np.abs(residuals) <= tol

    if keep.sum() >= 2:
        # Refine origin from the kept positions: origin minimises sum (pos - k*spacing - origin)^2
        origin = float(np.mean(pos[keep] - ks[keep] * spacing))
        snapped = origin + ks * spacing
        residuals = pos - snapped
        keep = np.abs(residuals) <= tol

    fitted_positions = pos[keep].tolist()
    fitted_indices = ks[keep].tolist()
    rejected_positions = pos[~keep].tolist()
    grid_residuals = residuals[keep].tolist()
    if fitted_indices:
        n_cells = int(max(fitted_indices) - min(fitted_indices) + 1)
        # Renormalize indices so the smallest is 0 — keeps downstream indices small/positive.
        shift = min(fitted_indices)
        fitted_indices = [k - shift for k in fitted_indices]
        origin = float(origin + shift * spacing)
    else:
        n_cells = 0

    return GridFit(
        spacing=float(spacing),
        origin=float(origin),
        fitted_positions=fitted_positions,
        fitted_indices=fitted_indices,
        rejected_positions=rejected_positions,
        grid_residuals=grid_residuals,
        n_grid_cells=n_cells,
        success=len(fitted_positions) >= min_ticks,
    )
