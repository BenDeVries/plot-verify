"""Pure point-editing math shared by arrow-key nudging and typed edits.

No shiny imports — unit-testable and safe to stage flat in the shinylive
bundle. Bounds are absolute data coordinates (matching ``EditableOverlay``),
never offsets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PointVals:
    """A point's editable values: position plus absolute interval bounds."""
    x: float
    y: float
    upper: Optional[float] = None
    lower: Optional[float] = None


def apply_nudge(
    vals: PointVals,
    dx: float,
    dy: float,
    *,
    part: str = "center",
    linked: bool = False,
    is_horizontal: bool = False,
    lock_y: bool = False,
) -> PointVals:
    """Apply one nudge (data-space deltas) to a point.

    - ``part == "center"``: gang-move — the point and both bounds shift
      together, preserving the interval width. ``lock_y`` pins ``y`` (forest
      rows).
    - ``part == "upper"`` / ``"lower"``: the value-axis delta moves that
      bound; when ``linked`` the opposite bound mirrors so the center stays
      fixed.

    ``is_horizontal`` selects which delta moves the value axis: ``dx`` for
    horizontal layouts (bounds bracket ``x``), ``dy`` otherwise.
    """
    x, y, upper, lower = vals.x, vals.y, vals.upper, vals.lower
    dv = dx if is_horizontal else dy
    center = x if is_horizontal else y

    if part == "upper":
        upper = (upper if upper is not None else center) + dv
        if linked:
            lower = 2.0 * center - upper
    elif part == "lower":
        lower = (lower if lower is not None else center) + dv
        if linked:
            upper = 2.0 * center - lower
    else:
        x += dx
        if not lock_y:
            y += dy
        if upper is not None:
            upper += dv
        if lower is not None:
            lower += dv

    return PointVals(x=x, y=y, upper=upper, lower=lower)


def half_width_of(
    center: float,
    upper: Optional[float],
    lower: Optional[float],
) -> Optional[float]:
    """The symmetric half-width best describing ``[lower, upper]``.

    Mean of the two offsets when both bounds exist (exact when they are
    already symmetric about ``center``); the single offset when only one
    bound exists; None when there is no interval.
    """
    if upper is None and lower is None:
        return None
    if upper is None:
        return abs(center - lower)
    if lower is None:
        return abs(upper - center)
    return ((upper - center) + (center - lower)) / 2.0


def linked_bounds(center: float, half_width: float) -> Tuple[float, float]:
    """Absolute (lower, upper) for a symmetric interval about ``center``."""
    hw = abs(half_width)
    return center - hw, center + hw
