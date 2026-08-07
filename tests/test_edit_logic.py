"""Pure nudge/symmetry math in shiny_app.edit_logic."""
import pytest

from shiny_app.edit_logic import (
    PointVals,
    apply_nudge,
    half_width_of,
    linked_bounds,
)


def _pt(x=1.0, y=10.0, upper=12.0, lower=8.0):
    return PointVals(x=x, y=y, upper=upper, lower=lower)


# ---------------------------------------------------------------------------
# Center gang-move
# ---------------------------------------------------------------------------

def test_center_gang_move_preserves_half_widths():
    new = apply_nudge(_pt(), dx=0.5, dy=2.0, part="center")
    assert new.x == 1.5
    assert new.y == 12.0
    assert new.upper == 14.0
    assert new.lower == 10.0
    assert new.upper - new.y == 2.0
    assert new.y - new.lower == 2.0


def test_center_gang_move_ignores_linked_flag():
    a = apply_nudge(_pt(), dx=0.0, dy=1.0, part="center", linked=False)
    b = apply_nudge(_pt(), dx=0.0, dy=1.0, part="center", linked=True)
    assert a == b


def test_center_horizontal_bounds_follow_dx():
    new = apply_nudge(_pt(), dx=1.0, dy=0.5, part="center",
                      is_horizontal=True)
    assert new.x == 2.0
    assert new.y == 10.5
    # bounds bracket x, so they follow dx not dy
    assert new.upper == 13.0
    assert new.lower == 9.0


def test_center_forest_lock_y():
    new = apply_nudge(_pt(), dx=1.0, dy=5.0, part="center",
                      is_horizontal=True, lock_y=True)
    assert new.y == 10.0
    assert new.x == 2.0
    assert new.upper == 13.0


def test_center_with_missing_bounds():
    new = apply_nudge(PointVals(1.0, 10.0, None, None), dx=0.0, dy=1.0,
                      part="center")
    assert new.y == 11.0
    assert new.upper is None and new.lower is None


# ---------------------------------------------------------------------------
# Bound nudges
# ---------------------------------------------------------------------------

def test_upper_nudge_unlinked_moves_one_bound():
    new = apply_nudge(_pt(), dx=0.0, dy=1.0, part="upper", linked=False)
    assert new.upper == 13.0
    assert new.lower == 8.0
    assert new.y == 10.0 and new.x == 1.0


def test_upper_nudge_linked_mirrors_lower():
    new = apply_nudge(_pt(), dx=0.0, dy=1.0, part="upper", linked=True)
    assert new.upper == 13.0
    assert new.lower == 7.0  # mirrored about y=10


def test_lower_nudge_linked_mirrors_upper():
    new = apply_nudge(_pt(), dx=0.0, dy=-1.0, part="lower", linked=True)
    assert new.lower == 7.0
    assert new.upper == 13.0


def test_bound_nudge_horizontal_uses_dx():
    new = apply_nudge(_pt(), dx=1.0, dy=99.0, part="upper", linked=True,
                      is_horizontal=True)
    assert new.upper == 13.0
    # mirrored about the value-axis center x=1
    assert new.lower == 1.0 - 12.0
    assert new.y == 10.0


def test_bound_nudge_seeds_from_center_when_missing():
    new = apply_nudge(PointVals(1.0, 10.0, None, None), dx=0.0, dy=2.0,
                      part="upper")
    assert new.upper == 12.0
    assert new.lower is None


# ---------------------------------------------------------------------------
# linked_bounds / half_width_of
# ---------------------------------------------------------------------------

def test_linked_bounds_round_trip():
    lower, upper = linked_bounds(10.0, 2.0)
    assert (lower, upper) == (8.0, 12.0)
    assert half_width_of(10.0, upper, lower) == 2.0


def test_linked_bounds_negative_half_width_abs():
    assert linked_bounds(10.0, -2.0) == (8.0, 12.0)


def test_half_width_of_asymmetric_returns_mean():
    assert half_width_of(10.0, 13.0, 8.0) == pytest.approx(2.5)


def test_half_width_of_missing_bounds():
    assert half_width_of(10.0, None, None) is None
    assert half_width_of(10.0, 12.0, None) == 2.0
    assert half_width_of(10.0, None, 7.0) == 3.0
