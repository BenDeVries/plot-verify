"""Tests for plotverify_core.overlay_model."""
import math

import pandas as pd
import pytest

from plotverify_core import EditableOverlay


def _basic_overlay():
    df = pd.DataFrame({
        "series": ["A", "A", "B"],
        "x": [1.0, 2.0, 5.0],
        "y": [10.0, 20.0, 7.0],
        "y_err_lower": [9.0, 19.0, None],
        "y_err_upper": [11.0, 21.0, None],
        "series_color": ["#ff0000", "#ff0000", "#0000ff"],
    })
    return EditableOverlay(df)


def test_loads_with_stable_ids():
    ov = _basic_overlay()
    ids = [p.point_id for p in ov.points()]
    assert ids == ["A#0", "A#1", "B#2"]
    assert len(ov) == 3


def test_initial_state_has_no_edits():
    ov = _basic_overlay()
    assert not ov.has_edits()
    for p in ov.points():
        assert p.edited is False
        assert p.edit_type is None


def test_edit_point_marks_edited():
    ov = _basic_overlay()
    ov.edit_point("A#0", new_x=1.5, new_y=12.0)
    p = ov.get("A#0")
    assert p.x == 1.5
    assert p.y == 12.0
    assert p.edited is True
    assert p.edit_type == "point"
    # Originals preserved.
    assert p.original_x == 1.0
    assert p.original_y == 10.0
    assert ov.has_edits()


def test_edit_err_upper_preserves_xy():
    ov = _basic_overlay()
    ov.edit_err_upper("A#0", 13.0)
    p = ov.get("A#0")
    assert p.y_err_upper == 13.0
    assert p.x == 1.0  # unchanged
    assert p.y == 10.0
    assert p.edit_type == "err_upper"
    assert p.original_y_err_upper == 11.0


def test_edit_err_lower_to_none():
    ov = _basic_overlay()
    ov.edit_err_lower("A#1", None)
    assert ov.get("A#1").y_err_lower is None


def test_reset_point_clears_edits():
    ov = _basic_overlay()
    ov.edit_point("A#0", 99.0, 99.0)
    ov.reset_point("A#0")
    p = ov.get("A#0")
    assert p.x == 1.0
    assert p.y == 10.0
    assert p.edited is False


def test_to_dataframe_basic():
    ov = _basic_overlay()
    ov.edit_point("A#0", 1.5, 12.0)
    df = ov.to_dataframe()
    assert list(df.columns) == ["series", "x", "y", "y_err_lower", "y_err_upper", "series_color"]
    assert df.loc[0, "x"] == 1.5
    assert df.loc[0, "y"] == 12.0


def test_to_dataframe_with_audit_cols():
    ov = _basic_overlay()
    ov.edit_point("A#0", 1.5, 12.0)
    df = ov.to_dataframe(include_audit_cols=True)
    assert "original_x" in df.columns
    assert "edited" in df.columns
    assert df.loc[0, "original_x"] == 1.0
    assert bool(df.loc[0, "edited"]) is True
    # Untouched points should have edited=False.
    assert bool(df.loc[1, "edited"]) is False
    assert df.loc[1, "edit_type"] == ""


def test_round_trip_no_edits_preserves_values():
    ov = _basic_overlay()
    out = ov.to_dataframe()
    assert list(out["x"].values) == [1.0, 2.0, 5.0]
    assert list(out["y"].values) == [10.0, 20.0, 7.0]


def test_series_names_preserves_first_seen_order():
    ov = _basic_overlay()
    assert ov.series_names() == ["A", "B"]


def test_handles_missing_error_columns():
    df = pd.DataFrame({
        "series": ["A"], "x": [1.0], "y": [2.0],
        "series_color": ["#888888"],
    })
    ov = EditableOverlay(df)
    p = ov.get("A#0")
    assert p.y_err_lower is None
    assert p.y_err_upper is None


# ---------------------------------------------------------------------------
# Batch mutators
# ---------------------------------------------------------------------------

def test_nudge_points_gang_moves_with_audit():
    ov = _basic_overlay()
    pids = [p.point_id for p in ov.points()][:2]
    before = {p.point_id: (p.x, p.y, p.y_err_upper, p.y_err_lower)
              for p in ov.points()}
    ov.nudge_points(pids, 0.5, 1.0)
    for pid in pids:
        p = ov.get(pid)
        bx, by, bu, bl = before[pid]
        assert p.x == bx + 0.5
        assert p.y == by + 1.0
        if bu is not None:
            assert p.y_err_upper == bu + 1.0
        if bl is not None:
            assert p.y_err_lower == bl + 1.0
        assert p.edited is True
        assert p.edit_timestamp is not None


def test_nudge_points_unknown_pid_raises_before_mutating():
    ov = _basic_overlay()
    pids = [p.point_id for p in ov.points()]
    with pytest.raises(KeyError):
        ov.nudge_points([pids[0], "Nope#99"], 1.0, 1.0)
    # nothing was applied
    assert not ov.get(pids[0]).edited


def test_reset_points_clears_flags():
    ov = _basic_overlay()
    pids = [p.point_id for p in ov.points()][:2]
    ov.nudge_points(pids, 1.0, 1.0)
    ov.reset_points(pids)
    for pid in pids:
        p = ov.get(pid)
        assert p.edited is False
        assert p.x == p.original_x
        assert p.y == p.original_y
