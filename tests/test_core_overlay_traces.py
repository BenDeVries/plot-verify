"""Tests for plotverify_core.overlay_traces."""
import numpy as np
import pandas as pd
import pytest

from plotverify_core import build_overlay_traces


def _df():
    return pd.DataFrame({
        "series": ["A", "A", "A", "B", "B"],
        "x": [1.0, 2.0, 3.0, 10.0, 20.0],
        "y": [10.0, 20.0, 30.0, 5.0, 6.0],
        "y_err_lower": [9.0, 19.0, None, None, None],
        "y_err_upper": [11.0, 21.0, None, None, None],
        "series_color": ["#ff0000", "#ff0000", "#ff0000", "#0000ff", "#0000ff"],
    })


def test_two_traces():
    traces = build_overlay_traces(_df())
    assert [t.series for t in traces] == ["A", "B"]


def test_has_err_per_row():
    [a, b] = build_overlay_traces(_df())
    # A: first two rows have errors, third doesn't.
    np.testing.assert_array_equal(a.has_err, [True, True, False])
    # B: no errors at all.
    np.testing.assert_array_equal(b.has_err, [False, False])


def test_err_plus_minus_zero_where_no_err():
    [a, _] = build_overlay_traces(_df())
    np.testing.assert_array_equal(a.err_array_plus, [1.0, 1.0, 0.0])
    np.testing.assert_array_equal(a.err_array_minus, [1.0, 1.0, 0.0])


def test_ribbon_sorted_by_x():
    df = pd.DataFrame({
        "series": ["A", "A"], "x": [2.0, 1.0], "y": [20.0, 10.0],
        "y_err_lower": [19.0, 9.0], "y_err_upper": [21.0, 11.0],
        "series_color": ["#ff0000"] * 2,
    })
    [t] = build_overlay_traces(df)
    np.testing.assert_array_equal(t.ribbon_x, [1.0, 2.0])
    np.testing.assert_array_equal(t.ribbon_y_upper, [11.0, 21.0])
    np.testing.assert_array_equal(t.ribbon_y_lower, [9.0, 19.0])


def test_visibility_default_true():
    traces = build_overlay_traces(_df())
    assert all(t.visible for t in traces)


def test_visibility_override():
    traces = build_overlay_traces(_df(), series_visibility={"A": False, "B": True})
    assert {t.series: t.visible for t in traces} == {"A": False, "B": True}


def test_color_override():
    traces = build_overlay_traces(_df(), series_colors={"A": "#00ff00"})
    a = next(t for t in traces if t.series == "A")
    assert a.color_hex == "#00ff00"


def test_invalid_csv_color_falls_back():
    df = pd.DataFrame({
        "series": ["A"], "x": [1.0], "y": [2.0],
        "y_err_lower": [None], "y_err_upper": [None],
        "series_color": ["not-a-color"],
    })
    [t] = build_overlay_traces(df)
    assert t.color_hex == "#888888"


def test_missing_error_columns_does_not_crash():
    df = pd.DataFrame({
        "series": ["A"], "x": [1.0], "y": [2.0],
        "series_color": ["#ff0000"],
    })
    [t] = build_overlay_traces(df)
    assert not t.has_err.any()
    assert len(t.ribbon_x) == 0


def test_overlay_trace_point_ids_default_is_empty_list():
    """Regression: point_ids must default to a fresh empty list (no None,
    no shared mutable default)."""
    from plotverify_core.overlay_traces import OverlayTrace
    arr = np.array([])
    a = OverlayTrace(
        series="A", x=arr, y=arr, has_err=arr.astype(bool),
        err_array_plus=arr, err_array_minus=arr,
        ribbon_x=arr, ribbon_y_upper=arr, ribbon_y_lower=arr,
        color_hex="#000000", marker_color_hex="#ffffff", visible=True,
    )
    b = OverlayTrace(
        series="B", x=arr, y=arr, has_err=arr.astype(bool),
        err_array_plus=arr, err_array_minus=arr,
        ribbon_x=arr, ribbon_y_upper=arr, ribbon_y_lower=arr,
        color_hex="#000000", marker_color_hex="#ffffff", visible=True,
    )
    assert a.point_ids == []
    assert b.point_ids == []
    a.point_ids.append("A#0")
    assert b.point_ids == []  # not shared
