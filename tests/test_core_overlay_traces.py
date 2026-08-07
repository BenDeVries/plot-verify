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


# ---------------------------------------------------------------------------
# Bar plot traces
# ---------------------------------------------------------------------------

def test_bar_traces_no_ribbon():
    """Bar traces should have empty ribbon arrays (no connecting band)."""
    [a, _] = build_overlay_traces(_df(), plot_type="bar")
    assert a.ribbon_x.size == 0
    assert a.ribbon_y_upper.size == 0


def test_bar_traces_have_vertical_error():
    """Bar traces use standard vertical error offsets (bracket y, not x)."""
    [a, _] = build_overlay_traces(_df(), plot_type="bar")
    np.testing.assert_array_equal(a.err_array_plus, [1.0, 1.0, 0.0])
    np.testing.assert_array_equal(a.err_array_minus, [1.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# Box plot traces
# ---------------------------------------------------------------------------

def _box_df():
    return pd.DataFrame({
        "series": ["A", "A", "A"],
        "x": [1.0, 2.0, 3.0],
        "y": [5.0, 6.0, 10.0],
        "y_err_lower": [2.0, 3.0, None],
        "y_err_upper": [8.0, 9.0, None],
        "box_q1": [3.0, 4.0, None],
        "box_median": [5.0, 6.0, None],
        "box_q3": [7.0, 8.0, None],
        "status": ["", "", "outlier"],
        "series_color": ["#ff0000"] * 3,
    })


def test_box_traces_carry_quartiles():
    [t] = build_overlay_traces(_box_df(), plot_type="box")
    assert t.box_q1 is not None
    np.testing.assert_array_equal(t.box_q1[:2], [3.0, 4.0])
    assert t.box_median is not None
    np.testing.assert_array_equal(t.box_median[:2], [5.0, 6.0])
    assert t.box_q3 is not None
    np.testing.assert_array_equal(t.box_q3[:2], [7.0, 8.0])


def test_box_traces_carry_status():
    [t] = build_overlay_traces(_box_df(), plot_type="box")
    assert t.status == ["", "", "outlier"]


def test_box_traces_no_ribbon():
    [t] = build_overlay_traces(_box_df(), plot_type="box")
    assert t.ribbon_x.size == 0


def test_box_quartiles_none_for_non_box():
    [t] = build_overlay_traces(_box_df(), plot_type="time_series")
    assert t.box_q1 is None
    assert t.box_median is None
    assert t.box_q3 is None


# ---------------------------------------------------------------------------
# Kaplan-Meier traces
# ---------------------------------------------------------------------------

def _km_df():
    return pd.DataFrame({
        "series": ["Arm A"] * 4,
        "x": [0.0, 1.0, 2.0, 3.0],
        "y": [1.0, 0.8, 0.6, 0.6],
        "y_err_lower": [1.0, 0.7, 0.5, 0.5],
        "y_err_upper": [1.0, 0.9, 0.7, 0.7],
        "at_risk": [100, 80, 50, 30],
        "status": ["", "", "censored", ""],
        "series_color": ["#0000ff"] * 4,
    })


def test_km_traces_carry_at_risk():
    [t] = build_overlay_traces(_km_df(), plot_type="kaplan_meier")
    assert t.at_risk is not None
    np.testing.assert_array_equal(t.at_risk, [100, 80, 50, 30])


def test_km_traces_carry_status():
    [t] = build_overlay_traces(_km_df(), plot_type="kaplan_meier")
    assert t.status == ["", "", "censored", ""]


def test_km_traces_have_ribbon():
    [t] = build_overlay_traces(_km_df(), plot_type="kaplan_meier")
    assert t.ribbon_x.size > 0
    assert t.ribbon_y_upper.size > 0


def test_km_at_risk_none_for_non_km():
    [t] = build_overlay_traces(_km_df(), plot_type="time_series")
    assert t.at_risk is None
