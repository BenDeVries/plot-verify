"""One-sided interval support: a row with only an upper *or* only a lower
bound still draws an interval, spanning the point estimate out to the bound.
Rows missing both bounds draw nothing.
"""
import types

import numpy as np
import pandas as pd

from plotverify_core import build_overlay_traces

_CAL = {"applied": True, "x_scale": 0.1, "x_offset": 0.0,
        "y_scale": -1.0, "y_offset": 100.0,
        "x_log_base": None, "y_log_base": None}


def _img():
    return np.full((100, 100, 3), 255, dtype=np.uint8)


def _ts_df():
    """Rows: two-sided, upper-only, lower-only, neither."""
    return pd.DataFrame({
        "series": ["A"] * 4,
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [10.0, 20.0, 30.0, 40.0],
        "y_err_lower": [9.0, None, 29.0, None],
        "y_err_upper": [11.0, 21.0, None, None],
        "series_color": ["#ff0000"] * 4,
    })


# ---------------------------------------------------------------------------
# Trace model
# ---------------------------------------------------------------------------

def test_has_err_includes_one_sided_rows():
    [t] = build_overlay_traces(_ts_df())
    np.testing.assert_array_equal(t.has_err, [True, True, True, False])
    np.testing.assert_array_equal(t.has_upper, [True, True, False, False])
    np.testing.assert_array_equal(t.has_lower, [True, False, True, False])


def test_missing_side_has_zero_error_offset():
    [t] = build_overlay_traces(_ts_df())
    np.testing.assert_array_equal(t.err_array_plus, [1.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(t.err_array_minus, [1.0, 0.0, 1.0, 0.0])


def test_ribbon_collapses_missing_bound_to_point_estimate():
    [t] = build_overlay_traces(_ts_df())
    # The two-bound row plus both one-sided rows; the bare row is excluded.
    np.testing.assert_array_equal(t.ribbon_x, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(t.ribbon_y_upper, [11.0, 21.0, 30.0])
    np.testing.assert_array_equal(t.ribbon_y_lower, [9.0, 20.0, 29.0])


def test_missing_bound_column_still_builds_ribbon():
    df = _ts_df().drop(columns=["y_err_lower"])
    [t] = build_overlay_traces(df)
    np.testing.assert_array_equal(t.has_err, [True, True, False, False])
    np.testing.assert_array_equal(t.ribbon_x, [1.0, 2.0])
    np.testing.assert_array_equal(t.ribbon_y_upper, [11.0, 21.0])
    np.testing.assert_array_equal(t.ribbon_y_lower, [10.0, 20.0])


def test_no_bounds_at_all_builds_no_ribbon():
    df = _ts_df().drop(columns=["y_err_lower", "y_err_upper"])
    [t] = build_overlay_traces(df)
    assert not t.has_err.any()
    assert t.ribbon_x.size == 0


def test_horizontal_one_sided_offsets_measured_from_x():
    df = pd.DataFrame({
        "series": ["A", "A"],
        "x": [5.0, 6.0], "y": [1.0, 0.0],
        "y_err_lower": [4.0, None],
        "y_err_upper": [None, 8.0],
        "series_color": ["#ff0000"] * 2,
    })
    [t] = build_overlay_traces(df, plot_type="forest")
    np.testing.assert_array_equal(t.err_array_minus, [1.0, 0.0])
    np.testing.assert_array_equal(t.err_array_plus, [0.0, 2.0])


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------

def test_time_series_caps_only_on_the_present_side():
    from shiny_app.figures import build_data_overlay_figure

    traces = build_overlay_traces(_ts_df())
    fig = build_data_overlay_figure(_img(), traces, _CAL)
    cap_u = next(t for t in fig.data if t.name == "_pv_cap_u_A")
    cap_l = next(t for t in fig.data if t.name == "_pv_cap_l_A")
    assert [float(v) for v in cap_u.y] == [11.0, 21.0]
    assert [float(v) for v in cap_l.y] == [9.0, 29.0]


def test_time_series_ribbon_rendered_for_one_sided_rows():
    from shiny_app.figures import build_data_overlay_figure

    df = _ts_df().drop(columns=["y_err_lower"])
    traces = build_overlay_traces(df)
    fig = build_data_overlay_figure(_img(), traces, _CAL)
    rib_u = next(t for t in fig.data if t.name == "_pv_rib_u_A")
    rib_l = next(t for t in fig.data if t.name == "_pv_rib_l_A")
    assert [float(v) for v in rib_u.y] == [11.0, 21.0]
    assert [float(v) for v in rib_l.y] == [10.0, 20.0]
    # The empty-side cap trace must still exist — the in-place edit push
    # looks both cap traces up by name.
    cap_l = next(t for t in fig.data if t.name == "_pv_cap_l_A")
    assert not cap_l.x


def test_forest_band_spans_point_to_single_bound():
    from shiny_app.figures import build_data_overlay_figure

    df = pd.DataFrame({
        "series": ["A"], "x": [5.0], "y": [0.0],
        "y_err_lower": [None], "y_err_upper": [8.0],
        "series_color": ["#ff0000"],
    })
    traces = build_overlay_traces(df, plot_type="forest")
    fig = build_data_overlay_figure(_img(), traces, _CAL, plot_type="forest")
    band = next(t for t in fig.data if t.name == "_pv_rib_A")
    xs = [v for v in band.x if v is not None]
    assert min(xs) == 5.0 and max(xs) == 8.0


def test_box_whisker_only_on_the_present_side():
    from shiny_app.figures import build_data_overlay_figure

    df = pd.DataFrame({
        "series": ["G"], "x": [1.0], "y": [5.0],
        "y_err_lower": [None], "y_err_upper": [9.0],
        "box_q1": [4.0], "box_median": [5.0], "box_q3": [6.0],
        "status": [""],
        "series_color": ["#ff0000"],
    })
    traces = build_overlay_traces(df, plot_type="box")
    fig = build_data_overlay_figure(_img(), traces, _CAL, plot_type="box")
    whisk = next(t for t in fig.data if t.name == "_pv_whisk_G")
    vals = [v for v in whisk.y if v is not None]
    # Upper whisker q3 -> 9 only; nothing drawn below q3.
    assert min(vals) == 6.0 and max(vals) == 9.0


def test_zoom_bubble_renders_one_sided_band():
    from shiny_app.figures import build_zoom_bubble_figure

    pt = types.SimpleNamespace(x=2.0, y=20.0, y_err_upper=21.0,
                               y_err_lower=None, color_hex="#ff0000")
    fig = build_zoom_bubble_figure(_img(), _CAL, pt, "upper")
    bub_pt = next(t for t in fig.data if t.name == "_bub_pt")
    assert float(bub_pt.error_y.array[0]) == 1.0
    assert float(bub_pt.error_y.arrayminus[0]) == 0.0
    rib = next(t for t in fig.data if t.name == "_bub_ribbon")
    assert min(float(v) for v in rib.y) == 20.0
    assert max(float(v) for v in rib.y) == 21.0
