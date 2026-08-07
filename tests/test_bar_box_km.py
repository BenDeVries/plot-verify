"""End-to-end tests for bar, box, and kaplan_meier plot types.

Follows the same pattern as ``test_forest.py``: load from JSON, build
overlay traces, verify column preservation and metadata survival.
"""
import json

import numpy as np
import pandas as pd

from plotverify_core import build_overlay_traces
from plotverify_core.csv_io import validate_and_normalize
from plotverify_core.json_io import parse_agent_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap_json(plot_type, rows, *, axes=None, series=None):
    doc = {
        "schema_version": "1.0",
        "image": {"filename": "test.png", "width_px": 100, "height_px": 100},
        "plot_type": plot_type,
        "axes": axes or {
            "x": {"scale": "linear",
                   "calibration": [{"pixel": 0, "value": 0},
                                   {"pixel": 100, "value": 10}]},
            "y": {"scale": "linear",
                   "calibration": [{"pixel": 0, "value": 100},
                                   {"pixel": 100, "value": 0}]},
        },
        "rows": rows,
        "series": series or [],
    }
    return json.dumps(doc)


# ---------------------------------------------------------------------------
# Bar
# ---------------------------------------------------------------------------

_BAR_ROWS = [
    {"series": "Control", "x": 1, "y": 5.0, "y_err_lower": 3.0, "y_err_upper": 7.0,
     "series_color": "#ff0000"},
    {"series": "Control", "x": 2, "y": 8.0, "y_err_lower": 6.0, "y_err_upper": 10.0,
     "series_color": "#ff0000"},
    {"series": "Treatment", "x": 1, "y": 6.0, "y_err_lower": 4.5, "y_err_upper": 7.5,
     "series_color": "#0000ff"},
]


def test_bar_json_maps_to_bar():
    result = parse_agent_json(_wrap_json("bar", _BAR_ROWS))
    assert result.plot_type == "bar"
    assert result.csv_df is not None


def test_bar_traces_no_ribbon():
    result = parse_agent_json(_wrap_json("bar", _BAR_ROWS))
    traces = build_overlay_traces(result.csv_df, plot_type="bar")
    for t in traces:
        assert t.ribbon_x.size == 0


def test_bar_traces_vertical_error():
    result = parse_agent_json(_wrap_json("bar", _BAR_ROWS))
    traces = build_overlay_traces(result.csv_df, plot_type="bar")
    ctrl = next(t for t in traces if t.series == "Control")
    assert ctrl.has_err.any()
    np.testing.assert_allclose(ctrl.err_array_plus[:2], [2.0, 2.0])
    np.testing.assert_allclose(ctrl.err_array_minus[:2], [2.0, 2.0])


# ---------------------------------------------------------------------------
# Box
# ---------------------------------------------------------------------------

_BOX_ROWS = [
    {"series": "Group A", "x": 1, "y": 5.0,
     "y_err_lower": 1.0, "y_err_upper": 9.0,
     "box_q1": 3.0, "box_median": 5.0, "box_q3": 7.0,
     "status": "", "series_color": "#ff0000"},
    {"series": "Group A", "x": 1, "y": 12.0,
     "status": "outlier", "series_color": "#ff0000"},
    {"series": "Group B", "x": 2, "y": 6.0,
     "y_err_lower": 2.0, "y_err_upper": 10.0,
     "box_q1": 4.0, "box_median": 6.0, "box_q3": 8.0,
     "status": "", "series_color": "#0000ff"},
]


def test_box_json_maps_to_box():
    result = parse_agent_json(_wrap_json("box", _BOX_ROWS))
    assert result.plot_type == "box"


def test_box_quartile_columns_preserved():
    result = parse_agent_json(_wrap_json("box", _BOX_ROWS))
    df = result.csv_df
    assert "box_q1" in df.columns
    assert "box_median" in df.columns
    assert "box_q3" in df.columns
    assert df.loc[0, "box_q1"] == 3.0


def test_box_status_preserved():
    result = parse_agent_json(_wrap_json("box", _BOX_ROWS))
    df = result.csv_df
    assert "status" in df.columns
    assert df.loc[1, "status"] == "outlier"


def test_box_traces_carry_quartiles():
    result = parse_agent_json(_wrap_json("box", _BOX_ROWS))
    traces = build_overlay_traces(result.csv_df, plot_type="box")
    ga = next(t for t in traces if t.series == "Group A")
    assert ga.box_q1 is not None
    assert ga.box_median is not None
    assert ga.box_q3 is not None


def test_box_traces_carry_status():
    result = parse_agent_json(_wrap_json("box", _BOX_ROWS))
    traces = build_overlay_traces(result.csv_df, plot_type="box")
    ga = next(t for t in traces if t.series == "Group A")
    assert "outlier" in ga.status


def test_box_traces_no_ribbon():
    result = parse_agent_json(_wrap_json("box", _BOX_ROWS))
    for t in build_overlay_traces(result.csv_df, plot_type="box"):
        assert t.ribbon_x.size == 0


def test_box_numeric_coercion():
    rows = [
        {"series": "A", "x": 1, "y": 5.0,
         "box_q1": "3.0", "box_median": "5.0", "box_q3": "7.0",
         "series_color": "#ff0000"},
    ]
    result = parse_agent_json(_wrap_json("box", rows))
    assert result.csv_df["box_q1"].dtype == float


# ---------------------------------------------------------------------------
# Kaplan-Meier
# ---------------------------------------------------------------------------

_KM_ROWS = [
    {"series": "Drug A", "x": 0, "y": 1.0,
     "y_err_lower": 1.0, "y_err_upper": 1.0,
     "at_risk": 100, "status": "", "series_color": "#ff0000"},
    {"series": "Drug A", "x": 6, "y": 0.85,
     "y_err_lower": 0.75, "y_err_upper": 0.95,
     "at_risk": 80, "status": "", "series_color": "#ff0000"},
    {"series": "Drug A", "x": 12, "y": 0.85,
     "at_risk": 50, "status": "censored", "series_color": "#ff0000"},
    {"series": "Drug A", "x": 18, "y": 0.70,
     "y_err_lower": 0.55, "y_err_upper": 0.85,
     "at_risk": 30, "status": "", "series_color": "#ff0000"},
]


def test_km_json_maps_to_kaplan_meier():
    result = parse_agent_json(_wrap_json("kaplan_meier", _KM_ROWS))
    assert result.plot_type == "kaplan_meier"


def test_km_at_risk_preserved():
    result = parse_agent_json(_wrap_json("kaplan_meier", _KM_ROWS))
    df = result.csv_df
    assert "at_risk" in df.columns
    np.testing.assert_array_equal(df["at_risk"].to_numpy(), [100, 80, 50, 30])


def test_km_status_preserved():
    result = parse_agent_json(_wrap_json("kaplan_meier", _KM_ROWS))
    df = result.csv_df
    assert df.loc[2, "status"] == "censored"


def test_km_traces_carry_at_risk():
    result = parse_agent_json(_wrap_json("kaplan_meier", _KM_ROWS))
    traces = build_overlay_traces(result.csv_df, plot_type="kaplan_meier")
    [t] = traces
    assert t.at_risk is not None
    np.testing.assert_array_equal(t.at_risk, [100, 80, 50, 30])


def test_km_traces_carry_status():
    result = parse_agent_json(_wrap_json("kaplan_meier", _KM_ROWS))
    traces = build_overlay_traces(result.csv_df, plot_type="kaplan_meier")
    [t] = traces
    assert t.status[2] == "censored"


def test_km_traces_have_ribbon():
    result = parse_agent_json(_wrap_json("kaplan_meier", _KM_ROWS))
    traces = build_overlay_traces(result.csv_df, plot_type="kaplan_meier")
    [t] = traces
    assert t.ribbon_x.size > 0


def test_km_at_risk_numeric_coercion():
    rows = [
        {"series": "A", "x": 0, "y": 1.0,
         "at_risk": "100", "series_color": "#ff0000"},
    ]
    result = parse_agent_json(_wrap_json("kaplan_meier", rows))
    assert pd.api.types.is_numeric_dtype(result.csv_df["at_risk"])


# ---------------------------------------------------------------------------
# validate_and_normalize: status/is_summary available outside forest
# ---------------------------------------------------------------------------

def test_status_normalized_for_non_forest():
    df = pd.DataFrame({
        "series": ["A"], "x": [1.0], "y": [2.0],
        "status": ["outlier"],
    })
    out, _ = validate_and_normalize(df, is_forest=False)
    assert out["status"].iloc[0] == "outlier"


def test_is_summary_normalized_for_non_forest():
    df = pd.DataFrame({
        "series": ["A"], "x": [1.0], "y": [2.0],
        "is_summary": ["true"],
    })
    out, _ = validate_and_normalize(df, is_forest=False)
    assert bool(out["is_summary"].iloc[0]) is True


def test_status_defaults_when_missing():
    df = pd.DataFrame({"series": ["A"], "x": [1.0], "y": [2.0]})
    out, _ = validate_and_normalize(df, is_forest=False)
    assert "status" in out.columns
    assert out["status"].iloc[0] == ""
