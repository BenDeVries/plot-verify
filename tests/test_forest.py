"""Forest-plot support across the loader, overlay traces, and dashboard.

A forest CSV carries one estimate per row (``value``) with a horizontal
interval (``value_err_*``) and a categorical row label (``series``). The
loader normalises it onto the canonical ``x``/``y``/``y_err_*`` columns so the
rest of the pipeline keeps working; ``x`` holds the value and ``y`` is a
synthesised row index (first CSV row on top).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from plotverify_core import build_overlay_traces, load_csv
from plotverify_core.dashboard import build_forest_display_df

FIXTURE = Path(__file__).resolve().parent.parent / "test_images" / "forest_B_50.csv"


def _small_forest_csv() -> str:
    return (
        "series,value,value_err_lower,value_err_upper,is_summary,status,series_color\n"
        "beta[2],5.0,3.0,7.0,false,,#ff0000\n"
        "beta[1],-2.0,-4.0,0.0,false,note1,#00ff00\n"
        "beta[0],1.0,0.5,1.5,true,pooled,#0000ff\n"
    )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def test_forest_detected_and_normalised():
    df, rep = load_csv(_small_forest_csv())
    assert rep.error is None
    assert rep.is_forest is True
    # value → x, interval → y_err_*, and no leftover forest columns.
    for col in ("x", "y", "y_err_lower", "y_err_upper"):
        assert col in df.columns
    assert "value" not in df.columns
    np.testing.assert_array_equal(df["x"].to_numpy(), [5.0, -2.0, 1.0])


def test_forest_row_index_top_to_bottom():
    # First CSV row sits at the top with the largest index (N-1).
    df, _ = load_csv(_small_forest_csv())
    np.testing.assert_array_equal(df["y"].to_numpy(), [2.0, 1.0, 0.0])


def test_forest_is_summary_and_status_preserved():
    df, _ = load_csv(_small_forest_csv())
    assert df["is_summary"].tolist() == [False, False, True]
    assert df["is_summary"].dtype == bool
    assert df["status"].tolist() == ["", "note1", "pooled"]


def test_forest_interval_brackets_value_not_y():
    # The bounds must bracket the estimate (x), not the row index (y).
    df, _ = load_csv(_small_forest_csv())
    assert (df["y_err_lower"] <= df["x"]).all()
    assert (df["y_err_upper"] >= df["x"]).all()


def test_forest_reversed_bars_swapped_against_value():
    # value=5, lower=7 (> value), upper=3 (< value) → reversed → swapped.
    csv = "series,value,value_err_lower,value_err_upper\nA,5,7,3\n"
    df, rep = load_csv(csv)
    assert rep.n_reversed_error_bars == 1
    assert df.loc[0, "y_err_lower"] == 3
    assert df.loc[0, "y_err_upper"] == 7


def test_forest_missing_value_row_dropped():
    csv = (
        "series,value,value_err_lower,value_err_upper\n"
        "A,5,3,7\n"
        "B,,1,2\n"
    )
    df, rep = load_csv(csv)
    assert len(df) == 1
    assert rep.n_rows_dropped_missing_xy == 1
    # Row index re-synthesised after the drop: single remaining row at 0.
    assert df["y"].tolist() == [0.0]


# ---------------------------------------------------------------------------
# Overlay traces
# ---------------------------------------------------------------------------

def test_forest_error_offsets_relative_to_x():
    df, _ = load_csv(_small_forest_csv())
    traces = build_overlay_traces(df, plot_type="forest")
    by_series = {t.series: t for t in traces}
    a = by_series["beta[2]"]  # value 5, [3, 7]
    np.testing.assert_allclose(a.err_array_plus, [7.0 - 5.0])
    np.testing.assert_allclose(a.err_array_minus, [5.0 - 3.0])


def test_forest_no_ribbon():
    df, _ = load_csv(_small_forest_csv())
    for t in build_overlay_traces(df, plot_type="forest"):
        assert t.ribbon_x.size == 0
        assert t.ribbon_y_upper.size == 0


def test_forest_is_summary_and_status_on_traces():
    df, _ = load_csv(_small_forest_csv())
    by_series = {t.series: t for t in build_overlay_traces(df, plot_type="forest")}
    assert by_series["beta[0]"].is_summary.tolist() == [True]
    assert by_series["beta[1]"].status == ["note1"]


def test_non_forest_traces_have_no_forest_metadata():
    df, _ = load_csv(_small_forest_csv())
    # Rendered as a time series, forest metadata must not leak in.
    for t in build_overlay_traces(df, plot_type="time_series"):
        assert t.is_summary is None
        assert t.status is None


# ---------------------------------------------------------------------------
# Dashboard reuse
# ---------------------------------------------------------------------------

def test_forest_dashboard_raw_view_has_no_sigma():
    df, _ = load_csv(_small_forest_csv())
    out = build_forest_display_df(df, None)
    assert list(out["Row"]) == ["beta[2]", "beta[1]", "beta[0]"]
    assert "σ" not in out.columns
    # Half-width = (upper - lower) / 2.
    np.testing.assert_allclose(out["Half-width"].to_numpy(), [2.0, 2.0, 0.5])


def test_forest_dashboard_sd_sigma_equals_half_width():
    df, _ = load_csv(_small_forest_csv())
    out = build_forest_display_df(df, "SD")
    np.testing.assert_allclose(out["σ"].to_numpy(), out["Half-width"].to_numpy())


def test_forest_dashboard_se_sigma_scales_by_sqrt_n():
    df, _ = load_csv(_small_forest_csv())
    out = build_forest_display_df(df, "SE", n=9)
    np.testing.assert_allclose(
        out["σ"].to_numpy(), out["Half-width"].to_numpy() * 3.0
    )


def test_forest_dashboard_status_column_only_when_present():
    df, _ = load_csv(_small_forest_csv())
    assert "Status" in build_forest_display_df(df, None).columns
    # A forest frame with no status notes omits the column.
    blank = df.copy()
    blank["status"] = ""
    assert "Status" not in build_forest_display_df(blank, None).columns


# ---------------------------------------------------------------------------
# Real fixture
# ---------------------------------------------------------------------------

def test_fixture_loads_as_forest():
    df, rep = load_csv(FIXTURE.read_text())
    assert rep.is_forest is True
    assert len(df) == 50
    # Evenly spaced descending row indices, top row = N-1.
    np.testing.assert_array_equal(
        df["y"].to_numpy(), np.arange(49, -1, -1, dtype=float)
    )
