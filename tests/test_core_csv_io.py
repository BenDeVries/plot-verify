"""Tests for plotverify_core.csv_io.load_csv."""
import pandas as pd
import pytest

from plotverify_core import load_csv, LoadReport


def test_minimal_csv():
    csv = "series,x,y\nA,1,2\nA,2,3\n"
    df, rep = load_csv(csv)
    assert df is not None
    assert rep.error is None
    assert len(df) == 2
    assert "y_err_lower" in df.columns
    assert df["y_err_lower"].isna().all()


def test_missing_required_returns_none():
    csv = "series,x\nA,1\n"
    df, rep = load_csv(csv)
    assert df is None
    assert "missing required" in rep.error


def test_drops_missing_xy():
    csv = "series,x,y\nA,1,2\nA,,3\n"
    df, rep = load_csv(csv)
    assert len(df) == 1
    assert rep.n_rows_dropped_missing_xy == 1


def test_invalid_series_color_falls_back():
    csv = "series,x,y,series_color\nA,1,2,nope\n"
    df, rep = load_csv(csv)
    assert df.loc[0, "series_color"] == "#888888"
    assert rep.n_invalid_series_colors == 1


def test_reversed_error_bars_auto_swapped():
    csv = "series,x,y,y_err_lower,y_err_upper\nA,1,2,5,1\n"
    df, rep = load_csv(csv)
    assert df.loc[0, "y_err_lower"] == 1
    assert df.loc[0, "y_err_upper"] == 5
    assert rep.n_reversed_error_bars == 1
    assert rep.reversed_samples == [("A", 1.0)]


def test_unparsable_csv_returns_error():
    df, rep = load_csv("not\x00a,real\x00csv")
    assert df is None or rep.error is not None  # tolerate either signal


def test_has_series_color_column_flag():
    df, rep = load_csv("series,x,y\nA,1,2\n")
    assert rep.has_series_color_column is False
    df, rep = load_csv("series,x,y,series_color\nA,1,2,#ff0000\n")
    assert rep.has_series_color_column is True


def test_missing_color_column_fills_palette():
    """When the CSV omits series_color, each unique series gets a palette hex.

    Regression: previously the column was filled with pd.NA, which stringified
    to "<NA>" downstream and broke Plotly's color validation in error_y.
    """
    from plotverify_core import is_valid_hex
    csv = "series,x,y\nA,1,2\nA,2,3\nB,1,5\nB,2,6\nC,1,7\n"
    df, rep = load_csv(csv)
    assert rep.has_series_color_column is False
    # Every cell must be a valid hex string — no pd.NA, no "<NA>".
    assert df["series_color"].apply(is_valid_hex).all()
    # Distinct series get distinct colors.
    by_series = df.drop_duplicates("series").set_index("series")["series_color"]
    assert by_series["A"] != by_series["B"]
    assert by_series["A"] != by_series["C"]
    assert by_series["B"] != by_series["C"]


def test_editable_overlay_handles_na_color_safely():
    """Belt-and-braces: even a DataFrame with pd.NA colors must not produce
    the literal string "<NA>" in OverlayPoint.color_hex."""
    from plotverify_core import EditableOverlay, is_valid_hex
    df = pd.DataFrame({
        "series": ["A", "A"],
        "x": [1.0, 2.0],
        "y": [10.0, 20.0],
        "series_color": [pd.NA, pd.NA],
    })
    overlay = EditableOverlay(df)
    for p in overlay.points():
        assert isinstance(p.color_hex, str)
        assert is_valid_hex(p.color_hex)
