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
