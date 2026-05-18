"""Pin down current CSV-loader behavior.

Streamlit IS importable here, so we call `_load_csv` directly. The function
calls `st.warning` / `st.error` internally — those are no-ops outside a session
runtime, so tests focus on the return DataFrame's columns, dtypes, and
auto-swap behaviour.
"""
import numpy as np
import pandas as pd
import pytest

from app_auto_axis import _load_csv


def test_minimal_csv_no_error_cols():
    csv = "series,x,y\nA,1,2\nA,2,3\n"
    df = _load_csv(csv)
    assert df is not None
    assert "y_err_lower" in df.columns
    assert "y_err_upper" in df.columns
    assert df["y_err_lower"].isna().all()
    assert df["y_err_upper"].isna().all()
    assert len(df) == 2


def test_csv_with_error_cols():
    csv = "series,x,y,y_err_lower,y_err_upper\nA,1,2,1,3\n"
    df = _load_csv(csv)
    assert df.loc[0, "y_err_lower"] == 1
    assert df.loc[0, "y_err_upper"] == 3


def test_reversed_error_bars_are_auto_swapped():
    # y=2, y_err_lower=5 > y, y_err_upper=1 < y → reversed
    csv = "series,x,y,y_err_lower,y_err_upper\nA,1,2,5,1\n"
    df = _load_csv(csv)
    # After auto-swap, lower should be ≤ y ≤ upper.
    assert df.loc[0, "y_err_lower"] == 1
    assert df.loc[0, "y_err_upper"] == 5


def test_missing_required_columns_returns_none():
    csv = "series,x\nA,1\n"
    df = _load_csv(csv)
    assert df is None


def test_drops_rows_with_missing_xy():
    csv = "series,x,y\nA,1,2\nA,,3\nA,4,\n"
    df = _load_csv(csv)
    # Only one row with both x and y present.
    assert len(df) == 1


def test_series_color_invalid_falls_back():
    csv = "series,x,y,series_color\nA,1,2,not-a-color\n"
    df = _load_csv(csv)
    assert df.loc[0, "series_color"] == "#888888"


def test_csv_with_missing_series_color_column():
    csv = "series,x,y\nA,1,2\n"
    df = _load_csv(csv)
    # The column should exist (NA-filled) so downstream code can read it.
    assert "series_color" in df.columns
