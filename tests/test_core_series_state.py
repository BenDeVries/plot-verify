"""Tests for plotverify_core.series_state."""
import pandas as pd
import pytest

from plotverify_core import SeriesState, init_series_states


def test_seriesstate_from_red():
    s = SeriesState.from_color("A", "#ff0000")
    assert s.series == "A"
    assert s.color_hex == "#ff0000"
    # Red is hue ~0; window should clamp at 0.
    assert s.h_min == 0
    assert s.h_max in range(0, 30)
    assert s.use_delta_e is True
    assert s.visible is True


def test_seriesstate_invalid_color_falls_back():
    s = SeriesState.from_color("X", "garbage")
    assert s.color_hex == "#888888"


def test_init_series_states_from_df():
    df = pd.DataFrame({
        "series": ["A", "A", "B"],
        "series_color": ["#ff0000", "#ff0000", "#00ff00"],
    })
    states = init_series_states(df)
    assert set(states.keys()) == {"A", "B"}
    assert states["A"].color_hex == "#ff0000"
    assert states["B"].color_hex == "#00ff00"
