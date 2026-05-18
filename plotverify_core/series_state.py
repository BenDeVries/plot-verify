"""Per-series mask configuration (HSV ranges, ΔE threshold, interpolation flag).

This is the typed version of the dict-shaped ``series_states`` previously
built by ``_init_series_states`` in app_auto_axis.py. Streamlit widgets bind
their values to ``SeriesState`` fields rather than to free-form dict keys.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import pandas as pd

from .colors import FALLBACK_HEX, hex_to_hsv_opencv, is_valid_hex


@dataclass
class SeriesState:
    """Mask + interpolation settings for one series."""
    series: str
    color_hex: str = FALLBACK_HEX
    use_delta_e: bool = True
    delta_e: int = 10
    h_min: int = 0
    h_max: int = 179
    s_min: int = 0
    s_max: int = 255
    v_min: int = 0
    v_max: int = 255
    interpolate: bool = False
    visible: bool = True

    @classmethod
    def from_color(cls, series: str, color_hex: str) -> "SeriesState":
        """Seed reasonable HSV defaults from the series's color hex."""
        if not is_valid_hex(color_hex):
            color_hex = FALLBACK_HEX
        try:
            h, s, v = hex_to_hsv_opencv(color_hex)
        except Exception:
            h, s, v = 0, 128, 128
        return cls(
            series=series,
            color_hex=color_hex,
            h_min=max(0, h - 15),
            h_max=min(179, h + 15),
            s_min=max(0, s - 60),
            s_max=min(255, s + 60),
            v_min=max(0, v - 60),
            v_max=min(255, v + 60),
        )


def init_series_states(df: pd.DataFrame) -> Dict[str, SeriesState]:
    """Build a ``{series_name: SeriesState}`` dict from a CSV DataFrame.

    Uses the first row's ``series_color`` for each series (matches legacy
    behavior). Series with no rows in the DataFrame are not included.
    """
    states: Dict[str, SeriesState] = {}
    for series_name in df["series"].drop_duplicates().tolist():
        rows = df[df["series"] == series_name]
        color_hex = rows["series_color"].iloc[0] if len(rows) else FALLBACK_HEX
        states[series_name] = SeriesState.from_color(str(series_name), color_hex)
    return states
