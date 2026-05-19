"""Color helpers: hex validation, hex↔HSV/BGR, and complementary colour pick.

All functions are pure; no Streamlit or Shiny imports. The dark-color branch of
`hex_complement` is intentionally non-deterministic (high-contrast random light
colour) — the contrast requirement matters more than reproducibility.
"""
from __future__ import annotations

import colorsys
from typing import Tuple

import numpy as np


FALLBACK_HEX = "#888888"

_rng = np.random.default_rng(12345)


def is_valid_hex(s) -> bool:
    """Return True if ``s`` is a 6-digit hex color (with or without leading #)."""
    if not isinstance(s, str):
        return False
    h = s.lstrip("#")
    if len(h) != 6:
        return False
    try:
        int(h, 16)
    except ValueError:
        return False
    return True


def hex_to_hsv_opencv(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex to OpenCV HSV (H: 0-179, S: 0-255, V: 0-255)."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color!r}")
    r, g, b = [int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return int(round(h * 179)), int(round(s * 255)), int(round(v * 255))


def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex to a BGR int tuple (for OpenCV drawing).

    Returns mid-grey when the input is not a valid hex string, so callers can
    feed in CSV values without pre-validating each row.
    """
    if not is_valid_hex(hex_color):
        return (136, 136, 136)
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b, g, r)


def hex_complement(hex_color: str) -> str:
    """Return the hue-opposite color of ``hex_color``.

    Extremely dark colors (value < 0.25) return a random light color instead,
    because a strict hue-opposite stays dark and gives poor contrast.
    """
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    hue, sat, val = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    if val < 0.25:
        r, g, b = colorsys.hsv_to_rgb(
            _rng.random(),
            _rng.uniform(0.25, 0.55),
            _rng.uniform(0.9, 1.0),
        )
        return "#{:02x}{:02x}{:02x}".format(
            int(round(r * 255)),
            int(round(g * 255)),
            int(round(b * 255)),
        )
    comp_hue = (hue + 0.5) % 1.0
    cr, cg, cb = colorsys.hsv_to_rgb(comp_hue, sat, val)
    return "#{:02x}{:02x}{:02x}".format(
        int(round(cr * 255)),
        int(round(cg * 255)),
        int(round(cb * 255)),
    )
