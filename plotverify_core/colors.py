"""Color helpers: hex validation, hex↔HSV/BGR, and complementary colour pick.

All functions are pure; no Streamlit or Shiny imports. The dark-color branch of
`hex_complement` is intentionally non-deterministic (high-contrast random light
colour) — the contrast requirement matters more than reproducibility.
"""
from __future__ import annotations

import colorsys
from typing import Iterable, List, Tuple

import numpy as np


FALLBACK_HEX = "#888888"

# Plotly's "D3" qualitative palette. Used to assign default per-series colors
# when the CSV does not supply a `series_color` column.
DEFAULT_PALETTE: Tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


def assign_palette_colors(series_names: Iterable[str]) -> List[Tuple[str, str]]:
    """Cycle through ``DEFAULT_PALETTE`` to assign a color per unique series.

    Returns a list of ``(series_name, hex)`` pairs in the input order. Duplicate
    names share a color (first occurrence wins).
    """
    seen: List[str] = []
    for n in series_names:
        if n not in seen:
            seen.append(n)
    return [(name, DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)])
            for i, name in enumerate(seen)]


def detect_background_color(img_bgr: np.ndarray) -> Tuple[int, int, int]:
    """Return the BGR triplet of the most common luminance bucket.

    Used by the masking compositor: pixels matched by a series's ΔE mask are
    repainted in this color so the series visually disappears from the source
    image (revealing the overlay drawn on top).
    """
    import cv2  # local import keeps the pure top-of-file lightweight
    grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hist = np.bincount(grey.ravel(), minlength=256)
    bg = int(np.argmax(hist))
    return (bg, bg, bg)

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
