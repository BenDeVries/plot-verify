"""Source-image compositing for the Overlay tab's ΔE mask preview.

The Shiny overlay can optionally repaint pixels of the source image that match
a series's chosen color so the underlying plot lines visually "disappear" and
the overlay points stand out on the background. This module is the pure helper
that produces that composite from a list of (color, threshold) specs.
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np

from .colors import detect_background_color, is_valid_hex
from .masking import delta_e_mask


SeriesMaskSpec = Tuple[str, float]  # (color_hex, delta_e_threshold)


def build_masked_overlay_image(
    img_bgr: np.ndarray,
    specs: Sequence[SeriesMaskSpec],
    *,
    background_bgr: Tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Return an RGB image with masked pixels painted in the background color.

    ``specs`` is a sequence of ``(color_hex, delta_e_threshold)`` pairs. Each
    spec's ΔE-Lab mask is computed against the source image; the union of all
    masks is then painted with ``background_bgr`` (auto-detected if None).

    Returns an RGB ndarray suitable for handing to ``encode_image_data_uri``.
    When ``specs`` is empty the source is returned converted to RGB.
    """
    if background_bgr is None:
        background_bgr = detect_background_color(img_bgr)
    bg_color = np.array(background_bgr, dtype=np.uint8)

    out = img_bgr.copy()
    union_mask: np.ndarray | None = None
    for color_hex, threshold in specs:
        if not is_valid_hex(color_hex):
            continue
        m = delta_e_mask(img_bgr, color_hex, float(threshold))
        union_mask = m if union_mask is None else cv2.bitwise_or(union_mask, m)

    if union_mask is not None:
        out[union_mask > 0] = bg_color

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
