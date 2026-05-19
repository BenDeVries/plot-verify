"""Color masking primitives.

`delta_e_mask` operates in CIE Lab space and is the preferred path; HSV
masking is exposed for legacy compatibility and fine-grained user control.
"""
from __future__ import annotations

import cv2
import numpy as np


def delta_e_mask(img_bgr: np.ndarray, hex_color: str,
                 threshold: float = 10.0) -> np.ndarray:
    """Return a binary mask of pixels within ``threshold`` ΔE76 of ``hex_color``.

    Works in CIE Lab — perceptually uniform, so equal distance = equal
    perceived colour difference. Much more reliable than HSV for plot colours
    near grey or near the hue wrap-around.
    """
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)

    h = hex_color.lstrip("#")
    rgb = np.uint8([[[int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)]]])
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    target_lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32)[0, 0]

    delta_e = np.sqrt(((img_lab - target_lab) ** 2).sum(axis=2))
    mask = (delta_e < threshold).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(mask, kernel, iterations=2)


def apply_color_mask(img_bgr: np.ndarray,
                     h_min: int, h_max: int,
                     s_min: int, s_max: int,
                     v_min: int, v_max: int) -> np.ndarray:
    """Build a binary mask isolating pixels in the given HSV range.

    Handles hue wraparound: when ``h_min > h_max`` the range is
    ``[h_min, 179] ∪ [0, h_max]`` (needed for reds spanning 0/179). Dilates
    the result 2x with a 3x3 kernel to fill anti-aliasing fringes around lines.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    if h_min <= h_max:
        mask = cv2.inRange(
            hsv,
            np.array([h_min, s_min, v_min], dtype=np.uint8),
            np.array([h_max, s_max, v_max], dtype=np.uint8),
        )
    else:
        mask1 = cv2.inRange(
            hsv,
            np.array([h_min, s_min, v_min], dtype=np.uint8),
            np.array([179, s_max, v_max], dtype=np.uint8),
        )
        mask2 = cv2.inRange(
            hsv,
            np.array([0, s_min, v_min], dtype=np.uint8),
            np.array([h_max, s_max, v_max], dtype=np.uint8),
        )
        mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(mask, kernel, iterations=2)
