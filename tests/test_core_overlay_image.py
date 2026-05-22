"""Tests for plotverify_core.overlay_image.build_masked_overlay_image."""
import numpy as np

from plotverify_core import (
    build_masked_overlay_image,
    detect_background_color,
)


def _img_with_red_square():
    """40x40 white image with a 10x10 saturated-red square at (10..20)."""
    img = np.full((40, 40, 3), 255, dtype=np.uint8)
    img[10:20, 10:20] = (40, 40, 200)  # BGR red
    return img


def test_detect_background_color_finds_white():
    img = _img_with_red_square()
    assert detect_background_color(img) == (255, 255, 255)


def test_masked_image_paints_match_with_background():
    img = _img_with_red_square()
    out = build_masked_overlay_image(img, [("#c83232", 20.0)])
    # Red square should now read white (background).
    assert tuple(int(c) for c in out[15, 15]) == (255, 255, 255)
    # Untouched area stays white.
    assert tuple(int(c) for c in out[5, 5]) == (255, 255, 255)


def test_masked_image_no_specs_returns_rgb_copy():
    img = _img_with_red_square()
    out = build_masked_overlay_image(img, [])
    # No specs → unchanged content, just BGR→RGB-converted.
    # The red square in BGR (40, 40, 200) → in RGB (200, 40, 40).
    assert tuple(int(c) for c in out[15, 15]) == (200, 40, 40)


def test_masked_image_skips_invalid_hex():
    """Non-hex specs are ignored rather than raising."""
    img = _img_with_red_square()
    out = build_masked_overlay_image(img, [("not-a-color", 10.0)])
    # Untouched: red square still shows as red in RGB.
    assert tuple(int(c) for c in out[15, 15]) == (200, 40, 40)
