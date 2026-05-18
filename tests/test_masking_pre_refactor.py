"""Pin down current masking/CV behavior.

The masking functions are pure (no Streamlit state); they take numpy arrays
and return numpy arrays. We assert shape, dtype, and a few key pixel values.
"""
import numpy as np

from app_auto_axis import _delta_e_mask, apply_color_mask


def _solid_image(color_bgr, h=20, w=30):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color_bgr
    return img


class TestDeltaEMask:
    def test_matching_color_returns_all_set(self):
        img = _solid_image((0, 0, 255))  # red in BGR
        mask = _delta_e_mask(img, "#ff0000", threshold=10.0)
        assert mask.shape == img.shape[:2]
        assert mask.dtype == np.uint8
        # All pixels should match (within tolerance).
        assert (mask > 0).all()

    def test_dissimilar_color_returns_zero(self):
        img = _solid_image((255, 255, 255))  # white
        mask = _delta_e_mask(img, "#ff0000", threshold=5.0)
        # White vs red — distance is huge — no matches expected.
        assert (mask > 0).sum() == 0


class TestApplyColorMask:
    def test_hue_wrap_around_no_match(self):
        img = _solid_image((0, 255, 0))  # green in BGR
        mask = apply_color_mask(img, 170, 10, 100, 255, 100, 255)
        # Green is hue ~60 in OpenCV's 0-179 scale; not in the [170-10] red wrap.
        assert (mask > 0).sum() == 0

    def test_in_range_matches_all(self):
        img = _solid_image((0, 255, 0))  # green
        # OpenCV BGR (0,255,0) → HSV roughly (60, 255, 255).
        mask = apply_color_mask(img, 50, 70, 200, 255, 200, 255)
        assert (mask > 0).all()
