"""Tests for plotverify_core.image_io."""
import io

import cv2
import numpy as np
import pytest
from PIL import Image

from plotverify_core import decode_and_maybe_downscale, decode_image_bytes


def _png_bytes(w: int, h: int) -> bytes:
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 1] = 255  # solid green
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def test_decode_small():
    b = _png_bytes(100, 80)
    load = decode_image_bytes(b)
    assert load.error is None
    assert load.img_bgr is not None
    assert load.img_bgr.shape == (80, 100, 3)
    assert load.image_hash != ""
    assert load.downscale_factor == 1.0


def test_decode_invalid_bytes_returns_error():
    load = decode_image_bytes(b"not an image")
    assert load.error is not None
    assert load.img_bgr is None


def test_large_image_warns_but_does_not_downscale():
    b = _png_bytes(5000, 100)
    load = decode_and_maybe_downscale(b, downscale=False, max_edge=4000)
    assert load.error is None
    assert load.img_bgr.shape[1] == 5000
    assert load.downscale_factor == 1.0
    assert any("Large image" in w for w in load.warnings)


def test_large_image_with_downscale():
    b = _png_bytes(5000, 100)
    load = decode_and_maybe_downscale(b, downscale=True, max_edge=4000,
                                       downscale_to=3000)
    assert load.img_bgr.shape[1] == 3000
    assert load.downscale_factor == pytest.approx(3000 / 5000)
    assert any("Downscaled" in w for w in load.warnings)


def test_hash_is_stable():
    b = _png_bytes(50, 50)
    a = decode_image_bytes(b)
    c = decode_image_bytes(b)
    assert a.image_hash == c.image_hash
