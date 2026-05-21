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


def test_image_load_warnings_default_is_empty_list():
    """Regression: warnings field must default to a fresh empty list (no None,
    no shared mutable default)."""
    from plotverify_core.image_io import ImageLoad
    a = ImageLoad(img_bgr=None, img_rgb=None, image_hash="x")
    b = ImageLoad(img_bgr=None, img_rgb=None, image_hash="y")
    assert a.warnings == []
    assert b.warnings == []
    a.warnings.append("only on a")
    assert b.warnings == []  # not shared


def test_decode_and_maybe_downscale_raises_on_silent_none():
    """If decode_image_bytes returned no error but also no array, surface the
    inconsistency as a RuntimeError rather than asserting (which is stripped
    under `python -O`)."""
    import plotverify_core.image_io as image_io

    def fake_decode(_b):
        return image_io.ImageLoad(img_bgr=None, img_rgb=None,
                                   image_hash="x", error=None)

    orig = image_io.decode_image_bytes
    image_io.decode_image_bytes = fake_decode
    try:
        with pytest.raises(RuntimeError, match="no image array"):
            image_io.decode_and_maybe_downscale(b"")
    finally:
        image_io.decode_image_bytes = orig
