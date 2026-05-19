"""Tests for the manual-mode overlay rendering path.

Manual calibration (no EasyOCR, no geometric frame detection) produces a
CalibrationResult with bbox=None. render_overlay must still draw the P1/P2/P3
anchor markers so the user can verify their placement. Bug #18 redesign
locks this contract in.
"""
from __future__ import annotations

import numpy as np

from axis_pipeline import manual_calibration
from axis_pipeline.overlay import render_overlay


def _white_image(h=300, w=400):
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    return img


def test_manual_calibration_yields_result_with_bbox_none():
    r = manual_calibration(
        p1_pixel=(40.0, 270.0),
        p2_pixel=(360.0, 270.0),
        p3_pixel=(40.0, 30.0),
        p1_data_x=0.0, p2_data_x=10.0,
        p1_data_y=0.0, p3_data_y=100.0,
    )
    assert r.success is True
    assert r.bbox is None, "manual mode must not synthesize a geometric bbox"


def test_render_overlay_draws_anchors_without_bbox():
    """With bbox=None, render_overlay should still mark P1/P2/P3.

    The anchor markers are magenta stars at (255, 0, 255) BGR. After conversion
    to RGB they're (255, 0, 255). Assert that pixels of that color appear at
    each anchor position (within a small radius — drawMarker rasterizes a
    star shape so the exact center pixel may not be the marker color).
    """
    img = _white_image()
    r = manual_calibration(
        p1_pixel=(40.0, 270.0),
        p2_pixel=(360.0, 270.0),
        p3_pixel=(40.0, 30.0),
        p1_data_x=0.0, p2_data_x=10.0,
        p1_data_y=0.0, p3_data_y=100.0,
    )
    rgb = render_overlay(img, r)
    assert rgb.shape == img.shape

    # The magenta marker color in RGB is (255, 0, 255).
    magenta_mask = (
        (rgb[:, :, 0] == 255) & (rgb[:, :, 1] == 0) & (rgb[:, :, 2] == 255)
    )
    assert magenta_mask.any(), "no magenta anchor markers drawn on output"

    def _anchor_has_marker(cx, cy, radius=15):
        x0 = max(0, int(cx) - radius)
        x1 = min(rgb.shape[1], int(cx) + radius + 1)
        y0 = max(0, int(cy) - radius)
        y1 = min(rgb.shape[0], int(cy) + radius + 1)
        return magenta_mask[y0:y1, x0:x1].any()

    assert _anchor_has_marker(40, 270), "no marker near P1"
    assert _anchor_has_marker(360, 270), "no marker near P2"
    assert _anchor_has_marker(40, 30), "no marker near P3"


def test_render_overlay_omits_frame_rectangle_without_bbox():
    """Without a bbox, the red frame rectangle (BGR 0,0,200 → RGB 200,0,0)
    must not be drawn. Magenta anchors should still be present.
    """
    img = _white_image()
    r = manual_calibration(
        p1_pixel=(40.0, 270.0),
        p2_pixel=(360.0, 270.0),
        p3_pixel=(40.0, 30.0),
        p1_data_x=0.0, p2_data_x=10.0,
        p1_data_y=0.0, p3_data_y=100.0,
    )
    rgb = render_overlay(img, r)

    red_frame_mask = (
        (rgb[:, :, 0] == 200) & (rgb[:, :, 1] == 0) & (rgb[:, :, 2] == 0)
    )
    assert not red_frame_mask.any(), (
        "frame rectangle was drawn despite bbox=None — the bbox-gating in "
        "render_overlay regressed."
    )


def test_render_overlay_returns_input_when_result_is_none():
    img = _white_image()
    rgb = render_overlay(img, None)
    assert rgb.shape == img.shape
    # Conversion BGR→RGB on an all-white image is a no-op.
    assert np.array_equal(rgb, img)
