"""Regression tests for the 1-SE outlier-exclusion rule in calibrate_axis.

Captures the bug where a mispaired y=0 tick stays in the calibration despite a
large RMSE reduction available by dropping it. The 1-SE threshold must be
derived from the min-rmse candidate's own residuals — never from the full
(outlier-contaminated) initial set.
"""
import pytest

from axis_pipeline import PairedTick
from axis_pipeline.calibration import calibrate_axis


def _tick(pixel, value):
    return PairedTick(
        pixel_position=float(pixel),
        fixed_axis_pixel=0.0,
        data_value=float(value),
        pair_distance_px=0.0,
        grid_index=None,
        label_bbox=(0, 0, 0, 0),
        raw_text=str(value),
        cleaned_text=str(value),
        ocr_confidence=0.9,
        parse_status="parsed",
        flag="",
        include=True,
        status="paired_to_tick_mark",
    )


def test_outlier_y0_is_excluded_by_1se_rule():
    """A 5-point y-axis with a deliberately mispaired y=0 should reduce to 4.

    Pixel grid 100..500 mapped to values 0,25,50,75,100 is perfectly linear.
    Shift the y=0 tick's pixel by +30 to simulate a mispairing; the other 4
    points still fit the original line cleanly. A correct 1-SE rule should
    drop the outlier and recover scale=0.25.
    """
    ticks = [
        _tick(130, 0),    # mispaired: should be pixel 100, actually at 130
        _tick(200, 25),
        _tick(300, 50),
        _tick(400, 75),
        _tick(500, 100),
    ]
    cal = calibrate_axis(ticks)
    assert cal is not None
    assert cal.n_points == 4, (
        f"Expected outlier exclusion → n_points=4, got {cal.n_points}; "
        f"rmse_data={cal.rmse_data:.4f}, scale={cal.scale:.4f}"
    )
    assert cal.scale == pytest.approx(0.25, rel=1e-6)
    assert cal.rmse_data < 0.1


def test_clean_5_points_kept_intact():
    """No outlier, no greedy removal — all 5 points stay in the fit."""
    ticks = [_tick(100 * (i + 1), 25 * i) for i in range(5)]  # 0,25,50,75,100
    cal = calibrate_axis(ticks)
    assert cal is not None
    assert cal.n_points == 5
    assert cal.scale == pytest.approx(0.25, rel=1e-6)


def test_two_outliers_both_excluded():
    """Both mispaired ticks should drop out, leaving the 3 clean ones."""
    ticks = [
        _tick(130, 0),     # outlier
        _tick(200, 25),
        _tick(300, 50),
        _tick(400, 75),
        _tick(540, 100),   # outlier
    ]
    cal = calibrate_axis(ticks)
    assert cal is not None
    assert cal.n_points == 3
    assert cal.scale == pytest.approx(0.25, rel=1e-6)
