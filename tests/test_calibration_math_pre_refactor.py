"""Pin down current calibration-math behavior in app_auto_axis.py.

After Refactor A, the same tests must pass against plotverify_core.calibration_math.
"""
import math

import numpy as np
import pytest

from app_auto_axis import (
    compute_calibration,
    data_to_px,
    px_to_data,
    _log10_or_none,
    P1P2_Y_TOLERANCE_PX,
)


class TestLog10OrNone:
    def test_no_log_returns_float(self):
        assert _log10_or_none(42.0, None) == 42.0

    def test_log_base_10_positive(self):
        assert _log10_or_none(100.0, 10.0) == pytest.approx(2.0)

    def test_log_base_10_non_positive_returns_none(self):
        assert _log10_or_none(0.0, 10.0) is None
        assert _log10_or_none(-1.0, 10.0) is None

    def test_log_base_other_passes_through(self):
        # Only log10 is supported.
        assert _log10_or_none(5.0, 2.0) == 5.0


class TestComputeCalibrationLinear:
    def test_basic_linear(self):
        cal = compute_calibration(
            100, 500, 900, 500,    # P1, P2 pixel
            100, 100,              # P3 pixel
            0.0, 100.0, 10.0, 0.0  # data values
        )
        assert cal is not None
        assert cal["applied"] is True
        assert cal["x_log_base"] is None
        assert cal["y_log_base"] is None
        # x_scale = (100-0) / (900-100) = 0.125 data/px
        assert cal["x_scale"] == pytest.approx(0.125)
        assert cal["x_offset"] == pytest.approx(-12.5)
        # y_scale across 400 px from 0 to 10 = -0.025 (pixel goes down)
        assert cal["y_scale"] == pytest.approx(-0.025)

    def test_p1_eq_p2_x_returns_none(self):
        cal = compute_calibration(
            100, 500, 100, 500, 100, 100,
            0.0, 100.0, 10.0, 0.0,
        )
        assert cal is None

    def test_p3_on_baseline_returns_none(self):
        cal = compute_calibration(
            100, 500, 900, 500,
            500, 500,   # P3 same y as baseline
            0.0, 100.0, 10.0, 0.0,
        )
        assert cal is None

    def test_p1p2_y_disagreement_recorded(self):
        cal = compute_calibration(
            100, 500, 900, 510,    # P2.y offset by 10
            100, 100,
            0.0, 100.0, 10.0, 0.0,
        )
        assert cal is not None
        assert cal["p1p2_y_disagreement_px"] == pytest.approx(10.0)

    def test_p1p2_y_disagreement_zero_when_aligned(self):
        cal = compute_calibration(
            100, 500, 900, 500,
            100, 100,
            0.0, 100.0, 10.0, 0.0,
        )
        assert cal["p1p2_y_disagreement_px"] == pytest.approx(0.0)


class TestComputeCalibrationLog:
    def test_log10_x_axis(self):
        cal = compute_calibration(
            100, 500, 900, 500,
            100, 100,
            1.0, 1000.0, 100.0, 1.0,
            x_log_base=10.0,
        )
        assert cal is not None
        assert cal["x_log_base"] == 10.0
        # In log10 space: x_scale = (3 - 0)/(900-100) = 0.00375
        assert cal["x_scale"] == pytest.approx(3.0 / 800)

    def test_log10_y_axis(self):
        cal = compute_calibration(
            100, 500, 900, 500,
            100, 100,
            0.0, 100.0, 100.0, 1.0,
            y_log_base=10.0,
        )
        assert cal is not None
        assert cal["y_log_base"] == 10.0

    def test_log_axis_rejects_zero_value(self):
        cal = compute_calibration(
            100, 500, 900, 500,
            100, 100,
            0.0, 100.0, 10.0, 0.0,
            x_log_base=10.0,
        )
        assert cal is None

    def test_log_axis_rejects_negative_value(self):
        cal = compute_calibration(
            100, 500, 900, 500,
            100, 100,
            1.0, 100.0, 10.0, -5.0,
            y_log_base=10.0,
        )
        assert cal is None


class TestPxToDataRoundTrip:
    def test_linear_round_trip(self):
        cal = compute_calibration(
            100, 500, 900, 500, 100, 100,
            0.0, 100.0, 10.0, 0.0,
        )
        for px, py in [(100, 500), (500, 300), (900, 100)]:
            dx, dy = px_to_data(px, py, cal)
            bx, by = data_to_px(dx, dy, cal)
            assert bx == pytest.approx(px, abs=1e-6)
            assert by == pytest.approx(py, abs=1e-6)

    def test_log10_round_trip(self):
        cal = compute_calibration(
            100, 500, 900, 500, 100, 100,
            1.0, 1000.0, 100.0, 1.0,
            x_log_base=10.0, y_log_base=10.0,
        )
        for px, py in [(100, 500), (500, 300), (900, 100)]:
            dx, dy = px_to_data(px, py, cal)
            assert dx > 0 and dy > 0, (dx, dy)
            bx, by = data_to_px(dx, dy, cal)
            assert bx == pytest.approx(px, abs=1e-6)
            assert by == pytest.approx(py, abs=1e-6)

    def test_data_to_px_nonpositive_data_under_log_returns_nan(self):
        cal = compute_calibration(
            100, 500, 900, 500, 100, 100,
            1.0, 1000.0, 100.0, 1.0,
            x_log_base=10.0,
        )
        bx, by = data_to_px(0.0, 50.0, cal)
        assert math.isnan(bx)
        assert math.isnan(by)


def test_tolerance_constant():
    # Locks in the magic value used by the warning logic.
    assert P1P2_Y_TOLERANCE_PX == 3.0
