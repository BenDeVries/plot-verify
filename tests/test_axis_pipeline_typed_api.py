"""Coverage for the typed axis_pipeline surface that doesn't need EasyOCR.

Tests parse_numeric_tick string parsing, manual_calibration edge cases
(degenerate P1=P2, log10 rejects zero), and to_legacy_dict shape. These
complement test_typed_tick_edits (which uses manual_calibration only as a
fixture) and test_real_image_regression (which exercises the full pipeline).
"""
import numpy as np
import pytest

from axis_pipeline import (
    AxisCalibration,
    AxisFrame,
    CalibrationConfig,
    CalibrationResult,
    manual_calibration,
    ocr_available,
    parse_numeric_tick,
)


def test_ocr_available_is_bool():
    assert isinstance(ocr_available(), bool)


class TestParseNumericTick:
    def test_plain_integer(self):
        v, _, _, _ = parse_numeric_tick("42")
        assert v == 42.0

    def test_negative(self):
        v, _, _, _ = parse_numeric_tick("-3.5")
        assert v == -3.5

    def test_unicode_minus(self):
        v, _, _, _ = parse_numeric_tick("−7")  # Unicode minus
        assert v == -7.0

    def test_superscript_log10(self):
        v, _, _, _ = parse_numeric_tick("10⁻³")
        assert v == pytest.approx(1e-3)

    def test_caret_log10(self):
        v, _, _, _ = parse_numeric_tick("10^3")
        assert v == pytest.approx(1000.0)

    def test_e_notation(self):
        v, _, _, _ = parse_numeric_tick("1.5e-2")
        assert v == pytest.approx(0.015)

    def test_non_numeric_returns_none(self):
        v, _, _, _ = parse_numeric_tick("Day")
        assert v is None


class TestManualCalibrationLinear:
    def test_basic(self):
        r = manual_calibration(
            p1_pixel=(100.0, 500.0),
            p2_pixel=(900.0, 500.0),
            p3_pixel=(100.0, 100.0),
            p1_data_x=0.0, p2_data_x=100.0,
            p3_data_y=10.0, p1_data_y=0.0,
        )
        assert r.success
        assert r.mode == "manual"
        assert r.x_calibration is not None
        assert r.x_calibration.log_base is None
        assert r.x_calibration.scale == pytest.approx(0.125)
        assert r.p3_data_x == pytest.approx(0.0)

    def test_degenerate(self):
        r = manual_calibration(
            p1_pixel=(100.0, 500.0),
            p2_pixel=(100.0, 500.0),  # equals P1
            p3_pixel=(100.0, 100.0),
            p1_data_x=0.0, p2_data_x=100.0, p3_data_y=10.0,
        )
        assert not r.success
        assert "differ in pixel X" in r.warnings[0]


class TestManualCalibrationLog:
    def test_log10_x(self):
        r = manual_calibration(
            p1_pixel=(100.0, 500.0),
            p2_pixel=(900.0, 500.0),
            p3_pixel=(100.0, 100.0),
            p1_data_x=1.0, p2_data_x=1000.0,
            p3_data_y=100.0, p1_data_y=1.0,
            x_log_base=10.0,
        )
        assert r.success
        assert r.x_calibration.log_base == 10.0
        # px=500 → data = 10^((500-100)/800 * 3 + 0) = 10^1.5 ≈ 31.62
        assert r.x_calibration.pixel_to_data(500.0) == pytest.approx(10 ** 1.5)

    def test_log10_x_rejects_zero(self):
        r = manual_calibration(
            p1_pixel=(100.0, 500.0),
            p2_pixel=(900.0, 500.0),
            p3_pixel=(100.0, 100.0),
            p1_data_x=0.0, p2_data_x=100.0,
            p3_data_y=10.0, p1_data_y=1.0,
            x_log_base=10.0,
        )
        assert not r.success


class TestCalibrationResultLegacy:
    def test_to_legacy_dict_shape(self):
        r = manual_calibration(
            p1_pixel=(100.0, 500.0),
            p2_pixel=(900.0, 500.0),
            p3_pixel=(100.0, 100.0),
            p1_data_x=0.0, p2_data_x=100.0,
            p3_data_y=10.0, p1_data_y=0.0,
            bbox=AxisFrame(100, 100, 900, 500),
        )
        d = r.to_legacy_dict()
        # Required keys for the Streamlit app's consumers.
        for key in ("success", "confidence", "mode", "bbox",
                    "p1", "p2", "p3",
                    "p1_data_x", "p2_data_x", "p3_data_x",
                    "p1_data_y", "p3_data_y",
                    "x_ticks", "y_ticks",
                    "x_tick_table", "y_tick_table",
                    "x_calibration", "y_calibration"):
            assert key in d, f"missing key: {key}"
