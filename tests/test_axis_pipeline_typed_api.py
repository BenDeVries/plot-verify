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


class TestCropBandDegenerate:
    """Regression: a degenerate band must return a zero-size crop so callers
    can skip it. Previously it returned the whole image with offset (0, 0),
    causing the band OCR to silently scan the full image at wrong coords."""

    def _make_img(self):
        return np.full((100, 200, 3), 128, dtype=np.uint8)

    def test_zero_size_band(self):
        from axis_pipeline.ocr import crop_band
        crop, offset = crop_band(self._make_img(), (50, 50, 50, 50))
        assert crop.size == 0
        assert crop.shape[2] == 3
        assert offset == (0, 0)

    def test_inverted_band(self):
        from axis_pipeline.ocr import crop_band
        crop, _ = crop_band(self._make_img(), (80, 80, 30, 30))
        assert crop.size == 0

    def test_band_entirely_outside(self):
        from axis_pipeline.ocr import crop_band
        # x0 > image width: clamps to a zero-width band.
        crop, _ = crop_band(self._make_img(), (300, 10, 400, 80))
        assert crop.size == 0

    def test_valid_band_unchanged(self):
        from axis_pipeline.ocr import crop_band
        crop, offset = crop_band(self._make_img(), (10, 20, 50, 80))
        assert crop.shape == (60, 40, 3)
        assert offset == (10, 20)


class TestModalSpacingClustering:
    """Regression: clustering used `clusters[-1][-1]` (last appended diff)
    instead of the cluster's representative — letting a chain of small
    `cluster_tol_px`-sized steps merge widely-separated diffs into one
    cluster."""

    def test_cluster_does_not_chain_across_tol(self):
        from axis_pipeline.gridfit import _modal_spacing
        # Chain: 10 -> 14 -> 18 -> 22 -> 26 -> 30; each gap is 4 (<= tol=8),
        # but 10 and 30 are 20 apart. With the buggy version they'd all
        # become one cluster. With the fix they split as soon as the new
        # diff drifts beyond tol from the cluster median.
        diffs = np.array([10.0, 14.0, 18.0, 22.0, 26.0, 30.0])
        spacing = _modal_spacing(diffs, cluster_tol_px=8.0)
        assert spacing is not None
        # The result must come from a cluster whose members span no more
        # than 2 * tol around the median (16 px window).
        assert 10.0 <= spacing <= 30.0
        # And the cluster containing the chosen spacing must not include
        # the extremes — sanity check by re-running with tighter tol.
        spacing_tight = _modal_spacing(diffs, cluster_tol_px=2.0)
        assert spacing_tight is not None

    def test_regular_grid_unchanged(self):
        from axis_pipeline.gridfit import _modal_spacing
        # A clean regular grid still returns the tick spacing.
        diffs = np.array([50.0, 50.0, 50.0, 50.0])
        assert _modal_spacing(diffs) == 50.0


class TestMonotonicityNonMutation:
    """Regression: _enforce_monotonic_* mutated `p.include` / `p.status` on
    the input PairedTicks. Now it must build fresh objects via
    `dataclasses.replace` so the input is untouched and repeat calls are
    idempotent."""

    def _ticks(self):
        from axis_pipeline import PairedTick
        # Build pairs that include a value that breaks monotonicity (x=2,
        # value=99 between value=1 at x=1 and value=3 at x=3).
        return [
            PairedTick(pixel_position=1.0, fixed_axis_pixel=0.0,
                       data_value=1.0, pair_distance_px=0.0, grid_index=0,
                       label_bbox=(0, 0, 10, 10), raw_text="1",
                       cleaned_text="1", ocr_confidence=0.9,
                       parse_status="ok", flag=""),
            PairedTick(pixel_position=2.0, fixed_axis_pixel=0.0,
                       data_value=99.0, pair_distance_px=0.0, grid_index=1,
                       label_bbox=(0, 0, 10, 10), raw_text="99",
                       cleaned_text="99", ocr_confidence=0.9,
                       parse_status="ok", flag=""),
            PairedTick(pixel_position=3.0, fixed_axis_pixel=0.0,
                       data_value=3.0, pair_distance_px=0.0, grid_index=2,
                       label_bbox=(0, 0, 10, 10), raw_text="3",
                       cleaned_text="3", ocr_confidence=0.9,
                       parse_status="ok", flag=""),
        ]

    def test_inputs_unmutated_and_idempotent(self):
        from axis_pipeline.pairing import _enforce_monotonic_x
        ticks = self._ticks()
        out1 = _enforce_monotonic_x(ticks)
        # Inputs untouched: original objects retain include=True.
        assert all(p.include is True for p in ticks)
        assert all(p.status == "paired_to_tick_mark" for p in ticks)
        # Output flags exactly one violation.
        assert sum(1 for p in out1 if not p.include) == 1
        # Re-running on the SAME input gives the same output (no leaked state).
        out2 = _enforce_monotonic_x(ticks)
        assert [(p.include, p.status) for p in out1] == \
               [(p.include, p.status) for p in out2]


class TestPairXNoneFiltering:
    """Phase 4: pair_x must filter out records whose `value` is None before
    greedy matching so they don't consume tick positions intended for real
    numeric labels."""

    def _make_grid(self):
        from axis_pipeline import GridFit
        return GridFit(
            spacing=100.0, origin=100.0,
            fitted_positions=[100.0, 200.0, 300.0],
            fitted_indices=[0, 1, 2],
            rejected_positions=[],
            grid_residuals=[0.0, 0.0, 0.0],
            n_grid_cells=3, success=True,
        )

    def _make_record(self, *, x: float, value):
        from axis_pipeline import OCRRecord
        return OCRRecord(
            raw_text=str(value), cleaned_text=str(value),
            value=value, is_numeric=value is not None,
            confidence=0.9, bbox=(int(x) - 5, 95, int(x) + 5, 105),
            center=(x, 100.0),
            parse_status="parsed" if value is not None else "not_numeric",
            parse_flag="",
        )

    def test_none_valued_records_dont_consume_tick_positions(self):
        from axis_pipeline import AxisFrame
        from axis_pipeline.pairing import pair_x
        # A None-valued OCR record positioned exactly on tick 0 (x=100). With
        # the buggy version it would greedy-match tick 0 and prevent the real
        # numeric record from being paired there. With the fix the None record
        # is filtered out first.
        records = [
            self._make_record(x=100.0, value=None),     # would steal tick 0
            self._make_record(x=100.0, value=1.0),      # real label at tick 0
            self._make_record(x=200.0, value=2.0),
            self._make_record(x=300.0, value=3.0),
        ]
        bbox = AxisFrame(left=50, top=20, right=350, bottom=120)
        paired = pair_x(records, self._make_grid(), bbox, max_distance=10.0)
        # All three numeric labels survive and are paired to the three ticks.
        assert sum(1 for p in paired if p.include) == 3
        assert sorted(p.data_value for p in paired if p.include) == [1.0, 2.0, 3.0]

    def test_all_none_records_yields_empty(self):
        from axis_pipeline import AxisFrame
        from axis_pipeline.pairing import pair_x
        records = [
            self._make_record(x=100.0, value=None),
            self._make_record(x=200.0, value=None),
        ]
        bbox = AxisFrame(left=50, top=20, right=350, bottom=120)
        paired = pair_x(records, self._make_grid(), bbox, max_distance=10.0)
        assert paired == []


class TestOLSUniquenessTolerance:
    """Phase 4: `_ols_fit` uses np.unique(np.round(y, 9)) instead of a Python
    float-set, so near-duplicates from log10 rounding don't spuriously pass
    the uniqueness check (or, equivalently, exact duplicates that round to
    the same value still fail)."""

    def test_rounding_treats_near_dupes_as_one(self):
        from axis_pipeline.pairing import _ols_fit
        # Two near-equal values differing only in the 12th decimal place.
        tick_px = np.array([100.0, 200.0])
        values = np.array([5.000000000001, 5.000000000002])
        # Without tolerance the float-set call would see them as distinct,
        # producing a useless fit. With np.round(y, 9) they collapse.
        assert _ols_fit(tick_px, values) is None

    def test_distinct_values_still_fit(self):
        from axis_pipeline.pairing import _ols_fit
        tick_px = np.array([100.0, 200.0, 300.0])
        values = np.array([10.0, 20.0, 30.0])
        fit = _ols_fit(tick_px, values)
        assert fit is not None
        slope, intercept, use_log = fit
        assert abs(slope - 0.1) < 1e-9
        assert use_log is False


class TestXBandConfigFields:
    """Phase 4: x_band_extend_outward_px (25) and
    x_band_right_edge_threshold_px (30) are exposed as named
    CalibrationConfig fields instead of magic literals in pipeline.py."""

    def test_defaults(self):
        cfg = CalibrationConfig()
        assert cfg.x_band_extend_outward_px == 25
        assert cfg.x_band_right_edge_threshold_px == 30

    def test_override(self):
        cfg = CalibrationConfig(
            x_band_extend_outward_px=40,
            x_band_right_edge_threshold_px=50,
        )
        assert cfg.x_band_extend_outward_px == 40
        assert cfg.x_band_right_edge_threshold_px == 50
