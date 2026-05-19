"""Refactor B: tick-table edits round-trip through typed CalibrationResult."""
import pandas as pd
import pytest

from axis_pipeline import (
    AxisFrame,
    PairedTick,
    manual_calibration,
)
from axis_pipeline.legacy import (
    rebuild_result_from_detection,
    update_detection_from_tick_tables,
    update_result_from_tick_edits,
)


def _seed_result():
    """Build a CalibrationResult with paired-tick tables we can edit."""
    r = manual_calibration(
        p1_pixel=(100.0, 500.0),
        p2_pixel=(900.0, 500.0),
        p3_pixel=(100.0, 100.0),
        p1_data_x=0.0, p2_data_x=100.0,
        p3_data_y=10.0, p1_data_y=0.0,
        bbox=AxisFrame(100, 100, 900, 500),
    )
    # manual_calibration produces no paired-tick rows; inject two for testing.
    r.x_paired_ticks = [
        PairedTick(
            pixel_position=200.0, fixed_axis_pixel=500.0,
            data_value=10.0, pair_distance_px=0.0, grid_index=0,
            label_bbox=(190, 510, 210, 530),
            raw_text="10", cleaned_text="10",
            ocr_confidence=0.9, parse_status="parsed", flag="",
            include=True, status="paired_to_tick_mark",
        ),
        PairedTick(
            pixel_position=800.0, fixed_axis_pixel=500.0,
            data_value=90.0, pair_distance_px=0.0, grid_index=1,
            label_bbox=(790, 510, 810, 530),
            raw_text="90", cleaned_text="90",
            ocr_confidence=0.9, parse_status="parsed", flag="",
            include=True, status="paired_to_tick_mark",
        ),
    ]
    return r


def test_typed_updater_returns_calibration_result():
    r = _seed_result()
    new_r = update_result_from_tick_edits(r, None, None)  # no edits
    # Type is preserved.
    assert hasattr(new_r, "x_paired_ticks")
    assert hasattr(new_r, "to_legacy_dict")


def test_typed_updater_applies_value_edit():
    r = _seed_result()
    # Edit x-tick rows: change first row's value from 10.0 to 12.0.
    x_edits = pd.DataFrame([
        {"include": True, "value": 12.0, "pixel_position": 200.0},
        {"include": True, "value": 90.0, "pixel_position": 800.0},
    ])
    new_r = update_result_from_tick_edits(r, x_edits, None)
    values = [t.data_value for t in new_r.x_paired_ticks]
    assert 12.0 in values


def test_typed_updater_unchecking_drops_from_calibration():
    r = _seed_result()
    # Uncheck the second row.
    x_edits = pd.DataFrame([
        {"include": True, "value": 10.0, "pixel_position": 200.0},
        {"include": False, "value": 90.0, "pixel_position": 800.0},
    ])
    new_r = update_result_from_tick_edits(r, x_edits, None)
    # Only one included row should remain.
    inc = [t for t in new_r.x_paired_ticks if t.include]
    assert len(inc) == 1
    assert inc[0].data_value == 10.0


def test_legacy_dict_updater_still_works():
    """The pre-existing dict-shaped path remains functional for external scripts."""
    r = _seed_result()
    d = r.to_legacy_dict()
    new_d = update_detection_from_tick_tables(d, None, None)
    # Roundtrip via rebuild_result_from_detection still yields a valid result.
    rebuilt = rebuild_result_from_detection(new_d)
    assert hasattr(rebuilt, "x_paired_ticks")
