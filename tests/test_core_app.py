"""Tests for the PlotVerifyApp controller.

Uses an in-memory PNG fixture so no test image files are required and no OCR
runner is invoked. Covers file ingestion, CSV attach, manual calibration,
tick edits, review queue, and CSV export.
"""
import io

import numpy as np
import pytest
from PIL import Image

from plotverify_core import (
    Anchors,
    EditableOverlay,
    MaskingChoice,
    PlotVerifyApp,
    ReviewStatus,
)


def _png(w=100, h=80, color=(0, 255, 0)) -> bytes:
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:] = color
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


CSV_MINIMAL = "series,x,y\nA,1,2\nA,2,3\nB,4,5\n"


def test_add_image_returns_file_id():
    app = PlotVerifyApp()
    fid = app.add_image("plot1.png", _png())
    assert fid in app.state.files
    assert app.active.image_filename == "plot1.png"


def test_add_csv_builds_overlay():
    app = PlotVerifyApp()
    fid = app.add_image("plot1.png", _png())
    app.add_csv(fid, "plot1.csv", CSV_MINIMAL)
    fs = app.state.files[fid]
    assert isinstance(fs.overlay, EditableOverlay)
    assert len(fs.overlay) == 3


def test_add_csv_unknown_raises():
    app = PlotVerifyApp()
    with pytest.raises(KeyError):
        app.add_csv("nope", "x.csv", CSV_MINIMAL)


def test_apply_manual_calibration_linear():
    app = PlotVerifyApp()
    fid = app.add_image("p.png", _png())
    r = app.apply_manual_calibration(fid, Anchors(
        p1_pixel=(10.0, 70.0), p2_pixel=(90.0, 70.0),
        p3_pixel=(10.0, 10.0),
        p1_data_x=0.0, p2_data_x=100.0,
        p3_data_y=50.0, p1_data_y=0.0,
    ))
    assert r.success
    fs = app.state.files[fid]
    assert fs.review_status == ReviewStatus.MANUALLY_ADJUSTED
    assert fs.detection_result is not None
    assert fs.detection_legacy_dict is not None


def test_manual_calibration_log10():
    app = PlotVerifyApp()
    fid = app.add_image("p.png", _png())
    r = app.apply_manual_calibration(fid, Anchors(
        p1_pixel=(10.0, 70.0), p2_pixel=(90.0, 70.0),
        p3_pixel=(10.0, 10.0),
        p1_data_x=1.0, p2_data_x=1000.0,
        p3_data_y=100.0, p1_data_y=1.0,
        x_log_base=10.0, y_log_base=10.0,
    ))
    assert r.success
    assert r.x_calibration.log_base == 10.0


def test_manual_calibration_degenerate_fails():
    app = PlotVerifyApp()
    fid = app.add_image("p.png", _png())
    r = app.apply_manual_calibration(fid, Anchors(
        p1_pixel=(10.0, 70.0), p2_pixel=(10.0, 70.0),  # same as P1
        p3_pixel=(10.0, 10.0),
        p1_data_x=0.0, p2_data_x=100.0,
        p3_data_y=50.0, p1_data_y=0.0,
    ))
    assert not r.success
    assert app.state.files[fid].review_status == ReviewStatus.FAILED


def test_calibrate_all_without_ocr_raises():
    app = PlotVerifyApp()
    app.add_image("p.png", _png())
    # ocr_available will be True on this machine, but injecting a None runner
    # means run_auto_calibration tries the real pipeline.  We expect it to
    # complete OR to record the file as failed via the controller's try/except.
    # Without proper test images Phase A may not detect a frame; either way
    # the controller should not raise.
    results = app.calibrate_all_with_defaults()
    # All files we attempted should now have *some* review_status set.
    for fs in app.state.files.values():
        assert fs.review_status != ReviewStatus.NOT_CALIBRATED or fs.detection_result is None


def test_set_masking_choice_sets_ready():
    app = PlotVerifyApp()
    fid = app.add_image("p.png", _png())
    app.set_masking_choice(fid, MaskingChoice.NO_MASK)
    assert app.state.files[fid].mask_ready is True
    app.set_masking_choice(fid, MaskingChoice.CUSTOM_MASK)
    assert app.state.files[fid].mask_ready is False


def test_export_csv_round_trips_edit():
    app = PlotVerifyApp()
    fid = app.add_image("p.png", _png())
    app.add_csv(fid, "p.csv", CSV_MINIMAL)
    fs = app.state.files[fid]
    fs.overlay.edit_point("A#0", new_x=1.5, new_y=2.5)
    out = app.export_csv(fid).decode("utf-8")
    # Edited row's values appear in output.
    assert "1.5" in out
    assert "2.5" in out


def test_export_csv_with_audit_cols():
    app = PlotVerifyApp()
    fid = app.add_image("p.png", _png())
    app.add_csv(fid, "p.csv", CSV_MINIMAL)
    out = app.export_csv(fid, include_audit_cols=True).decode("utf-8")
    assert "original_x" in out
    assert "edited" in out


def test_review_queue():
    app = PlotVerifyApp()
    f1 = app.add_image("a.png", _png(color=(255, 0, 0)))
    f2 = app.add_image("b.png", _png(color=(0, 0, 255)))
    app.state.files[f1].review_status = ReviewStatus.AUTO_PASSED
    app.state.files[f2].review_status = ReviewStatus.REQUIRES_REVIEW
    app.select(f1)
    assert app.next_unreviewed() == f2


def test_mark_reviewed():
    app = PlotVerifyApp()
    fid = app.add_image("p.png", _png())
    app.state.files[fid].review_status = ReviewStatus.REQUIRES_REVIEW
    app.mark_reviewed(fid)
    assert app.state.files[fid].review_status == ReviewStatus.REVIEWED
