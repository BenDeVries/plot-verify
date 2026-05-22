"""Tests for plotverify_core.session."""
import pytest

from plotverify_core import (
    Anchors,
    AppState,
    MaskingChoice,
    PerFileState,
    ReviewStatus,
)


def _state(file_id, status=ReviewStatus.NOT_CALIBRATED):
    return PerFileState(
        file_id=file_id,
        image_filename=f"{file_id}.png",
        image_bytes=b"\x89PNG",
        review_status=status,
    )


def test_app_state_add_select():
    app = AppState()
    app.add_file(_state("a"))
    assert app.active_file_id == "a"
    app.add_file(_state("b"))
    assert app.active_file_id == "a"  # selection sticks
    app.select("b")
    assert app.active.file_id == "b"


def test_app_state_select_unknown_raises():
    app = AppState()
    with pytest.raises(KeyError):
        app.select("nope")


def test_app_state_remove_active_falls_back():
    app = AppState()
    app.add_file(_state("a"))
    app.add_file(_state("b"))
    app.select("b")
    app.remove_file("b")
    assert app.active_file_id == "a"


def test_app_state_next_unreviewed_wraps():
    app = AppState()
    app.add_file(_state("a", ReviewStatus.AUTO_PASSED))
    app.add_file(_state("b", ReviewStatus.REQUIRES_REVIEW))
    app.add_file(_state("c", ReviewStatus.AUTO_PASSED))
    app.select("a")
    assert app.next_unreviewed() == "b"


def test_app_state_next_unreviewed_none_left():
    app = AppState()
    app.add_file(_state("a", ReviewStatus.AUTO_PASSED))
    app.add_file(_state("b", ReviewStatus.REVIEWED))
    assert app.next_unreviewed() is None


def test_per_file_state_defaults():
    s = _state("x")
    assert s.is_calibrated() is False
    assert s.masking_choice == MaskingChoice.NO_MASK
    assert s.review_status == ReviewStatus.NOT_CALIBRATED
    assert s.csv_has_series_color is False
    assert s.series_delta_e == {}
    assert s.series_color_overrides == {}


def test_has_intentional_color_csv_supplied():
    """When the CSV had a series_color column, every series is intentional."""
    s = _state("x")
    s.csv_has_series_color = True
    assert s.has_intentional_color("A") is True
    assert s.has_intentional_color("anything") is True


def test_has_intentional_color_only_with_override():
    """Without a CSV color column, only series with an override are intentional."""
    s = _state("x")
    assert s.has_intentional_color("A") is False
    s.series_color_overrides["A"] = "#abcdef"
    assert s.has_intentional_color("A") is True
    assert s.has_intentional_color("B") is False


def test_anchors_default_log_base_none():
    a = Anchors()
    assert a.x_log_base is None
    assert a.y_log_base is None
