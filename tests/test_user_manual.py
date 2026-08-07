"""User Manual section gating across OCR / JSON-only modes."""
from shiny import ui

from shiny_app.user_manual import user_manual_tab


def _html(**kwargs) -> str:
    # NavPanel objects render only inside a navset container.
    return str(ui.page_navbar(user_manual_tab(**kwargs), id="nav"))


def test_full_mode_with_ocr_has_all_sections():
    html = _html(ocr=True, json_only=False)
    for title in (
        "Getting started",
        "Calibration — overview",
        "Manual Values panel",
        "Automatic calibration",
        "X/Y label bands",
        "Plot type",
        "Series colors",
        "Overlay editing",
        "Keyboard shortcuts",
    ):
        assert title in html, title


def test_full_mode_without_ocr_hides_ocr_sections():
    html = _html(ocr=False, json_only=False)
    assert "Manual Values panel" in html
    assert "Series colors" in html
    for title in (
        "Automatic calibration",
        "X/Y label bands",
        "Detection settings",
        "Calibration points panel",
    ):
        assert title not in html, title


def test_json_only_mode_hides_calibration_sections():
    html = _html(ocr=True, json_only=True)
    # Calibrate-tab documentation is gone (OCR sections included). Assert on
    # the accordion panel values — titles can be mentioned in body text.
    for value in (
        "calibration_overview",
        "manual_calibration",
        "manual_values",
        "auto_calibration",
        "detect_frame",
        "plot_type",
        "series_colors",
    ):
        assert f'data-value="{value}"' not in html, value
    # Overlay-workflow sections stay.
    for value in (
        "getting_started",
        "overlay_editing",
        "export",
        "keyboard_shortcuts",
        "troubleshooting",
    ):
        assert f'data-value="{value}"' in html, value


def test_json_only_getting_started_describes_json_flow():
    html = _html(json_only=True)
    assert "Import JSON" in html
    full = _html(json_only=False)
    assert "Data CSV" in full or "data CSV" in full
