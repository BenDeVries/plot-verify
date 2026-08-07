"""Tests for shiny_app.runtime_flags.json_only_mode."""
import sys

from shiny_app.runtime_flags import json_only_mode


def test_default_false_on_native(monkeypatch):
    monkeypatch.delenv("PLOTVERIFY_JSON_ONLY", raising=False)
    assert json_only_mode() is False


def test_env_forces_on(monkeypatch):
    for val in ("1", "true", "TRUE", "yes"):
        monkeypatch.setenv("PLOTVERIFY_JSON_ONLY", val)
        assert json_only_mode() is True


def test_env_forces_off_even_under_emscripten(monkeypatch):
    monkeypatch.setattr(sys, "platform", "emscripten")
    for val in ("0", "false", "no"):
        monkeypatch.setenv("PLOTVERIFY_JSON_ONLY", val)
        assert json_only_mode() is False


def test_emscripten_auto_enables(monkeypatch):
    monkeypatch.delenv("PLOTVERIFY_JSON_ONLY", raising=False)
    monkeypatch.setattr(sys, "platform", "emscripten")
    assert json_only_mode() is True


def test_unrecognized_env_falls_through_to_platform(monkeypatch):
    monkeypatch.setenv("PLOTVERIFY_JSON_ONLY", "maybe")
    assert json_only_mode() is False


def test_user_manual_flag_matches_ocr_available():
    from axis_pipeline import ocr_available
    import shiny_app.user_manual as um

    assert um._OCR_AVAILABLE == ocr_available()
