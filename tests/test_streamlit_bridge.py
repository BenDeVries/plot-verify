"""Tests for plotverify_core.streamlit_bridge.

Verifies the flat-dict ↔ AppState bridge used by the Streamlit single-image
app to drive manual save_session / load_session — without any actual
Streamlit dependency. The mapping is exercised by simulating
``st.session_state`` as a plain dict.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from plotverify_core import (
    Anchors,
    PlotVerifyApp,
    load_session,
)
from plotverify_core.image_io import decode_and_maybe_downscale
from plotverify_core.streamlit_bridge import (
    DEFAULT_SESSION_DIR,
    SESSION_DIR_ENV,
    apply_app_to_flat_state,
    build_app_from_flat_state,
    ensure_session_id,
    resolve_session_dir,
    resolve_session_path,
)


CSV_SMALL = (
    "series,x,y,y_err_lower,y_err_upper\n"
    "A,1,2,1.8,2.2\n"
    "A,2,4,3.5,4.5\n"
    "B,3,9,8,10\n"
)


def _png_bytes(w=120, h=90, color=(0, 200, 0)) -> bytes:
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:] = color
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def _seed_flat_state() -> dict:
    """Build a flat dict that mirrors what the Streamlit app would hold
    after the user has uploaded an image+CSV and applied manual calibration."""
    img_bytes = _png_bytes()
    load = decode_and_maybe_downscale(img_bytes, downscale=False)
    df = pd.read_csv(io.StringIO(CSV_SMALL))
    # Give A a known series_color so the typed SeriesState picks it up.
    df["series_color"] = df["series"].map({"A": "#ff8800", "B": "#0088ff"})
    return {
        "image_bytes": img_bytes,
        "image_filename": "alpha.png",
        "image_bgr": load.img_bgr,
        "image_rgb": load.img_rgb,
        "image_hash": "deadbeefdeadbeef",
        "image_downscale_factor": 1.0,
        "csv_filename": "alpha.csv",
        "df": df,
        "csv_hash": "abc",
        "series_states": {
            "A": {"use_delta_e": True, "delta_e": 12,
                  "h_min": 5, "h_max": 25, "s_min": 30, "s_max": 220,
                  "v_min": 40, "v_max": 230, "interpolate": True},
            "B": {"use_delta_e": False, "delta_e": 10,
                  "h_min": 100, "h_max": 130, "s_min": 50, "s_max": 200,
                  "v_min": 60, "v_max": 240, "interpolate": False},
        },
        "vis_A": True,
        "vis_B": False,
        "series_color_overrides": {"A": "#ff8800"},
        # Manual anchors.
        "p1_px_x": 12.0, "p1_px_y": 80.0,
        "p2_px_x": 108.0, "p2_px_y": 80.0,
        "p3_px_x": 12.0, "p3_px_y": 10.0,
        "p1_data_x": 0.0, "p2_data_x": 10.0,
        "p1_data_y": 0.0, "p3_data_y": 100.0,
        "calibration": {"applied": True, "scale_x": 1.0, "offset_x": 0.0},
    }


# ----------------------------------------------------------------------
# Path resolver
# ----------------------------------------------------------------------

def test_resolve_session_dir_defaults_to_home(monkeypatch):
    monkeypatch.delenv(SESSION_DIR_ENV, raising=False)
    assert resolve_session_dir() == DEFAULT_SESSION_DIR


def test_resolve_session_dir_respects_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(SESSION_DIR_ENV, str(tmp_path))
    assert resolve_session_dir() == tmp_path


def test_resolve_session_path_combines_dir_and_id(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(SESSION_DIR_ENV, str(tmp_path))
    p = resolve_session_path("sess42")
    assert p == tmp_path / "sess42.pvsession"


def test_resolve_session_path_rejects_empty():
    with pytest.raises(ValueError):
        resolve_session_path("")


def test_ensure_session_id_is_stable():
    d: dict = {}
    sid1 = ensure_session_id(d)
    sid2 = ensure_session_id(d)
    assert sid1 == sid2
    assert len(sid1) == 12
    # Distinct dicts get distinct ids.
    assert ensure_session_id({}) != sid1


# ----------------------------------------------------------------------
# Build app from flat state
# ----------------------------------------------------------------------

def test_build_app_from_empty_state_returns_empty_app():
    app = build_app_from_flat_state({})
    assert app.state.active is None
    assert app.is_dirty is False


def test_build_app_carries_image_and_csv():
    state = _seed_flat_state()
    app = build_app_from_flat_state(state)
    fs = app.active
    assert fs is not None
    assert fs.image_filename == "alpha.png"
    assert fs.image_bytes == state["image_bytes"]
    assert fs.csv_filename == "alpha.csv"
    assert fs.csv_df is not None
    assert list(fs.csv_df["series"].unique()) == ["A", "B"]


def test_build_app_translates_typed_series_states():
    state = _seed_flat_state()
    app = build_app_from_flat_state(state)
    fs = app.active
    assert set(fs.series_states.keys()) == {"A", "B"}
    a = fs.series_states["A"]
    assert a.delta_e == 12
    assert a.interpolate is True
    assert a.visible is True
    assert a.color_hex == "#ff8800"
    b = fs.series_states["B"]
    assert b.visible is False
    assert b.use_delta_e is False


def test_build_app_picks_up_manual_anchors():
    state = _seed_flat_state()
    app = build_app_from_flat_state(state)
    a = app.active.manual_anchors
    assert a is not None
    assert a.p1_pixel == (12.0, 80.0)
    assert a.p3_pixel == (12.0, 10.0)
    assert a.p2_data_x == 10.0
    assert a.p3_data_y == 100.0


def test_build_app_omits_anchors_when_all_zero():
    state = _seed_flat_state()
    for k in ("p1_px_x", "p1_px_y", "p2_px_x", "p2_px_y",
              "p3_px_x", "p3_px_y", "p1_data_x", "p2_data_x",
              "p1_data_y", "p3_data_y"):
        state[k] = 0.0
    app = build_app_from_flat_state(state)
    assert app.active.manual_anchors is None


# ----------------------------------------------------------------------
# Round trip (build → save → load → apply)
# ----------------------------------------------------------------------

def test_round_trip_via_save_and_load(tmp_path: Path):
    state = _seed_flat_state()

    app = build_app_from_flat_state(state)
    path = tmp_path / "session.pvsession"
    app.save_session(path)
    assert path.exists()

    # Load through PlotVerifyApp, then rehydrate a fresh flat dict.
    app2 = PlotVerifyApp()
    app2.load_session(path)

    out: dict = {}
    file_id = apply_app_to_flat_state(app2, out)
    assert file_id is not None

    assert out["image_filename"] == "alpha.png"
    assert out["image_bytes"] == state["image_bytes"]
    assert out["csv_filename"] == "alpha.csv"
    # Anchors restored from manual_anchors.
    assert out["p1_px_x"] == 12.0 and out["p1_px_y"] == 80.0
    assert out["p3_data_y"] == 100.0
    # Calibration dict round-trips.
    assert out["calibration"]["applied"] is True
    # Series visibility restored.
    assert out["vis_A"] is True
    assert out["vis_B"] is False
    # Series-state numeric settings restored.
    assert out["series_states"]["A"]["delta_e"] == 12
    assert out["series_states"]["A"]["interpolate"] is True
    # Color overrides preserved.
    assert out["series_color_overrides"] == {"A": "#ff8800"}
    # DataFrame survives.
    assert isinstance(out["df"], pd.DataFrame)
    assert list(out["df"]["series"].unique()) == ["A", "B"]


def test_apply_clears_transient_caches(tmp_path: Path):
    state = _seed_flat_state()
    app = build_app_from_flat_state(state)
    path = tmp_path / "s.pvsession"
    app.save_session(path)

    app2 = PlotVerifyApp()
    app2.load_session(path)

    # Caller's flat dict has stale caches that should be cleared.
    out = {
        "cal_masked_img_bgr": np.zeros((4, 4, 3), dtype=np.uint8),
        "cal_masked_img_path": "/tmp/stale.png",
        "frame_preview_cache": {"key": "stale"},
    }
    apply_app_to_flat_state(app2, out)
    assert "cal_masked_img_bgr" not in out
    assert "cal_masked_img_path" not in out
    assert "frame_preview_cache" not in out


def test_apply_to_empty_app_returns_none():
    out: dict = {}
    assert apply_app_to_flat_state(PlotVerifyApp(), out) is None
    assert out == {}


def test_loaded_image_hash_matches_md5_of_bytes(tmp_path: Path):
    """Regression: ``image_hash`` in the rehydrated state must equal the full
    md5 of the original bytes.

    Why this matters: ``_load_image_from_upload`` in app_auto_axis.py uses
    that equality to decide whether the still-mounted file_uploader holds a
    *new* image. If the hash doesn't match, it RESETS the P1/P2/P3 anchor
    flat keys to default seed positions, silently overwriting the values
    just restored from the .pvsession.
    """
    import hashlib as _hl
    state = _seed_flat_state()
    app = build_app_from_flat_state(state)
    path = tmp_path / "s.pvsession"
    app.save_session(path)

    app2 = PlotVerifyApp()
    app2.load_session(path)
    out: dict = {}
    apply_app_to_flat_state(app2, out)

    assert out["image_hash"] == _hl.md5(state["image_bytes"]).hexdigest()
    # Anchors must NOT be the 0.10*w / 0.90*h defaults — they should be the
    # values from manual_anchors.
    assert out["p1_px_x"] == 12.0
    assert out["p3_px_y"] == 10.0
