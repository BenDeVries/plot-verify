"""Tests for plotverify_core.serialization (save_session / load_session).

Covers:
  - Round-trip an AppState with one or more files; manual calibration result
    survives if the pipeline version matches.
  - EditableOverlay edits + audit columns round-trip through overlays/.
  - Pipeline-version mismatch discards the saved CalibrationResult and re-runs
    manual_calibration (because manual_anchors are set).
  - Schema-version mismatch is rejected with a clear error.
  - Manifest references to missing zip entries are rejected.
  - is_dirty / mark_clean state on the controller.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from plotverify_core import (
    Anchors,
    MaskingChoice,
    PlotVerifyApp,
    ReviewStatus,
    save_session,
    load_session,
)
from plotverify_core import serialization as _ser


CSV_SMALL = "series,x,y,y_err_lower,y_err_upper\nA,1,2,1.8,2.2\nA,2,4,3.5,4.5\nB,3,9,8,10\n"


def _png(w=120, h=90, color=(0, 200, 0)) -> bytes:
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:] = color
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def _seed_app(tmp_path=None) -> PlotVerifyApp:
    """Build a representative app with one image, one CSV, and a manual cal."""
    app = PlotVerifyApp()
    fid = app.add_image("alpha.png", _png(color=(255, 0, 0)))
    app.add_csv(fid, "alpha.csv", CSV_SMALL)
    app.set_masking_choice(fid, MaskingChoice.NO_MASK)
    app.apply_manual_calibration(fid, Anchors(
        p1_pixel=(12.0, 80.0), p2_pixel=(108.0, 80.0),
        p3_pixel=(12.0, 10.0),
        p1_data_x=0.0, p2_data_x=10.0,
        p1_data_y=0.0, p3_data_y=100.0,
    ))
    return app


# ----------------------------------------------------------------------
# Basic round-trip
# ----------------------------------------------------------------------

def test_save_session_writes_expected_zip_layout(tmp_path: Path):
    app = _seed_app()
    path = tmp_path / "session.pvsession"
    app.save_session(path)
    assert path.exists() and path.stat().st_size > 0
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        # Exactly one image and one csv side-car for the seeded file.
        assert any(n.startswith("images/") and n.endswith(".png") for n in names)
        assert any(n.startswith("csvs/") and n.endswith(".csv") for n in names)
        # No overlays side-car because no edits were made yet.
        assert not any(n.startswith("overlays/") for n in names)


def test_round_trip_preserves_manual_calibration(tmp_path: Path):
    app = _seed_app()
    fid = next(iter(app.state.files))
    original = app.state.files[fid].detection_result
    assert original is not None and original.success

    path = tmp_path / "s.pvsession"
    app.save_session(path)

    app2 = PlotVerifyApp()
    app2.load_session(path)
    fs2 = app2.state.files[fid]
    assert fs2.detection_result is not None
    assert fs2.detection_result.success
    # Anchors and bbox should match within the legacy-dict round-trip tolerance.
    assert fs2.detection_result.p1_pixel == original.p1_pixel
    assert fs2.detection_result.p3_pixel == original.p3_pixel
    assert fs2.detection_result.p1_data_x == original.p1_data_x
    assert fs2.detection_result.p3_data_y == original.p3_data_y


def test_round_trip_active_file_and_order(tmp_path: Path):
    app = PlotVerifyApp()
    f1 = app.add_image("a.png", _png(color=(255, 0, 0)))
    f2 = app.add_image("b.png", _png(color=(0, 255, 0)))
    app.select(f2)

    path = tmp_path / "s.pvsession"
    app.save_session(path)

    app2 = PlotVerifyApp()
    app2.load_session(path)
    assert app2.state.active_file_id == f2
    assert app2.state.file_order == [f1, f2]


# ----------------------------------------------------------------------
# EditableOverlay edits
# ----------------------------------------------------------------------

def test_overlay_edits_round_trip(tmp_path: Path):
    app = _seed_app()
    fid = next(iter(app.state.files))
    fs = app.state.files[fid]
    pid = fs.overlay.points()[0].point_id
    fs.overlay.edit_point(pid, new_x=99.0, new_y=-7.0)
    fs.overlay.edit_err_upper(pid, value=5.5)
    assert fs.overlay.has_edits()

    path = tmp_path / "edits.pvsession"
    app.save_session(path)

    # An overlays side-car should now be present.
    with zipfile.ZipFile(path) as zf:
        assert any(n.startswith("overlays/") for n in zf.namelist())

    app2 = PlotVerifyApp()
    app2.load_session(path)
    fs2 = app2.state.files[fid]
    p = fs2.overlay.get(pid)
    assert p.x == pytest.approx(99.0)
    assert p.y == pytest.approx(-7.0)
    assert p.y_err_upper == pytest.approx(5.5)
    assert p.edited is True
    # Originals are also preserved.
    assert p.original_x == pytest.approx(1.0)
    assert p.original_y == pytest.approx(2.0)


# ----------------------------------------------------------------------
# Version handling
# ----------------------------------------------------------------------

def test_pipeline_version_mismatch_re_runs_manual_calibration(tmp_path: Path,
                                                              monkeypatch):
    app = _seed_app()
    path = tmp_path / "v.pvsession"
    app.save_session(path)

    # Bump the runtime PIPELINE_VERSION the manifest will be compared against.
    monkeypatch.setattr(_ser, "PIPELINE_VERSION", "999.999")

    call_count = {"n": 0}
    real_manual = _ser.manual_calibration

    def counting_manual(*args, **kwargs):
        call_count["n"] += 1
        return real_manual(*args, **kwargs)

    monkeypatch.setattr(_ser, "manual_calibration", counting_manual)

    app2 = PlotVerifyApp()
    app2.load_session(path)
    assert call_count["n"] >= 1, (
        "manual_calibration should be re-invoked when pipeline_version differs"
    )
    fid = next(iter(app2.state.files))
    assert app2.state.files[fid].detection_result is not None


def test_schema_version_mismatch_rejected(tmp_path: Path):
    app = _seed_app()
    path = tmp_path / "s.pvsession"
    app.save_session(path)

    # Rewrite the manifest with an unknown schema_version.
    bad = tmp_path / "bad.pvsession"
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(bad, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.namelist():
            data = zin.read(item)
            if item == "manifest.json":
                manifest = json.loads(data)
                manifest["schema_version"] = "9.9"
                data = json.dumps(manifest).encode("utf-8")
            zout.writestr(item, data)

    with pytest.raises(ValueError, match="schema_version"):
        load_session(bad)


def test_missing_image_in_zip_rejected(tmp_path: Path):
    app = _seed_app()
    path = tmp_path / "s.pvsession"
    app.save_session(path)

    # Strip the image side-car but keep manifest referencing it.
    bad = tmp_path / "broken.pvsession"
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(bad, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.namelist():
            if item.startswith("images/"):
                continue
            zout.writestr(item, zin.read(item))

    with pytest.raises(ValueError, match="not in the archive"):
        load_session(bad)


# ----------------------------------------------------------------------
# Dirty tracking
# ----------------------------------------------------------------------

def test_is_dirty_set_by_mutators_and_cleared_by_save(tmp_path: Path):
    app = PlotVerifyApp()
    assert app.is_dirty is False
    app.add_image("p.png", _png())
    assert app.is_dirty is True
    app.save_session(tmp_path / "s.pvsession")
    assert app.is_dirty is False
    # Subsequent edit re-dirties.
    fid = next(iter(app.state.files))
    app.set_masking_choice(fid, MaskingChoice.DEFAULT_MASK)
    assert app.is_dirty is True


def test_autosave_if_dirty_no_op_when_clean(tmp_path: Path):
    app = PlotVerifyApp()
    app.set_autosave_path(tmp_path / "auto.pvsession")
    # No mutations yet — nothing to save.
    saved = app.autosave_if_dirty()
    assert saved is False
    assert not (tmp_path / "auto.pvsession").exists()


def test_autosave_if_dirty_writes_when_dirty(tmp_path: Path):
    app = PlotVerifyApp()
    app.set_autosave_path(tmp_path / "auto.pvsession")
    app.add_image("p.png", _png())
    assert app.is_dirty
    assert app.autosave_if_dirty() is True
    assert (tmp_path / "auto.pvsession").exists()
    assert app.is_dirty is False


# ----------------------------------------------------------------------
# Misc
# ----------------------------------------------------------------------

def test_load_nonexistent_path_raises():
    with pytest.raises(FileNotFoundError):
        load_session(Path("/tmp/does-not-exist-pvsession-xyzzy.zip"))


def test_round_trip_preserves_series_states(tmp_path: Path):
    app = _seed_app()
    fid = next(iter(app.state.files))
    fs = app.state.files[fid]
    series_names_before = sorted(fs.series_states)
    assert series_names_before  # CSV had multiple series

    path = tmp_path / "ss.pvsession"
    app.save_session(path)

    app2 = PlotVerifyApp()
    app2.load_session(path)
    fs2 = app2.state.files[fid]
    assert sorted(fs2.series_states) == series_names_before
    # SeriesState fields survive.
    name = series_names_before[0]
    assert fs2.series_states[name].color_hex == fs.series_states[name].color_hex


# ----------------------------------------------------------------------
# plot_type / orientation persistence
# ----------------------------------------------------------------------

def test_round_trip_preserves_plot_type_and_orientation(tmp_path: Path):
    app = _seed_app()
    fid = next(iter(app.state.files))
    app.state.files[fid].plot_type = "bar"
    app.state.files[fid].orientation = "horizontal"

    path = tmp_path / "s.pvsession"
    app.save_session(path)

    app2 = PlotVerifyApp()
    app2.load_session(path)
    fs2 = app2.state.files[fid]
    assert fs2.plot_type == "bar"
    assert fs2.orientation == "horizontal"


def test_legacy_manifest_without_plot_type_loads_with_defaults(tmp_path: Path):
    """Manifests written before the plot_type/orientation keys still load."""
    app = _seed_app()
    path = tmp_path / "s.pvsession"
    app.save_session(path)

    # Strip the new keys to simulate an old session file.
    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        payload = {n: zf.read(n) for n in zf.namelist() if n != "manifest.json"}
    for entry in manifest["files"].values():
        entry.pop("plot_type", None)
        entry.pop("orientation", None)
    stripped = tmp_path / "legacy.pvsession"
    with zipfile.ZipFile(stripped, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
        for name, data in payload.items():
            zf.writestr(name, data)

    app2 = PlotVerifyApp()
    app2.load_session(stripped)
    fs2 = next(iter(app2.state.files.values()))
    assert fs2.plot_type == "time_series"
    assert fs2.orientation == "vertical"
