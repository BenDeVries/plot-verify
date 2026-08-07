"""Smoke tests for the Shiny app's non-UI surfaces.

These don't exercise the Shiny reactive layer (that would require Playwright
and a running browser). They cover the figure builders and the controller
flow that the Shiny callbacks invoke, so a regression in the data-path
won't ship silently.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from plotverify_core import (
    Anchors,
    PlotVerifyApp,
    build_overlay_traces,
)
from shiny_app.figures import (
    ANCHOR_LABELS,
    all_calibration_shapes,
    anchor_annotations,
    anchor_shapes,
    anchors_from_result,
    annotations_to_anchors,
    build_calibration_edit_figure,
    build_data_overlay_figure,
    cal_dict_from_result,
    default_anchors_for_image,
    enforce_anchor_constraints,
    guide_line_traces,
    shapes_to_anchors,
)


TEST_IMAGE = Path(__file__).resolve().parent.parent / "test_images" / "case3_three_arm_dose_response.png"
TEST_CSV = Path(__file__).resolve().parent.parent / "test_images" / "case3_three_arm_dose_response.csv"


def test_default_anchors_geometry():
    a = default_anchors_for_image(1000, 500)
    assert a.p1_pixel == (100.0, 450.0)
    assert a.p2_pixel == (900.0, 450.0)
    assert a.p3_pixel == (100.0, 50.0)


def test_anchor_shapes_round_trip():
    # anchor_shapes is a legacy helper (3-circle form); ANCHOR_LABELS now only
    # has 2 display labels, so compare against the legacy set directly.
    a = Anchors(p1_pixel=(10, 90), p2_pixel=(90, 90), p3_pixel=(10, 10))
    shapes = anchor_shapes(a)
    assert len(shapes) == 3
    assert {s["name"] for s in shapes} == {"P1", "P2", "P3"}
    a2 = shapes_to_anchors(shapes, a)
    assert a2.p1_pixel == a.p1_pixel
    assert a2.p2_pixel == a.p2_pixel
    assert a2.p3_pixel == a.p3_pixel


def test_calibration_edit_figure_builds():
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    a = default_anchors_for_image(300, 200)
    fig = build_calibration_edit_figure(img, a)
    assert isinstance(fig, go.Figure)
    # Anchors are now annotations (no resize handles); guide lines are
    # scatter traces; layout.shapes is empty.
    assert len(fig.layout.shapes) == 0
    assert len(fig.layout.annotations) == 2
    assert {a.name for a in fig.layout.annotations} == set(ANCHOR_LABELS)
    assert len(fig.data) == 4
    for trace in fig.data:
        assert trace.line.dash == "dash"
        assert trace.mode == "lines"


def test_anchor_annotations_round_trip():
    # Only P1 (top-left, internal p3) and P2 (bottom-right, internal p2) are shown.
    # Internal p1 (bottom-left) is derived as (p3.x, p2.y) on round-trip.
    a = Anchors(p1_pixel=(10, 90), p2_pixel=(90, 90), p3_pixel=(10, 10))
    anns = anchor_annotations(a)
    assert len(anns) == 2
    assert {ann["name"] for ann in anns} == set(ANCHOR_LABELS)
    a2 = annotations_to_anchors(anns, a)
    assert a2.p1_pixel == a.p1_pixel  # (p3.x, p2.y) = (10, 90) — derived correctly
    assert a2.p2_pixel == a.p2_pixel
    assert a2.p3_pixel == a.p3_pixel


def test_anchor_annotations_capture_events_and_selection_style():
    """``captureevents`` must be True for ``plotly_clickannotation`` to fire,
    and the selected anchor gets a distinct border so the keyboard target is
    visually obvious."""
    a = Anchors(p1_pixel=(10, 90), p2_pixel=(90, 90), p3_pixel=(10, 10))
    anns = anchor_annotations(a, selected="P2")
    by_name = {ann["name"]: ann for ann in anns}
    for name in ANCHOR_LABELS:
        assert by_name[name]["captureevents"] is True
    assert by_name["P2"]["bordercolor"] == "#000"
    assert by_name["P2"]["borderwidth"] == 3
    assert by_name["P1"]["bordercolor"] != "#000"


def test_guide_lines_track_anchors():
    a = Anchors(p1_pixel=(50, 150), p2_pixel=(250, 150), p3_pixel=(50, 30))
    traces = guide_line_traces(a, width=300, height=200)
    assert len(traces) == 4
    # 0,1 horizontal: y constant; 2,3 vertical: x constant.
    assert tuple(traces[0].y) == (150, 150)  # P1-P2 horizontal
    assert tuple(traces[1].y) == (30, 30)    # P3 horizontal
    assert tuple(traces[2].x) == (50, 50)    # P1-P3 vertical
    assert tuple(traces[3].x) == (250, 250)  # P2 vertical


def test_enforce_anchor_constraints_p1_leads():
    prev = Anchors(p1_pixel=(100, 200), p2_pixel=(300, 200), p3_pixel=(100, 50))
    # User moved P1 to (110, 205). Expect: P2.y follows (205); P3.x follows (110).
    raw = Anchors(p1_pixel=(110, 205), p2_pixel=(300, 200), p3_pixel=(100, 50))
    out = enforce_anchor_constraints(raw, prev)
    assert out.p1_pixel == (110, 205)
    assert out.p2_pixel == (300, 205)
    assert out.p3_pixel == (110, 50)


def test_enforce_anchor_constraints_p2_leads():
    prev = Anchors(p1_pixel=(100, 200), p2_pixel=(300, 200), p3_pixel=(100, 50))
    # User moved P2 to (320, 210). Expect: P1.y follows (210); P3 unchanged in x.
    raw = Anchors(p1_pixel=(100, 200), p2_pixel=(320, 210), p3_pixel=(100, 50))
    out = enforce_anchor_constraints(raw, prev)
    assert out.p1_pixel == (100, 210)
    assert out.p2_pixel == (320, 210)
    assert out.p3_pixel == (100, 50)


def test_enforce_anchor_constraints_p3_leads():
    prev = Anchors(p1_pixel=(100, 200), p2_pixel=(300, 200), p3_pixel=(100, 50))
    # User moved P3 to (115, 40). Expect: P1.x follows (115); P2.y unchanged.
    raw = Anchors(p1_pixel=(100, 200), p2_pixel=(300, 200), p3_pixel=(115, 40))
    out = enforce_anchor_constraints(raw, prev)
    assert out.p1_pixel == (115, 200)
    assert out.p2_pixel == (300, 200)
    assert out.p3_pixel == (115, 40)


def test_all_calibration_shapes_is_empty():
    """Anchors moved to annotations, so layout.shapes is empty."""
    a = Anchors(p1_pixel=(50, 150), p2_pixel=(250, 150), p3_pixel=(50, 30))
    assert all_calibration_shapes(a, width=300, height=200) == []


def test_manual_calibration_via_controller_produces_overlay():
    """Upload image + CSV, apply manual calibration, build the overlay figure."""
    pv = PlotVerifyApp()
    image_bytes = TEST_IMAGE.read_bytes()
    fid = pv.add_image(TEST_IMAGE.name, image_bytes)
    pv.add_csv(fid, TEST_CSV.name, TEST_CSV.read_text(encoding="utf-8"))

    fs = pv.active
    h, w = fs.image_rgb.shape[:2]
    # Set anchors that approximate the actual axis bounds of the test image —
    # the magnitudes are not what the test verifies, only that calibration
    # math succeeds and we end up with a renderable figure.
    anchors = Anchors(
        p1_pixel=(0.10 * w, 0.90 * h),
        p2_pixel=(0.90 * w, 0.90 * h),
        p3_pixel=(0.10 * w, 0.10 * h),
        p1_data_x=0.0,
        p2_data_x=24.0,
        p1_data_y=0.0,
        p3_data_y=500.0,
    )
    result = pv.apply_manual_calibration(fid, anchors)
    assert result.success
    cal = cal_dict_from_result(result)
    assert cal["applied"]
    assert cal["x_scale"] > 0
    assert cal["y_scale"] < 0  # pixel-y grows downward, so y_scale is negative

    df = fs.overlay.to_dataframe()
    traces = build_overlay_traces(df)
    fig = build_data_overlay_figure(fs.image_rgb, traces, cal)
    assert isinstance(fig, go.Figure)
    # At least one trace per series.
    assert len(fig.data) >= df["series"].nunique()


def test_export_csv_round_trips():
    pv = PlotVerifyApp()
    fid = pv.add_image(TEST_IMAGE.name, TEST_IMAGE.read_bytes())
    pv.add_csv(fid, TEST_CSV.name, TEST_CSV.read_text(encoding="utf-8"))
    data = pv.export_csv(fid)
    assert data.startswith(b"series,")
    # Round-trip with audit cols.
    fs = pv.active
    first_pt = next(iter(fs.overlay.points()))
    fs.overlay.edit_point(first_pt.point_id, first_pt.x + 1.0, first_pt.y + 1.0)
    data_audit = pv.export_csv(fid, include_audit_cols=True)
    assert b"original_x" in data_audit
    assert b"edited" in data_audit


def test_anchors_from_result_falls_back():
    """If the calibration has no result, anchors_from_result returns the fallback."""
    fallback = Anchors(p1_pixel=(1, 2), p2_pixel=(3, 4), p3_pixel=(5, 6))
    assert anchors_from_result(None, fallback) is fallback


def test_stale_file_id_returns_none_via_get():
    """Regression for Phase 1: a `file_id_rv` that lingers after the file was
    removed from `pv.state.files` must surface as `None` via `.get(fid)`
    rather than crashing with KeyError. The render-side panels in
    `shiny_app/app.py` rely on this idiom."""
    pv = PlotVerifyApp()
    fid = pv.add_image(TEST_IMAGE.name, TEST_IMAGE.read_bytes())
    assert pv.state.files.get(fid) is not None
    pv.state.remove_file(fid)
    assert pv.state.files.get(fid) is None
    # And a never-seen fid is also safe.
    assert pv.state.files.get("not-a-real-fid") is None


def test_safe_int_falls_back_and_traces():
    """Phase 3: _safe_int returns the default when the getter raises and
    emits a trace event (visible under PLOTVERIFY_TRACE=1)."""
    from shiny_app import app as app_mod

    def raising():
        raise RuntimeError("input not bound")

    traces = []
    orig_trace = app_mod._trace
    app_mod._trace = lambda tag, **kw: traces.append((tag, kw))
    try:
        assert app_mod._safe_int(raising, 90, "band_y_extra") == 90
    finally:
        app_mod._trace = orig_trace
    assert any("safe_int_fallback" in t for t, _ in traces)


def test_safe_int_coerces_valid_input():
    from shiny_app import app as app_mod
    assert app_mod._safe_int(lambda: "42", 0) == 42
    assert app_mod._safe_int(lambda: None, 7) == 7  # `or` fallback
    assert app_mod._safe_int(lambda: 0, 5) == 5     # `or` treats 0 as falsy
    assert app_mod._safe_int(lambda: 3, 5) == 3


def test_safe_float_and_safe_str():
    from shiny_app import app as app_mod
    assert app_mod._safe_float(lambda: "3.14", 0.0) == 3.14
    assert app_mod._safe_float(lambda: None, 0.1) == 0.1
    assert app_mod._safe_str(lambda: "Overlay", "Calibrate") == "Overlay"
    assert app_mod._safe_str(lambda: None, "Calibrate") == "Calibrate"


def test_user_error_traces_even_without_shiny_session():
    """_user_error must not raise when invoked outside a Shiny session
    (notifications fail, but tracing still happens)."""
    from shiny_app import app as app_mod

    traces = []
    orig_trace = app_mod._trace
    app_mod._trace = lambda tag, **kw: traces.append((tag, kw))
    try:
        # Should not raise even though there's no active Shiny session for
        # ui.notification_show to attach to.
        app_mod._user_error("Test failure", RuntimeError("boom"))
    finally:
        app_mod._trace = orig_trace
    # The error trace fires unconditionally.
    assert any("Test failure.error" == t for t, _ in traces)


# ---------------------------------------------------------------------------
# JSON-only UI gating
# ---------------------------------------------------------------------------

def test_json_only_ui_has_no_calibrate_surface():
    from shiny_app import app as app_mod

    html = str(app_mod._make_ui(json_only=True))
    assert ">Calibrate<" not in html
    for missing_id in ("image_upload", "csv_upload", "plot_type_select",
                       "ocr_banner", "reset_anchors"):
        assert missing_id not in html, missing_id
    for present_id in ("json_upload", "json_paste", "json_apply",
                       "session_status"):
        assert present_id in html, present_id


def test_full_ui_keeps_all_tabs_and_uploads():
    from shiny_app import app as app_mod

    html = str(app_mod._make_ui(json_only=False))
    assert ">Calibrate<" in html
    for present_id in ("image_upload", "csv_upload", "json_apply",
                       "plot_type_select"):
        assert present_id in html, present_id


# ---------------------------------------------------------------------------
# Edit panel: linked-bounds representation
# ---------------------------------------------------------------------------

def test_no_force_symmetry_input_remains():
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "shiny_app" / "app.py"
    assert "force_symmetry" not in src.read_text()


def test_overlay_tab_has_conditional_bound_rows():
    from shiny_app import app as app_mod

    html = str(app_mod._make_ui(json_only=False))
    # The static edit panel is server-rendered, but the accordion shell and
    # export controls are in the UI tree; Export starts collapsed.
    assert "overlay_controls_accordion" in html
