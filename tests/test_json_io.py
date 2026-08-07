"""Tests for the agent JSON import/export module."""
from __future__ import annotations

import base64
import json
import math

import numpy as np
import pandas as pd
import pytest

from plotverify_core.json_io import (
    JsonLoadResult,
    export_json,
    parse_agent_json,
)
from plotverify_core.session import Anchors, PerFileState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _tiny_png_b64() -> str:
    """A valid 2x2 red PNG as a data URI."""
    import struct
    import zlib

    def _chunk(ctype: bytes, data: bytes) -> bytes:
        c = ctype + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
    raw = b""
    for _ in range(2):
        raw += b"\x00" + b"\xff\x00\x00" * 2  # filter=None, 2 red pixels
    idat = _chunk(b"IDAT", zlib.compress(raw))
    iend = _chunk(b"IEND", b"")
    png_bytes = sig + ihdr + idat + iend
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _minimal_json(**overrides) -> str:
    doc = {
        "schema_version": "1.0",
        "image": {
            "filename": "test.png",
            "width_px": 2,
            "height_px": 2,
            "data_uri": _tiny_png_b64(),
        },
        "axes": {
            "x": {
                "scale": "linear",
                "calibration": [
                    {"pixel": 100, "value": 0},
                    {"pixel": 700, "value": 24},
                ],
            },
            "y": {
                "scale": "linear",
                "calibration": [
                    {"pixel": 50, "value": 100},
                    {"pixel": 500, "value": 0},
                ],
            },
        },
        "series": [{"key": "Drug A", "color": "#ff0000"}],
        "rows": [
            {"series": "Drug A", "x": 0, "y": 50, "series_color": "#ff0000"},
            {"series": "Drug A", "x": 12, "y": 75, "series_color": "#ff0000"},
            {"series": "Drug A", "x": 24, "y": 25, "series_color": "#ff0000"},
        ],
    }
    doc.update(overrides)
    return json.dumps(doc)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_invalid_json():
    result = parse_agent_json("not json {{{")
    assert result.error is not None
    assert "Invalid JSON" in result.error


def test_unsupported_schema_version():
    result = parse_agent_json(json.dumps({"schema_version": "99.0"}))
    assert result.error is not None
    assert "Unsupported schema_version" in result.error


def test_schema_version_prefix_match():
    doc = json.loads(_minimal_json())
    doc["schema_version"] = "1.2.3"
    result = parse_agent_json(json.dumps(doc))
    assert result.error is None


# ---------------------------------------------------------------------------
# Anchor mapping — linear
# ---------------------------------------------------------------------------

def test_anchor_mapping_linear():
    result = parse_agent_json(_minimal_json())
    assert result.error is None
    a = result.anchors
    assert a is not None

    # p3 = top-left: leftmost x-pixel, topmost y-pixel
    assert a.p3_pixel == (100.0, 50.0)
    # p2 = bottom-right: rightmost x-pixel, bottommost y-pixel
    assert a.p2_pixel == (700.0, 500.0)
    # p1 = bottom-left (derived)
    assert a.p1_pixel == (100.0, 500.0)

    assert a.p1_data_x == 0.0   # left edge
    assert a.p2_data_x == 24.0  # right edge
    assert a.p1_data_y == 0.0   # bottom (largest pixel)
    assert a.p3_data_y == 100.0 # top (smallest pixel)

    assert a.x_log_base is None
    assert a.y_log_base is None


def test_anchor_mapping_unsorted():
    """Calibration points not sorted by pixel are sorted internally."""
    doc = json.loads(_minimal_json())
    doc["axes"]["x"]["calibration"] = [
        {"pixel": 700, "value": 24},
        {"pixel": 100, "value": 0},
    ]
    doc["axes"]["y"]["calibration"] = [
        {"pixel": 500, "value": 0},
        {"pixel": 50, "value": 100},
    ]
    result = parse_agent_json(json.dumps(doc))
    a = result.anchors
    assert a is not None
    assert a.p3_pixel == (100.0, 50.0)
    assert a.p2_pixel == (700.0, 500.0)
    assert a.p1_data_x == 0.0
    assert a.p2_data_x == 24.0


def test_anchor_mapping_more_than_two():
    """When > 2 calibration points per axis, uses first and last by pixel."""
    doc = json.loads(_minimal_json())
    doc["axes"]["x"]["calibration"] = [
        {"pixel": 100, "value": 0},
        {"pixel": 400, "value": 12},
        {"pixel": 700, "value": 24},
    ]
    result = parse_agent_json(json.dumps(doc))
    a = result.anchors
    assert a is not None
    assert a.p1_data_x == 0.0
    assert a.p2_data_x == 24.0
    assert any("intermediate points ignored" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Anchor mapping — log
# ---------------------------------------------------------------------------

def test_anchor_mapping_log_axis():
    doc = json.loads(_minimal_json())
    doc["axes"]["y"]["scale"] = "log"
    doc["axes"]["y"]["calibration"] = [
        {"pixel": 50, "value": 1000},
        {"pixel": 500, "value": 0.1},
    ]
    result = parse_agent_json(json.dumps(doc))
    a = result.anchors
    assert a is not None
    assert a.y_log_base == 10.0
    assert a.p3_data_y == 1000.0  # top (smallest pixel)
    assert a.p1_data_y == 0.1     # bottom (largest pixel)


def test_anchor_mapping_log_negative_fails():
    doc = json.loads(_minimal_json())
    doc["axes"]["y"]["scale"] = "log"
    doc["axes"]["y"]["calibration"] = [
        {"pixel": 50, "value": -10},
        {"pixel": 500, "value": 100},
    ]
    result = parse_agent_json(json.dumps(doc))
    assert result.anchors is None
    assert any("positive" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# Anchor mapping — edge cases
# ---------------------------------------------------------------------------

def test_degenerate_x_axis():
    doc = json.loads(_minimal_json())
    doc["axes"]["x"]["calibration"] = [
        {"pixel": 400, "value": 0},
        {"pixel": 400, "value": 24},
    ]
    result = parse_agent_json(json.dumps(doc))
    assert result.anchors is None
    assert any("degenerate" in w.lower() for w in result.warnings)


def test_degenerate_y_axis():
    doc = json.loads(_minimal_json())
    doc["axes"]["y"]["calibration"] = [
        {"pixel": 300, "value": 0},
        {"pixel": 300, "value": 100},
    ]
    result = parse_agent_json(json.dumps(doc))
    assert result.anchors is None
    assert any("degenerate" in w.lower() for w in result.warnings)


def test_missing_axes_block():
    doc = json.loads(_minimal_json())
    del doc["axes"]
    result = parse_agent_json(json.dumps(doc))
    assert result.anchors is None
    assert result.image_bytes is not None  # partial import OK


def test_too_few_calibration_points():
    doc = json.loads(_minimal_json())
    doc["axes"]["x"]["calibration"] = [{"pixel": 100, "value": 0}]
    result = parse_agent_json(json.dumps(doc))
    assert result.anchors is None


# ---------------------------------------------------------------------------
# DataFrame extraction
# ---------------------------------------------------------------------------

def test_rows_to_dataframe():
    result = parse_agent_json(_minimal_json())
    assert result.csv_df is not None
    assert len(result.csv_df) == 3
    assert set(result.csv_df.columns) >= {"series", "x", "y", "y_err_lower", "y_err_upper", "series_color"}


def test_rows_missing_optional_columns():
    doc = json.loads(_minimal_json())
    doc["rows"] = [{"series": "A", "x": 1, "y": 2}]
    result = parse_agent_json(json.dumps(doc))
    df = result.csv_df
    assert df is not None
    assert "y_err_lower" in df.columns
    assert "y_err_upper" in df.columns
    assert pd.isna(df["y_err_lower"].iloc[0])


def test_rows_reversed_error_bars():
    doc = json.loads(_minimal_json())
    doc["rows"] = [
        {"series": "A", "x": 10, "y": 50, "y_err_lower": 60, "y_err_upper": 40},
    ]
    result = parse_agent_json(json.dumps(doc))
    df = result.csv_df
    assert df is not None
    assert df["y_err_lower"].iloc[0] == 40
    assert df["y_err_upper"].iloc[0] == 60


def test_no_rows():
    doc = json.loads(_minimal_json())
    del doc["rows"]
    result = parse_agent_json(json.dumps(doc))
    assert result.csv_df is None
    assert result.anchors is not None  # partial import OK


def test_default_series():
    doc = json.loads(_minimal_json())
    doc["rows"] = [{"x": 1, "y": 2}]
    result = parse_agent_json(json.dumps(doc))
    assert result.csv_df is not None
    assert result.csv_df["series"].iloc[0] == "Data"


# ---------------------------------------------------------------------------
# Image extraction
# ---------------------------------------------------------------------------

def test_image_data_uri_decode():
    result = parse_agent_json(_minimal_json())
    assert result.image_bytes is not None
    assert len(result.image_bytes) > 0
    assert result.image_filename == "test.png"


def test_missing_image():
    doc = json.loads(_minimal_json())
    del doc["image"]["data_uri"]
    result = parse_agent_json(json.dumps(doc))
    assert result.image_bytes is None
    assert any("data_uri" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Plot type mapping
# ---------------------------------------------------------------------------

def test_plot_type_scatter():
    result = parse_agent_json(_minimal_json(plot_type="scatter"))
    assert result.plot_type == "scatter"


def test_plot_type_line_timeseries():
    result = parse_agent_json(_minimal_json(plot_type="line_timeseries"))
    assert result.plot_type == "time_series"


def test_plot_type_forest():
    doc = json.loads(_minimal_json(plot_type="forest"))
    doc["rows"] = [
        {"series": "Study A", "x": 1.5, "y_err_lower": 0.8, "y_err_upper": 2.2},
    ]
    result = parse_agent_json(json.dumps(doc))
    assert result.plot_type == "forest"


def test_plot_type_unknown_warns():
    result = parse_agent_json(_minimal_json(plot_type="volcano"))
    assert result.plot_type == "time_series"
    assert any("Unknown plot type" in w for w in result.warnings)


def test_plot_type_bar():
    result = parse_agent_json(_minimal_json(plot_type="bar"))
    assert result.plot_type == "bar"
    assert not any("not fully supported" in w for w in result.warnings)


def test_plot_type_box():
    result = parse_agent_json(_minimal_json(plot_type="box"))
    assert result.plot_type == "box"
    assert not any("not fully supported" in w for w in result.warnings)


def test_plot_type_kaplan_meier():
    result = parse_agent_json(_minimal_json(plot_type="kaplan_meier"))
    assert result.plot_type == "kaplan_meier"
    assert not any("not fully supported" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Series colors
# ---------------------------------------------------------------------------

def test_series_colors_extracted():
    result = parse_agent_json(_minimal_json())
    assert result.series_colors is not None
    assert result.series_colors.get("Drug A") == "#ff0000"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def test_export_basic():
    """Export a PerFileState with overlay produces valid JSON."""
    from plotverify_core.overlay_model import EditableOverlay

    df = pd.DataFrame({
        "series": ["A", "A"],
        "x": [1.0, 2.0],
        "y": [10.0, 20.0],
        "y_err_lower": [8.0, 18.0],
        "y_err_upper": [12.0, 22.0],
        "series_color": ["#ff0000", "#ff0000"],
    })
    fs = PerFileState(
        file_id="test#abc",
        image_filename="test.png",
        image_bytes=base64.b64decode(
            _tiny_png_b64().split(",", 1)[1]
        ),
        image_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        csv_df=df,
        overlay=EditableOverlay(df),
        plot_type="scatter",
    )
    text = export_json(fs)
    doc = json.loads(text)
    assert doc["schema_version"] == "1.1"
    assert doc["plot_type"] == "scatter"
    assert len(doc["rows"]) == 2
    assert doc["image"]["filename"] == "test.png"


def test_export_round_trip():
    """export → re-parse produces equivalent data."""
    from plotverify_core.overlay_model import EditableOverlay

    df = pd.DataFrame({
        "series": ["A", "B"],
        "x": [1.0, 2.0],
        "y": [10.0, 20.0],
        "y_err_lower": [pd.NA, pd.NA],
        "y_err_upper": [pd.NA, pd.NA],
        "series_color": ["#ff0000", "#0000ff"],
    })
    fs = PerFileState(
        file_id="test#abc",
        image_filename="round.png",
        image_bytes=base64.b64decode(
            _tiny_png_b64().split(",", 1)[1]
        ),
        image_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        csv_df=df,
        overlay=EditableOverlay(df),
        plot_type="scatter",
    )
    exported = export_json(fs, include_image=True)
    result = parse_agent_json(exported)
    assert result.error is None
    assert result.image_bytes is not None
    assert result.csv_df is not None
    assert len(result.csv_df) == 2
    assert list(result.csv_df["x"]) == [1.0, 2.0]
    assert list(result.csv_df["y"]) == [10.0, 20.0]


# ---------------------------------------------------------------------------
# Controller integration
# ---------------------------------------------------------------------------

def test_add_json_controller():
    """PlotVerifyApp.add_json() integrates image + calibration + data."""
    from plotverify_core.app import PlotVerifyApp

    pv = PlotVerifyApp()
    fid, result = pv.add_json(_minimal_json())
    assert fid is not None
    fs = pv.state.files[fid]
    assert fs.image_rgb is not None
    assert fs.detection_result is not None
    assert fs.detection_result.success
    assert fs.csv_df is not None
    assert fs.overlay is not None
    assert len(fs.overlay.points()) == 3


def test_add_json_no_image_raises():
    """JSON without data_uri raises ValueError."""
    from plotverify_core.app import PlotVerifyApp

    doc = json.loads(_minimal_json())
    del doc["image"]["data_uri"]
    pv = PlotVerifyApp()
    with pytest.raises(ValueError, match="data_uri"):
        pv.add_json(json.dumps(doc))


def test_add_json_partial_no_rows():
    """JSON with image+calibration but no rows still works."""
    from plotverify_core.app import PlotVerifyApp

    doc = json.loads(_minimal_json())
    del doc["rows"]
    pv = PlotVerifyApp()
    fid, result = pv.add_json(json.dumps(doc))
    fs = pv.state.files[fid]
    assert fs.detection_result is not None
    assert fs.detection_result.success
    assert fs.csv_df is None


def test_add_json_partial_no_calibration():
    """JSON with image+rows but no calibration still works."""
    from plotverify_core.app import PlotVerifyApp

    doc = json.loads(_minimal_json())
    del doc["axes"]
    pv = PlotVerifyApp()
    fid, result = pv.add_json(json.dumps(doc))
    fs = pv.state.files[fid]
    assert fs.detection_result is None
    assert fs.csv_df is not None


def test_export_json_controller():
    """PlotVerifyApp.export_json() returns bytes."""
    from plotverify_core.app import PlotVerifyApp

    pv = PlotVerifyApp()
    fid, _ = pv.add_json(_minimal_json())
    data = pv.export_json(fid)
    assert isinstance(data, bytes)
    doc = json.loads(data.decode("utf-8"))
    assert doc["schema_version"] == "1.1"
    assert len(doc["rows"]) == 3


# ---------------------------------------------------------------------------
# Orientation (schema 1.1)
# ---------------------------------------------------------------------------

def test_orientation_default_vertical():
    result = parse_agent_json(_minimal_json())
    assert result.orientation == "vertical"


def test_orientation_horizontal_bar():
    result = parse_agent_json(
        _minimal_json(plot_type="bar", orientation="horizontal")
    )
    assert result.error is None
    assert result.orientation == "horizontal"


def test_orientation_horizontal_box():
    result = parse_agent_json(
        _minimal_json(plot_type="box", orientation="horizontal")
    )
    assert result.orientation == "horizontal"


def test_orientation_unknown_warns():
    result = parse_agent_json(_minimal_json(orientation="sideways"))
    assert result.orientation == "vertical"
    assert any("Unknown orientation" in w for w in result.warnings)


def test_orientation_horizontal_scatter_warns_and_coerces():
    result = parse_agent_json(
        _minimal_json(plot_type="scatter", orientation="horizontal")
    )
    assert result.orientation == "vertical"
    assert any("has no effect" in w for w in result.warnings)


def test_orientation_forest_implicit_horizontal():
    doc = json.loads(_minimal_json(plot_type="forest"))
    doc["rows"] = [
        {"series": "S", "value": 1.2, "value_err_lower": 0.9,
         "value_err_upper": 1.6},
        {"series": "S", "value": 0.8, "value_err_lower": 0.5,
         "value_err_upper": 1.1},
    ]
    result = parse_agent_json(json.dumps(doc))
    assert result.orientation == "horizontal"
    # No orientation key present -> no warning either
    assert not any("orientation" in w.lower() for w in result.warnings)


def test_orientation_forest_explicit_vertical_warns():
    doc = json.loads(_minimal_json(plot_type="forest", orientation="vertical"))
    doc["rows"] = [
        {"series": "S", "value": 1.2},
        {"series": "S", "value": 0.8},
    ]
    result = parse_agent_json(json.dumps(doc))
    assert result.orientation == "horizontal"
    assert any("always horizontal" in w for w in result.warnings)


def test_horizontal_bar_value_alias_maps_to_x():
    doc = json.loads(_minimal_json(plot_type="bar", orientation="horizontal"))
    doc["rows"] = [
        {"series": "S", "value": 5.0, "y": 1.0,
         "value_err_lower": 4.0, "value_err_upper": 6.0},
        {"series": "S", "value": 3.0, "y": 2.0,
         "value_err_lower": 2.5, "value_err_upper": 3.5},
    ]
    result = parse_agent_json(json.dumps(doc))
    assert result.error is None
    df = result.csv_df
    assert df is not None
    assert list(df["x"]) == [5.0, 3.0]
    assert list(df["y"]) == [1.0, 2.0]
    assert list(df["y_err_lower"]) == [4.0, 2.5]


def test_horizontal_reversed_error_bars_swap_against_x():
    doc = json.loads(_minimal_json(plot_type="bar", orientation="horizontal"))
    doc["rows"] = [
        # bounds bracket x=5 but reversed
        {"series": "S", "x": 5.0, "y": 1.0,
         "y_err_lower": 6.0, "y_err_upper": 4.0},
    ]
    result = parse_agent_json(json.dumps(doc))
    df = result.csv_df
    assert df is not None
    assert float(df["y_err_lower"].iloc[0]) == 4.0
    assert float(df["y_err_upper"].iloc[0]) == 6.0


# ---------------------------------------------------------------------------
# log_base (schema 1.1)
# ---------------------------------------------------------------------------

def _log_axes(x_base=None, y_base=None, x_scale="log", y_scale="log"):
    axes = {
        "x": {
            "scale": x_scale,
            "calibration": [
                {"pixel": 100, "value": 1},
                {"pixel": 700, "value": 100},
            ],
        },
        "y": {
            "scale": y_scale,
            "calibration": [
                {"pixel": 50, "value": 100},
                {"pixel": 500, "value": 1},
            ],
        },
    }
    if x_base is not None:
        axes["x"]["log_base"] = x_base
    if y_base is not None:
        axes["y"]["log_base"] = y_base
    return axes


def test_log_base_default_still_10():
    result = parse_agent_json(_minimal_json(axes=_log_axes()))
    assert result.anchors.x_log_base == 10.0
    assert result.anchors.y_log_base == 10.0


def test_log_base_explicit():
    result = parse_agent_json(_minimal_json(axes=_log_axes(x_base=2, y_base=5)))
    assert result.anchors.x_log_base == 2.0
    assert result.anchors.y_log_base == 5.0


def test_log_base_e_string():
    result = parse_agent_json(_minimal_json(axes=_log_axes(y_base="e")))
    assert result.anchors.y_log_base == pytest.approx(math.e)


def test_log_base_invalid_falls_back_to_10_warns():
    result = parse_agent_json(_minimal_json(axes=_log_axes(x_base=0.5)))
    assert result.anchors.x_log_base == 10.0
    assert any("Invalid axes.x.log_base" in w for w in result.warnings)

    result = parse_agent_json(_minimal_json(axes=_log_axes(y_base="huge")))
    assert result.anchors.y_log_base == 10.0
    assert any("Invalid axes.y.log_base" in w for w in result.warnings)


def test_log_base_with_linear_scale_ignored_warns():
    axes = _log_axes(x_base=2, x_scale="linear", y_scale="linear")
    result = parse_agent_json(_minimal_json(axes=axes))
    assert result.anchors.x_log_base is None
    assert any("log_base ignored" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Anchor rescaling
# ---------------------------------------------------------------------------

def test_rescale_anchors_scales_pixels_only():
    from plotverify_core.json_io import rescale_anchors

    a = Anchors(
        p1_pixel=(100.0, 500.0), p2_pixel=(700.0, 500.0),
        p3_pixel=(100.0, 50.0),
        p1_data_x=0.0, p2_data_x=24.0, p1_data_y=0.0, p3_data_y=100.0,
        x_log_base=None, y_log_base=2.0,
    )
    b = rescale_anchors(a, 0.5)
    assert b.p1_pixel == (50.0, 250.0)
    assert b.p2_pixel == (350.0, 250.0)
    assert b.p3_pixel == (50.0, 25.0)
    assert b.p2_data_x == 24.0
    assert b.y_log_base == 2.0
    # original untouched
    assert a.p1_pixel == (100.0, 500.0)


def test_add_json_warns_on_dimension_mismatch():
    """JSON declaring wrong image dims rescales anchors and warns."""
    from plotverify_core.app import PlotVerifyApp

    doc = json.loads(_minimal_json())
    doc["image"]["width_px"] = 4  # real decoded PNG is 2 px wide
    pv = PlotVerifyApp()
    fid, result = pv.add_json(json.dumps(doc))
    assert any("anchors rescaled" in w for w in result.warnings)
    fs = pv.state.files[fid]
    # anchors halved: x pixel 100 -> 50
    assert fs.manual_anchors.p3_pixel[0] == pytest.approx(50.0)


def test_add_json_sets_orientation():
    from plotverify_core.app import PlotVerifyApp

    doc = json.loads(_minimal_json(plot_type="bar", orientation="horizontal"))
    doc["rows"] = [
        {"series": "S", "x": 5.0, "y": 1.0},
        {"series": "S", "x": 3.0, "y": 2.0},
    ]
    pv = PlotVerifyApp()
    fid, result = pv.add_json(json.dumps(doc))
    assert pv.state.files[fid].orientation == "horizontal"
    assert pv.state.files[fid].plot_type == "bar"


# ---------------------------------------------------------------------------
# Export round-trip (schema 1.1)
# ---------------------------------------------------------------------------

def test_export_writes_schema_1_1():
    from plotverify_core.app import PlotVerifyApp

    pv = PlotVerifyApp()
    fid, _ = pv.add_json(_minimal_json())
    doc = json.loads(pv.export_json(fid).decode("utf-8"))
    assert doc["schema_version"] == "1.1"
    assert doc["orientation"] == "vertical"


def test_export_round_trip_orientation_and_log_base():
    from plotverify_core.app import PlotVerifyApp

    src = json.loads(_minimal_json(plot_type="bar", orientation="horizontal"))
    src["axes"] = _log_axes(x_base=2, y_base="e")
    src["rows"] = [
        {"series": "S", "x": 5.0, "y": 10.0},
        {"series": "S", "x": 3.0, "y": 20.0},
    ]
    pv = PlotVerifyApp()
    fid, _ = pv.add_json(json.dumps(src))

    exported = pv.export_json(fid, include_image=True).decode("utf-8")
    doc = json.loads(exported)
    assert doc["orientation"] == "horizontal"
    assert doc["axes"]["x"]["log_base"] == pytest.approx(2.0)
    assert doc["axes"]["y"]["log_base"] == pytest.approx(math.e)

    reparsed = parse_agent_json(exported)
    assert reparsed.error is None
    assert reparsed.orientation == "horizontal"
    assert reparsed.anchors.x_log_base == pytest.approx(2.0)
    assert reparsed.anchors.y_log_base == pytest.approx(math.e)
