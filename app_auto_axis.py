"""PlotVerify — Scientific plot verification Streamlit app.

Verifies AI-extracted plot data by overlaying it on the original image and
masking each series by its known color in the source figure. Optional monotone
cubic spline interpolation reconstructs occluded line segments.

Run with:
    streamlit run app.py
"""

import hashlib
import io
import os
import tempfile
import warnings as _py_warnings

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from scipy.interpolate import PchipInterpolator

# Silence deprecation warnings emitted by the dict-shaped legacy shims; the
# Streamlit app's transition to the typed API is being done incrementally and
# we don't want users to see the migration noise in their console.
_py_warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="axis_pipeline.legacy",
)

from axis_auto import auto_detect_axes_and_ticks, build_diagnostic_overlay
from ocr_axis import (
    auto_detect_axes_ticks_ocr,
    build_ocr_debug_overlay,
    rebuild_result_from_detection,
    update_detection_from_tick_tables,
)

# Direct pipeline access for the manual-band preview tab. The legacy shims
# above wrap the same package; using it directly here gives us the typed
# `FramePreview` / `CalibrationResult` objects without going through dicts.
from axis_pipeline import (
    CalibrationConfig,
    detect_axis_frame,
    manual_calibration,
    ocr_available,
    render_band_preview,
    render_overlay,
    run_calibration,
    x_label_band as _x_label_band,
    y_label_band as _y_label_band,
)

# UI-agnostic core (Refactor A). Pure logic — no Streamlit imports.
from plotverify_core import (
    FALLBACK_HEX,
    LARGE_IMAGE_DOWNSCALE_EDGE,
    LARGE_IMAGE_MAX_BYTES,
    LARGE_IMAGE_MAX_EDGE,
    P1P2_Y_TOLERANCE_PX,
    SeriesState,
    apply_color_mask,
    compute_calibration,
    data_to_px,
    decode_and_maybe_downscale,
    delta_e_mask as _delta_e_mask,
    hex_complement,
    hex_to_bgr,
    hex_to_hsv_opencv,
    init_series_states,
    is_valid_hex,
    load_csv as _core_load_csv,
    log10_or_none as _log10_or_none,
    px_to_data,
)


# Module constants (REQUIRED_COLUMNS, LARGE_IMAGE_*), color/mask/calibration
# math, image decode, CSV validation, and SeriesState are imported from
# plotverify_core at the top of this module. The ``cached_*`` Streamlit
# wrappers below delegate to those pure functions but layer on Streamlit's
# per-image caching.


@st.cache_data(show_spinner=False, max_entries=4)
def cached_grey_bg(image_hash, _img_bgr):
    """Pure greyscale background for the mask composite.

    Uses full desaturation (no original-colour bleed) so hidden series
    appear completely grey rather than a 40%-saturation echo of their colour.
    """
    grey = cv2.cvtColor(_img_bgr, cv2.COLOR_BGR2GRAY)
    grey_3ch = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    # Dim slightly so visible (fully-saturated) series pop clearly.
    return (grey_3ch * 0.55).astype(np.uint8)


@st.cache_data(show_spinner=False, max_entries=128)
def cached_delta_e_mask(image_hash, hex_color, threshold, _img_bgr):
    """Cache ΔE Lab masks keyed on image hash + color + threshold."""
    return _delta_e_mask(_img_bgr, hex_color, threshold)


@st.cache_data(show_spinner=False, max_entries=128)
def cached_mask(image_hash, h_min, h_max, s_min, s_max, v_min, v_max, _img_bgr):
    """Cache HSV color masks keyed on image hash + slider values."""
    return apply_color_mask(_img_bgr, h_min, h_max, s_min, s_max, v_min, v_max)


# ---------------------------------------------------------------------------
# Composite + interpolation
# ---------------------------------------------------------------------------

def interpolate_series(img_bgr, series_name, df, cal, occlusion_mask):
    """Render a PCHIP-spline path for ``series_name`` masked to occluded pixels.

    Returns a BGR overlay image (zero elsewhere) or ``None`` if the series is
    too sparse / degenerate to interpolate.
    """
    series_df = df[df["series"] == series_name]
    if len(series_df) < 2:
        return None

    color_hex = series_df["series_color"].iloc[0]
    bgr_color = hex_to_bgr(color_hex)

    # Convert each (data_x, data_y) to (pixel_x, pixel_y).
    px_pairs = []
    for x_val, y_val in zip(series_df["x"].to_numpy(), series_df["y"].to_numpy()):
        try:
            pxx, pxy = data_to_px(float(x_val), float(y_val), cal)
        except Exception:
            continue
        if not (np.isfinite(pxx) and np.isfinite(pxy)):
            continue
        px_pairs.append((pxx, pxy))

    if len(px_pairs) < 2:
        return None

    # PchipInterpolator requires strictly increasing x. Sort by pixel x and
    # drop duplicates (keeping the first y for each unique x).
    px_pairs.sort(key=lambda p: p[0])
    uniq_x = []
    uniq_y = []
    for pxx, pxy in px_pairs:
        if uniq_x and abs(pxx - uniq_x[-1]) < 1e-9:
            continue
        uniq_x.append(pxx)
        uniq_y.append(pxy)
    if len(uniq_x) < 2:
        return None

    try:
        spline = PchipInterpolator(np.asarray(uniq_x), np.asarray(uniq_y))
    except Exception:
        return None

    x_min_int = int(np.ceil(uniq_x[0]))
    x_max_int = int(np.floor(uniq_x[-1]))
    if x_max_int <= x_min_int:
        return None

    dense_px_x = np.arange(x_min_int, x_max_int + 1)
    dense_px_y = spline(dense_px_x)

    h, w = img_bgr.shape[:2]
    valid = (dense_px_x >= 0) & (dense_px_x < w) & np.isfinite(dense_px_y)
    dense_px_y_int = np.where(valid, dense_px_y, 0).astype(np.int32)
    valid &= (dense_px_y_int >= 0) & (dense_px_y_int < h)

    dense_px_x = dense_px_x[valid]
    dense_px_y_int = dense_px_y_int[valid]
    if len(dense_px_x) < 2:
        return None

    path_layer = np.zeros_like(img_bgr)
    pts = np.stack([dense_px_x, dense_px_y_int], axis=1).astype(np.int32)
    cv2.polylines(path_layer, [pts], isClosed=False, color=bgr_color, thickness=2)
    return cv2.bitwise_and(path_layer, path_layer, mask=occlusion_mask)


def _get_series_mask(image_hash, series_name, state, df, img_bgr):
    """Return the color mask for one series, using ΔE or HSV depending on state."""
    color_hex = FALLBACK_HEX
    if df is not None:
        rows = df[df["series"] == series_name]
        if len(rows):
            raw = rows["series_color"].iloc[0]
            if is_valid_hex(raw):
                color_hex = raw
    if state.get("use_delta_e", True) and df is not None:
        return cached_delta_e_mask(
            image_hash, color_hex, int(state.get("delta_e", 10)), img_bgr
        )
    return cached_mask(
        image_hash,
        int(state["h_min"]), int(state["h_max"]),
        int(state["s_min"]), int(state["s_max"]),
        int(state["v_min"]), int(state["v_max"]),
        img_bgr,
    )


def build_composite(img_bgr, image_hash, series_states, df, cal):
    """Build the Tab 1 composite image.

    Strategy:
    1.  Start from a pure-greyscale dimmed background.
    2.  Detect the plot's background grey (most common luminance value —
        for a white-background plot this is the dimmed white).
    3.  For every HIDDEN series: compute its mask and paint those pixels
        with the detected background colour, so the series vanishes entirely.
    4.  For every VISIBLE series: restore original pixel colours on top.
    """
    grey_bg = cached_grey_bg(image_hash, img_bgr)
    composite = grey_bg.copy()

    # Detect the background grey value from the most common luminance bucket.
    grey_1ch = cv2.cvtColor(grey_bg, cv2.COLOR_BGR2GRAY)
    hist = np.bincount(grey_1ch.ravel(), minlength=256)
    bg_grey = int(np.argmax(hist))
    bg_color = np.array([bg_grey, bg_grey, bg_grey], dtype=np.uint8)

    # Compute masks for every series regardless of visibility.
    all_masks = {}
    for series_name, state in series_states.items():
        all_masks[series_name] = _get_series_mask(
            image_hash, series_name, state, df, img_bgr
        )

    # Pass 1 — erase hidden series (paint their pixels with background colour).
    for series_name, m in all_masks.items():
        if not st.session_state.get(f"vis_{series_name}", True):
            composite[m > 0] = bg_color

    # Pass 2 — restore visible series in their original colours.
    visible_masks = {}
    for series_name, m in all_masks.items():
        if st.session_state.get(f"vis_{series_name}", True):
            composite[m > 0] = img_bgr[m > 0]
            visible_masks[series_name] = m

    # Optional interpolation overlay for visible series.
    if df is not None and cal is not None and cal.get("applied"):
        for series_name, state in series_states.items():
            if not st.session_state.get(f"vis_{series_name}", True):
                continue
            if not state.get("interpolate"):
                continue
            occlusion = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
            for other_name, other_mask in visible_masks.items():
                if other_name == series_name:
                    continue
                occlusion = cv2.bitwise_or(occlusion, other_mask)
            overlay = interpolate_series(img_bgr, series_name, df, cal, occlusion)
            if overlay is not None:
                composite = cv2.addWeighted(composite, 1.0, overlay, 0.7, 0)

    return cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)


def build_calibration_image(img_bgr, image_hash, series_states, df):
    """Build a BGR image for calibration with hidden series replaced by the plot background.

    Unlike build_composite, starts from the original image (no grey dimming) so
    the background color is preserved. Used by the calibration pipeline.
    """
    composite = img_bgr.copy()

    grey_1ch = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hist = np.bincount(grey_1ch.ravel(), minlength=256)
    bg_grey = int(np.argmax(hist))
    bg_color = np.array([bg_grey, bg_grey, bg_grey], dtype=np.uint8)

    for series_name, state in series_states.items():
        if st.session_state.get(f"vis_{series_name}", True):
            continue
        mask = _get_series_mask(image_hash, series_name, state, df, img_bgr)
        composite[mask > 0] = bg_color

    return composite


def _update_calibration_masked_image():
    """Rebuild the calibration masked image from current session state and write to a temp file."""
    img_bgr = st.session_state.get("image_bgr")
    if img_bgr is None:
        return
    image_hash = st.session_state.get("image_hash")
    series_states = st.session_state.get("series_states", {})
    df = st.session_state.get("df")

    masked = build_calibration_image(img_bgr, image_hash, series_states, df)

    path = st.session_state.get("cal_masked_img_path")
    if not path:
        fd, path = tempfile.mkstemp(suffix=".png", prefix="plotverify_cal_")
        os.close(fd)
        st.session_state["cal_masked_img_path"] = path

    cv2.imwrite(path, masked)
    st.session_state["cal_masked_img_bgr"] = masked


# ---------------------------------------------------------------------------
# Plotly overlay (Tab 2)
# ---------------------------------------------------------------------------

def build_overlay_figure(img_rgb, df, series_states, cal):
    """Return a Plotly figure: the calibrated image as a background with
    extracted-data scatter traces (with error bars) drawn on top in data coords.

    Honors ``cal["x_log_base"]`` / ``cal["y_log_base"]``: when set to 10, the
    corresponding Plotly axis is configured as ``type="log"`` and the image's
    layout coordinates are converted to log10 units (Plotly's layout-image
    coordinates use log10 of the data value when the axis is log-scaled).

    Data extraction is delegated to ``plotverify_core.build_overlay_traces``;
    this function is now only responsible for the Plotly rendering.
    """
    from plotverify_core import build_overlay_traces

    h, w = img_rgb.shape[:2]

    x_left = px_to_data(0, 0, cal)[0]
    x_right = px_to_data(w, 0, cal)[0]
    y_top = px_to_data(0, 0, cal)[1]
    y_bottom = px_to_data(0, h, cal)[1]

    x_log_base = cal.get("x_log_base")
    y_log_base = cal.get("y_log_base")

    fig = go.Figure()

    pil_img = Image.fromarray(img_rgb)
    # Plotly's layout-image x/y/sizex/sizey are specified in log10 of the data
    # value when the corresponding axis is set to type="log".
    img_x = min(x_left, x_right)
    img_y = max(y_top, y_bottom)
    img_sizex = abs(x_right - x_left)
    img_sizey = abs(y_top - y_bottom)
    if x_log_base and x_left > 0 and x_right > 0:
        img_x = float(np.log10(min(x_left, x_right)))
        img_sizex = float(abs(np.log10(x_right) - np.log10(x_left)))
    if y_log_base and y_top > 0 and y_bottom > 0:
        img_y = float(np.log10(max(y_top, y_bottom)))
        img_sizey = float(abs(np.log10(y_top) - np.log10(y_bottom)))
    fig.add_layout_image(
        dict(
            source=pil_img,
            xref="x", yref="y",
            x=img_x, y=img_y,
            sizex=img_sizex, sizey=img_sizey,
            xanchor="left", yanchor="top",
            sizing="stretch",
            opacity=1.0, layer="below",
        )
    )

    # Build series visibility from Streamlit widget keys; the trace builder
    # itself reads no st.session_state.
    series_visibility = {
        name: bool(st.session_state.get(f"vis_{name}", True))
        for name in df["series"].drop_duplicates().tolist()
    }
    traces = build_overlay_traces(df, series_visibility=series_visibility)

    for trace in traces:
        plot_visible = True if trace.visible else "legendonly"
        if trace.has_err.any():
            error_y = dict(
                type="data", symmetric=False,
                array=trace.err_array_plus,
                arrayminus=trace.err_array_minus,
                color=trace.overlay_color_hex, thickness=1.5, width=4,
            )
            h_str = trace.overlay_color_hex.lstrip("#")
            fill_rgba = "rgba({},{},{},0.2)".format(
                int(h_str[0:2], 16), int(h_str[2:4], 16), int(h_str[4:6], 16)
            )
            fig.add_trace(go.Scatter(
                x=trace.ribbon_x, y=trace.ribbon_y_upper,
                mode="lines", line=dict(width=0),
                legendgroup=trace.series,
                showlegend=False, visible=plot_visible, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=trace.ribbon_x, y=trace.ribbon_y_lower,
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor=fill_rgba,
                legendgroup=trace.series,
                showlegend=False, visible=plot_visible, hoverinfo="skip",
            ))
        else:
            error_y = None

        fig.add_trace(go.Scatter(
            x=trace.x, y=trace.y,
            mode="lines+markers",
            line=dict(color=trace.overlay_color_hex, width=2),
            marker=dict(color=trace.overlay_color_hex, size=8,
                        line=dict(color="rgba(0,0,0,0.5)", width=0.5)),
            name=trace.series,
            legendgroup=trace.series,
            error_y=error_y,
            visible=plot_visible,
        ))

    # When axis is log-scaled, Plotly expects `range` in log10 units.
    if x_log_base:
        x_lo, x_hi = sorted([x_left, x_right])
        fig.update_xaxes(type="log",
                         range=[float(np.log10(x_lo)), float(np.log10(x_hi))],
                         title="X (log10)")
    else:
        fig.update_xaxes(range=sorted([x_left, x_right]), title="X")
    if y_log_base:
        y_lo, y_hi = sorted([y_bottom, y_top])
        fig.update_yaxes(type="log",
                         range=[float(np.log10(y_lo)), float(np.log10(y_hi))],
                         title="Y (log10)")
    else:
        fig.update_yaxes(range=sorted([y_bottom, y_top]), title="Y")
    fig.update_layout(
        height=600,
        margin=dict(l=50, r=20, t=20, b=50),
        showlegend=True,
        legend=dict(x=1.02, y=1, xanchor="left"),
        plot_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# Session state + I/O
# ---------------------------------------------------------------------------

def _init_state():
    st.session_state.setdefault("calibration", {"applied": False})
    st.session_state.setdefault("series_states", {})
    st.session_state.setdefault("csv_hash", None)
    st.session_state.setdefault("image_hash", None)
    st.session_state.setdefault("auto_axis_detection", None)
    st.session_state.setdefault("auto_axis_result", None)
    st.session_state.setdefault("auto_axis_image_hash", None)
    st.session_state.setdefault("use_ocr_axis", True)
    st.session_state.setdefault("ocr_mask_all_text", False)
    st.session_state.setdefault("ocr_min_confidence", 0.20)
    st.session_state.setdefault("show_ocr_debug_overlay", True)
    st.session_state.setdefault("csv_has_series_color", True)
    st.session_state.setdefault("series_color_overrides", {})
    st.session_state.setdefault("picking_color_for", None)
    # Calibration-preview tab state.
    # Cache key: image_hash → FramePreview, so slider changes don't re-run
    # Phase A OCR. Per-image overrides are stored separately under
    # `band_overrides:{image_hash}` and restored when the same image reloads.
    st.session_state.setdefault("frame_preview_cache", None)
    st.session_state.setdefault("frame_preview_run_count", 0)
    st.session_state.setdefault("copy_detected_result", None)
    st.session_state.setdefault("apply_calibration_result", None)
    st.session_state.setdefault("cal_masked_img_path", None)
    st.session_state.setdefault("cal_masked_img_bgr", None)



def _auto_detection_available():
    """Return True when detection exists for the currently loaded image."""
    return (
        st.session_state.get("auto_axis_detection") is not None
        and st.session_state.get("auto_axis_image_hash") == st.session_state.get("image_hash")
    )



def _set_manual_fields_from_detection(detection):
    """Copy detected pixel coordinates and OCR data values into manual widgets."""
    if not detection:
        return False
    for label, x_key, y_key in [
        ("p1", "p1_px_x", "p1_px_y"),
        ("p2", "p2_px_x", "p2_px_y"),
        ("p3", "p3_px_x", "p3_px_y"),
    ]:
        point = detection.get(label)
        if not point:
            return False
        st.session_state[x_key] = float(point[0])
        st.session_state[y_key] = float(point[1])
    for key in ["p1_data_x", "p1_data_y", "p2_data_x", "p3_data_y"]:
        if key in detection and detection.get(key) is not None:
            st.session_state[key] = float(detection.get(key))
    return True



def _use_detected_values_in_manual_fields():
    """Button callback: pre-fill manual pixel-coordinate fields from auto detection."""
    if not _auto_detection_available():
        return
    _set_manual_fields_from_detection(st.session_state.auto_axis_detection)



def _callback_copy_detected_values():
    """on_click callback for 'Load detected values' inside the Manual override expander.

    Writes detected coords into the manual fields BEFORE the next render pass.
    """
    if not _auto_detection_available():
        st.session_state.copy_detected_result = "error:No auto-detection available for the current image."
        return
    if _set_manual_fields_from_detection(st.session_state.auto_axis_detection):
        st.session_state.copy_detected_result = "ok"
    else:
        st.session_state.copy_detected_result = "error:Could not load detection values — not all calibration points were found."


def _manual_fields_are_empty():
    """True when every P1/P2/P3 pixel field is unset (zero / unset)."""
    keys = ("p1_px_x", "p1_px_y", "p2_px_x", "p2_px_y", "p3_px_x", "p3_px_y")
    return all(abs(float(st.session_state.get(k, 0.0))) < 1e-9 for k in keys)


def _callback_apply_calibration():
    """on_click callback for the bottom 'Apply Calibration' primary button.

    If the manual P1/P2/P3 fields are empty AND a detection is available,
    pre-fills the fields from detection before computing the transform.
    Otherwise applies whatever the user has in the manual fields. Stores the
    outcome in session_state["apply_calibration_result"] for the render pass
    to display.
    """
    used_detection = False
    if _manual_fields_are_empty() and _auto_detection_available():
        ok = _set_manual_fields_from_detection(st.session_state.auto_axis_detection)
        if not ok:
            st.session_state.apply_calibration_result = (
                "error:Could not extract all calibration points from detection — "
                "open Manual override and enter values directly."
            )
            return
        used_detection = True

    # Inherit log-axis flags from the detected axis calibrations when available
    # (the new pipeline auto-detects log10 axes and sets `log_base=10.0`). The
    # manual fields hold data values in original (linear) space; compute_calibration
    # applies the log transform internally per `*_log_base`.
    x_log_base = None
    y_log_base = None
    if _auto_detection_available():
        det = st.session_state.auto_axis_detection
        x_cal = det.get("x_calibration") if det else None
        y_cal = det.get("y_calibration") if det else None
        if x_cal and x_cal.get("log_base"):
            x_log_base = float(x_cal["log_base"])
        if y_cal and y_cal.get("log_base"):
            y_log_base = float(y_cal["log_base"])

    cal = compute_calibration(
        st.session_state.get("p1_px_x", 0.0), st.session_state.get("p1_px_y", 0.0),
        st.session_state.get("p2_px_x", 0.0), st.session_state.get("p2_px_y", 0.0),
        st.session_state.get("p3_px_x", 0.0), st.session_state.get("p3_px_y", 0.0),
        st.session_state.get("p1_data_x", 0.0),
        st.session_state.get("p2_data_x", 1.0),
        st.session_state.get("p3_data_y", 1.0),
        st.session_state.get("p1_data_y", 0.0),
        x_log_base=x_log_base,
        y_log_base=y_log_base,
    )
    if cal is None:
        msg = (
            "Invalid points — P1 and P2 must differ in pixel X, "
            "and P3 must differ in pixel Y from the X-axis baseline."
        )
        if x_log_base or y_log_base:
            msg += (
                " Note: a log-scale axis was detected — data values must be "
                "strictly positive for log axes."
            )
        st.session_state.apply_calibration_result = "error:" + msg
        return
    if used_detection and _auto_detection_available():
        det = st.session_state.auto_axis_detection
        cal["auto_axis_confidence"] = float(det.get("confidence", 0.0))
        cal["auto_axis_mode"] = det.get("mode", "unknown")
    st.session_state.calibration = cal

    # When the user calibrates purely manually (no auto-detection was used),
    # synthesise a typed CalibrationResult so the diagnostic overlay can render
    # the manual P1/P2/P3 anchors. This keeps the overlay consistent with the
    # auto-detection path and unlocks features that depend on `auto_axis_result`.
    #
    # Preserve the bbox from any existing detection so render_overlay still has
    # a frame to draw on (manual_calibration's default bbox=None would short-
    # circuit the overlay renderer and make P1/P2/P3 disappear).
    if not used_detection:
        existing_result = st.session_state.get("auto_axis_result")
        existing_bbox = (
            existing_result.bbox
            if existing_result is not None and existing_result.bbox is not None
            else None
        )
        try:
            manual_result = manual_calibration(
                p1_pixel=(st.session_state.get("p1_px_x", 0.0),
                          st.session_state.get("p1_px_y", 0.0)),
                p2_pixel=(st.session_state.get("p2_px_x", 0.0),
                          st.session_state.get("p2_px_y", 0.0)),
                p3_pixel=(st.session_state.get("p3_px_x", 0.0),
                          st.session_state.get("p3_px_y", 0.0)),
                p1_data_x=float(st.session_state.get("p1_data_x", 0.0)),
                p2_data_x=float(st.session_state.get("p2_data_x", 1.0)),
                p3_data_y=float(st.session_state.get("p3_data_y", 1.0)),
                p1_data_y=float(st.session_state.get("p1_data_y", 0.0)),
                x_log_base=x_log_base,
                y_log_base=y_log_base,
                bbox=existing_bbox,
            )
            if manual_result.success:
                st.session_state.auto_axis_result = manual_result
                st.session_state.auto_axis_detection = manual_result.to_legacy_dict()
                st.session_state.auto_axis_image_hash = st.session_state.get("image_hash")
        except Exception:
            pass

    st.session_state.apply_calibration_result = "ok"


def _load_image_from_upload(image_file):
    """Decode an uploaded image into BGR + RGB arrays. No-op if unchanged."""
    img_bytes = image_file.getvalue()
    img_hash = hashlib.md5(img_bytes).hexdigest()
    if (st.session_state.image_hash == img_hash
            and "image_bgr" in st.session_state):
        return
    load = decode_and_maybe_downscale(
        img_bytes,
        downscale=bool(st.session_state.get("downscale_large_uploads", False)),
    )
    if load.error is not None:
        st.error(load.error)
        return
    # Surface size/downscale warnings to the UI.
    for msg in load.warnings:
        if msg.startswith("Downscaled"):
            st.info(msg)
        else:
            st.warning(msg)
    st.session_state["image_downscale_factor"] = load.downscale_factor

    img_rgb = load.img_rgb
    img_bgr = load.img_bgr
    st.session_state.image_rgb = img_rgb
    st.session_state.image_bgr = img_bgr
    st.session_state.image_hash = img_hash
    st.session_state.auto_axis_detection = None
    st.session_state.auto_axis_result = None
    st.session_state.auto_axis_image_hash = None
    st.session_state.frame_preview_cache = None
    # Drop any calibration tied to the previous image — pixel coords no longer match.
    st.session_state.calibration = {"applied": False}
    for k in (
        "p1_px_x", "p1_px_y", "p1_data_x", "p1_data_y",
        "p2_px_x", "p2_px_y", "p2_data_x",
        "p3_px_x", "p3_px_y", "p3_data_y",
        "copy_detected_result", "apply_calibration_result",
        "cal_masked_img_path", "cal_masked_img_bgr",
    ):
        st.session_state.pop(k, None)


def _load_csv(csv_source):
    """Streamlit wrapper around plotverify_core.load_csv.

    Surfaces the typed `LoadReport` as `st.error` / `st.warning` calls and
    writes the legacy session-state flags (`csv_has_series_color`,
    `error_bar_reversed_count`) used elsewhere in the app.
    """
    df, report = _core_load_csv(csv_source)
    if report.error is not None:
        st.error(report.error)
        return None
    st.session_state["csv_has_series_color"] = report.has_series_color_column
    st.session_state["error_bar_reversed_count"] = report.n_reversed_error_bars
    for msg in report.warnings:
        st.warning(msg)
    if report.n_reversed_error_bars > 0 and report.reversed_samples:
        st.caption(
            "Reversed-error sample rows: " +
            ", ".join(f"{s}@x={x:g}" for s, x in report.reversed_samples)
        )
    return df


def _init_series_states(df):
    """Build initial per-series control state from the DataFrame's colors."""
    states = {}
    for series_name in df["series"].drop_duplicates().tolist():
        color_hex = df[df["series"] == series_name]["series_color"].iloc[0]
        if not is_valid_hex(color_hex):
            color_hex = FALLBACK_HEX
        try:
            h, s, v = hex_to_hsv_opencv(color_hex)
        except Exception:
            h, s, v = 0, 128, 128
        states[series_name] = {
            "use_delta_e": True,
            "delta_e": 10,
            "h_min": max(0, h - 15),
            "h_max": min(179, h + 15),
            "s_min": max(0, s - 60),
            "s_max": min(255, s + 60),
            "v_min": max(0, v - 60),
            "v_max": min(255, v + 60),
            "interpolate": False,
        }
        # Seed the checkbox widget key directly in session state.
        # This ensures the widget and series_states["visible"] are always
        # in sync from the very first render, with no value= conflict.
        st.session_state[f"vis_{series_name}"] = True
    return states


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    df = st.session_state.get("df")

    with st.sidebar:
        st.header("File Inputs")
        image_file = st.file_uploader(
            "Plot image",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
        )
        st.toggle(
            "Downscale large uploads",
            key="downscale_large_uploads",
            help=(
                f"When the image exceeds {LARGE_IMAGE_MAX_EDGE}px on its longest "
                f"edge, resize it to {LARGE_IMAGE_DOWNSCALE_EDGE}px before "
                "processing. OCR and dark-mask preparation are much faster on "
                "smaller images."
            ),
        )
        csv_file = st.file_uploader("Extracted CSV", type=["csv"])
        csv_text = st.text_area(
            "...or paste CSV text",
            height=150,
            help="Ignored if a CSV file is uploaded above.",
        )

        if image_file is not None:
            _load_image_from_upload(image_file)

        # Resolve CSV source: uploaded file wins over pasted text.
        csv_source = None
        if csv_file is not None:
            try:
                csv_source = csv_file.getvalue().decode("utf-8", errors="replace")
            except Exception as e:
                st.error(f"Failed to read CSV file: {e}")
        elif csv_text and csv_text.strip():
            csv_source = csv_text

        if csv_source is not None:
            csv_hash = hashlib.md5(csv_source.encode("utf-8")).hexdigest()
            new_df = _load_csv(csv_source)
            if new_df is not None:
                if st.session_state.csv_hash != csv_hash:
                    # New CSV — clear any user-picked color overrides from the old file.
                    st.session_state.series_color_overrides = {}
                    # Drop per-series widget state from the previous CSV so a stale
                    # `vis_<old_name>=False` doesn't shadow a same-named series in
                    # the new file (and prevent _init_series_states from re-seeding
                    # to True). Same for the toggle/button keys.
                    new_series_names = set(new_df["series"].astype(str).tolist())
                    for k in list(st.session_state.keys()):
                        for prefix in ("vis_", "vis_btn_", "interp_btn_",
                                       "use_de_", "de_", "hmin_", "hmax_",
                                       "smin_", "smax_", "vmin_", "vmax_",
                                       "cpick_"):
                            if k.startswith(prefix):
                                suffix = k[len(prefix):]
                                if suffix not in new_series_names:
                                    st.session_state.pop(k, None)
                                break
                    st.session_state.series_states = _init_series_states(new_df)
                    st.session_state.csv_hash = csv_hash
                    # Clear the masked calibration image — series identities changed.
                    st.session_state.pop("cal_masked_img_path", None)
                    st.session_state.pop("cal_masked_img_bgr", None)
                    # New CSV → series identities can differ; the prior calibration
                    # may have been computed against a different y-baseline.
                    if st.session_state.calibration.get("applied"):
                        st.session_state.calibration = {"applied": False}
                        st.toast("Calibration cleared — re-apply after new data.", icon="🔄")
                else:
                    # Same CSV — re-apply any colors the user picked from the image.
                    for sname, color in st.session_state.get("series_color_overrides", {}).items():
                        new_df.loc[new_df["series"] == sname, "series_color"] = color
                st.session_state.df = new_df
                df = new_df

        # ----- Calibration status (read-only — controls are in the Calibrate tab) -----
        if "image_rgb" in st.session_state:
            if st.session_state.calibration.get("applied"):
                st.success("✓ Calibration applied")
            else:
                st.caption("✗ Not calibrated — use the **Calibrate** tab.")

        # ----- Series Controls -----
        if df is not None and st.session_state.series_states:
            st.header("Series Controls")
            for series_name in df["series"].drop_duplicates().tolist():
                state = st.session_state.series_states.get(series_name)
                if state is None:
                    continue
                color_hex = df[df["series"] == series_name]["series_color"].iloc[0]
                if not is_valid_hex(color_hex):
                    color_hex = FALLBACK_HEX

                with st.expander(f"■  {series_name}", expanded=False):
                    st.markdown(
                        "<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>"
                        f"<span style='display:inline-block;width:22px;height:22px;"
                        f"background:{color_hex};border:1px solid #555;border-radius:3px;'></span>"
                        f"<code>{color_hex}</code></div>",
                        unsafe_allow_html=True,
                    )
                    # Visibility toggle: read return value directly (True on the
                    # rerun triggered by the click) then st.rerun() to get a
                    # clean render pass where build_composite sees settled state.
                    is_visible = st.session_state.get(f"vis_{series_name}", True)
                    vis_label = "👁  Visible" if is_visible else "🚫  Hidden"
                    vis_type = "primary" if is_visible else "secondary"
                    if st.button(
                        vis_label,
                        key=f"vis_btn_{series_name}",
                        type=vis_type,
                        use_container_width=True,
                    ):
                        st.session_state[f"vis_{series_name}"] = not is_visible
                        _update_calibration_masked_image()
                        st.rerun()

                    state["use_delta_e"] = st.toggle(
                        "Use ΔE Lab masking (recommended)",
                        value=state.get("use_delta_e", True),
                        key=f"use_de_{series_name}",
                        help="ΔE masking is more reliable than HSV for plot colours.",
                    )
                    if state["use_delta_e"]:
                        state["delta_e"] = st.slider(
                            "ΔE threshold", 5, 80,
                            value=int(state.get("delta_e", 10)),
                            key=f"de_{series_name}",
                            help="Lower = tighter match. 10–20 works for most plots.",
                        )
                    else:
                        st.caption("HSV manual sliders:")
                        state["h_min"] = st.slider(
                            "Hue min", 0, 179, int(state["h_min"]),
                            key=f"hmin_{series_name}",
                        )
                        state["h_max"] = st.slider(
                            "Hue max", 0, 179, int(state["h_max"]),
                            key=f"hmax_{series_name}",
                        )
                        state["s_min"] = st.slider(
                            "Saturation min", 0, 255, int(state["s_min"]),
                            key=f"smin_{series_name}",
                        )
                        state["s_max"] = st.slider(
                            "Saturation max", 0, 255, int(state["s_max"]),
                            key=f"smax_{series_name}",
                        )
                        state["v_min"] = st.slider(
                            "Value min", 0, 255, int(state["v_min"]),
                            key=f"vmin_{series_name}",
                        )
                        state["v_max"] = st.slider(
                            "Value max", 0, 255, int(state["v_max"]),
                            key=f"vmax_{series_name}",
                        )

                    interp_on = bool(state.get("interpolate"))
                    btn_type = "primary" if interp_on else "secondary"
                    btn_label = "Interpolation: ON" if interp_on else "Interpolate occluded segments"
                    if st.button(
                        btn_label,
                        key=f"interp_btn_{series_name}",
                        type=btn_type,
                        use_container_width=True,
                    ):
                        state["interpolate"] = not interp_on
                        st.rerun()

                    if state.get("interpolate") and not st.session_state.calibration.get("applied"):
                        st.info("Interpolation requires axis calibration.")

        # ----- Debug status (helps diagnose state issues) -----
        if df is not None and st.session_state.series_states:
            st.divider()
            with st.expander("🔍 Debug: live state", expanded=False):
                rows = []
                for name in df["series"].drop_duplicates().tolist():
                    state = st.session_state.series_states.get(name, {})
                    rows.append({
                        "series": name,
                        "visible": st.session_state.get(f"vis_{name}", "<unset>"),
                        "mode": "ΔE" if state.get("use_delta_e") else "HSV",
                        "interpolate": bool(state.get("interpolate")),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

def _get_or_compute_frame_preview(img_bgr, image_hash):
    """Return a `FramePreview` for the current image, recomputing only on hash change.

    Phase A OCR is the most expensive step in the pipeline (several seconds with
    EasyOCR). The manual-band tab redraws on every slider change, so we cannot
    re-run OCR in the redraw loop. This helper memoises the FramePreview by
    (image_hash, min_ocr_confidence) so confidence-slider changes correctly
    invalidate; band-geometry sliders don't affect Phase A and stay cached.

    Returns ``(preview, was_cached)`` so callers can show a spinner only on the
    first compute. ``was_cached`` is True when this call was a cache hit.
    """
    min_conf = float(st.session_state.get("ocr_min_confidence", 0.20))
    cache_key = (image_hash, round(min_conf, 4))
    cached = st.session_state.get("frame_preview_cache")
    if cached and cached.get("key") == cache_key:
        return cached["preview"], True

    # Carry forward OCR-tuning settings so the preview matches
    # what `run_calibration` will see when invoked next.
    cfg = CalibrationConfig(
        use_gpu=False,
        min_ocr_confidence=min_conf,
    )
    preview = detect_axis_frame(img_bgr, config=cfg)
    st.session_state.frame_preview_cache = {"key": cache_key, "preview": preview}
    st.session_state.frame_preview_run_count = (
        int(st.session_state.get("frame_preview_run_count", 0)) + 1
    )
    return preview, False


def _band_overrides_key(image_hash):
    return f"band_overrides:{image_hash}"


def _slider_keys(image_hash):
    """Slider widget keys, scoped per image so reloading restores prior values.

    Each key embeds the image hash so a new image gets fresh sliders rather
    than inheriting the previous image's overrides. Streamlit retains the
    keys it has seen across reruns automatically.
    """
    return {
        "y_left": f"band_y_extra_left:{image_hash}",
        "y_vert": f"band_y_extra_vert:{image_hash}",
        "x_below": f"band_x_extra_below:{image_hash}",
        "x_horiz": f"band_x_extra_horiz:{image_hash}",
    }


def _last_calibration_has_degenerate_warning(image_hash):
    """Did the most recent calibration on THIS image produce a degenerate-fit warning?

    Used to surface a hint pointing at the y-band slider — the warning's most
    common cause is multi-digit y-labels bisected by a too-narrow band.
    """
    if st.session_state.get("auto_axis_image_hash") != image_hash:
        return False
    detection = st.session_state.get("auto_axis_detection") or {}
    for w in detection.get("warnings", []) or []:
        if "calibration refused" in w and "same value" in w:
            return True
    return False


def _render_detection_results(detection):
    """Inner contents of the 'Detection results' expander."""
    conf = float(detection.get("confidence", 0.0))
    st.metric("Detection confidence", f"{conf:.2f}",
              help=f"Mode: {detection.get('mode', 'unknown')}")
    for w in detection.get("warnings", []) or []:
        st.warning(w)

    pts = []
    for label in ["p1", "p2", "p3"]:
        p = detection.get(label)
        if not p:
            continue
        row = {"point": label.upper(), "pixel_x": float(p[0]), "pixel_y": float(p[1])}
        if label == "p1":
            row["data_x"] = detection.get("p1_data_x", st.session_state.get("p1_data_x", 0.0))
            row["data_y"] = detection.get("p1_data_y", st.session_state.get("p1_data_y", 0.0))
        elif label == "p2":
            row["data_x"] = detection.get("p2_data_x", st.session_state.get("p2_data_x", 1.0))
            row["data_y"] = detection.get("p1_data_y", st.session_state.get("p1_data_y", 0.0))
        else:
            row["data_x"] = detection.get("p3_data_x", 0.0)
            row["data_y"] = detection.get("p3_data_y", st.session_state.get("p3_data_y", 1.0))
        pts.append(row)
    if pts:
        st.markdown("**Detected calibration points**")
        st.dataframe(pd.DataFrame(pts), use_container_width=True, hide_index=True)

    x_cal = detection.get("x_calibration")
    y_cal = detection.get("y_calibration")
    x_grid = detection.get("x_grid_fit")
    y_grid = detection.get("y_grid_fit")
    if x_cal or y_cal or x_grid or y_grid:
        st.markdown("**Per-axis fit diagnostics**")
        diag_rows = []
        for axis_name, cal_d, grid_d in [("X", x_cal, x_grid), ("Y", y_cal, y_grid)]:
            if not cal_d:
                continue
            diag_rows.append({
                "axis": axis_name,
                "method": cal_d.get("method", ""),
                "n_points": cal_d.get("n_points", 0),
                "scale": cal_d.get("scale"),
                "offset": cal_d.get("offset"),
                "rmse_data": cal_d.get("rmse_data"),
                "rmse_px": cal_d.get("rmse_px"),
                "slope_SE": cal_d.get("slope_se"),
                "grid_spacing_px": grid_d.get("spacing") if grid_d else None,
                "grid_kept": len(grid_d.get("fitted_positions", [])) if grid_d else None,
                "grid_rejected": len(grid_d.get("rejected_positions", [])) if grid_d else None,
            })
        if diag_rows:
            st.dataframe(pd.DataFrame(diag_rows), use_container_width=True, hide_index=True)

    with st.expander("Raw detection diagnostics", expanded=False):
        st.json({k: v for k, v in detection.items() if k not in {"ocr_records"}})


def _render_ocr_tick_tables(detection):
    """Inner contents of the 'OCR tick tables' expander. Returns True if user re-calibrated."""
    st.caption(
        "`pixel_position` is the geometry-based tick position. "
        "Correct `value`, uncheck bad rows, then re-calibrate."
    )
    display_cols = ["include", "axis", "raw_text", "cleaned_text", "value", "pixel_position",
                    "ocr_confidence", "pair_distance_px", "parse_status", "status", "flag"]
    x_df = pd.DataFrame(detection.get("x_tick_table", []) or [])
    y_df = pd.DataFrame(detection.get("y_tick_table", []) or [])
    if len(x_df):
        x_df = x_df[[c for c in display_cols if c in x_df.columns]]
    if len(y_df):
        y_df = y_df[[c for c in display_cols if c in y_df.columns]]
    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown("**X tick labels**")
        edited_x = st.data_editor(
            x_df, key="ocr_x_tick_editor",
            use_container_width=True, hide_index=True,
            disabled=["axis", "raw_text", "cleaned_text", "ocr_confidence",
                      "pair_distance_px", "parse_status", "status", "flag"],
        )
    with right_col:
        st.markdown("**Y tick labels**")
        edited_y = st.data_editor(
            y_df, key="ocr_y_tick_editor",
            use_container_width=True, hide_index=True,
            disabled=["axis", "raw_text", "cleaned_text", "ocr_confidence",
                      "pair_distance_px", "parse_status", "status", "flag"],
        )
    if st.button("Re-calibrate from edits",
                 type="primary", use_container_width=True,
                 key="recalibrate_from_edits"):
        updated = update_detection_from_tick_tables(detection, edited_x, edited_y)
        st.session_state.auto_axis_detection = updated
        # Rebuild the typed CalibrationResult so the diagnostic overlay
        # (which renders from `auto_axis_result` when present) reflects the
        # post-edit anchors and paired ticks instead of the pre-edit ones.
        try:
            st.session_state.auto_axis_result = rebuild_result_from_detection(updated)
        except Exception:
            # If reconstruction fails for any reason, clear it so the overlay
            # falls back to rebuilding from the legacy dict.
            st.session_state.auto_axis_result = None
        _set_manual_fields_from_detection(updated)
        st.toast("Calibration updated from edited tick tables.", icon="✅")
        st.rerun()


def _render_manual_override(img_bgr):
    """Inner contents of the 'Manual override' expander."""
    h, w = img_bgr.shape[:2]
    st.caption(
        "Enter pixel and data coordinates for three calibration anchors. "
        "P1 + P2 anchor the X axis; P3 anchors the Y axis. "
        "Click **Load detected values** to pre-fill from the latest detection."
    )
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        st.markdown("**P1 — X-axis left**")
        st.number_input("Pixel X", key="p1_px_x", step=1.0,
                        min_value=0.0, max_value=float(w))
        st.number_input("Pixel Y", key="p1_px_y", step=1.0,
                        min_value=0.0, max_value=float(h))
        st.number_input("Data X",  key="p1_data_x")
        st.number_input("Data Y (baseline)", key="p1_data_y",
                        help="Data Y at the X-axis baseline — anchors the Y scale.")
    with pc2:
        st.markdown("**P2 — X-axis right**")
        st.number_input("Pixel X", key="p2_px_x", step=1.0,
                        min_value=0.0, max_value=float(w))
        st.number_input("Pixel Y", key="p2_px_y", step=1.0,
                        min_value=0.0, max_value=float(h))
        st.number_input("Data X",  key="p2_data_x")
    with pc3:
        st.markdown("**P3 — Y-axis top**")
        st.number_input("Pixel X", key="p3_px_x", step=1.0,
                        min_value=0.0, max_value=float(w))
        st.number_input("Pixel Y", key="p3_px_y", step=1.0,
                        min_value=0.0, max_value=float(h))
        st.number_input("Data Y", key="p3_data_y")

    # Live degeneracy hints — surfaced before the user clicks Apply.
    if abs(st.session_state.get("p1_px_x", 0.0) -
           st.session_state.get("p2_px_x", 0.0)) < 1e-9 and not _manual_fields_are_empty():
        st.warning("P1 and P2 share the same pixel X — calibration would fail.")
    if abs(st.session_state.get("p1_px_y", 0.0) -
           st.session_state.get("p3_px_y", 0.0)) < 1e-9 and not _manual_fields_are_empty():
        st.warning("P1 and P3 share the same pixel Y — Y-axis calibration would fail.")

    st.button(
        "Load detected values",
        on_click=_callback_copy_detected_values,
        disabled=not _auto_detection_available(),
        use_container_width=True,
        help="Pre-fill the fields above from the latest auto-detection.",
    )
    res = st.session_state.get("copy_detected_result")
    if res == "ok":
        st.success("Fields loaded from detection.")
        st.session_state.copy_detected_result = None
    elif res and res.startswith("error:"):
        st.error(res[len("error:"):])
        st.session_state.copy_detected_result = None


def _render_calibration_tab(img_bgr):
    """Calibrate tab: one button at the top to run detection, one at the bottom to apply.

    Everything in between is grouped into collapsible expanders so the workflow
    stays visible without scrolling through a wall of controls.
    """
    image_hash = st.session_state.get("image_hash")
    if image_hash is None:
        st.info("Upload an image to begin.")
        return

    # Use the series-masked image for display and calibration if one has been built.
    cal_img_bgr = st.session_state.get("cal_masked_img_bgr")
    if cal_img_bgr is None:
        cal_img_bgr = img_bgr

    detection_available = _auto_detection_available()
    ocr_is_available = ocr_available()

    # When EasyOCR is missing, force OCR off and inform the user that the
    # manual calibration path is the supported workflow. The pipeline still
    # runs in geometry-only mode for `Run Detection`, but accuracy is reduced.
    if not ocr_is_available:
        st.session_state["use_ocr_axis"] = False
        st.info(
            "EasyOCR is not installed — auto-calibration of tick labels is "
            "disabled. Use **Manual override** to set P1/P2/P3 by pixel and "
            "data values, then click **Apply Calibration**. "
            "(Install EasyOCR + pytorch to enable auto-calibration.)"
        )

    # ── 1. Detection settings (collapsed by default) ──────────────────────────
    with st.expander("Detection settings", expanded=not detection_available):
        c1, c2 = st.columns(2)
        with c1:
            st.toggle(
                "OCR-assisted detection",
                value=ocr_is_available,
                key="use_ocr_axis",
                disabled=not ocr_is_available,
                help=("EasyOCR is required for OCR-assisted detection."
                      if not ocr_is_available else None),
            )
            st.toggle("Mask all detected text", value=False, key="ocr_mask_all_text")
        with c2:
            st.toggle("Show OCR debug overlay", value=True, key="show_ocr_debug_overlay")
        st.slider("Min OCR confidence", 0.0, 1.0,
                  key="ocr_min_confidence", step=0.05,
                  disabled=not ocr_is_available)

    # ── 2. Primary action: Run Detection ──────────────────────────────────────
    if st.session_state.get("df") is None:
        st.caption("Upload a CSV in the sidebar to view the calibrated overlay after detection.")

    if cal_img_bgr is not img_bgr:
        n_hidden = sum(
            1 for k, v in st.session_state.items()
            if k.startswith("vis_") and not v
        )
        st.info(f"Using series-masked image ({n_hidden} series hidden). "
                "Toggle visibility in the sidebar to update.")

    run_clicked = st.button(
        "Run Detection",
        key="run_detection_primary",
        type="primary",
        use_container_width=True,
        help=(
            "Runs the full pipeline: EasyOCR text discovery, frame detection, "
            "tick parsing, grid-fit, and per-axis calibration regression."
        ),
    )
    if detection_available:
        det = st.session_state.auto_axis_detection
        st.caption(
            f"Last run: **{det.get('mode', 'unknown')}** — "
            f"confidence {float(det.get('confidence', 0.0)):.2f}"
        )

    # ── 3. Frame preview + band controls (always visible) ─────────────────────
    try:
        with st.spinner("Detecting plot frame..."):
            preview, _was_cached = _get_or_compute_frame_preview(img_bgr, image_hash)
    except Exception as e:
        st.error(f"Frame detection failed: {type(e).__name__}: {e}.")
        return

    if preview.bbox is None:
        st.error("Could not detect a plot frame in this image.")
        for w in preview.warnings:
            st.warning(w)
        st.info("Without a detected frame, label bands cannot be positioned.")
        return

    default_cfg = CalibrationConfig()
    overrides_key = _band_overrides_key(image_hash)
    saved = st.session_state.get(overrides_key, {})
    init_y_left  = int(saved.get("y_band_extra_px",            default_cfg.y_band_extra_px))
    init_y_vert  = int(saved.get("y_band_extra_vertical_px",   default_cfg.y_band_extra_vertical_px))
    init_x_below = int(saved.get("x_band_extra_px",            default_cfg.x_band_extra_px))
    init_x_horiz = int(saved.get("x_band_extra_horizontal_px", default_cfg.x_band_extra_horizontal_px))
    keys = _slider_keys(image_hash)
    img_h, img_w = cal_img_bgr.shape[:2]

    img_col, ctrl_col = st.columns([2, 1])

    with ctrl_col:
        st.markdown("**Y-label band**")
        y_extra_left = st.slider(
            "Extra left (px)", min_value=20, max_value=200, value=init_y_left, step=1,
            help=(
                "How far left of the y-axis the band extends. "
                "Increase for 4-digit labels like '2500' or negatives like '-1000'."
            ),
            key=keys["y_left"],
        )
        y_extra_vert = st.slider(
            "Trim top/bottom (px)", min_value=-(img_h // 10), max_value=img_h // 2, value=init_y_vert, step=1,
            help="Trims the y-band inward from the top and bottom. Negative values expand beyond the detected axis.",
            key=keys["y_vert"],
        )
        st.markdown("**X-label band**")
        x_extra_below = st.slider(
            "Extra below (px)", min_value=10, max_value=100, value=init_x_below, step=1,
            help=(
                "How far below the x-axis the band extends. "
                "Keep narrow to avoid capturing the axis title."
            ),
            key=keys["x_below"],
        )
        x_extra_horiz = st.slider(
            "Trim left/right (px)", min_value=-(img_w // 10), max_value=img_w // 2, value=init_x_horiz, step=1,
            help="Trims the x-band inward from the left and right. Negative values expand beyond the detected axis.",
            key=keys["x_horiz"],
        )
        if st.button("↺ Reset bands to defaults",
                     use_container_width=True,
                     key=f"band_reset:{image_hash}"):
            st.session_state.pop(overrides_key, None)
            for k in keys.values():
                st.session_state.pop(k, None)
            st.rerun()
        with st.expander("Preview diagnostics", expanded=False):
            st.write({
                "frame_preview_run_count": st.session_state.get("frame_preview_run_count", 0),
                "axis_confidence": float(preview.axis_confidence),
                "phase_a_records": len(preview.phase_a_records),
                "phase_a_numeric": sum(1 for r in preview.phase_a_records if r.is_numeric),
            })

    if detection_available:
        cal_result = st.session_state.get("auto_axis_result")
        # A failed re-run can leave cal_result with bbox=None; fall back to the
        # frame preview's bbox in that case so the band-overlay helpers don't
        # receive None.
        band_bbox = (
            cal_result.bbox
            if cal_result is not None and cal_result.bbox is not None
            else preview.bbox
        )
    else:
        cal_result = None
        band_bbox = preview.bbox

    if band_bbox is None:
        st.error(
            "Could not detect a plot frame. Adjust the Detection settings or "
            "use **Manual override** to set P1/P2/P3 directly."
        )
        return

    y_band = _y_label_band(band_bbox,
                           extra_left=int(y_extra_left),
                           extra_vertical=int(y_extra_vert))
    x_band = _x_label_band(band_bbox,
                           extra_below=int(x_extra_below),
                           extra_horizontal=int(x_extra_horiz))

    if detection_available:
        show_debug = st.session_state.get("show_ocr_debug_overlay", True)
        ocr_ran = cal_result is not None or (
            st.session_state.get("auto_axis_detection") or {}
        ).get("ocr_enabled", False)
        if cal_result is not None:
            diag_rgb = render_overlay(
                cal_img_bgr, cal_result,
                show_band_windows=False,
                show_grid_rejected=show_debug,
                show_frame=False,
            )
        else:
            det_img = st.session_state.auto_axis_detection
            if show_debug and det_img.get("ocr_enabled"):
                diag_rgb = build_ocr_debug_overlay(
                    cal_img_bgr, det_img, show_mask=True, show_pairing=True,
                )
            else:
                diag_rgb = build_diagnostic_overlay(cal_img_bgr, det_img)
        base_bgr = cv2.cvtColor(diag_rgb, cv2.COLOR_RGB2BGR)
        img_caption = "Calibration diagnostic with label bands overlaid."
    else:
        ocr_ran = False
        base_bgr = cal_img_bgr
        img_caption = (
            f"Band preview — bbox ({preview.bbox.left}, {preview.bbox.top})→"
            f"({preview.bbox.right}, {preview.bbox.bottom}). "
            "Click Run Detection above to see the full diagnostic."
        )

    preview_img = render_band_preview(
        base_bgr, band_bbox, y_band, x_band,
        phase_a_records=None if detection_available else preview.phase_a_records,
        show_frame=not ocr_ran,
    )

    with img_col:
        st.image(preview_img, caption=img_caption, use_container_width=True)

    # Persist slider values per-image so subsequent renders + Run Detection use them.
    st.session_state[overrides_key] = {
        "y_band_extra_px":            int(y_extra_left),
        "y_band_extra_vertical_px":   int(y_extra_vert),
        "x_band_extra_px":            int(x_extra_below),
        "x_band_extra_horizontal_px": int(x_extra_horiz),
    }

    if preview.warnings:
        with st.expander(f"Frame-detection warnings ({len(preview.warnings)})",
                         expanded=False):
            for w in preview.warnings:
                st.warning(w)

    if _last_calibration_has_degenerate_warning(image_hash):
        st.warning(
            "⚠️ The last calibration was refused because all paired y-labels read as the "
            "same value — the y-band crop likely bisected multi-digit labels. Increase "
            "**Y-label band → Extra left** until the band covers the longest label, "
            "then click **Run Detection** again."
        )

    # ── 4. Run Detection handler (uses current band slider values) ────────────
    if run_clicked:
        with st.spinner("Running calibration pipeline..."):
            override_cfg = CalibrationConfig(
                y_band_extra_px=int(y_extra_left),
                y_band_extra_vertical_px=int(y_extra_vert),
                x_band_extra_px=int(x_extra_below),
                x_band_extra_horizontal_px=int(x_extra_horiz),
                use_gpu=False,
                min_ocr_confidence=float(st.session_state.get("ocr_min_confidence", 0.20)),
            )
            new_detection_applied = False
            try:
                if not bool(st.session_state.get("use_ocr_axis", True)):
                    detection = auto_detect_axes_and_ticks(cal_img_bgr)
                    detection["ocr_enabled"] = False
                    # Only replace prior state when this run actually found
                    # a plot frame; otherwise keep the previous (working)
                    # calibration so the user can adjust settings and retry.
                    if detection.get("bbox"):
                        st.session_state.auto_axis_detection = detection
                        st.session_state.auto_axis_result = None
                        new_detection_applied = True
                    else:
                        st.warning(
                            "Detection found no plot frame — keeping previous "
                            "calibration. Adjust Detection settings and try again."
                        )
                        for w in detection.get("warnings", []) or []:
                            st.warning(w)
                else:
                    result = run_calibration(cal_img_bgr, config=override_cfg)
                    if result.bbox is None:
                        st.warning(
                            "Detection found no plot frame — keeping previous "
                            "calibration. Adjust Detection settings and try again."
                        )
                        for w in result.warnings:
                            st.warning(w)
                    else:
                        st.session_state.auto_axis_detection = result.to_legacy_dict()
                        st.session_state.auto_axis_result = result
                        new_detection_applied = True
                        if result.success:
                            st.toast(
                                f"Detection succeeded — confidence {result.confidence:.2f}",
                                icon="✅",
                            )
                        for w in result.warnings:
                            st.warning(w)
            except Exception as e:
                st.error(f"Detection failed: {type(e).__name__}: {e}")
                return
            st.session_state.auto_axis_image_hash = image_hash
            if new_detection_applied:
                # Drop the previous run's manual-field values so the next
                # _callback_apply_calibration repopulates them from the fresh
                # detection (taking the `used_detection=True` branch). Without
                # this, the second click of Run Detection would compute a
                # stale `cal` from the prior anchors AND synthesize a manual
                # CalibrationResult that overwrites the new one with bbox=None,
                # making P1/P2/P3 vanish from the overlay.
                for key in ("p1_px_x", "p1_px_y", "p1_data_x", "p1_data_y",
                            "p2_px_x", "p2_px_y", "p2_data_x",
                            "p3_px_x", "p3_px_y", "p3_data_y"):
                    st.session_state.pop(key, None)
            _callback_apply_calibration()
            st.rerun()

    # ── 5. Detection results expander (auto-expanded after first run) ─────────
    detection = st.session_state.get("auto_axis_detection") if _auto_detection_available() else None
    if detection is not None:
        with st.expander("Detection results", expanded=True):
            _render_detection_results(detection)

        if detection.get("ocr_enabled"):
            with st.expander("OCR tick tables", expanded=False):
                _render_ocr_tick_tables(detection)

    # ── 6. Manual override expander (collapsed by default) ────────────────────
    with st.expander("Manual override", expanded=False):
        _render_manual_override(img_bgr)

    # ── 7. Primary action: Apply Calibration ──────────────────────────────────
    st.button(
        "Apply Calibration",
        key="apply_calibration_primary",
        on_click=_callback_apply_calibration,
        type="primary",
        use_container_width=True,
        help=(
            "Applies the manual override values if set, otherwise applies the "
            "latest detection result. Enables the Overlay and Compare tabs."
        ),
    )

    res = st.session_state.get("apply_calibration_result")
    if res == "ok":
        st.success("Calibration applied.")
        st.session_state.apply_calibration_result = None
    elif res and res.startswith("error:"):
        st.error(res[len("error:"):])
        st.session_state.apply_calibration_result = None

    if st.session_state.calibration.get("applied"):
        cal_s = st.session_state.calibration
        x_log = cal_s.get("x_log_base")
        y_log = cal_s.get("y_log_base")
        log_tags = []
        if x_log:
            log_tags.append("X log10")
        if y_log:
            log_tags.append("Y log10")
        log_suffix = (" — " + ", ".join(log_tags)) if log_tags else ""
        x_unit = "log10(data)/px" if x_log else "data/px"
        y_unit = "log10(data)/px" if y_log else "data/px"
        st.caption(
            f"✓ Active calibration{log_suffix} — "
            f"X: {cal_s.get('x_scale', 0):.5f} {x_unit}  |  "
            f"Y: {cal_s.get('y_scale', 0):.5f} {y_unit}"
        )
        # Surface P1/P2 baseline mismatch — the calibration math averages the
        # two pixel-y values without warning, so a user-induced rotation between
        # the two anchors is silently absorbed.
        disagree = float(cal_s.get("p1p2_y_disagreement_px", 0.0) or 0.0)
        if disagree > P1P2_Y_TOLERANCE_PX:
            st.warning(
                f"P1 and P2 differ in pixel-Y by {disagree:.1f}px (tolerance "
                f"{P1P2_Y_TOLERANCE_PX:.0f}px). The x-axis baseline was set to "
                "the midpoint. Drag P2 to match P1's pixel-Y for a consistent "
                "x-axis baseline."
            )
    else:
        st.caption("✗ Not yet calibrated — apply to enable the Overlay and Compare tabs.")


def _render_mask_view(img_rgb, img_bgr, img_hash, df, series_states, cal,
                      container_height=600, key_suffix=""):
    """Render Tab 1's mask composite (or plain image if no CSV loaded)."""
    if df is None or not series_states:
        display = img_rgb
    else:
        display = build_composite(img_bgr, img_hash, series_states, df, cal)

    fig = px.imshow(display)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=container_height,
    )
    fig.update_xaxes(title="pixel X", showticklabels=True)
    fig.update_yaxes(title="pixel Y", showticklabels=True)
    st.plotly_chart(fig, use_container_width=True, key=f"mask_chart{key_suffix}")


def _render_overlay_view(img_rgb, df, series_states, cal,
                         container_height=600, key_suffix=""):
    """Render Tab 2's data overlay (or a calibration prompt)."""
    if df is None:
        st.info("Upload a CSV to see the data overlay.")
        return
    if not cal.get("applied"):
        st.warning("Apply axis calibration in the **Calibrate** tab to enable the overlay.")
        fig = px.imshow(img_rgb, binary_string=True)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=container_height)
        st.plotly_chart(
            fig, use_container_width=True,
            key=f"overlay_uncal{key_suffix}",
        )
        return

    fig = build_overlay_figure(img_rgb, df, series_states, cal)
    fig.update_layout(height=container_height)
    st.plotly_chart(
        fig, use_container_width=True,
        key=f"overlay_chart{key_suffix}",
    )


def _render_mask_color_picker(img_rgb, img_bgr, img_hash, df, series_states, cal):
    """Color assignment UI for the Masks tab when series_color was absent from the CSV.

    Shows a table of series with their current colors (initially NA) alongside the
    mask composite. Clicking 'Pick' then clicking the image samples the pixel color
    and assigns it to that series.
    """
    series_list = df["series"].drop_duplicates().tolist()
    picking_for = st.session_state.get("picking_color_for")

    table_col, img_col = st.columns([1, 2])

    with table_col:
        st.markdown("**Series Colors**")
        st.caption("Click **Pick**, then click anywhere on the image to sample a color.")
        for s in series_list:
            color = df[df["series"] == s]["series_color"].iloc[0]
            valid = is_valid_hex(color)
            is_picking = (picking_for == s)

            c1, c2 = st.columns([3, 2])
            with c1:
                if valid:
                    st.markdown(
                        f"<div style='display:flex;align-items:center;gap:6px;padding-top:6px'>"
                        f"<span style='width:14px;height:14px;background:{color};"
                        f"border:1px solid #555;display:inline-block;border-radius:2px'></span>"
                        f"<span>{s}</span></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div style='padding-top:6px'>{s} "
                        f"<em style='color:#888'>— no color</em></div>",
                        unsafe_allow_html=True,
                    )
            with c2:
                btn_label = "🎯 Picking..." if is_picking else ("Repick" if valid else "Pick")
                btn_type = "primary" if is_picking else "secondary"
                if st.button(btn_label, key=f"cpick_{s}",
                             type=btn_type, use_container_width=True):
                    # Toggle: click again to cancel.
                    st.session_state["picking_color_for"] = None if is_picking else s
                    st.rerun()

    with img_col:
        if picking_for:
            st.info(
                f"Click the image to set the color for **{picking_for}**. "
                "Click '🎯 Picking...' to cancel."
            )

        # In pick mode show the original so the user sees real colours to click on.
        # Otherwise show the composite so hide/show visibility controls are effective.
        if picking_for:
            display = img_rgb
        elif df is not None and series_states:
            display = build_composite(img_bgr, img_hash, series_states, df, cal)
        else:
            display = img_rgb

        fig = px.imshow(display, binary_string=True)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=580)
        fig.update_xaxes(title="pixel X", showticklabels=True)
        fig.update_yaxes(title="pixel Y", showticklabels=True)

        if picking_for:
            img_h, img_w = display.shape[:2]
            # px.imshow produces an Image trace which doesn't fire selection
            # events on click. Overlay a near-invisible scatter grid so that
            # clicking anywhere on the image selects the nearest grid point
            # and populates event.selection.points with (x, y) pixel coords.
            step_x = max(4, img_w // 100)
            step_y = max(4, img_h // 100)
            grid_xs = [x for x in range(0, img_w, step_x)
                       for _ in range(0, img_h, step_y)]
            grid_ys = [y for _ in range(0, img_w, step_x)
                       for y in range(0, img_h, step_y)]
            grid_colors = [
                "#{:02x}{:02x}{:02x}".format(int(img_bgr[gy, gx][2]),
                                             int(img_bgr[gy, gx][1]),
                                             int(img_bgr[gy, gx][0]))
                for gx, gy in zip(grid_xs, grid_ys)
            ]
            fig.add_trace(go.Scatter(
                x=grid_xs, y=grid_ys,
                mode="markers",
                marker=dict(size=max(step_x, step_y) + 4, opacity=0.01,
                            color="rgba(255,255,255,0.01)"),
                customdata=grid_colors,
                hovertemplate="x=%{x}, y=%{y}<br>%{customdata}<extra></extra>",
                showlegend=False,
            ))
            # clickmode="event+select" makes a single click on a scatter point
            # fire plotly_selected, which Streamlit captures via on_select.
            # dragmode is left at default (zoom) so the user can still pan/zoom.
            fig.update_layout(clickmode="event+select")

            event = st.plotly_chart(
                fig, use_container_width=True,
                key="mask_chart_color_pick",
                on_select="rerun",
                selection_mode="points",
            )
            if event.selection and event.selection.points:
                pt = event.selection.points[0]
                px_x = max(0, min(int(round(float(pt.get("x", 0)))), img_w - 1))
                px_y = max(0, min(int(round(float(pt.get("y", 0)))), img_h - 1))
                pixel = img_bgr[px_y, px_x]
                b_val, g_val, r_val = int(pixel[0]), int(pixel[1]), int(pixel[2])
                hex_color = "#{:02x}{:02x}{:02x}".format(r_val, g_val, b_val)
                # Persist override so it survives CSV reloads on rerun.
                overrides = st.session_state.get("series_color_overrides", {})
                overrides[picking_for] = hex_color
                st.session_state["series_color_overrides"] = overrides
                # Update HSV sliders in series_states immediately.
                if picking_for in series_states:
                    try:
                        hue, sat, val = hex_to_hsv_opencv(hex_color)
                    except Exception:
                        hue, sat, val = 0, 128, 128
                    series_states[picking_for].update({
                        "h_min": max(0, hue - 15),
                        "h_max": min(179, hue + 15),
                        "s_min": max(0, sat - 60),
                        "s_max": min(255, sat + 60),
                        "v_min": max(0, val - 60),
                        "v_max": min(255, val + 60),
                    })
                st.session_state["picking_color_for"] = None
                st.rerun()
        else:
            st.plotly_chart(fig, use_container_width=True, key="mask_chart_color_nopick")


def render_main():
    if "image_rgb" not in st.session_state:
        st.info("Upload an image and CSV in the sidebar to begin.")
        return

    img_rgb = st.session_state.image_rgb
    img_bgr = st.session_state.image_bgr
    img_hash = st.session_state.image_hash
    df = st.session_state.get("df")
    series_states = st.session_state.series_states
    cal = st.session_state.calibration

    mask_tab, calib_tab, overlay_tab, compare_tab = st.tabs([
        "Masks",
        "Calibrate",
        "Overlay",
        "Compare",
    ])

    with mask_tab:
        if not st.session_state.get("csv_has_series_color", True) and df is not None:
            _render_mask_color_picker(img_rgb, img_bgr, img_hash, df, series_states, cal)
        else:
            _render_mask_view(img_rgb, img_bgr, img_hash, df, series_states, cal,
                              key_suffix="_masks")

    with calib_tab:
        _render_calibration_tab(img_bgr)

    cal_masked = st.session_state.get("cal_masked_img_bgr")
    overlay_bg_rgb = (
        cv2.cvtColor(cal_masked, cv2.COLOR_BGR2RGB) if cal_masked is not None else img_rgb
    )

    with overlay_tab:
        _render_overlay_view(overlay_bg_rgb, df, series_states, cal,
                             key_suffix="_overlay")

    with compare_tab:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Color Masks**")
            _render_mask_view(img_rgb, img_bgr, img_hash, df, series_states, cal,
                              container_height=500, key_suffix="_cmp_left")
        with col2:
            st.markdown("**Extracted Overlay**")
            _render_overlay_view(overlay_bg_rgb, df, series_states, cal,
                                 container_height=500, key_suffix="_cmp_right")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(layout="wide", page_title="PlotVerify")
    st.title("PlotVerify")
    st.caption("Visually verify AI-extracted plot data against the original image.")
    _init_state()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
