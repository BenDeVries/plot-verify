"""PlotVerify — Scientific plot verification Streamlit app.

Verifies AI-extracted plot data by overlaying it on the original image and
masking each series by its known color in the source figure. Optional monotone
cubic spline interpolation reconstructs occluded line segments.

Run with:
    streamlit run app.py
"""

import colorsys
import hashlib
import io

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from scipy.interpolate import PchipInterpolator

from axis_auto import auto_detect_axes_and_ticks, build_diagnostic_overlay
from ocr_axis import auto_detect_axes_ticks_ocr, build_ocr_debug_overlay, update_detection_from_tick_tables

# Direct pipeline access for the manual-band preview tab. The legacy shims
# above wrap the same package; using it directly here gives us the typed
# `FramePreview` / `CalibrationResult` objects without going through dicts.
from axis_pipeline import (
    CalibrationConfig,
    detect_axis_frame,
    render_band_preview,
    render_overlay,
    run_calibration,
    x_label_band as _x_label_band,
    y_label_band as _y_label_band,
)


REQUIRED_COLUMNS = ["series", "x", "y", "y_err_lower", "y_err_upper", "series_color"]
FALLBACK_HEX = "#888888"


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def is_valid_hex(s):
    """Return True if ``s`` is a 6-digit hex color string (with or without #)."""
    if not isinstance(s, str):
        return False
    h = s.lstrip("#")
    if len(h) != 6:
        return False
    try:
        int(h, 16)
    except ValueError:
        return False
    return True


def hex_to_hsv_opencv(hex_color):
    """Convert hex color string to OpenCV HSV (H: 0-179, S: 0-255, V: 0-255)."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color!r}")
    r, g, b = [int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return int(round(h * 179)), int(round(s * 255)), int(round(v * 255))


def hex_to_bgr(hex_color):
    """Convert hex color string to a BGR int tuple (for OpenCV drawing)."""
    if not is_valid_hex(hex_color):
        return (136, 136, 136)
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b, g, r)


def _delta_e_mask(img_bgr, hex_color, threshold=30.0):
    """Return a binary mask of pixels within ``threshold`` ΔE76 of ``hex_color``.

    Works in CIE Lab — perceptually uniform, so equal distance = equal perceived
    colour difference.  Much more reliable than a 3-axis HSV box, especially for
    colours that are close to grey or near the hue wrap-around.
    """
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)

    h = hex_color.lstrip("#")
    rgb = np.uint8([[[int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)]]])
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    target_lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32)[0, 0]

    delta_e = np.sqrt(((img_lab - target_lab) ** 2).sum(axis=2))
    mask = (delta_e < threshold).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(mask, kernel, iterations=2)


# ---------------------------------------------------------------------------
# Mask + image ops
# ---------------------------------------------------------------------------

def apply_color_mask(img_bgr, h_min, h_max, s_min, s_max, v_min, v_max):
    """Build a binary mask isolating pixels whose HSV falls within the given range.

    Handles hue wraparound: when h_min > h_max the range is interpreted as
    [h_min, 179] ∪ [0, h_max], which is needed for reds that span 0/179.
    Dilates the result with a 3x3 kernel for 2 iterations to fill the
    anti-aliasing fringe around plotted lines.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    if h_min <= h_max:
        mask = cv2.inRange(
            hsv,
            np.array([h_min, s_min, v_min], dtype=np.uint8),
            np.array([h_max, s_max, v_max], dtype=np.uint8),
        )
    else:
        mask1 = cv2.inRange(
            hsv,
            np.array([h_min, s_min, v_min], dtype=np.uint8),
            np.array([179, s_max, v_max], dtype=np.uint8),
        )
        mask2 = cv2.inRange(
            hsv,
            np.array([0, s_min, v_min], dtype=np.uint8),
            np.array([h_max, s_max, v_max], dtype=np.uint8),
        )
        mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


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
# Calibration
# ---------------------------------------------------------------------------

def compute_calibration(p1_px_x, p1_px_y, p2_px_x, p2_px_y,
                        p3_px_x, p3_px_y,
                        p1_data_x, p2_data_x, p3_data_y, y_baseline):
    """Compute the affine pixel↔data transform from the three calibration points.

    Returns a dict with keys ``x_scale``, ``x_offset``, ``y_scale``,
    ``y_offset``, ``applied`` — or ``None`` if the inputs are degenerate.
    """
    if abs(p2_px_x - p1_px_x) < 1e-9:
        return None

    x_scale = (p2_data_x - p1_data_x) / (p2_px_x - p1_px_x)
    x_offset = p1_data_x - x_scale * p1_px_x

    x_axis_pixel_y = (p1_px_y + p2_px_y) / 2.0
    if abs(p3_px_y - x_axis_pixel_y) < 1e-9:
        return None

    y_scale = (p3_data_y - y_baseline) / (p3_px_y - x_axis_pixel_y)
    y_offset = p3_data_y - y_scale * p3_px_y

    if not (np.isfinite(x_scale) and np.isfinite(y_scale)):
        return None
    if abs(x_scale) < 1e-12 or abs(y_scale) < 1e-12:
        return None

    return {
        "x_scale": float(x_scale),
        "x_offset": float(x_offset),
        "y_scale": float(y_scale),
        "y_offset": float(y_offset),
        "applied": True,
    }


def px_to_data(px_x, px_y, cal):
    """Convert pixel coordinates to data coordinates."""
    return (cal["x_scale"] * px_x + cal["x_offset"],
            cal["y_scale"] * px_y + cal["y_offset"])


def data_to_px(data_x, data_y, cal):
    """Convert data coordinates to pixel coordinates."""
    return ((data_x - cal["x_offset"]) / cal["x_scale"],
            (data_y - cal["y_offset"]) / cal["y_scale"])


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
            color_hex = rows["series_color"].iloc[0]
    if state.get("use_delta_e", True) and df is not None:
        return cached_delta_e_mask(
            image_hash, color_hex, int(state.get("delta_e", 30)), img_bgr
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


# ---------------------------------------------------------------------------
# Plotly overlay (Tab 2)
# ---------------------------------------------------------------------------

def build_overlay_figure(img_rgb, df, series_states, cal):
    """Return a Plotly figure: the calibrated image as a background with
    extracted-data scatter traces (with error bars) drawn on top in data coords.
    """
    h, w = img_rgb.shape[:2]

    x_left = px_to_data(0, 0, cal)[0]
    x_right = px_to_data(w, 0, cal)[0]
    y_top = px_to_data(0, 0, cal)[1]
    y_bottom = px_to_data(0, h, cal)[1]

    fig = go.Figure()

    pil_img = Image.fromarray(img_rgb)
    fig.add_layout_image(
        dict(
            source=pil_img,
            xref="x",
            yref="y",
            x=min(x_left, x_right),
            y=max(y_top, y_bottom),
            sizex=abs(x_right - x_left),
            sizey=abs(y_top - y_bottom),
            xanchor="left",
            yanchor="top",
            sizing="stretch",
            opacity=1.0,
            layer="below",
        )
    )

    for series_name in df["series"].drop_duplicates().tolist():
        sdf = df[df["series"] == series_name]
        color_hex = sdf["series_color"].iloc[0]
        if not is_valid_hex(color_hex):
            color_hex = FALLBACK_HEX

        y_vals = sdf["y"].to_numpy(dtype=float)
        eu = sdf["y_err_upper"].to_numpy(dtype=float)
        el = sdf["y_err_lower"].to_numpy(dtype=float)
        has_err = np.isfinite(eu) & np.isfinite(el)

        if has_err.any():
            arr_plus = np.where(has_err, eu - y_vals, 0.0)
            arr_minus = np.where(has_err, y_vals - el, 0.0)
            error_y = dict(
                type="data", symmetric=False,
                array=arr_plus, arrayminus=arr_minus,
                color=color_hex, thickness=1.5, width=4,
            )
        else:
            error_y = None

        state = series_states.get(series_name, {})
        visible = True if st.session_state.get(f"vis_{series_name}", True) else "legendonly"

        fig.add_trace(go.Scatter(
            x=sdf["x"],
            y=sdf["y"],
            mode="lines+markers",
            line=dict(color=color_hex, width=2),
            marker=dict(color=color_hex, size=8,
                        line=dict(color="rgba(0,0,0,0.5)", width=0.5)),
            name=str(series_name),
            error_y=error_y,
            visible=visible,
        ))

    fig.update_xaxes(range=sorted([x_left, x_right]), title="X")
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
    st.session_state.setdefault("ocr_mask_all_text", True)
    st.session_state.setdefault("ocr_min_confidence", 0.20)
    st.session_state.setdefault("show_ocr_debug_overlay", True)
    st.session_state.setdefault("use_robust_regression", True)
    # Calibration-preview tab state.
    # Cache key: image_hash → FramePreview, so slider changes don't re-run
    # Phase A OCR. Per-image overrides are stored separately under
    # `band_overrides:{image_hash}` and restored when the same image reloads.
    st.session_state.setdefault("frame_preview_cache", None)
    st.session_state.setdefault("frame_preview_run_count", 0)
    st.session_state.setdefault("copy_detected_result", None)
    st.session_state.setdefault("apply_calibration_result", None)



def _run_auto_axis_detection():
    """Run button-triggered automatic axis/tick detection for the current image."""
    if "image_bgr" not in st.session_state:
        st.warning("Upload an image before running auto axis detection.")
        return

    image_hash = st.session_state.get("image_hash")
    overrides = st.session_state.get(_band_overrides_key(image_hash), {})
    default_cfg = CalibrationConfig()

    cal_result = None
    if not bool(st.session_state.get("use_ocr_axis", True)):
        detection = auto_detect_axes_and_ticks(st.session_state.image_bgr)
        detection["ocr_enabled"] = False
    else:
        cfg = CalibrationConfig(
            use_gpu=False,
            min_ocr_confidence=float(st.session_state.get("ocr_min_confidence", 0.20)),
            use_robust_regression=bool(st.session_state.get("use_robust_regression", True)),
            y_band_extra_px=int(overrides.get("y_band_extra_px", default_cfg.y_band_extra_px)),
            y_band_extra_vertical_px=int(overrides.get("y_band_extra_vertical_px", default_cfg.y_band_extra_vertical_px)),
            x_band_extra_px=int(overrides.get("x_band_extra_px", default_cfg.x_band_extra_px)),
            x_band_extra_horizontal_px=int(overrides.get("x_band_extra_horizontal_px", default_cfg.x_band_extra_horizontal_px)),
        )
        try:
            cal_result = run_calibration(st.session_state.image_bgr, config=cfg)
            detection = cal_result.to_legacy_dict()
        except Exception as e:
            detection = auto_detect_axes_and_ticks(st.session_state.image_bgr)
            detection.setdefault("warnings", []).append(
                f"OCR pipeline failed; fell back to geometry-only. {type(e).__name__}: {e}"
            )
            detection["ocr_enabled"] = False

    st.session_state.auto_axis_detection = detection
    st.session_state.auto_axis_result = cal_result
    st.session_state.auto_axis_image_hash = image_hash



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



def _callback_apply_detected_calibration():
    """on_click callback for 'Apply detected calibration now'.

    Runs BEFORE the next render pass so widget-keyed session state can be
    written without triggering StreamlitAPIException. Stores a result string
    in session_state["apply_calibration_result"] for the render pass to display.
    """
    if not _auto_detection_available():
        st.session_state.apply_calibration_result = "error:No auto-detection available for the current image."
        return
    detection = st.session_state.auto_axis_detection
    ok = _set_manual_fields_from_detection(detection)
    if not ok:
        st.session_state.apply_calibration_result = "error:Could not extract all calibration points from detection. Review detection output."
        return
    cal = compute_calibration(
        st.session_state.get("p1_px_x", 0.0), st.session_state.get("p1_px_y", 0.0),
        st.session_state.get("p2_px_x", 0.0), st.session_state.get("p2_px_y", 0.0),
        st.session_state.get("p3_px_x", 0.0), st.session_state.get("p3_px_y", 0.0),
        st.session_state.get("p1_data_x", 0.0),
        st.session_state.get("p2_data_x", 1.0),
        st.session_state.get("p3_data_y", 1.0),
        st.session_state.get("p1_data_y", 0.0),
    )
    if cal is None:
        st.session_state.apply_calibration_result = "error:Calibration math failed — check that P1 and P2 have different pixel X coordinates."
        return
    cal["auto_axis_confidence"] = float(detection.get("confidence", 0.0))
    cal["auto_axis_mode"] = detection.get("mode", "unknown")
    st.session_state.calibration = cal
    st.session_state.apply_calibration_result = "ok"



def _callback_copy_detected_values():
    """on_click callback for 'Copy detected values to calibration fields'.

    Runs BEFORE the next render pass so widget-keyed session state can be
    written without triggering StreamlitAPIException. Stores a result string
    in session_state["copy_detected_result"] for the render pass to display.
    """
    if not _auto_detection_available():
        st.session_state.copy_detected_result = "error:No auto-detection available for the current image."
        return
    if _set_manual_fields_from_detection(st.session_state.auto_axis_detection):
        st.session_state.copy_detected_result = "ok"
    else:
        st.session_state.copy_detected_result = "error:Could not copy detection values — not all calibration points were found."


def _toggle_interp(series_name):
    state = st.session_state.series_states.get(series_name)
    if state is not None:
        state["interpolate"] = not state.get("interpolate", False)


def _load_image_from_upload(image_file):
    """Decode an uploaded image into BGR + RGB arrays. No-op if unchanged."""
    img_bytes = image_file.getvalue()
    img_hash = hashlib.md5(img_bytes).hexdigest()
    if (st.session_state.image_hash == img_hash
            and "image_bgr" in st.session_state):
        return
    try:
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_rgb = np.array(pil)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        st.error(f"Failed to load image: {e}")
        return
    st.session_state.image_rgb = img_rgb
    st.session_state.image_bgr = img_bgr
    st.session_state.image_hash = img_hash
    st.session_state.auto_axis_detection = None
    st.session_state.auto_axis_image_hash = None


def _load_csv(csv_source):
    """Parse and validate CSV text. Returns a DataFrame or None on hard error."""
    try:
        df = pd.read_csv(io.StringIO(csv_source))
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")
        return None

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"CSV is missing required columns: {missing}")
        return None

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["y_err_lower"] = pd.to_numeric(df["y_err_lower"], errors="coerce")
    df["y_err_upper"] = pd.to_numeric(df["y_err_upper"], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=["x", "y"]).reset_index(drop=True)
    if len(df) < n_before:
        st.warning(f"Dropped {n_before - len(df)} row(s) with missing x or y.")

    df["series"] = df["series"].astype(str)

    invalid = ~df["series_color"].apply(is_valid_hex)
    if invalid.any():
        st.warning(
            f"{int(invalid.sum())} row(s) have invalid series_color values; "
            f"using {FALLBACK_HEX} as a fallback."
        )
        df.loc[invalid, "series_color"] = FALLBACK_HEX

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
            "visible": True,
            "use_delta_e": True,
            "delta_e": 30,
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
                st.session_state.df = new_df
                if st.session_state.csv_hash != csv_hash:
                    st.session_state.series_states = _init_series_states(new_df)
                    st.session_state.csv_hash = csv_hash
                df = new_df

        # ----- Calibration status (read-only — controls are in the Calibration tab) -----
        if "image_rgb" in st.session_state:
            if st.session_state.calibration.get("applied"):
                st.success("✓ Calibration applied")
            else:
                st.caption("✗ Not calibrated — use the **Calibration** tab.")

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
                        new_vis = not is_visible
                        st.session_state[f"vis_{series_name}"] = new_vis
                        if series_name in st.session_state.series_states:
                            st.session_state.series_states[series_name]["visible"] = new_vis
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
                            value=int(state.get("delta_e", 30)),
                            key=f"de_{series_name}",
                            help="Lower = tighter match. 20–35 works for most plots.",
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

                    btn_type = "primary" if state.get("interpolate") else "secondary"
                    btn_label = (
                        "Interpolation: ON"
                        if state.get("interpolate")
                        else "Interpolate occluded segments"
                    )
                    st.button(
                        btn_label,
                        key=f"interp_btn_{series_name}",
                        type=btn_type,
                        on_click=_toggle_interp,
                        args=(series_name,),
                    )

                    if state.get("interpolate") and not st.session_state.calibration.get("applied"):
                        st.info("Interpolation requires axis calibration.")

        # ----- Debug status (helps diagnose state issues) -----
        if df is not None and st.session_state.series_states:
            st.divider()
            with st.expander("🔍 Debug: live state", expanded=False):
                rows = []
                for name in df["series"].drop_duplicates().tolist():
                    rows.append({
                        "series": name,
                        "vis_key (st.session_state)":
                            st.session_state.get(f"vis_{name}", "<unset>"),
                        "visible (series_states dict)":
                            st.session_state.series_states.get(name, {}).get("visible", "<unset>"),
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
    image hash; slider movements hit the cache.

    Returns ``(preview, was_cached)`` so callers can show a spinner only on the
    first compute. ``was_cached`` is True when this call was a cache hit.
    """
    cached = st.session_state.get("frame_preview_cache")
    if cached and cached.get("hash") == image_hash:
        return cached["preview"], True

    # Carry forward OCR-tuning settings so the preview matches
    # what `run_calibration` will see when invoked next.
    cfg = CalibrationConfig(
        use_gpu=False,
        min_ocr_confidence=float(st.session_state.get("ocr_min_confidence", 0.20)),
    )
    preview = detect_axis_frame(img_bgr, config=cfg)
    st.session_state.frame_preview_cache = {"hash": image_hash, "preview": preview}
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


def _render_calibration_tab(img_bgr):
    """Combined calibration tab: detection settings, band preview, diagnostics, and manual entry."""
    image_hash = st.session_state.get("image_hash")
    if image_hash is None:
        st.info("Upload an image to begin.")
        return

    # ── Settings row ──────────────────────────────────────────────────────────
    s1, s2, s3, s4 = st.columns([1, 1, 1, 1])
    with s1:
        st.toggle("OCR-assisted detection", key="use_ocr_axis")
        st.toggle("Mask all detected text", key="ocr_mask_all_text")
    with s2:
        st.toggle("Show OCR debug overlay", key="show_ocr_debug_overlay")
        st.toggle(
            "Student-t robust regression", key="use_robust_regression",
            help="Heavy-tailed MLE downweights outlier label-tick pairs. Falls back to OLS for 2-point calibrations.",
        )
    with s3:
        st.slider("Min OCR confidence", 0.0, 1.0, key="ocr_min_confidence", step=0.05)
    with s4:
        st.button(
            "Run Auto Axis Detection",
            key="auto_detect_tab",
            on_click=_run_auto_axis_detection,
            type="primary",
            use_container_width=True,
            help="Runs EasyOCR text masking and tick-label parsing when OCR is enabled.",
        )
        if _auto_detection_available():
            det = st.session_state.auto_axis_detection
            conf_top = float(det.get("confidence", 0.0))
            st.caption(f"Last result: **{det.get('mode', 'unknown')}** — confidence {conf_top:.2f}")

    st.divider()

    # ── Phase A + frame detection (cached per image) ──────────────────────────
    preview = None
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
        st.info("Without a detected frame, label bands cannot be positioned. Try running detection above.")
        return

    default_cfg = CalibrationConfig()
    overrides_key = _band_overrides_key(image_hash)
    saved = st.session_state.get(overrides_key, {})

    init_y_left  = int(saved.get("y_band_extra_px",            default_cfg.y_band_extra_px))
    init_y_vert  = int(saved.get("y_band_extra_vertical_px",   default_cfg.y_band_extra_vertical_px))
    init_x_below = int(saved.get("x_band_extra_px",            default_cfg.x_band_extra_px))
    init_x_horiz = int(saved.get("x_band_extra_horizontal_px", default_cfg.x_band_extra_horizontal_px))

    keys = _slider_keys(image_hash)

    # ── Two-column layout: image (left, wider) | band controls (right) ────────
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
            "Extra vertical pad (px)", min_value=0, max_value=30, value=init_y_vert, step=1,
            help="Padding above/below the bbox for the y-band.",
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
            "Extra horizontal pad (px)", min_value=0, max_value=30, value=init_x_horiz, step=1,
            help="Padding left/right of the bbox for the x-band.",
            key=keys["x_horiz"],
        )
        st.markdown("---")
        col_reset, col_run = st.columns(2)
        with col_reset:
            if st.button("⟳ Defaults", use_container_width=True, key=f"band_reset:{image_hash}"):
                st.session_state.pop(overrides_key, None)
                for k in keys.values():
                    st.session_state.pop(k, None)
                st.rerun()
        with col_run:
            run_clicked = st.button(
                "▶ Run calibration",
                type="primary", use_container_width=True,
                key=f"band_run:{image_hash}",
            )
        with st.expander("Preview diagnostics", expanded=False):
            st.write({
                "frame_preview_run_count": st.session_state.get("frame_preview_run_count", 0),
                "axis_confidence": float(preview.axis_confidence),
                "phase_a_records": len(preview.phase_a_records),
                "phase_a_numeric": sum(1 for r in preview.phase_a_records if r.is_numeric),
            })

    with img_col:
        # Show calibration diagnostic overlay when detection is available;
        # fall back to the lightweight band-window preview otherwise.
        if _auto_detection_available():
            cal_result = st.session_state.get("auto_axis_result")
            show_debug = st.session_state.get("show_ocr_debug_overlay", True)
            if cal_result is not None:
                # Use the CalibrationResult directly so render_overlay reads the
                # actual config (band sizes) that was used — not legacy-dict defaults.
                overlay_img = render_overlay(
                    img_bgr, cal_result,
                    show_band_windows=show_debug,
                    show_grid_rejected=True,
                )
            else:
                # Fallback for cached detections that pre-date auto_axis_result.
                det_img = st.session_state.auto_axis_detection
                if show_debug and det_img.get("ocr_enabled"):
                    overlay_img = build_ocr_debug_overlay(img_bgr, det_img,
                                                          show_mask=True, show_pairing=True)
                else:
                    overlay_img = build_diagnostic_overlay(img_bgr, det_img)
            img_caption = "Calibration diagnostic — axes, ticks, pairings, and P1/P2/P3 anchors (magenta)."
        else:
            y_band = _y_label_band(preview.bbox, extra_left=int(y_extra_left), extra_vertical=int(y_extra_vert))
            x_band = _x_label_band(preview.bbox, extra_below=int(x_extra_below), extra_horizontal=int(x_extra_horiz))
            overlay_img = render_band_preview(
                img_bgr, preview.bbox, y_band, x_band,
                phase_a_records=preview.phase_a_records,
            )
            img_caption = (
                f"Band preview — bbox ({preview.bbox.left}, {preview.bbox.top})→"
                f"({preview.bbox.right}, {preview.bbox.bottom}). "
                f"Y-band x: {y_band[0]}–{y_band[2]}, X-band y: {x_band[1]}–{x_band[3]}. "
                "Run calibration to see full diagnostic."
            )
        st.image(overlay_img, caption=img_caption, use_container_width=True)

    # Persist slider values per-image.
    st.session_state[overrides_key] = {
        "y_band_extra_px":            int(y_extra_left),
        "y_band_extra_vertical_px":   int(y_extra_vert),
        "x_band_extra_px":            int(x_extra_below),
        "x_band_extra_horizontal_px": int(x_extra_horiz),
    }

    if preview.warnings:
        with st.expander(f"Frame-detection warnings ({len(preview.warnings)})", expanded=False):
            for w in preview.warnings:
                st.warning(w)

    if _last_calibration_has_degenerate_warning(image_hash):
        st.warning(
            "⚠️ The last calibration was refused because all paired y-labels read as the "
            "same value — the y-band crop likely bisected multi-digit labels. Increase "
            "**Y-label band → Extra left** until the band fully covers the longest label, "
            "then click **▶ Run calibration** again."
        )

    # ── Run calibration handler ───────────────────────────────────────────────
    if run_clicked:
        with st.spinner("Running full calibration with custom bands..."):
            override_cfg = CalibrationConfig(
                y_band_extra_px=int(y_extra_left),
                y_band_extra_vertical_px=int(y_extra_vert),
                x_band_extra_px=int(x_extra_below),
                x_band_extra_horizontal_px=int(x_extra_horiz),
                use_gpu=False,
                min_ocr_confidence=float(st.session_state.get("ocr_min_confidence", 0.20)),
                use_robust_regression=bool(st.session_state.get("use_robust_regression", True)),
            )
            try:
                result = run_calibration(img_bgr, config=override_cfg)
            except Exception as e:
                st.error(f"Calibration failed: {type(e).__name__}: {e}")
                return
            st.session_state.auto_axis_detection = result.to_legacy_dict()
            st.session_state.auto_axis_result = result
            st.session_state.auto_axis_image_hash = image_hash
            if result.success:
                st.success(f"Calibration succeeded — confidence {result.confidence:.2f}.")
            else:
                st.error("Calibration did not succeed. See warnings and try adjusting the bands.")
            for w in result.warnings:
                st.warning(w)

    st.divider()

    # ── Detection results ─────────────────────────────────────────────────────
    if _auto_detection_available():
        detection = st.session_state.auto_axis_detection
        conf = float(detection.get("confidence", 0.0))
        st.metric("Detection confidence", f"{conf:.2f}", help=f"Mode: {detection.get('mode', 'unknown')}")

        if detection.get("warnings"):
            for w in detection.get("warnings", []):
                st.warning(w)

        if detection.get("ocr_enabled"):
            st.markdown("#### OCR tick tables")
            st.caption(
                "`pixel_position` is the geometry-based tick position. "
                "Correct `value`, uncheck bad rows, then click Update."
            )
            display_cols = ["include", "axis", "raw_text", "cleaned_text", "value", "pixel_position",
                            "ocr_confidence", "pair_distance_px", "parse_status", "status", "flag"]
            x_df = pd.DataFrame(detection.get("x_tick_table", []) or [])
            y_df = pd.DataFrame(detection.get("y_tick_table", []) or [])
            if len(x_df): x_df = x_df[[c for c in display_cols if c in x_df.columns]]
            if len(y_df): y_df = y_df[[c for c in display_cols if c in y_df.columns]]
            left_col, right_col = st.columns(2)
            with left_col:
                st.markdown("**X tick labels**")
                edited_x = st.data_editor(
                    x_df, key="ocr_x_tick_editor", use_container_width=True, hide_index=True,
                    disabled=["axis", "raw_text", "cleaned_text", "ocr_confidence",
                              "pair_distance_px", "parse_status", "status", "flag"],
                )
            with right_col:
                st.markdown("**Y tick labels**")
                edited_y = st.data_editor(
                    y_df, key="ocr_y_tick_editor", use_container_width=True, hide_index=True,
                    disabled=["axis", "raw_text", "cleaned_text", "ocr_confidence",
                              "pair_distance_px", "parse_status", "status", "flag"],
                )
            if st.button("Update calibration from edited tick tables", type="primary", use_container_width=True):
                updated = update_detection_from_tick_tables(detection, edited_x, edited_y)
                st.session_state.auto_axis_detection = updated
                _set_manual_fields_from_detection(updated)
                st.success("Calibration updated from edited tick tables.")
                st.rerun()
        else:
            st.info("OCR was disabled or unavailable — geometry-only detection.")

        st.markdown("#### Detected calibration points")
        pts = []
        for label in ["p1", "p2", "p3"]:
            p = detection.get(label)
            if p:
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
            st.dataframe(pd.DataFrame(pts), use_container_width=True, hide_index=True)

        x_cal = detection.get("x_calibration")
        y_cal = detection.get("y_calibration")
        x_grid = detection.get("x_grid_fit")
        y_grid = detection.get("y_grid_fit")
        if x_cal or y_cal or x_grid or y_grid:
            st.markdown("#### Calibration diagnostics")
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

        st.divider()

    # ── Manual calibration points ─────────────────────────────────────────────
    st.markdown("#### Calibration Points")
    st.caption(
        "Pre-filled by **Apply detected calibration** or **Copy detected → fields**. "
        "Edit manually if needed, then click **Apply Calibration**."
    )
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        st.markdown("**P1 — X-axis left**")
        p1_px_x  = st.number_input("Pixel X", key="p1_px_x",  value=0.0, step=1.0)
        p1_px_y  = st.number_input("Pixel Y", key="p1_px_y",  value=0.0, step=1.0)
        p1_data_x = st.number_input("Data X",  key="p1_data_x", value=0.0)
        p1_data_y = st.number_input(
            "Data Y (baseline)", key="p1_data_y", value=0.0,
            help="Data Y at the X-axis baseline — anchors the Y scale.",
        )
    with pc2:
        st.markdown("**P2 — X-axis right**")
        p2_px_x  = st.number_input("Pixel X", key="p2_px_x",  value=0.0, step=1.0)
        p2_px_y  = st.number_input("Pixel Y", key="p2_px_y",  value=0.0, step=1.0)
        p2_data_x = st.number_input("Data X",  key="p2_data_x", value=1.0)
        st.number_input("Data Y", key="p2_data_y", value=0.0, disabled=True,
                        help="Disabled for X-axis points.")
    with pc3:
        st.markdown("**P3 — Y-axis top**")
        p3_px_x  = st.number_input("Pixel X", key="p3_px_x",  value=0.0, step=1.0)
        p3_px_y  = st.number_input("Pixel Y", key="p3_px_y",  value=0.0, step=1.0)
        st.number_input("Data X", key="p3_data_x", value=0.0, disabled=True,
                        help="Disabled for Y-axis points.")
        p3_data_y = st.number_input("Data Y",  key="p3_data_y", value=1.0)

    ba1, ba2, ba3 = st.columns(3)
    with ba1:
        if st.button("Apply Calibration", type="primary", use_container_width=True):
            cal = compute_calibration(
                p1_px_x, p1_px_y, p2_px_x, p2_px_y, p3_px_x, p3_px_y,
                p1_data_x, p2_data_x, p3_data_y, p1_data_y,
            )
            if cal is None:
                st.error("Invalid points — P1 and P2 must have different pixel X coordinates.")
            else:
                st.session_state.calibration = cal
                st.rerun()
    with ba2:
        st.button(
            "Copy detected → fields",
            on_click=_callback_copy_detected_values,
            use_container_width=True,
            help="Pre-fill P1/P2/P3 from the last auto-detection result.",
        )
        res = st.session_state.get("copy_detected_result")
        if res == "ok":
            st.success("Fields updated from detection.")
            st.session_state.copy_detected_result = None
        elif res and res.startswith("error:"):
            st.error(res[len("error:"):])
            st.session_state.copy_detected_result = None
    with ba3:
        st.button(
            "Apply detected calibration",
            on_click=_callback_apply_detected_calibration,
            use_container_width=True,
            help="Apply calibration directly from the last auto-detection result.",
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
        st.success(
            f"✓ Calibration applied — "
            f"X: {cal_s.get('x_scale', 0):.5f} data/px  |  "
            f"Y: {cal_s.get('y_scale', 0):.5f} data/px"
        )
    else:
        st.info("✗ Not calibrated — apply calibration to enable the Extracted Overlay tab.")


def _render_mask_view(img_rgb, img_bgr, img_hash, df, series_states, cal,
                      container_height=600, key_suffix=""):
    """Render Tab 1's mask composite (or plain image if no CSV loaded)."""
    if df is None or not series_states:
        display = img_rgb
    else:
        display = build_composite(img_bgr, img_hash, series_states, df, cal)

    fig = px.imshow(display, binary_string=True)
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
        st.warning("Apply axis calibration in the **Calibration** tab to enable the overlay.")
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

    tab1, tab2, tab3, tab4 = st.tabs([
        "Image & Masks",
        "Extracted Overlay",
        "Calibration",
        "Side by Side",
    ])

    with tab1:
        _render_mask_view(img_rgb, img_bgr, img_hash, df, series_states, cal,
                          key_suffix="_tab1")

    with tab2:
        _render_overlay_view(img_rgb, df, series_states, cal,
                             key_suffix="_tab2")

    with tab3:
        _render_calibration_tab(img_bgr)

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Color Masks**")
            _render_mask_view(img_rgb, img_bgr, img_hash, df, series_states, cal,
                              container_height=500, key_suffix="_tab4_left")
        with col2:
            st.markdown("**Extracted Overlay**")
            _render_overlay_view(img_rgb, df, series_states, cal,
                                 container_height=500, key_suffix="_tab4_right")


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
