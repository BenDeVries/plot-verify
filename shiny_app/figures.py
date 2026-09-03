"""Plotly figure builders for the Shiny PlotVerify front-end.

Pure UI-agnostic helpers: they take numpy arrays / dataclasses and return
plotly.graph_objects Figures. The Shiny `server` function wraps the result
in `FigureWidget` (via shinywidgets) so layout changes can be observed.

The two main outputs are:

- ``build_calibration_edit_figure`` — image with three draggable shapes
  representing the P1/P2/P3 anchors. The shapes use ``editable=True`` so a
  Plotly user can drag them; the caller listens for ``layout.shapes``
  changes on the FigureWidget to sync the Anchors back to app state.

- ``build_data_overlay_figure`` — the calibrated image with extracted-data
  scatter traces drawn in data coordinates. This mirrors
  ``app_auto_axis.build_overlay_figure`` but takes traces directly so it
  has no Streamlit dependency.
"""
from __future__ import annotations

import base64
import io
import math
from typing import Iterable, Optional, Union

import numpy as np
import plotly.graph_objects as go
from PIL import Image

from axis_pipeline import CalibrationResult
from plotverify_core import (
    Anchors,
    OverlayTrace,
    data_to_px,
    px_to_data,
)
from plotverify_core.overlay_traces import is_horizontal_layout
from plotverify_core.colors import FALLBACK_HEX, is_valid_hex


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def encode_image_data_uri(img_rgb: np.ndarray) -> str:
    """Encode an RGB ndarray as a ``data:image/png;base64,...`` URI.

    The result can be passed as ``source=`` to ``fig.add_layout_image``; this
    sidesteps Plotly's lazy PIL→PNG conversion at serialization time so the
    cost is paid once when the image is first loaded rather than on every
    ``FigureWidget`` re-render.
    """
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", compress_level=3)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


ANCHOR_LABELS = ("P1", "P2")
ANCHOR_COLORS = {"P1": "#e02020", "P2": "#1f9b4c", "P3": "#2060e0"}
ANCHOR_RADIUS_PX = 9  # half-width of each draggable circle, in image pixels
GUIDE_LINE_COLOR = "#444"


# ---------------------------------------------------------------------------
# cal-dict helpers
# ---------------------------------------------------------------------------


def cal_dict_from_result(result: Optional[CalibrationResult]) -> dict:
    """Project a typed ``CalibrationResult`` onto the legacy ``cal`` dict.

    The dict has the shape consumed by ``plotverify_core.px_to_data`` and
    ``plotverify_core.data_to_px``: ``x_scale``, ``x_offset``, ``y_scale``,
    ``y_offset``, ``x_log_base``, ``y_log_base``, ``applied``.
    """
    if (
        result is None
        or not result.success
        or result.x_calibration is None
        or result.y_calibration is None
    ):
        return {"applied": False}
    xc, yc = result.x_calibration, result.y_calibration
    return {
        "applied": True,
        "x_scale": float(xc.scale),
        "x_offset": float(xc.offset),
        "y_scale": float(yc.scale),
        "y_offset": float(yc.offset),
        "x_log_base": float(xc.log_base) if xc.log_base else None,
        "y_log_base": float(yc.log_base) if yc.log_base else None,
    }


def default_anchors_for_image(width: int, height: int) -> Anchors:
    """Seed P1/P2/P3 at (10%/90%, 10%) of image dimensions (Bug #18 default)."""
    return Anchors(
        p1_pixel=(0.10 * width, 0.90 * height),
        p2_pixel=(0.90 * width, 0.90 * height),
        p3_pixel=(0.10 * width, 0.10 * height),
    )


def anchors_from_result(result: Optional[CalibrationResult],
                        fallback: Anchors) -> Anchors:
    """Read P1/P2/P3 out of a calibration result if available."""
    if result is None or not result.success:
        return fallback
    return Anchors(
        p1_pixel=tuple(map(float, result.p1_pixel)),
        p2_pixel=tuple(map(float, result.p2_pixel)),
        p3_pixel=tuple(map(float, result.p3_pixel)),
        p1_data_x=float(result.p1_data_x) if result.p1_data_x is not None else fallback.p1_data_x,
        p2_data_x=float(result.p2_data_x) if result.p2_data_x is not None else fallback.p2_data_x,
        p1_data_y=float(result.p1_data_y) if result.p1_data_y is not None else fallback.p1_data_y,
        p3_data_y=float(result.p3_data_y) if result.p3_data_y is not None else fallback.p3_data_y,
        x_log_base=(float(result.x_calibration.log_base)
                    if result.x_calibration and result.x_calibration.log_base else None),
        y_log_base=(float(result.y_calibration.log_base)
                    if result.y_calibration and result.y_calibration.log_base else None),
    )


# ---------------------------------------------------------------------------
# Calibration-edit figure (image + draggable anchor shapes)
# ---------------------------------------------------------------------------


def guide_line_traces(anchors: Anchors, width: int, height: int) -> list:
    """Four dashed guide lines that visualise the axis-rectangle, as
    ``go.Scatter`` traces (not shapes).

    Returning traces rather than shapes matters because the calibration
    figure uses Plotly's *older* shape-edit mode (``config.edits.
    shapePosition``), which makes every shape draggable. By keeping the
    guide lines in ``data`` we sidestep that — they stay fixed and only
    the anchor circles in ``layout.shapes`` are user-draggable.

    Order is stable so ``_push_anchors_to_widget`` can update by index:

    - 0: P1-P2 horizontal (50% opacity)
    - 1: P3 horizontal (25% opacity)
    - 2: P1-P3 vertical (50% opacity)
    - 3: P2 vertical (25% opacity)
    """
    p2, p3 = anchors.p2_pixel, anchors.p3_pixel

    def _line(xs, ys, opacity):
        return go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=GUIDE_LINE_COLOR, width=1, dash="dash"),
            opacity=opacity,
            showlegend=False,
            hoverinfo="skip",
        )

    return [
        _line([0, width], [p2[1], p2[1]], 0.5),     # baseline (P2 y = derived P1 y)
        _line([0, width], [p3[1], p3[1]], 0.25),    # top (P1 displayed / internal P3)
        _line([p3[0], p3[0]], [0, height], 0.5),    # left edge (P1 displayed x = internal P3 x)
        _line([p2[0], p2[0]], [0, height], 0.25),   # right edge (P2 x)
    ]


def enforce_anchor_constraints(new: Anchors, prev: Anchors) -> Anchors:
    """Apply the axis-rectangle constraints: ``P1.y==P2.y`` and ``P1.x==P3.x``.

    Detects which anchor moved most relative to ``prev`` (one anchor at a
    time during a drag; the largest-delta anchor wins for numeric edits)
    and propagates the linked coordinate to the partner(s) so the moved
    anchor is preserved exactly and the rectangle stays closed.
    """
    d1 = (abs(new.p1_pixel[0] - prev.p1_pixel[0])
          + abs(new.p1_pixel[1] - prev.p1_pixel[1]))
    d2 = (abs(new.p2_pixel[0] - prev.p2_pixel[0])
          + abs(new.p2_pixel[1] - prev.p2_pixel[1]))
    d3 = (abs(new.p3_pixel[0] - prev.p3_pixel[0])
          + abs(new.p3_pixel[1] - prev.p3_pixel[1]))
    p1, p2, p3 = new.p1_pixel, new.p2_pixel, new.p3_pixel
    if d1 >= d2 and d1 >= d3:
        # P1 leads (or nothing moved): snap P2.y, P3.x to P1.
        p2 = (p2[0], p1[1])
        p3 = (p1[0], p3[1])
    elif d2 >= d3:
        # P2 leads: P1.y follows P2; P3.x re-anchors to the (unchanged) P1.x.
        p1 = (p1[0], p2[1])
        p3 = (p1[0], p3[1])
    else:
        # P3 leads: P1.x follows P3; P2.y re-anchors to the (unchanged) P1.y.
        p1 = (p3[0], p1[1])
        p2 = (p2[0], p1[1])
    return Anchors(
        p1_pixel=p1, p2_pixel=p2, p3_pixel=p3,
        p1_data_x=new.p1_data_x, p2_data_x=new.p2_data_x,
        p1_data_y=new.p1_data_y, p3_data_y=new.p3_data_y,
        x_log_base=new.x_log_base, y_log_base=new.y_log_base,
    )


def anchor_annotations(anchors: Anchors, selected: Optional[str] = None) -> list:
    """Build the three draggable anchor annotations.

    Annotations are used instead of shapes because Plotly always renders
    shapes with resize handles when they're editable (whether via per-shape
    ``editable=True`` or ``config.edits.shapePosition``). Annotations are
    inherently fixed-size text boxes — making them draggable via
    ``config.edits.annotationPosition=True`` yields move-only behavior with
    no distortion possible.

    ``captureevents=True`` is required for Plotly to fire
    ``plotly_clickannotation`` on click, which the Shiny app wires up to
    keyboard nudging. The annotation matching ``selected`` (if any) gets
    a darker border + white outline to make the current keyboard target
    visually obvious.
    """
    points = (
        ("P1", anchors.p3_pixel, ANCHOR_COLORS["P1"]),  # top-left (internal p3)
        ("P2", anchors.p2_pixel, ANCHOR_COLORS["P2"]),  # bottom-right (internal p2)
    )
    annotations = []
    for name, (px, py), color in points:
        is_selected = (name == selected)
        annotations.append(dict(
            x=px, y=py,
            xref="x", yref="y",
            xanchor="center", yanchor="middle",
            text=f"<b>{name}</b>",
            showarrow=False,
            font=dict(color="white", size=13 if is_selected else 12),
            bgcolor=color,
            bordercolor="#000" if is_selected else color,
            borderwidth=3 if is_selected else 2,
            borderpad=3,
            opacity=1.0 if is_selected else 0.9,
            captureevents=True,
            name=name,
        ))
    return annotations


def annotations_to_anchors(annotations: Iterable, existing: Anchors) -> Anchors:
    """Read P1/P2/P3 positions back out of a layout's annotations.

    Mirrors ``shapes_to_anchors`` but reads single-point ``x``/``y`` instead
    of ``x0,y0,x1,y1``. Non-matching annotations are ignored; data-value
    fields and log-base flags are carried over from ``existing``.
    """
    px_by_name: dict[str, tuple[float, float]] = {}
    for ann in annotations:
        name = ann.get("name") if isinstance(ann, dict) else getattr(ann, "name", None)
        if name not in ANCHOR_LABELS:
            continue
        x = ann.get("x") if isinstance(ann, dict) else getattr(ann, "x", None)
        y = ann.get("y") if isinstance(ann, dict) else getattr(ann, "y", None)
        if x is None or y is None:
            continue
        px_by_name[name] = (float(x), float(y))
    # Display "P1" is the top-left anchor (internal p3); "P2" is bottom-right (internal p2).
    # The hidden internal p1 (bottom-left) is always derived as (p3.x, p2.y).
    new_p3 = px_by_name.get("P1", existing.p3_pixel)
    new_p2 = px_by_name.get("P2", existing.p2_pixel)
    derived_p1 = (new_p3[0], new_p2[1])
    return Anchors(
        p1_pixel=derived_p1,
        p2_pixel=new_p2,
        p3_pixel=new_p3,
        p1_data_x=existing.p1_data_x,
        p2_data_x=existing.p2_data_x,
        p1_data_y=existing.p1_data_y,
        p3_data_y=existing.p3_data_y,
        x_log_base=existing.x_log_base,
        y_log_base=existing.y_log_base,
    )


# Legacy shape-based helpers kept for back-compat with the existing tests
# that exercise ``shapes_to_anchors``; the figure no longer uses them.
def anchor_shapes(anchors: Anchors, image_height: int = 0) -> list:
    """Build three Plotly circle shapes (one per anchor).

    No longer used by the calibration figure (anchors are now annotations
    so they can be dragged without resize handles), but retained for unit
    tests that exercise ``shapes_to_anchors``.
    """
    points = (
        ("P1", anchors.p1_pixel, ANCHOR_COLORS["P1"]),
        ("P2", anchors.p2_pixel, ANCHOR_COLORS["P2"]),
        ("P3", anchors.p3_pixel, ANCHOR_COLORS["P3"]),
    )
    shapes = []
    for name, (px, py), color in points:
        r = ANCHOR_RADIUS_PX
        shapes.append(dict(
            type="circle",
            xref="x", yref="y",
            x0=px - r, y0=py - r,
            x1=px + r, y1=py + r,
            line=dict(color=color, width=2),
            fillcolor=color,
            opacity=0.45,
            name=name,
        ))
    return shapes


def all_calibration_shapes(anchors: Anchors, width: int, height: int) -> list:
    """No layout shapes — anchors are annotations, guide lines are traces."""
    return []


def band_shapes(
    y_band: tuple,
    x_band: tuple,
) -> list:
    """Plotly rectangle shapes visualising the OCR label-scan bands.

    y_band / x_band are (x0, y0, x1, y1) tuples in image pixel coordinates
    (top-left origin, matching the cal_plot y-axis which runs range=[h, 0]).
    Green = y-axis label strip; blue = x-axis label strip.

    Coordinates are normalised (swapped if inverted) so negative extension
    values — which cause the band to appear on the opposite side of the axis —
    still render correctly.
    """
    shapes = []
    for band, fill, stroke in [
        (y_band, "rgba(0,180,0,0.12)", "rgba(0,180,0,0.8)"),
        (x_band, "rgba(0,100,200,0.12)", "rgba(0,100,200,0.8)"),
    ]:
        if not band:
            continue
        x0, y0, x1, y1 = band
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        if x0 == x1 or y0 == y1:
            continue  # degenerate (zero area)
        shapes.append(dict(
            type="rect", xref="x", yref="y",
            x0=x0, y0=y0, x1=x1, y1=y1,
            fillcolor=fill,
            line=dict(color=stroke, width=1.5),
            layer="above",
        ))
    return shapes


def build_calibration_edit_figure(
    img_rgb: np.ndarray,
    anchors: Anchors,
    *,
    diagnostic_rgb: Optional[np.ndarray] = None,
    image_data_uri: Optional[str] = None,
) -> go.Figure:
    """Build the calibration tab's editable figure.

    Pass ``image_data_uri`` (preferred) to avoid PNG-encoding the image on
    every rebuild; otherwise the RGB ndarray is encoded inline. When a
    ``diagnostic_rgb`` overlay is supplied it always supersedes the cached
    URI for this single render.
    """
    if diagnostic_rgb is not None:
        h, w = diagnostic_rgb.shape[:2]
        source: Union[str, Image.Image] = Image.fromarray(diagnostic_rgb)
    elif image_data_uri is not None:
        h, w = img_rgb.shape[:2]
        source = image_data_uri
    else:
        h, w = img_rgb.shape[:2]
        source = Image.fromarray(img_rgb)

    fig = go.Figure()
    fig.add_layout_image(dict(
        source=source,
        xref="x", yref="y",
        x=0, y=0,
        sizex=w, sizey=h,
        xanchor="left", yanchor="top",
        sizing="stretch",
        opacity=1.0,
        layer="below",
    ))

    # Anchors are layout annotations (draggable via config.edits.
    # annotationPosition); guide lines are scatter traces (non-interactive).
    for trace in guide_line_traces(anchors, w, h):
        fig.add_trace(trace)

    fig.update_layout(
        annotations=anchor_annotations(anchors),
        xaxis=dict(visible=False, range=[0, w], constrain="domain"),
        yaxis=dict(visible=False, range=[h, 0], scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, t=0, b=0),
        height=620,
        # ``dragmode="pan"`` so the background drag pans rather than zooms;
        # annotations are dragged in their own handler regardless of dragmode.
        dragmode="pan",
        plot_bgcolor="white",
    )
    return fig


def shapes_to_anchors(shapes: Iterable[dict], existing: Anchors) -> Anchors:
    """Read the three editable shapes out of a Plotly layout and rebuild Anchors.

    The data-value fields (``p1_data_x`` etc.) and log-base flags from
    ``existing`` are preserved — drag operations only change pixel positions.
    """
    px_by_name: dict[str, tuple[float, float]] = {}
    for shape in shapes:
        name = shape.get("name") if isinstance(shape, dict) else getattr(shape, "name", None)
        if name not in ANCHOR_LABELS:
            continue
        # Circle: centre = midpoint of (x0,y0)-(x1,y1).
        x0 = shape.get("x0") if isinstance(shape, dict) else getattr(shape, "x0", None)
        y0 = shape.get("y0") if isinstance(shape, dict) else getattr(shape, "y0", None)
        x1 = shape.get("x1") if isinstance(shape, dict) else getattr(shape, "x1", None)
        y1 = shape.get("y1") if isinstance(shape, dict) else getattr(shape, "y1", None)
        if None in (x0, y0, x1, y1):
            continue
        px_by_name[name] = ((float(x0) + float(x1)) / 2.0,
                            (float(y0) + float(y1)) / 2.0)
    return Anchors(
        p1_pixel=px_by_name.get("P1", existing.p1_pixel),
        p2_pixel=px_by_name.get("P2", existing.p2_pixel),
        p3_pixel=px_by_name.get("P3", existing.p3_pixel),
        p1_data_x=existing.p1_data_x,
        p2_data_x=existing.p2_data_x,
        p1_data_y=existing.p1_data_y,
        p3_data_y=existing.p3_data_y,
        x_log_base=existing.x_log_base,
        y_log_base=existing.y_log_base,
    )


# ---------------------------------------------------------------------------
# Data overlay figure (calibrated image + extracted points)
# ---------------------------------------------------------------------------

def _log_axis_title(axis_label: str, base: float) -> str:
    """Human-readable axis title that shows the actual log base."""
    if abs(base - math.e) < 1e-10:
        return f"{axis_label} (ln)"
    b_str = str(int(round(base))) if abs(base - round(base)) < 0.001 else f"{base:.4g}"
    return f"{axis_label} (log{b_str})"


def _log_axis_ticks(lo: float, hi: float, base: float):
    """Tick values/labels at integer powers of *base* that lie within [lo, hi].

    Plotly ``type="log"`` positions tick marks by their data value, so we can
    pass the actual power-of-base values as ``tickvals`` and supply matching
    ``ticktext`` strings — no manual log10 conversion needed.
    """
    lo_exp = math.floor(math.log(lo, base) - 1e-9)
    hi_exp = math.ceil(math.log(hi, base) + 1e-9)
    vals, texts = [], []
    for exp in range(lo_exp, hi_exp + 1):
        v = base ** exp
        if lo <= v <= hi:
            vals.append(v)
            if abs(base - math.e) < 1e-10:
                texts.append("1" if exp == 0 else f"e^{exp}")
            else:
                texts.append(str(int(round(v))) if abs(v - round(v)) < 0.001 else f"{v:.4g}")
    if not vals:
        return [lo, hi], [f"{lo:.4g}", f"{hi:.4g}"]
    return vals, texts


def build_data_overlay_figure(
    img_rgb: np.ndarray,
    traces: list[OverlayTrace],
    cal: dict,
    *,
    height: int = 620,
    edit_point_ids: Optional[set[str]] = None,
    image_data_uri: Optional[str] = None,
    plot_type: str = "time_series",
    orientation: str = "vertical",
) -> go.Figure:
    """Build the calibrated overlay (image as background + scatter traces).

    ``cal`` is the legacy dict produced by ``cal_dict_from_result``. Pass
    ``image_data_uri`` to skip PNG-encoding the image on every rebuild.
    """
    h, w = img_rgb.shape[:2]
    fig = go.Figure()

    if not cal.get("applied"):
        # No image embedded in the un-calibrated state — that keeps the tab
        # cheap when there is nothing useful to overlay yet.
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=0, r=0, t=0, b=0), height=height,
            plot_bgcolor="#fafafa",
        )
        fig.add_annotation(
            text="Apply calibration in the Calibrate tab to see the overlay.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="#666"),
        )
        return fig

    x_left, _ = px_to_data(0, 0, cal)
    x_right, _ = px_to_data(w, 0, cal)
    _, y_top = px_to_data(0, 0, cal)
    _, y_bottom = px_to_data(0, h, cal)

    x_log = cal.get("x_log_base")
    y_log = cal.get("y_log_base")

    source = image_data_uri if image_data_uri is not None else Image.fromarray(img_rgb)
    img_x = min(x_left, x_right)
    img_y = max(y_top, y_bottom)
    img_sx = abs(x_right - x_left)
    img_sy = abs(y_top - y_bottom)
    if x_log and x_left > 0 and x_right > 0:
        img_x = float(np.log10(min(x_left, x_right)))
        img_sx = float(abs(np.log10(x_right) - np.log10(x_left)))
    if y_log and y_top > 0 and y_bottom > 0:
        img_y = float(np.log10(max(y_top, y_bottom)))
        img_sy = float(abs(np.log10(y_top) - np.log10(y_bottom)))
    fig.add_layout_image(dict(
        source=source, xref="x", yref="y",
        x=img_x, y=img_y, sizex=img_sx, sizey=img_sy,
        xanchor="left", yanchor="top",
        sizing="stretch", opacity=1.0, layer="below",
    ))

    edit_point_ids = edit_point_ids or set()

    is_scatter = plot_type == "scatter"
    is_forest = plot_type == "forest"
    is_bar = plot_type == "bar"
    is_box = plot_type == "box"
    is_km = plot_type == "kaplan_meier"
    is_horiz = is_horizontal_layout(plot_type, orientation)

    for trace in traces:
        plot_visible = True if trace.visible else "legendonly"

        # --- Ribbon / band fills ---
        if is_km and trace.has_err.any() and len(trace.ribbon_x):
            # Step-shaped confidence band for Kaplan-Meier.
            h_str = trace.color_hex.lstrip("#")
            fill_rgba = "rgba({},{},{},0.2)".format(
                int(h_str[0:2], 16), int(h_str[2:4], 16), int(h_str[4:6], 16)
            )
            fig.add_trace(go.Scatter(
                x=trace.ribbon_x, y=trace.ribbon_y_upper,
                mode="lines", line=dict(width=0, shape="hv"),
                legendgroup=trace.series,
                name=f"_pv_rib_u_{trace.series}",
                showlegend=False, visible=plot_visible, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=trace.ribbon_x, y=trace.ribbon_y_lower,
                mode="lines", line=dict(width=0, shape="hv"),
                fill="tonexty", fillcolor=fill_rgba,
                legendgroup=trace.series,
                name=f"_pv_rib_l_{trace.series}",
                showlegend=False, visible=plot_visible, hoverinfo="skip",
            ))
        elif trace.has_err.any() and not is_scatter and not is_forest and not is_bar and not is_box:
            # Standard vertical ribbon for time_series / error_bar.
            h_str = trace.color_hex.lstrip("#")
            fill_rgba = "rgba({},{},{},0.2)".format(
                int(h_str[0:2], 16), int(h_str[2:4], 16), int(h_str[4:6], 16)
            )
            fig.add_trace(go.Scatter(
                x=trace.ribbon_x, y=trace.ribbon_y_upper,
                mode="lines", line=dict(width=0),
                legendgroup=trace.series,
                name=f"_pv_rib_u_{trace.series}",
                showlegend=False, visible=plot_visible, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=trace.ribbon_x, y=trace.ribbon_y_lower,
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor=fill_rgba,
                legendgroup=trace.series,
                name=f"_pv_rib_l_{trace.series}",
                showlegend=False, visible=plot_visible, hoverinfo="skip",
            ))

        if is_forest and trace.has_err.any():
            h_str = trace.color_hex.lstrip("#")
            fill_rgba = "rgba({},{},{},0.2)".format(
                int(h_str[0:2], 16), int(h_str[2:4], 16), int(h_str[4:6], 16)
            )
            band_x: list = []
            band_y: list = []
            hh = 0.32
            for i in range(len(trace.x)):
                if not trace.has_err[i]:
                    continue
                lo = float(trace.x[i] - trace.err_array_minus[i])
                hi = float(trace.x[i] + trace.err_array_plus[i])
                yc = float(trace.y[i])
                band_x += [lo, hi, hi, lo, lo, None]
                band_y += [yc - hh, yc - hh, yc + hh, yc + hh, yc - hh, None]
            fig.add_trace(go.Scatter(
                x=band_x, y=band_y,
                mode="lines", line=dict(width=0),
                fill="toself", fillcolor=fill_rgba,
                legendgroup=trace.series,
                name=f"_pv_rib_{trace.series}",
                showlegend=False, visible=plot_visible, hoverinfo="skip",
            ))

        # --- Box-plot components ---
        if is_box:
            h_str = trace.color_hex.lstrip("#")
            fill_rgba = "rgba({},{},{},0.15)".format(
                int(h_str[0:2], 16), int(h_str[2:4], 16), int(h_str[4:6], 16)
            )
            _status = trace.status or [""] * len(trace.x)
            has_q = (trace.box_q1 is not None and trace.box_q3 is not None)
            if has_q:
                # Built in (category, value) space; the axis assignment at the
                # end flips for horizontal boxes (value axis along x).
                box_cat: list = []
                box_val: list = []
                med_cat: list = []
                med_val: list = []
                whisk_cat: list = []
                whisk_val: list = []
                bw = 0.3
                for i in range(len(trace.x)):
                    if _status[i].lower() == "outlier":
                        continue
                    cat = float(trace.y[i]) if is_horiz else float(trace.x[i])
                    val_pt = float(trace.x[i]) if is_horiz else float(trace.y[i])
                    q1 = float(trace.box_q1[i])
                    q3 = float(trace.box_q3[i])
                    if not (np.isfinite(q1) and np.isfinite(q3)):
                        continue
                    box_cat += [cat - bw, cat + bw, cat + bw, cat - bw, cat - bw, None]
                    box_val += [q1, q1, q3, q3, q1, None]
                    if trace.box_median is not None and np.isfinite(trace.box_median[i]):
                        med = float(trace.box_median[i])
                        med_cat += [cat - bw, cat + bw, None]
                        med_val += [med, med, None]
                    if trace.has_lower[i]:
                        lo = val_pt - float(trace.err_array_minus[i])
                        whisk_cat += [cat, cat, None]
                        whisk_val += [lo, q1, None]
                        whisk_cat += [cat - bw * 0.5, cat + bw * 0.5, None]
                        whisk_val += [lo, lo, None]
                    if trace.has_upper[i]:
                        hi = val_pt + float(trace.err_array_plus[i])
                        whisk_cat += [cat, cat, None]
                        whisk_val += [q3, hi, None]
                        whisk_cat += [cat - bw * 0.5, cat + bw * 0.5, None]
                        whisk_val += [hi, hi, None]
                if is_horiz:
                    box_x, box_y = box_val, box_cat
                    med_x, med_y = med_val, med_cat
                    whisk_x, whisk_y = whisk_val, whisk_cat
                else:
                    box_x, box_y = box_cat, box_val
                    med_x, med_y = med_cat, med_val
                    whisk_x, whisk_y = whisk_cat, whisk_val
                if box_x:
                    fig.add_trace(go.Scatter(
                        x=box_x, y=box_y,
                        mode="lines", line=dict(width=1, color=trace.color_hex),
                        fill="toself", fillcolor=fill_rgba,
                        legendgroup=trace.series,
                        name=f"_pv_box_{trace.series}",
                        showlegend=False, visible=plot_visible, hoverinfo="skip",
                    ))
                if med_x:
                    fig.add_trace(go.Scatter(
                        x=med_x, y=med_y,
                        mode="lines", line=dict(width=2, color=trace.color_hex),
                        legendgroup=trace.series,
                        name=f"_pv_med_{trace.series}",
                        showlegend=False, visible=plot_visible, hoverinfo="skip",
                    ))
                if whisk_x:
                    fig.add_trace(go.Scatter(
                        x=whisk_x, y=whisk_y,
                        mode="lines", line=dict(width=1, color=trace.color_hex),
                        legendgroup=trace.series,
                        name=f"_pv_whisk_{trace.series}",
                        showlegend=False, visible=plot_visible, hoverinfo="skip",
                    ))

        # error-bar visualisation — along x for horizontal layouts (the
        # interval brackets the value axis), vertical otherwise. Box plots
        # show whiskers via the dedicated component above, so skip the
        # generic error bars.
        err_bar = None
        if trace.has_err.any() and not is_box:
            err_bar = dict(
                type="data", symmetric=False,
                array=trace.err_array_plus,
                arrayminus=trace.err_array_minus,
                color=trace.color_hex, thickness=1.2, width=4,
            )
        err_x = err_bar if is_horiz else None
        err_y = None if is_horiz else err_bar

        # If any point in this series has been edited, mark with a black ring.
        point_ids = trace.point_ids or [f"{trace.series}#{i}" for i in range(len(trace.x))]
        marker_line_colors = []
        marker_line_widths = []
        for pid in point_ids:
            if pid in edit_point_ids:
                marker_line_colors.append("#000")
                marker_line_widths.append(2.0)
            else:
                marker_line_colors.append("rgba(0,0,0,0.5)")
                marker_line_widths.append(0.5)

        # customdata and display mode depend on plot type.
        if is_forest:
            scatter_mode = "markers"
            line_width = 0
            marker_fill = trace.marker_color_hex
            _summary = (trace.is_summary if trace.is_summary is not None
                        else [False] * len(trace.x))
            marker_symbol = ["diamond" if s else "circle" for s in _summary]
            _status = trace.status if trace.status is not None else [""] * len(trace.x)
            customdata = [[pid, st] for pid, st in zip(point_ids, _status)]
            hovertemplate = ("%{fullData.name}: %{x:.4g}"
                             "<br>%{customdata[1]}"
                             "<extra>%{customdata[0]}</extra>")
        elif is_bar:
            scatter_mode = "markers"
            line_width = 0
            marker_fill = trace.marker_color_hex
            marker_symbol = "circle"
            customdata = [[pid] for pid in point_ids]
            hovertemplate = ("%{fullData.name}: (%{x:.4g}, %{y:.4g})"
                             "<extra>%{customdata[0]}</extra>")
        elif is_box:
            scatter_mode = "markers"
            line_width = 0
            marker_fill = trace.marker_color_hex
            _status = trace.status or [""] * len(trace.x)
            marker_symbol = ["diamond-open" if s.lower() == "outlier" else "circle"
                             for s in _status]
            customdata = [[pid, st] for pid, st in zip(point_ids, _status)]
            hovertemplate = ("%{fullData.name}: (%{x:.4g}, %{y:.4g})"
                             "<br>%{customdata[1]}"
                             "<extra>%{customdata[0]}</extra>")
        elif is_km:
            scatter_mode = "lines+markers"
            line_width = 2
            marker_fill = trace.marker_color_hex
            marker_symbol = "circle"
            _status = trace.status or [""] * len(trace.x)
            _at_risk = trace.at_risk
            if _at_risk is not None:
                customdata = [[pid, st, f"{ar:.0f}" if np.isfinite(ar) else ""]
                              for pid, st, ar in zip(point_ids, _status, _at_risk)]
                hovertemplate = ("%{fullData.name}: (%{x:.4g}, %{y:.4g})"
                                 "<br>At risk: %{customdata[2]}"
                                 "<extra>%{customdata[0]}</extra>")
            else:
                customdata = [[pid, st] for pid, st in zip(point_ids, _status)]
                hovertemplate = ("%{fullData.name}: (%{x:.4g}, %{y:.4g})"
                                 "<extra>%{customdata[0]}</extra>")
        else:
            scatter_mode = "markers" if is_scatter else "lines+markers"
            line_width = 0 if is_scatter else 2
            marker_fill = (_hex_to_rgba(trace.marker_color_hex, 0.35)
                           if is_scatter else trace.marker_color_hex)
            marker_symbol = "circle"
            customdata = [[pid] for pid in point_ids]
            hovertemplate = ("%{fullData.name}: (%{x:.4g}, %{y:.4g})"
                             "<extra>%{customdata[0]}</extra>")

        line_shape = "hv" if is_km else None
        fig.add_trace(go.Scatter(
            x=trace.x, y=trace.y,
            mode=scatter_mode,
            line=dict(color=trace.color_hex, width=line_width,
                      **({"shape": line_shape} if line_shape else {})),
            marker=dict(
                symbol=marker_symbol,
                color=marker_fill,
                size=10,
                line=dict(color=marker_line_colors, width=marker_line_widths)),
            error_x=err_x,
            error_y=err_y,
            name=trace.series,
            legendgroup=trace.series,
            showlegend=not is_forest,
            visible=plot_visible,
            customdata=customdata,
            hovertemplate=hovertemplate,
        ))

        # KM censored markers — small vertical tick marks on the step curve.
        if is_km:
            _status = trace.status or [""] * len(trace.x)
            _cens_idx = [i for i, s in enumerate(_status) if s.lower() == "censored"]
            if _cens_idx:
                fig.add_trace(go.Scatter(
                    x=[float(trace.x[i]) for i in _cens_idx],
                    y=[float(trace.y[i]) for i in _cens_idx],
                    mode="markers",
                    marker=dict(symbol="line-ns", size=10,
                                color=trace.color_hex, opacity=0.7,
                                line=dict(width=2, color=trace.color_hex)),
                    customdata=[[point_ids[i], "censored"] for i in _cens_idx],
                    name=f"_pv_cens_{trace.series}", showlegend=False,
                    hovertemplate=("%{fullData.legendgroup} (censored)"
                                   "<extra>%{customdata[0]}</extra>"),
                    legendgroup=trace.series, visible=plot_visible,
                ))

        # Clickable caps at error-bar endpoints (skip for box — whiskers drawn above).
        if trace.has_err.any() and not is_box:
            _u_idx = [int(i) for i in range(len(trace.x)) if trace.has_upper[i]]
            _l_idx = [int(i) for i in range(len(trace.x)) if trace.has_lower[i]]
            if is_horiz:
                _u_x = [float(trace.x[i] + trace.err_array_plus[i]) for i in _u_idx]
                _u_y = [float(trace.y[i]) for i in _u_idx]
                _l_x = [float(trace.x[i] - trace.err_array_minus[i]) for i in _l_idx]
                _l_y = [float(trace.y[i]) for i in _l_idx]
                _u_sym, _l_sym = "triangle-right", "triangle-left"
            else:
                _u_x = [float(trace.x[i]) for i in _u_idx]
                _u_y = [float(trace.y[i] + trace.err_array_plus[i]) for i in _u_idx]
                _l_x = [float(trace.x[i]) for i in _l_idx]
                _l_y = [float(trace.y[i] - trace.err_array_minus[i]) for i in _l_idx]
                _u_sym, _l_sym = "triangle-up", "triangle-down"
            fig.add_trace(go.Scatter(
                x=_u_x, y=_u_y, mode="markers",
                marker=dict(symbol=_u_sym, size=7,
                            color=trace.color_hex, opacity=0.4,
                            line=dict(width=0)),
                customdata=[[point_ids[i], "upper"] for i in _u_idx],
                name=f"_pv_cap_u_{trace.series}", showlegend=False,
                hovertemplate=("%{fullData.legendgroup} upper"
                               "<extra>%{customdata[0]}</extra>"),
                legendgroup=trace.series, visible=plot_visible,
            ))
            fig.add_trace(go.Scatter(
                x=_l_x, y=_l_y, mode="markers",
                marker=dict(symbol=_l_sym, size=7,
                            color=trace.color_hex, opacity=0.4,
                            line=dict(width=0)),
                customdata=[[point_ids[i], "lower"] for i in _l_idx],
                name=f"_pv_cap_l_{trace.series}", showlegend=False,
                hovertemplate=("%{fullData.legendgroup} lower"
                               "<extra>%{customdata[0]}</extra>"),
                legendgroup=trace.series, visible=plot_visible,
            ))

    # Selection-highlight traces — always present but start empty; updated
    # in-place by _push_overlay_selection_to_widget on click. Horizontal
    # intervals run along x, so the endpoint markers point left/right there.
    # _pv_sel_anchor marks the anchor point (the typed-edit target) with a
    # larger ring when more than one point is selected.
    _sel_specs = (
        (("_pv_sel_center", "circle-open", 22),
         ("_pv_sel_upper",  "triangle-right-open", 16),
         ("_pv_sel_lower",  "triangle-left-open", 16),
         ("_pv_sel_anchor", "circle-open", 30))
        if is_horiz else
        (("_pv_sel_center", "circle-open", 22),
         ("_pv_sel_upper",  "triangle-up-open", 16),
         ("_pv_sel_lower",  "triangle-down-open", 16),
         ("_pv_sel_anchor", "circle-open", 30))
    )
    for _sn, _sym, _sz in _sel_specs:
        fig.add_trace(go.Scatter(
            x=[], y=[], mode="markers",
            marker=dict(symbol=_sym, size=_sz, color="#ff6600",
                        line=dict(color="#ff6600", width=3)),
            name=_sn, showlegend=False, hoverinfo="skip",
        ))

    if x_log:
        x_lo, x_hi = sorted([x_left, x_right])
        x_tv, x_tt = _log_axis_ticks(x_lo, x_hi, x_log)
        fig.update_xaxes(type="log",
                         range=[float(np.log10(x_lo)), float(np.log10(x_hi))],
                         title=_log_axis_title("X", x_log),
                         tickvals=x_tv, ticktext=x_tt)
    else:
        fig.update_xaxes(range=sorted([x_left, x_right]), title="X")
    if y_log:
        y_lo, y_hi = sorted([y_bottom, y_top])
        y_tv, y_tt = _log_axis_ticks(y_lo, y_hi, y_log)
        fig.update_yaxes(type="log",
                         range=[float(np.log10(y_lo)), float(np.log10(y_hi))],
                         title=_log_axis_title("Y", y_log),
                         tickvals=y_tv, ticktext=y_tt)
    elif is_forest:
        # Categorical (row-index) axis — the source image already labels rows,
        # so hide the numeric ticks to avoid a misleading second scale.
        fig.update_yaxes(range=sorted([y_bottom, y_top]),
                         title="", showticklabels=False)
    else:
        fig.update_yaxes(range=sorted([y_bottom, y_top]), title="Y")
    fig.update_layout(
        height=height,
        autosize=True,
        margin=dict(l=50, r=20, t=20, b=50),
        showlegend=True,
        legend=dict(x=1.02, y=1, xanchor="left"),
        plot_bgcolor="white",
    )
    return fig


def build_zoom_bubble_figure(
    img_rgb: np.ndarray,
    cal: dict,
    pt,
    part: str,
    *,
    image_data_uri: Optional[str] = None,
    height: int = 260,
    plot_type: str = "time_series",
    orientation: str = "vertical",
) -> go.Figure:
    """Small zoomed inset centred on the selected point or error-bar cap.

    All four data traces carry the ``_bub_`` prefix so the server can update
    them in-place without rebuilding the figure on every arrow-key press.

    Trace inventory (in order):
      _bub_pt      — selected point + error bar (series colour)
      _bub_sel     — orange open marker at the focused position
      _bub_vline   — vertical dotted crosshair at focus_x
      _bub_hline   — horizontal dotted crosshair at focus_y
    """
    h_img, w_img = img_rgb.shape[:2]
    x_log = cal.get("x_log_base")
    y_log = cal.get("y_log_base")
    is_horiz = is_horizontal_layout(plot_type, orientation)

    x_c = float(pt.x)
    y_c = float(pt.y)
    upper = (float(pt.y_err_upper)
             if pt.y_err_upper is not None and np.isfinite(pt.y_err_upper) else None)
    lower = (float(pt.y_err_lower)
             if pt.y_err_lower is not None and np.isfinite(pt.y_err_lower) else None)
    _raw_color = getattr(pt, "color_hex", FALLBACK_HEX)
    color = _raw_color if is_valid_hex(_raw_color) else FALLBACK_HEX

    # A one-sided interval collapses its missing bound onto the point estimate
    # so the error bar and band still render, spanning point → bound.
    has_iv = upper is not None or lower is not None
    v_c = x_c if is_horiz else y_c
    eff_upper = upper if upper is not None else v_c
    eff_lower = lower if lower is not None else v_c

    if is_horiz:
        # Interval endpoints are x-values on a fixed row.
        if part == "upper" and upper is not None:
            focus_x, focus_y = upper, y_c
        elif part == "lower" and lower is not None:
            focus_x, focus_y = lower, y_c
        else:
            focus_x, focus_y = x_c, y_c
    elif part == "upper" and upper is not None:
        focus_x, focus_y = x_c, upper
    elif part == "lower" and lower is not None:
        focus_x, focus_y = x_c, lower
    else:
        focus_x, focus_y = x_c, y_c

    # Zoom radius in plot-coordinate space (log10 units for log axes).
    def _plot(val, log_base):
        return float(np.log10(val)) if (log_base and val > 0) else float(val)

    focus_xp = _plot(focus_x, x_log)
    focus_yp = _plot(focus_y, y_log)

    # In horizontal layouts the interval brackets the value axis (x); otherwise y.
    if has_iv:
        if is_horiz:
            err_w_plot = abs(_plot(eff_upper, x_log) - _plot(eff_lower, x_log))
            err_h_plot = None
        else:
            err_h_plot = abs(_plot(eff_upper, y_log) - _plot(eff_lower, y_log))
            err_w_plot = None
    else:
        err_h_plot = None
        err_w_plot = None

    # Full image extent in data coords — used for zoom radii, crosshairs, and
    # background image positioning.
    x_left_d, _ = px_to_data(0, 0, cal)
    x_right_d, _ = px_to_data(w_img, 0, cal)
    _, y_top_d = px_to_data(0, 0, cal)
    _, y_bot_d = px_to_data(0, h_img, cal)
    x_data_lo = min(x_left_d, x_right_d)
    x_data_hi = max(x_left_d, x_right_d)
    y_data_lo = min(y_top_d, y_bot_d)
    y_data_hi = max(y_top_d, y_bot_d)

    # Separate x and y zoom radii so the window is never blown out when the
    # two axes have very different numeric scales.
    x_full = abs(_plot(x_right_d, x_log) - _plot(x_left_d, x_log))
    y_full = abs(_plot(y_top_d, y_log) - _plot(y_bot_d, y_log))
    if is_horiz:
        # Interval runs along x; frame it horizontally and keep a few rows in view.
        y_zoom_r = y_full * 0.05
        if err_w_plot is not None and err_w_plot > 0:
            x_zoom_r = err_w_plot * 0.8
        else:
            x_zoom_r = x_full * 0.05
    else:
        x_zoom_r = x_full * 0.05
        if err_h_plot is not None and err_h_plot > 0:
            y_zoom_r = err_h_plot * 0.8
        else:
            y_zoom_r = y_full * 0.05

    x_lo_p, x_hi_p = focus_xp - x_zoom_r, focus_xp + x_zoom_r
    y_lo_p, y_hi_p = focus_yp - y_zoom_r, focus_yp + y_zoom_r

    # Background image (same positioning logic as build_data_overlay_figure).
    source = image_data_uri if image_data_uri is not None else Image.fromarray(img_rgb)
    img_x = min(x_left_d, x_right_d)
    img_y = max(y_top_d, y_bot_d)
    img_sx = abs(x_right_d - x_left_d)
    img_sy = abs(y_top_d - y_bot_d)
    if x_log and x_left_d > 0 and x_right_d > 0:
        img_x = float(np.log10(min(x_left_d, x_right_d)))
        img_sx = float(abs(np.log10(x_right_d) - np.log10(x_left_d)))
    if y_log and y_top_d > 0 and y_bot_d > 0:
        img_y = float(np.log10(max(y_top_d, y_bot_d)))
        img_sy = float(abs(np.log10(y_top_d) - np.log10(y_bot_d)))

    fig = go.Figure()
    fig.add_layout_image(dict(
        source=source, xref="x", yref="y",
        x=img_x, y=img_y, sizex=img_sx, sizey=img_sy,
        xanchor="left", yanchor="top",
        sizing="stretch", opacity=1.0, layer="below",
    ))

    # Selected point with error bar — horizontal layouts put the interval on x,
    # vertical otherwise (interval on y).
    err_x_dict = None
    err_y_dict = None
    if has_iv:
        _bar = dict(
            type="data", symmetric=False,
            array=[max(0.0, eff_upper - v_c)],
            arrayminus=[max(0.0, v_c - eff_lower)],
            color=color, thickness=2, width=8,
        )
        if is_horiz:
            err_x_dict = _bar
        else:
            err_y_dict = _bar
    fig.add_trace(go.Scatter(
        x=[x_c], y=[y_c], mode="markers",
        marker=dict(color=color, size=12,
                    line=dict(color="rgba(0,0,0,0.5)", width=0.5)),
        error_x=err_x_dict,
        error_y=err_y_dict,
        name="_bub_pt", showlegend=False, hoverinfo="skip",
    ))

    # Orange selection highlight at the focused position.
    sym_map = (
        {"upper": "triangle-right-open", "lower": "triangle-left-open"}
        if is_horiz else
        {"upper": "triangle-up-open", "lower": "triangle-down-open"}
    )
    sel_sym = sym_map.get(part, "circle-open")
    fig.add_trace(go.Scatter(
        x=[focus_x], y=[focus_y], mode="markers",
        marker=dict(symbol=sel_sym, size=24, color="#ff6600",
                    line=dict(color="#ff6600", width=2.5)),
        name="_bub_sel", showlegend=False, hoverinfo="skip",
    ))

    # Dotted crosshairs — span the full image extent and get clipped by zoom.
    _xhair = dict(color="rgba(255,102,0,0.55)", width=1, dash="dot")
    fig.add_trace(go.Scatter(
        x=[focus_x, focus_x], y=[y_data_lo, y_data_hi],
        mode="lines", line=_xhair,
        name="_bub_vline", showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[x_data_lo, x_data_hi], y=[focus_y, focus_y],
        mode="lines", line=_xhair,
        name="_bub_hline", showlegend=False, hoverinfo="skip",
    ))

    # Filled ribbon band showing the confidence interval at the selected point.
    # Horizontal: band spanning [lower, upper] in x, thin in y (rows).
    # Otherwise: vertical band spanning [lower, upper] in y, thin in x.
    if has_iv:
        h_str = color.lstrip("#")
        fill_rgba = "rgba({},{},{},0.25)".format(
            int(h_str[0:2], 16), int(h_str[2:4], 16), int(h_str[4:6], 16))
        if is_horiz:
            bh = y_zoom_r * 0.12
            rib_x = [eff_lower, eff_upper, eff_upper, eff_lower, eff_lower]
            rib_y = [y_c - bh,  y_c - bh,  y_c + bh,  y_c + bh,  y_c - bh]
        else:
            bw = x_zoom_r * 0.12
            rib_x = [x_c - bw,  x_c - bw,  x_c + bw,  x_c + bw,  x_c - bw]
            rib_y = [eff_lower, eff_upper, eff_upper, eff_lower, eff_lower]
        fig.add_trace(go.Scatter(
            x=rib_x, y=rib_y,
            mode="lines", fill="toself", fillcolor=fill_rgba,
            line=dict(width=0),
            name="_bub_ribbon", showlegend=False, hoverinfo="skip",
        ))

    x_axis = dict(
        type="log" if x_log else "linear",
        range=[x_lo_p, x_hi_p],
        showgrid=True, gridcolor="rgba(0,0,0,0.12)",
        tickfont=dict(size=9), showticklabels=True,
        **(dict(zip(("tickvals", "ticktext"),
                    _log_axis_ticks(10**x_lo_p, 10**x_hi_p, x_log)))
           if x_log else {}),
    )
    y_axis = dict(
        type="log" if y_log else "linear",
        range=[y_lo_p, y_hi_p],
        showgrid=True, gridcolor="rgba(0,0,0,0.12)",
        tickfont=dict(size=9), showticklabels=True,
        **(dict(zip(("tickvals", "ticktext"),
                    _log_axis_ticks(10**y_lo_p, 10**y_hi_p, y_log)))
           if y_log else {}),
    )
    fig.update_layout(
        xaxis=x_axis, yaxis=y_axis,
        height=height,
        width=280,
        autosize=False,
        margin=dict(l=42, r=8, t=8, b=32),
        plot_bgcolor="white",
        showlegend=False,
        dragmode="pan",
    )
    return fig


# Re-exported for callers that want the inverse transform.
__all__ = [
    "ANCHOR_COLORS",
    "ANCHOR_LABELS",
    "all_calibration_shapes",
    "band_shapes",
    "anchor_annotations",
    "anchor_shapes",
    "anchors_from_result",
    "annotations_to_anchors",
    "build_calibration_edit_figure",
    "build_data_overlay_figure",
    "build_zoom_bubble_figure",
    "cal_dict_from_result",
    "default_anchors_for_image",
    "enforce_anchor_constraints",
    "guide_line_traces",
    "shapes_to_anchors",
    "data_to_px",
    "px_to_data",
]
