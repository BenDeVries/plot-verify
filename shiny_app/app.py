"""Shiny single-image PlotVerify app (Milestone 3 of the shiny-plan).

Layout:

- Upload card (image + CSV) and an EasyOCR-availability banner at the top.
- ``Calibrate`` tab:
    * Main image with two draggable P1/P2 anchor circles.
    * Right column with three collapsible accordions: X/Y label bands,
      Calibration points, Manual Values.
    * Bottom accordion: detection settings + frame-detection warnings.
- ``Overlay`` tab:
    * Calibrated image with extracted data drawn in data coordinates.
    * Series-visibility toggles and ``Export updated CSV``.

The single source of truth is one ``PlotVerifyApp`` controller per session,
stored in ``state.app``. All non-trivial logic lives on that controller and
in ``plotverify_core`` / ``axis_pipeline``; this module only assembles UI
and reactive wiring.
"""
from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from shiny import App, reactive, render, req, ui
from shinywidgets import output_widget, render_widget

# Lightweight runtime tracing for diagnosing reactive cascades. Enable with
# PLOTVERIFY_TRACE=1 before launching `shiny run`. Each labelled effect logs
# its entry, so a runaway loop is visible as a tight repeating sequence.
_TRACE = os.environ.get("PLOTVERIFY_TRACE") == "1"
_log = logging.getLogger("plotverify.shiny")
if _TRACE:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def _trace(tag: str, **kw) -> None:
    if _TRACE:
        extras = " ".join(f"{k}={v}" for k, v in kw.items())
        _log.info("[%s] %s", tag, extras)


def _user_error(label: str, exc: BaseException) -> None:
    """Surface an unexpected exception to the user as a toast and trace it.

    Replaces the legacy `except Exception: pass` pattern that silently
    swallowed failures. Use only for unexpected errors — defensive catches
    around input bindings that fire before the UI is ready should call
    `_trace` directly instead so they don't spam toasts on every render.
    """
    _trace(f"{label}.error", error=repr(exc))
    try:
        ui.notification_show(f"{label}: {exc}", type="error", duration=8)
    except Exception:
        # Notification API can fail outside of a Shiny session (tests,
        # one-off scripts). Tracing already happened above.
        pass


def _safe_int(getter, default: int, label: str = "input") -> int:
    """Read an int-coercible Shiny input value, falling back if unbound.

    Shiny inputs raise during the brief window before they're bound to the
    UI; the band-slider readers were previously a six-deep try/except
    pyramid that silently defaulted. This helper does the same fallback
    but emits a trace so the fallback path is observable in trace mode.
    """
    try:
        return int(getter() or default)
    except Exception as exc:
        _trace(f"{label}.safe_int_fallback", default=default, error=repr(exc))
        return default


def _safe_float(getter, default: float, label: str = "input") -> float:
    """Float counterpart to `_safe_int`."""
    try:
        return float(getter() or default)
    except Exception as exc:
        _trace(f"{label}.safe_float_fallback", default=default, error=repr(exc))
        return default


def _safe_str(getter, default: str, label: str = "input") -> str:
    """String counterpart for `or`-able Shiny string inputs."""
    try:
        return getter() or default
    except Exception as exc:
        _trace(f"{label}.safe_str_fallback", default=default, error=repr(exc))
        return default


def _safe_bool(getter, default: bool, label: str = "input") -> bool:
    """Bool counterpart. Cannot use `or` — a real ``False`` must survive."""
    try:
        val = getter()
        return default if val is None else bool(val)
    except Exception as exc:
        _trace(f"{label}.safe_bool_fallback", default=default, error=repr(exc))
        return default


# ---------------------------------------------------------------------------
# Overlay selection model
# ---------------------------------------------------------------------------
# The selection reactive holds None or a dict:
#   {"pids": [pid, ...],   ordered, insertion order = click order
#    "anchor": pid,        last-clicked; target of typed edits + zoom bubble
#    "part": "center"|"upper"|"lower"}   meaningful only for single selection


def _sel_pids(sel) -> list:
    if not sel:
        return []
    return list(sel.get("pids") or [])


def _sel_anchor(sel):
    if not sel:
        return None
    return sel.get("anchor")


def _sel_part(sel) -> str:
    if not sel or len(_sel_pids(sel)) > 1:
        return "center"
    return sel.get("part", "center")


def _update_selection(sel, pid, part, shift):
    """Pure reducer for overlay point clicks.

    Plain click replaces the selection; shift-click toggles the pid in the
    set (the anchor falls back to the most recent remaining pid when the
    anchor itself is removed). ``part`` is only kept for single selections.
    """
    if not shift or not sel:
        return {"pids": [pid], "anchor": pid, "part": part}
    pids = _sel_pids(sel)
    if pid in pids:
        pids = [p for p in pids if p != pid]
        if not pids:
            return None
        anchor = _sel_anchor(sel)
        if anchor == pid or anchor not in pids:
            anchor = pids[-1]
        return {"pids": pids, "anchor": anchor,
                "part": part if len(pids) == 1 else "center"}
    pids = pids + [pid]
    return {"pids": pids, "anchor": pid,
            "part": part if len(pids) == 1 else "center"}


from axis_pipeline import (
    CalibrationConfig,
    detect_axis_frame,
    ocr_available,
    render_overlay,
    x_label_band,
    y_label_band,
)
from plotverify_core import (
    Anchors,
    PlotVerifyApp,
    build_masked_overlay_image,
    build_overlay_traces,
    hex_to_bgr,
    is_horizontal_layout,
    px_to_data,
)
from plotverify_core.dashboard import (
    VALID_ERROR_TYPES,
    build_forest_display_df,
    build_time_series_display_df,
    compute_scatter_stats,
)

from .figures import (
    ANCHOR_LABELS,
    anchor_annotations,
    anchors_from_result,
    annotations_to_anchors,
    band_shapes,
    build_calibration_edit_figure,
    build_data_overlay_figure,
    build_zoom_bubble_figure,
    cal_dict_from_result,
    default_anchors_for_image,
    encode_image_data_uri,
    enforce_anchor_constraints,
    guide_line_traces,
)
from .edit_logic import PointVals, apply_nudge, half_width_of, linked_bounds
from .runtime_flags import json_only_mode
from .user_manual import user_manual_tab

# Evaluated once at import: the UI tree is built when the module loads, so a
# mid-session env change cannot (and should not) retoggle the mode.
_JSON_ONLY = json_only_mode()

# First tab in the navbar — used as the fallback before `main_nav` binds, so
# render guards ("only draw when my tab is active") pass on the initial load.
_DEFAULT_NAV = "Overlay" if _JSON_ONLY else "Calibrate"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def _log_base_to_str(b: object) -> str:
    if b is None:
        return "10"
    if abs(float(b) - math.e) < 1e-10:
        return "e"
    return str(b)


def _parse_log_base(checked: object, base_str: object) -> object:
    if not checked:
        return None
    s = (base_str or "10").strip().lower()
    if s == "e":
        return math.e
    try:
        v = float(s)
        return v if v > 1.0 else None
    except ValueError:
        return None


def _anchor_inputs(prefix: str, label_x: str, label_y: str,
                   include_data_x: bool, include_data_y: bool,
                   *, px_x: float, px_y: float,
                   data_x: float, data_y: float) -> ui.TagList:
    """Render two pixel inputs + up-to-two data inputs for one anchor.

    Defaults are passed in so the inputs mount with the current anchors_rv
    values; without this they default to 0 and the input→anchor reactive
    immediately overwrites the anchors back to (0, 0, 0).
    """
    px_row = ui.row(
        ui.column(6, ui.input_numeric(f"{prefix}_px_x", "pixel x",
                                        value=round(float(px_x), 2), step=1)),
        ui.column(6, ui.input_numeric(f"{prefix}_px_y", "pixel y",
                                        value=round(float(px_y), 2), step=1)),
    )
    data_widgets = []
    if include_data_x:
        data_widgets.append(ui.column(6, ui.input_numeric(
            f"{prefix}_data_x", label_x, value=float(data_x))))
    if include_data_y:
        data_widgets.append(ui.column(6, ui.input_numeric(
            f"{prefix}_data_y", label_y, value=float(data_y))))
    return ui.TagList(
        ui.tags.strong(prefix.upper()),
        px_row,
        ui.row(*data_widgets) if data_widgets else None,
    )


def _calibration_tab() -> ui.Tag:
    return ui.nav_panel(
        "Calibrate",
        ui.layout_columns(
            ui.card(
                ui.card_header(
                    ui.div(
                        ui.div(
                            "Calibration image — drag P1/P2 onto the correct tick positions "
                            "(click an anchor, then arrow keys for ±1px; Shift = 10px)",
                            style="flex: 1 1 auto; min-width: 0;",
                        ),
                        ui.div(
                            ui.input_action_button("run_detection", "Run detection",
                                                    class_="btn-primary"),
                            ui.input_action_button("detect_frame", "Detect axis frame"),
                            ui.input_action_button("reset_anchors", "Reset anchors"),
                            style="display:flex; gap:6px; flex-shrink: 0;",
                        ),
                        style="display:flex; flex-wrap: wrap; gap: 8px; align-items: center;",
                    )
                ),
                output_widget("cal_plot"),
                ui.div(ui.output_ui("cal_summary"), style="padding:6px 12px;"),
                full_screen=True,
                min_height="660px",
            ),
            ui.div(
                ui.accordion(
                    ui.accordion_panel(
                        "X/Y label bands",
                        ui.output_ui("bands_panel"),
                        value="bands",
                    ),
                    ui.accordion_panel(
                        "Calibration points",
                        ui.output_ui("calib_points_panel"),
                        value="calib_points",
                    ),
                    ui.accordion_panel(
                        "Manual Values",
                        ui.output_ui("manual_values_panel"),
                        value="manual_values",
                    ),
                    ui.accordion_panel(
                        "Series colors",
                        ui.output_ui("series_color_panel"),
                        value="series_colors",
                    ),
                    ui.accordion_panel(
                        "Plot type",
                        ui.input_select(
                            "plot_type_select",
                            None,
                            {
                                "time_series": "Time series w/ intervals",
                                "scatter": "Scatter plots",
                                "forest": "Forest plot",
                                "bar": "Bar chart",
                                "box": "Box plot",
                                "kaplan_meier": "Kaplan-Meier",
                            },
                        ),
                        value="plot_type",
                    ),
                    id="right_accordion",
                    open=["bands", "manual_values", "series_colors"],
                    multiple=True,
                ),
            ),
            col_widths=(8, 4),
        ),
        ui.accordion(
            ui.accordion_panel(
                "Detection settings",
                ui.row(
                    ui.column(4, ui.input_numeric("cfg_min_ocr_conf",
                                                    "Min OCR confidence", value=0.20,
                                                    min=0.0, max=1.0, step=0.05)),
                ),
            ),
            ui.accordion_panel(
                "Frame-detection warnings",
                ui.output_ui("warnings_panel"),
            ),
            id="bottom_accordion",
            open=[],
            multiple=True,
        ),
    )


def _overlay_tab() -> ui.Tag:
    return ui.nav_panel(
        "Overlay",
        ui.layout_columns(
            ui.card(
                ui.card_header("Data overlay"),
                output_widget("overlay_plot"),
                full_screen=True,
                min_height="660px",
            ),
            ui.card(
                ui.card_header("Controls & export"),
                ui.card_body(
                    ui.accordion(
                        ui.accordion_panel(
                            "Series",
                            ui.output_ui("series_visibility_panel"),
                            value="series",
                        ),
                        ui.accordion_panel(
                            "Edit a point",
                            ui.output_ui("edit_point_panel"),
                            value="edit_point",
                        ),
                        ui.accordion_panel(
                            "Export",
                            ui.input_text("export_filename", "Filename",
                                           value="corrected.csv"),
                            ui.input_checkbox("include_audit_cols",
                                               "Include audit columns", value=False),
                            ui.download_button("export_csv", "Export updated CSV"),
                            ui.download_button("export_json", "Export as JSON"),
                            value="export",
                        ),
                        id="overlay_controls_accordion",
                        open=["series", "edit_point"],
                        multiple=True,
                    ),
                    fillable=False,
                    padding=0,
                ),
            ),
            col_widths=(8, 4),
        ),
        ui.output_ui("dashboard_panel"),
        # Floating zoom bubble — position:fixed so it hovers over the page
        # regardless of scroll position. Visibility is toggled via the
        # pv_bubble_show custom message handler in _ANCHOR_KEY_SCRIPT.
        ui.div(
            ui.div(
                "⠿ Zoom preview",
                id="pv-zoom-handle",
                style=(
                    "padding:3px 8px; font-size:11px; color:#666; "
                    "border-bottom:1px solid #e0e0e0; user-select:none; "
                    "text-align:center; letter-spacing:0.04em;"
                ),
            ),
            output_widget("zoom_bubble"),
            id="pv-zoom-bubble",
            style=(
                "position:fixed; top:80px; right:10px; width:310px; "
                "z-index:9000; border-radius:10px; "
                "border:1.5px solid rgba(0,0,0,0.14); "
                "box-shadow:0 8px 28px rgba(0,0,0,0.18); "
                "background:rgba(255,255,255,0.97); "
                "padding:0; overflow:hidden; "
                "opacity:0; transform:scale(0.95); "
                # visibility (not just pointer-events) — Plotly sets inline
                # `pointer-events:all` on its drag-layer rects, which would
                # otherwise keep hit-testing the hidden bubble and swallow
                # clicks on the accordion headers beneath it.
                "visibility:hidden; "
                "transition:opacity 0.12s ease, transform 0.12s ease, "
                "visibility 0.12s; "
                "pointer-events:none;"
            ),
        ),
    )


def _safe_series_token(series_name: str) -> str:
    """Sanitize a series name for use as a Shiny input ID suffix."""
    return "".join(c if (c.isalnum() or c == "_") else "_" for c in series_name)


def _vis_id(series_name: str) -> str:
    """Return a valid Shiny input ID for a series visibility checkbox."""
    return "vis_" + _safe_series_token(series_name)


def _color_id(series_name: str) -> str:
    """Return a valid Shiny input ID for a per-series color picker."""
    return "col_" + _safe_series_token(series_name)


def _delta_e_id(series_name: str) -> str:
    """Return a valid Shiny input ID for a per-series ΔE threshold field."""
    return "de_" + _safe_series_token(series_name)


BG_COLOR_INPUT_ID = "pv_bg_color"


def _mask_id(series_name: str) -> str:
    """Return a valid Shiny input ID for a per-series mask-preview toggle."""
    return "mask_" + _safe_series_token(series_name)


DEFAULT_DELTA_E = 10
DELTA_E_MIN = 1
DELTA_E_MAX = 100


def _coerce_delta_e(raw) -> Optional[int]:
    """Clamp a raw ΔE field value into range; None when it is not a number.

    ``input_numeric`` yields None while the field is empty mid-edit, which must
    leave the stored threshold untouched rather than snapping it to a default.
    """
    try:
        val = int(round(float(raw)))
    except (TypeError, ValueError, OverflowError):
        return None
    return max(DELTA_E_MIN, min(DELTA_E_MAX, val))


_ANCHOR_KEY_SCRIPT = """<style>
/* Per-series ΔE field: strip the label/margin so it sits on the series row */
.pv-de-cell .shiny-input-container { margin-bottom: 0; }
.pv-de-cell label { display: none; }
.pv-de-cell input { text-align: center; padding: 2px 4px; font-size: 13px; }

/* Allow file-upload progress text to wrap rather than clip in the sidebar */
.shiny-file-input-progress {
    overflow: visible;
    height: auto;
}
.shiny-file-input-progress .progress-bar {
    overflow: visible;
    white-space: normal;
    height: auto;
    min-height: 1.25rem;
    line-height: 1.25rem;
    padding: 2px 4px;
}
</style>
<script>
(function() {
    function log(m) { try { console.log('[plotverify] ' + m); } catch (e) {} }
    log('anchor key script loaded');

    // Set to true when a valid overlay data-point click is forwarded to Shiny;
    // cleared on every mousedown inside #overlay_plot.  The document mousedown
    // handler uses this to distinguish "clicked a point" from "clicked background".
    var pvOverlayClickHandled = false;

    function isFormTarget(t) {
        if (!t) return false;
        return t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' ||
               t.tagName === 'SELECT' || t.isContentEditable;
    }

    // Robust arrow detection: ``e.key`` is the modern path but some older
    // contexts (and some keyboard layouts) only populate ``e.keyCode``.
    function detectArrow(e) {
        if (e.key === 'ArrowUp'    || e.keyCode === 38) return 'ArrowUp';
        if (e.key === 'ArrowDown'  || e.keyCode === 40) return 'ArrowDown';
        if (e.key === 'ArrowLeft'  || e.keyCode === 37) return 'ArrowLeft';
        if (e.key === 'ArrowRight' || e.keyCode === 39) return 'ArrowRight';
        return null;
    }

    // Arrow-key coalescing: every keydown (including OS key-repeat while a
    // key is held) accumulates into step counters; a leading-edge send keeps
    // single taps instant while the trailing-edge timer batches held keys
    // into ~1 message per FLUSH_MS. The Shift x10 multiplier is applied at
    // accumulate time so mixed shift/plain presses in one window stay exact.
    var pvNudgeAcc = {dx: 0, dy: 0, timer: null, lastSend: 0};
    var PV_NUDGE_FLUSH_MS = 60;

    function flushNudge() {
        pvNudgeAcc.timer = null;
        if (!pvNudgeAcc.dx && !pvNudgeAcc.dy) return;
        if (typeof Shiny === 'undefined' || !Shiny.setInputValue) {
            log('Shiny.setInputValue not available');
            pvNudgeAcc.dx = 0; pvNudgeAcc.dy = 0;
            return;
        }
        Shiny.setInputValue(
            'anchor_nudge',
            {dx: pvNudgeAcc.dx, dy: pvNudgeAcc.dy, ts: Date.now()},
            {priority: 'event'}
        );
        log('nudge sent: dx=' + pvNudgeAcc.dx + ' dy=' + pvNudgeAcc.dy);
        pvNudgeAcc.dx = 0;
        pvNudgeAcc.dy = 0;
        pvNudgeAcc.lastSend = Date.now();
    }

    function onKey(e) {
        if ((e.key === 'Escape' || e.keyCode === 27) && !isFormTarget(e.target)) {
            if (typeof Shiny !== 'undefined' && Shiny.setInputValue) {
                Shiny.setInputValue('overlay_deselect', {ts: Date.now()}, {priority: 'event'});
            }
            return;
        }
        var arrow = detectArrow(e);
        if (!arrow) return;
        if (isFormTarget(e.target)) { log('form target, ignoring'); return; }
        e.preventDefault();
        e.stopPropagation();
        var m = e.shiftKey ? 10 : 1;
        if (arrow === 'ArrowRight')      pvNudgeAcc.dx += m;
        else if (arrow === 'ArrowLeft')  pvNudgeAcc.dx -= m;
        else if (arrow === 'ArrowUp')    pvNudgeAcc.dy += m;
        else                             pvNudgeAcc.dy -= m;
        if (Date.now() - pvNudgeAcc.lastSend > PV_NUDGE_FLUSH_MS) {
            flushNudge();
        } else if (!pvNudgeAcc.timer) {
            pvNudgeAcc.timer = setTimeout(flushNudge, PV_NUDGE_FLUSH_MS);
        }
    }

    // Attach in capture phase to every plausible scope so nothing can
    // intercept the event before us.
    window.addEventListener('keydown', onKey, true);
    document.addEventListener('keydown', onKey, true);
    if (document.body) document.body.addEventListener('keydown', onKey, true);
    log('keydown listeners attached (window, document, body)');

    // ``plotly_clickannotation`` → server input.anchor_selected.
    // ``plotly_click`` on a data point → server input.overlay_click.
    // Setting pvOverlayClickHandled=true on a valid point click lets the
    // document mousedown handler (below) know NOT to send overlay_deselect.
    function attachClickHandlers() {
        document.querySelectorAll('.js-plotly-plot').forEach(function(plot) {
            if (plot._pvAnchorAttached) return;
            plot._pvAnchorAttached = true;

            plot.on('plotly_clickannotation', function(ev) {
                var name = ev && ev.annotation && ev.annotation.name;
                if (name === 'P1' || name === 'P2' || name === 'P3') {
                    log('annotation clicked: ' + name);
                    var act = document.activeElement;
                    if (act && act !== document.body && act.blur) {
                        act.blur();
                        log('blurred ' + act.tagName + '#' + (act.id || ''));
                    }
                    if (typeof Shiny !== 'undefined' && Shiny.setInputValue) {
                        Shiny.setInputValue(
                            'anchor_selected',
                            {name: name, ts: Date.now()},
                            {priority: 'event'}
                        );
                    }
                }
            });

            var isOverlayPlot = !!(plot.closest && plot.closest('#overlay_plot'));
            plot.on('plotly_click', function(data) {
                var act = document.activeElement;
                if (act && act !== document.body && act.blur) {
                    act.blur();
                    log('blurred on plotly_click');
                }
                if (!isOverlayPlot) return;
                if (data && data.points && data.points.length > 0) {
                    var cd = data.points[0].customdata;
                    if (cd && cd.length >= 1) {
                        var pid = cd[0];
                        if (typeof pid === 'string' && pid.indexOf('#') !== -1) {
                            // Only cap traces tag a part; the forest main
                            // scatter puts a status note in cd[1], so restrict
                            // to the exact 'upper'/'lower' sentinels.
                            var part = (cd.length >= 2 && (cd[1] === 'upper' || cd[1] === 'lower')) ? cd[1] : 'center';
                            var shift = !!(data.event && data.event.shiftKey);
                            log('overlay click: ' + pid + ' part=' + part + ' shift=' + shift);
                            pvOverlayClickHandled = true;
                            if (typeof Shiny !== 'undefined' && Shiny.setInputValue) {
                                Shiny.setInputValue(
                                    'overlay_click',
                                    {pid: pid, part: part, shift: shift, ts: Date.now()},
                                    {priority: 'event'}
                                );
                            }
                        }
                    }
                }
            });

            // Legend entries and the "Overlay" checkboxes in the Series panel
            // are two views of one state. Forward the click so the server can
            // flip the matching checkbox; Plotly's own toggle still runs for
            // instant feedback and the rebuild lands on the same visibility.
            plot.on('plotly_legendclick', function(ev) {
                if (!isOverlayPlot) return;
                var tr = ev && ev.data && ev.data[ev.curveNumber];
                var series = tr && (tr.legendgroup || tr.name);
                if (!series) return;
                log('legend toggle: ' + series);
                if (typeof Shiny !== 'undefined' && Shiny.setInputValue) {
                    Shiny.setInputValue(
                        'overlay_legend_toggle',
                        {series: series, ts: Date.now()},
                        {priority: 'event'}
                    );
                }
            });

            // Double-click isolates one series client-side only, which would
            // leave the checkboxes describing a different figure.
            plot.on('plotly_legenddoubleclick', function() {
                return isOverlayPlot ? false : undefined;
            });

            // Box/lasso select → multi-selection. Dedupe pids because the
            // cap traces carry the same pid as the main scatter point.
            plot.on('plotly_selected', function(data) {
                if (!isOverlayPlot) return;
                if (!data || !data.points || !data.points.length) return;
                var seen = {};
                var pids = [];
                data.points.forEach(function(p) {
                    var cd = p.customdata;
                    var pid = cd && cd.length >= 1 ? cd[0] : null;
                    if (typeof pid === 'string' && pid.indexOf('#') !== -1 && !seen[pid]) {
                        seen[pid] = true;
                        pids.push(pid);
                    }
                });
                if (!pids.length) return;
                log('overlay box select: ' + pids.length + ' points');
                pvOverlayClickHandled = true;
                if (typeof Shiny !== 'undefined' && Shiny.setInputValue) {
                    Shiny.setInputValue(
                        'overlay_box_select',
                        {pids: pids, ts: Date.now()},
                        {priority: 'event'}
                    );
                }
            });
        });
    }
    attachClickHandlers();
    new MutationObserver(attachClickHandlers).observe(
        document.body, {childList: true, subtree: true});

    // Deselect the overlay point when the user clicks anywhere inside
    // #overlay_plot that is NOT a data point.  We use a document-level
    // mousedown (capture phase) so it fires before Plotly's own handlers.
    // pvOverlayClickHandled is set true inside plotly_click when a valid pid
    // is forwarded; the setTimeout(0) fires after all synchronous click
    // handlers have run, so the flag is already set (or not) by then.
    document.addEventListener('mousedown', function(e) {
        var container = document.getElementById('overlay_plot');
        if (!container || !container.contains(e.target)) return;
        // Starting a box/lasso selection must not clear the selection.
        var gd = container.querySelector('.js-plotly-plot');
        var dm = gd && gd._fullLayout && gd._fullLayout.dragmode;
        if (dm === 'select' || dm === 'lasso') return;
        pvOverlayClickHandled = false;
        setTimeout(function() {
            if (!pvOverlayClickHandled) {
                log('overlay deselect (background mousedown)');
                if (typeof Shiny !== 'undefined' && Shiny.setInputValue) {
                    Shiny.setInputValue('overlay_deselect', {ts: Date.now()}, {priority: 'event'});
                }
            }
        }, 0);
    }, true);

    // Fix horizontal squishing on first Overlay tab view.
    // Root cause: when the tab pane is display:none, Plotly renders with
    // clientWidth=0 and sets gd.style.width to a small pixel value.  Later
    // Plotly.Plots.resize() reads getComputedStyle(gd).width, which returns
    // that same inline pixel value — so the resize is a no-op.
    // Fix: clear gd's inline width/height so the CSS cascade uses the
    // container's real width, then call relayout with autosize:true.
    (function() {
        function fixOverlayWidth() {
            var container = document.getElementById('overlay_plot');
            if (!container) { setTimeout(fixOverlayWidth, 200); return; }
            var lastW = 0;
            new ResizeObserver(function(entries) {
                var w = entries[0] ? entries[0].contentRect.width : 0;
                if (lastW < 50 && w > 50) {
                    // Query gd inside rAF so we get the element that actually
                    // exists at fire time — Shiny may have replaced the widget
                    // between the ResizeObserver callback and the rAF on fast
                    // local servers, making a pre-captured gd reference stale.
                    requestAnimationFrame(function() {
                        var gd = container.querySelector('.js-plotly-plot');
                        if (gd) {
                            gd.style.width  = '';
                            gd.style.height = '';
                            try {
                                window.Plotly && Plotly.Plots.resize(gd);
                            } catch(_) {}
                        }
                    });
                }
                lastW = w;
            }).observe(container);
        }
        fixOverlayWidth();
    })();

    // Show/hide the floating zoom bubble via server message. Toggle
    // visibility as well as opacity/pointer-events: Plotly's drag-layer
    // rects carry inline `pointer-events:all`, so a merely-transparent
    // bubble would still hit-test and swallow clicks on the accordion
    // headers beneath it.
    Shiny.addCustomMessageHandler('pv_bubble_show', function(msg) {
        var el = document.getElementById('pv-zoom-bubble');
        if (!el) return;
        if (msg.show) {
            el.style.visibility = 'visible';
            el.style.opacity = '1';
            el.style.transform = 'scale(1)';
            el.style.pointerEvents = 'auto';
        } else {
            el.style.opacity = '0';
            el.style.transform = 'scale(0.95)';
            el.style.pointerEvents = 'none';
            el.style.visibility = 'hidden';
        }
    });

    // Draggable zoom bubble — drag via the handle bar at the top.
    (function initDrag() {
        var el     = document.getElementById('pv-zoom-bubble');
        var handle = document.getElementById('pv-zoom-handle');
        if (!el || !handle) {
            // Retry once the elements exist.
            new MutationObserver(function(_, obs) {
                el     = document.getElementById('pv-zoom-bubble');
                handle = document.getElementById('pv-zoom-handle');
                if (el && handle) { obs.disconnect(); initDrag(); }
            }).observe(document.body, {childList: true, subtree: true});
            return;
        }
        var mx = 0, my = 0;
        handle.style.cursor = 'grab';
        handle.addEventListener('mousedown', function(e) {
            // Convert right/bottom to left/top so dragging works in all cases.
            var rect = el.getBoundingClientRect();
            el.style.left  = rect.left + 'px';
            el.style.top   = rect.top  + 'px';
            el.style.right  = '';
            el.style.bottom = '';
            mx = e.clientX; my = e.clientY;
            handle.style.cursor = 'grabbing';
            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup',   onUp);
            e.preventDefault();
        });
        function onMove(e) {
            el.style.left = (el.offsetLeft + e.clientX - mx) + 'px';
            el.style.top  = (el.offsetTop  + e.clientY - my) + 'px';
            mx = e.clientX; my = e.clientY;
        }
        function onUp() {
            handle.style.cursor = 'grab';
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup',   onUp);
        }
    })();

    // Per-series color picker bridge. Each <input class="pv-color-picker"
    // data-input-id="col_<series>"> writes its value to Shiny on every change
    // (and on the `input` event for live-updates while the picker is open).
    // No priority:event — colors are persistent state, not one-shot events.
    function bindColorPickers() {
        document.querySelectorAll('input.pv-color-picker').forEach(function(el) {
            if (el._pvBound) return;
            el._pvBound = true;
            function send() {
                var id = el.getAttribute('data-input-id');
                if (id && typeof Shiny !== 'undefined' && Shiny.setInputValue) {
                    Shiny.setInputValue(id, el.value);
                }
            }
            el.addEventListener('change', send);
            el.addEventListener('input', send);
            // Seed Shiny with the initial value so first-render reads work.
            send();
        });
    }
    bindColorPickers();
    new MutationObserver(bindColorPickers).observe(
        document.body, {childList: true, subtree: true});
})();
</script>"""



def _make_sidebar(json_only: Optional[bool] = None) -> ui.Sidebar:
    if json_only is None:
        json_only = _JSON_ONLY
    json_block = [
        ui.h5("Agent JSON"),
        ui.input_file("json_upload", "JSON file",
                        accept=[".json"], multiple=False),
        ui.input_text_area("json_paste", "or paste JSON", rows=3,
                            placeholder='{"schema_version": "1.1", ...}'),
        ui.input_action_button("json_apply", "Import JSON",
                                class_="btn-outline-primary btn-sm"),
    ]
    if json_only:
        # JSON-only surface: the JSON carries the image, data, calibration,
        # plot type and orientation — no separate uploads, no OCR banner.
        return ui.sidebar(
            *json_block,
            ui.tags.small(
                "Import an Agent JSON with an embedded image, calibration "
                "and data rows.",
                style="color:#666;",
            ),
            ui.hr(),
            ui.output_ui("session_status"),
            width=320,
        )
    return ui.sidebar(
        ui.h4("Upload"),
        ui.input_file("image_upload", "Plot image",
                        accept=[".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"],
                        multiple=False),
        ui.input_file("csv_upload", "Data CSV",
                        accept=[".csv"], multiple=False),
        ui.output_ui("ocr_banner"),
        ui.hr(),
        *json_block,
        ui.hr(),
        ui.output_ui("session_status"),
        width=320,
    )


def _make_ui(json_only: Optional[bool] = None) -> ui.Tag:
    if json_only is None:
        json_only = _JSON_ONLY
    panels = []
    if not json_only:
        panels.append(_calibration_tab())
    panels.append(_overlay_tab())
    panels.append(user_manual_tab())
    return ui.page_navbar(
        *panels,
        title="PlotVerify (Shiny)",
        id="main_nav",
        **({"selected": "Overlay"} if json_only else {}),
        sidebar=_make_sidebar(json_only),
        header=ui.HTML(_ANCHOR_KEY_SCRIPT),
    )


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def server(input, output, session):  # noqa: A002 (`input` is a Shiny convention)
    # Single app controller per session.
    pv = PlotVerifyApp()

    # Reactive state. We use coarse-grained reactives so figure rebuilds happen
    # only when the underlying file or anchors actually change.
    file_id_rv = reactive.value(None)         # active file_id
    anchors_rv = reactive.value(Anchors())    # current per-file anchors (mirror)
    cal_revision = reactive.value(0)          # bump after auto/manual calibration
    overlay_revision = reactive.value(0)      # bump after EditableOverlay edits
    csv_revision = reactive.value(0)          # bump after CSV add/replace
    show_diagnostic = reactive.value(False)   # diagnostic-overlay toggle (placeholder)
    selected_anchor = reactive.value(None)    # "P1"/"P2"/None — keyboard target
    axis_frame_rv = reactive.value(None)      # AxisFrame from detect_frame / auto-cal
    selected_overlay_rv = reactive.value(None)  # {pid, part} for overlay point selection

    # CRITICAL: calling `cal_revision()` inside a reactive effect subscribes
    # that effect to the value — so an effect that *both* reads and writes
    # the counter would invalidate itself and loop. These helpers wrap the
    # read in ``reactive.isolate`` so the bump is a pure write.
    def _bump_cal() -> None:
        with reactive.isolate():
            cur = cal_revision()
        cal_revision.set(cur + 1)

    def _bump_overlay() -> None:
        with reactive.isolate():
            cur = overlay_revision()
        overlay_revision.set(cur + 1)

    # Used to suppress reactive loops when we programmatically update the widget
    # in response to numeric input edits and vice versa.
    syncing = {"shapes_to_inputs": False, "inputs_to_shapes": False}

    # Per-file image data-URI cache. Encoding the PNG once and reusing the
    # data URI on every render is far cheaper than letting Plotly re-encode
    # the PIL source on each FigureWidget rebuild.
    image_uri_cache: dict[str, str] = {}
    # Keyed by (fid, mask_signature) where signature is a frozenset of
    # (series_name, color_hex, delta_e). Only series whose visibility is on
    # AND whose color is intentional contribute to the signature.
    masked_image_uri_cache: dict[tuple, str] = {}

    def _get_image_uri(fid: str) -> str:
        cached = image_uri_cache.get(fid)
        if cached is not None:
            return cached
        fs = pv.state.files.get(fid)
        if fs is None:
            raise KeyError(f"file_id {fid!r} not in state")
        t0 = time.perf_counter()
        uri = encode_image_data_uri(fs.image_rgb)
        _trace("encode_image", file_id=fid, ms=int((time.perf_counter() - t0) * 1000),
                bytes=len(uri))
        image_uri_cache[fid] = uri
        return uri

    def _compute_mask_specs(fs, df, mask_active: dict[str, bool]
                             ) -> list[tuple[str, str, int]]:
        """Collect (series, color_hex, delta_e) for series eligible for masking.

        A series contributes a mask only when its mask toggle is ON AND its
        color is intentional (CSV-provided or user-picked via the calibration
        tab). Auto-palette defaults are skipped — they have no relation to the
        actual pixel colors in the source image.
        """
        specs: list[tuple[str, str, int]] = []
        for name in df["series"].drop_duplicates().tolist():
            if not mask_active.get(name, False):
                continue
            if not fs.has_intentional_color(name):
                continue
            color_hex = fs.series_color_overrides.get(name)
            if not color_hex:
                color_hex = str(df[df["series"] == name]
                                ["series_color"].iloc[0])
            de = int(fs.series_delta_e.get(name, DEFAULT_DELTA_E))
            specs.append((str(name), color_hex, de))
        return specs

    def _get_masked_image_uri(fid: str, mask_specs: list[tuple[str, str, int]]
                               ) -> str:
        """Return a cached data URI for the ΔE-masked composite.

        ``mask_specs`` is a list of (series_name, color_hex, delta_e). The cache
        key is fid + a frozenset of (color_hex.lower(), delta_e) — series order
        and name do not change pixel output. Falls back to the unmasked URI
        when ``mask_specs`` is empty.
        """
        if not mask_specs:
            return _get_image_uri(fid)
        fs = pv.state.files.get(fid)
        if fs is None or fs.image_bgr is None:
            return _get_image_uri(fid)
        bg_hex = fs.background_color_override
        sig_inner = frozenset((c.lower(), int(d)) for _, c, d in mask_specs)
        key = (fid, sig_inner, bg_hex.lower())
        cached = masked_image_uri_cache.get(key)
        if cached is not None:
            return cached
        t0 = time.perf_counter()
        masked_rgb = build_masked_overlay_image(
            fs.image_bgr,
            [(c, d) for _, c, d in mask_specs],
            background_bgr=hex_to_bgr(bg_hex),
        )
        uri = encode_image_data_uri(masked_rgb)
        _trace("encode_masked_image", file_id=fid,
               n=len(mask_specs),
               ms=int((time.perf_counter() - t0) * 1000),
               bytes=len(uri))
        masked_image_uri_cache[key] = uri
        return uri

    # ------------------------------------------------------------------
    # Upload + file lifecycle
    # ------------------------------------------------------------------

    @reactive.effect
    @reactive.event(input.image_upload)
    def _on_image_upload():
        files = input.image_upload()
        if not files:
            return
        info = files[0]
        path = Path(info["datapath"])
        t0 = time.perf_counter()
        try:
            data = path.read_bytes()
            fid = pv.add_image(info["name"], data, downscale=True)
        except Exception as e:
            ui.notification_show(f"Image load failed: {e}", type="error")
            return
        _trace("decode_image", file_id=fid,
                ms=int((time.perf_counter() - t0) * 1000),
                bytes=len(data))
        file_id_rv.set(fid)
        pv.select(fid)
        fs = pv.active
        h, w = fs.image_rgb.shape[:2]
        # Precompute the image data URI so the first cal_plot render reuses
        # it; without this the first render pays the PNG-encoding cost.
        _get_image_uri(fid)
        new_anchors = default_anchors_for_image(w, h)
        # Auto-detect axis frame on upload so band shapes appear immediately
        # without a separate "Detect axis frame" click.
        detected_frame = None
        if pv.ocr_available:
            with ui.Progress(min=0, max=1) as p:
                p.set(0.1, message="Detecting axis frame…")
                try:
                    preview = detect_axis_frame(fs.image_bgr)
                    if preview.bbox is not None:
                        detected_frame = preview.bbox
                        new_anchors = Anchors(
                            p1_pixel=(float(preview.bbox.left),
                                      float(preview.bbox.bottom)),
                            p2_pixel=(float(preview.bbox.right),
                                      float(preview.bbox.bottom)),
                            p3_pixel=(float(preview.bbox.left),
                                      float(preview.bbox.top)),
                            p1_data_x=new_anchors.p1_data_x,
                            p2_data_x=new_anchors.p2_data_x,
                            p1_data_y=new_anchors.p1_data_y,
                            p3_data_y=new_anchors.p3_data_y,
                        )
                except Exception as exc:
                    _trace("upload_auto_detect.error", error=repr(exc))
                p.set(1.0)
        # Suppress the input→anchors echo while we set defaults: when the
        # manual-values panel first mounts its inputs default to 0.0, and
        # without this guard `_inputs_to_anchors` would overwrite the just-
        # placed anchors back to (0, 0, 0).
        syncing["shapes_to_inputs"] = True
        try:
            anchors_rv.set(new_anchors)
        finally:
            syncing["shapes_to_inputs"] = False
        selected_anchor.set(None)
        selected_overlay_rv.set(None)
        axis_frame_rv.set(detected_frame)
        _bump_cal()
        _bump_overlay()
        if fs.image_downscale_factor < 1.0:
            ui.notification_show(
                f"Large image downscaled by {fs.image_downscale_factor:.2f}× "
                "for display speed; auto-calibration runs against the "
                "downscaled image.",
                type="warning", duration=8,
            )

    @reactive.effect
    @reactive.event(input.csv_upload)
    def _on_csv_upload():
        files = input.csv_upload()
        if not files:
            return
        with reactive.isolate():
            fid = file_id_rv()
        if fid is None:
            ui.notification_show("Upload an image before the CSV.", type="warning")
            return
        info = files[0]
        try:
            csv_text = Path(info["datapath"]).read_text(encoding="utf-8")
            pv.add_csv(fid, info["name"], csv_text)
        except Exception as e:
            ui.notification_show(f"CSV load failed: {e}", type="error")
            return
        # The loader auto-switches to "forest" for forest-shaped CSVs — reflect
        # that in the Plot type control so the UI and state agree.
        fs = pv.state.files.get(fid)
        if fs is not None:
            ui.update_select("plot_type_select", selected=fs.plot_type)
        selected_overlay_rv.set(None)
        with reactive.isolate():
            _cur_csv = csv_revision()
        csv_revision.set(_cur_csv + 1)
        _bump_overlay()

    # ------------------------------------------------------------------
    # JSON import
    # ------------------------------------------------------------------

    @reactive.effect
    @reactive.event(input.json_apply)
    def _on_json_import():
        json_text = None
        files = input.json_upload()
        if files:
            try:
                json_text = Path(files[0]["datapath"]).read_text(encoding="utf-8")
            except Exception as e:
                ui.notification_show(f"JSON file read failed: {e}", type="error")
                return
        if json_text is None:
            json_text = (input.json_paste() or "").strip()
        if not json_text:
            ui.notification_show(
                "No JSON provided (upload a file or paste text).", type="warning",
            )
            return

        try:
            fid, result = pv.add_json(json_text)
        except Exception as e:
            ui.notification_show(f"JSON import failed: {e}", type="error")
            return

        for w in result.warnings:
            ui.notification_show(w, type="warning", duration=6)

        file_id_rv.set(fid)
        fs = pv.active
        if fs is not None:
            _get_image_uri(fid)
            if result.anchors is not None:
                syncing["shapes_to_inputs"] = True
                try:
                    anchors_rv.set(result.anchors)
                finally:
                    syncing["shapes_to_inputs"] = False
            if fs.plot_type:
                ui.update_select("plot_type_select", selected=fs.plot_type)

        selected_anchor.set(None)
        selected_overlay_rv.set(None)
        _bump_cal()
        _bump_overlay()

        with reactive.isolate():
            _cur_csv = csv_revision()
        csv_revision.set(_cur_csv + 1)

        has_cal = (
            fs is not None
            and fs.detection_result is not None
            and fs.detection_result.success
        )
        has_data = fs is not None and fs.csv_df is not None
        if _JSON_ONLY and not has_cal:
            # There is no Calibrate tab to recover with in JSON-only mode:
            # calibration must come from the JSON itself.
            ui.notification_show(
                "This deployment requires calibration in the JSON "
                "(axes.x / axes.y with two pixel/value pairs each). "
                "Re-export the JSON with an axes block and import again.",
                type="error", duration=12,
            )
            return
        if has_cal and has_data:
            ui.update_navs("main_nav", selected="Overlay")
            ui.notification_show(
                "JSON imported — calibration and data loaded.", type="message",
            )
        elif has_cal:
            ui.notification_show(
                "JSON imported — calibration loaded (no data rows).", type="message",
            )
        elif has_data:
            ui.notification_show(
                "JSON imported — data loaded (no calibration).", type="message",
            )
        else:
            ui.notification_show("JSON imported — image only.", type="message")

    # ------------------------------------------------------------------
    # Plot type
    # ------------------------------------------------------------------

    @reactive.effect
    @reactive.event(input.plot_type_select)
    def _on_plot_type_change():
        with reactive.isolate():
            fid = file_id_rv()
        fs = pv.state.files.get(fid) if fid else None
        if fs:
            fs.plot_type = input.plot_type_select()
        _bump_overlay()

    # ------------------------------------------------------------------
    # OCR banner + status
    # ------------------------------------------------------------------

    @render.ui
    def ocr_banner():
        if ocr_available():
            return ui.div(
                ui.tags.span("EasyOCR detected — auto-calibration available.",
                              style="color:#1b5e20;"),
                style="font-size:12px; margin-top:6px;",
            )
        return ui.div(
            ui.tags.strong("EasyOCR not installed."),
            ui.tags.br(),
            "Auto-calibration is disabled. You can still calibrate manually "
            "using P1, P2 and P3 anchors.",
            class_="alert alert-warning",
            style="padding:6px 10px; margin-top:6px;",
        )

    @render.ui
    def session_status():
        # Subscribe to cal_revision + overlay_revision so the status text
        # refreshes after detection, manual calibration, and CSV upload.
        _ = cal_revision()
        _ = overlay_revision()
        fid = file_id_rv()
        if fid is None:
            return ui.tags.em("No image loaded.")
        fs = pv.state.files.get(fid)
        if fs is None:
            return ui.tags.em("(file missing)")
        h, w = fs.image_rgb.shape[:2]
        bits = [
            ui.div(ui.tags.strong("Image: "), fs.image_filename),
            ui.div(f"{w} × {h} px"),
        ]
        if fs.csv_filename:
            bits.append(ui.div(ui.tags.strong("CSV: "), fs.csv_filename))
            if fs.csv_df is not None:
                bits.append(ui.div(f"{len(fs.csv_df)} rows, "
                                    f"{fs.csv_df['series'].nunique()} series"))
        else:
            bits.append(ui.div(ui.tags.em("No CSV loaded.")))
        bits.append(ui.div(ui.tags.strong("Status: "), str(fs.review_status.value)))
        return ui.div(*bits, style="font-size:12px;")

    # ------------------------------------------------------------------
    # Calibration tab — figure
    # ------------------------------------------------------------------

    @render_widget
    def cal_plot():
        """Build the calibration FigureWidget once per file / calibration apply.

        The widget rebuilds whenever ``file_id_rv`` or ``cal_revision``
        changes; pure anchor edits update ``widget.layout.shapes`` in place
        via the separate ``_push_anchors_to_widget`` effect to avoid
        re-sending the entire figure (including the image) on every drag.
        """
        fid = file_id_rv()
        # `cal_revision()` participates in the dependency graph so the figure
        # rebuilds after auto/manual calibration applies (which may also
        # change `show_diagnostic`).
        _ = cal_revision()
        fs = pv.state.files.get(fid) if fid is not None else None
        if fs is None:
            return go.FigureWidget(go.Figure(
                layout=dict(height=620, plot_bgcolor="#fafafa",
                            annotations=[dict(
                                text="Upload an image to begin.",
                                showarrow=False, x=0.5, y=0.5,
                                xref="paper", yref="paper",
                                font=dict(size=16, color="#555"),
                            )])))
        diag = None
        if show_diagnostic() and fs.detection_result is not None:
            try:
                diag = render_overlay(fs.image_bgr, fs.detection_result)
            except Exception as exc:
                _trace("cal_plot.diagnostic_overlay_error", error=repr(exc))
                diag = None
        with reactive.isolate():
            initial_anchors = anchors_rv()
        t0 = time.perf_counter()
        fig = build_calibration_edit_figure(
            fs.image_rgb, initial_anchors, diagnostic_rgb=diag,
            image_data_uri=_get_image_uri(fid) if diag is None else None,
        )
        _trace("cal_plot.build", ms=int((time.perf_counter() - t0) * 1000))
        # Bake band shapes into the initial figure so they survive the widget
        # rebuild. ALL reads are inside reactive.isolate so cal_plot does not
        # subscribe to axis_frame_rv or the band-slider inputs — live in-place
        # updates flow through _push_bands_to_widget instead.
        with reactive.isolate():
            _frame = axis_frame_rv()
            _y_extra, _y_vert, _y_slide = 90, 0, 0
            _x_extra, _x_horiz, _x_slide = 28, 0, 0
            if _frame is not None:
                _y_extra = _safe_int(input.band_y_extra, 90, "band_y_extra")
                _y_vert = _safe_int(input.band_y_vert, 0, "band_y_vert")
                _y_slide = _safe_int(input.band_y_slide, 0, "band_y_slide")
                _x_extra = _safe_int(input.band_x_extra, 28, "band_x_extra")
                _x_horiz = _safe_int(input.band_x_horiz, 0, "band_x_horiz")
                _x_slide = _safe_int(input.band_x_slide, 0, "band_x_slide")
        if _frame is not None:
            _yb_raw = y_label_band(_frame, extra_left=_y_extra, extra_vertical=_y_vert)
            _yb = (_yb_raw[0] + _y_slide, _yb_raw[1], _yb_raw[2] + _y_slide, _yb_raw[3])
            _xb_raw = x_label_band(_frame, extra_below=_x_extra, extra_horizontal=_x_horiz)
            _xb = (_xb_raw[0], _xb_raw[1] + _x_slide, _xb_raw[2], _xb_raw[3] + _x_slide)
            fig.update_layout(shapes=band_shapes(_yb, _xb))
        widget = go.FigureWidget(fig)
        # Enable annotation-position dragging — shapes always render resize
        # handles when editable, so the anchors are annotations instead, and
        # this config flag is what makes them user-draggable.
        try:
            widget._config = {
                **getattr(widget, "_config", {}),
                "edits": {"annotationPosition": True},
                "displayModeBar": True,
                "displaylogo": False,
            }
        except Exception as exc:  # pragma: no cover - defensive
            _trace("cal_plot.config_error", error=repr(exc))

        # Plotly's ``Layout.on_change`` calls back with the layout object plus
        # one value per watched path: ``cb(layout, annotations)`` here.
        def _on_annotations_change(layout, annotations):  # pragma: no cover (UI callback)
            _trace("cal_plot.annotations_changed",
                   n=len(annotations) if annotations else 0)
            if syncing["inputs_to_shapes"]:
                return
            if not annotations:
                return
            try:
                with reactive.isolate():
                    current = anchors_rv()
                raw = annotations_to_anchors(annotations, current)
                # Snap P1.y↔P2.y and P1.x↔P3.x so the rectangle stays closed
                # — the moved anchor wins and its partner follows.
                new_anchors = enforce_anchor_constraints(raw, current)
                # Skip when nothing moved — this fires as an async echo of
                # programmatic widget pushes (nudge/drag-constraint), which
                # arrive after the syncing flag has been cleared. Skipping
                # prevents the push→echo→set→push→… oscillation loop.
                if (new_anchors.p1_pixel == current.p1_pixel
                        and new_anchors.p2_pixel == current.p2_pixel
                        and new_anchors.p3_pixel == current.p3_pixel):
                    return
            except Exception as exc:
                _trace("cal_plot.annotations_changed_error", error=repr(exc))
                return
            syncing["shapes_to_inputs"] = True
            try:
                anchors_rv.set(new_anchors)
            finally:
                syncing["shapes_to_inputs"] = False

        widget.layout.on_change(_on_annotations_change, ("annotations",))
        return widget

    @reactive.effect
    def _push_anchors_to_widget():
        """Mirror anchor changes into the live FigureWidget without rebuilding.

        Re-pushes annotation positions (anchors) and guide-line trace
        coordinates so a numeric edit or constraint-enforced propagation
        is reflected in the figure. ``inputs_to_shapes`` suppresses the
        echo from the ``on_change`` callback during this mutation.
        """
        a = anchors_rv()
        sel = selected_anchor()  # subscribed: re-render styling on selection change
        widget = cal_plot.widget
        if widget is None:
            return
        syncing["inputs_to_shapes"] = True
        try:
            with widget.batch_update():
                # Anchor positions live in layout.annotations.
                widget.layout.annotations = anchor_annotations(a, selected=sel)
                # Guide-line traces (added by ``build_calibration_edit_figure``
                # in this exact order): 0=P1-P2 horizontal, 1=P3 horizontal,
                # 2=P1-P3 vertical, 3=P2 vertical. Update only the changing
                # coordinate per trace.
                if len(widget.data) >= 4:
                    widget.data[0].y = [a.p2_pixel[1], a.p2_pixel[1]]  # baseline y
                    widget.data[1].y = [a.p3_pixel[1], a.p3_pixel[1]]  # top y
                    widget.data[2].x = [a.p3_pixel[0], a.p3_pixel[0]]  # left-edge x
                    widget.data[3].x = [a.p2_pixel[0], a.p2_pixel[0]]  # right-edge x
        finally:
            syncing["inputs_to_shapes"] = False

    @reactive.effect
    def _push_bands_to_widget():
        """Live-update band shapes when the axis frame or slider values change.

        Does NOT depend on cal_revision — the initial figure already has band
        shapes baked in at cal_plot build time (isolate-read there). This
        effect only handles the live in-place update path (slider edits after
        detection, or when axis_frame_rv first becomes available before the
        rebuilt widget is ready).
        """
        frame = axis_frame_rv()
        widget = cal_plot.widget
        if widget is None:
            return
        if frame is None:
            with widget.batch_update():
                widget.layout.shapes = []
            return
        y_extra = _safe_int(input.band_y_extra, 90, "band_y_extra")
        y_vert = _safe_int(input.band_y_vert, 0, "band_y_vert")
        y_slide = _safe_int(input.band_y_slide, 0, "band_y_slide")
        x_extra = _safe_int(input.band_x_extra, 28, "band_x_extra")
        x_horiz = _safe_int(input.band_x_horiz, 0, "band_x_horiz")
        x_slide = _safe_int(input.band_x_slide, 0, "band_x_slide")
        _yb = y_label_band(frame, extra_left=y_extra, extra_vertical=y_vert)
        yb = (_yb[0] + y_slide, _yb[1], _yb[2] + y_slide, _yb[3])
        _xb = x_label_band(frame, extra_below=x_extra, extra_horizontal=x_horiz)
        xb = (_xb[0], _xb[1] + x_slide, _xb[2], _xb[3] + x_slide)
        with widget.batch_update():
            widget.layout.shapes = band_shapes(yb, xb)

    def _selection_highlight_arrays(fs, sel):
        """Coordinate arrays for the four ``_pv_sel_*`` sentinel traces.

        Centers show every selected point; the bound markers track the
        anchor only (keeps the part-focus readable); the big anchor ring
        appears only when more than one point is selected.
        """
        empty = ([], [])
        out = {"_pv_sel_center": empty, "_pv_sel_upper": empty,
               "_pv_sel_lower": empty, "_pv_sel_anchor": empty}
        pids = _sel_pids(sel)
        if not pids:
            return out
        by_pid = {p.point_id: p for p in fs.overlay.points()}
        pts = [by_pid[p] for p in pids if p in by_pid]
        if not pts:
            return out
        out["_pv_sel_center"] = ([float(p.x) for p in pts],
                                 [float(p.y) for p in pts])
        anchor_pt = by_pid.get(_sel_anchor(sel))
        if anchor_pt is None:
            return out
        is_horiz = _fs_is_horizontal(fs)
        u, l = anchor_pt.y_err_upper, anchor_pt.y_err_lower
        _u_ok = u is not None and np.isfinite(float(u))
        _l_ok = l is not None and np.isfinite(float(l))
        if is_horiz:
            # Interval runs along x, so the endpoints sit at (bound, category-y).
            if _u_ok:
                out["_pv_sel_upper"] = ([float(u)], [float(anchor_pt.y)])
            if _l_ok:
                out["_pv_sel_lower"] = ([float(l)], [float(anchor_pt.y)])
        else:
            if _u_ok:
                out["_pv_sel_upper"] = ([float(anchor_pt.x)], [float(u)])
            if _l_ok:
                out["_pv_sel_lower"] = ([float(anchor_pt.x)], [float(l)])
        if len(pts) > 1:
            out["_pv_sel_anchor"] = ([float(anchor_pt.x)], [float(anchor_pt.y)])
        return out

    @reactive.effect
    def _push_overlay_selection_to_widget():
        """Update the selection-highlight traces in the overlay widget in-place.

        Runs when selected_overlay_rv changes or the widget is rebuilt, so
        the highlight follows the user's selection without a full figure rebuild.
        """
        sel = selected_overlay_rv()
        widget = overlay_plot.widget
        if widget is None:
            return
        # Find highlight traces by name (order-independent).
        trace_map = {}
        for tr in widget.data:
            n = getattr(tr, "name", None)
            if n in ("_pv_sel_center", "_pv_sel_upper",
                     "_pv_sel_lower", "_pv_sel_anchor"):
                trace_map[n] = tr
        if not trace_map:
            return
        if sel is None:
            with widget.batch_update():
                for tr in trace_map.values():
                    tr.x = []
                    tr.y = []
            return
        with reactive.isolate():
            fid = file_id_rv()
        if fid is None:
            return
        fs = pv.state.files.get(fid)
        if fs is None or fs.overlay is None:
            return
        arrays = _selection_highlight_arrays(fs, sel)
        with widget.batch_update():
            for name, (axs, ays) in arrays.items():
                if name in trace_map:
                    trace_map[name].x = axs
                    trace_map[name].y = ays

    @render.ui
    def cal_summary():
        fid = file_id_rv()
        fs = pv.state.files.get(fid) if fid is not None else None
        if fs is None:
            return ui.div()
        res = fs.detection_result
        if res is None:
            return ui.tags.em("No calibration applied yet.")
        if not res.success:
            return ui.div(
                ui.tags.strong("Calibration failed."),
                ui.tags.ul(*[ui.tags.li(w) for w in (res.warnings or ["unknown"])]),
                style="color:#a33;",
            )
        xc, yc = res.x_calibration, res.y_calibration
        return ui.div(
            ui.tags.strong(f"Mode: {res.mode}"),
            f" — confidence {res.confidence:.2f} | ",
            f"X: scale={xc.scale:.4g}, offset={xc.offset:.4g}",
            (f" (log10)" if xc.log_base else ""),
            " | ",
            f"Y: scale={yc.scale:.4g}, offset={yc.offset:.4g}",
            (f" (log10)" if yc.log_base else ""),
        )

    # ------------------------------------------------------------------
    # Anchor ↔ numeric-input sync
    # ------------------------------------------------------------------

    @reactive.effect
    def _push_anchors_to_inputs():
        """Whenever anchors change (from drag), push values into the inputs."""
        a = anchors_rv()
        if syncing["inputs_to_shapes"]:
            return
        # Block _inputs_to_anchors while the browser is processing these
        # updates — each ui.update_numeric echo triggers a "change" event that
        # fires _inputs_to_anchors with partially-stale values, causing
        # enforce_anchor_constraints to misidentify which anchor led and snap
        # the position back. Flag is cleared once all pixel inputs have settled.
        syncing["shapes_to_inputs"] = True
        ui.update_numeric("p3_px_x", value=round(a.p3_pixel[0], 2))
        ui.update_numeric("p3_px_y", value=round(a.p3_pixel[1], 2))
        ui.update_numeric("p2_px_x", value=round(a.p2_pixel[0], 2))
        ui.update_numeric("p2_px_y", value=round(a.p2_pixel[1], 2))
        ui.update_numeric("p1_data_x", value=a.p1_data_x)
        ui.update_numeric("p3_data_y", value=a.p3_data_y)
        ui.update_numeric("p2_data_x", value=a.p2_data_x)
        ui.update_numeric("p1_data_y", value=a.p1_data_y)

    @reactive.effect
    @reactive.event(input.apply_manual)
    def _apply_manual_calibration():
        fid = file_id_rv()
        if fid is None:
            ui.notification_show("Upload an image first.", type="warning")
            return
        # Pull current values from inputs (most up-to-date). Use the stored
        # anchor as fallback for any data input that is momentarily None (e.g.
        # during the ui.update_numeric echo after auto-calibration) so that a
        # None field never silently resets a value to a hardcoded constant.
        with reactive.isolate():
            cur = anchors_rv()

        def _num(v, fallback):
            return float(v) if v is not None else float(fallback)

        p3_px = (float(input.p3_px_x() or 0), float(input.p3_px_y() or 0))
        p2_px = (float(input.p2_px_x() or 0), float(input.p2_px_y() or 0))
        p1_data_y = _num(input.p1_data_y(), cur.p1_data_y)
        p3_data_y = _num(input.p3_data_y(), cur.p3_data_y)
        anchors = Anchors(
            p1_pixel=(p3_px[0], p2_px[1]),  # derived bottom-left corner
            p2_pixel=p2_px,
            p3_pixel=p3_px,
            p1_data_x=_num(input.p1_data_x(), cur.p1_data_x),
            p2_data_x=_num(input.p2_data_x(), cur.p2_data_x),
            p1_data_y=p1_data_y,
            p3_data_y=p3_data_y,
            x_log_base=_parse_log_base(input.x_log(), input.x_log_base_val()),
            y_log_base=_parse_log_base(input.y_log(), input.y_log_base_val()),
        )
        anchors_rv.set(anchors)
        try:
            result = pv.apply_manual_calibration(fid, anchors)
        except Exception as e:
            ui.notification_show(f"Manual calibration failed: {e}", type="error")
            return
        if not result.success:
            ui.notification_show(
                "Manual calibration failed: " + "; ".join(result.warnings),
                type="error",
            )
        else:
            ui.notification_show("Manual calibration applied.", type="message")
        _bump_cal()
        _bump_overlay()

    @reactive.effect
    @reactive.event(input.run_detection)
    def _run_auto_detection():
        fid = file_id_rv()
        if fid is None:
            ui.notification_show("Upload an image first.", type="warning")
            return
        if not pv.ocr_available:
            ui.notification_show(
                "EasyOCR is not installed — auto-calibration unavailable. "
                "Use Manual Values instead.", type="warning",
            )
            return
        cfg = CalibrationConfig(
            y_band_extra_px=_safe_int(input.band_y_extra, 90, "band_y_extra"),
            y_band_extra_vertical_px=_safe_int(input.band_y_vert, 0, "band_y_vert"),
            y_band_x_offset=_safe_int(input.band_y_slide, 0, "band_y_slide"),
            x_band_extra_px=_safe_int(input.band_x_extra, 28, "band_x_extra"),
            x_band_extra_horizontal_px=_safe_int(input.band_x_horiz, 0, "band_x_horiz"),
            x_band_y_offset=_safe_int(input.band_x_slide, 0, "band_x_slide"),
            min_ocr_confidence=_safe_float(input.cfg_min_ocr_conf, 0.20, "cfg_min_ocr_conf"),
        )
        with ui.Progress(min=0, max=1) as p:
            p.set(0.1, message="Running OCR + geometry…")
            try:
                result = pv.run_auto_calibration(fid, config=cfg)
            except Exception as e:
                ui.notification_show(f"Auto-calibration failed: {e}", type="error")
                return
            p.set(1.0)
        anchors_rv.set(anchors_from_result(result, anchors_rv()))
        if result.bbox is not None:
            axis_frame_rv.set(result.bbox)
        _bump_cal()
        _bump_overlay()
        if not result.success:
            ui.notification_show(
                "Detection failed; switch to Manual Values to calibrate.",
                type="warning",
            )

    @reactive.effect
    @reactive.event(input.detect_frame)
    def _detect_axis_frame_only():
        fid = file_id_rv()
        fs = pv.state.files.get(fid) if fid is not None else None
        if fs is None:
            return
        try:
            preview = detect_axis_frame(fs.image_bgr)
        except Exception as e:
            ui.notification_show(f"Frame detection failed: {e}", type="error")
            return
        if preview.bbox is None:
            ui.notification_show("No axis frame detected.", type="warning")
            return
        bbox = preview.bbox
        axis_frame_rv.set(bbox)
        a = anchors_rv()
        new_a = Anchors(
            p1_pixel=(float(bbox.left), float(bbox.bottom)),
            p2_pixel=(float(bbox.right), float(bbox.bottom)),
            p3_pixel=(float(bbox.left), float(bbox.top)),
            p1_data_x=a.p1_data_x, p2_data_x=a.p2_data_x,
            p1_data_y=a.p1_data_y, p3_data_y=a.p3_data_y,
            x_log_base=a.x_log_base, y_log_base=a.y_log_base,
        )
        anchors_rv.set(new_a)
        _bump_cal()

    @reactive.effect
    @reactive.event(input.reset_anchors)
    def _reset_anchors_to_default():
        fid = file_id_rv()
        fs = pv.state.files.get(fid) if fid is not None else None
        if fs is None:
            return
        h, w = fs.image_rgb.shape[:2]
        anchors_rv.set(default_anchors_for_image(w, h))
        _bump_cal()

    @reactive.effect
    def _reset_arrow_step_for_file():
        fid = file_id_rv()
        fs = pv.state.files.get(fid) if fid is not None else None
        if fs is None:
            return
        ui.update_numeric("overlay_arrow_step",
                          value=_pixel_step_for_file(fs))

    # ------------------------------------------------------------------
    # Keyboard nudging of P1/P2 and overlay points
    # ------------------------------------------------------------------

    def _pixel_step_for_file(fs) -> float:
        """Return the y-axis data-per-pixel magnitude from calibration, or 0.1."""
        try:
            scale = abs(fs.detection_result.y_calibration.scale)
            if scale > 0 and np.isfinite(scale):
                return round(scale, 6)
        except (AttributeError, TypeError):
            pass
        return 0.1

    def _fs_is_horizontal(fs) -> bool:
        """Value axis runs along x (forest, or horizontal bar/box)."""
        return fs is not None and is_horizontal_layout(
            fs.plot_type, getattr(fs, "orientation", "vertical"))

    def _model_vals(fs, pt):
        """A point's current values as ``PointVals`` (non-finite bounds → None)."""
        u, l = pt.y_err_upper, pt.y_err_lower
        return PointVals(
            x=float(pt.x), y=float(pt.y),
            upper=(float(u) if u is not None and np.isfinite(float(u)) else None),
            lower=(float(l) if l is not None and np.isfinite(float(l)) else None),
        )

    def _sync_edit_inputs(fs, pid):
        """Reflect a point's model values into the edit-panel inputs.

        Pushes exact floats so the echo events compare clean against the
        model inside ``_live_update_point_inputs`` (its 1e-10 guard).
        """
        pt = next((p for p in fs.overlay.points() if p.point_id == pid), None)
        if pt is None:
            return
        vals = _model_vals(fs, pt)
        ui.update_numeric("edit_point_x", value=vals.x)
        ui.update_numeric("edit_point_y", value=vals.y)
        if vals.upper is not None:
            ui.update_numeric("edit_err_upper", value=vals.upper)
        if vals.lower is not None:
            ui.update_numeric("edit_err_lower", value=vals.lower)
        center = vals.x if _fs_is_horizontal(fs) else vals.y
        hw = half_width_of(center, vals.upper, vals.lower)
        ui.update_numeric("edit_half_width", value=hw if hw is not None else 0.0)

    def _push_point_edits_to_widget(edits):
        """Update edited points in the overlay FigureWidget in-place.

        ``edits`` is a list of ``(pid, x, y, upper, lower)`` tuples whose
        values are already persisted to ``fs.overlay``. Trace lookup happens
        once and every mutation shares one ``batch_update()``, so a
        multi-point gang nudge costs a single round-trip. No full figure
        rebuild, so arrow-key editing is smooth.
        """
        widget = overlay_plot.widget
        if widget is None or not edits:
            return
        with reactive.isolate():
            fid = file_id_rv()
            sel = selected_overlay_rv()
        fs = pv.state.files.get(fid) if fid is not None else None
        if fs is None or fs.overlay is None:
            return
        is_horiz = _fs_is_horizontal(fs)
        tmap = {}
        for tr in widget.data:
            n = getattr(tr, "name", "") or ""
            if n:
                tmap[n] = tr

        by_series = {}
        for edit in edits:
            series = str(edit[0]).rsplit("#", 1)[0]
            by_series.setdefault(series, []).append(edit)

        with widget.batch_update():
            for series, series_edits in by_series.items():
                main_tr = tmap.get(series)
                if main_tr is None or main_tr.customdata is None:
                    continue
                pid_to_local = {str(list(row)[0]): ci
                                for ci, row in enumerate(main_tr.customdata)}
                xs = list(main_tr.x)
                ys = list(main_tr.y)
                # Horizontal layouts bracket the value axis (error_x)
                # measured from x; vertical ones bracket y.
                ebar = main_tr.error_x if is_horiz else main_tr.error_y
                arr_plus = arr_minus = None
                if ebar is not None and getattr(ebar, "array", None) is not None:
                    arr_plus = list(ebar.array)
                    _am = ebar.arrayminus
                    arr_minus = list(_am) if _am is not None else [0.0] * len(arr_plus)
                cap_u_tr = tmap.get(f"_pv_cap_u_{series}")
                cap_l_tr = tmap.get(f"_pv_cap_l_{series}")
                rib_u_tr = tmap.get(f"_pv_rib_u_{series}")
                rib_l_tr = tmap.get(f"_pv_rib_l_{series}")
                rib_band_tr = tmap.get(f"_pv_rib_{series}")
                caps = {}
                for cap_tr in (cap_u_tr, cap_l_tr):
                    if cap_tr is not None and cap_tr.customdata is not None:
                        caps[id(cap_tr)] = (
                            {str(list(row)[0]): ci
                             for ci, row in enumerate(cap_tr.customdata)},
                            list(cap_tr.x), list(cap_tr.y),
                        )
                rib = None
                if rib_u_tr is not None and rib_l_tr is not None:
                    rib = (list(rib_u_tr.x), list(rib_u_tr.y), list(rib_l_tr.y))

                for pid, new_x, new_y, new_upper, new_lower in series_edits:
                    li = pid_to_local.get(str(pid))
                    if li is None or not (0 <= li < len(xs)):
                        continue
                    old_x = float(xs[li])
                    xs[li] = float(new_x)
                    ys[li] = float(new_y)
                    _u_ok = new_upper is not None and np.isfinite(float(new_upper))
                    _l_ok = new_lower is not None and np.isfinite(float(new_lower))
                    _e_base = float(new_x) if is_horiz else float(new_y)
                    if arr_plus is not None and 0 <= li < len(arr_plus):
                        if _u_ok:
                            arr_plus[li] = max(0.0, float(new_upper) - _e_base)
                        if _l_ok:
                            arr_minus[li] = max(0.0, _e_base - float(new_lower))
                    # Ribbon: locate this point by its previous x. A missing
                    # bound collapses onto the point estimate (one-sided band).
                    if rib is not None and (_u_ok or _l_ok):
                        rxs, rys_u, rys_l = rib
                        rib_idx = next(
                            (ri for ri, rx in enumerate(rxs)
                             if abs(float(rx) - old_x) < 1e-9),
                            None,
                        )
                        if rib_idx is not None:
                            rxs[rib_idx] = float(new_x)
                            rys_u[rib_idx] = float(new_upper) if _u_ok else float(new_y)
                            rys_l[rib_idx] = float(new_lower) if _l_ok else float(new_y)
                    # Clickable caps: (bound, category-y) in horizontal
                    # layouts; (x, bound) in vertical ones.
                    for cap_tr, val in ((cap_u_tr, new_upper), (cap_l_tr, new_lower)):
                        if (cap_tr is None or val is None
                                or id(cap_tr) not in caps
                                or not np.isfinite(float(val))):
                            continue
                        cmap, cxs, cys = caps[id(cap_tr)]
                        ci = cmap.get(str(pid))
                        if ci is None:
                            continue
                        if is_horiz:
                            cxs[ci] = float(val)
                            cys[ci] = float(new_y)
                        else:
                            cxs[ci] = float(new_x)
                            cys[ci] = float(val)

                main_tr.x = xs
                main_tr.y = ys
                if arr_plus is not None and ebar is not None:
                    ebar.array = arr_plus
                    ebar.arrayminus = arr_minus
                if rib is not None:
                    rib_u_tr.x = rib[0]
                    rib_u_tr.y = rib[1]
                    rib_l_tr.x = rib[0]
                    rib_l_tr.y = rib[2]
                for cap_tr in (cap_u_tr, cap_l_tr):
                    if cap_tr is not None and id(cap_tr) in caps:
                        _, cxs, cys = caps[id(cap_tr)]
                        cap_tr.x = cxs
                        cap_tr.y = cys
                # Forest band (one rectangle per row): rebuild the whole
                # series band from the already-persisted overlay model so
                # multi-row edits stay consistent.
                if rib_band_tr is not None:
                    band_x = []
                    band_y = []
                    hh = 0.32
                    for p in fs.overlay.points():
                        if str(p.point_id).rsplit("#", 1)[0] != series:
                            continue
                        u, l = p.y_err_upper, p.y_err_lower
                        u_ok = u is not None and np.isfinite(float(u))
                        l_ok = l is not None and np.isfinite(float(l))
                        if not (u_ok or l_ok):
                            continue
                        lo = float(l) if l_ok else float(p.x)
                        hi = float(u) if u_ok else float(p.x)
                        yc = float(p.y)
                        band_x += [lo, hi, hi, lo, lo, None]
                        band_y += [yc - hh, yc - hh, yc + hh, yc + hh, yc - hh, None]
                    rib_band_tr.x = band_x
                    rib_band_tr.y = band_y

            # Selection highlights track the (already-persisted) model.
            arrays = _selection_highlight_arrays(fs, sel)
            for name, (axs, ays) in arrays.items():
                tr = tmap.get(name)
                if tr is not None:
                    tr.x = axs
                    tr.y = ays
        # Keep the zoom bubble in sync for the anchor point. (Defined below;
        # safe because this only runs after server initialisation.)
        anchor = _sel_anchor(sel)
        anchor_edit = next((e for e in edits if str(e[0]) == anchor), None)
        if anchor_edit is not None:
            _push_zoom_bubble_update(*anchor_edit)

    def _push_point_edit_to_widget(pid, new_x, new_y, new_upper, new_lower):
        """Single-point wrapper around :func:`_push_point_edits_to_widget`."""
        _push_point_edits_to_widget([(pid, new_x, new_y, new_upper, new_lower)])

    def _push_zoom_bubble_update(pid, new_x, new_y, new_upper, new_lower):
        """Update the zoom bubble widget in-place — no figure rebuild."""
        widget = zoom_bubble.widget
        if widget is None:
            return
        with reactive.isolate():
            sel = selected_overlay_rv()
            fid = file_id_rv()
        if sel is None or _sel_anchor(sel) != pid or fid is None:
            return
        fs = pv.state.files.get(fid)
        if fs is None:
            return
        cal = cal_dict_from_result(fs.detection_result)
        if not cal.get("applied"):
            return
        part = _sel_part(sel)
        is_horiz = _fs_is_horizontal(fs)
        _u_ok = new_upper is not None and np.isfinite(float(new_upper))
        _l_ok = new_lower is not None and np.isfinite(float(new_lower))

        if is_horiz:
            # Endpoints are x-values at a fixed category coordinate.
            if part == "upper" and _u_ok:
                focus_x, focus_y = float(new_upper), float(new_y)
            elif part == "lower" and _l_ok:
                focus_x, focus_y = float(new_lower), float(new_y)
            else:
                focus_x, focus_y = float(new_x), float(new_y)
        elif part == "upper" and _u_ok:
            focus_x, focus_y = float(new_x), float(new_upper)
        elif part == "lower" and _l_ok:
            focus_x, focus_y = float(new_x), float(new_lower)
        else:
            focus_x, focus_y = float(new_x), float(new_y)

        x_log = cal.get("x_log_base")
        y_log = cal.get("y_log_base")

        def _plot(v, log_base):
            return float(np.log10(v)) if (log_base and v > 0) else float(v)

        focus_xp = _plot(focus_x, x_log)
        focus_yp = _plot(focus_y, y_log)

        # Derive separate x and y zoom radii the same way the figure builder does.
        h_img, w_img = fs.image_rgb.shape[:2]
        x_left_d, _ = px_to_data(0, 0, cal)
        x_right_d, _ = px_to_data(w_img, 0, cal)
        _, y_top_d = px_to_data(0, 0, cal)
        _, y_bot_d = px_to_data(0, h_img, cal)
        x_full = abs(_plot(x_right_d, x_log) - _plot(x_left_d, x_log))
        y_full = abs(_plot(y_top_d, y_log) - _plot(y_bot_d, y_log))

        # A one-sided interval collapses its missing bound onto the point
        # estimate, matching build_zoom_bubble_figure.
        _has_iv = _u_ok or _l_ok
        _v_c = float(new_x) if is_horiz else float(new_y)
        _eff_u = float(new_upper) if _u_ok else _v_c
        _eff_l = float(new_lower) if _l_ok else _v_c
        if is_horiz:
            # Interval runs along x; keep the vertical (category) framing steady.
            err_w_plot = (abs(_plot(_eff_u, x_log) - _plot(_eff_l, x_log))
                          if _has_iv else None)
            x_zoom_r = err_w_plot * 0.8 if (err_w_plot and err_w_plot > 0) else x_full * 0.05
            try:
                cur_range = widget.layout.yaxis.range
                y_zoom_r = abs(cur_range[1] - cur_range[0]) / 2 if cur_range else y_full * 0.05
            except Exception as exc:
                _trace("zoom_bubble.yaxis_range_error", error=repr(exc))
                y_zoom_r = y_full * 0.05
        else:
            x_zoom_r = x_full * 0.05
            err_h_plot = (abs(_plot(_eff_u, y_log) - _plot(_eff_l, y_log))
                          if _has_iv else None)
            if err_h_plot is not None and err_h_plot > 0:
                y_zoom_r = err_h_plot * 0.8
            else:
                try:
                    cur_range = widget.layout.yaxis.range
                    y_zoom_r = abs(cur_range[1] - cur_range[0]) / 2 if cur_range else y_full * 0.05
                except Exception as exc:
                    _trace("zoom_bubble.yaxis_range_error", error=repr(exc))
                    y_zoom_r = y_full * 0.05

        x_lo_p, x_hi_p = focus_xp - x_zoom_r, focus_xp + x_zoom_r
        y_lo_p, y_hi_p = focus_yp - y_zoom_r, focus_yp + y_zoom_r

        bub_pt = bub_sel = bub_vline = bub_hline = bub_ribbon = None
        for tr in widget.data:
            n = getattr(tr, "name", "") or ""
            if n == "_bub_pt":
                bub_pt = tr
            elif n == "_bub_sel":
                bub_sel = tr
            elif n == "_bub_vline":
                bub_vline = tr
            elif n == "_bub_hline":
                bub_hline = tr
            elif n == "_bub_ribbon":
                bub_ribbon = tr

        with widget.batch_update():
            if bub_pt is not None:
                bub_pt.x = [float(new_x)]
                bub_pt.y = [float(new_y)]
                # Interval brackets x for horizontal layouts, y otherwise.
                ebar = bub_pt.error_x if is_horiz else bub_pt.error_y
                if ebar is not None and _has_iv:
                    ebar.array = [max(0.0, _eff_u - _v_c)]
                    ebar.arrayminus = [max(0.0, _v_c - _eff_l)]
            if bub_sel is not None:
                bub_sel.x = [focus_x]
                bub_sel.y = [focus_y]
            # Crosshairs: only the position changes, not the full extent.
            if bub_vline is not None:
                bub_vline.x = [focus_x, focus_x]
            if bub_hline is not None:
                bub_hline.y = [focus_y, focus_y]
            if bub_ribbon is not None and _has_iv:
                lo, hi = _eff_l, _eff_u
                if is_horiz:
                    bh = y_zoom_r * 0.12
                    y_c = float(new_y)
                    bub_ribbon.x = [lo, hi, hi, lo, lo]
                    bub_ribbon.y = [y_c - bh, y_c - bh, y_c + bh, y_c + bh, y_c - bh]
                else:
                    bw = x_zoom_r * 0.12
                    x_c = float(new_x)
                    bub_ribbon.x = [x_c - bw, x_c - bw, x_c + bw, x_c + bw, x_c - bw]
                    bub_ribbon.y = [lo, hi, hi, lo, lo]
            widget.layout.xaxis.range = [x_lo_p, x_hi_p]
            widget.layout.yaxis.range = [y_lo_p, y_hi_p]

    def _nudge_selection(dx_units, dy_units, sel):
        """Apply coalesced arrow-key steps to every selected overlay point.

        ``dx_units``/``dy_units`` are net step counts from the client-side
        accumulator (screen-up = +dy, Shift x10 already applied), so one call
        may represent several held-key repeats. All selected points gang-move
        with one model pass and one widget round-trip.
        """
        with reactive.isolate():
            fid = file_id_rv()
        if fid is None:
            return
        fs = pv.state.files.get(fid)
        if fs is None or fs.overlay is None:
            return
        pids = _sel_pids(sel)
        if not pids:
            return
        # Bound-part nudging only makes sense for a single selected point.
        part = _sel_part(sel)
        base_step = _safe_float(input.overlay_arrow_step,
                                _pixel_step_for_file(fs), "overlay_arrow_step")
        dx = base_step * dx_units
        dy = base_step * dy_units

        linked = _safe_bool(input.link_bounds, False, "link_bounds")
        is_horiz = _fs_is_horizontal(fs)
        lock_y = fs.plot_type == "forest"
        by_pid = {p.point_id: p for p in fs.overlay.points()}

        edits = []
        try:
            for pid in pids:
                pt = by_pid.get(pid)
                if pt is None:
                    continue
                new = apply_nudge(
                    _model_vals(fs, pt), dx, dy,
                    part=part, linked=linked,
                    is_horizontal=is_horiz, lock_y=lock_y,
                )
                fs.overlay.edit_point(pid, new.x, new.y)
                if new.upper is not None:
                    fs.overlay.edit_err_upper(pid, float(new.upper))
                if new.lower is not None:
                    fs.overlay.edit_err_lower(pid, float(new.lower))
                edits.append((pid, new.x, new.y, new.upper, new.lower))
        except Exception as e:
            _trace("nudge_selection.error", error=repr(e))
            return
        if not edits:
            return

        # Reflect the anchor's new values (including half-width) into the
        # edit panel. Exact floats so the echo events pass the guard.
        anchor = _sel_anchor(sel)
        if anchor is not None:
            _sync_edit_inputs(fs, anchor)
        # Update the widget in-place — no figure rebuild, no _bump_overlay().
        # The overlay rebuilds on the next Apply/Reset/tab-switch, at which
        # point fs.overlay carries the fully-updated state.
        _push_point_edits_to_widget(edits)

    @reactive.effect
    @reactive.event(input.anchor_selected)
    def _on_anchor_clicked():
        """JS-side ``plotly_clickannotation`` selects an anchor for nudging."""
        payload = input.anchor_selected()
        _trace("anchor_selected", payload=payload)
        if not isinstance(payload, dict):
            return
        name = payload.get("name")
        if name in ANCHOR_LABELS:
            selected_anchor.set(name)

    @reactive.effect
    @reactive.event(input.anchor_nudge)
    def _on_anchor_nudge():
        """Move the currently selected anchor by ±step px on arrow keys.

        Step defaults to 1 px (10 px with Shift held). Constraints run in
        the same pipeline as drag/typed edits, so the partner anchor follows.
        When the Overlay tab is active and a point is selected, arrow keys
        edit that point instead.
        """
        payload = input.anchor_nudge()
        _trace("anchor_nudge", payload=payload)
        if not isinstance(payload, dict):
            return
        # Coalesced protocol: the client accumulates held-key repeats and
        # sends net step counts (screen-up = +dy) with the Shift multiplier
        # already applied.
        try:
            dx_units = int(payload.get("dx", 0) or 0)
            dy_units = int(payload.get("dy", 0) or 0)
        except (TypeError, ValueError):
            return
        if dx_units == 0 and dy_units == 0:
            return
        with reactive.isolate():
            active_tab = input.main_nav()
            overlay_sel = selected_overlay_rv()
        if active_tab == "Overlay" and overlay_sel is not None:
            _nudge_selection(dx_units, dy_units, overlay_sel)
            return
        with reactive.isolate():
            sel = selected_anchor()
            current = anchors_rv()
        _trace("anchor_nudge.context", selected=sel)
        if sel not in ANCHOR_LABELS:
            return
        # Pixel-y grows downward, so screen-up (+dy) is a negative pixel move.
        dx, dy = dx_units, -dy_units
        # Display "P1" maps to internal p3_pixel; "P2" maps to internal p2_pixel.
        display_to_internal = {"P1": "p3", "P2": "p2"}
        internal_key = display_to_internal.get(sel)
        if internal_key is None:
            return
        pts = {"p1": current.p1_pixel, "p2": current.p2_pixel, "p3": current.p3_pixel}
        px, py = pts[internal_key]
        pts[internal_key] = (px + dx, py + dy)
        raw = Anchors(
            p1_pixel=pts["p1"], p2_pixel=pts["p2"], p3_pixel=pts["p3"],
            p1_data_x=current.p1_data_x, p2_data_x=current.p2_data_x,
            p1_data_y=current.p1_data_y, p3_data_y=current.p3_data_y,
            x_log_base=current.x_log_base, y_log_base=current.y_log_base,
        )
        anchors_rv.set(enforce_anchor_constraints(raw, current))

    # ------------------------------------------------------------------
    # Right-column accordions
    # ------------------------------------------------------------------

    @render.ui
    def bands_panel():
        if not ocr_available():
            return ui.tags.em("Disabled — EasyOCR is not installed.")
        fid = file_id_rv()
        if fid is None:
            return ui.tags.em("Upload an image to configure label bands.")
        return ui.div(
            ui.tags.strong("Y-label band"),
            ui.row(
                ui.column(4, ui.input_numeric(
                    "band_y_extra", "Left ext. (px)", value=90, step=5,
                )),
                ui.column(4, ui.input_numeric(
                    "band_y_vert", "V. trim (px)", value=0, step=5,
                )),
                ui.column(4, ui.input_numeric(
                    "band_y_slide", "H. slide (px)", value=0, step=5,
                )),
            ),
            ui.tags.strong("X-label band"),
            ui.row(
                ui.column(4, ui.input_numeric(
                    "band_x_extra", "Below ext. (px)", value=28, step=5,
                )),
                ui.column(4, ui.input_numeric(
                    "band_x_horiz", "H. trim (px)", value=0, step=5,
                )),
                ui.column(4, ui.input_numeric(
                    "band_x_slide", "V. slide (px)", value=0, step=5,
                )),
            ),
            ui.tags.small(
                "Ext: how far from axis. Trim: shrink from ends. "
                "Slide: shift band along axis (±into plot).",
                style="color:#666; margin-top:4px; display:block;",
            ),
        )

    @render.ui
    def calib_points_panel():
        fid = file_id_rv()
        fs = pv.state.files.get(fid) if fid is not None else None
        if fs is None:
            return ui.tags.em("No file loaded.")
        res = fs.detection_result
        if res is None or not res.success:
            return ui.tags.em("Run detection (auto) or apply Manual Values to populate.")
        # Build a compact two-column table: x-axis ticks + y-axis ticks.
        rows_x = []
        for tick in res.x_paired_ticks:
            rows_x.append(ui.tags.tr(
                ui.tags.td(f"{tick.data_value:g}"),
                ui.tags.td(f"{tick.pixel_position:.1f}"),
                ui.tags.td(f"{tick.source}"),
            ))
        rows_y = []
        for tick in res.y_paired_ticks:
            rows_y.append(ui.tags.tr(
                ui.tags.td(f"{tick.data_value:g}"),
                ui.tags.td(f"{tick.pixel_position:.1f}"),
                ui.tags.td(f"{tick.source}"),
            ))
        def _table(title, rows):
            if not rows:
                return ui.div(ui.tags.strong(title), ui.tags.em(" (no paired ticks)"))
            return ui.tags.div(
                ui.tags.strong(title),
                ui.tags.table(
                    ui.tags.thead(ui.tags.tr(
                        ui.tags.th("Data"), ui.tags.th("Pixel"), ui.tags.th("Source"))),
                    ui.tags.tbody(*rows),
                    class_="table table-sm",
                    style="margin-bottom:6px;",
                ),
            )
        return ui.div(
            _table("X-axis ticks", rows_x),
            _table("Y-axis ticks", rows_y),
            ui.tags.small(
                "Tick editing is not yet wired in Shiny — use the Streamlit app "
                "for now if you need to edit paired ticks.",
                style="color:#666;",
            ),
        )

    @render.ui
    def manual_values_panel():
        fid = file_id_rv()
        if fid is None:
            return ui.tags.em("Upload an image to set manual values.")
        # Read anchors *non-reactively* so the panel doesn't re-render on
        # every drag — values then flow via ui.update_numeric instead.
        with reactive.isolate():
            a = anchors_rv()
        fs = pv.state.files.get(fid)
        is_forest = fs is not None and fs.plot_type == "forest"
        # In forest mode the vertical axis is the categorical row index. As a
        # convenience, seed data-Y with the full row span (top = N-1, bottom =
        # 0) when the anchors are still at their generic defaults — but the
        # fields stay editable so the user can anchor any two rows they like.
        if (is_forest and fs.csv_df is not None
                and a.p3_data_y == 1.0 and a.p1_data_y == 0.0):
            n_rows = len(fs.csv_df)
            p1_top_y = float(n_rows - 1)
            p2_bottom_y = 0.0
        else:
            p1_top_y = float(a.p3_data_y)
            p2_bottom_y = float(a.p1_data_y)
        _y_label = "data Y"
        # Displayed P1 = top-left (internal p3); displayed P2 = bottom-right (internal p2).
        # Internal p1 (bottom-left) is derived as (p3.x, p2.y) and never shown.
        return ui.TagList(
            ui.tags.strong("P1"),
            ui.row(
                ui.column(6, ui.input_numeric("p3_px_x", "pixel x",
                                              value=round(float(a.p3_pixel[0]), 2), step=1)),
                ui.column(6, ui.input_numeric("p3_px_y", "pixel y",
                                              value=round(float(a.p3_pixel[1]), 2), step=1)),
            ),
            ui.row(
                ui.column(6, ui.input_numeric("p1_data_x", "data X",
                                              value=float(a.p1_data_x))),
                ui.column(6, ui.input_numeric("p3_data_y", _y_label,
                                              value=p1_top_y)),
            ),
            ui.tags.strong("P2"),
            ui.row(
                ui.column(6, ui.input_numeric("p2_px_x", "pixel x",
                                              value=round(float(a.p2_pixel[0]), 2), step=1)),
                ui.column(6, ui.input_numeric("p2_px_y", "pixel y",
                                              value=round(float(a.p2_pixel[1]), 2), step=1)),
            ),
            ui.row(
                ui.column(6, ui.input_numeric("p2_data_x", "data X",
                                              value=float(a.p2_data_x))),
                ui.column(6, ui.input_numeric("p1_data_y", _y_label,
                                              value=p2_bottom_y)),
            ),
            ui.tags.small(
                "Forest mode: data X is the value axis (P1 = left endpoint, "
                "P2 = right endpoint). Data Y is the row index — put P1 on any "
                "row and enter its index, P2 on any other row and enter its "
                "index (top CSV row = N-1, bottom row = 0). Rows in between are "
                "spaced evenly between the two anchors.",
                style="color:#666;",
            ) if is_forest else None,
            ui.hr(),
            ui.row(
                ui.column(6,
                    ui.input_checkbox("x_log", "X is log",
                                      value=(a.x_log_base is not None)),
                    ui.panel_conditional(
                        "input.x_log",
                        ui.input_text("x_log_base_val", "Base",
                                      value=_log_base_to_str(a.x_log_base)),
                    ),
                ),
                ui.column(6,
                    ui.input_checkbox("y_log", "Y is log",
                                      value=(a.y_log_base is not None)),
                    ui.panel_conditional(
                        "input.y_log",
                        ui.input_text("y_log_base_val", "Base",
                                      value=_log_base_to_str(a.y_log_base)),
                    ),
                ),
            ),
            ui.input_action_button("apply_manual", "Apply manual calibration",
                                    class_="btn-primary"),
            ui.tags.small(
                "Dragging an anchor on the image updates pixel values; "
                "click Apply when ready.",
                style="color:#666;",
            ),
        )

    # Numeric input → anchor (live mirror). We avoid an effect-storm by guarding
    # against the reverse path with `syncing`. Inputs are rendered conditionally
    # via `@render.ui`, so before the manual-values panel mounts the reads
    # raise SilentException — fall through to a no-op.
    @reactive.effect
    def _inputs_to_anchors():
        if syncing["shapes_to_inputs"]:
            # Pixel inputs are being updated by _push_anchors_to_inputs.
            # Read them (creating reactive dependencies so this effect re-runs
            # on each echo) and wait until they all match anchors_rv before
            # clearing the lock and returning control to the user.
            try:
                with reactive.isolate():
                    a = anchors_rv()
                p2x = input.p2_px_x()
                p2y = input.p2_px_y()
                p3x = input.p3_px_x()
                p3y = input.p3_px_y()
                if (p2x is not None and p2y is not None
                        and p3x is not None and p3y is not None
                        and round(float(p2x), 2) == round(a.p2_pixel[0], 2)
                        and round(float(p2y), 2) == round(a.p2_pixel[1], 2)
                        and round(float(p3x), 2) == round(a.p3_pixel[0], 2)
                        and round(float(p3y), 2) == round(a.p3_pixel[1], 2)):
                    syncing["shapes_to_inputs"] = False
            except Exception as exc:
                _trace("anchors_sync.compare_error", error=repr(exc))
                syncing["shapes_to_inputs"] = False
            return
        try:
            vals = (
                input.p3_px_x(), input.p3_px_y(),  # displayed P1 pixel (internal p3)
                input.p2_px_x(), input.p2_px_y(),  # displayed P2 pixel (internal p2)
                input.p1_data_x(), input.p3_data_y(),  # displayed P1 data
                input.p2_data_x(), input.p1_data_y(),  # displayed P2 data
                input.x_log(), input.x_log_base_val(),
                input.y_log(), input.y_log_base_val(),
            )
        except Exception as exc:
            _trace("inputs_to_anchors.read_error", error=repr(exc))
            return
        if any(v is None for v in vals[:4]):
            return
        # Read prev non-reactively — this effect should only depend on inputs.
        # Done here (before raw) so we can use prev data values as fallback
        # for any momentarily-None data inputs (e.g. during the ui.update_numeric
        # round-trip after auto-calibration) instead of resetting to hardcoded
        # defaults such as 1.0.
        with reactive.isolate():
            prev = anchors_rv()
        p3_px = (float(vals[0]), float(vals[1]))
        p2_px = (float(vals[2]), float(vals[3]))
        raw = Anchors(
            p1_pixel=(p3_px[0], p2_px[1]),  # derived bottom-left corner
            p2_pixel=p2_px,
            p3_pixel=p3_px,
            p1_data_x=float(vals[4]) if vals[4] is not None else prev.p1_data_x,
            p3_data_y=float(vals[5]) if vals[5] is not None else prev.p3_data_y,
            p2_data_x=float(vals[6]) if vals[6] is not None else prev.p2_data_x,
            p1_data_y=float(vals[7]) if vals[7] is not None else prev.p1_data_y,
            x_log_base=_parse_log_base(vals[8], vals[9]),
            y_log_base=_parse_log_base(vals[10], vals[11]),
        )
        # Apply rectangle constraints: a typed P1.x value pulls P3.x along, etc.
        a = enforce_anchor_constraints(raw, prev)
        if (a.p1_pixel == prev.p1_pixel
            and a.p2_pixel == prev.p2_pixel
            and a.p3_pixel == prev.p3_pixel
            and a.p1_data_x == prev.p1_data_x
            and a.p2_data_x == prev.p2_data_x
            and a.p1_data_y == prev.p1_data_y
            and a.p3_data_y == prev.p3_data_y
            and a.x_log_base == prev.x_log_base
            and a.y_log_base == prev.y_log_base):
            return
        syncing["inputs_to_shapes"] = True
        try:
            anchors_rv.set(a)
        finally:
            syncing["inputs_to_shapes"] = False

    @render.ui
    def warnings_panel():
        fid = file_id_rv()
        fs = pv.state.files.get(fid) if fid is not None else None
        if fs is None:
            return ui.tags.em("No file loaded.")
        res = fs.detection_result
        if res is None:
            return ui.tags.em("No detection run yet.")
        warnings = list(res.warnings or [])
        if not warnings:
            return ui.tags.em("No warnings.")
        return ui.tags.ul(*[ui.tags.li(w) for w in warnings])

    # ------------------------------------------------------------------
    # Overlay tab
    # ------------------------------------------------------------------

    @render_widget
    def overlay_plot():
        # Only do real work when the Overlay tab is active; otherwise emit a
        # cheap empty figure. This is the biggest win for upload time — the
        # image used to be PNG-encoded twice (cal_plot + overlay_plot) on
        # every upload, even though the Overlay tab is hidden.
        active = _safe_str(input.main_nav, _DEFAULT_NAV, "main_nav")
        if active != "Overlay":
            return go.FigureWidget(go.Figure(layout=dict(height=620, autosize=True)))
        _trace("overlay_plot.render")
        fid = file_id_rv()
        _ = overlay_revision()
        if fid is None:
            return go.FigureWidget(go.Figure(
                layout=dict(height=620, plot_bgcolor="#fafafa",
                            annotations=[dict(
                                text="Upload an image and CSV to see the overlay.",
                                showarrow=False, x=0.5, y=0.5,
                                xref="paper", yref="paper",
                                font=dict(size=16, color="#555"),
                            )])))
        fs = pv.state.files.get(fid)
        if fs is None:
            return go.FigureWidget(go.Figure(
                layout=dict(height=620, plot_bgcolor="#fafafa")))
        cal = cal_dict_from_result(fs.detection_result)
        if fs.csv_df is None or fs.overlay is None:
            fig = build_data_overlay_figure(fs.image_rgb, [], {"applied": False})
            return go.FigureWidget(fig)
        if not cal.get("applied"):
            fig = build_data_overlay_figure(fs.image_rgb, [], cal)
            return go.FigureWidget(fig)

        edited_ids = {p.point_id for p in fs.overlay.points() if p.edited}
        df = fs.overlay.to_dataframe()
        is_forest = fs.plot_type == "forest"
        visibility: dict[str, bool] = {}
        mask_active: dict[str, bool] = {}
        # Re-attach metadata columns that EditableOverlay.to_dataframe() drops.
        if fs.csv_df is not None and len(df) == len(fs.csv_df):
            _reattach = {
                "forest": ("is_summary", "status"),
                "box": ("box_q1", "box_median", "box_q3", "status", "is_summary"),
                "kaplan_meier": ("at_risk", "status"),
            }
            for col in _reattach.get(fs.plot_type, ()):
                if col in fs.csv_df.columns:
                    df[col] = fs.csv_df[col].to_numpy()
        if is_forest:
            # A single compact control drives every row.
            vis_all = _safe_bool(lambda: input.forest_vis(), True)
            mask_all = _safe_bool(lambda: input.forest_mask(), False)
            try:
                de_all = _coerce_delta_e(input.forest_de())
            except Exception:
                de_all = None
            names = df["series"].drop_duplicates().tolist()
            if de_all is None:
                de_all = int(fs.series_delta_e.get(names[0], DEFAULT_DELTA_E)
                             if names else DEFAULT_DELTA_E)
            visibility = {n: vis_all for n in names}
            mask_active = {n: mask_all for n in names}
            for n in names:
                fs.series_delta_e[n] = de_all
        else:
            for name in df["series"].drop_duplicates().tolist():
                try:
                    visibility[name] = bool(input[_vis_id(name)]())
                except Exception as exc:
                    _trace("overlay_plot.visibility_read_error",
                           series=name, error=repr(exc))
                    visibility[name] = True
                try:
                    mask_active[name] = bool(input[_mask_id(name)]())
                except Exception:
                    mask_active[name] = False
        traces = build_overlay_traces(df, orientation=fs.orientation,
                                       series_visibility=visibility,
                                       series_colors=fs.series_color_overrides,
                                       plot_type=fs.plot_type)

        mask_specs = _compute_mask_specs(fs, df, mask_active)

        t0 = time.perf_counter()
        fig = build_data_overlay_figure(
            fs.image_rgb, traces, cal,
            edit_point_ids=edited_ids,
            image_data_uri=_get_masked_image_uri(fid, mask_specs),
            plot_type=fs.plot_type,
            orientation=fs.orientation,
        )
        _trace("overlay_plot.build", ms=int((time.perf_counter() - t0) * 1000))
        widget = go.FigureWidget(fig)
        try:
            # Box Select drives multi-selection — make sure the modebar
            # offers it even if a Plotly default ever drops it for this
            # trace mix.
            widget._config = {
                **getattr(widget, "_config", {}),
                "modeBarButtonsToAdd": ["select2d"],
            }
        except Exception as exc:
            _trace("overlay_plot.config_error", error=repr(exc))
        return widget

    @render_widget
    def zoom_bubble():
        """Zoomed inset centred on the selected overlay point or error-bar cap.

        Rebuilds when the selection changes or the overlay is revised (Apply /
        Reset). Arrow-key edits update it in-place via _push_zoom_bubble_update
        so there is no per-keystroke rebuild.
        """
        sel = selected_overlay_rv()
        _ = overlay_revision()
        fid = file_id_rv()
        _empty = go.FigureWidget(go.Figure(layout=dict(
            height=260, plot_bgcolor="#fafafa",
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )))
        if sel is None or fid is None:
            return _empty
        fs = pv.state.files.get(fid)
        if fs is None or fs.overlay is None:
            return _empty
        cal = cal_dict_from_result(fs.detection_result)
        if not cal.get("applied"):
            return _empty
        pid = _sel_anchor(sel)
        part = _sel_part(sel)
        pt = next((p for p in fs.overlay.points() if p.point_id == pid), None)
        if pt is None:
            return _empty
        # Match the main overlay's masked image so the zoom inset shows the
        # same composite the user is comparing against.
        bubble_uri = _get_image_uri(fid)
        if fs.csv_df is not None:
            names = fs.csv_df["series"].drop_duplicates().tolist()
            mask_active: dict[str, bool] = {}
            if fs.plot_type == "forest":
                mask_all = _safe_bool(lambda: input.forest_mask(), False)
                mask_active = {n: mask_all for n in names}
            else:
                for name in names:
                    try:
                        mask_active[name] = bool(input[_mask_id(name)]())
                    except Exception:
                        mask_active[name] = False
            mask_specs = _compute_mask_specs(fs, fs.csv_df, mask_active)
            bubble_uri = _get_masked_image_uri(fid, mask_specs)
        fig = build_zoom_bubble_figure(
            fs.image_rgb, cal, pt, part,
            image_data_uri=bubble_uri,
            plot_type=fs.plot_type,
            orientation=fs.orientation,
        )
        widget = go.FigureWidget(fig)
        try:
            widget._config = {
                **getattr(widget, "_config", {}),
                "displayModeBar": False,
            }
        except Exception as exc:
            _trace("zoom_bubble.config_error", error=repr(exc))
        return widget

    @render.ui
    def series_visibility_panel():
        # Deliberately NOT subscribed to overlay_revision: this panel *writes*
        # to it (via _sync_series_delta_e), so re-rendering on it would tear
        # down and re-seed the controls mid-interaction — the ΔE field would
        # snap back to a stale value and the checkboxes would reset to their
        # defaults. csv_revision + file_id cover every case where the series
        # list itself changes.
        _ = csv_revision()
        fid = file_id_rv()
        fs = pv.state.files.get(fid) if fid is not None else None
        if fs is None or fs.csv_df is None:
            return ui.tags.em("Load a CSV to toggle series visibility.")
        if fs.plot_type == "forest":
            # One compact control drives all rows: a forest CSV can carry
            # dozens of rows, so 50 per-row widgets would be unusable.
            n_rows = len(fs.csv_df["series"].drop_duplicates())
            de_val = DEFAULT_DELTA_E
            for name in fs.csv_df["series"].drop_duplicates().tolist():
                de_val = fs.series_delta_e.get(name, DEFAULT_DELTA_E)
                break
            return ui.div(
                ui.div(
                    ui.tags.strong("All rows"),
                    ui.tags.span(f" ({n_rows})",
                                 style="color:#888;font-size:12px;"),
                    style="margin-bottom:6px;",
                ),
                ui.input_checkbox("forest_vis", "Show overlay", value=True),
                ui.input_checkbox("forest_mask", "Mask source colors",
                                  value=False),
                ui.input_numeric("forest_de", "ΔE threshold",
                                 value=int(de_val), min=DELTA_E_MIN,
                                 max=DELTA_E_MAX, step=1, width="90px"),
            )
        header = ui.div(
            ui.div("", style="flex:1 1 auto;"),
            ui.div("Overlay", style="width:60px;text-align:center;font-size:11px;color:#555;"),
            ui.div("Mask", style="width:50px;text-align:center;font-size:11px;color:#555;"),
            ui.div("ΔE", style="width:80px;text-align:center;font-size:11px;color:#555;"),
            style=("display:flex;align-items:center;gap:6px;"
                   "padding:2px 0;border-bottom:1px solid #ddd;"),
        )
        items = [header]
        for name in fs.csv_df["series"].drop_duplicates().tolist():
            de_val = fs.series_delta_e.get(name, DEFAULT_DELTA_E)
            row = ui.div(
                ui.div(str(name),
                       style=("flex:1 1 auto;min-width:0;overflow:hidden;"
                              "text-overflow:ellipsis;white-space:nowrap;"
                              "font-size:13px;")),
                ui.div(
                    ui.input_checkbox(_vis_id(name), None, value=True),
                    style="width:60px;display:flex;justify-content:center;",
                ),
                ui.div(
                    ui.input_checkbox(_mask_id(name), None, value=False),
                    style="width:50px;display:flex;justify-content:center;",
                ),
                ui.div(
                    ui.input_numeric(_delta_e_id(name), None,
                                     value=int(de_val), min=DELTA_E_MIN,
                                     max=DELTA_E_MAX, step=1, width="70px"),
                    class_="pv-de-cell",
                    style="width:80px;display:flex;justify-content:center;",
                ),
                style=("display:flex;align-items:center;gap:6px;"
                       "padding:2px 0;border-bottom:1px solid #eee;"),
            )
            items.append(row)
        return ui.div(*items)

    @reactive.effect
    @reactive.event(input.overlay_legend_toggle)
    def _on_overlay_legend_toggle():
        """Mirror a Plotly legend click onto the series' Overlay checkbox."""
        msg = input.overlay_legend_toggle() or {}
        clicked = str(msg.get("series") or "")
        fid = file_id_rv()
        fs = pv.state.files.get(fid) if fid is not None else None
        if not clicked or fs is None or fs.csv_df is None:
            return
        match = next((n for n in fs.csv_df["series"].drop_duplicates().tolist()
                      if str(n) == clicked), None)
        if match is None:
            _trace("legend_toggle.unknown_series", series=clicked)
            return
        vid = _vis_id(match)
        ui.update_checkbox(vid, value=not _safe_bool(
            lambda: input[vid](), True, vid))

    @render.ui
    def dashboard_panel():
        fid = file_id_rv()
        _ = csv_revision()
        _ = overlay_revision()
        fs = pv.state.files.get(fid) if fid else None
        if fs is None or fs.csv_df is None:
            return None

        df = fs.csv_df

        if fs.plot_type == "scatter":
            stats = compute_scatter_stats(df)
            rows = []
            for sname, s in stats["by_series"].items():
                r_str = f"{s['r']:.4f}" if math.isfinite(s["r"]) else "—"
                r2_str = f"{s['r2']:.4f}" if math.isfinite(s["r2"]) else "—"
                rows.append(ui.tags.tr(
                    ui.tags.td(sname),
                    ui.tags.td(str(s["n"])),
                    ui.tags.td(r_str),
                    ui.tags.td(r2_str),
                ))
            if len(stats["by_series"]) > 1:
                ov = stats["overall"]
                ov_r = f"{ov['r']:.4f}" if math.isfinite(ov["r"]) else "—"
                ov_r2 = f"{ov['r2']:.4f}" if math.isfinite(ov["r2"]) else "—"
                rows.append(ui.tags.tr(
                    ui.tags.th("Overall", scope="row"),
                    ui.tags.td(str(ov["n"])),
                    ui.tags.td(ov_r),
                    ui.tags.td(ov_r2),
                ))
            return ui.card(
                ui.card_header("Correlation"),
                ui.tags.table(
                    ui.tags.thead(ui.tags.tr(
                        ui.tags.th("Series"),
                        ui.tags.th("n"),
                        ui.tags.th("r"),
                        ui.tags.th("R²"),
                    )),
                    ui.tags.tbody(*rows),
                    class_="table table-sm table-striped",
                ),
                style="margin-top:12px;",
            )

        if fs.plot_type == "forest":
            # Value-as-estimate: the σ machinery treats the interval as
            # bracketing the estimate, which in forest mode lives in `x`.
            # Default to the raw view ("None") until the user picks a real type.
            eff = fs.effective_error_bar_type
            default_eb = eff if eff in VALID_ERROR_TYPES else "None"
            try:
                eb_type = input.forest_eb_type() or default_eb
            except Exception:
                eb_type = default_eb
            try:
                percent = float(input.forest_eb_percent() or fs.error_bar_percent)
            except Exception:
                percent = fs.error_bar_percent
            n_shared = _safe_int(lambda: input.forest_n(), 0, "forest_n") or None

            eb_arg = eb_type if eb_type in VALID_ERROR_TYPES else None
            cal = cal_dict_from_result(fs.detection_result)
            display_df = build_forest_display_df(
                df, eb_arg, percent, n_shared, cal.get("x_log_base")
            )

            needs_percent = eb_type in ("Confidence", "Prediction")
            needs_n = eb_type in ("Confidence", "Prediction", "SE")
            top_items = [
                ui.div(
                    ui.input_select(
                        "forest_eb_type", "Error bar type",
                        {"None": "None (raw)",
                         **{t: t for t in VALID_ERROR_TYPES}},
                        selected=eb_type,
                    ),
                    style="flex:0 0 auto;",
                ),
            ]
            if needs_percent:
                top_items.append(ui.div(
                    ui.input_numeric("forest_eb_percent", "Percent",
                                     value=percent, min=50, max=99.9, step=0.5),
                    style="flex:0 0 auto;width:110px;",
                ))
            if needs_n:
                top_items.append(ui.div(
                    ui.input_numeric("forest_n", "n (per row)",
                                     value=n_shared, min=2, step=1),
                    style="flex:0 0 auto;width:120px;",
                ))
            controls = ui.div(
                *top_items,
                style="display:flex;flex-wrap:wrap;gap:12px;align-items:flex-end;",
            )
            if display_df.empty:
                table_ui = ui.tags.em("No forest data available.")
            else:
                html_str = display_df.to_html(
                    classes=["table", "table-sm", "table-striped", "table-bordered"],
                    float_format=lambda x: f"{x:.4g}",
                    na_rep="—",
                    border=0,
                    index=False,
                )
                table_ui = ui.div(
                    ui.HTML(html_str),
                    style="overflow-x:auto;font-size:13px;width:100%;",
                )
            return ui.card(
                ui.card_header("Forest estimates"),
                controls,
                ui.hr(),
                table_ui,
                style="margin-top:12px;overflow:visible;",
            )

        if fs.plot_type == "box":
            series_names = list(dict.fromkeys(df["series"].astype(str).tolist()))
            rows = []
            for sname in series_names:
                sdf = df[df["series"] == sname]
                n = len(sdf)
                non_outlier = sdf[sdf.get("status", pd.Series(dtype=str)).str.lower() != "outlier"] if "status" in sdf.columns else sdf
                n_outlier = n - len(non_outlier)
                q1_str = f"{non_outlier['box_q1'].median():.4g}" if "box_q1" in sdf.columns and len(non_outlier) else "—"
                med_str = f"{non_outlier['box_median'].median():.4g}" if "box_median" in sdf.columns and len(non_outlier) else "—"
                q3_str = f"{non_outlier['box_q3'].median():.4g}" if "box_q3" in sdf.columns and len(non_outlier) else "—"
                rows.append(ui.tags.tr(
                    ui.tags.td(sname),
                    ui.tags.td(str(n)),
                    ui.tags.td(q1_str),
                    ui.tags.td(med_str),
                    ui.tags.td(q3_str),
                    ui.tags.td(str(n_outlier)),
                ))
            return ui.card(
                ui.card_header("Box plot summary"),
                ui.tags.table(
                    ui.tags.thead(ui.tags.tr(
                        ui.tags.th("Series"),
                        ui.tags.th("n"),
                        ui.tags.th("Q1"),
                        ui.tags.th("Median"),
                        ui.tags.th("Q3"),
                        ui.tags.th("Outliers"),
                    )),
                    ui.tags.tbody(*rows),
                    class_="table table-sm table-striped",
                ),
                style="margin-top:12px;",
            )

        if fs.plot_type == "kaplan_meier":
            series_names = list(dict.fromkeys(df["series"].astype(str).tolist()))
            rows = []
            for sname in series_names:
                sdf = df[df["series"] == sname]
                n = len(sdf)
                n_censored = int((sdf["status"].str.lower() == "censored").sum()) if "status" in sdf.columns else 0
                n_events = n - n_censored
                rows.append(ui.tags.tr(
                    ui.tags.td(sname),
                    ui.tags.td(str(n)),
                    ui.tags.td(str(n_events)),
                    ui.tags.td(str(n_censored)),
                ))
            return ui.card(
                ui.card_header("Kaplan-Meier summary"),
                ui.tags.table(
                    ui.tags.thead(ui.tags.tr(
                        ui.tags.th("Arm"),
                        ui.tags.th("n"),
                        ui.tags.th("Events"),
                        ui.tags.th("Censored"),
                    )),
                    ui.tags.tbody(*rows),
                    class_="table table-sm table-striped",
                ),
                style="margin-top:12px;",
            )

        # ---- Time series dashboard (also used for bar) ----
        default_eb = fs.effective_error_bar_type or "SD"
        try:
            eb_type = input.eb_type_input() or default_eb
        except Exception:
            eb_type = default_eb

        try:
            percent = float(input.eb_percent_input() or fs.error_bar_percent)
        except Exception:
            percent = fs.error_bar_percent

        try:
            display_x = input.display_x_input() or "None"
        except Exception:
            display_x = "None"

        series_names = list(dict.fromkeys(df["series"].astype(str).tolist()))
        n_per_series: dict = {}
        for sname in series_names:
            safe = _safe_series_token(sname)
            try:
                val = input[f"n_input_{safe}"]()
                n_per_series[sname] = int(val) if val is not None else None
            except Exception:
                n_per_series[sname] = None

        needs_percent = eb_type in ("Confidence", "Prediction")
        needs_n = eb_type in ("Confidence", "Prediction", "SE")

        cal = cal_dict_from_result(fs.detection_result)
        y_log_base = cal.get("y_log_base")

        stats_df = df
        if fs.plot_type == "bar" and _fs_is_horizontal(fs):
            # Horizontal bars carry the value in `x` (errors already bracket
            # it), so feed the σ machinery a swapped view and use the value
            # axis's (x) log base.
            stats_df = df.rename(columns={"x": "y", "y": "x"})
            y_log_base = cal.get("x_log_base")

        display_df = build_time_series_display_df(
            stats_df, eb_type, percent, n_per_series, y_log_base, display_x
        )

        # Top row: error bar type + percent + n inputs
        top_row_items = [
            ui.div(
                ui.input_select(
                    "eb_type_input", "Error bar type",
                    {t: t for t in VALID_ERROR_TYPES},
                    selected=eb_type,
                ),
                style="flex:0 0 auto;",
            ),
        ]
        if needs_percent:
            top_row_items.append(ui.div(
                ui.input_numeric("eb_percent_input", "Percent",
                                  value=percent, min=50, max=99.9, step=0.5),
                style="flex:0 0 auto;width:110px;",
            ))
        if needs_n:
            for sname in series_names:
                n_val = n_per_series.get(sname)
                top_row_items.append(ui.div(
                    ui.input_numeric(
                        f"n_input_{_safe_series_token(sname)}",
                        f"n ({sname})",
                        value=n_val, min=2, step=1,
                    ),
                    style="flex:0 0 auto;width:120px;",
                ))

        controls = ui.div(
            ui.div(
                *top_row_items,
                style="display:flex;flex-wrap:wrap;gap:12px;align-items:flex-end;",
            ),
            ui.div(
                ui.input_select(
                    "display_x_input", "Display x",
                    {"None": "None", "Single column": "Single column", "Multi column": "Multi column"},
                    selected=display_x,
                ),
                style="margin-top:8px;",
            ),
        )

        if display_df.empty:
            table_ui = ui.tags.em("No error bar data available.")
        else:
            html_str = display_df.to_html(
                classes=["table", "table-sm", "table-striped", "table-bordered"],
                float_format=lambda x: f"{x:.4g}",
                na_rep="—",
                border=0,
            )
            table_ui = ui.div(
                ui.HTML(html_str),
                style="overflow-x:auto;font-size:13px;width:100%;",
            )

        return ui.card(
            ui.card_header("Data summary"),
            controls,
            ui.hr(),
            table_ui,
            style="margin-top:12px;overflow:visible;",
        )

    @reactive.calc
    def _active_series_info():
        """Series-level info for the active file, recomputed when CSV changes.

        Returns ``None`` when no CSV is loaded; otherwise a list of dicts:
        ``{"name": str, "color": "#hex"}``.
        """
        _ = csv_revision()
        _ = overlay_revision()
        fid = file_id_rv()
        if fid is None:
            return None
        fs = pv.state.files.get(fid)
        if fs is None or fs.csv_df is None:
            return None
        df = fs.csv_df
        out = []
        for name in df["series"].drop_duplicates().tolist():
            color = fs.series_color_overrides.get(name) or str(
                df[df["series"] == name]["series_color"].iloc[0]
            )
            out.append({"name": str(name), "color": color})
        return out

    @render.ui
    def series_color_panel():
        """Per-series color picker (Calibration tab).

        Native HTML color input is wired to Shiny via the pv-color-picker JS
        binding in _ANCHOR_KEY_SCRIPT. The ΔE slider and mask toggle live on
        the Overlay tab next to the visibility checkboxes.
        """
        info = _active_series_info()
        if info is None:
            return ui.tags.em("Load a CSV to pick colors.")
        fid = file_id_rv()
        fs = pv.state.files.get(fid) if fid is not None else None
        if fs is None:
            return ui.tags.em("Load a CSV to pick colors.")

        rows = []
        for item in info:
            name = item["name"]
            initial = item["color"]
            picker = ui.HTML(
                f'<input type="color" class="pv-color-picker" '
                f'data-input-id="{_color_id(name)}" '
                f'value="{initial}" '
                f'style="width:28px;height:28px;border:1px solid #aaa;'
                f'padding:0;background:none;cursor:pointer;vertical-align:middle;">'
            )
            label = ui.tags.span(
                name,
                style=("flex:1 1 auto;min-width:0;overflow:hidden;"
                       "text-overflow:ellipsis;white-space:nowrap;"
                       "padding:0 8px;font-size:13px;"),
            )
            rows.append(ui.div(
                picker, label,
                style=("display:flex;align-items:center;gap:6px;"
                       "padding:4px 0;border-bottom:1px solid #eee;"),
            ))
        if not fs.csv_has_series_color and not fs.series_color_overrides:
            rows.insert(0, ui.tags.small(
                "Pick a color for a series to enable its mask toggle on the "
                "Overlay tab.",
                style="color:#888;display:block;margin-bottom:6px;",
            ))

        bg_initial = fs.background_color_override
        bg_picker = ui.HTML(
            f'<input type="color" class="pv-color-picker" '
            f'data-input-id="{BG_COLOR_INPUT_ID}" '
            f'value="{bg_initial}" '
            f'style="width:28px;height:28px;border:1px solid #aaa;'
            f'padding:0;background:none;cursor:pointer;vertical-align:middle;">'
        )
        bg_label = ui.tags.span(
            "Background",
            style=("flex:1 1 auto;min-width:0;overflow:hidden;"
                   "text-overflow:ellipsis;white-space:nowrap;"
                   "padding:0 8px;font-size:13px;"),
        )
        rows.append(ui.div(
            ui.tags.hr(style="margin:6px 0;"),
            ui.div(
                bg_picker, bg_label,
                style=("display:flex;align-items:center;gap:6px;padding:4px 0;"),
            ),
        ))

        return ui.div(*rows)

    @reactive.effect
    def _sync_series_color_overrides():
        """Mirror per-series color picker values into PerFileState.

        Subscribes to csv_revision so the effect re-runs *after* the panel
        mounts and the dynamic color inputs exist — that's when we capture
        the per-input reactive dependencies that fire on subsequent picks.
        """
        _ = csv_revision()
        fid = file_id_rv()
        fs = pv.state.files.get(fid) if fid is not None else None
        if fs is None or fs.csv_df is None:
            return
        changed = False
        for name in fs.csv_df["series"].drop_duplicates().tolist():
            try:
                val = input[_color_id(name)]()
            except Exception:
                continue
            if not isinstance(val, str) or not val.startswith("#"):
                continue
            # Treat the picker value as an "intentional" choice only when it
            # differs from the auto-palette default. Otherwise we'd flip every
            # series into masking just by mounting the panel.
            csv_default = str(fs.csv_df[fs.csv_df["series"] == name]
                              ["series_color"].iloc[0])
            if val.lower() == csv_default.lower() and not fs.csv_has_series_color:
                # User has not actually picked — drop any prior override.
                if fs.series_color_overrides.pop(name, None) is not None:
                    changed = True
                continue
            if fs.series_color_overrides.get(name) != val:
                fs.series_color_overrides[name] = val
                changed = True
        if changed:
            with reactive.isolate():
                _cur = overlay_revision()
            overlay_revision.set(_cur + 1)

    @reactive.effect
    def _sync_bg_color():
        """Mirror background color picker into PerFileState."""
        try:
            val = input[BG_COLOR_INPUT_ID]()
        except Exception:
            return
        if not isinstance(val, str) or not val.startswith("#"):
            return
        fid = file_id_rv()
        fs = pv.state.files.get(fid) if fid is not None else None
        if fs is None:
            return
        if fs.background_color_override != val:
            fs.background_color_override = val
            masked_image_uri_cache.clear()
            with reactive.isolate():
                _cur = overlay_revision()
            overlay_revision.set(_cur + 1)

    @reactive.effect
    def _sync_series_delta_e():
        """Mirror ΔE fields into PerFileState. See sibling effect for csv_revision rationale."""
        _ = csv_revision()
        fid = file_id_rv()
        fs = pv.state.files.get(fid) if fid is not None else None
        if fs is None or fs.csv_df is None:
            return
        touched: list[str] = []
        for name in fs.csv_df["series"].drop_duplicates().tolist():
            try:
                raw = input[_delta_e_id(name)]()
            except Exception:
                continue
            val = _coerce_delta_e(raw)
            if val is None:
                continue
            if fs.series_delta_e.get(name) != val:
                fs.series_delta_e[name] = val
                touched.append(name)
        if not touched:
            return
        # A threshold only changes pixels for a series that is currently
        # masked; editing it with the mask off just parks the value in state.
        # Read the mask toggles isolated so this effect stays driven purely by
        # the ΔE fields.
        with reactive.isolate():
            repaint = any(
                _safe_bool(lambda n=n: input[_mask_id(n)](), False)
                for n in touched
            )
        if repaint:
            _bump_overlay()

    @render.ui
    def overlay_selection_status():
        sel = selected_overlay_rv()
        if sel is None:
            return ui.tags.small(
                "Click a point (Shift-click or Box Select for several) to "
                "select it.",
                style="color:#888; display:block; margin-bottom:4px;",
            )
        pids = _sel_pids(sel)
        anchor = _sel_anchor(sel)
        if len(pids) > 1:
            return ui.div(
                ui.tags.small(
                    ui.tags.strong(f"{len(pids)} points selected"),
                    f" — arrows move all; typed edits apply to {anchor}.",
                    style="color:#333;",
                ),
                style="margin-bottom:4px;",
            )
        part_label = {
            "center": "center point",
            "upper": "upper error bar",
            "lower": "lower error bar",
        }.get(_sel_part(sel), "center point")
        return ui.div(
            ui.tags.small(
                ui.tags.strong("Editing: "),
                f"{anchor} — {part_label}",
                style="color:#333;",
            ),
            style="margin-bottom:4px;",
        )

    @render.ui
    def edit_point_panel():
        # Static across edits: rebuilds only when the file or its data
        # change (pids are stable across edits/resets), never on
        # overlay_revision — values flow through ui.update_numeric instead,
        # which keeps typing and held arrow keys smooth.
        fid = file_id_rv()
        _ = csv_revision()
        fs = pv.state.files.get(fid) if fid is not None else None
        if fs is None or fs.overlay is None:
            return ui.tags.em("Load data to enable point editing.")
        ids = [p.point_id for p in fs.overlay.points()]
        if not ids:
            return ui.tags.em("No points loaded.")
        with reactive.isolate():
            sel = selected_overlay_rv()
        initial_pid = (_sel_anchor(sel) if sel and _sel_anchor(sel) in ids
                       else ids[0])
        init_pt = next((p for p in fs.overlay.points()
                        if p.point_id == initial_pid), None)
        init = (_model_vals(fs, init_pt) if init_pt is not None
                else PointVals(0.0, 0.0, None, None))
        init_center = init.x if _fs_is_horizontal(fs) else init.y
        init_hw = half_width_of(init_center, init.upper, init.lower)
        # Preserve settings across rebuilds (file switches).
        with reactive.isolate():
            cur_step = _safe_float(input.overlay_arrow_step,
                                    _pixel_step_for_file(fs), "overlay_arrow_step")
            cur_link = _safe_bool(input.link_bounds, False, "link_bounds")
        # Forest y is the fixed row index; editing it would break the even
        # row spacing, so the field is labelled locked and ignored on Apply.
        is_forest = fs.plot_type == "forest"
        y_label = "y (row, locked)" if is_forest else "y"
        return ui.div(
            ui.output_ui("overlay_selection_status"),
            ui.input_select("edit_point_id", "Point", choices=ids,
                            selected=initial_pid),
            ui.row(
                ui.column(6, ui.input_numeric("edit_point_x", "x",
                                               value=init.x)),
                ui.column(6, ui.input_numeric("edit_point_y", y_label,
                                               value=init.y)),
            ),
            # Both interval representations stay in the DOM; the checkbox
            # only toggles visibility, so no re-render on switch.
            ui.panel_conditional(
                "!input.link_bounds",
                ui.row(
                    ui.column(6, ui.input_numeric(
                        "edit_err_lower", "Lower bound",
                        value=init.lower if init.lower is not None else 0.0)),
                    ui.column(6, ui.input_numeric(
                        "edit_err_upper", "Upper bound",
                        value=init.upper if init.upper is not None else 0.0)),
                ),
            ),
            ui.panel_conditional(
                "input.link_bounds",
                ui.row(
                    ui.column(6, ui.input_numeric(
                        "edit_half_width", "± half-width",
                        value=init_hw if init_hw is not None else 0.0,
                        min=0.0)),
                ),
            ),
            ui.row(
                ui.column(6, ui.input_numeric(
                    "overlay_arrow_step", "Arrow step",
                    value=cur_step, step=0.01, min=0.0001,
                )),
                ui.column(
                    6,
                    ui.input_checkbox("link_bounds", "Symmetric interval",
                                       value=cur_link),
                    style="display:flex; align-items:flex-end; padding-bottom:8px;",
                ),
            ),
            ui.tags.small(
                "Arrow keys nudge the selection (Shift = 10×); moving the "
                "center keeps the interval width. With a symmetric interval, "
                "nudging one bound mirrors the other.",
                style="color:#666; display:block; margin-bottom:4px;",
            ),
            ui.input_action_button("apply_point_edit", "Apply edit",
                                    class_="btn-primary btn-sm"),
            ui.input_action_button("reset_point_edit", "Reset point",
                                    class_="btn-sm"),
        )

    @reactive.effect
    def _update_reset_button_label():
        """`Reset point` becomes `Reset selected (N)` for multi-selections."""
        sel = selected_overlay_rv()
        n = len(_sel_pids(sel))
        label = f"Reset selected ({n})" if n > 1 else "Reset point"
        ui.update_action_button("reset_point_edit", label=label)

    @reactive.effect
    @reactive.event(input.link_bounds)
    def _seed_half_width_on_link():
        """Toggling the symmetric-interval checkbox seeds the half-width from
        the current bounds (mean of the two offsets when asymmetric). The
        bounds themselves only change on the next edit, not on the toggle."""
        if not _safe_bool(input.link_bounds, False, "link_bounds"):
            return
        with reactive.isolate():
            fid = file_id_rv()
        fs = pv.state.files.get(fid) if fid is not None else None
        if fs is None or fs.overlay is None:
            return
        pid = _safe_str(input.edit_point_id, "", "edit_point_id")
        pt = next((p for p in fs.overlay.points() if p.point_id == pid), None)
        if pt is None:
            return
        vals = _model_vals(fs, pt)
        center = vals.x if _fs_is_horizontal(fs) else vals.y
        hw = half_width_of(center, vals.upper, vals.lower)
        if hw is not None:
            ui.update_numeric("edit_half_width", value=hw)

    # Whenever the user changes the selected point, populate the inputs with
    # the current values so editing is incremental.
    @reactive.effect
    @reactive.event(input.edit_point_id)
    def _populate_point_editor():
        fid = file_id_rv()
        fs = pv.state.files.get(fid) if fid is not None else None
        if fs is None or fs.overlay is None:
            return
        pid = input.edit_point_id()
        pt = next((p for p in fs.overlay.points() if p.point_id == pid), None)
        if pt is None:
            return
        _sync_edit_inputs(fs, pid)
        # Sync selected_overlay_rv so arrow keys work when using the dropdown.
        # A genuine dropdown change (anchor differs) replaces the selection
        # with this single point; the programmatic update_select echo after a
        # click/box-select (anchor already matches) must NOT collapse a
        # multi-selection.
        with reactive.isolate():
            current_sel = selected_overlay_rv()
        if current_sel is None or _sel_anchor(current_sel) != pid:
            selected_overlay_rv.set({"pids": [pid], "anchor": pid,
                                     "part": "center"})

    @reactive.effect
    @reactive.event(input.apply_point_edit)
    def _apply_point_edit():
        fid = file_id_rv()
        fs = pv.state.files.get(fid) if fid is not None else None
        if fs is None or fs.overlay is None:
            return
        pid = input.edit_point_id()
        try:
            new_x = float(input.edit_point_x() or 0)
            new_y = float(input.edit_point_y() or 0)
            if fs.plot_type == "forest":
                # Row index is fixed — keep the point on its evenly-spaced row.
                cur_pt = next((p for p in fs.overlay.points()
                               if p.point_id == pid), None)
                if cur_pt is not None:
                    new_y = float(cur_pt.y)
            if _safe_bool(input.link_bounds, False, "link_bounds"):
                hw = _safe_float(input.edit_half_width, 0.0, "edit_half_width")
                center = new_x if _fs_is_horizontal(fs) else new_y
                new_lower, new_upper = linked_bounds(center, hw)
            else:
                new_upper = float(input.edit_err_upper() or 0)
                new_lower = float(input.edit_err_lower() or 0)
            fs.overlay.edit_point(pid, new_x, new_y)
            fs.overlay.edit_err_lower(pid, float(new_lower))
            fs.overlay.edit_err_upper(pid, float(new_upper))
        except Exception as e:
            ui.notification_show(f"Edit failed: {e}", type="error")
            return
        _bump_overlay()
        # The panel no longer rebuilds on overlay_revision — reflect any
        # derived values (linked bounds / half-width) back into the inputs.
        _sync_edit_inputs(fs, pid)

    @reactive.effect
    @reactive.event(input.reset_point_edit)
    def _reset_point_edit():
        fid = file_id_rv()
        fs = pv.state.files.get(fid) if fid is not None else None
        if fs is None or fs.overlay is None:
            return
        with reactive.isolate():
            sel = selected_overlay_rv()
        pids = _sel_pids(sel) or [input.edit_point_id()]
        try:
            fs.overlay.reset_points(pids)
        except Exception as e:
            ui.notification_show(f"Reset failed: {e}", type="error")
            return
        _bump_overlay()
        _sync_edit_inputs(fs, _sel_anchor(sel) or pids[-1])

    @reactive.effect
    @reactive.event(
        input.edit_point_x, input.edit_point_y,
        input.edit_err_upper, input.edit_err_lower,
        input.edit_half_width,
    )
    def _live_update_point_inputs():
        """Push input changes to the overlay widget immediately (no full rebuild).

        Fires whenever any of the numeric edit inputs change — whether from
        the user typing, arrowing, or a programmatic ui.update_numeric. The
        model-value comparison guards against spurious updates from the
        latter: if the inputs match the stored model values the change came
        from a populate/sync call, so we skip.
        """
        with reactive.isolate():
            fid = file_id_rv()
        if fid is None:
            return
        fs = pv.state.files.get(fid)
        if fs is None or fs.overlay is None:
            return
        try:
            pid = input.edit_point_id()
            new_x = float(input.edit_point_x())
            new_y = float(input.edit_point_y())
        except Exception as exc:
            _trace("live_update.read_inputs_error", error=repr(exc))
            return
        if pid is None:
            return
        pt = next((p for p in fs.overlay.points() if p.point_id == pid), None)
        if pt is None:
            return
        cur = _model_vals(fs, pt)
        linked = _safe_bool(input.link_bounds, False, "link_bounds")
        is_horiz = _fs_is_horizontal(fs)

        if linked:
            try:
                hw = float(input.edit_half_width())
            except Exception as exc:
                _trace("live_update.read_inputs_error", error=repr(exc))
                return
            cur_center = cur.x if is_horiz else cur.y
            cur_hw = half_width_of(cur_center, cur.upper, cur.lower)
            hw_echo = ((cur_hw is None and hw == 0.0)
                       or (cur_hw is not None and abs(hw - cur_hw) < 1e-10))
            if (abs(new_x - cur.x) < 1e-10 and abs(new_y - cur.y) < 1e-10
                    and hw_echo):
                return
            if cur_hw is None and hw == 0.0:
                # No interval on this point and none requested.
                new_lower = new_upper = None
            else:
                center = new_x if is_horiz else new_y
                new_lower, new_upper = linked_bounds(center, hw)
        else:
            try:
                new_upper = float(input.edit_err_upper())
                new_lower = float(input.edit_err_lower())
            except Exception as exc:
                _trace("live_update.read_inputs_error", error=repr(exc))
                return
            cur_upper = cur.upper if cur.upper is not None else 0.0
            cur_lower = cur.lower if cur.lower is not None else 0.0
            if (abs(new_x - cur.x) < 1e-10 and abs(new_y - cur.y) < 1e-10
                    and abs(new_upper - cur_upper) < 1e-10
                    and abs(new_lower - cur_lower) < 1e-10):
                return
        # Persist to model silently — no _bump_overlay, widget updated below.
        try:
            fs.overlay.edit_point(pid, new_x, new_y)
            if new_upper is not None:
                fs.overlay.edit_err_upper(pid, float(new_upper))
            if new_lower is not None:
                fs.overlay.edit_err_lower(pid, float(new_lower))
        except Exception as exc:
            _user_error("Live point edit failed", exc)
            return
        # Keep the hidden representation in sync so switching the checkbox
        # shows current numbers. Exact floats → the echo passes the guard.
        if linked:
            if new_upper is not None:
                ui.update_numeric("edit_err_upper", value=float(new_upper))
            if new_lower is not None:
                ui.update_numeric("edit_err_lower", value=float(new_lower))
        else:
            center = new_x if is_horiz else new_y
            hw_now = half_width_of(center, new_upper, new_lower)
            if hw_now is not None:
                ui.update_numeric("edit_half_width", value=hw_now)
        _push_point_edit_to_widget(pid, new_x, new_y, new_upper, new_lower)

    # ------------------------------------------------------------------
    # Overlay point selection (JS → Shiny.setInputValue → overlay_click)
    # ------------------------------------------------------------------

    @reactive.effect
    @reactive.event(input.overlay_click)
    def _on_overlay_click():
        """Handle a data-point click forwarded from JS via Shiny.setInputValue.

        The JS plotly_click handler in _ANCHOR_KEY_SCRIPT sends the pid,
        part ("center"/"upper"/"lower"), and whether Shift was held. Plain
        clicks replace the selection; shift-clicks toggle membership in the
        multi-selection (see ``_update_selection``).
        """
        payload = input.overlay_click()
        if not isinstance(payload, dict):
            return
        pid = payload.get("pid")
        part = payload.get("part", "center")
        shift = bool(payload.get("shift", False))
        if not pid:
            return
        with reactive.isolate():
            fid = file_id_rv()
            cur_sel = selected_overlay_rv()
        if fid is None:
            return
        fs = pv.state.files.get(fid)
        if fs is None or fs.overlay is None:
            return
        if not any(p.point_id == pid for p in fs.overlay.points()):
            return
        new_sel = _update_selection(cur_sel, pid, part, shift)
        anchor = _sel_anchor(new_sel)
        if anchor is not None:
            ui.update_select("edit_point_id", selected=anchor)
        selected_overlay_rv.set(new_sel)

    @reactive.effect
    @reactive.event(input.overlay_box_select)
    def _on_overlay_box_select():
        """Box/lasso select from Plotly replaces the selection with the set."""
        payload = input.overlay_box_select()
        if not isinstance(payload, dict):
            return
        pids = payload.get("pids")
        if not isinstance(pids, list) or not pids:
            return
        with reactive.isolate():
            fid = file_id_rv()
        if fid is None:
            return
        fs = pv.state.files.get(fid)
        if fs is None or fs.overlay is None:
            return
        known = {p.point_id for p in fs.overlay.points()}
        valid = [str(p) for p in pids if str(p) in known]
        if not valid:
            return
        anchor = valid[-1]
        ui.update_select("edit_point_id", selected=anchor)
        selected_overlay_rv.set({"pids": valid, "anchor": anchor,
                                 "part": "center"})

    @reactive.effect
    @reactive.event(input.overlay_deselect)
    def _on_overlay_deselect():
        selected_overlay_rv.set(None)

    # ------------------------------------------------------------------
    # Zoom bubble visibility sync
    # ------------------------------------------------------------------

    @reactive.effect
    async def _sync_bubble_visibility():
        sel = selected_overlay_rv()
        active_tab = _safe_str(input.main_nav, _DEFAULT_NAV, "main_nav")
        show = sel is not None and active_tab == "Overlay"
        await session.send_custom_message("pv_bubble_show", {"show": show})

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @render.download(filename=lambda: input.export_filename() or "corrected.csv")
    def export_csv():
        fid = file_id_rv()
        fs = pv.state.files.get(fid) if fid is not None else None
        if fs is None or fs.overlay is None:
            yield b"# No CSV loaded.\n"
            return
        data = pv.export_csv(fid, include_audit_cols=bool(input.include_audit_cols()))
        yield data

    @render.download(
        filename=lambda: (input.export_filename() or "corrected").rsplit(".", 1)[0] + ".json",
    )
    def export_json():
        fid = file_id_rv()
        fs = pv.state.files.get(fid) if fid is not None else None
        if fs is None or fs.overlay is None:
            yield b'{"error": "No data loaded."}\n'
            return
        data = pv.export_json(
            fid, include_audit_cols=bool(input.include_audit_cols()),
        )
        yield data


# ---------------------------------------------------------------------------
# Wrap and expose
# ---------------------------------------------------------------------------

app = App(_make_ui(), server)
