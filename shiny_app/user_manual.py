"""User Manual tab for the PlotVerify Shiny app.

A single ``ui.nav_panel`` containing an accordion of reference sections that
documents every tool in the workflow. All sections start collapsed.

Two gates shape the section list so the same source file serves every
deployment:

- OCR-only sections (auto-detection, axis-frame, label bands, detection
  settings, calibration points table) require a working EasyOCR install
  (``axis_pipeline.ocr_available()``).
- Calibration/plot-type/series-color sections are omitted entirely in
  JSON-only mode (``runtime_flags.json_only_mode()``) — the shinylive
  Pyodide deployment, where the Calibrate tab does not exist and the Agent
  JSON supplies calibration, plot type and orientation.
"""
from __future__ import annotations

from typing import Optional

from shiny import ui

from .runtime_flags import json_only_mode

try:
    from axis_pipeline import ocr_available
    _OCR_AVAILABLE = ocr_available()
except ImportError:
    _OCR_AVAILABLE = False

_JSON_ONLY = json_only_mode()


def _p(*body) -> ui.Tag:
    return ui.tags.p(*body, style="margin:0 0 8px 0;")


def _ul(*items) -> ui.Tag:
    return ui.tags.ul(
        *[ui.tags.li(it) for it in items],
        style="margin:0 0 8px 18px; padding:0;",
    )


def _code(text: str) -> ui.Tag:
    return ui.tags.code(text)


def _section(title: str, value: str, *body) -> ui.Tag:
    return ui.accordion_panel(title, *body, value=value)


def _getting_started(json_only: bool = False) -> ui.Tag:
    if json_only:
        return _section(
            "Getting started",
            "getting_started",
            _p("This deployment verifies AI-extracted plot data from a "
               "single Agent JSON file. Upload a ", _code(".json"),
               " file from the sidebar (or paste the JSON text) and click ",
               ui.tags.strong("Import JSON"), "."),
            _ul(
                "The JSON carries everything: the plot image (embedded as "
                "a data URI), the extracted data rows, the axis "
                "calibration, the plot type, and the orientation.",
                "After import you land directly on the Overlay tab with "
                "the data drawn over the source image — review, adjust "
                "points, and export.",
                "A JSON without an axes block cannot be calibrated here "
                "and is rejected with an error.",
                "The sidebar status block shows the active image filename, "
                "dimensions, row count, series count, and review status.",
            ),
        )
    return _section(
        "Getting started",
        "getting_started",
        _p("Upload a plot image, then a data CSV, from the sidebar. The image "
           "must be uploaded first; the CSV is attached to the most recently "
           "loaded image."),
        _ul(
            ui.TagList("Image formats: ", _code(".png"), ", ", _code(".jpg"),
                       ", ", _code(".jpeg"), ", ", _code(".tif"), ", ",
                       _code(".tiff"), ", ", _code(".bmp"), ", ",
                       _code(".webp"), ". Images over 4000 px on the long "
                       "edge or 25 MB are automatically downscaled to about "
                       "3000 px for display; a notification reports the "
                       "scale factor."),
            ui.TagList("CSV required columns: ", _code("x"), ", ",
                       _code("y"), "."),
            ui.TagList("CSV optional columns: ", _code("series"), " (defaults "
                       "to ", _code("Data"), " when absent), ",
                       _code("y_err_lower"), ", ", _code("y_err_upper"),
                       ", ", _code("series_color"), " (6-digit hex), ",
                       _code("error_bar_type"), " (SD / SE / Confidence / "
                       "Prediction)."),
            ui.TagList("Rows with non-numeric ", _code("x"), " or ",
                       _code("y"), " are dropped silently. Reversed error "
                       "bars (where lower > y or upper < y) are auto-swapped "
                       "and a notification reports how many were fixed."),
            "The sidebar status block below the uploads shows the active "
            "image filename, dimensions, attached CSV, row count, series "
            "count, and the per-file review status.",
        ),
    )


def _calibration_overview() -> ui.Tag:
    return _section(
        "Calibration — overview",
        "calibration_overview",
        _p("Calibration maps pixel coordinates on the image to data "
           "coordinates on the plot. The UI exposes two draggable anchor "
           "circles on the calibration image:"),
        _ul(
            ui.TagList(ui.tags.strong("P1"),
                       " (red) — top-left corner of the axis rectangle: "
                       "pixel position of the leftmost x-tick at the "
                       "topmost y-tick."),
            ui.TagList(ui.tags.strong("P2"),
                       " (green) — bottom-right corner: pixel position of "
                       "the rightmost x-tick at the bottommost y-tick."),
        ),
        _p("The bottom-left corner is implicit — it's derived as "
           "(P1.x, P2.y), so dragging P1 vertically also moves the "
           "baseline and dragging P2 horizontally also moves the right edge."),
        _p(ui.TagList(
            "Internally the model is ",
            _code("data = scale · pixel + offset"),
            " per axis. Y-axis pixels grow downward, so a normal upward "
            "y-axis has a negative scale. Calibration is applied to the "
            "downscaled image when one was generated on upload.",
        )),
    )


def _manual_calibration() -> ui.Tag:
    return _section(
        "Manual calibration — dragging anchors",
        "manual_calibration",
        _p("Drag P1 and P2 onto the correct tick positions on the image."),
        _ul(
            ui.TagList(ui.tags.strong("Keyboard nudge:"),
                       " click an anchor to select it, then arrow keys "
                       "move ±1 px; hold ", _code("Shift"),
                       " for ±10 px. Dashed guide lines on the image show "
                       "the rectangle implied by the anchors."),
            ui.TagList(ui.tags.strong("Reset anchors"),
                       " (button in the calibration card header) "
                       "returns the anchors to a 10%/90% rectangle inside "
                       "the image."),
            "Rectangle constraints are enforced live: a horizontal drag of "
            "one anchor moves the rectangle's vertical edge with it, and a "
            "vertical drag moves the baseline.",
            ui.TagList(
                "Dragging or nudging an anchor updates the pixel inputs in "
                "the ", ui.tags.strong("Manual Values"),
                " panel. The corresponding ",
                _code("data X"), " / ", _code("data Y"),
                " values you supplied are preserved.",
            ),
        ),
    )


def _manual_values() -> ui.Tag:
    return _section(
        "Manual Values panel",
        "manual_values",
        _p("Numeric inputs that mirror the on-image anchors and let you "
           "enter the data values at each tick."),
        _ul(
            ui.TagList(
                ui.tags.strong("P1 — top-left."), " ", _code("pixel x"),
                " / ", _code("pixel y"),
                " = anchor position. ", _code("data X"),
                " = x-axis value at the LEFT edge. ", _code("data Y"),
                " = y-axis value at the TOP edge.",
            ),
            ui.TagList(
                ui.tags.strong("P2 — bottom-right."), " ", _code("pixel x"),
                " / ", _code("pixel y"),
                " = anchor position. ", _code("data X"),
                " = x-axis value at the RIGHT edge. ", _code("data Y"),
                " = y-axis value at the BOTTOM edge.",
            ),
            ui.TagList(
                ui.tags.strong("X is log / Y is log"),
                " checkboxes — tick to mark an axis as log-scale. The ",
                _code("Base"),
                " field accepts a positive number greater than 1 (",
                _code("10"), ", ", _code("e"), ", ", _code("2"), ", …). "
                "Enter the data values in their natural linear units; the "
                "calibration handles the log transform internally.",
            ),
            ui.TagList(
                ui.tags.strong("Apply manual calibration"),
                " — required after typing values to commit the calibration. "
                "Pure anchor drags do not auto-apply; nothing downstream "
                "(overlay, dashboard) updates until you click this button.",
            ),
        ),
    )


def _auto_calibration() -> ui.Tag:
    return _section(
        "Automatic calibration — \"Run detection\"",
        "auto_calibration",
        _p("Runs the full multi-phase OCR + geometry pipeline and applies "
           "the result. Available when EasyOCR is installed (a green banner "
           "in the sidebar confirms this)."),
        _p("Phases:"),
        _ul(
            "Phase A — full-image OCR discovery scan, then mask all "
            "detected text so it doesn't interfere with line detection.",
            "Geometric axis-frame detection on the masked image "
            "(projection profiles + Hough).",
            "Phase B — re-OCR a tight band left of the y-axis with a "
            "numeric allowlist.",
            "Phase C — re-OCR a tight band below the x-axis with a "
            "numeric allowlist.",
            "Geometric tick detection, then a modal-spacing grid fit to "
            "reject non-tick peaks.",
            "Spatial one-to-one pairing of OCR labels to grid ticks "
            "(with monotonicity enforced).",
            ui.TagList(
                "Per-axis regression of ", _code("data = scale · pixel + offset"),
                " (OLS, with Student-t MLE for robustness).",
            ),
        ),
        _p("On success the result is summarized under the calibration image "
           "(mode, confidence, X/Y scale + offset, log base if any) and the "
           "anchors snap to the detected P1/P2 ticks. Review the result and "
           "edit individual fields in Manual Values if a tick was misread, "
           "then click Apply."),
        _p(ui.TagList(
            "Note: ", ui.tags.strong("Detect axis frame"),
            " also runs Phase A automatically on every image upload (so the "
            "anchors start at the detected rectangle corners). The "
            "\"Run detection\" button is for the full multi-phase pipeline.",
        )),
    )


def _detect_frame() -> ui.Tag:
    return _section(
        "\"Detect axis frame\" button",
        "detect_frame",
        _p("Runs Phase A and geometric frame detection only — no OCR "
           "regression, no calibration. Useful to re-seed the X/Y label "
           "bands and reposition the anchors at the detected rectangle "
           "corners after a manual edit."),
        _p("This also runs implicitly on every image upload; clicking the "
           "button is only needed if you want to re-detect after the user "
           "has dragged the anchors away from the frame."),
    )


def _label_bands() -> ui.Tag:
    return _section(
        "X/Y label bands",
        "label_bands",
        _p("Two rectangles overlaid on the image marking the regions that "
           "Phase B (y-tick labels, left of the axis) and Phase C (x-tick "
           "labels, below the axis) re-OCR. The bands only appear after an "
           "axis frame has been detected."),
        _p(ui.tags.strong("Y-label band:")),
        _ul(
            ui.TagList(_code("Left ext. (px)"),
                       " — how far left of the y-axis the band reaches. "
                       "Default 90."),
            ui.TagList(_code("V. trim (px)"),
                       " — shrink the band's vertical extent from the top "
                       "and bottom ends. Default 0."),
            ui.TagList(_code("H. slide (px)"),
                       " — shift the entire band horizontally without "
                       "resizing it. Default 0."),
        ),
        _p(ui.tags.strong("X-label band:")),
        _ul(
            ui.TagList(_code("Below ext. (px)"),
                       " — how far below the x-axis the band reaches. "
                       "Default 28."),
            ui.TagList(_code("H. trim (px)"),
                       " — shrink the band's horizontal extent from the "
                       "left and right ends. Default 0."),
            ui.TagList(_code("V. slide (px)"),
                       " — shift the entire band vertically without "
                       "resizing it. Default 0."),
        ),
        _p("Slider edits update the rectangles live on the image. The next "
           "\"Run detection\" call uses the current band values."),
    )


def _calibration_points() -> ui.Tag:
    return _section(
        "Calibration points panel",
        "calibration_points",
        _p("After a successful detection or Apply, two tables list the "
           "paired ticks the pipeline used:"),
        _ul(
            ui.TagList(_code("Data"), " — the data value of the tick."),
            ui.TagList(_code("Pixel"),
                       " — the pixel position of the tick."),
            ui.TagList(_code("Source"),
                       " — which OCR phase / heuristic produced the label "
                       "(useful for diagnosing misread ticks)."),
        ),
        _p("This view is read-only in Shiny. To edit individual paired "
           "ticks, use the legacy Streamlit app for now."),
    )


def _detection_settings() -> ui.Tag:
    return _section(
        "Detection settings & warnings",
        "detection_settings",
        _p("Bottom accordion on the Calibrate tab (collapsed by default)."),
        _ul(
            ui.TagList(ui.tags.strong("Min OCR confidence"),
                       " — discard OCR records below this score. Lower it "
                       "(e.g. 0.10) when labels are small or low-contrast; "
                       "raise it (e.g. 0.40) when stray text is being "
                       "misread as ticks. Default: 0.20."),
            ui.TagList(ui.tags.strong("Frame-detection warnings"),
                       " — any warning strings the pipeline emitted on the "
                       "last run (low confidence, ambiguous frame, missing "
                       "ticks). Empty when detection succeeded cleanly."),
        ),
    )


def _plot_type() -> ui.Tag:
    return _section(
        "Plot type",
        "plot_type",
        _p("Selects how the Overlay tab renders the data and which "
           "Dashboard layout appears. Located in the Calibrate tab's right "
           "accordion."),
        _ul(
            ui.TagList(
                ui.tags.strong("Time series w/ intervals"),
                " — points connected per series, error bars from ",
                _code("y_err_lower"), " / ", _code("y_err_upper"),
                ", and a shaded ribbon between the upper and lower bounds. "
                "Dashboard shows a Data summary table with σ converted "
                "from the half-widths.",
            ),
            ui.TagList(
                ui.tags.strong("Scatter plots"),
                " — points only, no connecting line. Dashboard shows "
                "Pearson r and R² per series and overall.",
            ),
            ui.TagList(
                ui.tags.strong("Forest plot"),
                " — one horizontal row per estimate with a horizontal "
                "confidence interval and a categorical vertical axis. "
                "Auto-selected when a forest CSV is loaded. See the ",
                ui.tags.em("Forest plots"), " section below.",
            ),
            ui.TagList(
                ui.tags.strong("Bar chart"),
                " — one marker per bar top with error bars; the Dashboard "
                "reuses the time-series σ table.",
            ),
            ui.TagList(
                ui.tags.strong("Box plot"),
                " — box/median/whisker glyphs from ", _code("box_q1"),
                " / ", _code("box_median"), " / ", _code("box_q3"),
                " plus the interval bounds as whisker ends; rows with ",
                _code("status = outlier"), " render as open diamonds.",
            ),
            ui.TagList(
                ui.tags.strong("Kaplan–Meier"),
                " — step curves with a step-shaped confidence band, "
                "censoring tick marks (", _code("status = censored"),
                "), and per-point ", _code("at_risk"), " counts.",
            ),
            ui.TagList(
                ui.tags.strong("Orientation"),
                " — bar and box plots can render horizontally (value axis "
                "along x). Orientation comes from the Agent JSON's ",
                _code("orientation"),
                " field; forest plots are always horizontal.",
            ),
        ),
    )


def _forest_plots() -> ui.Tag:
    return _section(
        "Forest plots",
        "forest_plots",
        _p("A forest plot lays out one estimate per horizontal row: each row "
           "has a point estimate, a horizontal confidence interval, and a "
           "categorical label on the vertical axis. PlotVerify auto-detects "
           "this layout and switches Plot type to Forest plot."),
        ui.tags.p(ui.tags.strong("CSV schema:"),
                  style="margin:8px 0 4px 0;"),
        _ul(
            ui.TagList(
                "Detected when the CSV has a ", _code("value"),
                " column and no ", _code("x"), " column. ",
                _code("value"), " → the estimate, ",
                _code("value_err_lower"), " / ", _code("value_err_upper"),
                " → the interval bounds.",
            ),
            ui.TagList(
                "The ", _code("series"), " column is the row label "
                "(e.g. ", _code("beta[0]"), "). Rows are placed top-to-bottom "
                "in CSV order — the first CSV row sits at the top.",
            ),
            ui.TagList(
                "Optional ", _code("is_summary"),
                " (bool) renders that row as a diamond marker; ",
                _code("status"),
                " (text) is appended to the point's hover label.",
            ),
        ),
        ui.tags.p(ui.tags.strong("Calibration — what to enter:"),
                  style="margin:8px 0 4px 0;"),
        _ul(
            ui.TagList(
                ui.tags.strong("data X"), " is the value axis. Set ",
                _code("P1 data X"), " to the value at P1's horizontal "
                "position (the left value gridline) and ", _code("P2 data X"),
                " to the value at P2 (the right value gridline).",
            ),
            ui.TagList(
                ui.tags.strong("data Y"), " is the row index. Rows are "
                "numbered so the top CSV row = ", _code("N-1"),
                " and the bottom row = ", _code("0"),
                " (N = number of rows). Place P1 on any row and enter that "
                "row's index as ", _code("P1 data Y"),
                "; place P2 on any other row and enter its index as ",
                _code("P2 data Y"), ".",
            ),
            "The two anchors do not have to be the very top and bottom rows — "
            "pick whichever two rows are easiest to line up. The remaining "
            "rows are spaced evenly between them by the linear y-calibration. "
            "As a convenience the data-Y fields are pre-filled with the full "
            "span (N-1 and 0), but you can change them. Click Apply when set.",
        ),
        ui.tags.p(ui.tags.strong("Overlay:"),
                  style="margin:8px 0 4px 0;"),
        _ul(
            "Each row draws a marker with a horizontal error bar and a "
            "low-opacity interval band. Because a forest CSV can carry dozens "
            "of rows, the Series panel collapses to a single compact control: "
            "one Show-overlay toggle, one Mask toggle, and one ΔE slider that "
            "apply to every row at once.",
            ui.TagList(
                "Click a marker to select its center, or click the left/right "
                "cap to select the lower/upper interval endpoint. Arrow keys "
                "then nudge the selection horizontally (Shift = 10×); you can "
                "also type exact values into ", _code("x"), ", ",
                _code("y_err_lower"), " and ", _code("y_err_upper"),
                " and click Apply.",
            ),
            ui.TagList(
                "The ", _code("y"),
                " field is the fixed row index and is locked, so editing only "
                "moves values along the horizontal axis and rows stay evenly "
                "spaced.",
            ),
        ),
        ui.tags.p(ui.tags.strong("Dashboard:"),
                  style="margin:8px 0 4px 0;"),
        _ul(
            "One row per estimate: label, estimate, CI lower/upper, and the "
            "interval half-width. A Status column appears when any row has a "
            "status note.",
            ui.TagList(
                "Set an ", _code("Error bar type"),
                " (SD / SE / Confidence / Prediction) and, where required, a "
                "shared ", _code("n"), " to add a σ column, computed with "
                "the same half-width→σ machinery as the time-series "
                "dashboard. Leave the type on ", _code("None (raw)"),
                " to show only the raw estimates and intervals.",
            ),
        ),
    )


def _series_colors() -> ui.Tag:
    return _section(
        "Series colors",
        "series_colors",
        _p("One color swatch per series in the CSV (Calibrate tab, right "
           "accordion). Click the swatch to pick a new color."),
        _ul(
            ui.TagList(
                "If the CSV had a ", _code("series_color"),
                " column, those colors are used. Otherwise an auto-palette "
                "is assigned (Plotly's D3 qualitative palette).",
            ),
            ui.TagList(
                "Picking a color marks the series as having an ",
                ui.tags.em("intentional override"),
                " — only then does the per-series mask toggle on the "
                "Overlay tab become meaningful. Selecting the auto-palette "
                "default again clears the override.",
            ),
            "Each rendered point on the overlay is drawn with the series "
            "color, then a smaller marker is drawn on top in the "
            "hue-complement color so the centers remain visible against "
            "the matching plot line.",
        ),
    )


def _mask_preview() -> ui.Tag:
    return _section(
        "Mask preview (ΔE)",
        "mask_preview",
        _p("Each series row on the Overlay tab's ", _code("Series"),
           " panel exposes a ", _code("Mask"),
           " checkbox and a ", _code("ΔE"), " slider (range 1–40, "
           "default 10). Enabling Mask repaints all source-image pixels "
           "within ΔE of the series color, replacing them with the "
           "detected background color so the underlying plot line "
           "visually \"disappears\" and the overlay points stand out."),
        ui.tags.p(ui.tags.strong("ΔE formula (CIE 1976 Lab):"),
                  style="margin:8px 0 4px 0;"),
        _ul(
            ui.TagList(
                _code("ΔE = √((L − L₀)² + (a − a₀)² + (b − b₀)²)"),
                " — Euclidean distance in CIE Lab between each pixel and "
                "the chosen color.",
            ),
            ui.TagList(
                "Pixels with ", _code("ΔE < threshold"),
                " form the mask; the mask is then dilated 2× with a 3×3 "
                "kernel to fill anti-aliasing fringes around line edges.",
            ),
            ui.TagList(
                "Background color is auto-detected as the modal grey "
                "level in the source image; masked pixels are repainted "
                "with that grey.",
            ),
            "Mask mode is only available for series with an intentional "
            "color (CSV-supplied or user-picked). Auto-palette defaults "
            "don't match real pixels in the image.",
        ),
        _p("Multiple series masks are combined with a bitwise OR before "
           "compositing, so you can hide several plot lines at once."),
    )


def _agent_json() -> ui.Tag:
    _row = ui.tags.tr
    _td = ui.tags.td

    def _field(name, desc):
        return _row(_td(_code(name), style="white-space:nowrap;"),
                    _td(desc))

    return _section(
        "Agent JSON — import, export & schema",
        "agent_json",
        _p("A single JSON file can carry a complete verification job: the "
           "plot image, the extracted data rows, the axis calibration, the "
           "plot type, and the orientation. Upload or paste it in the "
           "sidebar and click ", ui.tags.strong("Import JSON"),
           "; export the corrected state with ",
           ui.tags.strong("Export as JSON"), " on the Overlay tab."),
        ui.tags.table(
            ui.tags.thead(_row(ui.tags.th("Field"), ui.tags.th("Meaning"))),
            ui.tags.tbody(
                _field("schema_version",
                       'Any "1.x" is accepted; exports are written as "1.1".'),
                _field("image",
                       ui.TagList(
                           _code("filename"), ", ", _code("width_px"), ", ",
                           _code("height_px"), ", and the image itself as a "
                           "base64 ", _code("data_uri"),
                           ". If the declared width differs from the decoded "
                           "image (e.g. a large image was auto-downscaled), "
                           "the calibration anchors are rescaled to match "
                           "and a warning reports the factor.")),
                _field("plot_type",
                       ui.TagList(
                           "One of ", _code("scatter"), ", ",
                           _code("line_timeseries"), ", ", _code("error_bar"),
                           ", ", _code("forest"), ", ", _code("bar"), ", ",
                           _code("box"), ", ", _code("kaplan_meier"), ".")),
                _field("orientation",
                       ui.TagList(
                           _code('"vertical"'), " (default) or ",
                           _code('"horizontal"'),
                           " — horizontal applies to bar and box plots "
                           "(value axis along x); forest plots are always "
                           "horizontal.")),
                _field("axes.x / axes.y",
                       ui.TagList(
                           _code('scale: "linear"|"log"'), ", optional ",
                           _code("log_base"),
                           " (a number > 1 or ", _code('"e"'),
                           "; defaults to 10 for log axes), and ",
                           _code("calibration"),
                           ": a list of at least two ",
                           _code("{pixel, value}"),
                           " pairs mapping pixel positions to data values.")),
                _field("rows",
                       ui.TagList(
                           "Data records in image-axis coordinates: ",
                           _code("series"), ", ", _code("x"), ", ",
                           _code("y"), ", ", _code("y_err_lower"), ", ",
                           _code("y_err_upper"),
                           " (absolute bounds), plus per-plot-type extras (",
                           _code("box_q1"), "/", _code("box_median"), "/",
                           _code("box_q3"), ", ", _code("at_risk"), ", ",
                           _code("is_summary"), ", ", _code("status"),
                           "). In horizontal layouts the value lives in ",
                           _code("x"), " and ", _code("y_err_*"),
                           " bracket ", _code("x"), "; forest-style ",
                           _code("value"), "/", _code("value_err_*"),
                           " aliases are accepted for any horizontal "
                           "layout.")),
                _field("series",
                       ui.TagList("Per-series display colors: ",
                                  _code("{key, color}"), " entries.")),
            ),
            class_="table table-sm",
        ),
        _p("A JSON without an ", _code("axes"),
           " block cannot be calibrated — on the JSON-only deployment the "
           "import is rejected with an error; on the full app you can still "
           "calibrate manually from the Calibrate tab."),
    )


def _overlay_editing() -> ui.Tag:
    return _section(
        "Overlay editing",
        "overlay_editing",
        _p("The Overlay tab draws the extracted data in data coordinates on "
           "top of the (optionally masked) source image. Use it to verify "
           "and correct individual points."),
        _ul(
            "Click a point's center, upper error-bar cap, or lower "
            "error-bar cap to select it. Selection populates the "
            "Edit-a-point panel and shows the floating zoom-preview "
            "bubble.",
            ui.TagList(
                ui.tags.strong("Edit a point"), " panel — dropdown to "
                "choose the point, then numeric inputs for ", _code("x"),
                " and ", _code("y"),
                ". Typed edits update the overlay live (no Apply needed); ",
                ui.tags.strong("Apply edit"),
                " is provided for explicit confirmation, and ",
                ui.tags.strong("Reset point"),
                " restores the original values (it becomes ",
                _code("Reset selected (N)"),
                " when several points are selected).",
            ),
            ui.TagList(
                ui.tags.strong("Symmetric interval"),
                " — with the checkbox ON the interval is edited as "
                "center ± ", _code("half-width"),
                " (one number instead of two bounds); nudging one bound "
                "mirrors the other about the center. With it OFF you edit "
                "the absolute ", _code("Lower bound"), " and ",
                _code("Upper bound"),
                " independently. Toggling it on seeds the half-width from "
                "the current bounds (their mean offset when asymmetric).",
            ),
            ui.TagList(
                ui.tags.strong("Arrow step"),
                " — size of each arrow-key nudge in data units. Defaults "
                "to the y-axis data-per-pixel magnitude from the current "
                "calibration (or 0.1 when not calibrated). Shift = 10×.",
            ),
            "Moving a point's center always carries its error bounds "
            "along, preserving the interval width.",
            "Original values are preserved — every edited row carries "
            "original_* audit columns alongside the edited values, and "
            "edited points are visually marked on the overlay.",
            ui.TagList(
                ui.tags.strong("Floating zoom bubble"),
                " — appears when a point is selected and the Overlay tab "
                "is active. Drag the title bar to reposition it. The "
                "bubble shares the masked composite with the main overlay.",
            ),
        ),
    )


def _multi_select() -> ui.Tag:
    return _section(
        "Selecting multiple points",
        "multi_select",
        _p("Several points can be selected and edited together:"),
        _ul(
            ui.TagList(ui.tags.strong("Shift-click"),
                       " a point to add it to (or remove it from) the "
                       "selection."),
            ui.TagList(ui.tags.strong("Box Select"),
                       " — choose the box-select tool from the plot's "
                       "modebar and drag a rectangle; every point inside "
                       "replaces the current selection."),
            ui.TagList(
                "The last-clicked point is the ",
                ui.tags.strong("anchor"),
                " (marked with a larger ring): typed edits and the zoom "
                "bubble follow the anchor, while arrow keys move the whole "
                "selection together, preserving each point's interval.",
            ),
            ui.TagList(_code("Reset selected (N)"),
                       " restores every selected point to its original "
                       "values."),
            ui.TagList(ui.tags.strong("Esc"),
                       " (or clicking empty background) clears the "
                       "selection."),
        ),
    )


def _export() -> ui.Tag:
    return _section(
        "Export",
        "export",
        _p("Download the corrected data as CSV (Overlay tab, Export "
           "accordion)."),
        _ul(
            ui.TagList(ui.tags.strong("Filename"),
                       " — defaults to ", _code("corrected.csv"), "."),
            ui.TagList(
                ui.tags.strong("Include audit columns"),
                " — when on, the export carries ", _code("original_x"),
                ", ", _code("original_y"), ", ",
                _code("original_y_err_lower"), ", ",
                _code("original_y_err_upper"), ", ", _code("edited"),
                " (bool), and ", _code("edit_type"),
                " (\"point\" / \"err_upper\" / \"err_lower\" / empty) "
                "alongside the edited values. Off for a clean export.",
            ),
            ui.TagList(
                "Standard columns always included: ", _code("series"),
                ", ", _code("x"), ", ", _code("y"), ", ",
                _code("y_err_lower"), ", ", _code("y_err_upper"),
                ", ", _code("series_color"), ".",
            ),
        ),
    )


def _dashboard_scatter() -> ui.Tag:
    return _section(
        "Dashboard — Scatter plots",
        "dashboard_scatter",
        _p("A correlation table appears below the Overlay when Plot type "
           "= Scatter plots. One row per series, plus an overall row when "
           "there are two or more series."),
        ui.tags.p(ui.tags.strong("Definitions:"),
                  style="margin:8px 0 4px 0;"),
        _ul(
            ui.TagList(_code("n"),
                       " — count of rows with finite x and y."),
            ui.TagList(_code("r"),
                       " — Pearson correlation coefficient between x and "
                       "y, r ∈ [−1, 1]. Computed via ",
                       _code("scipy.stats.pearsonr"),
                       ". Per-series r uses only that series's rows; "
                       "Overall r pools all rows across all series."),
            ui.TagList(_code("R²"), " = ", _code("r²"),
                       " — fraction of variance explained by a linear fit."),
            ui.TagList("When ", _code("n < 2"), ", r and R² are reported "
                       "as ", _code("—"), "."),
        ),
    )


def _dashboard_time_series() -> ui.Tag:
    table_html = """
<table class="table table-sm table-striped table-bordered" style="margin-top:6px;">
  <thead>
    <tr>
      <th>Type</th>
      <th>Formula for &sigma;</th>
      <th>Requires</th>
      <th>Interpretation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>SD</code></td>
      <td><code>&sigma; = half_w</code></td>
      <td>&mdash;</td>
      <td>Population dispersion (1 SD).</td>
    </tr>
    <tr>
      <td><code>SE</code></td>
      <td><code>&sigma; = half_w &middot; &radic;n</code></td>
      <td><code>n &ge; 1</code></td>
      <td>Standard error of the mean.</td>
    </tr>
    <tr>
      <td><code>Confidence</code></td>
      <td><code>&sigma; = (half_w &middot; &radic;n) / t<sub>&alpha;/2,&nbsp;n&minus;1</sub></code></td>
      <td><code>n &ge; 2</code>, percent</td>
      <td>Range that should contain the true mean with the chosen probability.</td>
    </tr>
    <tr>
      <td><code>Prediction</code></td>
      <td><code>&sigma; = half_w / (t<sub>&alpha;/2,&nbsp;n&minus;1</sub> &middot; &radic;(1 + 1/n))</code></td>
      <td><code>n &ge; 2</code>, percent</td>
      <td>Range that should contain a single future observation.</td>
    </tr>
  </tbody>
</table>
"""
    return _section(
        "Dashboard — Time series w/ intervals",
        "dashboard_time_series",
        _p("A Data summary table appears below the Overlay when Plot type "
           "= Time series w/ intervals. It reports μ (the point estimate "
           "from the CSV's ", _code("y"), " column) and σ (a "
           "standard-deviation analog derived from the CSV's error-bar "
           "half-widths)."),
        ui.tags.p(ui.tags.strong("Half-width per row:"),
                  style="margin:8px 0 4px 0;"),
        _ul(
            ui.TagList("Linear scale: ",
                       _code("half_w = (y_err_upper − y_err_lower) / 2"),
                       "."),
            ui.TagList("Log scale (y-axis log-calibrated with base ",
                       _code("b"), "): ",
                       _code("half_w = (log_b(y_err_upper) − log_b(y_err_lower)) / 2"),
                       ". The half-width is reported in the same base the "
                       "user picked for the y-axis (Manual Values → ",
                       _code("Y is log"), " → ", _code("Base"),
                       ") — so base 10 gives a log10-scale σ_log, base ",
                       _code("e"), " gives a natural-log σ_log, and so on. "
                       "Rows with non-positive ", _code("y"), ", ",
                       _code("y_err_lower"), ", or ", _code("y_err_upper"),
                       " are dropped (the log transform is undefined)."),
        ),
        ui.tags.p(
            ui.tags.strong("Conversion from half-width to σ:"),
            " depends on the ", _code("Error bar type"), " selector.",
            style="margin:8px 0 4px 0;",
        ),
        ui.HTML(table_html),
        _p(ui.TagList(
            "where ", _code("α = 1 − percent / 100"),
            " and ", _code("t"),
            " is the Student-t critical value, ",
            _code("scipy.stats.t.ppf(1 − α/2, df = n − 1)"),
            ".",
        )),
        ui.tags.p(ui.tags.strong("Log-scale calibration:"),
                  style="margin:8px 0 4px 0;"),
        _p(ui.TagList(
            "The σ above is on the user's chosen log scale (reported as the "
            "column ", _code("σ_log"),
            "). The table also reports the back-transformed multiplicative "
            "(geometric) SD ",
            _code("σ = b ** σ_log"),
            " in a second column, using the same base ", _code("b"),
            ". The back-transformed σ is invariant under the choice of "
            "base — picking base 10 vs base ", _code("e"),
            " vs base 2 gives the same geometric spread, only the σ_log "
            "column rescales.",
        )),
        ui.tags.p(ui.tags.strong("Controls:"),
                  style="margin:8px 0 4px 0;"),
        _ul(
            ui.TagList(_code("Error bar type"),
                       " — selects which σ formula to apply. Defaults to "
                       "whatever the CSV's ", _code("error_bar_type"),
                       " column reported, or SD if absent."),
            ui.TagList(_code("Percent"),
                       " — only shown for Confidence / Prediction; "
                       "range 50–99.9, default 95."),
            ui.TagList(_code("n (series)"),
                       " — only shown when needed (SE, CI, PI). "
                       "One numeric input per series; enter the sample "
                       "size that produced each reported interval."),
            ui.TagList(_code("Display x"),
                       " — ", _code("None"), " (rows indexed by Obs "
                       "number, no x column), ",
                       _code("Single column"),
                       " (index is the mean x across series per Obs), ",
                       _code("Multi column"),
                       " (one extra x column per series before its μ/σ "
                       "columns)."),
        ),
    )


def _keyboard_shortcuts() -> ui.Tag:
    return _section(
        "Keyboard shortcuts",
        "keyboard_shortcuts",
        _ul(
            ui.TagList(ui.tags.strong("Calibrate tab — anchor nudge:"),
                       " click P1 or P2, then arrow keys = ±1 px; ",
                       _code("Shift + arrow"), " = ±10 px."),
            ui.TagList(ui.tags.strong("Overlay tab — point nudge:"),
                       " select a point part (center / upper cap / lower "
                       "cap), then arrow keys move by ",
                       _code("Arrow step"), " data units. ",
                       _code("Shift + arrow"), " = 10× that step. With "
                       "several points selected, arrows move all of them "
                       "together. Held keys are batched client-side, so "
                       "the motion stays smooth."),
            ui.TagList(ui.tags.strong("Bound nudge:"),
                       " with a bound cap selected and ",
                       _code("Symmetric interval"),
                       " on, nudging one bound mirrors the other about "
                       "the center."),
            ui.TagList(ui.tags.strong("Esc:"),
                       " clear the overlay selection."),
            "Arrow keys do NOT fire while a form input is focused — click "
            "elsewhere first.",
        ),
    )


def _troubleshooting() -> ui.Tag:
    return _section(
        "Tips & troubleshooting",
        "troubleshooting",
        _ul(
            ui.TagList(
                ui.tags.strong("CSV doesn't load after the image:"),
                " upload the image first. CSV uploads while no image is "
                "active show a warning notification.",
            ),
            ui.TagList(
                ui.tags.strong("Calibration values don't take effect:"),
                " typing in Manual Values does NOT auto-apply. Click ",
                _code("Apply manual calibration"),
                " to commit. (Drags also need an Apply if you typed any "
                "data values after the drag.)",
            ),
            ui.TagList(
                ui.tags.strong("Auto-detect misses a tick:"),
                " widen the corresponding label band, lower Min OCR "
                "confidence, or just enter the tick value manually in the "
                "Manual Values panel and click Apply.",
            ),
            ui.TagList(
                ui.tags.strong("Log-scale plot:"),
                " tick the X is log / Y is log checkbox, set the base, "
                "and enter ", _code("data X"), " / ", _code("data Y"),
                " as the underlying values (not their logs). The "
                "calibration handles the log transform.",
            ),
            ui.TagList(
                ui.tags.strong("Mask is too aggressive / too loose:"),
                " adjust the per-series ΔE slider. Lower = stricter match. "
                "A series whose color is the auto-palette default can't be "
                "masked — pick a color that actually appears in the image "
                "first.",
            ),
            ui.TagList(
                ui.tags.strong("Overlay tab looks horizontally squished on "
                                "first load:"),
                " the page layout self-corrects after the tab is first "
                "shown; if it persists, switch tabs and back.",
            ),
            ui.TagList(
                ui.tags.strong("Large image notification:"),
                " images over 4000 px or 25 MB are auto-downscaled to "
                "~3000 px for display. Auto-calibration runs against the "
                "downscaled image, so the calibration is exact in pixel "
                "space for the version you see.",
            ),
        ),
    )


def user_manual_tab(ocr: Optional[bool] = None,
                    json_only: Optional[bool] = None) -> ui.Tag:
    """Build the User Manual nav panel.

    Two gates shape the section list:

    - ``ocr`` (defaults to module-level ``_OCR_AVAILABLE``): OCR-only
      sections require a working EasyOCR install.
    - ``json_only`` (defaults to ``runtime_flags.json_only_mode()``): the
      JSON-only deployment has no Calibrate tab, so every calibration /
      plot-type / series-color section is omitted (OCR sections included —
      they document Calibrate-tab controls).
    """
    if ocr is None:
        ocr = _OCR_AVAILABLE
    if json_only is None:
        json_only = _JSON_ONLY

    panels = [
        _getting_started(json_only=json_only),
        _agent_json(),
    ]
    if not json_only:
        panels.extend([
            _calibration_overview(),
            _manual_calibration(),
            _manual_values(),
        ])
        if ocr:
            panels.extend([
                _auto_calibration(),
                _detect_frame(),
                _label_bands(),
                _calibration_points(),
                _detection_settings(),
            ])
        panels.extend([
            _plot_type(),
            _series_colors(),
        ])
    panels.extend([
        _mask_preview(),
        _overlay_editing(),
        _multi_select(),
        _export(),
        _dashboard_scatter(),
        _dashboard_time_series(),
        _forest_plots(),
        _keyboard_shortcuts(),
        _troubleshooting(),
    ])

    return ui.nav_panel(
        "User Manual",
        ui.card(
            ui.card_header("PlotVerify user manual"),
            ui.card_body(
                ui.tags.p(
                    "Reference for every tool in the PlotVerify workflow. "
                    "Sections start collapsed — click a header to expand.",
                    style="color:#555; margin:0 0 12px 0;",
                ),
                ui.accordion(
                    *panels,
                    id="user_manual_accordion",
                    open=[],
                    multiple=True,
                ),
            ),
            full_screen=True,
        ),
    )
