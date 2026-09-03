# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

PlotVerify is a Shiny app for verifying AI-extracted data from scientific plots. It overlays user-supplied data (series, x/y values, error intervals, colors) on the source image and calibrates pixel coordinates to data coordinates using a multi-phase OCR + geometry pipeline, two manually-placed anchors, or the calibration embedded in an Agent JSON (schema 1.1: image, rows, axes with per-axis log bases, plot type, orientation). The Overlay tab supports per-point and multi-point editing, ΔE color masking of the source image, and a Dashboard with derived statistics (Pearson r/R² for scatter plots; CSV-half-width → σ conversion for time-series plots).

## Running the app

```bash
shiny run app_shiny.py                        # full app
PLOTVERIFY_JSON_ONLY=1 shiny run app_shiny.py # JSON-only mode (simulates the shinylive deploy)
```

The legacy single-image Streamlit workflow is still available at `app_auto_axis.py` (`streamlit run app_auto_axis.py`), but active development targets the Shiny app.

## Tests and CI

```bash
pip install -r requirements-dev.txt
pytest -q
```

Current baseline: **349 passed, 1 xfailed** with EasyOCR installed; **~325 passed** without (the real-image regression tests in `tests/test_real_image_regression.py` auto-skip when `easyocr` is missing).

GitHub Actions (`.github/workflows/tests.yml`) runs the full suite on push and pull request against Python 3.10/3.11/3.12. The EasyOCR model is cached at `~/.EasyOCR` across runs.

A local pre-push hook is available at `scripts/git-hooks/pre-push`. Enable it once per clone with:

```bash
git config core.hooksPath scripts/git-hooks
```

## Branch structure

| Branch | Purpose |
|---|---|
| `main` | Active development branch |
| `shiny-manual` | Deployment branch — GitHub Pages deploy infrastructure on top of `main` |

### JSON-only mode (runtime flag)

The shinylive deploy is a **JSON-only** surface: no Calibrate tab, no image/CSV uploads — the sidebar accepts only an Agent JSON, which carries the image, data, calibration, plot type and orientation. This is a **runtime flag**, not a branch difference: `shiny_app/runtime_flags.py::json_only_mode()` returns True under Pyodide (`sys.platform == "emscripten"`) and can be forced locally with `PLOTVERIFY_JSON_ONLY=1` (or suppressed with `=0`). `shiny_app/app.py` and `shiny_app/user_manual.py` read it once at import (`_JSON_ONLY`) to shape the UI tree, so the same source files serve both deployments.

Historically `shiny-manual` hard-deleted OCR UI code from `shiny_app/app.py`; with the runtime flag that divergence is obsolete. **At the next `main → shiny-manual` merge**, resolve `shiny_app/` conflicts by taking main wholesale:

```bash
git checkout shiny-manual
git merge main            # conflicts in shiny_app/ expected once
git checkout main -- shiny_app/app.py shiny_app/user_manual.py
# ALSO in this same merge commit: add shiny_app/runtime_flags.py and
# shiny_app/edit_logic.py to the FILES list in scripts/build_shinylive.py
# (staged flat next to app.py), or the Pages deploy fails at import time.
git commit && git push origin shiny-manual
git checkout main
```

After that merge, future merges should be conflict-free in `shiny_app/`.

The push triggers `.github/workflows/deploy-shinylive.yml` (which lives only on `shiny-manual`), rebuilding and deploying the shinylive bundle to GitHub Pages. `shiny-manual` carries deployment-only files not present on `main`:
- `.github/workflows/deploy-shinylive.yml` — GitHub Pages deploy trigger
- `shinylive_app/requirements.txt` — Pyodide-compatible deps (no EasyOCR/PyTorch)
- `scripts/build_shinylive.py` — shinylive bundle builder (stages `shiny_app/*.py` flat at the bundle root plus the `axis_pipeline/` and `plotverify_core/` packages)

The `shiny_app/user_manual.py` module additionally gates OCR-only sections on `axis_pipeline.ocr_available()` and calibration sections on `json_only_mode()` — see its docstring.

If a new package is added to `requirements.txt` on `main`, manually update `shinylive_app/requirements.txt` on `shiny-manual` if the package is Pyodide-compatible, or omit it if it requires native extensions.

## Architecture

Three layers sit on top of the calibration engine:

```
app_shiny.py                       — entry point: re-exports shiny_app.app:app
    └── shiny_app/app.py           — Shiny UI: page_navbar with three tabs
        │                            (Calibrate, Overlay, User Manual; the
        │                            Calibrate tab and image/CSV uploads are
        │                            omitted in JSON-only mode), reactive
        │                            layout, all user interaction
        shiny_app/figures.py       — Plotly figure builders: calibration edit
        │                            widget, data overlay, floating zoom bubble,
        │                            band/anchor/guide shapes
        shiny_app/edit_logic.py    — Pure nudge/symmetry math (PointVals,
        │                            apply_nudge, linked_bounds, half_width_of)
        shiny_app/runtime_flags.py — json_only_mode(): Pyodide detection +
        │                            PLOTVERIFY_JSON_ONLY env override
        shiny_app/user_manual.py   — User Manual tab content (collapsible
                                     accordion). Gates: OCR sections on
                                     ocr_available(), calibration sections
                                     on json_only_mode().

plotverify_core/                   — UI-agnostic business logic. No streamlit,
    │                                no shiny imports anywhere in this package.
    app.py                         — PlotVerifyApp controller, owns AppState;
    │                                wraps axis_pipeline + serialization
    session.py                     — AppState, PerFileState, Anchors,
    │                                ReviewStatus, WorkflowStage, MaskingChoice
    csv_io.py                      — load_csv() + LoadReport: required vs
    │                                optional columns, auto-swap reversed
    │                                error bars, error_bar_type sniff, palette
    │                                color assignment when no series_color col
    image_io.py                    — decode_and_maybe_downscale: 4000 px / 25 MB
    │                                large-image warning; auto-downscale to
    │                                3000 px when downscale=True
    matching.py                    — match_files(): pair image+CSV by lowercase
    │                                stem; reports duplicates and unmatched
    overlay_model.py               — EditableOverlay: per-point x/y/err edits
    │                                + batch mutators (nudge_points,
    │                                reset_points), preserves originals,
    │                                edit_type audit
    overlay_traces.py              — build_overlay_traces(): UI-agnostic trace
    │                                records with error bars + ribbon coords;
    │                                one-sided intervals (has_upper/has_lower);
    │                                is_horizontal_layout(plot_type, orientation)
    json_io.py                     — Agent JSON schema 1.1 parse/export:
    │                                embedded image, axes calibration (incl.
    │                                per-axis log_base), plot_type,
    │                                orientation, rows; rescale_anchors for
    │                                downscaled images
    overlay_image.py               — build_masked_overlay_image(): composites
    │                                a source image with one or more series
    │                                ΔE masks painted in the background color
    colors.py                      — Hex validation, hex↔HSV/BGR, palette,
    │                                hex_complement (marker color), background
    │                                color autodetect (modal grey)
    masking.py                     — delta_e_mask (CIE Lab ΔE76), HSV masking
    series_state.py                — SeriesState: per-series mask config (HSV
    │                                ranges, ΔE threshold, interpolate flag)
    calibration_math.py            — Legacy three-point manual calibration math
    │                                (compute_calibration, px_to_data,
    │                                data_to_px). Still consumed via the dict-
    │                                shaped `cal` by Plotly figure builders.
    serialization.py               — Session save/load as .pvsession zip
    │                                (manifest.json + images + csvs + overlays)
    streamlit_bridge.py            — Translation layer between flat
    │                                st.session_state keys and AppState so the
    │                                Streamlit app can use save/load_session
    dashboard.py                   — compute_time_series_stats(),
                                     compute_scatter_stats(),
                                     build_time_series_display_df() — pure
                                     stats for the Overlay-tab dashboard

axis_pipeline/                     — Core calibration engine (see below)
```

### Shiny UI surface (`shiny_app/app.py`)

`ui.page_navbar(...)` built by `_make_ui(json_only=None)` / `_make_sidebar(json_only=None)` (the parameter defaults to the module constant `_JSON_ONLY`; tests pass it explicitly).

**Full mode** — three tabs:

1. **Calibrate** — calibration image with two draggable anchors (P1 red top-left, P2 green bottom-right; the bottom-left corner is implicit as `(P1.x, P2.y)`); card header buttons: `Run detection`, `Detect axis frame`, `Reset anchors`; right-column accordion: `X/Y label bands`, `Calibration points`, `Manual Values` (pixel inputs + data inputs + log-base checkboxes + `Apply manual calibration` button), `Series colors`, `Plot type` (time_series / scatter / forest / bar / box / kaplan_meier); bottom accordion: `Detection settings`, `Frame-detection warnings`.
2. **Overlay** — calibrated image with extracted data; controls accordion: `Series` (per-series visibility checkbox + mask checkbox + ΔE numeric field), `Edit a point` (point selector + x/y inputs + either absolute Lower/Upper bound inputs or a `± half-width` input behind the `Symmetric interval` checkbox + Arrow step + Apply edit / Reset point), `Export` (filename, audit-columns toggle, CSV + JSON downloads; starts collapsed); dashboard panel rendered below; floating draggable zoom-preview bubble keyed to the selection anchor.
3. **User Manual** — single-card body with one collapsible accordion of reference sections; all sections start collapsed, gated per mode (see `user_manual.py` docstring).

**JSON-only mode** (shinylive / `PLOTVERIFY_JSON_ONLY=1`) — the Calibrate tab is omitted (Overlay is first and selected), and the sidebar carries only the Agent JSON block (`json_upload` / `json_paste` / `json_apply`) plus `session_status`. A JSON without an `axes` block is rejected with an error since there is no Calibrate tab to recover with.

The sidebar in full mode carries `image_upload`, `csv_upload`, an OCR-available banner, the Agent JSON block, and the per-file `session_status`. Image upload on `main` auto-runs `detect_axis_frame()` so the anchors seed at the detected rectangle corners.

### Overlay selection & editing model

- Selection state (`selected_overlay_rv`) is `None` or `{"pids": [ordered], "anchor": pid, "part": "center"|"upper"|"lower"}`. Plain click replaces; Shift-click toggles membership; Plotly Box Select replaces with the boxed set; Esc or a background click clears. The pure reducer `_update_selection` and helpers `_sel_pids`/`_sel_anchor`/`_sel_part` live at module level in `app.py` (tested in `tests/test_multi_select_logic.py`).
- Typed edits and the zoom bubble follow the **anchor** (last-clicked, marked with a large ring when N>1); arrow keys nudge the whole selection; `Reset point` becomes `Reset selected (N)`.
- Arrow-key nudging is **coalesced client-side**: keydowns accumulate into `{dx, dy}` step counts (Shift ×10 applied at accumulate time) with a leading-edge send plus a 60 ms trailing flush, so held keys cost ≤ ~16 messages/s. The server applies the math via `shiny_app/edit_logic.py::apply_nudge` (center moves always gang-move the bounds; bound moves mirror when `Symmetric interval` is on) and pushes all edited points to the FigureWidget in one `batch_update` (`_push_point_edits_to_widget`) — no figure rebuild.
- The `Edit a point` panel is static across edits (rebuilds only on file/data changes); values flow via `ui.update_numeric`. The 1e-10 model-comparison guard in `_live_update_point_inputs` distinguishes user edits from programmatic echoes — pushes must use exact model floats.
- The `Series` panel is likewise static across edits: it rebuilds only on `csv_revision` / file change, never on `overlay_revision`, because `_sync_series_delta_e` *writes* that counter. Subscribing to it would re-seed the ΔE field from stale state mid-edit and reset the visibility/mask checkboxes to their defaults. ΔE edits only bump `overlay_revision` when the edited series is currently masked (nothing else about the composite depends on the threshold).
- Series visibility is one state with two views: the `Overlay` checkbox and the Plotly legend entry. `plotly_legendclick` forwards the series name to `input.overlay_legend_toggle`, and `_on_overlay_legend_toggle` flips the matching `_vis_id` checkbox — Plotly's own toggle still runs for instant feedback, and the ensuing rebuild lands on the same visibility. `plotly_legenddoubleclick` is suppressed on the overlay because its client-side isolate has no checkbox equivalent.

- **One-sided intervals**: a row with only `y_err_lower` *or* only `y_err_upper` (including when the whole column is absent) still draws an interval — the missing bound collapses to the point estimate, so the ribbon/band runs from the point out to the bound it has. `OverlayTrace.has_err` means "at least one bound"; `has_upper`/`has_lower` say which side, and gate the per-side clickable caps and box whiskers. A row missing both bounds draws nothing.

The sidebar carries `image_upload`, `csv_upload`, an OCR-available banner, and a per-file `session_status` block. Image upload on `main` auto-runs `detect_axis_frame()` so the anchors seed at the detected rectangle corners.

### `axis_pipeline/` — the core calibration engine

All new code should use this package directly. The public surface is exported from `axis_pipeline/__init__.py`:

- `run_calibration(img_bgr, config=None, ocr_runner=None) -> CalibrationResult` — full multi-phase pipeline
- `detect_axis_frame(img_bgr, ...) -> FramePreview` — Phase A + geometric detection only (used on every image upload and by the "Detect axis frame" button)
- `manual_calibration(p1_pixel=, p2_pixel=, p3_pixel=, ...) -> CalibrationResult` — fits a typed result from three anchor points without OCR; used by Apply manual calibration
- `render_overlay(img_bgr, result) -> RGB` — diagnostic visualization
- `render_band_preview(img_bgr, bbox, y_band, x_band) -> RGB`
- `CalibrationConfig` — all tunables (band sizes, OCR thresholds, regression settings)
- `PIPELINE_VERSION` — bumped manually when `run_calibration`'s output semantics change; consumed by `serialization.py` to decide whether to trust a saved `CalibrationResult` or re-invoke the pipeline on load

### Pipeline phases (in `pipeline.py`)

```
Phase A: full-image EasyOCR discovery scan
   → mask all detected text
   → detect axis frame geometrically (geometry.py)
Phase B: re-OCR a tight band left of the y-axis (numeric allowlist)
Phase C: re-OCR a tight band below the x-axis (numeric allowlist)
   → merge records, preferring band-phase over full-phase
   → detect geometric tick positions (geometry.py)
   → grid-fit each axis to reject non-tick peaks (gridfit.py)
   → pair OCR labels to grid-fitted ticks (pairing.py)
   → calibrate each axis independently (calibration.py)
   → select anchors P1/P2/P3 (typed CalibrationResult)
```

### Module map

| Module | Responsibility |
|---|---|
| `types.py` | All dataclasses: `OCRRecord`, `AxisFrame`, `GridFit`, `PairedTick`, `AxisCalibration`, `CalibrationConfig`, `CalibrationResult`, `FramePreview` |
| `pipeline.py` | `run_calibration`, `detect_axis_frame`, `manual_calibration`, phase orchestration |
| `geometry.py` | Dark-mask preparation, projection-profile + Hough axis line detection, tick position detection |
| `ocr.py` | EasyOCR integration, `parse_numeric_tick`, band crop helpers (`x_label_band`, `y_label_band`), text masking |
| `gridfit.py` | Modal-spacing linear grid fit — rejects non-tick peaks from the geometric detector |
| `pairing.py` | Spatial filter + one-to-one matching of OCR labels to grid-fitted ticks; monotonicity enforcement |
| `calibration.py` | OLS and Student-t MLE regression (`data = scale * pixel + offset`) |
| `overlay.py` | Diagnostic overlay drawing |
| `legacy.py` | Adaptor that re-exposes the old dict-shaped API for the legacy Streamlit app; also exposes `rebuild_result_from_detection` and `update_result_from_tick_edits` |

### Legacy shims

`axis_auto.py` and `ocr_axis.py` at the repo root are thin re-export shims for the old API. The Streamlit app (`app_auto_axis.py`) imports from them. Do not add logic there — delegate to `axis_pipeline`.

### OCR injection point

`pipeline.py` accepts an `ocr_runner` callable so EasyOCR can be replaced in tests without mocking. The signature is:

```python
def my_runner(img_bgr, *, gpu, min_confidence, allowlist, phase, bbox_offset, upsample, detection_params) -> List[OCRRecord]
```

### Calibration coordinate conventions

`AxisCalibration` stores `data = scale * pixel + offset`. Y-axis pixel origin is top-left (pixel-y increases downward), so a plot with a normal upward y-axis will have a negative `scale`. `AxisCalibration.pixel_to_data(px)` and `data_to_pixel(value)` handle the conversion. For log-calibrated axes, the regression is fitted in log-space and `px_to_data` exponentiates: `data = log_base ** (scale * pixel + offset)`.

### Anchor model — UI vs typed result

The **typed CalibrationResult** carries three anchor points:

- `P1` = leftmost / bottommost paired tick (x-axis left edge, y-axis bottom edge).
- `P2` = rightmost paired x-tick at the same row as P1.
- `P3` = topmost paired y-tick; its `data_x` is derived from the x-calibration at P3's pixel-x position.

The **Shiny calibration UI** only renders TWO draggable circles:

- Display **"P1"** (red) is the top-left corner — internally it's the `p3_pixel` field on `Anchors`.
- Display **"P2"** (green) is the bottom-right corner — internally it's the `p2_pixel` field.
- The bottom-left corner is **derived** as `(display-P1.x, display-P2.y)` and stored as `p1_pixel`. It is never shown to the user.

The mapping is enforced by `enforce_anchor_constraints` in `shiny_app/figures.py`: any drag of one visible anchor pulls its rectangle partner along so the implicit bottom-left stays consistent. The Manual Values panel labels the two anchors "P1" / "P2" but exposes four data values — `P1 data X` is `p1_data_x` (left edge), `P1 data Y` is `p3_data_y` (top), `P2 data X` is `p2_data_x` (right edge), `P2 data Y` is `p1_data_y` (bottom).

### Dashboard statistics (in `plotverify_core/dashboard.py`)

Pure, framework-agnostic functions. The Shiny dashboard panel calls these directly.

**Scatter (`compute_scatter_stats`):**

- Per-series and overall `n`, Pearson `r` (via `scipy.stats.pearsonr`), `R² = r²`. Falls back to NaN when `n < 2`.

**Time series (`compute_time_series_stats` + `build_time_series_display_df` + `_sd_from_half_width`):**

The CSV's reported half-width is `(y_err_upper − y_err_lower) / 2` on linear axes, and `(log_b(y_err_upper) − log_b(y_err_lower)) / 2` on a log-axis where `b` is the user-chosen base (Manual Values → `Y is log` → `Base`). Both functions take a `log_base: Optional[float]` parameter (positive float > 1, or `None` for linear; the legacy bool form `True` aliases to base 10 for back-compat). Implementation: `(np.log(upper) − np.log(lower)) / (2 · ln b)` to stay numerically stable with NumPy.

Displayed σ:

| Error bar type | σ formula |
|---|---|
| `SD` | `half_w` |
| `SE` | `half_w · √n` |
| `Confidence` | `(half_w · √n) / t_{α/2, n−1}` |
| `Prediction` | `half_w / (t_{α/2, n−1} · √(1 + 1/n))` |

where `α = 1 − percent / 100` and `t` is `scipy.stats.t.ppf(1 − α/2, df = n−1)`. On a log-axis the table reports both `σ_log` (in base `b`) and the back-transformed geometric SD `σ = b ** σ_log`. The geometric SD is invariant under the choice of base; `σ_log` rescales by a constant factor `1 / ln b`.

The shiny app pipes `cal.get("y_log_base")` (from the typed calibration result) straight through, so the dashboard reads the same base the user committed via the Manual Values → Apply path.

### ΔE mask preview (in `plotverify_core/masking.py` + `overlay_image.py`)

`delta_e_mask(img_bgr, hex_color, threshold)` computes the CIE 1976 Lab Euclidean distance per pixel against `hex_color`, returns a binary mask of pixels with `ΔE < threshold` (dilated 2× with a 3×3 kernel to fill anti-aliasing fringes). `build_masked_overlay_image` ORs together one mask per series spec and repaints the union with the modal grey level from `detect_background_color`. Mask mode is only available for series with an "intentional" color — i.e. CSV-supplied or user-picked, never auto-palette defaults (gated by `PerFileState.has_intentional_color`).

## Key dependencies

- `easyocr` — production OCR engine (lazy import in `ocr.py`; not needed if injecting a custom runner)
- `opencv-python` (cv2), `numpy`, `scipy` — image processing and calibration math
- `shiny`, `shinywidgets`, `plotly`, `pandas`, `Pillow` — UI layer
- `streamlit` — only required for the legacy `app_auto_axis.py`; the Shiny app does not import it
