# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

PlotVerify is a Shiny app for verifying AI-extracted data from scientific plots. It overlays user-supplied data (series, x/y values, confidence intervals, colors) on the source image and calibrates pixel coordinates to data coordinates using a multi-phase OCR + geometry pipeline.

## Running the app

```bash
shiny run app_shiny.py
```

The legacy single-image Streamlit workflow is still available at `app_auto_axis.py` (`streamlit run app_auto_axis.py`), but active development targets the Shiny app.

## Tests and CI

```bash
pip install -r requirements-dev.txt
pytest -q
```

Baseline: **156 passed, 1 xfailed** with EasyOCR installed; **137 passed** without (the 20 real-image regression tests in `tests/test_real_image_regression.py` auto-skip when `easyocr` is missing).

GitHub Actions (`.github/workflows/tests.yml`) runs the full suite on push and pull request against Python 3.10/3.11/3.12. The EasyOCR model is cached at `~/.EasyOCR` across runs.

A local pre-push hook is available at `scripts/git-hooks/pre-push`. Enable it once per clone with:

```bash
git config core.hooksPath scripts/git-hooks
```

## Branch structure

| Branch | Purpose |
|---|---|
| `main` | Active development branch |
| `shiny-manual` | Deployment branch — `main` + GitHub Pages deploy infrastructure |

On every push to `main`, `.github/workflows/sync-to-shiny-manual.yml` automatically merges `main` into `shiny-manual`. That merge triggers `.github/workflows/deploy-shinylive.yml` (which lives only on `shiny-manual`), rebuilding and deploying the shinylive bundle to GitHub Pages.

`shiny-manual` carries deployment-only files not present on `main`:
- `.github/workflows/deploy-shinylive.yml` — GitHub Pages deploy trigger
- `shinylive_app/requirements.txt` — Pyodide-compatible deps (no EasyOCR/PyTorch)
- `scripts/build_shinylive.py` — shinylive bundle builder

If a new package is added to `requirements.txt` on `main`, manually update `shinylive_app/requirements.txt` on `shiny-manual` if the package is Pyodide-compatible, or omit it if it requires native extensions.

**Prerequisite:** A fine-grained PAT with "Contents: write" permission must be stored as repository secret `SYNC_PAT` for the sync workflow to push and trigger downstream workflows.

## Architecture

Three layers sit on top of the calibration engine:

```
app_shiny.py
    └── shiny_app/app.py       — Shiny UI: reactive layout, user interaction
        shiny_app/figures.py   — Plotly figure builders (calibration edit + data overlay)

plotverify_core/               — UI-agnostic business logic
    app.py                     — PlotVerifyApp controller, owns AppState
    session.py                 — AppState, PerFileState, ReviewStatus dataclasses
    csv_io.py                  — CSV loading, validation, audit reports
    image_io.py                — Image decode + downscaling
    matching.py                — Image/CSV pairing by canonical stem
    overlay_model.py           — EditableOverlay: per-point edits, preserves originals
    overlay_traces.py          — build_overlay_traces(): UI-agnostic trace records
    overlay_image.py           — ΔE mask preview compositing
    colors.py                  — Hex validation, hex↔HSV/BGR, complementary picker
    masking.py                 — CIE Lab delta_e_mask, HSV masking primitives
    series_state.py            — SeriesState: per-series mask config
    serialization.py           — Session save/load as zip (manifest.json + images + csvs)

axis_pipeline/                 — Core calibration engine (see section below)
```

### `axis_pipeline/` — the core calibration engine

All new code should use this package directly. The public surface is exported from `axis_pipeline/__init__.py`:

- `run_calibration(img_bgr, config=None, ocr_runner=None) -> CalibrationResult` — full multi-phase pipeline
- `detect_axis_frame(img_bgr, ...) -> FramePreview` — Phase A + geometric detection only (used by the manual-band UI tab)
- `render_overlay(img_bgr, result) -> RGB` — diagnostic visualization
- `render_band_preview(img_bgr, bbox, y_band, x_band) -> RGB`
- `CalibrationConfig` — all tunables (band sizes, OCR thresholds, regression settings)

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
   → select anchors P1/P2/P3
```

### Module map

| Module | Responsibility |
|---|---|
| `types.py` | All dataclasses: `OCRRecord`, `AxisFrame`, `GridFit`, `PairedTick`, `AxisCalibration`, `CalibrationConfig`, `CalibrationResult`, `FramePreview` |
| `pipeline.py` | `run_calibration`, `detect_axis_frame`, phase orchestration |
| `geometry.py` | Dark-mask preparation, projection-profile + Hough axis line detection, tick position detection |
| `ocr.py` | EasyOCR integration, `parse_numeric_tick`, band crop helpers (`x_label_band`, `y_label_band`), text masking |
| `gridfit.py` | Modal-spacing linear grid fit — rejects non-tick peaks from the geometric detector |
| `pairing.py` | Spatial filter + one-to-one matching of OCR labels to grid-fitted ticks; monotonicity enforcement |
| `calibration.py` | OLS and Student-t MLE regression (`data = scale * pixel + offset`) |
| `overlay.py` | Diagnostic overlay drawing |
| `legacy.py` | Adaptor that re-exposes the old dict-shaped API for the legacy Streamlit app |

### Legacy shims

`axis_auto.py` and `ocr_axis.py` are thin re-export shims for the old API. The Streamlit app (`app_auto_axis.py`) imports from them. Do not add logic there — delegate to `axis_pipeline`.

### OCR injection point

`pipeline.py` accepts an `ocr_runner` callable so EasyOCR can be replaced in tests without mocking. The signature is:

```python
def my_runner(img_bgr, *, gpu, min_confidence, allowlist, phase, bbox_offset, upsample, detection_params) -> List[OCRRecord]
```

### Calibration coordinate convention

`AxisCalibration` stores `data = scale * pixel + offset`. Y-axis pixel origin is top-left (pixel-y increases downward), so a plot with a normal upward y-axis will have a negative `scale`. `AxisCalibration.pixel_to_data(px)` and `data_to_pixel(value)` handle the conversion.

### `CalibrationResult` anchor points

P1 = leftmost/bottommost paired tick (x-axis left edge, y-axis bottom edge).  
P2 = rightmost paired x-tick at the same row as P1.  
P3 = topmost paired y-tick; its `data_x` is derived from the x-calibration at P3's pixel-x position.

## Key dependencies

- `easyocr` — production OCR engine (lazy import in `ocr.py`; not needed if injecting a custom runner)
- `opencv-python` (cv2), `numpy`, `scipy` — image processing and calibration math
- `shiny`, `shinywidgets`, `plotly`, `pandas`, `Pillow` — UI layer
