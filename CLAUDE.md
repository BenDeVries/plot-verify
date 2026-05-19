# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

PlotVerify is a Streamlit app for verifying AI-extracted data from scientific plots. It overlays user-supplied data (series, x/y values, confidence intervals, colors) on the source image and calibrates pixel coordinates to data coordinates using a multi-phase OCR + geometry pipeline.

## Running the app

```bash
streamlit run app_auto_axis.py
```

## Running the pipeline in a script (with tesseract fallback)

EasyOCR requires a model download; in sandboxes or CI use the tesseract shim defined in `scripts/run_pipeline_with_tesseract.py`:

```bash
python scripts/run_pipeline_with_tesseract.py
```

The shim implements the same `OCRRunner` callable signature, so it is a drop-in for `run_calibration(..., ocr_runner=tesseract_runner)`.

## Architecture

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
| `legacy.py` | Adaptor that re-exposes the old dict-shaped API for the Streamlit app |

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
- `pytesseract` — test/sandbox OCR shim only
- `opencv-python` (cv2), `numpy`, `scipy` — image processing and calibration math
- `streamlit`, `plotly`, `pandas`, `Pillow` — UI layer
