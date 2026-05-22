# PlotVerify — shinylive bundle

A browser-only build of the PlotVerify Shiny app, packaged with
[shinylive](https://shiny.posit.co/py/docs/shinylive.html) so it runs
entirely client-side via Pyodide.

## Manual-only mode

EasyOCR cannot run under Pyodide (PyTorch + large model weights), so
auto-calibration is disabled in this build. The app already handles
this gracefully: when `axis_pipeline.ocr_available()` returns `False`,
the upload banner shows "EasyOCR not installed" and the OCR controls
are hidden. The manual P1/P2/P3 anchor workflow stays fully functional.

## Layout

This directory holds only the build-specific files:

```
shinylive_app/
├── requirements.txt   # Pyodide-installable deps (no easyocr)
└── README.md          # this file
```

The Shiny app source itself lives at `shiny_app/app.py` and
`shiny_app/figures.py`. The `axis_pipeline/` and `plotverify_core/`
packages live at the repo root. The build script copies all of these
into a staging directory at build time so nothing here can drift.

## Build

```bash
pip install shinylive
python scripts/build_shinylive.py
```

Output: `dist/shinylive/site/` — a static directory ready to deploy
anywhere (GitHub Pages, Netlify, S3, etc.).

## Serve locally

```bash
python -m http.server -d dist/shinylive/site 8000
```

Then open <http://localhost:8000>. First load takes ~30-60 s while
Pyodide and the wheels download; subsequent loads are cached.
