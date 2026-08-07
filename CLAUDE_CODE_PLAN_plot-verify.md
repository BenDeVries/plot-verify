# Claude Code work order — plot-verify: JSON auto-calibration, new plot types, clean overlay editor

**Repo:** `BenDeVries/plot-verify` · **Deploy:** shinylive at `bendevries.github.io/plot-verify`
**Prepared for:** Claude Code (autonomous implementation) · **Prepared by:** Plot Digitization agent maintainer

## 0. Goal in one paragraph
The Plot Digitization agent now emits a **plot-verify JSON** (contract v1.0, see
`plotverify_json.md`, reproduced in §3) that bundles axis **calibration** plus **all
digitized rows**. Update plot-verify so a user can paste/upload that JSON and land
**directly on an overlay-review screen with zero manual calibration** (delete the
calibrate step from the happy path), review the digitized layer drawn on top of the
original figure, **edit** points/intervals/values with simple controls, and export a
**versioned** edited JSON + CSV. Add rendering for **all seven** unified plot types.

---

## 1. First, orient yourself in the repo (do this before coding)
> These are assumptions to **verify against the actual code** — adjust the plan to fit.

**Framework is confirmed: Python Shiny, deployed via `shinylive export` to GitHub
Pages.** Assume `app.py` (or a `shiny`/`shinyapp` package layout) using
`shiny.express` or the classic `ui`/`server` split. Build with `shiny` +
`shinywidgets` (for Plotly), and re-run `shinylive export <appdir> docs/` so the
`bendevries.github.io/plot-verify` Pages deploy refreshes. All code, tests, and
dependencies below are **Python**.

1. **Confirm the app entry.** Locate `app.py`/`app_ui`/`app_server` (or the
   `shiny.express` module) and how `shinylive export` is invoked (Makefile, CI, or
   a script). Note the pinned `shiny`/`shinylive` versions.
2. **Inventory existing modules.** Locate: (a) the **calibrate** screen (axis-point
   clicking + scale selection), (b) the **overlay/plot** view, (c) the **image loader**,
   (d) any existing **data model / state store**, (e) the **export** path, (f) which
   **plot types** are already supported. Grep for `calibrat`, `overlay`, `axis`,
   `pixel`, `download`, `reactive.Value`/`reactive.calc`, `render.image`/`render_widget`.
3. **Map the current happy path** (upload image → calibrate → digitize/overlay → export)
   and mark exactly where the JSON path will **short-circuit** the calibrate step.
4. **Coordinate spaces.** Confirm how the app currently maps between the image's
   **natural** pixels and the **displayed/canvas** pixels (this is the #1 source of
   overlay misalignment — see §6.4). Note the display approach in use (`shinywidgets`
   + Plotly `FigureWidget`, `render.image`, an `ui.HTML` `<canvas>`, or `matplotlib`
   via `render.plot`). **Target Plotly via `shinywidgets`** for drag-edit UX.
5. Only after this inventory, implement per the milestones in §9.

---

### Python dependencies (add to the app's requirements / shinylive assets)
`shiny`, `shinywidgets`, `plotly`, `numpy`, `pandas`. (Optional for a matplotlib
fallback overlay: `matplotlib`.) Everything must be **pure-Python / WASM-friendly**
for shinylive — avoid native-only libs (no OpenCV in-app; the agent already did the
pixel work, the app only transforms/renders). Tests use `pytest`.

## 2. Architecture target (keep it small and modular)
Introduce (or refactor toward) four cohesive modules with a single reactive **state**:

- **`ingest`** — accept JSON (paste box **and** file upload), validate against the
  contract, load the image (from `image.data_uri` or a user-attached file), populate state.
- **`calib`** — pure transform layer: build `to_data`/`to_pixel` per axis (linear+log),
  plus display↔natural scaling. **No UI.** Reused by render + edit + export.
- **`overlay`** — the review/edit screen: draw the figure + digitized layer, host the
  controls, handle drag/click/table edits, keep state in sync.
- **`io_export`** — serialize edited state back to **JSON v(n+1)** and **CSV** (canonical
  schema), plus a **PNG snapshot** of the current overlay; versioned filenames.

Central **state** (reactive) fields: `image{natural_w,natural_h,src}`, `plot_type`,
`orientation`, `axes{x,y}` (scale/label/unit/calibration), `series[]`, `rows[]`
(each row is the contract row + derived `px_geom`), `view{zoom,pan,scale}`,
`ui{visible_series, show_markers, show_intervals, show_labels, opacity, color_mode}`,
`dirty` (edited?), `version`.

---

## 3. The JSON contract v1.0 (authoritative input — implement exactly)
Full spec lives in `plotverify_json.md`; the essentials Claude Code must honor:

- **Top level:** `schema_version`, `generator`, `image{filename,width_px,height_px,
  data_uri?,sha256?}`, `plot_type`, `orientation` (`vertical`=value on y,
  `horizontal`=value on x), `axes{x,y}`, `plot_area_px?`, `series[]`, `rows[]`, `notes`.
- **Axis:** `{scale:"linear"|"log", label, unit, calibration:[{pixel,value}, …≥2]}`.
  For `x`, `pixel` is the **horizontal** natural-image coordinate; for `y`, the
  **vertical** one (**top-left origin, y grows downward**).
- **Row keys mirror the CSV canonical columns** + type extensions:
  `series, x, y, x_err_lower, x_err_upper, y_err_lower, y_err_upper, n, sd, x_label,
  y_label, x_unit, y_unit, x_scale, y_scale, error_bar_type, is_summary, value_source,
  series_color, status` (+ `box_q1, box_median, box_q3` for box; `at_risk` for KM;
  optional `px,py` QC hints). `n`/`sd` may be the string `"NA"`. **Interval columns
  are absolute axis coordinates, never half-lengths.**
- **Source of truth:** **calibration + row data**. The app **derives** every pixel
  position; it must not require `px`/`py`. On edit it recomputes values from pixels
  (inverse) and pixels from values (forward). Round-trip must be exact.

### Transforms (implement once in `calib`)
Using first & last calibration anchors `(p1,v1),(p2,v2)`:
- **linear:** `to_data(px)=v1+(v2−v1)/(p2−p1)·(px−p1)`;
  `to_pixel(v)=p1+(v−v1)·(p2−p1)/(v2−v1)`.
- **log** (`L=log10`): `to_data(px)=10^(L(v1)+(L(v2)−L(v1))/(p2−p1)·(px−p1))`;
  `to_pixel(v)=p1+(L(v)−L(v1))·(p2−p1)/(L(v2)−L(v1))`.
Guard: log requires `v>0`; on non-positive edited value, reject the edit and flash the field.

### Validation on ingest (fail loud, degrade gracefully)
1. `schema_version` starts with `1.`. 2. Each **used** axis has ≥2 anchors with distinct
`pixel`; log axes have positive `value`. 3. Every `row.series` exists in `series[]`.
4. Required columns for `plot_type` present (per §5). 5. Image available (`data_uri`
**or** user attaches an image whose natural size equals `width_px/height_px` — warn on
mismatch and offer to scale calibration). On any failure: show a clear, specific message;
allow the user to open the **manual calibrate fallback for the offending axis only**.

---

## 4. Auto-calibration flow — remove the calibrate screen from the happy path
1. **New landing action:** "Load plot-verify JSON" with two inputs — a **paste
   textarea** and a **file picker** (`.json`). Accept either.
2. On valid JSON → **skip calibrate entirely** and route straight to the **overlay**
   screen with transforms already built from `axes.*.calibration`.
3. **Image resolution:** prefer `image.data_uri`; if absent, prompt "Attach the source
   image" and verify natural dimensions against `image.width_px/height_px`.
4. **Do not delete the calibrate code** — demote it to an optional **"Recalibrate"**
   affordance (per-axis) reachable from the overlay for the degraded/edge cases in §3.
   Default happy path never shows it.
5. Preserve any legacy "upload image → manual calibrate" entry as a secondary tab so
   existing users aren't broken; the JSON path is the promoted default.

---

## 5. Render every unified plot type (drive rendering from `plot_type`)
All geometry comes from `to_pixel(value)` on the value axis/axes; then scale to display
(§6.4). Respect `orientation` for forest/box/bar (`horizontal` → value on x, use
`x_err_*`; else value on y, use `y_err_*`).

- **scatter** — one hollow marker per row at `(to_pixel_x(x), to_pixel_y(y))`.
- **error_bar** — marker at `(x,y)` + a vertical segment from `to_pixel_y(y_err_lower)`
  to `to_pixel_y(y_err_upper)` at `to_pixel_x(x)`, with end caps.
- **line_timeseries** — per series, sort rows by `x`, draw a **polyline** through
  vertices + markers; where `y_err_lower/upper` present, draw band-edge marks (or a
  translucent band if both edges exist across consecutive x).
- **bar** — marker at each bar top `(x,y)`; whisker between `y_err_*` when present.
  (Horizontal bars: value on x.)
- **forest** — marker at the estimate + a **value-axis** segment between the two interval
  ends; rows with `is_summary=true` drawn as a **diamond**. Log value axes common —
  transforms already handle it. Ignore null/reference lines (they're not rows).
- **box** — per category `x`: draw hinge lines at `box_q1`,`box_median`,`box_q3` and
  whisker caps at `y_err_lower`/`y_err_upper`; rows with `status="outlier"` are single
  markers (blank box columns).
- **kaplan_meier** — per arm, draw a **step** polyline in `x` order (horizontal then
  vertical segments — **never interpolate across a drop**); `status="censored"` rows as
  small tick marks on the curve; `y_err_*` as band edges; expose `at_risk` in tooltips.

**Series identity & color:** group by `series`; default the drawn layer to the
**complement** of `series_color` (per-channel `255−c`) so it pops against the original —
this matches the agent's overlay convention and the user's stated preference. Offer a
toggle for a single flat high-contrast color (magenta/lime) as an alternative.

---

## 6. The overlay/review screen (clean, simple, fast)
### 6.1 Layout
A two-pane responsive layout with `ui.layout_sidebar` (or `ui.layout_columns` + cards):
- **Left (canvas):** the figure with the digitized layer on top; zoom/pan; hover
  tooltips; selection highlight.
- **Right (controls + table):** the control panel (6.2) above an **editable data table**
  bound to `rows[]` (6.3).

### 6.2 Controls (keep them few and obvious)
- **Series visibility:** a checkbox per series (color swatch = drawn color).
- **Layer toggles:** Markers · Intervals/whiskers · Connecting lines (line/KM) ·
  Value labels · Band fill.
- **Layer opacity:** slider 0–100%.
- **Marker size:** slider.
- **Color mode:** `Complement (default)` · `Flat magenta` · `Flat lime`.
- **Zoom:** slider + scroll-to-zoom + "Fit" reset; **Pan:** drag on empty canvas.
- **Snap:** optional "snap edits to nearest lit pixel of the original mark" (nice-to-have).
- **Recalibrate (advanced):** collapsed; opens per-axis manual calibration fallback.

### 6.3 Editable table (two-way bound to the overlay)
Use Shiny's **`render.data_frame` with `render.DataGrid(..., editable=True)`** and the
`.data_view()` / cell-edit reactive to capture edits; keep it bound to `state.rows`.
- Columns = the row schema relevant to `plot_type` (hide always-blank columns).
- Editing a value cell → forward-transform → **marker moves** immediately.
- Selecting a table row → highlights its mark on canvas (and vice-versa).
- Row actions: **add**, **duplicate**, **delete**, set `status` (e.g., mark `outlier`/
  `censored`). New rows get sensible defaults (current series, mid-view position).
- Show derived read-outs (e.g., recomputed `sd` if you port `back_calc_sd`, optional).

### 6.4 Coordinate handling (critical — prevents misalignment)
- All calibration/pixels are in **natural** image space. Compute
  `scale = displayed_px / natural_px` (guard non-uniform scaling — keep aspect ratio).
- Draw at `display = natural * scale + pan_offset`, inside a zoom transform.
- On pointer events, invert: `natural = (display − pan_offset) / scale`, **then**
  `to_data(natural)`. Never mix spaces.
- **Implementation (Python Shiny):** use **`shinywidgets.render_widget`** with a Plotly
  **`go.FigureWidget`**:
  - Background = the figure via `fig.add_layout_image(source=data_uri, xref="x",
    yref="y", sizing="stretch", layer="below")`, with axes set to the image's natural
    pixel extents and `yaxis.autorange="reversed"` (image origin top-left).
  - Marks = one `go.Scatter` trace **per series** (markers) plus line/interval traces;
    plot marks in **natural-pixel** coordinates (`to_pixel(value)`) so they sit on the
    image, or plot in data space with a second overlaid axis — pixel space is simpler
    and matches the fixtures.
  - Edits: enable `fig.update_layout(dragmode="drawopenpath"/"pan")` and capture point
    drags via `FigureWidget` `on_click`/selection callbacks, or a "select + nudge with
    number inputs" fallback (Plotly-in-Shiny drag callbacks can be finicky — see §6.5).
  - Keep the widget in a `reactive.Value`; mutate `trace.x/trace.y` in a `with
    fig.batch_update():` block on every edit so the canvas re-renders in place.
- **Fallback** if `FigureWidget` drag proves unreliable in shinylive (WASM): render a
  static Plotly/`matplotlib` overlay via `render.plot`, and drive all edits from the
  **editable table** (§6.3) — clicks on the plot select the nearest row.
- Either way: **image is the background, marks are an overlay layer**, and one edit
  updates exactly one row.

### 6.5 Editing → data flow
1. Drag mark end N pixels → `natural` → `to_data` → write the corresponding row field
   (`x`/`y` or an interval end) → re-render → set `dirty=true`.
2. Edit a table value → validate (log positivity, numeric) → `to_pixel` → move mark.
3. Every edit updates a single source of truth (`state.rows`); canvas and table both
   read from it (no divergent copies).

---

## 7. Export (versioned, faithful, round-trippable)
- **Edited JSON:** re-serialize the **full contract** from state (refresh `rows[*].px/py`
  from current geometry, bump a `generator.edited_utc`, keep original `generator` info,
  add `"edited_in":"plot-verify"`). Filename: **increment version** — if input was
  `foo.json`, write `foo_v2.json`, then `foo_v3.json`, … (mirror the agent's
  `versioned_path` rule; never overwrite).
- **CSV:** canonical column order (§3) + this type's extension columns appended; RFC 4180;
  `n`/`sd` default `"NA"`. Same versioned stem.
- **PNG snapshot:** current overlay as displayed (for pasting into QC decks).
- Provide all three as download buttons; if practical, also a "Copy JSON" button.

---

## 8. Validation, errors, and edge cases to handle
- Missing image + no `data_uri` → prompt attach; verify dimensions; offer proportional
  calibration rescale if the attached image was resized.
- Log axis with an anchor `value ≤ 0` → reject with a specific message.
- `orientation` mismatch (e.g., forest with only `y_err_*` but `horizontal`) → warn and
  fall back to the populated interval columns; note in a banner.
- Unknown `plot_type` or unknown extension columns → render the generic
  point+interval view and surface a non-blocking warning.
- Very large `data_uri` → lazy-load; don't block the UI thread.
- Duplicate/overlapping marks → selection picks the nearest; allow nudge via arrow keys.

---

## 9. Milestones / PR breakdown (ship incrementally)
1. **M1 — Ingest + transforms + validation** (`ingest`, `calib`): paste/upload JSON,
   build transforms, unit-test round-trip (linear & log) and validation. No UI change yet.
2. **M2 — Auto-calibration routing:** JSON path bypasses calibrate; image via `data_uri`
   or attach; demote calibrate to advanced fallback.
3. **M3 — Overlay render (all 7 types):** background image + complement-colored layer,
   correct natural↔display scaling; series visibility + basic toggles.
4. **M4 — Controls polish:** opacity, marker size, color mode, zoom/pan/fit, labels, bands.
5. **M5 — Editable table + drag-edit:** two-way binding, add/duplicate/delete, status
   edits, log-guard.
6. **M6 — Export:** versioned JSON + CSV + PNG; "Copy JSON".
7. **M7 — Tests + docs:** fixtures per type (§10), README update, and re-run
   `shinylive export` so the GitHub Pages deploy refreshes.

---

## 10. Testing (block the merge until these pass)
- **Transform unit tests:** linear and log `to_data∘to_pixel == identity` (±1e-6);
  known-anchor sanity (e.g., `y=0.1`→p1, `y=10`→p2 on a log axis).
- **Contract fixtures:** one minimal JSON per `plot_type` (scatter, error_bar, forest
  [horizontal + a summary diamond], line_timeseries [with a band], bar [grouped, with
  whiskers], box [with an outlier], kaplan_meier [with censor ticks + at_risk]). Assert
  they ingest, render the expected number of marks per series, and export a byte-stable CSV.
- **Round-trip test:** ingest → export JSON → re-ingest → deep-equal (modulo refreshed
  `px/py` and timestamps).
- **Alignment test:** with a `data_uri` fixture whose anchors are known, assert drawn
  marker natural-pixels equal `to_pixel(value)` within 1px.
- **Regression:** legacy image-only manual-calibrate path still works.

---

## 11. Acceptance criteria (definition of done)
- Pasting/uploading a v1.0 JSON opens the **overlay with no calibrate step**.
- All **seven** plot types render correctly, series-separated, in complement colors, with
  markers landing on the original marks (±1px on fixtures).
- Controls (visibility, toggles, opacity, size, color mode, zoom/pan) all work.
- Users can **drag marks and edit table values**, changes stay in sync both ways, and
  log-axis positivity is enforced.
- Export produces a **versioned** edited JSON (round-trippable), a canonical CSV, and a
  PNG snapshot.
- `shinylive export` succeeds and the Pages deploy shows the new flow.

---

## 12. Non-goals (out of scope for this pass)
- Re-running image digitization inside the app (extraction stays in the agent/`digitize.py`).
- OCR, auto-detection of axes, or auto-segmentation in-app.
- Multi-figure/batch projects (single figure per session for now).
- Server-side persistence (state lives in-session; export is the save mechanism).

---

## 13. Notes for reviewers / handoff
- Keep the JSON contract in lockstep with the agent: any schema change updates
  **both** `plotverify_json.md` (agent side) and the app's `ingest`/`io_export`.
- Prefer **plotly-based** rendering for the smoothest drag-edit UX in either framework;
  fall back to a table-driven / `render.plot` edit model only if `FigureWidget` drag
  proves limiting under shinylive/WASM.
- Everything the app needs is in **calibration + rows** — resist adding new required
  fields; extend via optional keys so old JSON keeps loading.
