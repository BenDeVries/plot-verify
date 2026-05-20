# Second Plan: Post-Detection-Fix Migration From Streamlit to Shiny With Batch Processing

## Execution Order

This plan should be executed only after completing the first plan, which addresses:

1. Inward tick detection.
2. Scientific notation parsing.
3. Right-side text exclusion.
4. Improved calibration diagnostics.
5. Better OCR/tick pairing reliability.

The goal of this second plan is to migrate the app from Streamlit to Shiny while shifting the primary user workflow from single-image review to batch image/CSV processing.

## Current Status (as of 2026-05-20)

### Completed

- **All Part 1 bugs (#10–#19)** — see inline status notes per bug below.
- **All Part 8 refactors A–G** — `plotverify_core/` package fully extracted (~2000-line `app_auto_axis.py` now consumes the core).
- **Part 8 Refactor H, Phases 1–2** — `axis_pipeline/legacy.py` entries emit `DeprecationWarning`; `update_result_from_tick_edits` / `rebuild_result_from_detection` exist as typed replacements.
- **Milestones 1 and 2** — algorithm stabilization (per the first plan) and UI-independent core extraction.
- **Milestone 3** — single-image Shiny app. See detailed status below.
- **Real-image regression infrastructure** — `tests/test_real_image_regression.py` parametrizes over every verified diagnostic in `test_images/verified_raw_detection_diagnostics/`. Current test count: **156 passed, 1 xfailed** (only `lin_log` remains xfail; see drift section below).
- **1-SE outlier-exclusion regression test** — `tests/test_calibration_1se.py` locks in the rule that the 1-SE threshold must come from the min-rmse candidate's own residuals (the previous bug let a mispaired y=0 inflate the threshold past the point at which it could exclude itself).
- **Manual-mode overlay contract** — `tests/test_overlay_manual_mode.py` confirms `manual_calibration() + render_overlay()` draws P1/P2/P3 anchors without any geometric bbox.
- **AppState JSON serialization** — `plotverify_core/serialization.py` saves/loads sessions as `.pvsession` zip artifacts (manifest.json + images/ + csvs/ + overlays/). `PIPELINE_VERSION = "0.1"` lives in `axis_pipeline/__init__.py`; on load, a mismatch with the saved manifest's `pipeline_version` discards the saved `CalibrationResult` and re-invokes the pipeline (manual or auto). `PlotVerifyApp` carries `is_dirty` / `mark_dirty` / `autosave_if_dirty` hooks. Covered by `tests/test_serialization.py` (12 tests).
- **Streamlit session save/load** — `plotverify_core/streamlit_bridge.py` translates between the legacy flat `st.session_state` (used by `app_auto_axis.py`) and `AppState`/`PerFileState` so the single-image Streamlit app can use `save_session` / `load_session` without being refactored onto `PlotVerifyApp` as its source of truth. Surfaces in the sidebar as a **💾 Session** expander with explicit Save/Load buttons (autosave-at-callback was deferred in favor of manual save). Default file location is `~/.plotverify/sessions/<session_id>.pvsession`, overridable via the `PLOTVERIFY_SESSION_DIR` env var; `session_id` is a uuid4 prefix stored under `__pv_session_id` and stable across reruns of the same tab. **Critical invariant** (regression-pinned in `tests/test_streamlit_bridge.py`): the rehydrated `image_hash` flat key must equal `md5(image_bytes)` — `_load_image_from_upload` uses that equality to early-return; a mismatch makes it treat the still-mounted file_uploader as a NEW image and reset the P1/P2/P3 anchor flat keys to default seed positions, silently overwriting the values just restored. Covered by `tests/test_streamlit_bridge.py` (14 tests).

### Algorithmic Drift Resolution (closed 2026-05-19)

The 6 xfailed cases from the initial drift sweep collapsed into 3 distinct root causes, all of which have now been fixed in-tree. The final state is **5 of 6 cases green**, with only `lin_log` still xfail (a synthetic image with extreme 1e+90 y-values that is more an OCR/numerics edge case than a pipeline bug).

| Image | Resolution |
|---|---|
| `case1_single_arm_pk` | ✓ Passing — fix #1 (geometry length-filter) |
| `case2_two_arm_pk_ddi` | ✓ Passing — fix #1 (geometry length-filter) |
| `iga_zigakibart` | ✓ Passing — fix #1 + side-effect fix (gated x-band rightward extension) |
| `IgA_povetacicept` | ✓ Passing — fix #2 (trailing-sign-strip in parser) + re-baselined |
| `iga_sc_povetacicept` | ✓ Passing — fix #2 (trailing-sign-strip) + re-baselined |
| `lin_log` | ✗ xfail — rotated y-band missed `0e+00` this run; with only 4 ticks the 1-SE rule then drops the topmost `4e+90` as numerically ill-conditioned (lin_log has linear y-values on the order of 1e+90, not log10). OCR-stability + ill-conditioned-regression, not a deterministic pipeline bug. |

Fixes landed:
- **Fix #1 — `axis_pipeline/geometry.py:_choose_axes`** — drop left-edge vertical candidates whose length is < 60% of the longest before applying the leftmost-by-position rule. On case1/case2/zigakibart, unmasked strokes of the rotated y-axis title (at x≈41/60 for case1, x≈96/102/108 for zigakibart) were beating the real y-axis (at x=169 and x=210 respectively) because the algorithm picked `min(pos)`. The real axis is always the strongest candidate, so a length floor selects it correctly without regressing any passing case.
- **Fix #2 — `axis_pipeline/ocr.py:parse_numeric_tick`** — when `float(s2)` raises, retry once after stripping a single trailing `+`/`-`. EasyOCR sometimes pulls a tick-mark glyph into the label producing strings like `'-20-'`, `'-60-'`, `'-40-'`. No legitimate numeric format ends in a sign, so the strip-and-retry is safe. Leading signs are never stripped (would silently sign-flip a real negative label).
- **Side-effect fix — `axis_pipeline/ocr.py:x_label_band` + `axis_pipeline/pipeline.py`** — add opt-in `extend_outward` parameter that extends the x-band rightward past `bbox.right`. The pipeline gates this on `image_width - bbox.right < 30` so it only fires when bbox.right is near the image edge (the configuration that clips a centered rightmost label like zigakibart's `'120'` to `'12'`). Crucially this gate prevents the extension from masking the `loglog_inner_tick` deliberate-failure case (right margin = 54 px). The band is NOT extended leftward — the y-tick label strip sits there, and capturing y labels in the x-band confuses pairing (verified by the case3 regression that left-extension introduced).

Re-baselined diagnostics:
- `test_images/verified_raw_detection_diagnostics/iga_povetacicept_success.txt` — 4 y-ticks now paired (was 2); OLS calibration over -20/-40/-60/-80 within 0.5% of the previous two-point scale.
- `test_images/verified_raw_detection_diagnostics/iga_sc_povetacicept_success.txt` — calibration now uses the real `-40`/`-60`/`-80` labels (scale ≈ -0.28) instead of accidentally calibrating off rotated y-axis-title fragments (verified scale was -0.099, which encoded the broken-parser state).

Each `pytest.mark.xfail(strict=True)` will fail loudly if the case starts passing, so any future regression here surfaces immediately and prompts removal of the marker.

### Band-Override Capture Protocol (Do Not Simplify)

The diagnostics in `test_images/verified_raw_detection_diagnostics/*.txt` were captured with **per-image** custom y/x band extras (e.g. `case3` = (130, 64); `iga_sc_povetacicept` = (140, 55)). Those values are stored inside each diagnostic under `diagnostics.y_band_extra_used` / `diagnostics.x_band_extra_used` and replayed by `tests/test_real_image_regression.py:130-135`:

```python
cfg = CalibrationConfig(
    y_band_extra_px=int(used.get("y_band_extra_used", 90)),
    x_band_extra_px=int(used.get("x_band_extra_used", 28)),
)
```

This is the regression protocol — do **not** rewrite the test to use only `CalibrationConfig()` defaults, and do **not** drop the `y_band_extra_used`/`x_band_extra_used` keys when re-baselining a diagnostic. The (90, 28) fallback in `.get(...)` is a defensive default for diagnostics produced before the keys were recorded; new baselines must include them. Removing or renaming `y_band_extra_px` / `x_band_extra_px` on `CalibrationConfig` would break every case in the suite.

### Outstanding Before Resuming Migration

1. ~~**Investigate the 6 drift cases**~~ — closed 2026-05-19; 5 of 6 cases fixed (see "Algorithmic Drift Resolution" above). `lin_log` remains as a single xfail and is OCR-stability rather than a pipeline bug — acceptable to ship the migration without resolving it.
2. ~~**CI setup**~~ — closed 2026-05-19. GitHub Actions workflow at `.github/workflows/tests.yml` runs the full suite (156 passed, 1 xfailed) on push and PR across Python 3.10/3.11/3.12 with `~/.EasyOCR` cached between runs. Optional local pre-push hook at `scripts/git-hooks/pre-push` (enable via `git config core.hooksPath scripts/git-hooks`). Dependency manifests: `requirements.txt` (runtime) and `requirements-dev.txt` (adds pytest).
3. ~~**Begin Milestone 3**~~ — substantially complete. See Milestone 3 status below.
4. **Overlay plot horizontal squishing on first tab visit** — known JS bug. When `overlay_plot` first renders while the tab pane is `display:none`, Plotly sets an explicit pixel `width` on the `gd` element (using a 0-measurement default). Subsequent `Plotly.Plots.resize()` calls read `getComputedStyle(gd).width` which returns the stale inline value, making the resize a no-op. Current fix attempts: one-shot MutationObserver, `shown.bs.tab` + polling, ResizeObserver + `gd.style.width = ''` + `Plotly.relayout(gd, {autosize: true})`. The ResizeObserver approach is the current strategy but has not yet been confirmed fixed by the user.

> Persistence is no longer a blocker: the Streamlit app gained a manual Save/Load UI backed by `plotverify_core/streamlit_bridge.py`. End-of-callback autosave via `PlotVerifyApp.autosave_if_dirty()` remains available on the controller — it can be wired in later if needed without further surgery to the bridge.

## High-Level Migration Goals

1. Preserve the existing single-image workflow when only one image is uploaded.
2. Add a batch workflow for multiple plot images and matching CSV files.
3. Require CSV files to match uploaded plot image names.
4. Surface unmatched files clearly before processing.
5. Allow per-file masking strategy selection before calibration.
6. Support batch calibration with automatic confidence-based manual review.
7. Improve calibration editing UX using Shiny’s reactive UI model.
8. Improve overlay correction UX by allowing direct point/error-bar edits.
9. Export corrected CSV outputs after interactive review.
10. Support a fully manual calibration workflow that does not require EasyOCR or pytorch.
11. Make EasyOCR an optional runtime feature; detect its availability at startup and adjust the UI accordingly.

# Part 1: Bugs to Address Before Migrating From Streamlit to Shiny

These bugs should be fixed in the current codebase before starting the Shiny migration. The purpose is to avoid migrating known defects and to establish a stable reference implementation.

## 1. Inward Tick Detection Failure

### Problem

The current geometry tick detection misses ticks drawn inside the plotting region because it primarily searches outside the axis frame.

### Why Fix Before Migration

If this remains unresolved, the Shiny app will inherit incorrect calibration results and it will be difficult to distinguish algorithm bugs from migration bugs.

### Outline to Address

1. Complete the first-plan changes in `geometry.py`.
2. Add inward and outward tick candidate detection.
3. Add candidate source diagnostics.
4. Confirm x-axis inward ticks are detected on the failing image.
5. Confirm existing outward-tick plots still pass.
6. Add regression fixtures before migrating.

## 2. Scientific Notation Parsing Failure

### Problem

The x-axis scientific notation labels are not parsed reliably.

Likely problematic examples include:

- `10^-3`
- `10⁻³`
- `1×10^3`
- `1 x 10 -3`
- `1E−03`

### Why Fix Before Migration

The parser should be UI-independent. It should be fully tested before moving to Shiny so that parsing behavior is stable across frameworks.

### Outline to Address

1. Strengthen `parse_numeric_tick()` in `ocr.py`.
2. Normalize superscripts correctly:
   - `10³` → `10^3`
   - `10⁻³` → `10^-3`
3. Add explicit parser branches for:
   - Plain numbers.
   - E notation.
   - Power-of-ten notation.
   - Multiplied power-of-ten notation.
4. Add parse status and parse flag fields.
5. Add unit tests for all expected scientific notation forms.
6. Ensure parser failures are visible in the OCR/calibration point table.

## 3. Right-Side Text Contamination Risk

### Problem

Right-side plot text may be detected by OCR and could accidentally be treated as a tick label if label filters are too broad.

The right-side axis text in the supplied image is not a concern and should not be used for calibration.

### Why Fix Before Migration

The Shiny batch workflow will process many plots automatically. Any right-side text contamination could silently affect calibration across a batch.

### Outline to Address

1. Keep x-axis labels constrained to the x-label band below the bottom axis.
2. Keep y-axis labels constrained to the left-side y-label band.
3. Allow right-side OCR text to be masked during geometry detection if helpful.
4. Prevent right-side OCR text from entering x/y tick tables.
5. Add diagnostics for ignored numeric OCR records outside axis label bands.
6. Add a regression test where right-side numeric text is present but ignored.

## 4. Ambiguous Tick Candidate Source

### Problem

The app currently does not clearly distinguish whether a calibration point came from:

- Outward tick detection.
- Inward tick detection.
- Gridline-supported detection.
- OCR-label-center fallback.
- Manual edit.

### Why Fix Before Migration

The Shiny workflow will rely heavily on batch confidence flags and manual review. Users need to understand why an image was flagged.

### Outline to Address

1. Add tick source/status fields in the pipeline result.
2. Surface the source in the calibration point table.
3. Include source counts in diagnostics:
   - `x_tick_candidates_outward`
   - `x_tick_candidates_inward`
   - `x_tick_candidates_gridline`
   - `y_tick_candidates_outward`
   - `y_tick_candidates_inward`
   - `y_tick_candidates_gridline`
4. Render source-specific markers in the diagnostic overlay.
5. Preserve these statuses in the legacy dict until the Shiny app directly consumes typed results.

## 5. Calibration Confidence Threshold Needs Validation

### Problem

The planned Shiny workflow flags every image with detection confidence below `0.95` for manual review. This threshold may be too strict or too permissive depending on the confidence calculation.

### Why Fix Before Migration

Batch processing depends on confidence to decide which images need human attention.

### Outline to Address

1. Evaluate confidence values on known good and known bad cases.
2. Confirm that high-confidence detections are actually reliable.
3. Confirm that low-confidence detections are appropriately flagged.
4. Add diagnostic components to confidence:
   - Axis frame confidence.
   - Grid fit quality.
   - Number of calibration points.
   - Calibration residual.
   - OCR parse reliability.
5. Consider adding a secondary review flag separate from numeric confidence:
   - `requires_manual_review: true/false`
   - `review_reason: list[str]`
6. Use `requires_manual_review` as the primary batch workflow flag.
7. Keep `confidence < 0.95` as one trigger for manual review.

## 6. OCR Tick Table Editing Fragility

### Problem

The current OCR tick table supports editing values and include/exclude flags, but the state update path may be fragile when recalibrating repeatedly.

### Why Fix Before Migration

The Shiny app will replace this with a more central "Calibration points" editor. The underlying recalibration logic should already be stable.

### Outline to Address

1. Confirm edited x/y tick tables round-trip correctly through `update_detection_from_tick_tables()`.
2. Confirm include/exclude flags persist after recalibration.
3. Confirm edited numeric values update calibration anchors.
4. Confirm invalid edited values are handled gracefully.
5. Add validation messages for:
   - Fewer than two included x points.
   - Fewer than two included y points.
   - Degenerate same-value axis.
   - Non-monotonic values.
6. Add tests for repeated edit/recalibrate cycles.

## 7. Manual Calibration Anchor Inconsistencies

### Problem

Manual calibration uses P1, P2, and P3, but prior issues suggest anchors can land on visually confusing positions, such as the axis bar instead of the intended data-value location.

### Why Fix Before Migration

The Shiny app will introduce draggable P1/P2/P3 anchors. If the underlying anchor semantics are unclear, drag editing will be confusing.

### Outline to Address

1. Define anchor semantics explicitly:
   - P1: first x-axis calibration point.
   - P2: second x-axis calibration point.
   - P3: y-axis calibration point.
2. Store both pixel coordinates and data values for each point.
3. Ensure P1/P2 can represent the x-axis baseline without incorrectly inheriting a y tick value.
4. Ensure P3 can have both:
   - A y data value.
   - An x data value if needed for display or future extension.
5. Add clear warnings when anchor values are missing or inconsistent.
6. Make anchor display labels match the same names used in the future Shiny UI.

## 8. Image/CSV Name Matching Rules Need Definition

### Problem

The batch workflow requires every CSV to have the same name as an uploaded plot, but matching rules are not yet formalized.

### Why Fix Before Migration

File matching is central to the Shiny batch workflow. Ambiguous matching rules will cause user confusion and data mismatches.

### Outline to Address

1. Define canonical file stem matching:
   - `plot1.png` matches `plot1.csv`
   - `plot1.jpg` matches `plot1.csv`
   - Matching should ignore extension only.
2. Decide whether matching is case-sensitive.
   - Recommended: case-insensitive stem matching, but preserve original file names.
3. Decide how to handle duplicate stems.
   - Example: `plot1.png` and `plot1.jpg` both uploaded.
4. Add unmatched file categories:
   - Image without CSV.
   - CSV without image.
   - Duplicate image stem.
   - Duplicate CSV stem.
5. Build a matching utility independent of Streamlit/Shiny.
6. Test this utility before migration.

## 9. Overlay Coordinate Editing Not Yet Available

### Problem

The current overlay displays extracted data but does not support directly moving points or editing error bars by dragging.

### Why Fix Before Migration

This feature is planned for Shiny, but the data model should be prepared before migration.

### Outline to Address

1. Define an editable overlay data model:
   - `series`
   - `point_id`
   - `x`
   - `y`
   - `y_err_lower`
   - `y_err_upper`
   - `series_color`
   - `source_file`
   - `edited`
   - `edit_timestamp`
2. Ensure every plotted point has a stable ID.
3. Decide how to handle duplicate points.
4. Define how interval fill updates when a point moves.
5. Define how upper/lower error bars update independently.
6. Add a pure data-update function that can be called from either Streamlit or Shiny.

## 10. EasyOCR Must Be an Optional Dependency  ✅ Done

`axis_pipeline.ocr_available()` and `axis_pipeline.manual_calibration()` exist; the Streamlit app shows a banner and disables auto-only controls when EasyOCR is absent. Covered by `tests/test_core_no_streamlit.py` (subprocess-isolated, runs without Streamlit) and the typed manual-calibration path.

### Problem

EasyOCR requires pytorch, which is a large dependency tree. Users who only need to verify plots manually should not be forced to install GPU drivers, CUDA toolkits, or multi-gigabyte model weights just to run the app.

Currently the codebase already uses a lazy EasyOCR import and an `ocr_runner` injection point, but the app does not degrade gracefully when EasyOCR is absent.

### Why Fix Before Migration

The Shiny app is designed around manual calibration as a first-class workflow. If this is not implemented before migration, the Shiny UI will implicitly assume OCR is always available, and removing that assumption later will require reworking large sections of the UI.

### Outline to Address

1. Add an `ocr_available()` helper in `axis_pipeline` that attempts to import `easyocr` and returns `True`/`False`.
2. Ensure the pipeline raises a clear, catchable error (not a bare import error) when `run_calibration` is called without an `ocr_runner` and EasyOCR is not installed.
3. Confirm that `detect_axis_frame` works with no OCR dependency (geometry only).
4. Confirm that manual anchor values (P1/P2/P3 + data values) can produce a valid `CalibrationResult` without calling the full pipeline.
5. Add a pure manual calibration function:
   - Accepts P1, P2, P3 pixel positions and their corresponding data values.
   - Constructs an `AxisCalibration` from those three points directly via OLS or direct solve.
   - Returns a `CalibrationResult` with a `source: "manual"` field.
6. Document the two modes clearly:
   - **Auto mode**: full pipeline including OCR (requires EasyOCR / pytorch).
   - **Manual mode**: user-supplied anchors only (no OCR, no pytorch required beyond numpy/opencv).
7. Add unit tests for the manual calibration path that run without EasyOCR installed.

## 11. Log-Scale Axes Silently Miscalibrate in the Streamlit Overlay  ✅ Done

`plotverify_core.calibration_math` carries `x_log_base` / `y_log_base` through `compute_calibration`, `px_to_data`, and `data_to_px`. Streamlit `cal` dict propagates the log base from `AxisCalibration`; overlay axis type is set accordingly.

### Problem

`axis_pipeline` auto-detects log10 axes and returns `AxisCalibration.log_base=10.0` with the slope/offset fitted in log space. The Streamlit app's `compute_calibration` (`app_auto_axis.py`), `px_to_data`, and `data_to_px` are linear-only — they ignore `log_base` and treat the values as if pixel and data were related by a single straight line. When the pipeline detects log10 and the user clicks **Apply Calibration**, the OCR-derived data values (e.g. 1, 10, 100, 1000) are linearly regressed against pixel positions. The overlay is then geometrically off by a factor that grows exponentially across the axis.

### Why Fix Before Migration

This is silent, visually-plausible miscalibration on any log-axis plot — the most common failure mode the verification tool is supposed to catch. Migrating it into Shiny would preserve the bug across the whole batch workflow.

### Outline to Address

1. Extend the Streamlit `cal` dict to carry `x_log_base` and `y_log_base` (both optional, default `None`).
2. Update `compute_calibration` to accept and emit log-axis parameters, or replace it with a thin adapter around `AxisCalibration`.
3. Update `px_to_data` / `data_to_px` to apply `10**(linear)` or `log10(...)` per axis when `log_base` is set.
4. Update `build_overlay_figure` so the image's data-coordinate bounds, the Plotly axis `type="log"` setting, and the axis range all reflect log scaling.
5. When loading detection values into manual fields, carry through `x_calibration.log_base` / `y_calibration.log_base` instead of dropping them.
6. Add regression tests with a known log-axis plot.

## 12. Tick-Table Edits Desync the Typed CalibrationResult  ✅ Done

`axis_pipeline.legacy.rebuild_result_from_detection` reconstructs the typed `CalibrationResult` after tick-table edits; the Streamlit app calls it on every "Re-calibrate from edits" click. Typed-edit path covered by `tests/test_typed_tick_edits.py`.

### Problem

`_render_calibration_tab` renders the diagnostic overlay from `st.session_state.auto_axis_result` (the typed `CalibrationResult`). The "Re-calibrate from edits" button updates `st.session_state.auto_axis_detection` (the legacy dict) but never touches `auto_axis_result`. So after the user edits a tick value, the detection-results table reflects the edit but the diagnostic overlay still shows the pre-edit paired ticks and anchors.

### Why Fix Before Migration

Calibration-points editing is one of the centerpieces of the Shiny calibration tab. The Streamlit app's stale-overlay bug will reappear and be harder to diagnose once the UI is more interactive.

### Outline to Address

1. After `update_detection_from_tick_tables` completes, rebuild `auto_axis_result` from the new legacy dict via `_legacy_dict_to_result`, OR clear `auto_axis_result` so the overlay falls back to `build_ocr_debug_overlay`.
2. Preferred: extend `update_detection_from_tick_tables` (in `axis_pipeline/legacy.py`) to also return an updated `CalibrationResult`, so the app can keep both in sync without round-tripping through dicts.
3. Add a regression test that edits a tick value, recalibrates, and confirms `result.x_paired_ticks[i].data_value` reflects the edit.

## 13. Frame-Preview Cache Ignores OCR Confidence Slider  ✅ Done

`_get_or_compute_frame_preview` cache key now includes `min_ocr_confidence`; slider changes invalidate the cached Phase A records.

### Problem

`_get_or_compute_frame_preview` keys its cache only on `image_hash`. When the user changes the **Min OCR confidence** slider in Detection Settings, the band preview tab continues to show stale Phase A records computed with the old threshold. The next **Run Detection** uses the new value, but the live preview lies.

### Why Fix Before Migration

Reactive previews are central to the Shiny editing workflow. Carrying this caching bug forward means slider feedback will be inconsistent across panels.

### Outline to Address

1. Include `min_ocr_confidence` (and any other config that affects Phase A output) in the cache key.
2. Consider hashing a stable subset of `CalibrationConfig` so future config additions are caught automatically.
3. Add a test that changes the slider and verifies the cached preview is invalidated.

## 14. Stale `vis_<series>` Widget Keys Leak Across CSV Reloads  ✅ Done

Streamlit `render_sidebar` sweeps `vis_*` (and related per-series widget keys) on CSV change. Workaround until per-file state is fully scoped by `PerFileState` (which now exists; Streamlit can adopt the scoped approach as part of incremental integration).

### Problem

When a new CSV is loaded, `_init_series_states` seeds `st.session_state[f"vis_{name}"] = True` for the new series, but the old CSV's `vis_<old_name>` keys are never removed. `build_calibration_image` and `build_composite` iterate `series_states` (which is rebuilt) but read visibility via `st.session_state.get(f"vis_{name}", True)`. A leftover `vis_X=False` from a previous CSV could shadow a same-named series in the new CSV, producing an incorrectly masked calibration image on first detection.

### Why Fix Before Migration

In the Shiny batch workflow, per-file state needs to be cleanly partitioned. A stray pattern of "widget keys persist forever" will spread across files and make state debugging far harder.

### Outline to Address

1. On CSV change in `render_sidebar`, sweep `st.session_state` for keys starting with `vis_` and remove any whose suffix is not in the new series list.
2. Same for `cpick_*`, `interp_btn_*`, `vis_btn_*`, and other per-series widget keys.
3. Add a debug-state check that flags orphaned per-series keys.
4. In Shiny: scope per-series state under a per-file dict, not the global reactive registry.

## 15. CSV Loader Requires `y_err_lower` / `y_err_upper` Columns Even When Unavailable  ✅ Done

`REQUIRED_COLUMNS` is now `["series", "x", "y"]`; `plotverify_core.csv_io.load_csv` adds NaN-filled error columns if absent and surfaces an info note via `LoadReport`.

### Problem

`REQUIRED_COLUMNS` includes `y_err_lower` and `y_err_upper`. CSVs without error-bar columns (common output from many extraction pipelines) are rejected with a hard error, even though the rest of the app handles `NaN` error values correctly.

### Why Fix Before Migration

This is a real adoption blocker — users discover the requirement only after they have already extracted data, and have to manually add empty columns to every CSV.

### Outline to Address

1. Move `y_err_lower` and `y_err_upper` out of `REQUIRED_COLUMNS`. Required set becomes `["series", "x", "y"]`.
2. In `_load_csv`, add missing error columns filled with `NaN` if absent.
3. Confirm `build_overlay_figure` already handles all-NaN error columns (it does via `has_err = np.isfinite(eu) & np.isfinite(el)`).
4. Add a test using a CSV with no error columns.
5. Surface a small info note ("no error bars detected — overlay will show points only") rather than a warning.

## 16. Reversed Error Bars Are Silently Displayed Inverted  ✅ Done

`plotverify_core.csv_io.load_csv` auto-swaps reversed `y_err_lower` / `y_err_upper` rows and records the count in `LoadReport`. The Streamlit app surfaces the warning.

### Problem

The convention is `y_err_lower ≤ y ≤ y_err_upper`. AI-extracted CSVs occasionally swap the columns. The current overlay code computes `arr_plus = eu - y` and `arr_minus = y - el` without checking the sign, so swapped rows produce error bars pointing the wrong direction and a "ribbon" that inverts vertically. The user has no warning that the data is malformed.

### Why Fix Before Migration

Verification of error bars is one of the app's selling points. Silent display of inverted bars defeats the purpose.

### Outline to Address

1. In `_load_csv`, after parsing, count rows where `y_err_lower > y` or `y_err_upper < y`.
2. If > 0 rows are inverted, surface a Streamlit warning with the count and a sample of affected (`series`, `x`) pairs.
3. Offer a one-click "Swap reversed error columns" button that normalises just the affected rows.
4. Add an audit column in the future export CSV (per Part 3 §11) recording whether the row was auto-swapped.

## 17. Very Large Image Uploads Silently Degrade or Crash  ✅ Done

`plotverify_core.image_io.decode_and_maybe_downscale` returns an `ImageLoad` with the downscale factor recorded. Streamlit shows a warning and offers an opt-out toggle when the threshold is hit.

### Problem

No size check exists on uploaded images. An 8000×6000 plot quietly consumes minutes of CPU during Phase A OCR and several GB during dark-mask preparation. Streamlit may appear to hang or run out of memory with no progress feedback. EasyOCR scales roughly quadratically with image area.

### Why Fix Before Migration

The Shiny batch workflow will multiply this problem across many files. A single oversized image can stall the batch queue.

### Outline to Address

1. On upload, check image dimensions and file size.
2. If the image exceeds a configurable threshold (default: max edge > 4000 px or file > 25 MB), show a banner offering automatic downscale (e.g. max edge 3000 px) before proceeding.
3. Cache the downscaled image and use it for calibration; map calibration anchors back to the original-image coordinate space if the user later needs them.
4. Record the downscale factor in `diagnostics` so the overlay can compose against the original image when appropriate.
5. Add a regression test using a synthetic large image.

## 18. Manual Mode No Longer Relies on Geometric Detection  ✅ Done

### Decision (2026-05-18)

The original outline proposed adding a non-ML text-region heuristic so geometry-only frame detection would be reliable enough for the manual workflow. That approach was abandoned in favor of removing the dependency entirely: **manual mode never invokes geometric axis detection.** P1/P2/P3 anchors are placed at sensible defaults on image load (10%/90% of image dimensions), and the user drags them to the correct tick positions. Calibration is computed from the three anchor points and their data values via `axis_pipeline.manual_calibration`. `CalibrationResult.bbox` remains `None` in this mode and the overlay renderer tolerates that.

This eliminates the geometry-drift failure mode entirely on the manual path. The tradeoff is a small cold-start cost (three explicit drags instead of nudging seeded anchors), accepted because (a) the geometry seed was unreliable enough that "Bug #18 fixes" would have been an open-ended quality investment, and (b) the manual-mode UI is the ground-truth path — it should not pretend to know where the axes are.

### Scope of `detect_axis_frame` going forward (option c)

- **Stays as a public API.** `axis_pipeline.detect_axis_frame` is still used by the auto-mode "X/Y label bands" tuning panel and by tests, and `run_calibration` internally relies on the same `_detect_frame_internal` helper.
- **No surface in manual mode.** No "Detect axis frame" button, no band controls, no frame-preview cache reads. These UI elements are gated on `ocr_available()`.
- **The Streamlit "preserve previous bbox so the overlay works" workaround is removed.** The overlay renderer now draws P1/P2/P3 unconditionally when calibration succeeded; the bbox-dependent decorations (frame rectangle, geometric tick markers) are gated locally on `bbox is not None`.

### Outline to Address

1. `axis_pipeline/overlay.py::render_overlay` — drop the early return on `bbox is None`. Always draw P1/P2/P3 markers and the calibration anchor lines when `result.success`. Gate frame rectangle, geometric tick markers, grid-fit markers, and Phase B/C band shading on `bbox is not None`.
2. `app_auto_axis.py` — gate the X/Y label bands panel and the Detect-axis-frame button on `ocr_available()`. Remove the `_callback_apply_calibration` workaround that salvages `existing_result.bbox` from a previous detection.
3. `app_auto_axis.py` — when the user uploads an image and no detection has run, seed P1/P2/P3 at (10%, 90%), (90%, 90%), (10%, 10%) of the decoded image's dimensions.
4. Add a unit test that confirms `manual_calibration(...) + render_overlay(...)` produces an image with the three anchor markers visible without any bbox.
5. ✅ **Done** — `tests/test_real_image_regression.py::test_every_image_axis_frame_detection` has been deleted; the rationale (`tests/test_real_image_regression.py:302-310`) is preserved as a removed-with-explanation comment so future readers don't try to re-add it. Auto-mode frame detection is already covered by the verified-success regression cases.

## 19. Manual P1/P2 With Differing Pixel-Y Is Silently Averaged  ✅ Done

`compute_calibration` now records `p1p2_y_disagreement_px`; the Streamlit app surfaces a warning when the delta exceeds tolerance. A 3-anchor least-squares variant is still on the table for the Shiny UI (snap-on-drag would be the simpler alternative).

### Problem

`compute_calibration` computes `x_axis_pixel_y = (p1_px_y + p2_px_y) / 2.0` and uses that as the x-axis baseline for y-scale fitting. If the user drags P1 and P2 to different pixel-y positions (intentionally or accidentally), only the midpoint is used; the discrepancy is silently absorbed. Because the manual-only workflow encourages free placement of anchors, this will be a common user mistake on rotated or distorted plots — the calibration will accept inconsistent input and produce a quietly-wrong transform.

### Why Fix Before Migration

The manual calibration path is now a first-class workflow. Anchor semantics must be unambiguous before drag UX is added in Shiny, or users will not be able to debug their calibrations.

### Outline to Address

1. After the user applies manual calibration, if `|p1_px_y - p2_px_y| > tolerance` (e.g. 3 px), surface a warning with the actual delta.
2. Either:
   - Enforce that P2.pixel_y mirrors P1.pixel_y in the UI (snap on drag/edit), OR
   - Switch to a 3-anchor least-squares fit so the calibration honours both points independently.
3. Document the chosen convention in the Manual Values panel UI itself.
4. Add a test that verifies the warning fires and the calibration result reflects the chosen behaviour.

# Part 2: Bugs That Could Be Encountered During Migration to Shiny

These are likely migration-specific issues caused by differences between Streamlit and Shiny in state management, reactivity, event handling, file handling, and UI rendering.

## 1. Reactive State Drift Across Files

### Problem

In Streamlit, state is often stored in `st.session_state`. In Shiny, state will be split across reactive values, reactive calculations, observers, and UI inputs.

In batch mode, calibration state for one image could accidentally leak into another image.

### Outline to Address

1. Create a centralized per-file state object.
2. Key all state by canonical file stem or unique file ID.
3. Store independent per-file objects for:
   - Uploaded image.
   - Uploaded CSV.
   - Masking choice.
   - Custom masks.
   - Calibration result.
   - Manual calibration edits.
   - Overlay edits.
   - Review status.
4. Avoid global mutable state except for app-level configuration.
5. Add debug output showing the currently active file ID.
6. Add tests or simulated workflows that switch files repeatedly.

## 2. Uploaded File Temporary Path Issues

### Problem

Shiny and Streamlit handle uploaded files differently. File paths may be temporary, renamed, or unavailable after session changes.

### Outline to Address

1. On upload, immediately read file bytes into app-managed storage.
2. Compute a stable file hash.
3. Store:
   - Original filename.
   - Canonical stem.
   - Extension.
   - MIME type if available.
   - File hash.
   - Decoded image array or CSV DataFrame.
4. Never rely on the temporary upload path after initial read.
5. Use the hash plus canonical stem as a stable file ID.

## 3. Large Batch Memory Pressure

### Problem

Batch upload of many images and CSVs may consume more memory than the single-image Streamlit workflow.

### Outline to Address

1. Store original image bytes and decoded arrays separately.
2. Decode images lazily when selected, if needed.
3. Cache expensive outputs:
   - OCR results.
   - Mask images.
   - Calibration overlays.
   - Plotly figures.
4. Evict derived images for inactive files if memory gets high.
5. Provide a batch size warning if many large images are uploaded.
6. Consider downscaled previews for file explorer thumbnails.

## 4. Batch OCR Runtime Bottlenecks

### Problem

EasyOCR and image processing may be slow when run across many files.

Note: batch calibration requires EasyOCR. If EasyOCR is not installed, `Calibrate all with defaults` is disabled and users must calibrate each image manually. The bottlenecks below apply only when EasyOCR is available.

### Outline to Address

1. Add progress indicators during batch calibration.
2. Run calibration sequentially first for simplicity.
3. Cache OCR results by image hash.
4. Avoid rerunning OCR if only UI state changes.
5. Add a cancel or stop mechanism if feasible.
6. Consider background processing only after the synchronous workflow is stable.
7. Record per-file runtime diagnostics.

## 5. Plotly Drag/Edit Event Handling Differences

### Problem

Streamlit and Shiny differ in how Plotly events are captured and propagated. Point dragging and error-bar dragging may require custom JavaScript or specific Plotly/Shiny event handling.

### Outline to Address

1. Prototype a minimal Shiny Plotly chart with draggable points.
2. Confirm whether native Plotly edit events provide:
   - Point index.
   - New x/y coordinates.
   - Error bar endpoint changes.
3. If native events are insufficient, add custom JavaScript event listeners.
4. Map browser event data back to the per-file editable CSV model.
5. Ensure edits update:
   - Point x/y.
   - Error bar lower/upper.
   - Interval fill.
   - Local data table.
6. Add visual indication that a point has been edited.

## 6. Image Coordinate Mapping Mismatch

### Problem

Interactive dragging requires converting between:

- Image pixels.
- Plotly display coordinates.
- Data coordinates.
- Calibrated overlay coordinates.

Differences in Shiny/Plotly layout sizing may cause coordinate drift.

### Outline to Address

1. Maintain a single calibration transform per file:
   - Pixel to data.
   - Data to pixel.
2. Lock image display aspect ratio.
3. Avoid autoscaling that changes coordinate systems unexpectedly.
4. Store figure dimensions used for event mapping.
5. Test dragging at:
   - Corners.
   - Axis boundaries.
   - Dense regions.
   - Resized browser windows.
6. Add a visual debug mode showing cursor pixel/data coordinates.

## 7. Mutually Exclusive Checkbox Columns

### Problem

The batch masking selection menu uses three checkbox columns where only one can be selected per file.

Checkboxes are not inherently mutually exclusive, so users may select multiple options unless handled carefully.

### Outline to Address

1. Implement the three choices as radio buttons visually styled as columns, if possible.
2. If checkboxes must be used, add observer logic:
   - Selecting one option clears the other two.
   - Each row always has exactly one selected option.
3. Default every file to `Don't mask series`.
4. Validate before proceeding:
   - No row has zero choices.
   - No row has multiple choices.
5. Store masking choice as one enum field:
   - `no_precalibration_mask`
   - `default_mask`
   - `custom_mask`

## 8. Custom Masking Workflow Branching Bugs

### Problem

If any file is marked `Apply custom masking`, the user should be taken to the masking tab. Other files should still get their selected masking behavior automatically.

This branching may be error-prone.

### Outline to Address

1. After masking-choice confirmation, compute:
   - Files needing custom masking.
   - Files using default masking.
   - Files using no pre-calibration mask.
2. Apply default masks immediately for default-mask files.
3. Mark no-mask files as ready for calibration.
4. Route user to masking tab only if at least one file needs custom masking.
5. In the masking tab, show only files requiring custom masking by default.
6. Allow file explorer navigation to all files if needed.
7. Mark custom-mask files as ready after user saves mask settings.

## 9. Manual Review Status Bugs

### Problem

The calibration tab requires manual review for confidence below `0.95`, while also allowing users to jump between files. Review status could be lost or incorrectly assigned.

### Outline to Address

1. Store review status per file:
   - `not_calibrated`
   - `auto_passed`
   - `requires_review`
   - `reviewed`
   - `manually_adjusted`
2. Store review reason separately:
   - Low confidence.
   - Calibration failed.
   - OCR parse warning.
   - Degenerate axis.
   - User manually flagged.
3. Enable the red `Proceed to next image` button only when there is another unreviewed/flagged image.
4. When clicked, navigate to the next file in review queue.
5. Do not mark a file reviewed just because it was viewed.
6. Add explicit `Mark reviewed` behavior, or automatically mark reviewed after user applies/saves calibration.

## 10. Collapsible Panel Interaction Bugs

### Problem

The calibration UI requires multiple collapsible panels with linked behavior:

- X/Y label bands panel.
- Calibration points panel.
- Manual Values panel.
- Detection settings moved to the bottom.
- Frame-detection warnings moved to the bottom.
- Collapsing one panel opens another in some cases.

This can create reactive loops.

### Outline to Address

1. Define panel state as an explicit enum or named booleans.
2. Avoid observers that recursively trigger each other.
3. For the right-side stack:
   - Default: X/Y label bands open.
   - Calibration points below it.
   - Manual Values below that.
4. Implement controlled transitions:
   - If the user opens Calibration points, optionally collapse X/Y bands.
   - If the user collapses X/Y bands, optionally open Calibration points.
5. Do not force panel changes while a user is editing unless necessary.
6. Add UI tests for panel open/close behavior.

## 11. Export Location Limitations

### Problem

Browser-based Shiny apps generally cannot freely choose arbitrary local export locations without browser download behavior. The requested "choosing export location" may depend on browser capabilities.

### Outline to Address

1. Provide a filename input in the app.
2. Provide an export/download button that downloads the updated CSV through the browser.
3. Let the browser handle final save location.
4. If deployed in a controlled desktop environment, consider a local app mode for native file save dialogs.
5. Clearly label the export behavior:
   - User chooses filename in app.
   - Browser chooses or prompts for save location depending on browser settings.
6. Add batch export later:
   - Single corrected CSV.
   - ZIP of corrected CSVs.

# Part 3: Components That Can Be Optimized Using Shiny Over Streamlit

Shiny’s reactive programming model should be used to improve batch workflows, state control, panel behavior, and interactive editing.

## 1. Batch File Matching and Upload Validation

### Streamlit Limitation

The current app is oriented around one image and one CSV at a time.

### Shiny Optimization

Use reactive file registries to track uploaded images, uploaded CSVs, matched pairs, unmatched files, and duplicate stems.

### Outline to Address

1. Build a reactive file registry.
2. Automatically match image/CSV stems.
3. Display a small unmatched-files box:
   - Images without CSV.
   - CSVs without image.
   - Duplicate stems.
4. Disable `Start` until critical matching errors are resolved.
5. Allow single-image mode to proceed with the current single-image behavior.

## 2. Batch Start Menu With Per-File Masking Choices

### Streamlit Limitation

Streamlit reruns can make large per-file selection tables cumbersome.

### Shiny Optimization

Render a stable reactive table/list of files with previews and mutually exclusive masking choices.

### Outline to Address

1. After upload, show a `Start` button.
2. When clicked, show the batch setup menu.
3. For each matched file, display:
   - Small plot preview.
   - Filename.
   - Three mutually exclusive masking options:
     - `Don't mask series`
     - `Use default masking`
     - `Apply custom masking`
4. Default every file to `Don't mask series`.
5. Store choice as one per-file enum.
6. Validate choices before continuing.

## 3. Conditional Routing to Masking Tab

### Streamlit Limitation

Routing across tabs can be cumbersome when workflow depends on per-file state.

### Shiny Optimization

Use reactive navigation/state to move users to the appropriate next workflow stage.

### Outline to Address

1. After masking choices are confirmed:
   - If no files require custom masking, go directly to calibration.
   - If any files require custom masking, go to masking tab.
2. In masking tab:
   - Default file explorer filter to custom-mask files.
   - Allow navigation to all files if needed.
3. Apply default masks automatically to files with `Use default masking`.
4. Apply no pre-calibration mask to files with `Don't mask series`.
5. Track mask readiness per file.

## 4. Batch Calibration Queue

### Streamlit Limitation

Batch detection can be awkward because Streamlit reruns the script and requires careful session-state management.

### Shiny Optimization

Use a reactive batch job state to process all files and update per-file results.

### Outline to Address

1. Add `Calibrate all with defaults` button.
2. On click:
   - Iterate through all ready files.
   - Apply the same defaults as the current app’s `Run detection` behavior.
   - Store detection result per file.
3. Compute review flags:
   - Confidence below `0.95`.
   - Detection failed.
   - Scientific notation parse warning.
   - Degenerate calibration.
   - Too few calibration points.
4. Update the file explorer with status badges:
   - Passed.
   - Needs review.
   - Failed.
   - Edited.
5. Route to first image needing review.

## 5. File Explorer With Preview Thumbnails

### Streamlit Limitation

The current app is not designed around a persistent file navigator.

### Shiny Optimization

Use a reactive file explorer component across tabs.

### Outline to Address

1. Create a shared file explorer component.
2. Use it in:
   - Masking tab.
   - Calibration tab.
   - Overlay tab.
3. Each file row should show:
   - Thumbnail preview.
   - Filename.
   - Calibration status.
   - Review status.
   - Masking status.
4. Make the file list scrollable.
5. Make the file list collapsible.
6. In calibration tab, place it where detection settings currently are.
7. Move detection settings to the bottom of the app.

## 6. Calibration Tab Right-Side Editing Stack

### Streamlit Limitation

Streamlit expanders and data editors are useful, but tightly linked editor panels and image interactions are difficult to coordinate.

### Shiny Optimization

Use reactive collapsible panels and image interaction events to keep the most-used calibration tools beside the image.

### Outline to Address

1. Main image on the left.
2. Right-side stack of collapsible panels:
   - X/Y label bands.
   - Calibration points.
   - Manual Values.
3. X/Y label bands:
   - Collapsible.
   - Open by default.
4. Calibration points:
   - Rename from `OCR tick tables`.
   - Place directly below bands.
   - If one of bands/calibration points collapses, optionally open the other.
5. Manual Values:
   - Third collapsible panel.
   - Show P1/P2/P3 pixel and data values.
   - Allow direct edits to pixel values and data values.
6. Detection settings:
   - Move to bottom.
   - Collapsible.
7. Frame-detection warnings:
   - Move to bottom.
   - Collapsible.

## 7. Draggable Calibration Anchors P1/P2/P3

### Streamlit Limitation

Dragging calibration anchors in Streamlit is difficult without custom components.

### Shiny Optimization

Use Plotly/Shiny events or custom JavaScript to allow direct manipulation of calibration anchors.

### Outline to Address

1. Render P1/P2/P3 on the calibration image.
2. Allow the user to drag anchors with the mouse.
3. Update pixel values in the Manual Values panel after dragging.
4. Allow manual numeric edits to update anchor positions on the image.
5. Allow x/y data values for P1/P2/P3 to be edited.
6. Recompute calibration when the user applies changes.
7. Mark file status as `manually_adjusted`.
8. Preserve original auto-detected values for audit/debug comparison.

## 8. Calibration Points Editor

### Streamlit Limitation

The current OCR tick tables are functional but somewhat separated from the visual calibration workflow.

### Shiny Optimization

Rename and reorganize the OCR tick table concept into a central calibration point editor.

### Outline to Address

1. Rename `OCR tick tables` to `Calibration points`.
2. Show x and y calibration points in a compact editable table.
3. Include columns:
   - Include/exclude.
   - Axis.
   - Raw OCR text.
   - Cleaned text.
   - Parsed value.
   - Pixel position.
   - Pairing status.
   - Source.
   - Confidence.
4. Allow edits to:
   - Include/exclude.
   - Parsed value.
   - Pixel position if needed.
5. Recalibrate on user command.
6. Sync selected table row with visual highlight on image.

## 9. Overlay Tab Interactive Point Editing

### Streamlit Limitation

The current overlay tab is mainly visual verification, not correction.

### Shiny Optimization

Use interactive Plotly events to edit extracted data directly from the overlay.

### Outline to Address

1. Use the same shared file explorer as the calibration tab.
2. User selects a file.
3. Display the calibrated overlay as before.
4. Allow user to click and drag a point.
5. When point moves:
   - Update local x/y data values.
   - Keep error interval height unchanged.
   - Move the upper/lower interval fill with the point.
   - Update low-opacity fill.
6. Mark the point as edited.
7. Store changes in the per-file editable CSV model.

## 10. Overlay Error-Bar Editing

### Streamlit Limitation

Error bars are currently displayed but not interactively adjustable.

### Shiny Optimization

Allow direct manipulation of upper and lower error-bar handles.

### Outline to Address

1. Render draggable handles at:
   - Upper error-bar endpoint.
   - Lower error-bar endpoint.
2. If user drags upper error bar:
   - Update `y_err_upper`.
   - Leave x/y point coordinates unchanged.
   - Leave lower error unchanged.
3. If user drags lower error bar:
   - Update `y_err_lower`.
   - Leave x/y point coordinates unchanged.
   - Leave upper error unchanged.
4. Update interval fill immediately.
5. Validate that error bars remain nonnegative where appropriate.
6. Mark row as edited.

## 11. Corrected CSV Export

### Streamlit Limitation

The current app does not export interactively corrected CSV data.

### Shiny Optimization

Use a reactive corrected data model and download handler.

### Outline to Address

1. Add an `Export updated CSV` button.
2. Allow user to choose or type export filename.
3. Export the currently selected file’s corrected CSV.
4. Include edited values:
   - x
   - y
   - y_err_lower
   - y_err_upper
   - series
   - series_color if present
5. Optionally add audit columns:
   - original_x
   - original_y
   - original_y_err_lower
   - original_y_err_upper
   - edited
   - edit_type
6. Browser handles save location unless using a local desktop deployment.
7. Later extension:
   - Export all corrected CSVs as a ZIP.

## 12. Manual Calibration Without OCR (No-PyTorch Mode)

### Streamlit Limitation

The current app always attempts to run EasyOCR. There is no UI path for a user who has not installed pytorch.

### Shiny Optimization

Support a fully usable workflow when EasyOCR is not installed, using only P1/P2/P3 dragging and data-value entry.

### Outline to Address

1. At app startup, call `ocr_available()` from `axis_pipeline` and store the result in app-level state.
2. When EasyOCR is unavailable:
   - Show a non-blocking informational banner: _"EasyOCR is not installed. Auto-calibration is disabled. You can still calibrate manually using P1, P2, and P3."_
   - Grey out and disable:
     - `Calibrate all with defaults`.
     - `Run detection` / `Run calibration` (the auto-calibration buttons).
     - X/Y label bands panel (no OCR band controls needed).
     - Calibration points panel (no OCR tick table to display).
   - Keep fully functional:
     - Image display.
     - Draggable P1/P2/P3 anchors.
     - Manual Values panel (open by default in this mode).
     - Data value entry fields for P1/P2/P3.
     - `Apply manual calibration` button.
     - Optional `Detect axis frame` button (geometry-only, no OCR).
     - Overlay display after manual calibration is applied.
     - Corrected CSV export.
3. P1/P2/P3 default pixel positions when no auto-calibration has run:
   - P1: lower-left corner of the image at 10%/90% of width/height.
   - P2: lower-right corner at 90%/90%.
   - P3: upper-left corner at 10%/10%.
   - These are starting points; the user drags them to the correct tick positions.
4. If the user clicks `Detect axis frame` (geometry only):
   - Run `detect_axis_frame` without OCR.
   - Move P1/P2/P3 to the inferred axis corners.
   - Do not run calibration; wait for user to enter data values.
5. `Apply manual calibration` constructs a `CalibrationResult` from the three anchor points and their data values using the manual calibration function added in Part 1, Bug 10.
6. After applying manual calibration, the overlay tab becomes available exactly as in auto mode.
7. When EasyOCR is available:
   - All panels are shown normally.
   - Manual Values panel is collapsible and closed by default (auto-calibration is preferred).
   - `Detect axis frame` button is still available as a fallback.
   - If auto-calibration fails, automatically open the Manual Values panel and show the failure reason.

# Part 4: Proposed Shiny Batch Workflow

## 1. Upload Stage

### Single-Image Case

If only one image is uploaded, the workflow branches on whether EasyOCR is installed.

**Auto-calibration path (EasyOCR installed):**

1. Upload image.
2. Upload corresponding CSV if available.
3. Run calibration (full pipeline).
4. Review calibration result.
5. Optionally edit using calibration points or Manual Values panel.
6. Review overlay.
7. Optionally edit overlay points/error bars.
8. Export corrected CSV.

**Manual calibration path (EasyOCR not installed, or auto-calibration failed):**

1. Upload image.
2. Upload corresponding CSV if available.
3. Optionally click `Detect axis frame` (geometry only, no OCR).
4. Drag P1, P2, P3 to the correct tick positions on the image.
5. Enter data values for P1, P2, P3 in the Manual Values panel.
6. Click `Apply manual calibration`.
7. Review overlay.
8. Optionally edit overlay points/error bars.
9. Export corrected CSV.

### Multi-Image Case

If multiple images are uploaded:

1. User uploads plot images.
2. User uploads CSV files.
3. App matches files by stem.
4. App shows unmatched files in a small box.
5. App enables `Start` only when matching is acceptable.

## 2. File Matching Rules

### Matching Rule

Every CSV should have the same stem as an uploaded plot image.

Examples:

- `plot_a.png` matches `plot_a.csv`
- `plot_b.jpg` matches `plot_b.csv`
- `figure-1.tiff` matches `figure-1.csv`

### Unmatched Files Box

Display a compact unmatched-files box with:

1. Images without CSV.
2. CSVs without image.
3. Duplicate image stems.
4. Duplicate CSV stems.

### Recommended Behavior

- If image has no CSV, allow calibration but warn that overlay cannot be generated until data are supplied.
- If CSV has no image, block processing for that CSV.
- If duplicate stems exist, require user resolution.

## 3. Start Menu

After the user clicks `Start`, show a menu listing all matched files.

Each row includes:

1. Small plot preview.
2. Filename.
3. Three mutually exclusive checkbox-style choices:
   - `Don't mask series`
   - `Use default masking`
   - `Apply custom masking`

### Default

The default for every file is:

- `Don't mask series`

### Choice Definitions

#### Don't mask series

The app does not apply a pre-calibration mask.

The user can still mask series later in the overlay/masking UI.

#### Use default masking

The app applies the default Delta E mask to each series in the image before calibration.

#### Apply custom masking

The user is taken to the masking tab to apply masks manually before calibration.

### Routing

If any file uses `Apply custom masking`:

1. Go to the masking tab.
2. User applies custom masks as needed.
3. Other files are processed according to their selected setting.

If no file uses `Apply custom masking`:

1. Skip masking tab.
2. Go directly to calibration.

## 4. Masking Tab

### Purpose

The masking tab is used only when at least one file requires custom masking.

### File Explorer

Use the shared file explorer.

Default filter:

- Show files marked `Apply custom masking`.

Allow the user to switch to all files if needed.

### Mask Application

For each file:

1. User adjusts masks.
2. User saves mask settings.
3. File is marked `mask_ready`.

Files with `Use default masking` are automatically assigned default Delta E masks.

Files with `Don't mask series` are marked ready without a pre-calibration mask.

## 5. Calibration Tab

### Main Layout

The calibration tab should have:

1. Main image/calibration overlay area.
2. Right-side collapsible editing stack.
3. Collapsible file explorer.
4. Bottom collapsible detection settings.
5. Bottom collapsible frame-detection warnings.

### File Explorer

The file explorer is located where detection settings currently are.

It should include:

- Small image previews.
- File names.
- Calibration status.
- Manual review flag.
- Scrollable list.
- Collapsible container.

The user can jump directly to any file.

### Calibrate All With Defaults

Add a button:

- `Calibrate all with defaults`

This button requires EasyOCR. When EasyOCR is not installed, the button is disabled and a tooltip explains why.

When clicked (EasyOCR available):

1. Run calibration for every ready file.
2. Use the same defaults as the current app’s `Run detection` action.
3. Do not apply user modifications unless already saved as per-file settings.
4. Store result per file.
5. Flag every image with confidence below `0.95` for manual review.
6. Also flag calibration failures or warnings requiring review.

When EasyOCR is not installed, all images begin in `requires_review` status and the user must manually calibrate each one using the P1/P2/P3 anchor workflow.

### Proceed to Next Image Button

Add a red button in the upper right:

- `Proceed to next image`

Behavior:

1. Button is active only if there are images the user has not reviewed.
2. Clicking moves to the next unreviewed or flagged image.
3. If no unreviewed images remain, button is disabled.
4. The button should not skip failed images silently.

## 6. Calibration Tab Right-Side Stack

The most-used editing controls should sit on the right side of the image.

### Panel 1: X/Y Label Bands

- Collapsible.
- Open by default.
- Contains x/y label band controls.
- Used to adjust OCR label detection windows.

### Panel 2: Calibration Points

Rename the old `OCR tick tables` tab to:

- `Calibration points`

Place it directly below the bands section.

It should contain:

- Editable x calibration points.
- Editable y calibration points.
- Include/exclude controls.
- OCR raw/cleaned values.
- Parsed data values.
- Pixel positions.
- Pairing status.
- Recalibration action.

### Linked Collapse Behavior

When one of the following is collapsed, the other may open:

- X/Y Label Bands.
- Calibration Points.

Recommended behavior:

1. X/Y Label Bands open by default.
2. If user opens Calibration Points, keep Bands open unless screen space is limited.
3. If user collapses Bands, open Calibration Points.
4. Avoid automatic changes while the user is editing a field.

### Panel 3: Manual Values

Add a third collapsible panel:

- `Manual Values`

When opened, show:

- P1 pixel x/y.
- P2 pixel x/y.
- P3 pixel x/y.
- P1 data x/y.
- P2 data x/y.
- P3 data x/y.

User can:

1. Edit pixel values directly.
2. Edit data values directly.
3. Drag P1/P2/P3 with the mouse on the image.
4. Apply recalibration from these manual values.

### Bottom Panels

Move these to the bottom of the calibration app:

1. Detection settings.
2. Frame-detection warnings.

Both should be collapsible.

## 7. Overlay Tab

### File Explorer

The overlay tab uses the same shared file explorer.

The user selects the file they want to inspect.

### Existing Behavior to Preserve

All current overlay behavior should remain the same, including:

- Displaying the calibrated image.
- Displaying extracted points.
- Displaying series colors.
- Displaying error bars.
- Displaying interval/fill behavior.
- Existing masking controls where relevant.

### New Interactive Editing Behavior

The user can click and drag a point.

When the point moves:

1. The local x value updates.
2. The local y value updates.
3. The interval follows the point.
4. The interval remains the same height.
5. The low-opacity fill follows and updates.
6. Error bars move with the point.
7. The corresponding row in the editable CSV model updates.

### Error-Bar Editing

The user can drag upper/lower error bars.

When upper error bar is dragged:

1. Update upper error value.
2. Keep point x/y unchanged.
3. Keep lower error unchanged.
4. Update interval fill.

When lower error bar is dragged:

1. Update lower error value.
2. Keep point x/y unchanged.
3. Keep upper error unchanged.
4. Update interval fill.

### Export Updated CSV

Add a button:

- `Export updated CSV`

The user can:

1. Choose or type the file name.
2. Export the corrected CSV for the current file.
3. Use browser download behavior for save location.
4. Later, optionally export all corrected CSVs as a ZIP.

# Part 5: Recommended Shiny State Model

## App-Level State

Track:

- Uploaded images.
- Uploaded CSVs.
- Matched file pairs.
- Unmatched files.
- Current workflow stage.
- Current selected file.
- Batch processing status.

## Per-File State

Each file should have a state object containing:

1. File metadata:
   - Original image filename.
   - Original CSV filename.
   - Canonical stem.
   - File hash.
2. Image data:
   - Original image bytes.
   - Decoded image array.
   - Thumbnail.
3. CSV data:
   - Original CSV DataFrame.
   - Editable CSV DataFrame.
4. Masking:
   - Masking choice.
   - Default mask result.
   - Custom mask settings.
   - Mask readiness.
5. Calibration:
   - Detection result.
   - Calibration result.
   - Confidence.
   - Review status.
   - Review reasons.
   - Manual anchor values.
   - Calibration point table.
6. Overlay edits:
   - Edited point IDs.
   - Edited x/y values.
   - Edited error bars.
   - Edit history if needed.
7. Export:
   - Suggested output filename.
   - Export status.

# Part 6: Suggested Implementation Milestones

## Milestone 1: Stabilize Current Algorithm  ✅ Done

Complete first-plan fixes:

1. Inward tick detection.
2. Scientific notation parsing.
3. Right-side text exclusion.
4. Diagnostics.
5. Regression tests.

## Milestone 2: Extract UI-Independent Core Functions  ✅ Done

Refactors A, F, G, C, E, B, D from **Part 8** are complete (executed in that order). H Phases 1–2 (deprecation warnings on legacy entry points) are also complete. Produced:

1. `plotverify_core/` package containing all pure logic — no Streamlit/Shiny imports (Refactor A).
2. `match_files` utility for image/CSV pairing (Refactor F).
3. `EditableOverlay` data model + `OverlayPoint` records (Refactor G).
4. `PerFileState` and `AppState` dataclasses replacing flat `st.session_state` keys (Refactor C).
5. `build_overlay_traces` + `render_plotly_overlay` split (Refactor E).
6. Typed `CalibrationResult` as the sole runtime representation; dict-shaped legacy paths deprecated (Refactor B, H Phase 1–2).
7. `PlotVerifyApp` controller class (Refactor D).

Side-effect goals reached by this milestone:
- File matching, image loading, CSV validation, mask application, calibration execution (auto + manual), `ocr_available()`, calibration-result serialization, overlay data editing, and corrected CSV export all live in `plotverify_core/`.
- `app_auto_axis.py` is < 1,000 lines (down from ~2,000), behaviour-preserving.
- Every controller method has a unit test that runs without Streamlit installed.

Still outstanding from the Part 8 acceptance criteria:
- `AppState` JSON serialization round-trip (currently not exercised — see Outstanding §3).
- Streamlit's adoption of `PlotVerifyApp` is still partial: callbacks call through the core for shared logic, but `st.session_state` keys remain the source of truth in places. Full migration is deferred until Shiny exists, so the Streamlit app keeps shipping.

## Milestone 3: Build Minimal Shiny Single-Image App  ✅ Substantially complete (2026-05-20)

Entry points: `shiny_app/app.py` (server + UI + JS), `shiny_app/figures.py` (Plotly figure builders).
Run with: `shiny run shiny_app/app.py` (or `python app_shiny.py`).

### Implemented

**Upload (sidebar)**
- Image upload (PNG/JPG/TIFF/BMP/WebP) + optional CSV upload.
- OCR availability banner (informational; auto-detection controls hidden when EasyOCR absent).

**Calibrate tab**
- Main calibration image rendered as a Plotly figure.
- Draggable P1/P2/P3 anchor annotations; click annotation to select, then arrow keys for ±1 px (Shift = ±10 px).
- Run detection (auto-calibration via `axis_pipeline`), Detect axis frame, Reset anchors buttons.
- Right-side accordion stack (open by default: X/Y label bands + Manual Values):
  - **X/Y label bands** — band-width sliders, band preview images.
  - **Calibration points** — detected tick tables (x and y).
  - **Manual Values** — P1/P2/P3 pixel-x/y and data-x/y numeric inputs; log-base checkboxes; Apply manual calibration button; P1/P2 y-disagreement warning when delta > 3 px.
- Calibration summary (status, anchor pixel/data coordinates, calibration residual).
- Bottom accordion: Detection settings (min OCR confidence), Frame-detection warnings.

**Overlay tab**
- Calibrated image with all series drawn as scatter + error bars + low-opacity ribbon fills.
- Right-side card: series visibility checkboxes; Export updated CSV (filename input + optional audit columns toggle + download button).
- Edit point panel: click a data point on the plot to select it (highlights center + upper/lower error caps); arrow keys nudge the selected part; step-size input; force-symmetry dropdown (none / upper → lower / both from centre).
- Floating zoom bubble: draggable preview centred on the selected point or error-bar cap; auto-shows on selection, auto-hides on deselect. Drag handle at top; positioned top-right.
- Click plot background to deselect point and hide zoom bubble.

**Data model / state**
- One `PlotVerifyApp` instance per session, holding `AppState` with one `PerFileState`.
- `EditableOverlay` is the source of truth for all point edits; `export_csv()` exports the current state with or without audit columns.
- `anchors_rv` reactive value is the single source of truth for P1/P2/P3; pixel inputs, drag events, and auto-calibration all write through it.

### Outstanding / known issues

- **Horizontal squishing on first Overlay tab visit** — Plotly renders while the tab pane is `display:none`, sets an explicit pixel width on the `gd` element (measuring 0), and subsequent resize calls are no-ops because `getComputedStyle(gd).width` returns the stale inline value. Current JS strategy: `ResizeObserver` on `#overlay_plot` container; when container width goes from 0 → real, clear `gd.style.width`/`height` then call `Plotly.relayout(gd, {autosize: true})`. Not yet confirmed resolved.
- Smoke test (`tests/test_shiny_smoke.py`) covers import and basic render; no end-to-end Shiny interaction tests yet.

## Milestone 4: Add Batch Upload and File Matching

Add:

1. Multi-image upload.
2. Multi-CSV upload.
3. File matching.
4. Unmatched files box.
5. Start button.

## Milestone 5: Add Batch Masking Selection

Add:

1. Batch setup menu.
2. Three mutually exclusive masking choices.
3. Conditional routing to masking tab.
4. Per-file mask readiness.

## Milestone 6: Add Batch Calibration

Add:

1. `Calibrate all with defaults`.
2. Per-file calibration results.
3. Confidence threshold flagging.
4. Review queue.
5. `Proceed to next image`.

## Milestone 7: Add Calibration Editing UX

Add:

1. File explorer with thumbnails.
2. Right-side X/Y label bands panel.
3. Calibration points panel.
4. Manual Values panel.
5. Draggable P1/P2/P3 anchors.
6. Bottom detection settings and warnings.

## Milestone 8: Add Overlay Editing UX

Add:

1. Draggable points.
2. Draggable error bars.
3. Updating interval fill.
4. Editable local CSV model.
5. Export updated CSV.

## Milestone 9: Batch Export and Final Polish

Add:

1. Export current corrected CSV.
2. Optional export all corrected CSVs as ZIP.
3. Audit columns.
4. Final regression testing.
5. Documentation.

# Part 7: Final Acceptance Criteria

## Algorithm Stability

- First-plan detection fixes are complete.
- Inward ticks are detected.
- Scientific notation parses correctly.
- Right-side text is ignored.
- Existing single-image cases still work.

## Batch Upload

- Multiple images and CSVs can be uploaded.
- Matching by file stem works.
- Unmatched files are clearly displayed.
- Duplicate stems are handled.

## Batch Masking

- Every file has one masking choice.
- Choices are mutually exclusive.
- Default is `Don't mask series`.
- Custom-mask files route to masking tab.
- Other files apply selected masking automatically.

## Batch Calibration

- `Calibrate all with defaults` runs calibration across ready files.
- Confidence below `0.95` flags manual review.
- Failed detections are flagged.
- Review queue works.
- Red `Proceed to next image` button routes correctly.

## Calibration Editing

- File explorer is scrollable and collapsible.
- X/Y label bands are collapsible and open by default.
- OCR tick tables are renamed `Calibration points`.
- Calibration points sit directly below bands.
- Manual Values panel shows and edits P1/P2/P3.
- P1/P2/P3 can be dragged on the image.
- Detection settings and warnings are moved to the bottom.

## Overlay Editing

- Overlay tab uses the shared file explorer.
- Existing overlay behavior is preserved.
- Points can be dragged to update x/y values.
- Error bars can be dragged to update uncertainty values.
- Interval fill follows edited points/error bars.
- Corrected CSV can be exported.

## Manual Calibration (No-PyTorch Mode)

- App runs and is usable without EasyOCR or pytorch installed.
- Informational banner is shown when EasyOCR is absent (not an error).
- `Calibrate all with defaults` is disabled and explains why when EasyOCR is absent.
- P1/P2/P3 anchors appear at sensible default positions on upload.
- `Detect axis frame` button is available without EasyOCR and moves anchors to inferred axis corners.
- User can drag P1/P2/P3 to correct tick positions.
- User can enter data values for P1/P2/P3 in the Manual Values panel.
- `Apply manual calibration` produces a valid overlay identical in format to auto-calibration output.
- Manual calibration path is covered by unit tests that run without EasyOCR installed.
- After manual calibration, the overlay tab and CSV export work without restriction.

## Migration Quality

- State does not leak between files.
- Large batches do not cause unnecessary recomputation.
- OCR/calibration results are cached per image hash.
- Exported CSVs reflect all interactive edits.

# Part 8: Codebase Refactoring Strategy

The current Streamlit app (`app_auto_axis.py`, ~2,000 lines, ~175 `st.session_state` references) mixes pure logic with Streamlit-specific state plumbing. Several modules also carry dict-shaped legacy adaptors that exist only to serve the Streamlit app and will become dead weight once Shiny is the primary UI. This section names concrete refactors, the file moves they imply, and the order in which to apply them so the Streamlit app stays working throughout.

The overall principle: **introduce a UI-agnostic core package, migrate Streamlit to consume it incrementally, then build the Shiny UI on top of the same core.** Both UIs end up as thin renderers over the same logic.

## Refactor A — Carve out `plotverify_core/` package

### Motivation

About a dozen functions in `app_auto_axis.py` are pure logic but live alongside Streamlit code. They cannot be reused by Shiny without rewriting. They also have no unit tests because they're impossible to import without spinning up Streamlit.

### Target layout

```
plotverify_core/
    __init__.py            # public API surface
    colors.py              # is_valid_hex, hex_to_hsv_opencv, hex_to_bgr, hex_complement
    masking.py             # _delta_e_mask, apply_color_mask, build_composite, build_calibration_image
    interpolation.py       # interpolate_series (PCHIP overlay generation)
    calibration_math.py    # compute_calibration, px_to_data, data_to_px, _log10_or_none
                           #   (and bridges into axis_pipeline.manual_calibration)
    csv_io.py              # load_csv (returns DataFrame + LoadReport; never calls st.*)
    image_io.py            # decode_image_bytes, downscale_if_large
    series_state.py        # SeriesState dataclass, init_series_states
    matching.py            # match_files (see Refactor F)
    overlay_model.py       # EditableOverlay, OverlayPoint (see Refactor G)
    session.py             # PerFileState, AppState (see Refactor C)
    app.py                 # PlotVerifyApp controller (see Refactor D)
```

### Outline to Address

1. Create the package directory and move the pure functions identified above out of `app_auto_axis.py`. Imports in `app_auto_axis.py` change to `from plotverify_core import ...`.
2. Strip every `st.` call from moved functions. `_load_csv` returns a `LoadReport` dataclass (warnings, error message, n_dropped); the Streamlit wrapper translates the report into `st.warning`/`st.error` calls.
3. Add `tests/` next to the package and add unit tests for each pure function (uses an inline image fixture, no Streamlit import).
4. Confirm `app_auto_axis.py` still runs end-to-end after the moves; no behaviour change in this refactor.

## Refactor B — Typed `CalibrationResult` as the single runtime representation

### Motivation

The Streamlit app currently keeps two parallel copies of every calibration:
- `st.session_state.auto_axis_detection` — legacy dict (consumed by `update_detection_from_tick_tables`, the overlay shims, and `_set_manual_fields_from_detection`).
- `st.session_state.auto_axis_result` — typed `CalibrationResult` (consumed by `render_overlay`).

Keeping them in sync requires manual bookkeeping after every edit. Bug #12 (just fixed) was a direct symptom; future edits will hit the same trap.

### Outline to Address

1. Make `CalibrationResult` the only in-memory representation in the Streamlit app and in `PlotVerifyApp`. The legacy dict is no longer stored in state.
2. Replace `update_detection_from_tick_tables(detection_dict, x_df, y_df) -> dict` with `update_result_from_tick_edits(result: CalibrationResult, x_edits, y_edits) -> CalibrationResult`. Keep the dict-shaped function as a one-line shim that round-trips through the typed form (for any external scripts still on the old API).
3. Replace `_set_manual_fields_from_detection(dict)` with `anchors_from_result(result: CalibrationResult) -> Anchors` (a small dataclass) and a `populate_manual_widgets(anchors)` helper that does the session-state writes in one place.
4. Treat `CalibrationResult.to_legacy_dict()` strictly as a serialization helper for disk caches, never a runtime interchange format.
5. After the Shiny app is the primary UI, delete `auto_detect_axes_and_ticks`, `auto_detect_axes_ticks_ocr`, `build_diagnostic_overlay`, `build_ocr_debug_overlay`, and `update_detection_from_tick_tables` from `axis_pipeline/legacy.py`. Keep only `rebuild_result_from_detection` for any saved-state file restores.

## Refactor C — `PerFileState` dataclass

### Motivation

The Streamlit app spreads each file's state across ~15 flat `st.session_state` keys (`image_bgr`, `image_hash`, `df`, `calibration`, `auto_axis_result`, `series_states`, `frame_preview_cache`, etc.). For multi-file batch mode, this is unworkable — each file would need its own copy of every key, with bespoke namespacing. The bug-14 fix (sweeping `vis_*` keys on CSV change) is a workaround that points at the missing abstraction.

### Target shape

```python
@dataclass
class PerFileState:
    file_id: str                            # canonical_stem + image_hash[:8]
    image_filename: str
    image_bytes: bytes                       # original; kept for hash + re-decode
    image_bgr: np.ndarray                    # decoded (downscaled if needed)
    image_downscale_factor: float = 1.0
    csv_filename: Optional[str] = None
    csv_df: Optional[pd.DataFrame] = None
    csv_load_report: Optional[LoadReport] = None
    overlay: Optional[EditableOverlay] = None
    series_states: Dict[str, SeriesState] = field(default_factory=dict)
    series_color_overrides: Dict[str, str] = field(default_factory=dict)
    masking_choice: str = "no_precalibration_mask"  # enum value
    mask_ready: bool = False
    cal_masked_img_bgr: Optional[np.ndarray] = None
    frame_preview: Optional[FramePreview] = None
    frame_preview_key: Optional[tuple] = None        # invalidation key
    detection_result: Optional[CalibrationResult] = None
    manual_anchors: Optional[Anchors] = None
    review_status: str = "not_calibrated"
    review_reasons: List[str] = field(default_factory=list)
    export_filename: Optional[str] = None

@dataclass
class AppState:
    files: Dict[str, PerFileState] = field(default_factory=dict)
    active_file_id: Optional[str] = None
    matching: Optional[MatchResult] = None    # see Refactor F
    workflow_stage: str = "upload"           # upload/masking/calibration/overlay
```

### Outline to Address

1. Define `PerFileState` and `AppState` in `plotverify_core/session.py`.
2. The Streamlit app stores a single `AppState` object under one session-state key. Direct access (`st.session_state.image_bgr`) is replaced by helper accessors (`active_file_state().image_bgr`).
3. Per-file caches that today key on `image_hash` (e.g. `cached_delta_e_mask`) keep the same key; the cache layer is unaffected.
4. `_init_series_states`, `_load_image_from_upload`, `_load_csv` write into a `PerFileState` instead of into flat session-state keys.
5. Add a migration shim so the existing widget keys (`vis_{name}`, `p1_px_x`, `p1_data_y`, etc.) continue to work for the duration of the refactor — they are still owned by Streamlit widgets, but their values are mirrored into the active `PerFileState` on every change.
6. After the Shiny app exists, the migration shim is deleted and per-widget keys are removed.

## Refactor D — Pure callbacks + `PlotVerifyApp` controller

### Motivation

The current `_callback_apply_calibration` (and friends) read ~10 session-state keys, call pure functions, and write ~6 keys. There is no single function describing "apply manual calibration"; it's a transaction smeared across UI plumbing. The same applies to `_callback_copy_detected_values`, `_init_series_states`, `_update_calibration_masked_image`.

### Target shape

```python
class PlotVerifyApp:
    """UI-agnostic controller for the multi-file PlotVerify workflow."""

    def __init__(self, ocr_runner: Optional[OCRRunner] = None) -> None: ...

    # File ingest
    def add_image(self, filename: str, image_bytes: bytes) -> str: ...
    def add_csv(self, filename: str, csv_text: str) -> str: ...
    def remove_file(self, file_id: str) -> None: ...
    def match_files(self) -> MatchResult: ...

    # Selection
    def select(self, file_id: str) -> None: ...
    @property
    def active(self) -> Optional[PerFileState]: ...

    # Calibration
    def run_auto_calibration(self, file_id: str, *, config: CalibrationConfig) -> CalibrationResult: ...
    def apply_manual_calibration(self, file_id: str, anchors: Anchors,
                                  x_log_base=None, y_log_base=None) -> CalibrationResult: ...
    def update_tick_edits(self, file_id: str, x_edits, y_edits) -> CalibrationResult: ...
    def calibrate_all_with_defaults(self) -> Dict[str, CalibrationResult]: ...

    # Masking
    def set_masking_choice(self, file_id: str, choice: MaskingChoice) -> None: ...

    # Review
    def mark_reviewed(self, file_id: str) -> None: ...
    def next_unreviewed(self) -> Optional[str]: ...

    # Export
    def export_csv(self, file_id: str) -> bytes: ...
    def export_all_zip(self) -> bytes: ...
```

### Outline to Address

1. Define `PlotVerifyApp` in `plotverify_core/app.py`. No Streamlit/Shiny imports.
2. Each method is a pure transformation on `AppState` plus the injected `ocr_runner`.
3. The Streamlit app instantiates one `PlotVerifyApp` and keeps it in session state. Every callback in `app_auto_axis.py` becomes a one-line call into the controller.
4. The Shiny app instantiates `PlotVerifyApp` per-session and wraps method calls in `reactive` blocks. Reactivity is at the UI layer; the controller is plain Python.
5. Add unit tests for each controller method using a fixture image + fixture CSV (no UI required).

## Refactor E — Decouple Plotly trace generation from rendering

### Motivation

`build_overlay_figure` builds Plotly traces inside a loop that also reads `st.session_state.get(f"vis_{series_name}", True)`. That coupling means the function cannot be reused as-is in Shiny (where visibility lives in a reactive value rather than a Streamlit widget key) and cannot be unit-tested without a session-state mock.

### Target shape

```python
@dataclass
class OverlayTrace:
    series: str
    x: np.ndarray
    y: np.ndarray
    y_err_lower: np.ndarray
    y_err_upper: np.ndarray
    color_hex: str
    visible: bool

def build_overlay_traces(
    df: pd.DataFrame,
    series_visibility: Dict[str, bool],
    series_colors: Dict[str, str],
) -> List[OverlayTrace]: ...

def render_plotly_overlay(
    traces: List[OverlayTrace],
    img_rgb: np.ndarray,
    cal: dict,
) -> go.Figure: ...
```

### Outline to Address

1. Split `build_overlay_figure` into the two functions above. Place both in `plotverify_core/overlay_plot.py` (Plotly is still used by Shiny via `shinywidgets`, so no need to abstract it further).
2. `series_visibility` and `series_colors` are passed in explicitly. No `st.session_state` reads inside the trace builder.
3. Streamlit caller builds the visibility/colors dicts from `PerFileState`; Shiny caller builds them from reactive values.
4. Add a unit test that builds traces from a fixture DataFrame and asserts trace count, error-bar shapes, and ribbon ordering.

## Refactor F — File-matching utility

### Motivation

Part 1 §8 of this plan calls for image/CSV stem matching with case-insensitive rules and duplicate-stem reporting, but the implementation does not exist yet. Without it, the batch-upload UI has nothing to bind to.

### Target shape

```python
@dataclass
class FileEntry:
    filename: str
    canonical_stem: str
    extension: str
    payload: bytes

@dataclass
class MatchResult:
    pairs: Dict[str, Tuple[FileEntry, FileEntry]]   # stem -> (image, csv)
    images_without_csv: List[FileEntry]
    csvs_without_image: List[FileEntry]
    duplicate_image_stems: Dict[str, List[FileEntry]]
    duplicate_csv_stems: Dict[str, List[FileEntry]]

def match_files(images: List[FileEntry], csvs: List[FileEntry]) -> MatchResult: ...
```

### Outline to Address

1. Implement `match_files` in `plotverify_core/matching.py`. Case-insensitive stem comparison; preserves original filenames in the returned entries.
2. Unit-test every category (matched pair, image-only, csv-only, dup image, dup csv, mixed-case match, extension permutations).
3. Used by `PlotVerifyApp.match_files()` and by both the Streamlit batch entry point (when introduced) and the Shiny upload tab.

## Refactor G — Editable overlay data model

### Motivation

Part 1 §9 of this plan defines the editable overlay schema but does not provide a Python representation. Without a typed model the Streamlit overlay tab cannot accept edits, and the Shiny drag interactions have nothing to write to.

### Target shape

```python
@dataclass
class OverlayPoint:
    series: str
    point_id: str                # stable across edits
    x: float
    y: float
    y_err_lower: Optional[float]
    y_err_upper: Optional[float]
    color_hex: str
    original_x: float
    original_y: float
    original_y_err_lower: Optional[float]
    original_y_err_upper: Optional[float]
    edited: bool
    edit_timestamp: Optional[str]
    edit_type: Optional[str]     # "point" | "err_upper" | "err_lower"

class EditableOverlay:
    def __init__(self, df: pd.DataFrame): ...
    def points(self) -> Iterable[OverlayPoint]: ...
    def edit_point(self, point_id: str, new_x: float, new_y: float) -> None: ...
    def edit_err_upper(self, point_id: str, value: float) -> None: ...
    def edit_err_lower(self, point_id: str, value: float) -> None: ...
    def to_dataframe(self, *, include_audit_cols: bool = False) -> pd.DataFrame: ...
    def reset_point(self, point_id: str) -> None: ...
```

### Outline to Address

1. Implement in `plotverify_core/overlay_model.py`.
2. `point_id` is `f"{series}#{row_index}"` from the source CSV; stable across edits but not across CSV reloads.
3. `to_dataframe(include_audit_cols=True)` produces the export CSV with `original_*` and `edited`/`edit_type` columns per Part 3 §11.
4. Streamlit `_load_csv` constructs an `EditableOverlay` and stores it in `PerFileState.overlay`. Drag handlers (Shiny) and tick-row edits (both UIs) call the methods above.
5. Unit tests for round-trip: load → edit one point → export → re-load → confirm original columns are preserved.

## Refactor H — Cleanup of legacy shims

### Motivation

`axis_auto.py` and `ocr_axis.py` are tiny re-export shims that exist only to keep the old Streamlit imports working. `axis_pipeline/legacy.py` is several hundred lines of dict-shaped adapters serving the same audience. Once the Streamlit app moves to the typed API and the Shiny app exists, all of this can be deleted or radically reduced.

### Outline to Address

1. **Phase 1 (concurrent with Refactor B)**: deprecate `axis_pipeline/legacy.auto_detect_axes_and_ticks` and `axis_pipeline/legacy.auto_detect_axes_ticks_ocr`. The Streamlit app calls `run_calibration` and `manual_calibration` directly.
2. **Phase 2 (after Refactor B is done in Streamlit)**: deprecate `update_detection_from_tick_tables` and `build_ocr_debug_overlay`/`build_diagnostic_overlay`. Both UIs use the typed equivalents.
3. **Phase 3 (after Shiny is primary)**: delete `axis_auto.py` and `ocr_axis.py`. Reduce `axis_pipeline/legacy.py` to just `to_legacy_dict` / `from_legacy_dict` helpers used for disk caches.
4. Add a `DeprecationWarning` at each shim's import path so external scripts find out before the deletion.
5. Search the repo (and any scripts users have shared) for references before deleting.

## Refactor Phasing

The refactors are interdependent. The recommended order is:

| Order | Refactor | Gates | Status |
|-------|----------|-------|--------|
| 1 | A (carve out `plotverify_core/`) | None — additive only | ✅ Done |
| 2 | F (file matching) | None — pure new module | ✅ Done |
| 3 | G (editable overlay model) | A | ✅ Done |
| 4 | C (`PerFileState` / `AppState`) | A, F, G | ✅ Done |
| 5 | E (Plotly trace decoupling) | A, C | ✅ Done |
| 6 | B (typed `CalibrationResult` everywhere) | A, C (`PerFileState` carries the typed result) | ✅ Done |
| 7 | D (`PlotVerifyApp` controller) | A–C, F | ✅ Done |
| 8 | H Phase 1–2 (deprecate legacy shims) | B, D | ✅ Done |
| — | *Shiny milestones 3–8 happen here* | — | ✅ M3 substantially complete; M4–M9 not started |
| 9 | H Phase 3 (delete legacy shims) | Shiny is the primary UI | ⏳ Not started |

Refactors 1–7 keep the existing Streamlit app working at every step. Each can be merged independently and shipped to users. Refactor 9 is the only step that removes capabilities; gate it on the Shiny app passing the Part 7 acceptance criteria.

## Refactor Acceptance Criteria

- ✅ `plotverify_core/` package has zero `import streamlit` and zero `import shiny` (verified by `tests/test_core_no_streamlit.py`, subprocess-isolated).
- ✅ Every controller method on `PlotVerifyApp` has at least one unit test that runs without Streamlit or Shiny installed (`tests/test_core_app.py`).
- ✅ All public APIs that take or return dict-shaped detections are marked deprecated and emit a `DeprecationWarning` (`tests/test_legacy_deprecation.py`).
- ⏳ `app_auto_axis.py` is < 1,000 lines — partially met. Streamlit code now delegates to `plotverify_core` for the bulk of computation, but full controller adoption is deferred until the Shiny app exists so the Streamlit app keeps shipping. Verify the line count after the next pass of Streamlit cleanup.
- ✅ `AppState` JSON serialization — `plotverify_core.serialization` round-trips a session through a `.pvsession` zip; `tests/test_serialization.py` covers happy path, EditableOverlay edits, version mismatch, and malformed archives.
- ⏳ Shiny main module < 1,500 lines, reads/writes `AppState` exclusively through `PlotVerifyApp` — gated on Milestone 3.

# Part 9: Pre-Migration Cleanup Audit

## Purpose

Catalog every piece of code identified as removable *without changing application behavior*, so a single cleanup pass can be reviewed and committed before the Shiny migration proper begins. Each item below was located by reading the codebase as it stood on **2026-05-19** with `rg`/AST scans; the `Evidence` line records the verification step. If a later change re-introduces a use of one of these symbols, the audit is wrong for that item — re-verify before deletion.

**Scope rule:** an item belongs here only if all three are true:
1. It has zero non-import references in the repo, OR is a test file whose coverage is provided elsewhere.
2. Removing it does not change any user-visible Streamlit behavior.
3. Removing it does not change the pass/fail outcome of the surviving pytest suite.

Items that *look* removable but are actually used are listed under **§9.6 Do Not Remove** so a future audit doesn't relitigate them.

## 9.1 Unused imports (mechanical)

Pyflakes-style; all are zero-reference outside their own `from ... import` line. Verification command (used for each): `rg -n '\b<name>\b' <file> | rg -v '^[0-9]+:(from |import )'` returns no rows.

| File | Line | Symbol | Evidence |
|---|---|---|---|
| `plotverify_core/app.py` | 14 | `Any` (from `typing`) | 0 non-import refs |
| `plotverify_core/app.py` | 14 | `replace` (from `dataclasses`) | 0 non-import refs |
| `plotverify_core/app.py` | 18 | `np` | 0 non-import refs |
| `plotverify_core/app.py` | 22 | `detect_axis_frame` | 0 non-import refs (used in `app_auto_axis.py`, not here) |
| `plotverify_core/app.py` | 28 | `rebuild_result_from_detection` | 0 non-import refs (used in `serialization.py`, not here) |
| `plotverify_core/app.py` | 33 | `hash_bytes` | 0 non-import refs |
| `plotverify_core/app.py` | 43 | `WorkflowStage` | 0 non-import refs |
| `axis_pipeline/legacy.py` | 24 | `List` (from `typing`) | 0 non-import refs |
| `axis_pipeline/legacy.py` | 26 | `np` | 0 non-import refs |
| `axis_pipeline/overlay.py` | 14 | `Dict` (from `typing`) | 0 non-import refs |
| `plotverify_core/serialization.py` | 35 | `List` (from `typing`) | 0 non-import refs |
| `plotverify_core/overlay_model.py` | 13 | `field` (from `dataclasses`) | 0 calls to `field(` in file |
| `plotverify_core/series_state.py` | 9 | `field` (from `dataclasses`) | 0 calls to `field(` in file |

Estimated removal: ~13 lines of imports.

## 9.2 Dead test files

These are pytest modules that either fail to collect today, or whose every assertion is duplicated by a surviving `tests/test_core_*` file that loads the same logic from `plotverify_core` instead of through the `app_auto_axis` re-export shim.

### 9.2.1 Already broken at collection

- **`tests/test_colors_pre_refactor.py`** — Imports `hex_complement` from `app_auto_axis`, which no longer re-exports it. `python -m pytest --collect-only` errors on this file (verified). Coverage is provided by `tests/test_core_no_streamlit.py` via `plotverify_core.colors`. **Safe to delete.**
- **`test_new_pipeline.py`** (repo root) — Hardcodes `sys.path.insert(0, '/home/claude/plotverify')`, which only existed in the original sandbox. Pytest auto-discovers it and errors on collection. CLAUDE.md documents it as a manual pipeline runner, not a unit test. **Options:** (a) delete; (b) move to a `scripts/` directory and remove from pytest discovery; (c) rename without `test_` prefix.

### 9.2.2 Pre-refactor checkpoints — outcome of the diff pass

These were originally written as "pin current behavior before extracting" snapshots during Refactor A. The cleanup-PR diff against the surviving `test_core_*` files showed that **only one of the four was genuinely duplicated**; the other three carried unique behavioral coverage that the audit had assumed lived elsewhere.

**Audit correction:** `tests/test_core_no_streamlit.py` is a 40-line subprocess smoke test that only asserts `callable(compute_calibration)` / `callable(delta_e_mask)` / etc. It does **not** verify any values. The original §9.2.2 claim that it superseded `test_calibration_math_pre_refactor.py` and `test_masking_pre_refactor.py` was wrong.

Outcome by file:

- **`tests/test_csv_io_pre_refactor.py` → deleted.** Every assertion is mirrored by `tests/test_core_csv_io.py` (which already imports from `plotverify_core`).
- **`tests/test_calibration_math_pre_refactor.py` → renamed to `tests/test_core_calibration_math.py`** and imports repointed from `app_auto_axis` to `plotverify_core`. Preserves 17 value-correctness tests: `log10_or_none` semantics, `compute_calibration` linear/log/degenerate cases, `p1p2_y_disagreement_px`, `px_to_data`/`data_to_px` round-trips, `P1P2_Y_TOLERANCE_PX == 3.0`.
- **`tests/test_masking_pre_refactor.py` → renamed to `tests/test_core_masking.py`** and imports repointed to `plotverify_core` (`delta_e_mask`, `apply_color_mask`). Preserves 4 tests: matching-color all-set, dissimilar-zero, hue-wrap, in-range.
- **`tests/test_axis_pipeline_pre_refactor.py` → renamed to `tests/test_axis_pipeline_typed_api.py`** (imports were already from `axis_pipeline`, no repointing needed). Preserves 13 tests: 7 `parse_numeric_tick` parsing cases (unicode minus, superscript/caret log10, e-notation, …), 5 `manual_calibration` edge cases (`log10_x_rejects_zero`, `to_legacy_dict_shape`), and `ocr_available` type guard. `test_typed_tick_edits.py` uses `manual_calibration` only as a fixture builder; it does not exercise `parse_numeric_tick` at all.

Lesson for future audits: an "X is superseded by Y" claim must be verified by diffing assertion bodies, not just by matching imported symbol names. Smoke-test files (`test_core_no_streamlit.py`) prove a function exists, not that it behaves correctly.

Net deletion from §9.2.2: 1 test file (~65 lines). Three files renamed (no behavior delta).

## 9.3 Unreferenced helpers and unused dataclass fields

- **`plotverify_core/matching.py:103-105`** — `find_pair(result, stem) -> Optional[Tuple[FileEntry, FileEntry]]`. Evidence: `rg -n 'find_pair'` yields one hit (the `def`). Not in `plotverify_core/__init__.py.__all__`. Was scoped for "look up a pair by case-insensitive stem" but no caller materialized; `MatchResult.pairs` is consulted directly. **Safe to delete.**
- **`plotverify_core/session.py:98-99`** — `frame_preview_key: Optional[tuple] = None` and `frame_preview: Any = None` on `PerFileState`. Evidence: `rg -n '\.frame_preview(_key)?\b' plotverify_core app_auto_axis.py tests` returns 0 hits on the dataclass fields. The Streamlit app uses its own global `st.session_state.frame_preview_cache` (separate from the dataclass) — see `app_auto_axis.py:452`, `:934`. `serialization.py:119` already documents these fields as "transient" and excludes them from save. **Safe to delete from the dataclass.** When the Shiny app needs a per-file frame preview cache, it can re-add the fields with a typed shape.

## 9.4 Stale parameters with no read inside the body

- **`axis_pipeline/legacy.py:89`** — `mask_all_text: bool = True` parameter on `auto_detect_axes_ticks_ocr`. Comment: `# kept for signature compatibility`. Body never reads it. Evidence: `rg -n 'mask_all_text' axis_pipeline/legacy.py` shows the parameter declaration and nothing else; only other hits are an unrelated `st.session_state.ocr_mask_all_text` toggle in `app_auto_axis.py:1222` which never flows into this call. **Caveat:** this is a public-ish legacy API. External scripts may pass it as a keyword. Two options:
  - **(a)** Remove now and bump `PIPELINE_VERSION` to `0.2`. Lower friction, breaks any external caller using the kwarg.
  - **(b)** Keep until Refactor H Phase 3 (legacy-shim deletion), since the whole function is going away then anyway.
  - Recommendation: **(b)** — defer. The cost of keeping a one-line dead parameter is zero, and we don't want a `PIPELINE_VERSION` bump on a cosmetic change.

## 9.5 Cleanup PR — actual outcome

Executed 2026-05-19. Pytest before: 150 collected + 2 collection errors. Pytest after: 143 collected, **137 passed, 6 xfailed**, no new failures, same xfail set.

- **Unused imports** — 13 lines across 7 files.
- **Dead helpers / fields** — `find_pair` (~3 lines), `PerFileState.frame_preview_key` + `.frame_preview` (~2 lines); `serialization.py:119` docstring also tightened.
- **Broken-at-collection tests** — `test_colors_pre_refactor.py` deleted; `test_new_pipeline.py` relocated to a `scripts/` runner with a portable `sys.path.insert(Path(__file__).resolve().parent.parent)`; `CLAUDE.md` updated. (The relocated runner has since been removed.)
- **Pre-refactor tests** — `test_csv_io_pre_refactor.py` deleted (~65 lines); the other three renamed + import-repointed (see §9.2.2 above).
- **Deferred until Refactor H Phase 3:** `mask_all_text` parameter; `axis_auto.py`; `ocr_axis.py`; the bulk of `axis_pipeline/legacy.py`.

## 9.6 Do Not Remove

Symbols/files initially suspected but verified as still used. Listed so a future audit doesn't have to re-verify.

- **`axis_auto.py`, `ocr_axis.py`** — Imported by `app_auto_axis.py:35-46`. Removal is Refactor H Phase 3 work and is gated on the Shiny app being the primary UI.
- **`axis_pipeline/legacy.py`** as a whole — `rebuild_result_from_detection` is called by `plotverify_core/serialization.py:220`; `update_result_from_tick_edits` is called by `plotverify_core/app.py:213` and `app_auto_axis.py`; `update_detection_from_tick_tables` is called by `app_auto_axis.py:1078`. Deletion candidates inside this file are limited to the lines listed in §9.1 and §9.4.
- **`frame_preview_run_count` Streamlit state** (`app_auto_axis.py:453, 946-947, 1334`) — surfaced in the "Preview diagnostics" expander. Diagnostic UI, not dead code.
- **Symbols in `plotverify_core/__init__.py` `__all__`** that the AST scan flagged — they are intentional public re-exports; the AST tool doesn't model `__all__`. Do not remove based on AST output for `__init__.py` files.
- **`bbox is None` guard branches in `app_auto_axis.py`** (`:1277`, `:1342-1347`, `:1462`) — these are Bug #18 (manual mode) tolerance, not workarounds for a fixed bug. They must stay so manual calibration without `detect_axis_frame` keeps rendering.
- **`csv_io.py:87`** `df["series_color"] = df["series_color"].astype(object)` — this is a fix for the pandas all-NaN→float64 inference, exercised by `tests/test_serialization.py` round-trips. Required.

## 9.7 Recommended execution order for the cleanup PR

1. Delete the imports listed in §9.1 (one commit per file, or one commit total — they're independent).
2. Delete `find_pair` (§9.3) and the two `PerFileState` fields (§9.3).
3. Decide on `test_new_pipeline.py` (§9.2.1) — recommended: move under `scripts/` as a manual pipeline runner and update `CLAUDE.md` accordingly so the manual-run instruction still works.
4. Delete `tests/test_colors_pre_refactor.py` (§9.2.1).
5. Diff each `*_pre_refactor.py` test against its `test_core_*` replacement (§9.2.2); delete only after diff confirms no unique assertion is lost.
6. Run the full suite: `python -m pytest`. Expected delta: same pass count as before deletion, minus the deleted tests. No xfailed count change.
7. Run the Streamlit app and smoke-test: upload → run detection → manual override → apply calibration → export CSV. No visible difference.

After this PR lands, revisit §9.4 only once Refactor H Phase 3 begins.

