# Second Plan: Post-Detection-Fix Migration From Streamlit to Shiny With Batch Processing

## Execution Order

This plan should be executed only after completing the first plan, which addresses:

1. Inward tick detection.
2. Scientific notation parsing.
3. Right-side text exclusion.
4. Improved calibration diagnostics.
5. Better OCR/tick pairing reliability.

The goal of this second plan is to migrate the app from Streamlit to Shiny while shifting the primary user workflow from single-image review to batch image/CSV processing.

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

# Part 4: Proposed Shiny Batch Workflow

## 1. Upload Stage

### Single-Image Case

If only one image is uploaded, the workflow remains approximately the same as the current app:

1. Upload image.
2. Upload corresponding CSV if available.
3. Calibrate.
4. Review overlay.
5. Optionally edit overlay points/error bars.
6. Export corrected CSV.

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

When clicked:

1. Run calibration for every ready file.
2. Use the same defaults as the current app’s `Run detection` action.
3. Do not apply user modifications unless already saved as per-file settings.
4. Store result per file.
5. Flag every image with confidence below `0.95` for manual review.
6. Also flag calibration failures or warnings requiring review.

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

## Milestone 1: Stabilize Current Algorithm

Complete first-plan fixes:

1. Inward tick detection.
2. Scientific notation parsing.
3. Right-side text exclusion.
4. Diagnostics.
5. Regression tests.

## Milestone 2: Extract UI-Independent Core Functions

Before writing Shiny UI, isolate reusable functions:

1. File matching.
2. Image loading.
3. CSV validation.
4. Mask application.
5. Calibration execution.
6. Calibration result serialization.
7. Overlay data editing.
8. Corrected CSV export.

## Milestone 3: Build Minimal Shiny Single-Image App

Recreate current single-image workflow first:

1. Upload one image.
2. Upload one CSV.
3. Run calibration.
4. Show calibration points.
5. Show overlay.
6. Export corrected CSV.

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

## Migration Quality

- State does not leak between files.
- Large batches do not cause unnecessary recomputation.
- OCR/calibration results are cached per image hash.
- Exported CSVs reflect all interactive edits.

