# Plan: Improve Auto Axis Detection for Inward Ticks and X-Axis Scientific Notation

## 1. Current Architecture Summary

The app currently routes axis detection through compatibility shims. `axis_auto.py` delegates geometry-only detection to `axis_pipeline.legacy`, and `ocr_axis.py` delegates OCR-assisted detection plus numeric parsing to the same package.

The main detection flow is organized around:

1. OCR discovery and band OCR in `ocr.py`
2. Axis frame detection and geometric tick detection in `geometry.py`
3. Linear grid fitting in `gridfit.py`
4. OCR-label-to-tick pairing in `pairing.py`
5. Calibration fitting in `calibration.py`
6. Pipeline orchestration in `pipeline.py`
7. Legacy dictionary conversion in `legacy.py`
8. Streamlit display/edit controls in `app_auto_axis.py`

The described failure is consistent with the current geometry logic:

- `detect_x_tick_positions()` searches below the bottom x-axis line.
- `detect_y_tick_positions()` searches left of the y-axis line.
- Therefore, ticks drawn inside the plotting region can be missed.

The current OCR parser already attempts to support scientific notation using characters such as `e`, `E`, `^`, `×`, and `x`, and it includes superscript normalization support. However, the parser should be strengthened for common OCR variants of scientific notation.

The axis text on the right side of the image is out of scope and should be ignored.

## 2. Goals

### Primary Goals

1. Detect ticks drawn inside the plotting region.
2. Correctly parse x-axis tick labels written in scientific notation.
3. Preserve existing behavior for ordinary outward ticks and standard decimal/integer tick labels.
4. Avoid using right-side axis text as calibration evidence.
5. Improve diagnostics so future failures are easier to inspect.

### Non-Goals

1. Do not implement full right-side y-axis support.
2. Do not add dual-axis calibration.
3. Do not use axis title or right-side annotation text as tick labels.
4. Do not change the user-facing manual calibration workflow unless needed for diagnostics.
5. Do not replace EasyOCR; keep the current EasyOCR-based pipeline.

## 3. Root Causes

### 3.1 Inward Ticks Are Missed

Current x-axis tick detection searches only outside the plot, below the bottom x-axis. This works for outward ticks below the axis but misses inward ticks above the axis line.

Current y-axis tick detection searches only outside the plot, left of the y-axis. This works for outward ticks left of the axis but misses inward ticks right of the y-axis line.

### 3.2 Grid Fit Has Too Few Candidate Ticks

`gridfit.py` expects a list of candidate tick positions and then rejects outliers that do not fit the modal spacing of a linear axis.

If inward tick candidates are never detected, the grid fit cannot recover them.

### 3.3 Pairing Depends on Geometric Ticks

`pairing.py` pairs OCR numeric labels to grid-fitted geometric tick positions.

If geometric ticks are missing, labels may not pair even if OCR reads the label correctly.

### 3.4 Scientific Notation Parsing Is Too Fragile

`ocr.py` already contains normalization logic for superscripts and Unicode dashes, but the parser should explicitly handle multiple OCR variants of scientific tick labels, including:

- `10^-3`
- `10−3`
- `10⁻³`
- `1×10^3`
- `1 x 10 -3`
- `1.0E−03`
- split or partially merged OCR text

## 4. Proposed File-Level Changes

## 4.1 `geometry.py`

### Add Bidirectional Tick Detection

Update `detect_x_tick_positions()` and `detect_y_tick_positions()` so they search both outward and inward tick bands.

For x-axis ticks:

- Existing outward band: from `bbox.bottom` downward.
- New inward band: from slightly above `bbox.bottom` into the plot region.
- Combine candidates from both bands.
- Deduplicate candidates by x-position.
- Preserve candidate source metadata internally where possible.

For y-axis ticks:

- Existing outward band: from left of `bbox.left` to the y-axis.
- New inward band: from `bbox.left` rightward into the plot region.
- Combine candidates from both bands.
- Deduplicate candidates by y-position.

### Add Tick Direction Diagnostics

Add diagnostics that report:

- Number of outward x tick candidates.
- Number of inward x tick candidates.
- Number of outward y tick candidates.
- Number of inward y tick candidates.
- Which side contributed the final fitted grid.

### Recommended Search Windows

For x-axis:

- Outward band: existing behavior.
- Inward band: `bbox.bottom - inward_tick_depth_px` to `bbox.bottom + 1`.
- Default inward depth: max of `10 px` and approximately `2.5%` of image height or plot height.
- Clamp to `bbox.top` and image bounds.

For y-axis:

- Outward band: existing behavior.
- Inward band: `bbox.left` to `bbox.left + inward_tick_depth_px`.
- Default inward depth: max of `10 px` and approximately `2.5%` of image width or plot width.
- Clamp to `bbox.right` and image bounds.

### Improve Peak Detection Robustness

Use separate projection profiles for inward and outward bands, then combine.

Avoid pooling the full cross-axis band before peak detection, because axis lines and gridlines can dominate the signal.

Recommended process:

1. Extract inward band.
2. Extract outward band.
3. Remove the axis-line row or column contribution if it dominates.
4. Compute projection along the tick direction.
5. Use local percentile subtraction as currently done.
6. Run `find_peaks()`.
7. Deduplicate peaks within a small pixel tolerance.

### Avoid False Positives from Gridlines

Because inward bands may include gridlines, add filtering:

- Require candidate tick ink to be concentrated near the axis line.
- For x ticks, reject vertical structures that extend too far upward into the plot unless they also align with expected short tick length.
- For y ticks, reject horizontal structures that extend too far right into the plot unless they are short.
- Keep full gridlines only if they help identify tick positions, and flag them as `gridline_supported`.

### Add Optional Gridline-Supported Tick Detection

If inward short-tick detection is weak, use gridline positions as a fallback:

- Detect vertical gridlines for x tick positions.
- Detect horizontal gridlines for y tick positions.
- Only accept gridline-derived ticks when they align with OCR labels and pass grid fitting.
- Mark them separately in diagnostics.

This is useful for plots where tick marks are minimal but gridlines are present.

## 4.2 `types.py`

### Extend `CalibrationConfig`

Add configuration fields for inward tick detection:

- `detect_inward_ticks: bool = True`
- `inward_tick_depth_frac: float = 0.025`
- `inward_tick_min_depth_px: int = 10`
- `tick_dedup_tolerance_px: float = 4.0`
- `allow_gridline_supported_ticks: bool = True`
- `gridline_tick_max_pair_distance_frac: float = 0.65`

`CalibrationConfig` already centralizes OCR, grid fit, pairing, and calibration tunables, so adding tick-detection tunables there keeps the design consistent.

### Optionally Extend Diagnostics Schema

No new dataclass is strictly required, but `CalibrationResult.diagnostics` should include new fields such as:

- `x_tick_candidates_outward`
- `x_tick_candidates_inward`
- `x_tick_candidates_gridline`
- `y_tick_candidates_outward`
- `y_tick_candidates_inward`
- `y_tick_candidates_gridline`
- `x_tick_detection_mode`
- `y_tick_detection_mode`

`CalibrationResult` already exposes a flexible diagnostics dictionary, so this can be done without breaking the legacy API.

## 4.3 `pipeline.py`

### Pass Configuration Into Geometry Tick Detection

Update the pipeline so tick detection uses the new `CalibrationConfig` inward-tick parameters.

### Recommended Pipeline Order

For each axis:

1. Detect outward tick candidates.
2. Detect inward tick candidates.
3. Merge and deduplicate.
4. Fit a linear grid.
5. If grid fit fails or has too few ticks:
   - Try outward-only grid.
   - Try inward-only grid.
   - Try gridline-supported positions.
   - Try OCR-label-center fallback.
6. Pair OCR labels to the best candidate grid.
7. Calibrate.

### Add OCR-Label-Center Fallback

If geometric tick detection fails but OCR labels are available:

- For x-axis, use the x-center of each x tick label as a provisional tick position.
- For y-axis, use the y-center of each y tick label as a provisional tick position.
- Fit a grid to these label-center-derived positions.
- Pair labels to these inferred positions with a different status, such as `paired_to_label_center_fallback`.

This fallback should only be used when:

- At least 3 numeric labels are detected for the axis.
- The labels are monotonic in the expected direction.
- The spacing is approximately regular or calibrates with low residual error.

This is not ideal for publication-quality extraction, but it is preferable to complete failure when tick marks are not visible or are inside the plotting region.

### Preserve Existing Calibration Logic

Do not change `calibration.py` unless scientific notation produces numeric values that reveal a new calibration edge case.

The current linear OLS calibration and outlier-removal strategy should continue to work once correct numeric values and pixel positions are supplied.

## 4.4 `ocr.py`

### Strengthen Scientific Notation Parsing

Update `parse_numeric_tick()` and supporting normalization functions to robustly parse x-axis scientific notation.

Supported forms should include:

- `1e3`
- `1E3`
- `1e-3`
- `1E−03`
- `1.0e+03`
- `10^3`
- `10^-3`
- `10⁻³`
- `10³`
- `1×10^3`
- `1 × 10^3`
- `1x10^3`
- `1 x 10^3`
- `1.0×10−3`
- `1.0 x 10 -3`

The parser should normalize:

- Unicode minus signs to `-`
- Superscript digits to exponent notation
- Superscript minus to exponent minus
- Multiplication signs `×`, `x`, `X`, and possibly `*`
- OCR spacing artifacts
- OCR variants like `l` or `I` only when safely surrounded by numeric/scientific-notation context

### Fix Superscript Expansion Semantics

The current superscript logic should produce exponent notation.

Desired behavior:

- `10³` should become `10^3`
- `10⁻³` should become `10^-3`

This prevents `10³` from being parsed as `103`.

### Add Scientific Parser Stages

Implement parsing in ordered stages:

1. Plain decimal/integer parse.
2. Standard Python scientific notation parse, such as `1e-3`.
3. Power notation parse, such as `10^-3`.
4. Multiplier notation parse, such as `1.5×10^-3`.
5. Compact OCR notation parse, such as `1.5x10-3`, when unambiguous.
6. Return parse failure with a clear flag if no pattern matches.

### Add Parse Status Values

Recommended parse statuses:

- `plain_number`
- `scientific_e_notation`
- `power_of_ten`
- `multiplied_power_of_ten`
- `ocr_corrected_scientific`
- `parse_failed`

Recommended flags:

- `none`
- `unicode_minus_normalized`
- `superscript_normalized`
- `scientific_notation`
- `ocr_ambiguous`
- `right_side_text_ignored`

### Keep Right-Side Text Out of X-Axis Pairing

Do not broaden x-axis label filtering to include right-side text.

The user explicitly stated that the axis text on the right side of the image is not a concern.

## 4.5 `pairing.py`

### Make Pairing Robust to Inward Ticks

The pairing logic should not need major changes if `grid.fitted_positions` contains the correct tick positions.

It already pairs x labels using label x-centers and y labels using label y-centers.

However, update statuses to distinguish different sources:

- `paired_to_tick_mark_outward`
- `paired_to_tick_mark_inward`
- `paired_to_gridline_supported_tick`
- `paired_to_label_center_fallback`

### Prevent Right-Side Text From Pairing to X Ticks

Ensure `filter_x_axis_labels()` only keeps labels in the x-axis label band below the x-axis and horizontally within or very near the plot frame.

For this image-specific requirement:

- Text on the right side of the plot should not be considered an x-axis tick label.
- Right-side text should not be included in `x_tick_table`.
- Right-side text should not influence calibration.

### Handle Scientific Values During Monotonicity Checks

The current monotonicity enforcement should work after parsing values correctly.

Add a diagnostic warning if x values fail monotonicity after scientific parsing, because that often indicates OCR parsed the exponent incorrectly.

## 4.6 `gridfit.py`

### Preserve Existing Grid Fit But Improve Inputs

`gridfit.py` currently fits a regular 1-D grid and rejects non-grid outliers.

This should be retained.

### Optional Enhancement

If merged inward/outward candidates contain duplicates or near-duplicates, grid fit may be sensitive to duplicated peaks.

Add pre-grid deduplication either in `geometry.py` or immediately before calling `fit_linear_grid()`.

Recommended behavior:

1. Sort positions.
2. Merge positions within `tick_dedup_tolerance_px`.
3. Use the median position of each cluster.
4. Preserve source labels in diagnostics, not necessarily in the grid input.

## 4.7 `overlay.py`

### Show Inward Tick Diagnostics

Update the diagnostic overlay to visually distinguish:

- Outward tick candidates
- Inward tick candidates
- Gridline-supported candidates
- Rejected candidates
- Final grid-fitted candidates

`overlay.py` already renders detected frame, geometric ticks, grid-fit kept/rejected positions, OCR boxes, and paired anchors, so this is a natural extension.

### Recommended Overlay Conventions

Use different visual markers:

- Outward tick candidate: small blue mark
- Inward tick candidate: small purple mark
- Gridline-supported candidate: cyan mark
- Kept grid tick: green mark
- Rejected candidate: gray mark
- Paired OCR label: connector line to tick
- Calibration anchors: keep existing anchor display

## 4.8 `legacy.py`

### Preserve Existing Public API

The app depends on legacy dictionary keys such as:

- `x_ticks`
- `y_ticks`
- `x_tick_table`
- `y_tick_table`
- `x_grid_fit`
- `y_grid_fit`
- `x_calibration`
- `y_calibration`
- `warnings`
- `diagnostics`

Keep the existing structure, but add the new diagnostic fields inside `diagnostics`.

### Add Source Status to Tick Tables

Ensure the legacy tick-table rows carry the new pairing status values from `PairedTick.status`.

No breaking schema change is needed because the table already includes `status`.

## 4.9 `app_auto_axis.py`

### Minimal UI Changes

The current app already displays OCR tick tables and diagnostic overlays.

Add optional diagnostics to the calibration tab:

- Count of inward/outward x tick candidates.
- Count of inward/outward y tick candidates.
- Warning if scientific notation labels were parsed using fallback corrections.
- Warning if x-axis calibration used label-center fallback instead of true geometric ticks.

### Do Not Add Right-Side Axis Controls

No UI should be added for right-side axis text, because it is out of scope for this fix.

## 5. Detailed Implementation Sequence

## Phase 1: Add Tests Before Changing Logic

Create a test image fixture or synthetic minimal image that mimics the failing case:

- Bottom x-axis.
- Left y-axis.
- X ticks drawn inward.
- Scientific notation x tick labels.
- Optional irrelevant text on the right side.
- Light background.
- Dark axis lines.

Add tests for:

1. `parse_numeric_tick("10^-3")`
2. `parse_numeric_tick("10⁻³")`
3. `parse_numeric_tick("1×10^3")`
4. `parse_numeric_tick("1.0 x 10 -3")`
5. `parse_numeric_tick("1E−03")`
6. Inward x tick detection returns the expected tick x-positions.
7. Right-side text is not included in x tick labels.
8. Full `run_calibration()` succeeds on the failing image.

## Phase 2: Improve Scientific Notation Parser

Modify `ocr.py` first.

Acceptance criteria:

- All plain numeric labels continue to parse.
- Scientific notation labels parse to correct floats.
- Superscript exponents parse as exponents, not appended digits.
- Parser returns meaningful `parse_status` and `flag`.

## Phase 3: Add Inward Tick Detection

Modify `geometry.py`.

Acceptance criteria:

- Existing outward tick plots still work.
- Inward tick plots return tick candidates.
- Mixed inward/outward ticks do not duplicate positions.
- Candidate counts are available for diagnostics.

## Phase 4: Update Pipeline Candidate Selection

Modify `pipeline.py`.

Acceptance criteria:

- The pipeline tries merged inward/outward candidates first.
- If merged detection fails, it can try inward-only or outward-only candidates.
- If geometric detection still fails, OCR-label-center fallback is available.
- Diagnostics explain which path was used.

## Phase 5: Update Pairing Statuses

Modify `pairing.py` and possibly `types.py`.

Acceptance criteria:

- Tick table status identifies whether the tick came from inward detection, outward detection, gridline support, or label-center fallback.
- Monotonicity filtering still rejects bad OCR parses.
- Right-side text is ignored for x-axis pairing.

## Phase 6: Update Overlay and UI Diagnostics

Modify `overlay.py` and `app_auto_axis.py`.

Acceptance criteria:

- Diagnostic overlay shows inward tick candidates distinctly.
- User can see why a calibration succeeded or failed.
- OCR tick tables remain editable.
- Existing manual override behavior is unchanged.

## Phase 7: Regression Test Existing Cases

Run the app or tests on known prior examples:

1. Boxed plot with outward ticks.
2. Open left/bottom axes with outward ticks.
3. Plot with gridlines.
4. Plot without gridlines.
5. Plot with zero baseline on x-axis.
6. Plot with scientific notation on x-axis.
7. Plot with irrelevant right-side text.

## 6. Specific Scientific Notation Parsing Rules

### Plain Numeric

- `0` should parse as `0`
- `1` should parse as `1`
- `-1` should parse as `-1`
- `1.25` should parse as `1.25`
- `.25` should parse as `0.25`
- `1,000` should parse as `1000`

### E Notation

- `1e3` should parse as `1000`
- `1E3` should parse as `1000`
- `1e-3` should parse as `0.001`
- `1E−03` should parse as `0.001`

### Power-of-Ten Notation

- `10^3` should parse as `1000`
- `10^-3` should parse as `0.001`
- `10³` should parse as `1000`
- `10⁻³` should parse as `0.001`

### Multiplied Power Notation

- `1×10^3` should parse as `1000`
- `1.0×10^-3` should parse as `0.001`
- `1 x 10^3` should parse as `1000`
- `1.5x10-3` should parse as `0.0015`

### OCR Corrections to Allow Carefully

Only apply OCR corrections inside scientific-notation-like strings:

- `1O^3` may be corrected to `10^3` if clearly intended as power notation.
- `l×10^3` may be corrected to `1×10^3` if the leading character is isolated and followed by `×10`.
- `I×10^-3` may be corrected to `1×10^-3` if the leading character is isolated and followed by `×10`.

Avoid broad OCR corrections that could turn arbitrary axis text into numeric tick labels.

## 7. Tick Detection Algorithm Details

## 7.1 X-Axis Inward Tick Detection

Use this conceptual process:

1. Identify the x-axis line at `bbox.bottom`.
2. Define outward band below the axis.
3. Define inward band above the axis.
4. For each band:
   - Extract dark pixels.
   - Remove the axis line itself if it dominates.
   - Project dark pixels by column.
   - Detect peaks.
   - Convert local peak positions to image x-coordinates.
5. Merge outward and inward peaks.
6. Deduplicate by x-coordinate.
7. Fit a linear grid.
8. Pair with OCR x labels.

## 7.2 Y-Axis Inward Tick Detection

Use this conceptual process:

1. Identify the y-axis line at `bbox.left`.
2. Define outward band left of the axis.
3. Define inward band right of the axis.
4. For each band:
   - Extract dark pixels.
   - Remove the axis line itself if it dominates.
   - Project dark pixels by row.
   - Detect peaks.
   - Convert local peak positions to image y-coordinates.
5. Merge outward and inward peaks.
6. Deduplicate by y-coordinate.
7. Fit a linear grid.
8. Pair with OCR y labels.

## 8. Right-Side Text Handling

The right-side axis text in the provided image is not a concern and should not be used.

Implementation guardrails:

1. Do not expand x-label OCR search to the right-side region.
2. Keep x-axis OCR labels constrained to the x-label band below the x-axis.
3. Keep y-axis OCR labels constrained to the left-side y-label band.
4. If right-side text appears in Phase A full-image OCR, allow it to be masked for geometry detection, but do not pair it as a tick label.
5. Add a diagnostic count of numeric OCR records ignored because they were outside the axis label bands.

## 9. Acceptance Criteria

## Detection Criteria

- Inward x ticks are detected on the failing image.
- X-axis tick candidates are sufficient to fit a regular grid.
- X-axis scientific notation labels parse to correct numeric values.
- X-axis labels pair to the correct tick positions.
- Calibration anchors are placed at the correct axis positions.
- Right-side text does not appear in the x tick table.
- Right-side text does not influence x calibration.

## Regression Criteria

- Existing ordinary decimal tick labels still parse.
- Existing outward tick plots still calibrate.
- Existing OCR tick table editing still works.
- Existing manual override workflow still works.
- Existing overlay plotting still works.

## Diagnostic Criteria

The detection result should report:

- Whether inward tick detection was used.
- How many inward tick candidates were found.
- Whether scientific notation parsing was used.
- Whether label-center fallback was used.
- Which OCR records were ignored as outside the axis label bands.

## 10. Recommended Test Matrix

### Case 1: Existing Standard Plot

- Tick direction: outward
- X labels: integers
- Gridlines: no
- Expected result: pass

### Case 2: Existing Gridline Plot

- Tick direction: outward
- X labels: integers
- Gridlines: yes
- Expected result: pass

### Case 3: Failing Image Case

- Tick direction: inward
- X labels: scientific notation
- Gridlines: unknown
- Expected result: pass

### Case 4: Inward Ticks Without Gridlines

- Tick direction: inward
- X labels: decimals
- Gridlines: no
- Expected result: pass

### Case 5: Inward Ticks With Gridlines

- Tick direction: inward
- X labels: scientific notation
- Gridlines: yes
- Expected result: pass

### Case 6: Mixed Inward/Outward Ticks

- Tick direction: mixed
- X labels: integers
- Gridlines: no
- Expected result: pass

### Case 7: Right-Side Numeric Text

- Tick direction: inward or outward
- X labels: scientific notation
- Gridlines: any
- Expected result: right-side text ignored

### Case 8: OCR Split Scientific Notation

- Tick direction: inward
- X labels: `1 × 10^-3` or similar OCR-split variant
- Gridlines: any
- Expected result: parse correctly or warn clearly

## 11. Risk Areas

## 11.1 False Tick Detection From Gridlines

Inward tick detection may confuse gridlines with ticks.

Mitigation:

- Prefer short dark structures near the axis line.
- Use gridline-derived ticks only as fallback.
- Require OCR pairing and monotonicity consistency.

## 11.2 Scientific Notation OCR Ambiguity

EasyOCR may read `10^-3` as `10-3`, `1O^-3`, or split the label.

Mitigation:

- Add conservative parser corrections.
- Use monotonicity checks in `pairing.py`.
- Surface ambiguous parses in the tick table for manual correction.

## 11.3 Overfitting to the Provided Image

Avoid hard-coding positions from the provided image.

The implementation should rely on:

- Axis frame geometry.
- Relative tick-band windows.
- OCR label bands.
- Grid regularity.
- Monotonicity.

## 11.4 Breaking Existing Plots

Because many prior plots may have outward ticks, all new inward logic should be additive and configurable through `CalibrationConfig`.

## 12. Final File Change Checklist

## Required Changes

### `geometry.py`

- Add inward tick detection.
- Add candidate merging and deduplication.
- Add optional gridline-supported tick candidates.

### `ocr.py`

- Strengthen scientific notation normalization and parsing.
- Fix superscript expansion semantics.
- Add parse statuses and flags.

### `pipeline.py`

- Use inward/outward tick candidates.
- Add fallback candidate selection.
- Add diagnostics.

### `types.py`

- Add configuration fields.
- Preserve legacy-compatible result structure.

### `pairing.py`

- Ensure right-side text is ignored.
- Add clearer pairing statuses.
- Preserve monotonicity filtering.

## Recommended Changes

### `overlay.py`

- Show inward/outward/gridline-supported tick candidates distinctly.

### `app_auto_axis.py`

- Display new diagnostics in the calibration tab.

## Likely No Direct Changes Needed

### `axis_auto.py`

- Should remain a compatibility shim.

### `ocr_axis.py`

- Should remain a compatibility shim.

### `calibration.py`

- Should not need changes if parsed numeric values and tick positions are correct.

### `gridfit.py`

- Should not need major changes, aside from optional deduplication if not handled earlier.

### `legacy.py`

- Only minor updates if new diagnostics or statuses need to be surfaced in the legacy dictionary.

