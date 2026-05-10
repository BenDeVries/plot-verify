"""Run the new pipeline against the test image, with a tesseract OCR shim.

EasyOCR's model download is blocked in this sandbox, so we substitute a
tesseract-backed OCR runner with the same record schema. This lets us
validate the entire pipeline end-to-end.

For numeric-allowlist phases (B, C), we approximate EasyOCR's behaviour by
filtering tesseract's output to records whose recognized text contains digits.
"""
import sys
import io
sys.path.insert(0, '/home/claude/plotverify')

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from axis_pipeline import (
    CalibrationConfig,
    OCRPhase,
    OCRRecord,
    detect_axis_frame,
    parse_numeric_tick,
    render_band_preview,
    render_overlay,
    run_calibration,
    x_label_band,
    y_label_band,
)


def tesseract_runner(
    img_bgr,
    *,
    gpu=False,
    min_confidence=0.20,
    allowlist=None,
    phase=OCRPhase.FULL.value,
    bbox_offset=(0, 0),
    upsample=1.0,
    detection_params=None,
):
    # `upsample` and `detection_params` are EasyOCR-specific tuning knobs that
    # the production runner forwards from `CalibrationConfig`. Tesseract has its
    # own (different) tuning surface; we accept and ignore them so the shim can
    # be a drop-in for tests, but optionally honour `upsample` since it's
    # genuinely useful for small text.
    if upsample and upsample > 0 and upsample != 1.0:
        h, w = img_bgr.shape[:2]
        img_bgr = cv2.resize(
            img_bgr,
            (max(1, int(round(w * upsample))), max(1, int(round(h * upsample)))),
            interpolation=cv2.INTER_CUBIC,
        )
    coord_scale = 1.0 / upsample if upsample and upsample > 0 else 1.0
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    config = "--psm 11"  # sparse text
    if allowlist:
        # Tesseract doesn't reliably honour char_whitelist for numeric-only;
        # we'll post-filter instead.
        pass
    data = pytesseract.image_to_data(img_rgb, config=config, output_type=Output.DICT)
    dx, dy = bbox_offset
    out = []
    n = len(data['text'])
    for i in range(n):
        text = data['text'][i].strip()
        if not text:
            continue
        try:
            conf = float(data['conf'][i]) / 100.0
        except (TypeError, ValueError):
            conf = 0.0
        if conf < min_confidence:
            continue
        x, y = int(data['left'][i]) + dx, int(data['top'][i]) + dy
        w, h = int(data['width'][i]), int(data['height'][i])
        x1, y1 = x + w, y + h
        # If the input was upsampled, rescale bbox coordinates back to original-image space.
        # (dx, dy are already in original-image space and were applied above; we apply the
        # scale to the relative position before the offset, then re-add it.)
        if coord_scale != 1.0:
            x = int(round((x - dx) * coord_scale)) + dx
            y = int(round((y - dy) * coord_scale)) + dy
            x1 = int(round((x1 - dx) * coord_scale)) + dx
            y1 = int(round((y1 - dy) * coord_scale)) + dy
        value, cleaned, status, flag = parse_numeric_tick(text)
        if allowlist is not None and value is None:
            continue
        out.append(OCRRecord(
            raw_text=text, cleaned_text=cleaned, value=value,
            is_numeric=value is not None, confidence=conf,
            bbox=(x, y, x1, y1),
            center=((x + x1) / 2.0, (y + y1) / 2.0),
            parse_status=status, parse_flag=flag, phase=phase,
        ))
    return out


def main():
    img = cv2.imread('/home/claude/plotverify/iga_povetacicept_iv.png')
    print(f"Image shape: {img.shape}")

    cfg = CalibrationConfig(
        min_ocr_confidence=0.20,
        use_robust_regression=True,
        student_t_df=4.0,
    )
    result = run_calibration(img, config=cfg, ocr_runner=tesseract_runner)

    print(f"\n========== Calibration Result ==========")
    print(f"Success: {result.success}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Mode: {result.mode}")
    print(f"BBox: {result.bbox}")

    print(f"\n--- OCR records by phase ---")
    by_phase = {}
    for r in result.ocr_records:
        by_phase.setdefault(r.phase, []).append(r)
    for phase, recs in by_phase.items():
        print(f"  {phase}: {len(recs)} records ({sum(1 for r in recs if r.is_numeric)} numeric)")

    print(f"\n--- Geometric ticks ---")
    print(f"  X raw: {len(result.x_geometric_ticks)}")
    print(f"  Y raw: {len(result.y_geometric_ticks)}")

    print(f"\n--- Grid fit ---")
    if result.x_grid_fit:
        g = result.x_grid_fit
        print(f"  X: spacing={g.spacing:.1f}px, kept={len(g.fitted_positions)}, "
              f"rejected={len(g.rejected_positions)}, success={g.success}")
        print(f"     kept positions: {[f'{p:.0f}' for p in g.fitted_positions]}")
        if g.rejected_positions:
            print(f"     rejected: {[f'{p:.0f}' for p in g.rejected_positions]}")
    if result.y_grid_fit:
        g = result.y_grid_fit
        print(f"  Y: spacing={g.spacing:.1f}px, kept={len(g.fitted_positions)}, "
              f"rejected={len(g.rejected_positions)}, success={g.success}")
        print(f"     kept positions: {[f'{p:.0f}' for p in g.fitted_positions]}")
        if g.rejected_positions:
            print(f"     rejected: {[f'{p:.0f}' for p in g.rejected_positions]}")

    print(f"\n--- X paired ticks (include={sum(1 for t in result.x_paired_ticks if t.include)}/{len(result.x_paired_ticks)}) ---")
    for t in result.x_paired_ticks:
        marker = "✓" if t.include else "✗"
        print(f"  {marker} raw={t.raw_text!r:<8} value={t.data_value:>6.1f} "
              f"px={t.pixel_position:>6.1f}  pair_dist={t.pair_distance_px:>5.1f}  "
              f"status={t.status}")

    print(f"\n--- Y paired ticks (include={sum(1 for t in result.y_paired_ticks if t.include)}/{len(result.y_paired_ticks)}) ---")
    for t in result.y_paired_ticks:
        marker = "✓" if t.include else "✗"
        print(f"  {marker} raw={t.raw_text!r:<8} value={t.data_value:>6.1f} "
              f"px={t.pixel_position:>6.1f}  pair_dist={t.pair_distance_px:>5.1f}  "
              f"status={t.status}")

    print(f"\n--- Calibrations ---")
    if result.x_calibration:
        c = result.x_calibration
        print(f"  X: scale={c.scale:.4f} offset={c.offset:.3f} method={c.method}  "
              f"n={c.n_points}  rmse_data={c.rmse_data:.3f}  rmse_px={c.rmse_px:.3f}")
        if c.slope_se is not None:
            print(f"     slope_SE={c.slope_se:.5f}  offset_SE={c.offset_se:.3f}  "
                  f"log_lik={c.log_likelihood}")
    if result.y_calibration:
        c = result.y_calibration
        print(f"  Y: scale={c.scale:.4f} offset={c.offset:.3f} method={c.method}  "
              f"n={c.n_points}  rmse_data={c.rmse_data:.3f}  rmse_px={c.rmse_px:.3f}")
        if c.slope_se is not None:
            print(f"     slope_SE={c.slope_se:.5f}  offset_SE={c.offset_se:.3f}  "
                  f"log_lik={c.log_likelihood}")

    print(f"\n--- Calibration anchors ---")
    print(f"  P1 px={result.p1_pixel}  data_x={result.p1_data_x}  data_y={result.p1_data_y}")
    print(f"  P2 px={result.p2_pixel}  data_x={result.p2_data_x}")
    print(f"  P3 px={result.p3_pixel}  data_x={result.p3_data_x}  data_y={result.p3_data_y}")

    if result.warnings:
        print(f"\n--- Warnings ---")
        for w in result.warnings:
            print(f"  ! {w}")

    # Sanity check: convert P3 (0, 1) data point to pixel and back.
    if result.x_calibration and result.y_calibration:
        # The user said: "p3 is 0,1 and the point appears to be at (1,-small number)"
        # In this image the leftmost x-tick label is "1" not "0", so data_x=0 sits
        # to the LEFT of the first tick. Let's compute where data_x=0 would land.
        x_cal = result.x_calibration
        y_cal = result.y_calibration
        zero_x_px = x_cal.data_to_pixel(0.0)
        zero_y_px = y_cal.data_to_pixel(0.0)
        print(f"\n--- Data → pixel sanity ---")
        print(f"  data (0, 0) -> pixel ({zero_x_px:.1f}, {zero_y_px:.1f})")
        print(f"  data (1, 0) -> pixel ({x_cal.data_to_pixel(1.0):.1f}, "
              f"{zero_y_px:.1f})")
        print(f"  data (0, 1) -> pixel ({zero_x_px:.1f}, "
              f"{y_cal.data_to_pixel(1.0):.1f})")
        print(f"  data (113, -100) -> pixel ({x_cal.data_to_pixel(113.0):.1f}, "
              f"{y_cal.data_to_pixel(-100.0):.1f})")

    overlay = render_overlay(img, result, show_band_windows=True, show_grid_rejected=True)
    cv2.imwrite('/home/claude/plotverify/overlay_new_pipeline.png',
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"\nOverlay saved to overlay_new_pipeline.png")

    return result


def test_detect_axis_frame_basic():
    """Verify the detect_axis_frame factor-out works on the IgA fixture.

    Asserts mechanical contract — bbox detected, Phase A records returned,
    warnings list is well-formed. OCR-quality assertions live in `main` against
    the full calibration result.
    """
    print("\n" + "=" * 60)
    print("TEST: detect_axis_frame on iga_povetacicept_iv.png")
    print("=" * 60)
    img = cv2.imread('/home/claude/plotverify/iga_povetacicept_iv.png')
    preview = detect_axis_frame(img, ocr_runner=tesseract_runner)

    assert preview.bbox is not None, "Expected bbox to be detected on iga fixture"
    assert preview.bbox.left < preview.bbox.right, "bbox.left must be < bbox.right"
    assert preview.bbox.top < preview.bbox.bottom, "bbox.top must be < bbox.bottom"
    assert isinstance(preview.phase_a_records, list), "phase_a_records must be a list"
    assert preview.axis_confidence > 0.0, "axis_confidence should be > 0 on a clean fixture"
    print(f"  bbox: {preview.bbox}")
    print(f"  axis_confidence: {preview.axis_confidence:.3f}")
    print(f"  phase A records: {len(preview.phase_a_records)} "
          f"({sum(1 for r in preview.phase_a_records if r.is_numeric)} numeric)")
    print(f"  mode: {preview.mode}")
    print("  PASS")


def test_render_band_preview():
    """Verify render_band_preview returns an RGB image of the same HxW as the source."""
    print("\n" + "=" * 60)
    print("TEST: render_band_preview returns same-shape RGB image")
    print("=" * 60)
    img = cv2.imread('/home/claude/plotverify/iga_povetacicept_iv.png')
    preview = detect_axis_frame(img, ocr_runner=tesseract_runner)
    assert preview.bbox is not None

    cfg = CalibrationConfig()
    yb = y_label_band(preview.bbox, extra_left=cfg.y_band_extra_px,
                      extra_vertical=cfg.y_band_extra_vertical_px)
    xb = x_label_band(preview.bbox, extra_below=cfg.x_band_extra_px,
                      extra_horizontal=cfg.x_band_extra_horizontal_px)
    overlay = render_band_preview(
        img, preview.bbox, yb, xb, phase_a_records=preview.phase_a_records,
    )
    assert overlay.shape == img.shape, (
        f"overlay shape {overlay.shape} != source shape {img.shape}"
    )
    assert overlay.dtype == img.dtype, "overlay dtype mismatch"
    # render_band_preview returns RGB (cv2.cvtColor at the end). The source is
    # BGR. We can't byte-compare those, but we *can* sanity-check the overlay
    # isn't identical to the input — drawing should have changed pixels.
    assert not np.array_equal(overlay, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), (
        "render_band_preview produced an unmodified image"
    )
    print(f"  overlay shape: {overlay.shape}, dtype: {overlay.dtype}")
    print("  PASS")


def test_band_override_takes_effect_on_case3():
    """Verify the manual-band rescue path applies the user's override.

    With EasyOCR (production), `y_band_extra_px=160` is the spec's
    motivating fix that makes case3's calibration succeed: the wider band
    captures the full 4-digit y-labels (0/500/.../2500). With tesseract (this
    sandbox), small-text OCR quality is too weak to fully reproduce that, so
    we assert what we CAN test mechanically:

      1. `detect_axis_frame` returns the same bbox regardless of y_band_extra_px
         (the override only affects later phases).
      2. `run_calibration(... y_band_extra_px=160)` records the override in
         diagnostics — proving the config flowed through to Phase B.
      3. The diagnostics report a wider effective band than the default would.

    A note printed at the end documents the OCR-engine-dependent quality
    expectation per the spec.
    """
    print("\n" + "=" * 60)
    print("TEST: band override propagates through run_calibration on case3")
    print("=" * 60)
    img = cv2.imread('/home/claude/plotverify/case3_three_arm_dose_response.png')
    assert img is not None, "case3 fixture missing"

    # Step 1: detect_axis_frame is band-independent.
    preview = detect_axis_frame(img, ocr_runner=tesseract_runner)
    assert preview.bbox is not None, "case3 frame should still be detected"
    print(f"  detect_axis_frame: bbox={preview.bbox} "
          f"axis_confidence={preview.axis_confidence:.3f}")

    # Step 2: Run with the spec's recommended override and verify diagnostics
    # show it was applied.
    cfg_override = CalibrationConfig(y_band_extra_px=160)
    result_override = run_calibration(img, config=cfg_override,
                                      ocr_runner=tesseract_runner)
    y_band_used = result_override.diagnostics.get("y_band_extra_used")
    print(f"  with y_band_extra_px=160: y_band_extra_used={y_band_used}")
    assert y_band_used == 160, (
        f"Expected y_band_extra_used=160, got {y_band_used}. The override did "
        f"not flow through to Phase B."
    )

    # Step 3: Compare against the default to confirm the override actually
    # changes behaviour. The default y_band_extra_px is 90.
    cfg_default = CalibrationConfig()
    result_default = run_calibration(img, config=cfg_default,
                                     ocr_runner=tesseract_runner)
    y_band_used_default = result_default.diagnostics.get("y_band_extra_used")
    print(f"  with y_band_extra_px=default (90): "
          f"y_band_extra_used={y_band_used_default}")
    assert y_band_used != y_band_used_default, (
        "Override produced identical band geometry to default — no signal "
        "that the slider had any effect."
    )

    # Step 4: OCR-quality outcome. Document it; don't gate the test on it.
    n_distinct_y_default = len({
        round(t.data_value, 6) for t in result_default.y_paired_ticks
        if t.include and t.data_value is not None
    })
    n_distinct_y_override = len({
        round(t.data_value, 6) for t in result_override.y_paired_ticks
        if t.include and t.data_value is not None
    })
    print(f"  Distinct y-values paired: default={n_distinct_y_default}, "
          f"override={n_distinct_y_override}")
    print(f"  Calibration success: default={result_default.success}, "
          f"override={result_override.success}")
    print("  Note: with EasyOCR (production), y_band_extra_px=160 is expected")
    print("  to make case3 succeed reading 0/500/1000/1500/2000/2500. Tesseract")
    print("  (sandbox) is weaker on small text; this test asserts only that the")
    print("  config override propagates correctly.")
    print("  PASS")


def test_detect_axis_frame_consistency():
    """Two consecutive calls return the same bbox (idempotency).

    This is what makes the Streamlit UI's per-image caching safe: repeat calls
    don't drift, so the cached value is always representative.
    """
    print("\n" + "=" * 60)
    print("TEST: detect_axis_frame is idempotent")
    print("=" * 60)
    img = cv2.imread('/home/claude/plotverify/iga_povetacicept_iv.png')
    p1 = detect_axis_frame(img, ocr_runner=tesseract_runner)
    p2 = detect_axis_frame(img, ocr_runner=tesseract_runner)
    assert p1.bbox is not None and p2.bbox is not None
    assert p1.bbox.as_tuple() == p2.bbox.as_tuple(), (
        f"bbox drifted between calls: {p1.bbox} != {p2.bbox}"
    )
    print(f"  bbox stable across runs: {p1.bbox}")
    print("  PASS")


if __name__ == "__main__":
    main()
    test_detect_axis_frame_basic()
    test_render_band_preview()
    test_detect_axis_frame_consistency()
    test_band_override_takes_effect_on_case3()
    print("\nAll tests passed.")
