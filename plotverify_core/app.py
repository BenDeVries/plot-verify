"""UI-agnostic controller for the multi-file PlotVerify workflow.

`PlotVerifyApp` owns the `AppState` and exposes the high-level operations
(add files, match, calibrate, edit, export) as plain methods. Both the
Streamlit and Shiny UI layers instantiate one `PlotVerifyApp` per session
and bind their widgets to its method calls.

This module imports `axis_pipeline` for the typed calibration API but does
NOT import streamlit or shiny.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

from axis_pipeline import (
    CalibrationConfig,
    CalibrationResult,
    manual_calibration,
    ocr_available,
    run_calibration,
)
from axis_pipeline.legacy import update_result_from_tick_edits

from .csv_io import load_csv
from .image_io import decode_and_maybe_downscale
from .matching import FileEntry, MatchResult, make_file_entry, match_files
from .overlay_model import EditableOverlay
from .series_state import init_series_states
from .session import (
    Anchors,
    AppState,
    MaskingChoice,
    PerFileState,
    ReviewStatus,
)


class PlotVerifyApp:
    """High-level multi-file workflow controller. Pure Python, no UI."""

    def __init__(self, *, ocr_runner: Optional[Callable] = None) -> None:
        self.state = AppState()
        self.ocr_runner = ocr_runner
        self._dirty = False
        self._autosave_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # OCR runtime check
    # ------------------------------------------------------------------

    @property
    def ocr_available(self) -> bool:
        """Whether auto-calibration can run (EasyOCR importable)."""
        if self.ocr_runner is not None:
            return True
        return ocr_available()

    # ------------------------------------------------------------------
    # File ingest
    # ------------------------------------------------------------------

    def add_image(self, filename: str, image_bytes: bytes,
                   *, downscale: bool = False) -> str:
        """Decode and register an image. Returns the file_id."""
        load = decode_and_maybe_downscale(image_bytes, downscale=downscale)
        if load.error:
            raise ValueError(load.error)
        file_id = self._make_file_id(filename, load.image_hash)
        existing = self.state.files.get(file_id)
        if existing is not None:
            existing.image_bgr = load.img_bgr
            existing.image_rgb = load.img_rgb
            existing.image_downscale_factor = load.downscale_factor
            return file_id
        fs = PerFileState(
            file_id=file_id,
            image_filename=filename,
            image_bytes=image_bytes,
            image_bgr=load.img_bgr,
            image_rgb=load.img_rgb,
            image_downscale_factor=load.downscale_factor,
        )
        self.state.add_file(fs)
        self.mark_dirty()
        return file_id

    def add_csv(self, file_id: str, csv_filename: str, csv_text: str) -> None:
        """Attach a CSV to an existing image file_id, building EditableOverlay."""
        if file_id not in self.state.files:
            raise KeyError(f"unknown file_id: {file_id}")
        df, report = load_csv(csv_text)
        if df is None:
            raise ValueError(report.error or "CSV load failed")
        fs = self.state.files[file_id]
        fs.csv_filename = csv_filename
        fs.csv_df = df
        fs.csv_has_series_color = report.has_series_color_column
        fs.csv_error_bar_type = report.error_bar_type
        if report.is_forest:
            fs.plot_type = "forest"
        fs.overlay = EditableOverlay(df)
        fs.series_states = init_series_states(df)
        self.mark_dirty()

    def remove_file(self, file_id: str) -> None:
        self.state.remove_file(file_id)
        self.mark_dirty()

    # ------------------------------------------------------------------
    # File matching (for batch upload UI)
    # ------------------------------------------------------------------

    def match_files(self, images: List[FileEntry],
                     csvs: List[FileEntry]) -> MatchResult:
        """Pure helper exposed on the controller for UI convenience."""
        return match_files(images, csvs)

    @staticmethod
    def make_entry(filename: str, payload: bytes) -> FileEntry:
        return make_file_entry(filename, payload)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    @property
    def active(self) -> Optional[PerFileState]:
        return self.state.active

    def select(self, file_id: str) -> None:
        self.state.select(file_id)
        self.mark_dirty()

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------

    def set_masking_choice(self, file_id: str, choice: MaskingChoice) -> None:
        fs = self._require(file_id)
        fs.masking_choice = choice
        # `NO_MASK` and `DEFAULT_MASK` are ready immediately; `CUSTOM_MASK`
        # requires the user to save their adjustments in the masking tab.
        fs.mask_ready = (choice in (MaskingChoice.NO_MASK, MaskingChoice.DEFAULT_MASK))
        self.mark_dirty()

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def run_auto_calibration(self, file_id: str,
                              *, config: Optional[CalibrationConfig] = None
                              ) -> CalibrationResult:
        """Run the full OCR+geometry pipeline on the image. Stores the result."""
        if not self.ocr_available:
            raise RuntimeError(
                "EasyOCR is not available. Use apply_manual_calibration instead."
            )
        fs = self._require(file_id)
        if fs.image_bgr is None:
            raise RuntimeError(f"file_id {file_id} has no decoded image.")
        cfg = config or CalibrationConfig()
        result = run_calibration(fs.image_bgr, config=cfg, ocr_runner=self.ocr_runner)
        fs.detection_result = result
        fs.detection_legacy_dict = result.to_legacy_dict()
        fs.review_status = (
            ReviewStatus.AUTO_PASSED if result.success and result.confidence >= 0.95
            else ReviewStatus.REQUIRES_REVIEW
        )
        if not result.success:
            fs.review_status = ReviewStatus.FAILED
            fs.review_reasons = list(result.warnings)
        self.mark_dirty()
        return result

    def apply_manual_calibration(self, file_id: str,
                                  anchors: Anchors) -> CalibrationResult:
        """Build a CalibrationResult from three anchors. Always available."""
        fs = self._require(file_id)
        result = manual_calibration(
            p1_pixel=anchors.p1_pixel,
            p2_pixel=anchors.p2_pixel,
            p3_pixel=anchors.p3_pixel,
            p1_data_x=anchors.p1_data_x,
            p2_data_x=anchors.p2_data_x,
            p3_data_y=anchors.p3_data_y,
            p1_data_y=anchors.p1_data_y,
            x_log_base=anchors.x_log_base,
            y_log_base=anchors.y_log_base,
            bbox=fs.detection_result.bbox if fs.detection_result else None,
        )
        fs.manual_anchors = anchors
        fs.detection_result = result
        fs.detection_legacy_dict = result.to_legacy_dict()
        fs.review_status = (
            ReviewStatus.MANUALLY_ADJUSTED if result.success
            else ReviewStatus.FAILED
        )
        if not result.success:
            fs.review_reasons = list(result.warnings)
        self.mark_dirty()
        return result

    def update_tick_edits(self, file_id: str, x_edits, y_edits) -> CalibrationResult:
        """Apply edited tick tables. Requires a prior calibration result."""
        fs = self._require(file_id)
        if fs.detection_result is None:
            raise RuntimeError("No calibration result to edit.")
        new_result = update_result_from_tick_edits(
            fs.detection_result, x_edits, y_edits,
        )
        fs.detection_result = new_result
        fs.detection_legacy_dict = new_result.to_legacy_dict()
        fs.review_status = ReviewStatus.MANUALLY_ADJUSTED
        self.mark_dirty()
        return new_result

    def calibrate_all_with_defaults(self) -> Dict[str, CalibrationResult]:
        """Run auto-calibration on every file with no pending custom masking."""
        results: Dict[str, CalibrationResult] = {}
        for fid in list(self.state.file_order):
            fs = self.state.files[fid]
            if not fs.mask_ready and fs.masking_choice == MaskingChoice.CUSTOM_MASK:
                continue
            try:
                results[fid] = self.run_auto_calibration(fid)
            except Exception as e:
                fs.review_status = ReviewStatus.FAILED
                fs.review_reasons = [str(e)]
        return results

    # ------------------------------------------------------------------
    # Review queue
    # ------------------------------------------------------------------

    def mark_reviewed(self, file_id: str) -> None:
        fs = self._require(file_id)
        fs.review_status = ReviewStatus.REVIEWED
        self.mark_dirty()

    def next_unreviewed(self) -> Optional[str]:
        return self.state.next_unreviewed()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(self, file_id: str, *,
                    include_audit_cols: bool = False) -> bytes:
        """Return the currently-edited CSV for ``file_id`` as UTF-8 bytes."""
        fs = self._require(file_id)
        if fs.overlay is None:
            raise RuntimeError("No CSV loaded for this file.")
        df = fs.overlay.to_dataframe(include_audit_cols=include_audit_cols)
        return df.to_csv(index=False).encode("utf-8")

    # ------------------------------------------------------------------
    # Persistence (.pvsession zip)
    # ------------------------------------------------------------------

    @property
    def is_dirty(self) -> bool:
        """True when state has changed since the last successful save."""
        return self._dirty

    def mark_dirty(self) -> None:
        """Flag the session as having unsaved changes. Called by mutators."""
        self._dirty = True

    def mark_clean(self) -> None:
        self._dirty = False

    def set_autosave_path(self, path: Path) -> None:
        """Bind the controller to a file location for ``autosave_if_dirty``.

        Caller's responsibility to choose a path inside a writable directory.
        """
        self._autosave_path = Path(path)

    def save_session(self, path: Path) -> None:
        """Write the current AppState to ``path`` (a ``.pvsession`` zip)."""
        from .serialization import save_session as _save
        _save(self.state, Path(path))
        self.mark_clean()

    def load_session(self, path: Path) -> None:
        """Replace the current AppState with the contents of ``path``."""
        from .serialization import load_session as _load
        self.state = _load(Path(path), ocr_runner=self.ocr_runner)
        self.mark_clean()

    def autosave_if_dirty(self) -> bool:
        """Write to the bound autosave path when dirty. Returns True iff saved."""
        if not self._dirty or self._autosave_path is None:
            return False
        self.save_session(self._autosave_path)
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_file_id(self, filename: str, image_hash: str) -> str:
        stem = filename.rsplit(".", 1)[0].lower()
        return f"{stem}#{image_hash[:8]}"

    def _require(self, file_id: str) -> PerFileState:
        if file_id not in self.state.files:
            raise KeyError(f"unknown file_id: {file_id}")
        return self.state.files[file_id]
