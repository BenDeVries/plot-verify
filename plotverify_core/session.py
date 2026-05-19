"""Per-file and app-level session data.

The Streamlit app currently spreads each file's state across many flat
`st.session_state` keys; the Shiny batch workflow needs one isolated bundle
per file. This module defines those bundles plus a minimal `AppState`
container.

These are plain dataclasses — no Streamlit / Shiny imports. UI code holds
the `AppState` reference and reads/writes its fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .overlay_model import EditableOverlay
from .series_state import SeriesState


class MaskingChoice(str, Enum):
    """How to mask the image before calibration. Default: no pre-cal mask."""
    NO_MASK = "no_precalibration_mask"
    DEFAULT_MASK = "default_mask"
    CUSTOM_MASK = "custom_mask"


class ReviewStatus(str, Enum):
    NOT_CALIBRATED = "not_calibrated"
    AUTO_PASSED = "auto_passed"
    REQUIRES_REVIEW = "requires_review"
    REVIEWED = "reviewed"
    MANUALLY_ADJUSTED = "manually_adjusted"
    FAILED = "failed"


class WorkflowStage(str, Enum):
    UPLOAD = "upload"
    MASKING = "masking"
    CALIBRATION = "calibration"
    OVERLAY = "overlay"


@dataclass
class Anchors:
    """The three manual calibration anchors plus their data values."""
    p1_pixel: Tuple[float, float] = (0.0, 0.0)
    p2_pixel: Tuple[float, float] = (0.0, 0.0)
    p3_pixel: Tuple[float, float] = (0.0, 0.0)
    p1_data_x: float = 0.0
    p2_data_x: float = 1.0
    p1_data_y: float = 0.0
    p3_data_y: float = 1.0
    x_log_base: Optional[float] = None
    y_log_base: Optional[float] = None


@dataclass
class PerFileState:
    """All state associated with one uploaded image+CSV pair."""
    file_id: str
    image_filename: str
    image_bytes: bytes
    image_bgr: Optional[np.ndarray] = None
    image_rgb: Optional[np.ndarray] = None
    image_downscale_factor: float = 1.0

    csv_filename: Optional[str] = None
    csv_df: Optional[pd.DataFrame] = None
    overlay: Optional[EditableOverlay] = None

    series_states: Dict[str, SeriesState] = field(default_factory=dict)
    series_color_overrides: Dict[str, str] = field(default_factory=dict)

    masking_choice: MaskingChoice = MaskingChoice.NO_MASK
    mask_ready: bool = False
    cal_masked_img_bgr: Optional[np.ndarray] = None

    # Typed pipeline output. The legacy dict (`detection_legacy_dict`) is
    # kept only for the duration of the deprecation window (Refactor H).
    detection_result: Any = None      # CalibrationResult; Any avoids a forward import
    detection_legacy_dict: Optional[Dict[str, Any]] = None
    manual_anchors: Optional[Anchors] = None

    # Per-file calibration result for the legacy three-point math.
    calibration: Dict[str, Any] = field(default_factory=lambda: {"applied": False})

    review_status: ReviewStatus = ReviewStatus.NOT_CALIBRATED
    review_reasons: List[str] = field(default_factory=list)

    export_filename: Optional[str] = None

    def is_calibrated(self) -> bool:
        return bool(self.calibration.get("applied"))


@dataclass
class AppState:
    """App-level (multi-file) state container."""
    files: Dict[str, PerFileState] = field(default_factory=dict)
    file_order: List[str] = field(default_factory=list)
    active_file_id: Optional[str] = None
    workflow_stage: WorkflowStage = WorkflowStage.UPLOAD

    def add_file(self, file_state: PerFileState) -> None:
        if file_state.file_id not in self.files:
            self.file_order.append(file_state.file_id)
        self.files[file_state.file_id] = file_state
        if self.active_file_id is None:
            self.active_file_id = file_state.file_id

    def remove_file(self, file_id: str) -> None:
        self.files.pop(file_id, None)
        if file_id in self.file_order:
            self.file_order.remove(file_id)
        if self.active_file_id == file_id:
            self.active_file_id = self.file_order[0] if self.file_order else None

    @property
    def active(self) -> Optional[PerFileState]:
        if self.active_file_id is None:
            return None
        return self.files.get(self.active_file_id)

    def select(self, file_id: str) -> None:
        if file_id not in self.files:
            raise KeyError(f"unknown file_id: {file_id}")
        self.active_file_id = file_id

    def next_unreviewed(self) -> Optional[str]:
        """Return the next file_id that still needs review, or None."""
        if not self.file_order:
            return None
        # Start search after the active file (with wraparound).
        start = 0
        if self.active_file_id in self.file_order:
            start = self.file_order.index(self.active_file_id) + 1
        n = len(self.file_order)
        for k in range(n):
            idx = (start + k) % n
            fid = self.file_order[idx]
            if fid == self.active_file_id:
                continue
            status = self.files[fid].review_status
            if status in (ReviewStatus.REQUIRES_REVIEW, ReviewStatus.NOT_CALIBRATED,
                          ReviewStatus.FAILED):
                return fid
        return None
