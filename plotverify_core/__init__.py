"""UI-agnostic core for PlotVerify.

This package contains all pure logic shared between the Streamlit and Shiny
front-ends: color helpers, masking, calibration math, CSV/image loading,
file matching, the editable overlay model, per-file state, and the
PlotVerifyApp controller.

No module in this package imports `streamlit` or `shiny`. UI code lives
elsewhere and consumes this package.
"""
from .app import PlotVerifyApp
from .colors import (
    FALLBACK_HEX,
    hex_complement,
    hex_to_bgr,
    hex_to_hsv_opencv,
    is_valid_hex,
)
from .calibration_math import (
    P1P2_Y_TOLERANCE_PX,
    compute_calibration,
    data_to_px,
    log10_or_none,
    px_to_data,
)
from .csv_io import (
    OPTIONAL_ERROR_COLUMNS,
    REQUIRED_COLUMNS,
    LoadReport,
    load_csv,
)
from .image_io import (
    LARGE_IMAGE_DOWNSCALE_EDGE,
    LARGE_IMAGE_MAX_BYTES,
    LARGE_IMAGE_MAX_EDGE,
    ImageLoad,
    decode_and_maybe_downscale,
    decode_image_bytes,
    hash_bytes,
)
from .masking import apply_color_mask, delta_e_mask
from .matching import FileEntry, MatchResult, make_file_entry, match_files
from .overlay_model import EditableOverlay, OverlayPoint
from .overlay_traces import OverlayTrace, build_overlay_traces
from .series_state import SeriesState, init_series_states
from .session import (
    Anchors,
    AppState,
    MaskingChoice,
    PerFileState,
    ReviewStatus,
    WorkflowStage,
)

__all__ = [
    "FALLBACK_HEX",
    "Anchors",
    "AppState",
    "EditableOverlay",
    "FileEntry",
    "ImageLoad",
    "MaskingChoice",
    "MatchResult",
    "OverlayPoint",
    "OverlayTrace",
    "PerFileState",
    "PlotVerifyApp",
    "ReviewStatus",
    "WorkflowStage",
    "LARGE_IMAGE_DOWNSCALE_EDGE",
    "LARGE_IMAGE_MAX_BYTES",
    "LARGE_IMAGE_MAX_EDGE",
    "LoadReport",
    "OPTIONAL_ERROR_COLUMNS",
    "P1P2_Y_TOLERANCE_PX",
    "REQUIRED_COLUMNS",
    "SeriesState",
    "apply_color_mask",
    "build_overlay_traces",
    "compute_calibration",
    "data_to_px",
    "decode_and_maybe_downscale",
    "decode_image_bytes",
    "delta_e_mask",
    "hash_bytes",
    "hex_complement",
    "hex_to_bgr",
    "hex_to_hsv_opencv",
    "init_series_states",
    "is_valid_hex",
    "load_csv",
    "log10_or_none",
    "make_file_entry",
    "match_files",
    "px_to_data",
]
