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
    DEFAULT_PALETTE,
    FALLBACK_HEX,
    assign_palette_colors,
    detect_background_color,
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
    validate_and_normalize,
)
from .json_io import (
    JsonLoadResult,
    export_json,
    parse_agent_json,
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
from .overlay_image import build_masked_overlay_image
from .overlay_model import EditableOverlay, OverlayPoint
from .overlay_traces import OverlayTrace, build_overlay_traces, is_horizontal_layout
from .serialization import SCHEMA_VERSION, load_session, save_session
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
    "DEFAULT_PALETTE",
    "FALLBACK_HEX",
    "Anchors",
    "AppState",
    "EditableOverlay",
    "FileEntry",
    "JsonLoadResult",
    "ImageLoad",
    "MaskingChoice",
    "MatchResult",
    "OverlayPoint",
    "OverlayTrace",
    "is_horizontal_layout",
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
    "SCHEMA_VERSION",
    "SeriesState",
    "apply_color_mask",
    "assign_palette_colors",
    "build_masked_overlay_image",
    "build_overlay_traces",
    "compute_calibration",
    "data_to_px",
    "decode_and_maybe_downscale",
    "decode_image_bytes",
    "delta_e_mask",
    "detect_background_color",
    "hash_bytes",
    "hex_complement",
    "hex_to_bgr",
    "hex_to_hsv_opencv",
    "init_series_states",
    "is_valid_hex",
    "export_json",
    "load_csv",
    "load_session",
    "log10_or_none",
    "make_file_entry",
    "parse_agent_json",
    "match_files",
    "px_to_data",
    "save_session",
    "validate_and_normalize",
]
