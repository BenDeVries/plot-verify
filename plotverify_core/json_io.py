"""Parse and export the agent-produced 'plotverify JSON' format.

The JSON bundles axis calibration anchors, digitised data rows, and
optionally an embedded image (as a ``data_uri``).  Parsing produces the
same ``Anchors`` / ``DataFrame`` / ``LoadReport`` objects the rest of the
app expects, so the existing overlay, editing, and export pipelines work
unchanged.
"""
from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .csv_io import LoadReport, validate_and_normalize
from .session import Anchors

SUPPORTED_SCHEMA_PREFIXES = ("1.",)
EXPORT_SCHEMA_VERSION = "1.1"

# Row coordinates are always image-axis coordinates: `x` is position along
# the image's x-axis, `y` along its y-axis. For horizontal layouts (forest,
# or bar/box with orientation "horizontal") the value therefore lives in `x`
# and `y_err_lower`/`y_err_upper` bracket `x`. Forest-style `value` /
# `value_err_*` aliases are accepted for any horizontal layout.
_ORIENTATIONS = ("vertical", "horizontal")
_ORIENTABLE_PLOT_TYPES = ("bar", "box")

_PLOT_TYPE_MAP: Dict[str, str] = {
    "scatter": "scatter",
    "line_timeseries": "time_series",
    "error_bar": "time_series",
    "forest": "forest",
    "bar": "bar",
    "box": "box",
    "kaplan_meier": "kaplan_meier",
}


@dataclass
class JsonLoadResult:
    """Outcome of :func:`parse_agent_json`."""
    image_bytes: Optional[bytes] = None
    image_filename: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    anchors: Optional[Anchors] = None
    csv_df: Optional[pd.DataFrame] = None
    csv_report: Optional[LoadReport] = None
    plot_type: Optional[str] = None
    orientation: str = "vertical"
    series_colors: Optional[Dict[str, str]] = None
    json_doc: Optional[dict] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_agent_json(raw: str) -> JsonLoadResult:
    """Parse an agent-produced JSON string into app-ready components.

    Returns a :class:`JsonLoadResult`. On validation failure
    ``result.error`` is set; partial fields may still be populated
    (e.g. ``image_bytes`` present even when calibration fails).
    """
    result = JsonLoadResult()
    try:
        doc = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        result.error = f"Invalid JSON: {exc}"
        return result

    if not isinstance(doc, dict):
        result.error = "JSON root must be an object."
        return result
    result.json_doc = doc

    sv = str(doc.get("schema_version", ""))
    if not any(sv.startswith(p) for p in SUPPORTED_SCHEMA_PREFIXES):
        result.error = (
            f"Unsupported schema_version '{sv}' "
            f"(expected prefix: {', '.join(SUPPORTED_SCHEMA_PREFIXES)})."
        )
        return result

    img_bytes, img_name, img_w, img_h, img_warns = _extract_image(doc)
    result.image_bytes = img_bytes
    result.image_filename = img_name
    result.image_width = img_w
    result.image_height = img_h
    result.warnings.extend(img_warns)

    anchors, anch_warns = _extract_anchors(doc)
    result.anchors = anchors
    result.warnings.extend(anch_warns)

    raw_type = str(doc.get("plot_type", "scatter"))
    mapped_type = _PLOT_TYPE_MAP.get(raw_type, "time_series")
    if raw_type not in _PLOT_TYPE_MAP:
        result.warnings.append(
            f"Unknown plot type '{raw_type}'; defaulting to '{mapped_type}'."
        )
    result.plot_type = mapped_type

    result.orientation = _parse_orientation(doc, mapped_type, result.warnings)

    is_forest = mapped_type == "forest"
    df, report, df_warns = _extract_dataframe(
        doc,
        is_forest=is_forest,
        horizontal=(not is_forest and result.orientation == "horizontal"),
    )
    result.csv_df = df
    result.csv_report = report
    result.warnings.extend(df_warns)

    series_list = doc.get("series")
    if isinstance(series_list, list):
        colors: Dict[str, str] = {}
        for s in series_list:
            if isinstance(s, dict):
                key = s.get("key") or s.get("name") or s.get("label", "")
                color = s.get("color", "")
                if key and color:
                    colors[str(key)] = str(color)
        if colors:
            result.series_colors = colors

    return result


def export_json(
    fs: Any,
    *,
    include_audit_cols: bool = False,
    include_image: bool = False,
) -> str:
    """Serialize a :class:`PerFileState` back to the agent JSON schema."""
    doc: Dict[str, Any] = {"schema_version": EXPORT_SCHEMA_VERSION}

    img: Dict[str, Any] = {
        "filename": fs.image_filename or "image.png",
    }
    if fs.image_rgb is not None:
        h, w = fs.image_rgb.shape[:2]
        img["width_px"] = w
        img["height_px"] = h
    if include_image and fs.image_bytes:
        ext = (fs.image_filename or "").rsplit(".", 1)[-1].lower()
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png",
                "webp": "webp", "tif": "tiff", "tiff": "tiff",
                "bmp": "bmp"}.get(ext, "png")
        b64 = base64.b64encode(fs.image_bytes).decode("ascii")
        img["data_uri"] = f"data:image/{mime};base64,{b64}"
    doc["image"] = img

    doc["plot_type"] = fs.plot_type or "scatter"
    doc["orientation"] = (
        "horizontal"
        if (fs.plot_type == "forest"
            or getattr(fs, "orientation", "vertical") == "horizontal")
        else "vertical"
    )

    axes: Dict[str, Any] = {}
    result = fs.detection_result
    if result is not None and result.success:
        if result.x_calibration is not None:
            xcal = result.x_calibration
            x_anchors = _calibration_to_anchors(
                xcal, result.p1_pixel, result.p2_pixel, axis="x",
            )
            axes["x"] = {
                "scale": "log" if xcal.log_base else "linear",
                "calibration": x_anchors,
            }
            if xcal.log_base:
                axes["x"]["log_base"] = float(xcal.log_base)
        if result.y_calibration is not None:
            ycal = result.y_calibration
            y_anchors = _calibration_to_anchors(
                ycal, result.p1_pixel, result.p3_pixel, axis="y",
            )
            axes["y"] = {
                "scale": "log" if ycal.log_base else "linear",
                "calibration": y_anchors,
            }
            if ycal.log_base:
                axes["y"]["log_base"] = float(ycal.log_base)
    doc["axes"] = axes

    if fs.overlay is not None:
        df = fs.overlay.to_dataframe(include_audit_cols=include_audit_cols)
        if fs.csv_df is not None and len(df) == len(fs.csv_df):
            for col in ("box_q1", "box_median", "box_q3", "at_risk",
                         "is_summary", "status"):
                if col in fs.csv_df.columns and col not in df.columns:
                    df[col] = fs.csv_df[col].to_numpy()
        rows = df.to_dict(orient="records")
        for row in rows:
            for k, v in list(row.items()):
                if pd.isna(v):
                    row[k] = None
        doc["rows"] = rows
    else:
        doc["rows"] = []

    unique_series = []
    if fs.csv_df is not None:
        seen: set = set()
        for _, r in fs.csv_df.iterrows():
            s = str(r.get("series", ""))
            if s and s not in seen:
                seen.add(s)
                entry: Dict[str, str] = {"key": s}
                color = r.get("series_color")
                if pd.notna(color):
                    entry["color"] = str(color)
                unique_series.append(entry)
    doc["series"] = unique_series

    return json.dumps(doc, indent=2, default=str)


def rescale_anchors(anchors: Anchors, scale: float) -> Anchors:
    """Return a copy of *anchors* with every pixel coordinate multiplied by
    *scale*. Data values and log bases are unchanged. Used when the decoded
    image was downscaled relative to the coordinates the JSON was written in.
    """
    return replace(
        anchors,
        p1_pixel=(anchors.p1_pixel[0] * scale, anchors.p1_pixel[1] * scale),
        p2_pixel=(anchors.p2_pixel[0] * scale, anchors.p2_pixel[1] * scale),
        p3_pixel=(anchors.p3_pixel[0] * scale, anchors.p3_pixel[1] * scale),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_orientation(doc: dict, plot_type: str, warnings: List[str]) -> str:
    """Validate the top-level ``orientation`` field against the plot type."""
    raw = doc.get("orientation")
    orientation = "vertical"
    if raw is not None:
        val = str(raw).strip().lower()
        if val not in _ORIENTATIONS:
            warnings.append(
                f"Unknown orientation '{raw}'; defaulting to 'vertical'."
            )
        else:
            orientation = val

    if plot_type == "forest":
        if orientation == "vertical" and raw is not None:
            warnings.append(
                "Forest plots are always horizontal; ignoring "
                "orientation 'vertical'."
            )
        return "horizontal"

    if orientation == "horizontal" and plot_type not in _ORIENTABLE_PLOT_TYPES:
        warnings.append(
            f"orientation 'horizontal' has no effect for plot type "
            f"'{plot_type}'; treating as 'vertical'."
        )
        return "vertical"

    return orientation


def _parse_axis_log_base(
    axis_block: dict, axis_name: str, warnings: List[str],
) -> Optional[float]:
    """Resolve the effective log base for one axis block.

    Returns ``None`` for linear axes; otherwise the validated ``log_base``
    (default 10, the string ``"e"`` maps to Euler's number).
    """
    scale = str(axis_block.get("scale", "linear")).lower()
    raw = axis_block.get("log_base")
    if scale != "log":
        if raw is not None:
            warnings.append(
                f"axes.{axis_name}.log_base ignored because scale is not 'log'."
            )
        return None
    if raw is None:
        return 10.0
    if isinstance(raw, str) and raw.strip().lower() == "e":
        return math.e
    try:
        base = float(raw)
    except (TypeError, ValueError):
        base = float("nan")
    if not math.isfinite(base) or base <= 1:
        warnings.append(
            f"Invalid axes.{axis_name}.log_base {raw!r} "
            "(must be a number > 1 or 'e'); using 10."
        )
        return 10.0
    return base


def _extract_image(
    doc: dict,
) -> Tuple[Optional[bytes], Optional[str], Optional[int], Optional[int], List[str]]:
    """Decode the optional ``image.data_uri``."""
    warnings: List[str] = []
    image_block = doc.get("image")
    if not isinstance(image_block, dict):
        return None, None, None, None, ["JSON has no 'image' block."]

    filename = str(image_block.get("filename", "imported.png"))
    width = image_block.get("width_px")
    height = image_block.get("height_px")

    data_uri = image_block.get("data_uri")
    if not data_uri or not isinstance(data_uri, str):
        warnings.append(
            "JSON does not contain an image (no data_uri). "
            "Upload the image separately first, then paste the JSON."
        )
        return None, filename, width, height, warnings

    try:
        if "," in data_uri:
            _, encoded = data_uri.split(",", 1)
        else:
            encoded = data_uri
        image_bytes = base64.b64decode(encoded)
    except Exception as exc:
        warnings.append(f"Failed to decode image data_uri: {exc}")
        return None, filename, width, height, warnings

    return image_bytes, filename, width, height, warnings


def _extract_anchors(
    doc: dict,
) -> Tuple[Optional[Anchors], List[str]]:
    """Map JSON ``axes.*.calibration`` arrays to the 3-anchor model."""
    warnings: List[str] = []
    axes = doc.get("axes")
    if not isinstance(axes, dict):
        return None, ["JSON has no 'axes' block; calibration skipped."]

    x_axis = axes.get("x")
    y_axis = axes.get("y")
    if not isinstance(x_axis, dict) or not isinstance(y_axis, dict):
        return None, ["Both axes.x and axes.y are required for calibration."]

    x_cal = x_axis.get("calibration")
    y_cal = y_axis.get("calibration")
    if not isinstance(x_cal, list) or len(x_cal) < 2:
        return None, ["axes.x.calibration must have >= 2 entries."]
    if not isinstance(y_cal, list) or len(y_cal) < 2:
        return None, ["axes.y.calibration must have >= 2 entries."]

    try:
        x_sorted = sorted(x_cal, key=lambda c: float(c["pixel"]))
        y_sorted = sorted(y_cal, key=lambda c: float(c["pixel"]))
    except (KeyError, TypeError, ValueError) as exc:
        return None, [f"Calibration entries must have numeric 'pixel' and 'value': {exc}"]

    if len(x_sorted) > 2:
        warnings.append(
            f"Using first and last of {len(x_sorted)} x-axis calibration "
            "points (by pixel position); intermediate points ignored."
        )
    if len(y_sorted) > 2:
        warnings.append(
            f"Using first and last of {len(y_sorted)} y-axis calibration "
            "points (by pixel position); intermediate points ignored."
        )

    x_left = x_sorted[0]
    x_right = x_sorted[-1]
    y_top = y_sorted[0]      # smallest pixel = highest on screen
    y_bottom = y_sorted[-1]  # largest pixel = lowest on screen

    try:
        x_left_px = float(x_left["pixel"])
        x_right_px = float(x_right["pixel"])
        y_top_px = float(y_top["pixel"])
        y_bottom_px = float(y_bottom["pixel"])
        x_left_val = float(x_left["value"])
        x_right_val = float(x_right["value"])
        y_top_val = float(y_top["value"])
        y_bottom_val = float(y_bottom["value"])
    except (KeyError, TypeError, ValueError) as exc:
        return None, [f"Calibration entries must have numeric 'pixel' and 'value': {exc}"]

    if x_left_px == x_right_px:
        return None, ["Degenerate x-axis: both calibration points have the same pixel position."]
    if y_top_px == y_bottom_px:
        return None, ["Degenerate y-axis: both calibration points have the same pixel position."]

    x_log_base = _parse_axis_log_base(x_axis, "x", warnings)
    y_log_base = _parse_axis_log_base(y_axis, "y", warnings)

    if x_log_base is not None:
        if x_left_val <= 0 or x_right_val <= 0:
            return None, [
                f"Log x-axis requires positive calibration values, "
                f"got {x_left_val} and {x_right_val}."
            ]
    if y_log_base is not None:
        if y_top_val <= 0 or y_bottom_val <= 0:
            return None, [
                f"Log y-axis requires positive calibration values, "
                f"got {y_top_val} and {y_bottom_val}."
            ]

    anchors = Anchors(
        p1_pixel=(x_left_px, y_bottom_px),     # bottom-left (derived)
        p2_pixel=(x_right_px, y_bottom_px),     # bottom-right
        p3_pixel=(x_left_px, y_top_px),         # top-left
        p1_data_x=x_left_val,
        p2_data_x=x_right_val,
        p1_data_y=y_bottom_val,
        p3_data_y=y_top_val,
        x_log_base=x_log_base,
        y_log_base=y_log_base,
    )
    return anchors, warnings


def _extract_dataframe(
    doc: dict, *, is_forest: bool = False, horizontal: bool = False,
) -> Tuple[Optional[pd.DataFrame], Optional[LoadReport], List[str]]:
    """Convert JSON ``rows`` to a validated DataFrame."""
    warnings: List[str] = []
    rows = doc.get("rows")
    if not isinstance(rows, list) or len(rows) == 0:
        return None, None, ["JSON has no 'rows'; data loading skipped."]

    df = pd.DataFrame(rows)

    # Any horizontal layout accepts the forest-style value/value_err_*
    # aliases; the value lands in `x` (see module docstring). For non-forest
    # horizontal layouts `y` (the category coordinate) remains required.
    if (is_forest or horizontal) and "value" in df.columns and "x" not in df.columns:
        from .csv_io import FOREST_COLUMN_MAP
        df = df.rename(columns=FOREST_COLUMN_MAP)

    df, report = validate_and_normalize(
        df, is_forest=is_forest, horizontal=horizontal,
    )
    if df is None:
        return None, report, [report.error or "Data validation failed."]

    warnings.extend(report.warnings)
    return df, report, warnings


def _calibration_to_anchors(
    cal: Any,
    p_low: Optional[Tuple[float, float]],
    p_high: Optional[Tuple[float, float]],
    *,
    axis: str,
) -> List[Dict[str, float]]:
    """Reconstruct JSON calibration entries from an AxisCalibration."""
    entries: List[Dict[str, float]] = []
    if p_low is None or p_high is None:
        return entries

    if axis == "x":
        px_vals = [p_low[0], p_high[0]]
    else:
        px_vals = [p_low[1], p_high[1]]

    for px in px_vals:
        data_val = cal.pixel_to_data(px)
        if data_val is not None and math.isfinite(data_val):
            entries.append({"pixel": round(px, 2), "value": round(data_val, 6)})

    return entries
