"""Bridge between the legacy flat ``st.session_state`` and ``AppState``.

The Streamlit single-image app (``app_auto_axis.py``) stores its state in
flat ``st.session_state`` keys (``image_bgr``, ``df``, ``calibration``,
``series_states``, ``p1_px_x``...). The Shiny-bound persistence layer
operates on ``AppState`` / ``PerFileState``. These helpers translate
between the two representations so the Streamlit app can use
``save_session`` / ``load_session`` without being refactored.

No Streamlit import — operates on plain ``MutableMapping`` objects so the
helpers are unit-testable with a vanilla dict.
"""
from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

import pandas as pd

from axis_pipeline.legacy import rebuild_result_from_detection

from .app import PlotVerifyApp
from .image_io import decode_and_maybe_downscale
from .overlay_model import EditableOverlay
from .series_state import SeriesState
from .session import Anchors, MaskingChoice, PerFileState, ReviewStatus


SESSION_DIR_ENV = "PLOTVERIFY_SESSION_DIR"
DEFAULT_SESSION_DIR = Path.home() / ".plotverify" / "sessions"
SESSION_FILE_SUFFIX = ".pvsession"

# Flat keys consumed/written by the bridge. Kept in one place so callers
# can clear them in bulk on load.
ANCHOR_KEYS = (
    "p1_px_x", "p1_px_y", "p1_data_x", "p1_data_y",
    "p2_px_x", "p2_px_y", "p2_data_x",
    "p3_px_x", "p3_px_y", "p3_data_y",
)


# ----------------------------------------------------------------------
# Session path
# ----------------------------------------------------------------------

def resolve_session_dir() -> Path:
    """Return the directory where ``.pvsession`` files live.

    Honors ``PLOTVERIFY_SESSION_DIR`` when set; otherwise uses
    ``~/.plotverify/sessions``. The directory is NOT created here —
    callers create it lazily when they actually write.
    """
    override = os.environ.get(SESSION_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return DEFAULT_SESSION_DIR


def resolve_session_path(session_id: str) -> Path:
    """Return ``<session_dir>/<session_id>.pvsession``."""
    if not session_id:
        raise ValueError("session_id must be non-empty")
    return resolve_session_dir() / f"{session_id}{SESSION_FILE_SUFFIX}"


def ensure_session_id(state: MutableMapping[str, Any],
                      key: str = "__pv_session_id") -> str:
    """Return a stable session id for this Streamlit tab.

    Stored under ``key`` in ``state`` so it survives reruns. Generated
    once with ``uuid.uuid4().hex[:12]``.
    """
    sid = state.get(key)
    if not sid:
        sid = uuid.uuid4().hex[:12]
        state[key] = sid
    return sid


# ----------------------------------------------------------------------
# Flat → AppState (build a PlotVerifyApp ready to save)
# ----------------------------------------------------------------------

def build_app_from_flat_state(state: Mapping[str, Any],
                              *, ocr_runner=None) -> PlotVerifyApp:
    """Build a single-file ``PlotVerifyApp`` mirroring ``state``.

    Returns an app with zero files when ``state`` has no decoded image
    (``image_bgr`` missing) — save_session will still write a valid
    (empty) manifest, which load_session round-trips cleanly.

    The app's ``_dirty`` flag is *cleared* before returning so the
    caller can decide whether to call ``mark_dirty`` themselves.
    """
    app = PlotVerifyApp(ocr_runner=ocr_runner)

    image_bgr = state.get("image_bgr")
    image_bytes = state.get("image_bytes")
    if image_bgr is None or image_bytes is None:
        app.mark_clean()
        return app

    image_filename = state.get("image_filename") or "image.png"
    file_id = _make_file_id(image_filename, state.get("image_hash") or "")

    fs = PerFileState(
        file_id=file_id,
        image_filename=image_filename,
        image_bytes=bytes(image_bytes),
        image_bgr=image_bgr,
        image_rgb=state.get("image_rgb"),
        image_downscale_factor=float(state.get("image_downscale_factor", 1.0)),
    )

    df = state.get("df")
    if df is not None and isinstance(df, pd.DataFrame):
        fs.csv_filename = state.get("csv_filename") or "data.csv"
        fs.csv_df = df.copy()
        fs.overlay = EditableOverlay(fs.csv_df)

    fs.series_states = _typed_series_states(state, df)
    fs.series_color_overrides = dict(state.get("series_color_overrides") or {})

    fs.masking_choice = MaskingChoice.NO_MASK
    fs.mask_ready = True

    detection_legacy = state.get("auto_axis_detection")
    detection_typed = state.get("auto_axis_result")
    if detection_typed is not None:
        fs.detection_result = detection_typed
        fs.detection_legacy_dict = detection_typed.to_legacy_dict()
    elif detection_legacy:
        fs.detection_legacy_dict = dict(detection_legacy)
        try:
            fs.detection_result = rebuild_result_from_detection(detection_legacy)
        except Exception:
            fs.detection_result = None

    fs.manual_anchors = _anchors_from_flat(state)

    cal = state.get("calibration") or {"applied": False}
    fs.calibration = dict(cal)

    fs.review_status = (
        ReviewStatus.MANUALLY_ADJUSTED if cal.get("applied")
        else ReviewStatus.NOT_CALIBRATED
    )

    app.state.add_file(fs)
    app.mark_clean()
    return app


# ----------------------------------------------------------------------
# AppState → flat (rehydrate st.session_state after load_session)
# ----------------------------------------------------------------------

def apply_app_to_flat_state(app: PlotVerifyApp,
                            state: MutableMapping[str, Any]) -> Optional[str]:
    """Write the active file's data back into ``state`` (flat keys).

    Returns the active ``file_id`` or None when the loaded session was
    empty. The caller is responsible for clearing transient caches
    (``cal_masked_img_bgr`` etc.) before calling — they're not reset
    here because the caller may want to preserve some keys.
    """
    fs = app.active
    if fs is None:
        return None

    # Decode the image if the loader didn't fill it (defensive — load_session
    # always decodes, but make the bridge robust if a caller hands us a
    # partially-rebuilt PerFileState).
    if fs.image_bgr is None and fs.image_bytes:
        load = decode_and_maybe_downscale(fs.image_bytes, downscale=False)
        if load.error is None:
            fs.image_bgr = load.img_bgr
            fs.image_rgb = load.img_rgb

    state["image_filename"] = fs.image_filename
    state["image_bytes"] = fs.image_bytes
    state["image_bgr"] = fs.image_bgr
    state["image_rgb"] = fs.image_rgb
    # MUST match what _load_image_from_upload computes (full md5 of the raw
    # bytes). If it doesn't, the next rerun's sidebar will treat the
    # still-mounted file_uploader as a NEW image and reset the anchors.
    state["image_hash"] = (
        hashlib.md5(fs.image_bytes).hexdigest() if fs.image_bytes else ""
    )
    state["image_downscale_factor"] = fs.image_downscale_factor

    if fs.csv_df is not None:
        state["df"] = fs.csv_df.copy()
        state["csv_filename"] = fs.csv_filename
        # The sidebar computes csv_hash from raw uploaded CSV text. We don't
        # have those bytes after a session load, but seeding csv_hash from
        # the DataFrame is good enough as long as the user doesn't re-upload
        # the same CSV: an identical re-upload will hash-mismatch and trigger
        # series_states reset, which is acceptable because that path also
        # re-initializes from the DataFrame's series_color.
        state["csv_hash"] = hashlib.md5(
            fs.csv_df.to_csv(index=False).encode("utf-8")
        ).hexdigest()
    else:
        state.pop("df", None)
        state.pop("csv_filename", None)
        state.pop("csv_hash", None)

    state["series_states"] = _flat_series_states(fs.series_states)
    state["series_color_overrides"] = dict(fs.series_color_overrides)
    # Per-series visibility lives in separate widget keys (`vis_<name>`).
    for name, typed in fs.series_states.items():
        state[f"vis_{name}"] = bool(typed.visible)

    state["calibration"] = dict(fs.calibration)

    if fs.detection_legacy_dict is not None:
        state["auto_axis_detection"] = dict(fs.detection_legacy_dict)
    else:
        state["auto_axis_detection"] = None
    state["auto_axis_result"] = fs.detection_result
    state["auto_axis_image_hash"] = state["image_hash"]

    _flat_set_anchors(state, fs)

    # Transient caches must be invalidated — they were keyed on the previous
    # image and are cheap to recompute.
    for k in ("cal_masked_img_bgr", "cal_masked_img_path",
              "frame_preview_cache"):
        state.pop(k, None)

    return fs.file_id


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _make_file_id(filename: str, image_hash: str) -> str:
    stem = filename.rsplit(".", 1)[0].lower() if "." in filename else filename.lower()
    suffix = (image_hash or "anon")[:8]
    return f"{stem}#{suffix}"


def _typed_series_states(state: Mapping[str, Any],
                         df: Optional[pd.DataFrame]) -> Dict[str, SeriesState]:
    """Convert the flat dict-shaped series_states + visibility keys into
    typed ``SeriesState`` instances. Pulls ``color_hex`` from the DataFrame
    when available; otherwise leaves the SeriesState default.
    """
    flat: Mapping[str, Any] = state.get("series_states") or {}
    out: Dict[str, SeriesState] = {}
    for name, d in flat.items():
        if isinstance(d, SeriesState):
            out[name] = d
            continue
        color_hex = ""
        if df is not None and "series_color" in df.columns:
            rows = df[df["series"] == name]
            if len(rows):
                color_hex = str(rows["series_color"].iloc[0])
        if not color_hex:
            color_hex = SeriesState.__dataclass_fields__["color_hex"].default
        ss = SeriesState(
            series=str(name),
            color_hex=color_hex,
            use_delta_e=bool(d.get("use_delta_e", True)),
            delta_e=int(d.get("delta_e", 10)),
            h_min=int(d.get("h_min", 0)),
            h_max=int(d.get("h_max", 179)),
            s_min=int(d.get("s_min", 0)),
            s_max=int(d.get("s_max", 255)),
            v_min=int(d.get("v_min", 0)),
            v_max=int(d.get("v_max", 255)),
            interpolate=bool(d.get("interpolate", False)),
            visible=bool(state.get(f"vis_{name}", True)),
        )
        out[name] = ss
    return out


def _flat_series_states(typed: Mapping[str, SeriesState]
                         ) -> Dict[str, Dict[str, Any]]:
    """Inverse of ``_typed_series_states``: emit dict-shaped series state
    matching the keys the existing Streamlit widgets bind to."""
    out: Dict[str, Dict[str, Any]] = {}
    for name, ss in typed.items():
        out[str(name)] = {
            "use_delta_e": bool(ss.use_delta_e),
            "delta_e": int(ss.delta_e),
            "h_min": int(ss.h_min),
            "h_max": int(ss.h_max),
            "s_min": int(ss.s_min),
            "s_max": int(ss.s_max),
            "v_min": int(ss.v_min),
            "v_max": int(ss.v_max),
            "interpolate": bool(ss.interpolate),
        }
    return out


def _anchors_from_flat(state: Mapping[str, Any]) -> Optional[Anchors]:
    """Build an ``Anchors`` from the flat ``p*_px_*`` / ``p*_data_*`` keys.

    Returns None when no manual-calibration keys have been written (all
    zero), so an unconfigured session doesn't get a spurious anchor entry.
    """
    if all(abs(float(state.get(k, 0.0))) < 1e-9 for k in ANCHOR_KEYS):
        return None
    return Anchors(
        p1_pixel=(float(state.get("p1_px_x", 0.0)),
                  float(state.get("p1_px_y", 0.0))),
        p2_pixel=(float(state.get("p2_px_x", 0.0)),
                  float(state.get("p2_px_y", 0.0))),
        p3_pixel=(float(state.get("p3_px_x", 0.0)),
                  float(state.get("p3_px_y", 0.0))),
        p1_data_x=float(state.get("p1_data_x", 0.0)),
        p2_data_x=float(state.get("p2_data_x", 1.0)),
        p1_data_y=float(state.get("p1_data_y", 0.0)),
        p3_data_y=float(state.get("p3_data_y", 1.0)),
    )


def _flat_set_anchors(state: MutableMapping[str, Any],
                      fs: PerFileState) -> None:
    """Write anchors back to flat keys. Prefers ``fs.manual_anchors``
    but falls back to the CalibrationResult's pixel/data fields so that
    auto-detected sessions also restore the on-image draggable points.
    """
    a = fs.manual_anchors
    if a is None and fs.detection_result is not None:
        r = fs.detection_result
        try:
            a = Anchors(
                p1_pixel=tuple(r.p1_pixel),
                p2_pixel=tuple(r.p2_pixel),
                p3_pixel=tuple(r.p3_pixel),
                p1_data_x=float(r.p1_data_x),
                p2_data_x=float(r.p2_data_x),
                p1_data_y=float(r.p1_data_y),
                p3_data_y=float(r.p3_data_y),
            )
        except Exception:
            a = None
    if a is None:
        return
    state["p1_px_x"], state["p1_px_y"] = float(a.p1_pixel[0]), float(a.p1_pixel[1])
    state["p2_px_x"], state["p2_px_y"] = float(a.p2_pixel[0]), float(a.p2_pixel[1])
    state["p3_px_x"], state["p3_px_y"] = float(a.p3_pixel[0]), float(a.p3_pixel[1])
    state["p1_data_x"] = float(a.p1_data_x)
    state["p2_data_x"] = float(a.p2_data_x)
    state["p1_data_y"] = float(a.p1_data_y)
    state["p3_data_y"] = float(a.p3_data_y)
