"""Save / load PlotVerify sessions as a single zip artifact.

A session zip has this layout:

    session.pvsession (zip)
    ├── manifest.json        # full AppState as JSON
    ├── images/<file_id>.<ext>
    ├── csvs/<file_id>.csv   # original uploaded CSV bytes (verbatim)
    └── overlays/<file_id>.csv  # EditableOverlay audit DataFrame
                                   (only written when has_edits() is True)

Decoded images, mask caches, and frame previews are NEVER serialized —
they're cheap to recompute from ``image_bytes`` plus the saved
``CalibrationConfig``. The numerical pipeline output (``CalibrationResult``)
IS persisted via ``CalibrationResult.to_legacy_dict`` and rebuilt on load
via ``rebuild_result_from_detection``, but only when the saved
``pipeline_version`` matches the current ``axis_pipeline.PIPELINE_VERSION``;
otherwise it's discarded and the pipeline is re-invoked.

Schema policy
-------------
``SCHEMA_VERSION`` covers the structure of this manifest. Bump it when
field shape changes in a non-backward-compatible way. ``pipeline_version``
covers the semantics of the calibration algorithm and is owned by
``axis_pipeline``.
"""
from __future__ import annotations

import dataclasses
import io
import json
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd

from axis_pipeline import PIPELINE_VERSION, CalibrationResult, manual_calibration
from axis_pipeline.legacy import rebuild_result_from_detection

from .csv_io import load_csv
from .image_io import decode_and_maybe_downscale
from .overlay_model import EditableOverlay
from .series_state import SeriesState
from .session import (
    Anchors,
    AppState,
    MaskingChoice,
    PerFileState,
    ReviewStatus,
    WorkflowStage,
)


SCHEMA_VERSION = "0.1"

MANIFEST_NAME = "manifest.json"
IMAGES_DIR = "images"
CSVS_DIR = "csvs"
OVERLAYS_DIR = "overlays"


# ----------------------------------------------------------------------
# Anchors
# ----------------------------------------------------------------------

def _anchors_to_dict(a: Optional[Anchors]) -> Optional[Dict[str, Any]]:
    if a is None:
        return None
    return {
        "p1_pixel": list(a.p1_pixel),
        "p2_pixel": list(a.p2_pixel),
        "p3_pixel": list(a.p3_pixel),
        "p1_data_x": a.p1_data_x,
        "p2_data_x": a.p2_data_x,
        "p1_data_y": a.p1_data_y,
        "p3_data_y": a.p3_data_y,
        "x_log_base": a.x_log_base,
        "y_log_base": a.y_log_base,
    }


def _anchors_from_dict(d: Optional[Dict[str, Any]]) -> Optional[Anchors]:
    if d is None:
        return None
    return Anchors(
        p1_pixel=tuple(d["p1_pixel"]),
        p2_pixel=tuple(d["p2_pixel"]),
        p3_pixel=tuple(d["p3_pixel"]),
        p1_data_x=float(d["p1_data_x"]),
        p2_data_x=float(d["p2_data_x"]),
        p1_data_y=float(d["p1_data_y"]),
        p3_data_y=float(d["p3_data_y"]),
        x_log_base=d.get("x_log_base"),
        y_log_base=d.get("y_log_base"),
    )


# ----------------------------------------------------------------------
# SeriesState
# ----------------------------------------------------------------------

def _series_state_to_dict(s: SeriesState) -> Dict[str, Any]:
    return asdict(s)


def _series_state_from_dict(d: Dict[str, Any]) -> SeriesState:
    return SeriesState(**d)


# ----------------------------------------------------------------------
# PerFileState (the heavy lifting)
# ----------------------------------------------------------------------

def _per_file_state_to_dict(fs: PerFileState) -> Dict[str, Any]:
    """Convert a PerFileState to its JSON-serializable form.

    Excludes derived/transient fields (image_bgr, etc.) — they're
    recomputed on load. Binary payloads (image bytes, CSV bytes, overlay
    edits) are emitted into the zip side-cars; this dict only records
    their relative paths.
    """
    image_ext = _extension(fs.image_filename) or ".png"
    image_path = f"{IMAGES_DIR}/{fs.file_id}{image_ext}"
    csv_path = f"{CSVS_DIR}/{fs.file_id}.csv" if fs.csv_filename else None
    overlay_path = (
        f"{OVERLAYS_DIR}/{fs.file_id}.csv"
        if fs.overlay is not None and fs.overlay.has_edits()
        else None
    )

    detection_dict: Optional[Dict[str, Any]] = None
    if fs.detection_result is not None:
        detection_dict = fs.detection_result.to_legacy_dict()

    return {
        "file_id": fs.file_id,
        "image_filename": fs.image_filename,
        "image_path": image_path,
        "image_downscale_factor": fs.image_downscale_factor,
        "csv_filename": fs.csv_filename,
        "csv_path": csv_path,
        "overlay_path": overlay_path,
        "series_states": {
            name: _series_state_to_dict(s) for name, s in fs.series_states.items()
        },
        "series_color_overrides": dict(fs.series_color_overrides),
        "masking_choice": fs.masking_choice.value,
        "mask_ready": fs.mask_ready,
        "detection_result": detection_dict,
        "manual_anchors": _anchors_to_dict(fs.manual_anchors),
        "calibration": _jsonify(fs.calibration),
        "review_status": fs.review_status.value,
        "review_reasons": list(fs.review_reasons),
        "export_filename": fs.export_filename,
    }


def _per_file_state_from_dict(
    d: Dict[str, Any],
    *,
    image_bytes: bytes,
    csv_text: Optional[str],
    overlay_text: Optional[str],
    pipeline_version_matched: bool,
    ocr_runner: Optional[Callable],
) -> PerFileState:
    """Rebuild a PerFileState from its manifest entry + zipped binaries."""
    # Decode the image (always — derived fields can be empty but the BGR array
    # is needed by every operation downstream).
    load = decode_and_maybe_downscale(
        image_bytes,
        downscale=False,   # already-decoded factor will be preserved below
    )
    if load.error:
        raise ValueError(f"image decode failed for {d['file_id']}: {load.error}")

    fs = PerFileState(
        file_id=d["file_id"],
        image_filename=d["image_filename"],
        image_bytes=image_bytes,
        image_bgr=load.img_bgr,
        image_rgb=load.img_rgb,
        image_downscale_factor=float(d.get("image_downscale_factor", 1.0)),
    )

    fs.csv_filename = d.get("csv_filename")
    if csv_text is not None:
        df, _report = load_csv(csv_text)
        fs.csv_df = df
        if df is not None:
            if overlay_text is not None:
                overlay_df = pd.read_csv(io.StringIO(overlay_text))
                fs.overlay = EditableOverlay.from_audit_dataframe(overlay_df)
            else:
                fs.overlay = EditableOverlay(df)

    fs.series_states = {
        name: _series_state_from_dict(s)
        for name, s in (d.get("series_states") or {}).items()
    }
    fs.series_color_overrides = dict(d.get("series_color_overrides") or {})

    masking_value = d.get("masking_choice", MaskingChoice.NO_MASK.value)
    fs.masking_choice = MaskingChoice(masking_value)
    fs.mask_ready = bool(d.get("mask_ready", False))

    fs.manual_anchors = _anchors_from_dict(d.get("manual_anchors"))
    fs.calibration = dict(d.get("calibration") or {"applied": False})

    review_value = d.get("review_status", ReviewStatus.NOT_CALIBRATED.value)
    fs.review_status = ReviewStatus(review_value)
    fs.review_reasons = list(d.get("review_reasons") or [])
    fs.export_filename = d.get("export_filename")

    # Detection result: trust on match, else re-run.
    saved_detection = d.get("detection_result")
    if saved_detection is not None and pipeline_version_matched:
        result = rebuild_result_from_detection(saved_detection)
        fs.detection_result = result
        fs.detection_legacy_dict = result.to_legacy_dict()
    elif saved_detection is not None or fs.manual_anchors is not None:
        # Mismatched version OR there was a result that needs rebuilding.
        fs.detection_result = _re_run_pipeline(fs, ocr_runner)
        if fs.detection_result is not None:
            fs.detection_legacy_dict = fs.detection_result.to_legacy_dict()

    return fs


def _re_run_pipeline(fs: PerFileState,
                      ocr_runner: Optional[Callable]) -> Optional[CalibrationResult]:
    """Re-invoke the calibration pipeline when the saved version is stale.

    - If manual anchors are set, recompute via ``manual_calibration``.
    - Else, run ``run_calibration`` (requires EasyOCR or an injected runner).
      Returns None if auto-run is unavailable.
    """
    if fs.manual_anchors is not None:
        a = fs.manual_anchors
        return manual_calibration(
            p1_pixel=a.p1_pixel,
            p2_pixel=a.p2_pixel,
            p3_pixel=a.p3_pixel,
            p1_data_x=a.p1_data_x,
            p2_data_x=a.p2_data_x,
            p3_data_y=a.p3_data_y,
            p1_data_y=a.p1_data_y,
            x_log_base=a.x_log_base,
            y_log_base=a.y_log_base,
        )
    from axis_pipeline import ocr_available, run_calibration
    if fs.image_bgr is None:
        return None
    if ocr_runner is None and not ocr_available():
        # Can't re-run; leave detection unset and rely on the user to redo.
        return None
    return run_calibration(fs.image_bgr, ocr_runner=ocr_runner)


# ----------------------------------------------------------------------
# AppState
# ----------------------------------------------------------------------

def _app_state_to_manifest(state: AppState) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "workflow_stage": state.workflow_stage.value,
        "active_file_id": state.active_file_id,
        "file_order": list(state.file_order),
        "files": {
            fid: _per_file_state_to_dict(fs) for fid, fs in state.files.items()
        },
    }


# ----------------------------------------------------------------------
# Save / Load
# ----------------------------------------------------------------------

def save_session(state: AppState, path: Path) -> None:
    """Write the full AppState to a single ``.pvsession`` zip.

    Atomic via write-to-temp then rename.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    manifest = _app_state_to_manifest(state)

    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2, sort_keys=True))
        for fid, fs in state.files.items():
            file_dict = manifest["files"][fid]
            zf.writestr(file_dict["image_path"], fs.image_bytes)
            if fs.csv_df is not None and file_dict["csv_path"]:
                zf.writestr(
                    file_dict["csv_path"],
                    fs.csv_df.to_csv(index=False),
                )
            if fs.overlay is not None and file_dict.get("overlay_path"):
                overlay_df = fs.overlay.to_dataframe(include_audit_cols=True)
                zf.writestr(
                    file_dict["overlay_path"],
                    overlay_df.to_csv(index=False),
                )

    tmp_path.replace(path)


def load_session(
    path: Path,
    *,
    ocr_runner: Optional[Callable] = None,
) -> AppState:
    """Read a ``.pvsession`` zip and rehydrate into an AppState.

    When the saved manifest's ``pipeline_version`` doesn't match the current
    ``axis_pipeline.PIPELINE_VERSION``, the saved CalibrationResult for every
    file is discarded and the pipeline is re-invoked (manual_calibration for
    files with manual_anchors, run_calibration otherwise).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
        if MANIFEST_NAME not in names:
            raise ValueError(f"{path.name}: missing {MANIFEST_NAME}")
        manifest = json.loads(zf.read(MANIFEST_NAME))

        schema = manifest.get("schema_version")
        if schema != SCHEMA_VERSION:
            raise ValueError(
                f"{path.name}: schema_version {schema!r} is incompatible "
                f"with this build (expected {SCHEMA_VERSION!r})."
            )

        saved_pipeline = manifest.get("pipeline_version")
        pipeline_matched = (saved_pipeline == PIPELINE_VERSION)

        state = AppState()
        state.workflow_stage = WorkflowStage(
            manifest.get("workflow_stage", WorkflowStage.UPLOAD.value)
        )
        state.file_order = list(manifest.get("file_order") or [])

        for fid, fd in (manifest.get("files") or {}).items():
            image_path = fd["image_path"]
            if image_path not in names:
                raise ValueError(
                    f"{path.name}: manifest references {image_path!r}, "
                    f"which is not in the archive."
                )
            image_bytes = zf.read(image_path)
            csv_text = None
            if fd.get("csv_path") and fd["csv_path"] in names:
                csv_text = zf.read(fd["csv_path"]).decode("utf-8")
            overlay_text = None
            if fd.get("overlay_path") and fd["overlay_path"] in names:
                overlay_text = zf.read(fd["overlay_path"]).decode("utf-8")

            fs = _per_file_state_from_dict(
                fd,
                image_bytes=image_bytes,
                csv_text=csv_text,
                overlay_text=overlay_text,
                pipeline_version_matched=pipeline_matched,
                ocr_runner=ocr_runner,
            )
            state.files[fid] = fs

        # Restore active selection AFTER files are populated.
        active = manifest.get("active_file_id")
        if active in state.files:
            state.active_file_id = active
        elif state.file_order:
            state.active_file_id = state.file_order[0]

    return state


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _extension(filename: Optional[str]) -> str:
    if not filename:
        return ""
    if "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[1].lower()


def _jsonify(obj: Any) -> Any:
    """Recursive best-effort coercion to JSON-safe primitives.

    Used for the ``calibration`` dict, which is constructed by ad-hoc UI code
    and may contain numpy scalars, tuples, or other non-JSON types.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _jsonify(asdict(obj))
    # numpy scalars
    for attr in ("item",):
        if hasattr(obj, attr):
            try:
                return _jsonify(getattr(obj, attr)())
            except Exception:
                break
    return str(obj)
