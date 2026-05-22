"""Editable overlay data model.

Wraps a CSV DataFrame in an ``EditableOverlay`` that tracks per-point edits
to (x, y, y_err_lower, y_err_upper), preserves the original values for audit
columns, and round-trips back to a DataFrame for export.

This model is the data target for Shiny's drag-to-edit overlay interactions
(see Part 4 §7 of the Shiny migration plan). It is intentionally framework-
agnostic — neither Streamlit nor Shiny is imported.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .colors import FALLBACK_HEX, is_valid_hex


def _safe_color_hex(value) -> str:
    """Coerce a raw cell value to a valid hex string, falling back to grey.

    Handles ``pd.NA``, ``NaN``, ``None``, and non-string types — all of which
    would otherwise stringify into garbage that Plotly rejects (e.g. ``"<NA>"``).
    """
    if isinstance(value, str) and is_valid_hex(value):
        return value
    return FALLBACK_HEX


@dataclass
class OverlayPoint:
    """One extracted data point with its current and original values."""
    series: str
    point_id: str
    x: float
    y: float
    y_err_lower: Optional[float]
    y_err_upper: Optional[float]
    color_hex: str

    original_x: float
    original_y: float
    original_y_err_lower: Optional[float]
    original_y_err_upper: Optional[float]

    edited: bool = False
    edit_timestamp: Optional[str] = None
    edit_type: Optional[str] = None    # "point" | "err_upper" | "err_lower" | "reset"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class EditableOverlay:
    """Mutable view over an extracted-data DataFrame with per-point edits."""

    def __init__(self, df: pd.DataFrame):
        # `_points` is keyed by stable point_id so edits survive reordering.
        self._points: Dict[str, OverlayPoint] = {}
        for i, row in df.reset_index(drop=True).iterrows():
            pid = f"{row['series']}#{int(i)}"
            self._points[pid] = OverlayPoint(
                series=str(row["series"]),
                point_id=pid,
                x=float(row["x"]),
                y=float(row["y"]),
                y_err_lower=_opt_float(row.get("y_err_lower")),
                y_err_upper=_opt_float(row.get("y_err_upper")),
                color_hex=_safe_color_hex(row.get("series_color")),
                original_x=float(row["x"]),
                original_y=float(row["y"]),
                original_y_err_lower=_opt_float(row.get("y_err_lower")),
                original_y_err_upper=_opt_float(row.get("y_err_upper")),
            )

    @classmethod
    def from_records(cls, records: Iterable[dict]) -> "EditableOverlay":
        """Convenience constructor for unit tests."""
        return cls(pd.DataFrame(list(records)))

    @classmethod
    def from_audit_dataframe(cls, df: pd.DataFrame) -> "EditableOverlay":
        """Rehydrate from a DataFrame produced by ``to_dataframe(include_audit_cols=True)``.

        Unlike the regular constructor (which treats incoming x/y as the
        originals), this reads the explicit ``original_*`` columns plus the
        ``edited``/``edit_type`` flags so saved edits survive a round-trip.
        Falls back to the regular constructor if any audit column is missing.
        """
        required = {"original_x", "original_y", "edited"}
        if not required.issubset(df.columns):
            return cls(df)

        instance = cls.__new__(cls)
        instance._points = {}
        for i, row in df.reset_index(drop=True).iterrows():
            pid = f"{row['series']}#{int(i)}"
            edit_type = row.get("edit_type", "")
            edit_type_val: Optional[str] = str(edit_type) if edit_type else None
            edited = bool(row["edited"])
            instance._points[pid] = OverlayPoint(
                series=str(row["series"]),
                point_id=pid,
                x=float(row["x"]),
                y=float(row["y"]),
                y_err_lower=_opt_float(row.get("y_err_lower")),
                y_err_upper=_opt_float(row.get("y_err_upper")),
                color_hex=_safe_color_hex(row.get("series_color")),
                original_x=float(row["original_x"]),
                original_y=float(row["original_y"]),
                original_y_err_lower=_opt_float(row.get("original_y_err_lower")),
                original_y_err_upper=_opt_float(row.get("original_y_err_upper")),
                edited=edited,
                edit_timestamp=None,        # not persisted across save/load
                edit_type=edit_type_val if edited else None,
            )
        return instance

    # ---- read API -----------------------------------------------------

    def __len__(self) -> int:
        return len(self._points)

    def points(self) -> List[OverlayPoint]:
        return list(self._points.values())

    def get(self, point_id: str) -> OverlayPoint:
        return self._points[point_id]

    def series_names(self) -> List[str]:
        seen: List[str] = []
        for p in self._points.values():
            if p.series not in seen:
                seen.append(p.series)
        return seen

    # ---- mutate API ---------------------------------------------------

    def edit_point(self, point_id: str, new_x: float, new_y: float) -> None:
        p = self._points[point_id]
        p.x = float(new_x)
        p.y = float(new_y)
        p.edited = True
        p.edit_timestamp = _now_iso()
        p.edit_type = "point"

    def edit_err_upper(self, point_id: str, value: Optional[float]) -> None:
        p = self._points[point_id]
        p.y_err_upper = _opt_float(value)
        p.edited = True
        p.edit_timestamp = _now_iso()
        p.edit_type = "err_upper"

    def edit_err_lower(self, point_id: str, value: Optional[float]) -> None:
        p = self._points[point_id]
        p.y_err_lower = _opt_float(value)
        p.edited = True
        p.edit_timestamp = _now_iso()
        p.edit_type = "err_lower"

    def reset_point(self, point_id: str) -> None:
        """Restore the point to its original values; clears the edited flag."""
        p = self._points[point_id]
        p.x = p.original_x
        p.y = p.original_y
        p.y_err_lower = p.original_y_err_lower
        p.y_err_upper = p.original_y_err_upper
        p.edited = False
        p.edit_timestamp = None
        p.edit_type = None

    # ---- export -------------------------------------------------------

    def to_dataframe(self, *, include_audit_cols: bool = False) -> pd.DataFrame:
        """Export the current state as a DataFrame.

        With ``include_audit_cols=True`` the original_* columns and ``edited``/
        ``edit_type`` columns are appended for downstream provenance tracking.
        """
        rows = []
        for p in self._points.values():
            row = {
                "series": p.series,
                "x": p.x,
                "y": p.y,
                "y_err_lower": p.y_err_lower,
                "y_err_upper": p.y_err_upper,
                "series_color": p.color_hex,
            }
            if include_audit_cols:
                row.update({
                    "original_x": p.original_x,
                    "original_y": p.original_y,
                    "original_y_err_lower": p.original_y_err_lower,
                    "original_y_err_upper": p.original_y_err_upper,
                    "edited": p.edited,
                    "edit_type": p.edit_type or "",
                })
            rows.append(row)
        return pd.DataFrame(rows)

    def has_edits(self) -> bool:
        return any(p.edited for p in self._points.values())


def _opt_float(value) -> Optional[float]:
    """Coerce a possibly-NA value to float-or-None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
