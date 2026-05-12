"""Multi-phase OCR for axis calibration.

Phase A — Full-image discovery scan
    Catches every text region (axis labels, axis titles, plot title, legend,
    annotations) so they can all be masked out before geometric axis detection.

Phase B — Y-tick label band scan (numeric-allowlist)
    After axis bbox is known, mask everything except a tight band immediately
    left of the y-axis. Re-run OCR with `allowlist='0123456789.+-eE^×x'` so the
    recognizer's output space cannot include letters that look like digits in
    title fonts. Drastically cuts false-positive numeric matches that came from
    title text fragments in Phase A.

Phase C — X-tick label band scan
    Same idea below the x-axis.

Notes
-----
- We import easyocr lazily; the module is usable for tests with a tesseract
  shim (set `OCREngine.run = ...`) without easyocr installed.
- Records from all phases are returned with a `phase` field so downstream
  pairing can prefer band-phase records over discovery-phase records.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .types import OCRPhase, OCRRecord


_NUMERIC_ALLOWLIST = "0123456789.+-eE^×x"

# Superscript digits and sign used on log-scale tick labels (e.g. "10³", "10⁻³").
_SUPERSCRIPT_DIGITS: Dict[str, str] = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
}
_SUPERSCRIPT_MINUS = "⁻"
_SUPERSCRIPT_ALL = frozenset(_SUPERSCRIPT_DIGITS) | {_SUPERSCRIPT_MINUS}

# Dash-like Unicode chars that should all normalise to ASCII hyphen-minus.
_UNICODE_DASHES = str.maketrans({"−": "-", "–": "-", "—": "-"})


def _expand_superscripts(s: str) -> str:
    """Replace runs of Unicode superscript chars with '^' + ASCII equivalent.

    "10³"    → "10^3"
    "10⁻³"   → "10^-3"
    "10⁻¹⁰"  → "10^-10"

    Inserting the literal '^' means the log-10 regex can require an explicit
    caret instead of making it optional — which was causing "1000" to parse as
    10^(00) = 1.
    """
    out: list = []
    i = 0
    while i < len(s):
        c = s[i]
        if c in _SUPERSCRIPT_ALL:
            out.append("^")
            while i < len(s) and s[i] in _SUPERSCRIPT_ALL:
                if s[i] == _SUPERSCRIPT_MINUS:
                    out.append("-")
                else:
                    out.append(_SUPERSCRIPT_DIGITS[s[i]])
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


# ----------------------------------------------------------------------
# Numeric parsing
# ----------------------------------------------------------------------

def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    s = _expand_superscripts(s)       # "10³" → "10^3", "10⁻³" → "10^-3"
    s = s.translate(_UNICODE_DASHES)  # "−" / "–" / "—" → "-"
    s = s.replace(" ", "")
    s = s.replace("，", ",").replace("．", ".")
    return s


def parse_numeric_tick(text: str) -> Tuple[Optional[float], str, str, str]:
    """Parse OCR'd text as a numeric tick value.

    Returns: (value, cleaned_text, parse_status, flag)
    """
    raw = normalize_text(text)
    if not raw:
        return None, raw, "not_numeric", "empty"

    # Reject obviously non-numeric strings before trying OCR repairs;
    # this prevents words like "Day" from being parsed after I→1 swaps.
    allowed_letters = set("OoDdIlSsBeE")
    letters = [ch for ch in raw if ch.isalpha()]
    if any(ch not in allowed_letters for ch in letters):
        return None, raw, "not_numeric", "contains non-numeric letters"

    repairs: List[str] = []
    trans = {
        "O": "0", "o": "0", "D": "0",
        "I": "1", "l": "1", "|": "1", "!": "1",
        "S": "5", "s": "5",
        "B": "8",
    }
    repaired = []
    for ch in raw:
        if ch in trans:
            repaired.append(trans[ch])
            repairs.append(f"{ch}->{trans[ch]}")
        else:
            repaired.append(ch)
    s = "".join(repaired)
    s = s.replace("×10", "e").replace("x10", "e").replace("X10", "e")

    m = re.fullmatch(r"10\^([+-]?\d+)", s)
    if m:
        exp = int(m.group(1))
        flag = "; ".join(repairs) if repairs else ""
        return float(10 ** exp), s, "auto_corrected" if repairs else "parsed_log10", flag

    s2 = s.replace(",", "")
    s2 = re.sub(r"[^0-9eE+\-.]", "", s2)
    if not re.search(r"\d", s2) or s2 in {"-", "+", ".", "-.", "+.", "--", "+-"}:
        return None, s, "not_numeric", "no numeric digits"
    try:
        value = float(s2)
    except ValueError:
        return None, s, "parse_failed", "could not parse as float/log10"

    status = "auto_corrected" if repairs or s != raw else "parsed"
    flag = "; ".join(repairs) if repairs else ""
    return value, s, status, flag


# ----------------------------------------------------------------------
# OCR engine wrapper
# ----------------------------------------------------------------------

@lru_cache(maxsize=4)
def _easyocr_reader(languages: Tuple[str, ...] = ("en",), gpu: bool = False):
    import easyocr  # lazy import
    return easyocr.Reader(list(languages), gpu=gpu, verbose=False)


def _bbox_from_polygon(box) -> Tuple[int, int, int, int]:
    arr = np.asarray(box, dtype=float)
    return (int(np.floor(arr[:, 0].min())),
            int(np.floor(arr[:, 1].min())),
            int(np.ceil(arr[:, 0].max())),
            int(np.ceil(arr[:, 1].max())))


def run_easyocr(
    img_bgr: np.ndarray,
    *,
    gpu: bool = False,
    min_confidence: float = 0.20,
    allowlist: Optional[str] = None,
    phase: str = OCRPhase.FULL.value,
    bbox_offset: Tuple[int, int] = (0, 0),
    upsample: float = 1.0,
    detection_params: Optional[Dict[str, object]] = None,
) -> List[OCRRecord]:
    """Run EasyOCR; convert results to OCRRecord, offsetting bboxes if cropped.

    `upsample` scales the input image before recognition (helpful for small
    matplotlib tick labels — at typical 10pt rendering they're ~10px tall,
    right at EasyOCR's default `min_size=10` detection floor). Bboxes are
    scaled back to original image coordinates before bbox_offset is applied.

    `detection_params` is a dict of overrides forwarded to `reader.readtext`.
    Useful keys for plot tick-label detection:
        - text_threshold (default 0.7): lower → more candidate regions
        - low_text (default 0.4): lower → include weaker text scores
        - link_threshold (default 0.4): lower → keep isolated chars separate
        - min_size (default 10): lower → detect smaller text
        - mag_ratio (default 1.5): EasyOCR's internal upscale factor
        - canvas_size (default 2560): max edge length before downscaling
    """
    reader = _easyocr_reader(gpu=gpu)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    coord_scale = 1.0
    if upsample > 0 and upsample != 1.0:
        h, w = img_rgb.shape[:2]
        new_h = max(1, int(round(h * upsample)))
        new_w = max(1, int(round(w * upsample)))
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        coord_scale = 1.0 / upsample

    kwargs: Dict[str, object] = {"detail": 1, "paragraph": False}
    if allowlist:
        kwargs["allowlist"] = allowlist
    if detection_params:
        kwargs.update(detection_params)
    raw = reader.readtext(img_rgb, **kwargs)
    return _records_from_raw(
        raw,
        min_confidence=min_confidence,
        phase=phase,
        bbox_offset=bbox_offset,
        coord_scale=coord_scale,
    )


def _records_from_raw(
    raw,
    *,
    min_confidence: float,
    phase: str,
    bbox_offset: Tuple[int, int],
    coord_scale: float = 1.0,
) -> List[OCRRecord]:
    dx, dy = bbox_offset
    out: List[OCRRecord] = []
    for box, text, conf in raw:
        c = float(conf)
        if c < min_confidence:
            continue
        x0, y0, x1, y1 = _bbox_from_polygon(box)
        if coord_scale != 1.0:
            x0 = x0 * coord_scale; x1 = x1 * coord_scale
            y0 = y0 * coord_scale; y1 = y1 * coord_scale
        x0 = int(round(x0 + dx)); x1 = int(round(x1 + dx))
        y0 = int(round(y0 + dy)); y1 = int(round(y1 + dy))
        value, cleaned, status, flag = parse_numeric_tick(text)
        out.append(OCRRecord(
            raw_text=str(text),
            cleaned_text=cleaned,
            value=value,
            is_numeric=value is not None,
            confidence=c,
            bbox=(x0, y0, x1, y1),
            center=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
            parse_status=status,
            parse_flag=flag,
            phase=phase,
        ))
    return out


# ----------------------------------------------------------------------
# Region masking
# ----------------------------------------------------------------------

def mask_records(
    img_bgr: np.ndarray,
    records: List[OCRRecord],
    pad: int = 4,
) -> np.ndarray:
    """Return a copy of `img_bgr` with `records`' bboxes painted white.

    Used to remove text from an image before geometric axis detection.
    """
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()
    for r in records:
        x0, y0, x1, y1 = r.bbox
        x0 = max(0, int(x0) - pad); y0 = max(0, int(y0) - pad)
        x1 = min(w, int(x1) + pad); y1 = min(h, int(y1) + pad)
        out[y0:y1, x0:x1] = 255
    return out


def mask_non_numeric_records(
    img_bgr: np.ndarray,
    records: List[OCRRecord],
    pad: int = 4,
) -> np.ndarray:
    """Like `mask_records` but only whitens records where `is_numeric` is False.

    Used before band-scan OCR: keeps tick labels visible (so the band re-scan
    can still find them) while removing rotated axis titles, legend text, plot
    titles, and similar clutter that would otherwise confuse the recognizer
    when the band geometry has to be wide enough to capture multi-digit
    labels.
    """
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()
    for r in records:
        if r.is_numeric:
            continue
        x0, y0, x1, y1 = r.bbox
        x0 = max(0, int(x0) - pad); y0 = max(0, int(y0) - pad)
        x1 = min(w, int(x1) + pad); y1 = min(h, int(y1) + pad)
        out[y0:y1, x0:x1] = 255
    return out


def keep_only_band(
    img_bgr: np.ndarray,
    band: Tuple[int, int, int, int],
) -> np.ndarray:
    """Return a copy of `img_bgr` with everything OUTSIDE `band` painted white.

    `band` = (x0, y0, x1, y1). This is the inverse of `mask_records`: instead of
    blanking text, blank everything except a region. Used to isolate the y-tick
    label strip (or x-tick label strip) before re-running OCR with a numeric
    allowlist, so the recognizer cannot stray into title or legend text.
    """
    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = band
    x0 = max(0, int(x0)); y0 = max(0, int(y0))
    x1 = min(w, int(x1)); y1 = min(h, int(y1))
    out = np.full_like(img_bgr, 255)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = img_bgr[y0:y1, x0:x1]
    return out


def crop_band(
    img_bgr: np.ndarray,
    band: Tuple[int, int, int, int],
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Return a cropped sub-image plus the (dx, dy) offset to map crop coords back."""
    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = band
    x0 = max(0, int(x0)); y0 = max(0, int(y0))
    x1 = min(w, int(x1)); y1 = min(h, int(y1))
    if x1 <= x0 or y1 <= y0:
        return img_bgr.copy(), (0, 0)
    return img_bgr[y0:y1, x0:x1].copy(), (x0, y0)


# ----------------------------------------------------------------------
# Band geometry helpers
# ----------------------------------------------------------------------

def y_label_band(bbox, *, extra_left: int = 55, extra_vertical: int = 0) -> Tuple[int, int, int, int]:
    """Strip immediately left of the y-axis where y-tick labels live.

    `extra_vertical` trims inward from the top and bottom of the bbox, letting
    the user exclude labels at the extreme ends of the y-axis.
    """
    top    = max(0, int(bbox.top) + extra_vertical)
    bottom = max(top, int(bbox.bottom) - extra_vertical)
    return (
        max(0, int(bbox.left) - extra_left),
        top,
        max(0, int(bbox.left) - 1),                      # stop just left of axis line
        bottom,
    )


def x_label_band(bbox, *, extra_below: int = 28, extra_horizontal: int = 0) -> Tuple[int, int, int, int]:
    """Strip immediately below the x-axis where x-tick labels live.

    `extra_horizontal` trims inward from the left and right of the bbox, letting
    the user exclude labels at the extreme ends of the x-axis.
    """
    left  = max(0, int(bbox.left) + extra_horizontal)
    right = max(left, int(bbox.right) - extra_horizontal)
    return (
        left,
        int(bbox.bottom) + 2,                            # start just below axis line
        right,
        int(bbox.bottom) + 2 + extra_below,
    )
