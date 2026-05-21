"""Image decode + downscale utilities (UI-agnostic).

Replaces the file-loading half of ``_load_image_from_upload`` from
``app_auto_axis.py``. The UI layer (Streamlit/Shiny) holds session state and
shows banners; this module just decodes bytes and optionally downscales.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image


LARGE_IMAGE_MAX_EDGE = 4000
LARGE_IMAGE_MAX_BYTES = 25 * 1024 * 1024  # 25 MB
LARGE_IMAGE_DOWNSCALE_EDGE = 3000


@dataclass
class ImageLoad:
    """Outcome of `decode_image_bytes` / `decode_and_maybe_downscale`."""
    img_bgr: Optional[np.ndarray]
    img_rgb: Optional[np.ndarray]
    image_hash: str
    error: Optional[str] = None
    downscale_factor: float = 1.0
    warnings: List[str] = field(default_factory=list)


def hash_bytes(data: bytes) -> str:
    """MD5 hex digest — used as a stable file ID."""
    return hashlib.md5(data).hexdigest()


def decode_image_bytes(img_bytes: bytes) -> ImageLoad:
    """Decode raw image bytes to BGR + RGB numpy arrays.

    Always returns the original-resolution image (no downscaling). For the
    over-sized check + auto-downscale, call ``decode_and_maybe_downscale``.
    """
    image_hash = hash_bytes(img_bytes)
    try:
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_rgb = np.array(pil)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        return ImageLoad(
            img_bgr=None, img_rgb=None, image_hash=image_hash,
            error=f"Failed to load image: {e}",
        )
    return ImageLoad(img_bgr=img_bgr, img_rgb=img_rgb, image_hash=image_hash)


def decode_and_maybe_downscale(
    img_bytes: bytes,
    *,
    downscale: bool = False,
    max_edge: int = LARGE_IMAGE_MAX_EDGE,
    max_bytes: int = LARGE_IMAGE_MAX_BYTES,
    downscale_to: int = LARGE_IMAGE_DOWNSCALE_EDGE,
) -> ImageLoad:
    """Decode and optionally downscale oversized inputs.

    ``downscale=True`` triggers automatic resize when the image exceeds
    ``max_edge``; otherwise the warning is recorded in ``ImageLoad.warnings``
    but the original resolution is kept. ``image_hash`` is always the hash of
    the original input bytes.
    """
    load = decode_image_bytes(img_bytes)
    if load.error is not None:
        return load

    if load.img_bgr is None:
        raise RuntimeError("decode produced no image array despite no error")
    h, w = load.img_bgr.shape[:2]
    max_edge_actual = max(h, w)
    size_bytes = len(img_bytes)
    over_dim = max_edge_actual > max_edge
    over_bytes = size_bytes > max_bytes

    if over_dim or over_bytes:
        reasons = []
        if over_dim:
            reasons.append(f"{w}x{h}px > {max_edge}px on longest edge")
        if over_bytes:
            reasons.append(
                f"{size_bytes / (1024 * 1024):.1f} MB > "
                f"{max_bytes // (1024 * 1024)} MB"
            )
        load.warnings.append(
            f"Large image: {'; '.join(reasons)}. "
            f"Detection will be slower and more memory-heavy."
        )

        if downscale and over_dim:
            scale = downscale_to / float(max_edge_actual)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            load.img_bgr = cv2.resize(
                load.img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA
            )
            load.img_rgb = cv2.cvtColor(load.img_bgr, cv2.COLOR_BGR2RGB)
            load.downscale_factor = scale
            load.warnings.append(f"Downscaled image to {new_w}x{new_h}px for processing.")

    return load
