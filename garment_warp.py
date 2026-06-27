"""
Garment alignment / warping preprocessing.

Detects garment orientation, removes excess background padding, centers the
garment on the canvas, and optionally applies mild affine alignment to match
canonical presentation.

Design:
  - Orientation detection: the garment's foreground bounding-box aspect ratio
    determines whether it should be rotated (e.g. a horizontal product shot
    becomes vertical).
  - Padding removal: crops to the tightest foreground bounding box so the
    diffusion model sees maximum garment pixels.
  - Centering: places the cropped garment at canvas centre with proportional
    scaling.

All operations are PIL-based and run before the image enters the VAE encoder.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

logger = logging.getLogger("idm-vton.worker.garment_warp")

TARGET_W = 768
TARGET_H = 1024


def _foreground_bbox(
    img: Image.Image,
    bg_threshold: int = 240,
) -> tuple[int, int, int, int] | None:
    """
    Compute tight bounding box of non-background pixels.

    Returns (left, top, right, bottom) or None if no foreground found.
    """
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    bg = (
        (arr[:, :, 0] > bg_threshold)
        & (arr[:, :, 1] > bg_threshold)
        & (arr[:, :, 2] > bg_threshold)
    )
    fg = ~bg
    ys, xs = np.where(fg)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _detect_orientation(img: Image.Image, fg_ratio_threshold: float = 1.0) -> str:
    """
    Detect whether the garment should be presented as portrait or landscape.

    Returns "portrait" or "landscape".  When the foreground bounding box is
    wider than tall (ratio > threshold), the garment is treated as landscape
    and should be rotated 90° for a vertical canvas.
    """
    bbox = _foreground_bbox(img)
    if bbox is None:
        return "portrait"
    l, t, r, b = bbox
    fw = r - l
    fh = b - t
    if fh == 0:
        return "portrait"
    aspect = fw / fh
    return "landscape" if aspect > fg_ratio_threshold else "portrait"


def align_garment(
    garment_img: Image.Image,
    cloth_type: str = "",
    target_size: tuple[int, int] = (TARGET_W, TARGET_H),
) -> Image.Image:
    """
    Align and centre a garment image on a target canvas.

    Steps:
      1. Foreground bounding-box crop (remove white/background padding).
      2. Orientation check — landscape garments are rotated 90° for a
         vertical canvas.
      3. Aspect-ratio-preserving resize to fit within target_size.
      4. Centre-paste on mid-gray canvas (128,128,128).

    Returns aligned PIL Image in RGB mode.
    """
    img = garment_img.convert("RGB")

    # ── Step 1: tight crop to non-background pixels ─────────────────────
    bbox = _foreground_bbox(img)
    if bbox is not None:
        l, t, r, b = bbox
        fw = r - l
        fh = b - t
        if fw >= 16 and fh >= 16:
            img = img.crop(bbox)
            logger.info(
                "foreground_crop bbox=(%d,%d,%d,%d) size_before=%s size_after=%s",
                l, t, r, b, garment_img.size, img.size,
            )

    # ── Step 2: orientation correction ──────────────────────────────────
    tgt_w, tgt_h = target_size
    is_vertical = tgt_h > tgt_w
    if is_vertical:
        orient = _detect_orientation(img, fg_ratio_threshold=1.0)
        if orient == "landscape":
            img = img.rotate(90, expand=True, resample=Image.BICUBIC)
            logger.info("orientation_rotated_90 landscape_to_portrait")

    # ── Step 3: aspect-ratio-preserving resize ──────────────────────────
    gw, gh = img.size
    scale = min(tgt_w / max(gw, 1), tgt_h / max(gh, 1))
    nw = max(1, int(gw * scale))
    nh = max(1, int(gh * scale))
    img_resized = img.resize((nw, nh), Image.LANCZOS)

    # ── Step 4: centre-paste on mid-gray canvas ─────────────────────────
    # mid-gray (128) matches the preprocessing service.
    canvas = Image.new("RGB", target_size, (128, 128, 128))
    x_offset = (tgt_w - nw) // 2
    y_offset = (tgt_h - nh) // 2
    canvas.paste(img_resized, (x_offset, y_offset))

    logger.info(
        "align_complete original_size=%s target_size=%s placement=(%d,%d,%d,%d)",
        garment_img.size, target_size, x_offset, y_offset, x_offset + nw, y_offset + nh,
    )
    return canvas
