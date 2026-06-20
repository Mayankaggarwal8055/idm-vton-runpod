"""
Post-processing pipeline for TryLix — compositing, blending, color correction.

Each stage can be disabled independently via env var:
  ENABLE_FACE_COMPOSITE=1       (default: 1)
  ENABLE_SEAMLESS_CLONE=1       (default: 0)
  ENABLE_SKIN_TONE_CORRECTION=1 (default: 1)

Runs after inference, before Cloudinary upload.
"""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("idm-vton.worker.post_processing")


def apply_face_composite(
    result: Image.Image,
    person_original: Image.Image,
    protected_mask: Image.Image | None,
) -> Image.Image:
    """
    Composite the ORIGINAL face and protected regions back into the result.

    Uses the protected mask (face + hands + accessories) to determine which
    pixels should be preserved from the original person photo.

    This is more reliable than face detection — it preserves whatever the
    preprocessing pipeline marked as protected (face, phone, clutch, watch).
    """
    if os.environ.get("ENABLE_FACE_COMPOSITE", "0") != "1":
        return result
    if protected_mask is None:
        return result

    result_np = np.array(result.convert("RGB"), dtype=np.uint8)
    person_np = np.array(person_original.convert("RGB"), dtype=np.uint8)
    mask_np = np.array(protected_mask.convert("L"), dtype=np.uint8)

    # Resize mask if shape mismatch
    if mask_np.shape[:2] != result_np.shape[:2]:
        mask_pil = protected_mask.convert("L").resize(result.size, Image.NEAREST)
        mask_np = np.array(mask_pil, dtype=np.uint8)

    # Dilate protected region to ensure full coverage with soft falloff.
    # Larger kernel (7 vs 5) and more iterations (3 vs 2) create a wider
    # transition band, reducing visible seams where the original face
    # composite meets the diffusion output.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_dilated = cv2.dilate(mask_np, kernel, iterations=3)

    # Feather the edges with a wider Gaussian for smooth blending.
    # The larger sigma=12 spreads the blend over ~30px instead of ~20px,
    # hiding lighting/color mismatches at the composite boundary.
    feather = cv2.GaussianBlur(mask_dilated.astype(np.float32), (31, 31), 12)
    feather_3d = np.stack([feather / 255.0] * 3, axis=-1)

    # Blend: original person where mask is high, result where mask is low
    blended = (
        person_np.astype(np.float32) * feather_3d
        + result_np.astype(np.float32) * (1.0 - feather_3d)
    )

    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def apply_seamless_clone(
    result: Image.Image,
    person_original: Image.Image,
    inpaint_mask: Image.Image,
) -> Image.Image:
    """
    Apply cv2.seamlessClone on the garment edge band to eliminate visible seams.

    Extracts the garment region from the result and blends it into the original
    using Poisson image editing at the mask boundary.
    """
    if os.environ.get("ENABLE_SEAMLESS_CLONE", "0") != "1":
        return result

    result_np = np.array(result.convert("RGB"), dtype=np.uint8)
    person_np = np.array(person_original.convert("RGB"), dtype=np.uint8)
    mask_np = np.array(inpaint_mask.convert("L"), dtype=np.uint8)

    if mask_np.shape[:2] != result_np.shape[:2]:
        mask_pil = inpaint_mask.convert("L").resize(result.size, Image.NEAREST)
        mask_np = np.array(mask_pil, dtype=np.uint8)

    # Only apply seamlessClone on the mask edge band (not the whole mask)
    eroded = cv2.erode(mask_np, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    edge_band = mask_np.copy()
    edge_band[eroded > 127] = 0

    if np.sum(edge_band > 127) < 500:
        return result  # Edge band too small — skip

    try:
        # Use the result garment as the source
        src = result_np.copy()
        src[edge_band == 0] = 0  # Only keep edge band pixels

        # Center of mask for the clone point
        ys, xs = np.where(mask_np > 127)
        if len(xs) == 0:
            return result
        center = (int(np.mean(xs)), int(np.mean(ys)))

        cloned = cv2.seamlessClone(src, person_np, mask_np, center, cv2.NORMAL_CLONE)
        return Image.fromarray(cloned)
    except Exception as exc:
        logger.warning("seamless_clone_failed error=%s", exc)
        return result


def apply_skin_tone_correction(
    result: Image.Image,
    person_original: Image.Image,
    face_bbox: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    """
    Correct skin tone in the result to match the original person.

    Computes per-channel gain in skin regions (face if detected, otherwise
    full image) and applies to the entire result.
    """
    if os.environ.get("ENABLE_SKIN_TONE_CORRECTION", "0") != "1":
        return result

    result_np = np.array(result.convert("RGB"), dtype=np.float32)
    person_np = np.array(person_original.convert("RGB"), dtype=np.float32)

    # Resize person to match result if needed
    if person_np.shape[:2] != result_np.shape[:2]:
        person_np = np.array(
            person_original.convert("RGB").resize(result.size, Image.LANCZOS),
            dtype=np.float32,
        )

    # Define the skin reference region
    if face_bbox is not None:
        x1, y1, x2, y2 = face_bbox
        skin_ref_person = person_np[y1:y2, x1:x2]
        skin_ref_result = result_np[y1:y2, x1:x2]
    else:
        # Fallback: use upper-center region of image (expected face area).
        # Center shifted to 20% height (was 25%) to capture faces in
        # full-body shots where the face is at ~10-15% of image height.
        h, w = person_np.shape[:2]
        cx, cy = w // 2, h // 5
        roi_w, roi_h = w // 2, h // 5
        skin_ref_person = person_np[cy - roi_h // 2:cy + roi_h // 2, cx - roi_w // 2:cx + roi_w // 2]
        skin_ref_result = result_np[cy - roi_h // 2:cy + roi_h // 2, cx - roi_w // 2:cx + roi_w // 2]

    if skin_ref_person.size == 0 or skin_ref_result.size == 0:
        return result

    # Compute per-channel gain: person_mean / result_mean (clamped)
    person_mean = np.mean(skin_ref_person, axis=(0, 1))
    result_mean = np.mean(skin_ref_result, axis=(0, 1))

    # Avoid divide-by-zero
    result_mean = np.maximum(result_mean, 1.0)
    gain = person_mean / result_mean
    gain = np.clip(gain, 0.85, 1.15)  # Max ±15% correction

    # Only apply if drift exceeds 3% threshold
    drift = float(np.max(np.abs(gain - 1.0)))
    if drift < 0.03:
        logger.info("skin_tone_correction skipped (drift=%.3f < 0.03)", drift)
        return result

    logger.info(
        "skin_tone_correction applied gain_r=%.3f gain_g=%.3f gain_b=%.3f drift=%.3f",
        gain[0], gain[1], gain[2], drift,
    )

    corrected = result_np * gain.reshape(1, 1, 3)
    return Image.fromarray(np.clip(corrected, 0, 255).astype(np.uint8))
