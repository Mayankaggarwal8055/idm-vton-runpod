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

    The dilation and feather profile MUST match `apply_protected_mask` in
    mask_pipeline.py (3x9 dilation + DT/40 feather) so the composite zone
    exactly aligns with the zone that was protected during diffusion.
    This prevents visible rectangular seams at face/hand boundaries.
    """
    if os.environ.get("ENABLE_FACE_COMPOSITE", "1") != "1":
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

    # Dilate with same kernel/iterations as apply_protected_mask (3x9)
    # so the composite zone exactly matches the protected zone.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_dilated = cv2.dilate(mask_np, kernel, iterations=3)

    # Distance-transform feather matching mask_pipeline.py (DT/40).
    # Linear falloff over 40px creates a smooth, seam-free transition.
    prot_binary = (mask_dilated > 127).astype(np.uint8)
    dist = cv2.distanceTransform(prot_binary, cv2.DIST_L2, 5)
    feather = np.clip(dist.astype(np.float32) / 40.0, 0, 1)
    feather_3d = np.stack([feather] * 3, axis=-1)

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
    if os.environ.get("ENABLE_SEAMLESS_CLONE", "1") != "1":
        return result

    result_np = np.array(result.convert("RGB"), dtype=np.uint8)
    person_np = np.array(person_original.convert("RGB"), dtype=np.uint8)
    mask_np = np.array(inpaint_mask.convert("L"), dtype=np.uint8)

    if mask_np.shape[:2] != result_np.shape[:2]:
        mask_pil = inpaint_mask.convert("L").resize(result.size, Image.NEAREST)
        mask_np = np.array(mask_pil, dtype=np.uint8)

    # Create a wider edge band for seamlessClone — wider band = smoother transition.
    # Previous erosion kernel was (11,11) with 1 iteration, producing a ~5px band.
    # Now (7,7) with 2 iterations produces an ~8px band for better blending.
    eroded = cv2.erode(mask_np, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=2)
    edge_band = mask_np.copy()
    edge_band[eroded > 127] = 0

    if np.sum(edge_band > 127) < 500:
        return result  # Edge band too small — skip

    try:
        # Use the result garment as the source with the edge band
        src = result_np.copy()
        src[edge_band == 0] = 0  # Only keep edge band pixels

        # Center of mask for the clone point
        ys, xs = np.where(mask_np > 127)
        if len(xs) == 0:
            return result
        center = (int(np.mean(xs)), int(np.mean(ys)))

        # Use MIXED_CLONE for garment boundaries — preserves texture better
        # than NORMAL_CLONE which can wash out fabric patterns.
        cloned = cv2.seamlessClone(src, person_np, mask_np, center, cv2.MIXED_CLONE)
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
    Correct skin tone in the FACE REGION ONLY to match the original person.

    Computes per-channel gain from the face region, then applies it ONLY
    within a generously padded face+neck mask with feathered edges.
    Hands, garment, background, and hair are NOT modified — preventing
    the "hand skin tone changes" and "global color shift" failure modes.
    """
    if os.environ.get("ENABLE_SKIN_TONE_CORRECTION", "1") != "1":
        return result

    result_np = np.array(result.convert("RGB"), dtype=np.float32)
    person_np = np.array(person_original.convert("RGB"), dtype=np.float32)

    # Resize person to match result if needed
    if person_np.shape[:2] != result_np.shape[:2]:
        person_np = np.array(
            person_original.convert("RGB").resize(result.size, Image.LANCZOS),
            dtype=np.float32,
        )

    h_img, w_img = result_np.shape[:2]

    # Define the skin reference region
    if face_bbox is not None:
        x1, y1, x2, y2 = face_bbox
        skin_ref_person = person_np[y1:y2, x1:x2]
        skin_ref_result = result_np[y1:y2, x1:x2]
    else:
        # Fallback: use upper-center region of image (expected face area).
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

    # ── LOCAL correction: only apply gain within a padded face+neck mask ──
    # Create a generously padded face mask so correction covers face + jawline
    # + upper neck while leaving hands, garment, hair, and background untouched.
    if face_bbox is not None:
        x1, y1, x2, y2 = face_bbox
        fw = x2 - x1
        fh = y2 - y1
        pad = int(min(fw, fh) * 0.5)  # 50% padding for neck buffer
        mx1 = max(0, x1 - pad)
        my1 = max(0, y1 - pad)
        mx2 = min(w_img, x2 + pad)
        my2 = min(h_img, y2 + pad)
    else:
        h, w = h_img, w_img
        cx, cy = w // 2, h // 5
        roi_w, roi_h = w // 2, h // 5
        pad = int(min(roi_w, roi_h) * 0.5)
        mx1 = max(0, cx - roi_w // 2 - pad)
        my1 = max(0, cy - roi_h // 2 - pad)
        mx2 = min(w, cx + roi_w // 2 + pad)
        my2 = min(h, cy + roi_h // 2 + pad)

    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    mask[my1:my2, mx1:mx2] = 255

    # Feather the mask edges for seamless blend
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_dilated = cv2.dilate(mask, kernel, iterations=3)
    prot_binary = (mask_dilated > 127).astype(np.uint8)
    dist = cv2.distanceTransform(prot_binary, cv2.DIST_L2, 5)
    feather = np.clip(dist.astype(np.float32) / 40.0, 0, 1)
    feather_3d = np.stack([feather] * 3, axis=-1)

    # Correct only within the feathered mask region
    corrected_rgb = result_np * gain.reshape(1, 1, 3)
    blended = corrected_rgb * feather_3d + result_np * (1.0 - feather_3d)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def apply_region_freeze(
    result: Image.Image,
    person_original: Image.Image,
    inpaint_mask: Image.Image,
) -> Image.Image:
    """
    Copy ALL non-masked pixels from original person image back into result.

    The "region freezing" step ensures the diffusion model only modified the
    garment area. Every pixel outside the inpaint mask (background, face,
    hair, hands, body contour) is restored from the original photo.

    This is the final safety net — even if a few non-garment pixels leak
    into the mask, region freezing puts them back.
    """
    result_np = np.array(result.convert("RGB"), dtype=np.uint8)
    person_np = np.array(person_original.convert("RGB"), dtype=np.uint8)
    mask_np = np.array(inpaint_mask.convert("L"), dtype=np.uint8)

    if mask_np.shape[:2] != result_np.shape[:2]:
        mask_pil = inpaint_mask.convert("L").resize(result.size, Image.NEAREST)
        mask_np = np.array(mask_pil, dtype=np.uint8)

    if person_np.shape[:2] != result_np.shape[:2]:
        person_pil = person_original.convert("RGB").resize(result.size, Image.LANCZOS)
        person_np = np.array(person_pil, dtype=np.uint8)

    # Every pixel where mask is 0 → copy from original person
    frozen = result_np.copy()
    outside_mask = mask_np < 128
    frozen[outside_mask] = person_np[outside_mask]

    logger.info(
        "region_freeze applied frozen_pixels=%.1f%%",
        100.0 * float(np.sum(outside_mask)) / outside_mask.size,
    )
    return Image.fromarray(frozen, mode="RGB")
