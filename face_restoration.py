"""
Face Restoration for TryLix — optional post-processing step on the GPU worker.

Applies face enhancement to the output image AFTER diffusion inference and
BEFORE Cloudinary upload. Only the face region is modified; the rest of the
image passes through unchanged.

Two tiers:
  1. (Preferred) GFPGAN / CodeFormer — loaded on first use, cached per worker.
  2. (Fallback) OpenCV pipeline — always available, no additional deps.

Controlled by env var:  ENABLE_FACE_RESTORATION=1  (default: 0 = disabled)
"""

from __future__ import annotations

import logging
import os
import time

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("idm-vton.worker.face_restoration")

# Whether the GFPGAN/CodeFormer model was loaded successfully
_restoration_model = None


# ── Public API ────────────────────────────────────────────────────────────


def enhance_face(
    result: Image.Image,
    person_original: Image.Image | None = None,
    trace_id: str = "",
) -> tuple[Image.Image, dict[str, object]]:
    """
    Detect face in the result image and apply enhancement.

    Args:
        result: The output image from diffusion (to be enhanced).
        person_original: The original person image (for color reference).
        trace_id: Trace ID for logging.

    Returns:
        (enhanced_image, meta_dict):
            enhanced_image — the output with face region enhanced.
            meta_dict — keys: face_detected, restoration_time_ms, restoration_method.
    """
    if os.environ.get("ENABLE_FACE_RESTORATION", "1") != "1":
        return result, {"face_restoration": "disabled"}

    meta: dict[str, object] = {"face_restoration": "enabled"}
    t0 = time.perf_counter()

    result_np = np.array(result.convert("RGB"))

    # ── Step 1: Detect face ────────────────────────────────────────────
    face_bbox = _detect_face_bbox_from_array(result_np)
    if face_bbox is None:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("face_restoration no_face_detected time_ms=%.0f trace_id=%s", elapsed, trace_id)
        meta["face_detected"] = False
        meta["restoration_time_ms"] = round(elapsed, 1)
        meta["restoration_method"] = "none"
        return result, meta

    meta["face_detected"] = True
    x1, y1, x2, y2 = face_bbox
    meta["face_bbox"] = [int(x1), int(y1), int(x2), int(y2)]
    logger.info(
        "face_restoration detected bbox=(%d,%d,%d,%d) trace_id=%s",
        x1, y1, x2, y2, trace_id,
    )

    # ── Step 2: Extract face region from ORIGINAL person photo ─────────
    # Using person_original instead of result_np ensures the enhanced face
    # shares the same identity/skin tone as the blend background, making
    # the feathered transition nearly invisible.  Extracting from result_np
    # (diffusion output) causes a visible rectangular seam because the
    # diffusion may alter face appearance.
    if person_original is not None:
        person_np = np.array(person_original.convert("RGB"))
        face_source = person_np
    else:
        face_source = result_np
    face_region = face_source[y1:y2, x1:x2]
    if face_region.size == 0:
        elapsed = (time.perf_counter() - t0) * 1000
        meta["restoration_time_ms"] = round(elapsed, 1)
        meta["restoration_method"] = "none"
        return result, meta

    h, w = face_region.shape[:2]
    if h < 20 or w < 20:
        meta["restoration_time_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        meta["restoration_method"] = "none"
        return result, meta

    # ── Step 3: Apply restoration ───────────────────────────────────────
    method = "opencv_fallback"
    try:
        enhanced_face = _apply_gfpgan(face_region, person_original, trace_id)
        if enhanced_face is not None:
            method = "gfpgan"
            face_region_enhanced = enhanced_face
        else:
            face_region_enhanced = _apply_opencv_enhance(face_region)
    except Exception:
        face_region_enhanced = _apply_opencv_enhance(face_region)

    meta["restoration_method"] = method

    # ── Step 4: Blend back with feathering ─────────────────────────────
    result_np = _blend_face(result_np, face_region_enhanced, x1, y1, x2, y2)

    elapsed = (time.perf_counter() - t0) * 1000
    meta["restoration_time_ms"] = round(elapsed, 1)

    logger.info(
        "face_restoration done method=%s time_ms=%.0f face_size=%dx%d trace_id=%s",
        method, elapsed, w, h, trace_id,
    )

    return Image.fromarray(result_np), meta


# ── Face Detection ────────────────────────────────────────────────────────


def _detect_face_bbox_from_array(
    img_np: np.ndarray,
    min_face_ratio: float = 0.03,
) -> tuple[int, int, int, int] | None:
    """
    Detect the largest face using OpenCV Haar cascade.

    Returns (x1, y1, x2, y2) or None.
    """
    h_img, w_img = img_np.shape[:2]
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    cascade_path = os.path.join(
        cv2.data.haarcascades, "haarcascade_frontalface_default.xml",
    )
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        return None

    min_dim = max(30, int(min(w_img, h_img) * min_face_ratio))
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(min_dim, min_dim),
    )
    if len(faces) == 0:
        return None

    (fx, fy, fw, fh) = max(faces, key=lambda r: r[2] * r[3])
    pad_x = int(fw * 0.18)
    pad_y_top = int(fh * 0.40)
    pad_y_bottom = int(fh * 0.30)
    return (
        max(0, fx - pad_x),
        max(0, fy - pad_y_top),
        min(w_img, fx + fw + pad_x),
        min(h_img, fy + fh + pad_y_bottom),
    )


# ── OpenCV Fallback Pipeline ──────────────────────────────────────────────


def _apply_opencv_enhance(face_rgb: np.ndarray) -> np.ndarray:
    """
    Enhance face region using OpenCV-only operations:
      1. Denoise with non-local means (mild).
      2. Unsharp mask for detail sharpening.
      3. CLAHE on the L channel (Lab space) for local contrast.
      4. Slight saturation boost.

    All operations are intentionally conservative to avoid over-processing.
    """
    result = face_rgb.copy().astype(np.uint8)

    # 1. Mild denoise
    denoised = cv2.fastNlMeansDenoisingColored(result, None, h=6, hColor=6, templateWindowSize=7, searchWindowSize=21)

    # 2. Unsharp mask
    blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)

    # 3. CLAHE on L channel in Lab space
    lab = cv2.cvtColor(sharpened, cv2.COLOR_RGB2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    contrast_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_Lab2RGB)

    # 4. Very slight saturation boost
    hsv = cv2.cvtColor(contrast_enhanced, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.1, 0, 255)
    saturated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return saturated


# ── GFPGAN Integration (optional) ─────────────────────────────────────────


def _apply_gfpgan(
    face_rgb: np.ndarray,
    person_original: Image.Image | None,
    trace_id: str,
) -> np.ndarray | None:
    """
    Apply GFPGAN restoration if the model is available.

    Returns enhanced face region or None (fall back to OpenCV).
    """
    global _restoration_model
    if _restoration_model is None:
        _restoration_model = _load_gfpgan()
    if _restoration_model is None:
        return None

    try:
        # GFPGAN expects BGR uint8
        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
        _, _, restored_bgr = _restoration_model.enhance(
            face_bgr, has_aligned=False, only_center_face=False, paste_back=True,
        )
        if restored_bgr is not None:
            return cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
    except Exception as exc:
        logger.warning("gfpgan_enhance_failed error=%s trace_id=%s", exc, trace_id)

    return None


_GFPGAN_LOADED = False


def _load_gfpgan():
    """Lazy-load GFPGAN model. Returns None if unavailable."""
    global _GFPGAN_LOADED
    if _GFPGAN_LOADED:
        return _restoration_model

    try:
        from gfpgan import GFPGANer  # type: ignore[import-untyped]
        model = GFPGANer(
            model_path=None,  # downloads default if not found
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
        _GFPGAN_LOADED = True
        logger.info("face_restoration gfpgan_loaded")
        return model
    except ImportError:
        logger.info("face_restoration gfpgan_not_available (using OpenCV fallback)")
        _GFPGAN_LOADED = True
        return None
    except Exception as exc:
        logger.warning("face_restoration gfpgan_load_failed error=%s", exc)
        _GFPGAN_LOADED = True
        return None


# ── Blending ──────────────────────────────────────────────────────────────


def _blend_face(
    image: np.ndarray,
    enhanced_face: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
) -> np.ndarray:
    """
    Blend the enhanced face back into the image with a Gaussian falloff
    at the edges to prevent visible seams.
    """
    result = image.copy()
    fh = y2 - y1
    fw = x2 - x1

    # Ensure enhanced face matches the extracted region size
    enhanced_resized = cv2.resize(enhanced_face, (fw, fh))

    # Create a feather mask: white center, Gaussian falloff at edges
    feather_mask = np.ones((fh, fw), dtype=np.float32)
    feather_pixels = min(fh, fw) // 3  # ~33% feather border (was ~17% — too narrow, caused visible seam)
    if feather_pixels > 2:
        kernel_1d = cv2.getGaussianKernel(feather_pixels * 2 + 1, sigma=feather_pixels / 3)
        kernel_1d = kernel_1d[feather_pixels:-feather_pixels].flatten()
        # horizontal gradient
        h_grad = np.ones(fw, dtype=np.float32)
        h_grad[:feather_pixels] = kernel_1d[:min(feather_pixels, len(kernel_1d))]
        h_grad[-feather_pixels:] = kernel_1d[-min(feather_pixels, len(kernel_1d)):][::-1]
        # vertical gradient
        v_grad = np.ones(fh, dtype=np.float32)
        v_grad[:feather_pixels] = kernel_1d[:min(feather_pixels, len(kernel_1d))]
        v_grad[-feather_pixels:] = kernel_1d[-min(feather_pixels, len(kernel_1d)):][::-1]
        feather_mask = np.outer(v_grad, h_grad)

    feather_mask_3d = np.stack([feather_mask] * 3, axis=-1)

    # Blend
    roi = result[y1:y2, x1:x2].astype(np.float32)
    blended = roi * (1.0 - feather_mask_3d) + enhanced_resized.astype(np.float32) * feather_mask_3d
    result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    return result
