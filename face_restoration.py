"""
Face Restoration — optional post-inference sharpening.

Default: DISABLED (ENABLE_FACE_RESTORATION=0).

When enabled, sharpens the face region IN the diffusion output only —
does NOT paste original person pixels (which caused halos, identity mismatch,
and visible seams on ethnic skin tones).

Optional GFPGAN tier if packages are installed.
"""

from __future__ import annotations

import logging
import os
import time

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("idm-vton.worker.face_restoration")

_restoration_model = None
_GFPGAN_LOADED = False


def enhance_face(
    result: Image.Image,
    person_original: Image.Image | None = None,
    trace_id: str = "",
) -> tuple[Image.Image, dict[str, object]]:
    """
    Detect face in the result and apply mild in-place enhancement.

    person_original is ignored for blending — kept for API compatibility.
    """
    if os.environ.get("ENABLE_FACE_RESTORATION", "0") != "1":
        return result, {"face_restoration": "disabled"}

    meta: dict[str, object] = {"face_restoration": "enabled"}
    t0 = time.perf_counter()

    result_np = np.array(result.convert("RGB"))
    face_bbox = _detect_face_bbox_from_array(result_np)
    if face_bbox is None:
        elapsed = (time.perf_counter() - t0) * 1000
        meta.update({
            "face_detected": False,
            "restoration_time_ms": round(elapsed, 1),
            "restoration_method": "none",
        })
        return result, meta

    meta["face_detected"] = True
    x1, y1, x2, y2 = face_bbox
    meta["face_bbox"] = [int(x1), int(y1), int(x2), int(y2)]

    face_region = result_np[y1:y2, x1:x2]
    if face_region.size == 0 or face_region.shape[0] < 20 or face_region.shape[1] < 20:
        meta["restoration_time_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        meta["restoration_method"] = "none"
        return result, meta

    method = "opencv_sharpen"
    try:
        gfpgan_face = _apply_gfpgan(face_region, trace_id)
        if gfpgan_face is not None:
            method = "gfpgan"
            enhanced_face = gfpgan_face
        else:
            enhanced_face = _apply_opencv_enhance(face_region)
    except Exception:
        enhanced_face = _apply_opencv_enhance(face_region)

    meta["restoration_method"] = method
    result_np = _blend_face(result_np, enhanced_face, x1, y1, x2, y2, feather_fraction=0.12)

    elapsed = (time.perf_counter() - t0) * 1000
    meta["restoration_time_ms"] = round(elapsed, 1)
    logger.info(
        "face_restoration done method=%s time_ms=%.0f trace_id=%s",
        method, elapsed, trace_id,
    )
    return Image.fromarray(result_np), meta


def _detect_face_bbox_from_array(
    img_np: np.ndarray,
    min_face_ratio: float = 0.03,
) -> tuple[int, int, int, int] | None:
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
    pad_x = int(fw * 0.12)
    pad_y_top = int(fh * 0.22)
    pad_y_bottom = int(fh * 0.12)
    return (
        max(0, fx - pad_x),
        max(0, fy - pad_y_top),
        min(w_img, fx + fw + pad_x),
        min(h_img, fy + fh + pad_y_bottom),
    )


def _apply_opencv_enhance(face_rgb: np.ndarray) -> np.ndarray:
    """Mild unsharp mask on diffusion output — no denoise/CLAHE/saturation."""
    blurred = cv2.GaussianBlur(face_rgb, (0, 0), sigmaX=0.6)
    sharpened = cv2.addWeighted(face_rgb, 1.25, blurred, -0.25, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _apply_gfpgan(face_rgb: np.ndarray, trace_id: str) -> np.ndarray | None:
    global _restoration_model, _GFPGAN_LOADED
    if _restoration_model is None and not _GFPGAN_LOADED:
        _restoration_model = _load_gfpgan()
    if _restoration_model is None:
        return None
    try:
        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
        _, _, restored_bgr = _restoration_model.enhance(
            face_bgr, has_aligned=False, only_center_face=True, paste_back=True,
        )
        if restored_bgr is not None:
            return cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
    except Exception as exc:
        logger.warning("gfpgan_enhance_failed error=%s trace_id=%s", exc, trace_id)
    return None


def _load_gfpgan():
    global _GFPGAN_LOADED, _restoration_model
    try:
        from gfpgan import GFPGANer  # type: ignore[import-untyped]
        model = GFPGANer(
            model_path=None,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
        _GFPGAN_LOADED = True
        logger.info("face_restoration gfpgan_loaded")
        return model
    except ImportError:
        logger.info("face_restoration gfpgan_not_available")
        _GFPGAN_LOADED = True
        return None
    except Exception as exc:
        logger.warning("face_restoration gfpgan_load_failed error=%s", exc)
        _GFPGAN_LOADED = True
        return None


def _blend_face(
    image: np.ndarray,
    enhanced_face: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    feather_fraction: float = 0.12,
) -> np.ndarray:
    """Blend enhanced face with tight feather to avoid neck/chest halos."""
    result = image.copy()
    fh = y2 - y1
    fw = x2 - x1
    enhanced_resized = cv2.resize(enhanced_face, (fw, fh))

    feather_pixels = max(2, int(min(fh, fw) * feather_fraction))
    feather_mask = np.ones((fh, fw), dtype=np.float32)
    if feather_pixels > 2:
        kernel_1d = cv2.getGaussianKernel(feather_pixels * 2 + 1, sigma=feather_pixels / 3)
        kernel_1d = kernel_1d[feather_pixels:-feather_pixels].flatten()
        h_grad = np.ones(fw, dtype=np.float32)
        h_grad[:feather_pixels] = kernel_1d[:min(feather_pixels, len(kernel_1d))]
        h_grad[-feather_pixels:] = kernel_1d[-min(feather_pixels, len(kernel_1d)):][::-1]
        v_grad = np.ones(fh, dtype=np.float32)
        v_grad[:feather_pixels] = kernel_1d[:min(feather_pixels, len(kernel_1d))]
        v_grad[-feather_pixels:] = kernel_1d[-min(feather_pixels, len(kernel_1d)):][::-1]
        feather_mask = np.outer(v_grad, h_grad)

    feather_mask_3d = np.stack([feather_mask] * 3, axis=-1)
    roi = result[y1:y2, x1:x2].astype(np.float32)
    blended = roi * (1.0 - feather_mask_3d) + enhanced_resized.astype(np.float32) * feather_mask_3d
    result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return result
