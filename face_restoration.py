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

    person_original is used for identity comparison — kept for logging.
    Returns (enhanced_image, metadata) where metadata includes before/after
    face quality scores for diagnostics.
    """
    if os.environ.get("ENABLE_FACE_RESTORATION", "1") != "0":
        enabled = True
    else:
        enabled = False
    if not enabled:
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

    # ── Face quality BEFORE restoration ─────────────────────────────────
    face_sharpness_before = _estimate_face_sharpness(face_region)
    meta["face_sharpness_before"] = round(face_sharpness_before, 3)

    # If person_original is available, compute identity similarity
    if person_original is not None:
        try:
            orig_np = np.array(person_original.convert("RGB"))
            orig_face = _detect_face_bbox_from_array(orig_np)
            if orig_face is not None:
                ox1, oy1, ox2, oy2 = orig_face
                orig_face_region = orig_np[oy1:oy2, ox1:ox2]
                if orig_face_region.shape == face_region.shape:
                    identity_sim = _compute_identity_similarity(orig_face_region, face_region)
                    meta["identity_similarity_before"] = round(identity_sim, 4)
        except Exception:
            pass

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
    result_np = _blend_face(result_np, enhanced_face, x1, y1, x2, y2, feather_fraction=0.20)

    # ── Face quality AFTER restoration ──────────────────────────────────
    enhanced_region = result_np[y1:y2, x1:x2]
    face_sharpness_after = _estimate_face_sharpness(enhanced_region)
    meta["face_sharpness_after"] = round(face_sharpness_after, 3)
    meta["face_sharpness_delta"] = round(face_sharpness_after - face_sharpness_before, 3)

    # Identity similarity after restoration
    if person_original is not None:
        try:
            orig_np = np.array(person_original.convert("RGB"))
            orig_face = _detect_face_bbox_from_array(orig_np)
            if orig_face is not None:
                ox1, oy1, ox2, oy2 = orig_face
                orig_face_region = orig_np[oy1:oy2, ox1:ox2]
                if orig_face_region.shape == enhanced_region.shape:
                    identity_sim_after = _compute_identity_similarity(orig_face_region, enhanced_region)
                    meta["identity_similarity_after"] = round(identity_sim_after, 4)
                    meta["identity_drift"] = round(
                        abs(identity_sim_after - meta.get("identity_similarity_before", 0)), 4
                    )
        except Exception:
            pass

    elapsed = (time.perf_counter() - t0) * 1000
    meta["restoration_time_ms"] = round(elapsed, 1)
    logger.info(
        "face_restoration done method=%s sharp_before=%.3f sharp_after=%.3f "
        "sharp_delta=%.3f time_ms=%.0f trace_id=%s",
        method, face_sharpness_before, face_sharpness_after,
        face_sharpness_after - face_sharpness_before, elapsed, trace_id,
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
    blurred = cv2.GaussianBlur(face_rgb, (0, 0), sigmaX=0.8)
    sharpened = cv2.addWeighted(face_rgb, 1.15, blurred, -0.15, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _apply_gfpgan(face_rgb: np.ndarray, trace_id: str) -> np.ndarray | None:
    global _restoration_model, _GFPGAN_LOADED
    if _restoration_model is None and not _GFPGAN_LOADED:
        _restoration_model = _load_gfpgan()
    if _restoration_model is None:
        return None
    try:
        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
        # Use paste_back=False to get raw restored face, then we blend ourselves
        # to avoid double-blending artifacts
        cropped, restored, _ = _restoration_model.enhance(
            face_bgr, has_aligned=False, only_center_face=True, paste_back=False,
        )
        if restored and len(restored) > 0:
            return cv2.cvtColor(restored[0], cv2.COLOR_BGR2RGB)
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
        kernel_1d = cv2.getGaussianKernel(feather_pixels * 2 + 1, sigma=feather_pixels / 3).flatten()
        h_grad = np.ones(fw, dtype=np.float32)
        h_grad[:feather_pixels] = kernel_1d[:feather_pixels]
        h_grad[-feather_pixels:] = kernel_1d[-feather_pixels:][::-1]
        v_grad = np.ones(fh, dtype=np.float32)
        v_grad[:feather_pixels] = kernel_1d[:feather_pixels]
        v_grad[-feather_pixels:] = kernel_1d[-feather_pixels:][::-1]
        feather_mask = np.outer(v_grad, h_grad)

    feather_mask_3d = np.stack([feather_mask] * 3, axis=-1)
    roi = result[y1:y2, x1:x2].astype(np.float32)
    blended = roi * (1.0 - feather_mask_3d) + enhanced_resized.astype(np.float32) * feather_mask_3d
    result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


def _estimate_face_sharpness(face_rgb: np.ndarray) -> float:
    """Estimate face sharpness using Laplacian variance.

    Higher values indicate sharper edges. Used to compare face quality
    before and after restoration.
    """
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def _compute_identity_similarity(face_a: np.ndarray, face_b: np.ndarray) -> float:
    """Compute structural similarity between two face regions.

    Uses normalized correlation on grayscale face patches. Values close to 1.0
    indicate high similarity (good identity preservation).
    """
    gray_a = cv2.cvtColor(face_a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(face_b, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Resize to same dimensions if needed
    if gray_a.shape != gray_b.shape:
        gray_b = cv2.resize(gray_b, (gray_a.shape[1], gray_a.shape[0]))

    # Normalize to zero mean, unit variance
    mean_a, std_a = gray_a.mean(), max(gray_a.std(), 1e-6)
    mean_b, std_b = gray_b.mean(), max(gray_b.std(), 1e-6)
    norm_a = (gray_a - mean_a) / std_a
    norm_b = (gray_b - mean_b) / std_b

    # Normalized cross-correlation
    similarity = float(np.mean(norm_a * norm_b))
    return max(-1.0, min(1.0, similarity))
