"""
GPU worker mask pipeline — retry, hybrid fusion, failure detection.

Runs on RunPod where SCHP, OpenPose, and DensePose are available.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("idm-vton.worker.mask")

# Legacy scoring gate — set SCORE_LEGACY=1 to restore the original scoring
# formulas (top-22% identity band, flat-texture-peak curve, zero-padded
# gradient, and no protected-region exclusion).  Default 0 = calibrated
# scoring that reliably selects the visually best candidate.
_SCORE_LEGACY = os.environ.get("SCORE_LEGACY", "") in ("1", "true", "True")


class WorkerMaskStrategy(str, Enum):
    EXTERNAL = "external"
    AUTOMASKER = "automasker"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class WorkerMaskAttempt:
    strategy: WorkerMaskStrategy
    mask: Image.Image
    score: float


@dataclass(frozen=True)
class InferenceQualityReport:
    passed: bool
    identity_drift_score: float
    floating_garment_score: float
    missing_arms_score: float
    visible_original_clothing_score: float
    color_fidelity_score: float
    color_drift_mean_rgb: float
    failure_reasons: tuple[str, ...]


def validate_mask_coverage(
    mask: Image.Image,
    cloth_type: str,
    min_coverage: float = 0.04,
    protected_mask: Image.Image | None = None,
) -> dict[str, object]:
    """
    Pre-inference mask sanity check.

    Returns a dict with:
      - valid: True if the mask is reasonable enough for inference
      - coverage_percent: percentage of image covered by mask
      - reason: human-readable failure reason (empty if valid)

    This runs on the GPU worker before ~8s of inference to catch
    pathological masks early.

    If protected_mask is provided, it is used for a precise face-in-mask
    check instead of the heuristic top-12% band.
    """
    mask_np = np.array(mask.convert("L"), dtype=np.uint8)
    h, w = mask_np.shape[:2]
    binary = (mask_np > 127).astype(np.uint8)
    coverage = float(np.sum(binary)) / binary.size

    # Mask too small — nothing to inpaint
    if coverage < min_coverage:
        return {
            "valid": False,
            "coverage_percent": round(coverage * 100.0, 2),
            "reason": f"mask_too_small:{coverage*100:.1f}%",
        }

    # Face zone check: use protected mask if available (more precise),
    # otherwise fall back to heuristic top-12% band.
    if protected_mask is not None:
        prot_np = np.array(protected_mask.convert("L"), dtype=np.uint8)
        prot_binary = (prot_np > 127).astype(np.uint8)
        overlap = np.sum((binary == 1) & (prot_binary == 1))
        prot_total = np.sum(prot_binary)
        if prot_total > 0 and overlap / prot_total > 0.70:
            return {
                "valid": False,
                "coverage_percent": round(coverage * 100.0, 2),
                "reason": f"mask_covers_protected:{overlap/prot_total*100:.0f}%",
            }
    else:
        face_zone = binary[:int(0.12 * h), :]
        face_coverage = float(np.sum(face_zone)) / face_zone.size
        if face_coverage > 0.85:
            return {
                "valid": False,
                "coverage_percent": round(coverage * 100.0, 2),
                "reason": f"mask_covers_face:{face_coverage*100:.0f}%",
            }

    # For lower_body: check that bottom half has meaningful coverage
    if cloth_type in ("lower_body", "dresses"):
        lower_zone = binary[h * 3 // 5:, :]
        lower_coverage = float(np.sum(lower_zone)) / lower_zone.size
        if lower_coverage < 0.03:
            return {
                "valid": False,
                "coverage_percent": round(coverage * 100.0, 2),
                "reason": f"lower_body_too_sparse:{lower_coverage*100:.1f}%",
            }

    return {
        "valid": True,
        "coverage_percent": round(coverage * 100.0, 2),
        "reason": "",
    }


def fuse_hybrid_mask(
    external: Image.Image | None,
    automasker: Image.Image,
    cloth_type: str,
) -> Image.Image:
    """
    GPU hybrid: union external rembg mask with AutoMasker semantic mask,
    then keep AutoMasker-fixed regions (head, shoes, hands).

    Morphology kernel sizes are cloth-type-aware: lower_body and dresses
    get stronger dilation to ensure leg coverage.
    """
    auto_np = np.array(automasker.convert("L"), dtype=np.uint8)
    if external is None:
        return automasker

    ext_np = np.array(external.convert("L"), dtype=np.uint8)
    if ext_np.shape != auto_np.shape:
        ext_np = np.array(
            external.convert("L").resize(automasker.size, Image.NEAREST),
            dtype=np.uint8,
        )

    # Union inpaint regions — external often covers arms rembg caught
    fused = np.maximum(
        (ext_np > 127).astype(np.uint8) * 255,
        (auto_np > 127).astype(np.uint8) * 255,
    )

    # Cloth-type-aware morphology: lower-body needs stronger dilation
    # to ensure full leg coverage from the union mask.
    is_lower = cloth_type in ("lower_body", "dresses")
    erode_size = 5  # Same for all types — prevents background bleed
    dilate_size = 15 if is_lower else 9

    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
    fused = cv2.erode(fused, erode_k, iterations=1)
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
    fused = cv2.dilate(fused, dilate_k, iterations=1)

    return Image.fromarray(fused, mode="L")


def apply_protected_mask(inpaint_mask: Image.Image, protected: Image.Image | None) -> Image.Image:
    if protected is None:
        return inpaint_mask
    inp = np.array(inpaint_mask.convert("L"), dtype=np.uint8)
    prot = np.array(protected.convert("L"), dtype=np.uint8)
    if prot.shape != inp.shape:
        prot = np.array(protected.convert("L").resize(inpaint_mask.size, Image.NEAREST))
    prot_binary = (prot > 127).astype(np.uint8)
    dist = cv2.distanceTransform(prot_binary, cv2.DIST_L2, 5)
    feather = np.clip(dist.astype(np.float32) / 15.0, 0, 1)
    result = inp.astype(np.float32)
    result[prot_binary > 0] = 0.0
    outside = (prot_binary == 0)
    result[outside] = result[outside] * feather[outside]
    return Image.fromarray(result.astype(np.uint8), mode="L")


def detect_inference_failures(
    original: Image.Image,
    result: Image.Image,
    inpaint_mask: Image.Image,
    protected: Image.Image | None = None,
    *,
    identity_threshold: float = 28.0,
    garment_ref: Image.Image | None = None,
) -> InferenceQualityReport:
    """
    Heuristic post-inference QA — triggers retry with alternate mask.

    Args:
        original: Person image BEFORE try-on (used for identity/shape checks).
        result: Try-on output image.
        inpaint_mask: Binary mask of the garment region to evaluate.
        protected: Optional mask of regions to exclude from evaluation.
        identity_threshold: Sensitivity for face-region drift detection.
        garment_ref: Source garment image (used for TRUE color fidelity
            comparison — compares garment source vs output garment color.
            If omitted, a heuristic fallback is used (less accurate).
    """
    orig = np.array(original.convert("RGB"), dtype=np.float32)
    out = np.array(result.convert("RGB"), dtype=np.float32)
    if orig.shape != out.shape:
        out_img = result.convert("RGB").resize(original.size, Image.LANCZOS)
        out = np.array(out_img, dtype=np.float32)

    mask_np = np.array(inpaint_mask.convert("L"), dtype=np.uint8)
    if mask_np.shape[:2] != orig.shape[:2]:
        mask_np = np.array(inpaint_mask.convert("L").resize(original.size, Image.NEAREST))

    reasons: list[str] = []

    # Identity drift — measure only the face/hair region, not shoulders.
    # When a protected mask is available use it as the precise identity
    # region (face + hair + accessories that should not change).
    # Fallback: top 10% of image (tighter than the old 22% band).
    h = orig.shape[0]
    if _SCORE_LEGACY:
        face_band = slice(0, int(0.22 * h), None)
        face_diff = float(np.mean(np.abs(orig[face_band] - out[face_band])))
    elif protected is not None:
        prot_arr = np.array(protected.convert("L"), dtype=np.uint8)
        if prot_arr.shape[:2] != orig.shape[:2]:
            prot_arr = np.array(
                protected.convert("L").resize(orig.shape[1::-1], Image.NEAREST),
                dtype=np.uint8,
            )
        prot_mask = prot_arr > 127
        if np.any(prot_mask):
            face_diff = float(np.mean(np.abs(orig[prot_mask] - out[prot_mask])))
        else:
            face_diff = float(np.mean(np.abs(orig[:int(0.10 * h), :] - out[:int(0.10 * h), :])))
    else:
        face_diff = float(np.mean(np.abs(orig[:int(0.10 * h), :] - out[:int(0.10 * h), :])))
    identity_drift = face_diff
    if identity_drift > identity_threshold:
        reasons.append(f"identity_drift:{identity_drift:.1f}")

    # Missing arms — inpaint region at elbow height has low change (original leaked)
    elbow_y = int(0.38 * h)
    band = slice(max(0, elbow_y - 20), min(h, elbow_y + 20), None)
    inpaint_band = mask_np[band] > 127
    if np.any(inpaint_band):
        diff_band = np.mean(np.abs(orig[band] - out[band]), axis=2)
        change_ratio = float(np.mean(diff_band[inpaint_band] < 8.0))
        missing_arms = change_ratio * 100.0
        if change_ratio > 0.55:
            reasons.append(f"missing_inpaint_at_arms:{change_ratio:.2f}")
    else:
        missing_arms = 0.0

    # Floating garment — high variance at mask edge but flat interior
    edges = cv2.Canny((mask_np > 127).astype(np.uint8) * 255, 50, 150)
    edge_pixels = edges > 0
    if np.any(edge_pixels):
        edge_diff = np.mean(np.abs(orig - out)[edge_pixels])
        floating = float(edge_diff)
        if edge_diff < 6.0:
            reasons.append(f"floating_garment:{edge_diff:.1f}")
    else:
        floating = 0.0

    # Original clothing visible — inpaint area barely changed
    inpaint_region = mask_np > 127
    if np.any(inpaint_region):
        diff_inpaint = np.mean(np.abs(orig - out), axis=2)
        unchanged = float(np.mean(diff_inpaint[inpaint_region] < 10.0))
        visible_original = unchanged * 100.0
        if unchanged > 0.45:
            reasons.append(f"original_clothing_visible:{unchanged:.2f}")
    else:
        visible_original = 0.0

    # ── Color fidelity: compare GARMENT SOURCE vs OUTPUT in inpaint region ──
    # CRITICAL: We compare garment_ref (source product image) against the
    #           output garment, NOT the original person against the output.
    #           Comparing person vs output is backwards — a SUCCESSFUL try-on
    #           SHOULD have a large color change (new garment != old clothing),
    #           so it would ALWAYS register as "drift" — the exact opposite of
    #           what the metric should detect.
    color_drift_mean_rgb = 0.0
    color_fidelity_score = 100.0
    inpaint_region = mask_np > 127
    if np.any(inpaint_region) and garment_ref is not None:
        # Extract mean RGB from source garment (non-white pixels)
        garm = np.array(garment_ref.convert("RGB").resize(out.shape[1::-1], Image.LANCZOS), dtype=np.float32)
        garm_mask = (garm[:, :, 0] < 240) | (garm[:, :, 1] < 240) | (garm[:, :, 2] < 240)
        if np.any(garm_mask):
            garm_mean = np.mean(garm[garm_mask], axis=0)
        else:
            garm_mean = np.mean(garm, axis=(0, 1))

        # Extract mean RGB from output — but exclude protected regions
        # (face, hair, hands, accessories) that inflate drift with
        # skin-tone pixels that aren't part of the garment.
        if _SCORE_LEGACY or protected is None:
            out_garm_mask = inpaint_region
        else:
            prot_arr = np.array(protected.convert("L"), dtype=np.uint8)
            if prot_arr.shape[:2] != out.shape[:2]:
                prot_arr = np.array(
                    protected.convert("L").resize(out.shape[1::-1], Image.NEAREST),
                    dtype=np.uint8,
                )
            out_garm_mask = inpaint_region & ~(prot_arr > 127)

        out_garm = out[out_garm_mask]
        if len(out_garm) > 100:
            out_mean = np.mean(out_garm, axis=0)
            # Per-channel absolute difference between source garment and output
            drift = float(np.mean(np.abs(garm_mean - out_mean)))
            color_drift_mean_rgb = drift

            # Threshold: 50 means ~20% per-channel drift
            # Dark garments get tighter threshold (mean < 80)
            garm_luminance = float(np.mean(garm_mean))
            color_threshold = 30.0 if garm_luminance < 80.0 else 50.0
            color_fidelity_score = max(0.0, 100.0 - (drift / color_threshold) * 100.0)
            if color_fidelity_score < 50.0:
                reasons.append(f"color_drift:{drift:.1f} garm_mean={garm_luminance:.0f}")
    elif np.any(inpaint_region):
        # Fallback: no garment_ref provided — use heuristic
        inpaint_flat = orig[inpaint_region] - out[inpaint_region]
        color_drift_mean_rgb = float(np.mean(np.abs(inpaint_flat)))
        orig_mean = float(np.mean(orig[inpaint_region]))
        color_threshold = 25.0 if orig_mean < 80.0 else 35.0
        color_fidelity_score = max(0.0, 100.0 - (color_drift_mean_rgb / color_threshold) * 100.0)
        if color_fidelity_score < 50.0:
            reasons.append(f"color_drift_fallback:{color_drift_mean_rgb:.1f}")

    passed = len(reasons) == 0
    return InferenceQualityReport(
        passed=passed,
        identity_drift_score=identity_drift,
        floating_garment_score=floating,
        missing_arms_score=missing_arms,
        visible_original_clothing_score=visible_original,
        color_fidelity_score=round(color_fidelity_score, 1),
        color_drift_mean_rgb=round(color_drift_mean_rgb, 1),
        failure_reasons=tuple(reasons),
    )


def _compute_texture_score(
    result: Image.Image,
    garment_ref: Image.Image,
    inpaint_mask: Image.Image,
) -> float:
    """Score texture retention in garment region (0-100, higher = better).

    Measures Laplacian variance (high-frequency energy) in the garment
    region of the output and compares it to the source garment. A ratio
    near 1.0 means texture fidelity is preserved.
    """
    out_np = np.array(result.convert("L"), dtype=np.float32)
    garm_np = np.array(
        garment_ref.convert("L").resize(result.size, Image.LANCZOS),
        dtype=np.float32,
    )
    mask_np = np.array(inpaint_mask.convert("L"), dtype=np.uint8) > 127

    if not np.any(mask_np):
        return 50.0

    out_lap = cv2.Laplacian(out_np, cv2.CV_32F, ksize=3)
    garm_lap = cv2.Laplacian(garm_np, cv2.CV_32F, ksize=3)

    out_energy = float(np.std(out_lap[mask_np]))
    garm_energy = float(np.std(garm_lap[mask_np]))

    if garm_energy < 2.0:
        return 75.0

    ratio = out_energy / max(garm_energy, 1e-6)

    if _SCORE_LEGACY:
        if ratio < 0.2:
            return 0.0
        if ratio < 0.6:
            return (ratio - 0.2) / 0.4 * 60.0
        if ratio < 1.0:
            return 60.0 + (ratio - 0.6) / 0.4 * 30.0
        if ratio < 1.8:
            return 90.0 - (ratio - 1.0) / 0.8 * 20.0
        return max(0.0, 70.0 - (ratio - 1.8) / 2.2 * 70.0)

    # Calibrated curve: peak at ratio ≈ 1.5 where body-worn fabric
    # has natural wrinkle/fold texture energy above a flat product shot.
    #   ratio < 0.3  → severely under-textured (0)
    #   ratio 0.3-0.8 → under-textured, flat (0→40)
    #   ratio 0.8-1.5 → good body-worn texture (40→90)
    #   ratio 1.5-2.5 → over-sharp but acceptable (90→60)
    #   ratio > 2.5   → noisy / artifact (60→5)
    if ratio < 0.3:
        return 0.0
    if ratio < 0.8:
        return (ratio - 0.3) / 0.5 * 40.0
    if ratio < 1.5:
        return 40.0 + (ratio - 0.8) / 0.7 * 50.0
    if ratio < 2.5:
        return 90.0 - (ratio - 1.5) / 1.0 * 30.0
    return max(5.0, 60.0 - (ratio - 2.5) / 2.5 * 55.0)


def _compute_artifact_score(
    result: Image.Image,
    inpaint_mask: Image.Image,
) -> float:
    """Score freedom from artifacts in garment region (0-100, higher = better).

    Checks for two common artifact types:
      1. Dead/flat regions — unnaturally low gradient (zero-detail patches)
      2. Clipped pixels — burned highlights or crushed shadows
    """
    out_np = np.array(result.convert("L"), dtype=np.float32)
    mask_np = np.array(inpaint_mask.convert("L"), dtype=np.uint8) > 127

    if not np.any(mask_np):
        return 50.0

    if _SCORE_LEGACY:
        gx = np.abs(np.diff(out_np, axis=1, append=0))
        gy = np.abs(np.diff(out_np, axis=0, append=0))
        gradient = (gx + gy) * 0.5
    else:
        # Use edge-padding instead of append=0 to avoid creating an
        # artificial large gradient at the image boundary.
        gx = np.abs(np.diff(out_np, axis=1))
        gy = np.abs(np.diff(out_np, axis=0))
        gx = np.pad(gx, ((0, 0), (0, 1)), mode='edge')
        gy = np.pad(gy, ((0, 1), (0, 0)), mode='edge')
        gradient = (gx + gy) * 0.5

    garm_grad = gradient[mask_np]
    dead_ratio = float(np.mean(garm_grad < 0.5)) if len(garm_grad) > 0 else 0.0

    garm_flat = out_np[mask_np]
    extreme_ratio = float(np.mean((garm_flat < 3) | (garm_flat > 252)))

    if _SCORE_LEGACY:
        score = 100.0 - dead_ratio * 50.0 - extreme_ratio * 50.0
    else:
        # Increased penalty coefficients give meaningful differentiation:
        # clean outputs score ~96, artifact-heavy outputs score ~65.
        score = 100.0 - dead_ratio * 80.0 - extreme_ratio * 100.0
    return max(5.0, min(100.0, score))


def compute_aggregate_quality_score(
    quality_report: InferenceQualityReport,
    result: Image.Image,
    inpaint_mask: Image.Image,
    garment_ref: Image.Image,
) -> dict:
    """Compute aggregate quality score (0-100, higher = better).

    Combines identity preservation, color fidelity, texture retention,
    and artifact freedom into a single weighted score used for
    multi-candidate selection.

    Weights:
      identity_drift (inverted) : 30 %
      color_fidelity            : 30 %
      texture_retention         : 25 %
      artifact_freedom          : 15 %
    """
    identity = max(0.0, 100.0 - quality_report.identity_drift_score * 2.5)
    color = quality_report.color_fidelity_score
    texture = _compute_texture_score(result, garment_ref, inpaint_mask)
    artifact = _compute_artifact_score(result, inpaint_mask)

    # Calibrated weights (v2): identity reduced from 30→25% because the
    # tighter face band produces higher scores for good candidates, so it
    # needs less influence. Texture increased from 25→30% because the
    # reworked curve now correctly rewards realistic body-worn fabric,
    # making it a more reliable signal for visual quality.
    if _SCORE_LEGACY:
        aggregate = (
            identity * 0.30 +
            color * 0.30 +
            texture * 0.25 +
            artifact * 0.15
        )
    else:
        aggregate = (
            identity * 0.25 +
            color * 0.30 +
            texture * 0.30 +
            artifact * 0.15
        )

    return {
        "aggregate_score": round(aggregate, 1),
        "identity_score": round(identity, 1),
        "color_fidelity_score": color,
        "texture_score": round(texture, 1),
        "artifact_score": round(artifact, 1),
    }


def select_worker_mask_strategy(
    external_mask: Image.Image | None,
    mask_quality_score: float | None,
    min_quality: float = 62.0,
) -> WorkerMaskStrategy:
    """
    Decide initial mask strategy on GPU.

    Low-quality external masks are ignored in favour of AutoMasker.
    """
    if external_mask is None:
        return WorkerMaskStrategy.AUTOMASKER
    if mask_quality_score is not None and mask_quality_score < min_quality:
        logger.info(
            "external_mask_rejected score=%.1f min=%.1f",
            mask_quality_score,
            min_quality,
        )
        return WorkerMaskStrategy.AUTOMASKER
    return WorkerMaskStrategy.EXTERNAL
