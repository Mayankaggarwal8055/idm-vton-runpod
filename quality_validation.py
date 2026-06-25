"""
Post-inference quality validation and candidate scoring.

Provides:
  - Face-region validation: detect face distortion / identity drift.
  - Garment-region validation: measure replacement strength, texture detail.
  - Candidate scoring: aggregate metrics into a single quality score for
    best-candidate selection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

logger = logging.getLogger("idm-vton.worker.quality")


@dataclass
class ValidationResult:
    """Per-candidate quality assessment."""

    passed: bool
    face_quality: float          # 0..1, higher = better
    garment_quality: float       # 0..1, higher = better
    identity_drift: float        # 0..inf, lower = better
    garment_replacement: float   # 0..1, higher = more replacement
    sharpness: float             # 0..inf, higher = sharper
    failure_reasons: list[str] = field(default_factory=list)
    score: float = 0.0           # aggregate weighted score


# ═══════════════════════════════════════════════════════════════════════
# Face-region validation
# ═══════════════════════════════════════════════════════════════════════


def validate_face_region(
    original: Image.Image,
    result: Image.Image,
    mask_np: np.ndarray | None = None,
    protect_np: np.ndarray | None = None,
    schp_labels: np.ndarray | None = None,
) -> tuple[float, float, list[str]]:
    """
    Check face-region quality in the result image.

    Returns (face_quality, identity_drift, failure_reasons) where:
      - face_quality = 0..1 (1 = perfect)
      - identity_drift = mean pixel diff in face zone (0 = identical)
      - failure_reasons = list of issues found

    When schp_labels is available, uses only SCHP label 13 (FACE) to
    define the face zone — excludes hair/glasses/shoes from identity
    measurement. Falls back to protect_np, then top 25% of image.
    """
    orig = np.array(original.convert("RGB"), dtype=np.float32)
    out = np.array(result.convert("RGB"), dtype=np.float32)
    if orig.shape != out.shape:
        out = np.array(
            result.convert("RGB").resize(original.size, Image.LANCZOS),
            dtype=np.float32,
        )

    h = orig.shape[0]
    reasons: list[str] = []

    # ── Face zone: SCHP label 13 (FACE) only when available —────────────
    #    Using label 13 isolates face pixels from hair/sunglasses which
    #    change naturally without indicating identity drift.
    face_mask = np.zeros(orig.shape[:2], dtype=bool)
    if schp_labels is not None:
        if schp_labels.shape[:2] != orig.shape[:2]:
            schp_resized = np.array(
                Image.fromarray(schp_labels.astype(np.uint8)).resize(
                    orig.shape[1::-1], Image.NEAREST,
                ),
                dtype=np.int32,
            )
        else:
            schp_resized = schp_labels.astype(np.int32)
        face_mask = schp_resized == 13  # SCHP FACE label only
    elif protect_np is not None:
        if protect_np.shape[:2] != orig.shape[:2]:
            protect_np = np.array(
                Image.fromarray(protect_np).resize(orig.shape[1::-1], Image.NEAREST),
                dtype=np.uint8,
            )
        face_mask = protect_np > 127

    # Fallback: top 25% region when face mask is sparse
    top_zone = int(0.25 * h)
    if np.sum(face_mask) < 500:
        zone = np.zeros(orig.shape[:2], dtype=bool)
        zone[:top_zone, :] = True
        face_mask = zone

    # ── Identity drift (pixel diff in face zone) ────────────────────────
    if np.any(face_mask):
        face_diff = float(np.mean(np.abs(orig[face_mask] - out[face_mask])))
    else:
        face_diff = float(np.mean(np.abs(orig[:top_zone, :] - out[:top_zone, :])))
    identity_drift = face_diff

    # Normalize to quality score: 30.0 pixel diff = 0.0 quality
    face_quality = max(0.0, min(1.0, 1.0 - identity_drift / 30.0))

    if identity_drift > 20.0:
        reasons.append(f"face_identity_drift:{identity_drift:.1f}")
    if identity_drift > 35.0:
        reasons.append("face_severe_distortion")

    # ── Face-region sharpness (2D Laplacian on face bounding box) ───────
    if np.any(face_mask):
        gray_out = np.array(result.convert("L"), dtype=np.uint8)
        face_ys, face_xs = np.where(face_mask)
        if len(face_ys) > 0 and len(face_xs) > 0:
            y1 = int(face_ys.min())
            y2 = int(face_ys.max()) + 1
            x1 = int(face_xs.min())
            x2 = int(face_xs.max()) + 1
            if y2 - y1 >= 10 and x2 - x1 >= 10:
                face_crop = gray_out[y1:y2, x1:x2]
                import cv2
                lap = cv2.Laplacian(face_crop, cv2.CV_64F)
                lap_sharpness = float(lap.var())
                if lap_sharpness < 10.0:
                    reasons.append("face_blurry")

    return face_quality, identity_drift, reasons


# ═══════════════════════════════════════════════════════════════════════
# Garment-region validation
# ═══════════════════════════════════════════════════════════════════════


def validate_garment_region(
    original: Image.Image,
    result: Image.Image,
    garment_img: Image.Image,
    mask_np: np.ndarray | None,
) -> tuple[float, float, float, list[str]]:
    """
    Check garment-region quality in the result image.

    Returns (garment_quality, garment_replacement, color_coherence, failure_reasons) where:
      - garment_quality = 0..1 (1 = perfect texture/color match)
      - garment_replacement = 0..1 fraction of mask region changed from orig
      - color_coherence = 0..1 (1 = perfect color match with garment)
      - failure_reasons = list of issues found

    Measures:
      1. Replacement strength — how much the inpaint region differs from
         the original (low = ghosting / transparent overlay).
      2. Texture detail — local contrast within the garment region.
      3. Color coherence — deviation from the input garment colour.
    """
    orig = np.array(original.convert("RGB"), dtype=np.float32)
    out = np.array(result.convert("RGB"), dtype=np.float32)
    garm = np.array(garment_img.convert("RGB"), dtype=np.float32)
    if out.shape[:2] != orig.shape[:2]:
        out = np.array(
            result.convert("RGB").resize(original.size, Image.LANCZOS),
            dtype=np.float32,
        )

    reasons: list[str] = []
    h, w = orig.shape[:2]

    if mask_np is None or mask_np.shape[:2] != (h, w):
        mask_np = np.ones((h, w), dtype=np.uint8) * 255
    inpaint_region = mask_np > 127

    if not np.any(inpaint_region):
        return 0.5, 0.0, 0.5, ["no_inpaint_region_found"]

    # ── 1. Replacement strength ─────────────────────────────────────────
    diff = np.mean(np.abs(orig - out), axis=2)
    region_diff = diff[inpaint_region]
    replacement = float(np.mean(region_diff > 10.0))

    if replacement < 0.20:
        reasons.append(f"garment_ghosting:{replacement:.2f}")
    if replacement < 0.05:
        reasons.append("garment_unchanged")

    # ── 2. Texture detail (local contrast in garment region) ─────────────
    gray_out = np.array(result.convert("L"), dtype=np.float32)
    if np.any(inpaint_region):
        garm_gray = gray_out[inpaint_region]
        texture_var = float(np.var(garm_gray))
        # Normalise: variance of 2000+ = detailed, <500 = smooth/plastic
        texture_detail = min(1.0, texture_var / 2000.0)
    else:
        texture_detail = 0.5

    if texture_detail < 0.2:
        reasons.append("garment_over_smooth")

    # ── 3. Colour coherence with input garment (mask region only) ───────
    if np.any(inpaint_region):
        garm_region = garm[inpaint_region]
        out_region = out[inpaint_region]
        garm_mean = np.mean(garm_region, axis=0)
        out_mean = np.mean(out_region, axis=0)
        color_diff = float(np.linalg.norm(garm_mean - out_mean))
    else:
        color_diff = 0.0
    color_coherence = max(0.0, min(1.0, 1.0 - color_diff / 150.0))

    if color_coherence < 0.4:
        reasons.append(f"garment_color_drift:{color_diff:.0f}")

    # ── Aggregate garment quality ───────────────────────────────────────
    garment_quality = 0.35 * replacement + 0.35 * texture_detail + 0.30 * color_coherence

    return garment_quality, replacement, color_coherence, reasons


# ═══════════════════════════════════════════════════════════════════════
# Global / overall quality
# ═══════════════════════════════════════════════════════════════════════


def _sharpness_score(img: Image.Image) -> float:
    """Laplacian-based sharpness metric (higher = sharper)."""
    gray = np.array(img.convert("L"), dtype=np.uint8)
    lap = cv2_laplacian(gray)
    return float(lap.var()) if lap.size > 0 else 0.0


def cv2_laplacian(gray: np.ndarray) -> np.ndarray:
    """Laplacian of grayscale image via convolution (no cv2 import needed at top)."""
    import cv2
    return cv2.Laplacian(gray, cv2.CV_64F)


# ═══════════════════════════════════════════════════════════════════════
# Candidate scoring
# ═══════════════════════════════════════════════════════════════════════

# Garment-aware scoring weights: different garment families need different
# priorities. Structured garments (jackets, blazers) need higher garment
# quality weight. Draped garments (sarees) need higher color coherence.
# Tight garments (bodycon) need higher sharpness.
_GARMENT_WEIGHTS: dict[str, dict[str, float]] = {
    # Structured outerwear: garment geometry matters most
    "jacket":      {"face_quality": 0.30, "garment_quality": 0.45, "sharpness": 0.15, "color_coherence": 0.10},
    "blazer":      {"face_quality": 0.30, "garment_quality": 0.45, "sharpness": 0.15, "color_coherence": 0.10},
    "coat":        {"face_quality": 0.30, "garment_quality": 0.45, "sharpness": 0.15, "color_coherence": 0.10},
    "leather_jacket": {"face_quality": 0.30, "garment_quality": 0.45, "sharpness": 0.15, "color_coherence": 0.10},
    "denim_jacket":   {"face_quality": 0.30, "garment_quality": 0.45, "sharpness": 0.15, "color_coherence": 0.10},
    # Draped garments: color coherence matters most (drape is IP-Adapter driven)
    "saree":       {"face_quality": 0.30, "garment_quality": 0.35, "sharpness": 0.15, "color_coherence": 0.20},
    "sari":        {"face_quality": 0.30, "garment_quality": 0.35, "sharpness": 0.15, "color_coherence": 0.20},
    "lehenga":     {"face_quality": 0.30, "garment_quality": 0.35, "sharpness": 0.15, "color_coherence": 0.20},
    "kimono":      {"face_quality": 0.30, "garment_quality": 0.35, "sharpness": 0.15, "color_coherence": 0.20},
    "abaya":       {"face_quality": 0.30, "garment_quality": 0.35, "sharpness": 0.15, "color_coherence": 0.20},
    # Tight/fitted: sharpness matters (fabric texture must be crisp)
    "bodycon":     {"face_quality": 0.30, "garment_quality": 0.35, "sharpness": 0.25, "color_coherence": 0.10},
    "leggings":    {"face_quality": 0.30, "garment_quality": 0.35, "sharpness": 0.25, "color_coherence": 0.10},
    "cheongsam":   {"face_quality": 0.30, "garment_quality": 0.35, "sharpness": 0.25, "color_coherence": 0.10},
    # Formal: face quality matters more (formal wear = close-up portraits)
    "evening_gown": {"face_quality": 0.40, "garment_quality": 0.35, "sharpness": 0.15, "color_coherence": 0.10},
    "ball_gown":   {"face_quality": 0.40, "garment_quality": 0.35, "sharpness": 0.15, "color_coherence": 0.10},
    "wedding":     {"face_quality": 0.40, "garment_quality": 0.35, "sharpness": 0.15, "color_coherence": 0.10},
}

_DEFAULT_WEIGHTS = {
    "face_quality": 0.35,
    "garment_quality": 0.40,
    "sharpness": 0.15,
    "color_coherence": 0.10,
}


def _get_garment_weights(garment_subtype: str = "") -> dict[str, float]:
    """Get scoring weights adapted to the garment type."""
    key = (garment_subtype or "").strip().lower().replace(" ", "_").replace("-", "_")
    if key in _GARMENT_WEIGHTS:
        return _GARMENT_WEIGHTS[key]
    # Fuzzy match — prefer longest/most-specific match
    best_len = 0
    best_val = _DEFAULT_WEIGHTS
    for geo_key, geo_val in _GARMENT_WEIGHTS.items():
        if key and geo_key in key and len(geo_key) > best_len:
            best_val = geo_val
            best_len = len(geo_key)
    if best_len > 0:
        return best_val
    for geo_key, geo_val in _GARMENT_WEIGHTS.items():
        if key and key in geo_key and len(geo_key) > best_len:
            best_val = geo_val
            best_len = len(geo_key)
    return best_val


def score_candidate(
    original: Image.Image,
    result: Image.Image,
    garment_img: Image.Image,
    mask_np: np.ndarray | None = None,
    protect_np: np.ndarray | None = None,
    schp_labels: np.ndarray | None = None,
    weights: dict[str, float] | None = None,
    garment_subtype: str = "",
) -> ValidationResult:
    """
    Run all validations and compute an aggregate quality score.

    Weights are garment-aware: structured garments prioritize garment quality,
    draped garments prioritize color coherence, tight garments prioritize
    sharpness. Can be overridden via the `weights` parameter.

    Returns a ValidationResult with `score` being the weighted sum.
    """
    w = weights or _get_garment_weights(garment_subtype)

    all_reasons: list[str] = []

    # Face validation (uses SCHP label 13 only when schp_labels available)
    face_quality, identity_drift, face_reasons = validate_face_region(
        original, result, mask_np, protect_np, schp_labels,
    )
    all_reasons.extend(face_reasons)

    # Garment validation
    garment_quality, replacement, color_coherence, garm_reasons = validate_garment_region(
        original, result, garment_img, mask_np,
    )
    all_reasons.extend(garm_reasons)

    # Global sharpness
    sharpness = _sharpness_score(result)

    # Adjusted sharpness score (0..1): var ~100 = sharp, <10 = blurry
    sharpness_score_val = min(1.0, sharpness / 100.0)
    if sharpness < 10.0:
        all_reasons.append("image_blurry")

    aggregate = (
        w["face_quality"] * face_quality
        + w["garment_quality"] * garment_quality
        + w["sharpness"] * sharpness_score_val
        + w["color_coherence"] * color_coherence
    )

    # Only hard-fail on severe issues (identity drift, garment unchanged, blurry)
    # Soft warnings (ghosting, over_smooth, color_drift) reduce score but don't fail
    severe_reasons = [r for r in all_reasons if any(
        kw in r for kw in ("face_severe_distortion", "garment_unchanged", "image_blurry", "face_identity_drift")
    )]
    passed = len(severe_reasons) == 0

    return ValidationResult(
        passed=passed,
        face_quality=round(face_quality, 4),
        garment_quality=round(garment_quality, 4),
        identity_drift=round(identity_drift, 2),
        garment_replacement=round(replacement, 4),
        sharpness=round(sharpness, 2),
        failure_reasons=all_reasons,
        score=round(aggregate, 4),
    )
