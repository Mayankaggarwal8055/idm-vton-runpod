"""
Post-inference quality validation and candidate scoring.

Provides:
  - Face-region validation: detect face distortion / identity drift.
  - Garment-region validation: measure replacement strength, texture detail.
  - Structural similarity: SSIM for perceptual quality.
  - Garment geometry correctness: verify target region was edited.
  - Candidate scoring: aggregate metrics into a single quality score for
    best-candidate selection. Garment-family-aware weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

logger = logging.getLogger("idm-vton.worker.quality")


@dataclass
class ValidationResult:
    """Per-candidate quality assessment with all 9 metrics."""

    passed: bool
    face_quality: float          # 0..1, higher = better
    garment_quality: float       # 0..1, higher = better
    identity_drift: float        # 0..inf, lower = better
    garment_replacement: float   # 0..1, higher = more replacement
    sharpness: float             # 0..inf, higher = sharper
    failure_reasons: list[str] = field(default_factory=list)
    score: float = 0.0           # aggregate weighted score
    # Extended metrics (all 0..1)
    ssim: float = 0.0            # structural similarity
    region_edit: float = 0.0     # correct body region edited
    boundary_quality: float = 0.0  # smooth mask edges
    pose_consistency: float = 0.0  # pose preserved
    geometry_correctness: float = 0.0  # correct garment geometry
    leakage_penalty: float = 0.0  # old garment leakage (lower = better)
    color_coherence: float = 0.0  # color match with garment


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
    # NOTE: color_coherence is counted separately in the aggregate score
    # (line 693), so it must not be included here to avoid double-counting.
    garment_quality = 0.50 * replacement + 0.50 * texture_detail

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


def _ssim_score(original: Image.Image, result: Image.Image, mask_np: np.ndarray | None = None) -> float:
    """Compute structural similarity (SSIM) between original and result.

    When mask_np is provided, computes SSIM only within the inpaint region
    (how well the structure was preserved outside the edit). When not provided,
    computes global SSIM.

    Returns 0..1 where 1 = identical structure.
    """
    import cv2
    orig = np.array(original.convert("L"), dtype=np.float64)
    out = np.array(result.convert("L"), dtype=np.float64)
    if orig.shape != out.shape:
        out = np.array(result.convert("L").resize(original.size, Image.NEAREST), dtype=np.float64)

    # SSIM computation using OpenCV (simplified Wang et al. 2004)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = orig
    img2 = out

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if mask_np is not None:
        # Compute SSIM only in protected region (non-inpaint) — measures
        # how well identity/background was preserved
        if mask_np.shape[:2] != orig.shape[:2]:
            mask_np = np.array(
                Image.fromarray(mask_np).resize(original.size, Image.NEAREST),
                dtype=np.uint8,
            )
        protected = mask_np < 127  # non-inpaint region
        if np.any(protected):
            return float(np.mean(ssim_map[protected]))
    return float(np.mean(ssim_map))


def _region_editing_score(
    original: Image.Image,
    result: Image.Image,
    mask_np: np.ndarray,
    schp_labels: np.ndarray | None = None,
    source_cloth_type: str = "",
    target_cloth_type: str = "",
) -> float:
    """Check that the correct body region was edited.

    Verifies that the inpaint region (mask) was actually modified, and that
    non-inpaint regions were preserved. Returns 0..1 where 1 = correct editing.

    For cross-category swaps, also checks that the source garment region
    was fully erased (high replacement fraction in source labels).
    """
    orig = np.array(original.convert("RGB"), dtype=np.float32)
    out = np.array(result.convert("RGB"), dtype=np.float32)
    if orig.shape != out.shape:
        out = np.array(result.convert("RGB").resize(original.size, Image.LANCZOS), dtype=np.float32)

    h, w = orig.shape[:2]
    if mask_np is None or mask_np.shape[:2] != (h, w):
        return 0.5

    inpaint = mask_np > 127
    protected = ~inpaint

    if not np.any(inpaint):
        return 0.0

    # 1. Replacement strength in inpaint region
    diff = np.mean(np.abs(orig - out), axis=2)
    inpaint_replacement = float(np.mean(diff[inpaint] > 10.0))

    # 2. Preservation in protected region
    if np.any(protected):
        protected_change = float(np.mean(diff[protected] > 10.0))
        # Low change in protected region is good
        preservation = 1.0 - min(1.0, protected_change * 5.0)
    else:
        preservation = 1.0

    # 3. For cross-category: check source garment erasure
    erasure_score = 1.0
    if source_cloth_type and schp_labels is not None:
        source_label_map = {
            "upper_body": {_LABEL_UPPER_CLOTHES},
            "lower_body": {_LABEL_PANTS, _LABEL_SKIRT},
            "dresses": {_LABEL_UPPER_CLOTHES, _LABEL_DRESS,
                       _LABEL_PANTS, _LABEL_SKIRT, _LABEL_SCARF},
            "full_body": {_LABEL_UPPER_CLOTHES, _LABEL_DRESS,
                          _LABEL_PANTS, _LABEL_SKIRT, _LABEL_SCARF},
        }
        src_labels = source_label_map.get(source_cloth_type, set())
        if src_labels and schp_labels.shape[:2] == (h, w):
            src_region = np.isin(schp_labels, list(src_labels))
            if np.any(src_region & inpaint):
                src_diff = diff[src_region & inpaint]
                erasure_score = float(np.mean(src_diff > 10.0))

    # Combine: correct editing = inpaint changed + protected preserved + source erased
    score = 0.40 * inpaint_replacement + 0.30 * preservation + 0.30 * erasure_score
    return max(0.0, min(1.0, score))


def _boundary_quality_score(
    original: Image.Image,
    result: Image.Image,
    mask_np: np.ndarray,
) -> float:
    """Measure boundary quality at the inpaint mask edges.

    Checks that transitions between edited and non-edited regions are smooth,
    not jagged or harsh. Uses gradient magnitude at mask boundaries.

    Returns 0..1 where 1 = smooth, natural boundaries.
    """
    import cv2

    orig = np.array(original.convert("L"), dtype=np.float32)
    out = np.array(result.convert("L"), dtype=np.float32)
    if orig.shape != out.shape:
        out = np.array(result.convert("L").resize(original.size, Image.NEAREST), dtype=np.float32)

    h, w = orig.shape[:2]
    if mask_np.shape[:2] != (h, w):
        return 0.5

    # Find mask boundary pixels (dilate - erode)
    mask_u8 = mask_np.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(mask_u8, kernel, iterations=1)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    boundary = (dilated > 127) != (eroded > 127)

    if not np.any(boundary):
        return 1.0  # no boundary = no boundary artifacts

    # Compute gradient magnitude at boundary in result
    gx = cv2.Sobel(out, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(out, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    boundary_grad = grad_mag[boundary]
    mean_grad = float(np.mean(boundary_grad))

    # High gradient at boundary = sharp/harsh transition (bad)
    # Low gradient = smooth transition (good)
    # Normalize: grad > 50 is harsh, < 15 is smooth
    score = max(0.0, min(1.0, 1.0 - (mean_grad - 15.0) / 35.0))
    return score


def _pose_consistency_score(
    original: Image.Image,
    result: Image.Image,
    protect_np: np.ndarray | None = None,
    schp_labels: np.ndarray | None = None,
) -> float:
    """Verify that pose/limb positions are consistent between original and result.

    Uses edge detection and structural comparison in the protected region
    (face, hair, hands, shoes) to ensure the model didn't hallucinate
    different limb positions.

    Returns 0..1 where 1 = perfect pose consistency.
    """
    import cv2

    orig = np.array(original.convert("L"), dtype=np.uint8)
    out = np.array(result.convert("L"), dtype=np.uint8)
    if orig.shape != out.shape:
        out = np.array(result.convert("L").resize(original.size, Image.NEAREST), dtype=np.uint8)

    # Focus on protected region (identity areas)
    if protect_np is not None:
        if protect_np.shape[:2] == orig.shape[:2]:
            protected = protect_np > 127
        else:
            protected = np.zeros_like(orig, dtype=bool)
            protected[:int(0.25 * orig.shape[0]), :] = True
    else:
        # Top 25% as fallback
        protected = np.zeros_like(orig, dtype=bool)
        protected[:int(0.25 * orig.shape[0]), :] = True

    if not np.any(protected):
        return 1.0

    # Edge detection
    edges_orig = cv2.Canny(orig, 50, 150)
    edges_out = cv2.Canny(out, 50, 150)

    # Compare edges in protected region
    orig_edges_in_protected = edges_orig[protected]
    out_edges_in_protected = edges_out[protected]

    # If both have edges in same positions, pose is preserved
    both_have_edges = np.sum((orig_edges_in_protected > 0) & (out_edges_in_protected > 0))
    either_has_edges = np.sum((orig_edges_in_protected > 0) | (out_edges_in_protected > 0))

    if either_has_edges == 0:
        return 1.0  # no edges to compare

    consistency = both_have_edges / either_has_edges
    return float(min(1.0, consistency))


def _garment_geometry_score(
    result: Image.Image,
    mask_np: np.ndarray,
    target_cloth_type: str = "",
    garment_subtype: str = "",
) -> float:
    """Verify that the generated garment has correct geometry for its type.

    Checks basic geometric properties:
      - Upper body garments should have upper-body-shaped inpaint region
      - Lower body garments should have lower-body-shaped inpaint region
      - Full body garments should have full-body coverage

    Returns 0..1 where 1 = correct geometry.
    """
    if mask_np is None or not target_cloth_type:
        return 0.5

    out = np.array(result.convert("L"), dtype=np.uint8)
    h, w = out.shape[:2]
    if mask_np.shape[:2] != (h, w):
        return 0.5

    inpaint = mask_np > 127
    if not np.any(inpaint):
        return 0.0

    # Compute centroid of inpaint region
    ys, xs = np.where(inpaint)
    centroid_y = np.mean(ys) / h
    centroid_x = np.mean(xs) / w

    # Compute vertical extent
    y_min = np.min(ys) / h
    y_max = np.max(ys) / h
    vertical_span = y_max - y_min

    score = 0.5  # base

    if target_cloth_type == "upper_body":
        # Upper body: centroid should be in top 60%, span should be moderate
        if 0.2 < centroid_y < 0.6:
            score += 0.2
        if 0.2 < vertical_span < 0.7:
            score += 0.15
        # Check it doesn't extend too far into legs
        if y_max < 0.85:
            score += 0.15
    elif target_cloth_type == "lower_body":
        # Lower body: centroid should be in bottom 50%, span moderate
        if 0.4 < centroid_y < 0.9:
            score += 0.2
        if 0.2 < vertical_span < 0.7:
            score += 0.15
        # Check it doesn't extend too far into upper body
        if y_min > 0.15:
            score += 0.15
    elif target_cloth_type in ("dresses", "full_body"):
        # Full body: should span most of the image
        if vertical_span > 0.5:
            score += 0.25
        if 0.15 < centroid_y < 0.6:
            score += 0.15
        if y_min < 0.2:
            score += 0.1

    return max(0.0, min(1.0, score))


# SCHP LIP label constants (must match mask_pipeline.py)
_LABEL_UPPER_CLOTHES = 4
_LABEL_DRESS = 7
_LABEL_PANTS = 6
_LABEL_SKIRT = 5
_LABEL_SCARF = 17


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
    source_cloth_type: str = "",
    target_cloth_type: str = "",
    trace_id: str = "",
) -> ValidationResult:
    """Production-level candidate scoring with 9 quality metrics.

    Metrics:
      1. Face quality — identity preservation
      2. Garment quality — replacement, texture, color
      3. Sharpness — global image sharpness
      4. SSIM — structural similarity
      5. Region editing — correct body region was edited
      6. Boundary quality — smooth mask edges
      7. Pose consistency — limb/face positions preserved
      8. Garment geometry — correct shape for garment type
      9. Garment leakage — old garment suppression

    Returns ValidationResult with aggregate score on 0..1 scale.
    """
    w = weights or _get_garment_weights(garment_subtype)

    all_reasons: list[str] = []

    # ── 1. Face quality ────────────────────────────────────────────────
    face_quality, identity_drift, face_reasons = validate_face_region(
        original, result, mask_np, protect_np, schp_labels,
    )
    all_reasons.extend(face_reasons)

    # ── 2. Garment quality ─────────────────────────────────────────────
    garment_quality, replacement, color_coherence, garm_reasons = validate_garment_region(
        original, result, garment_img, mask_np,
    )
    all_reasons.extend(garm_reasons)

    # ── 3. Sharpness ───────────────────────────────────────────────────
    sharpness = _sharpness_score(result)
    sharpness_score_val = min(1.0, sharpness / 100.0)
    if sharpness < 10.0:
        all_reasons.append("image_blurry")

    # ── 4. SSIM ────────────────────────────────────────────────────────
    ssim_val = _ssim_score(original, result, mask_np)

    # ── 5. Region editing correctness ──────────────────────────────────
    region_edit = _region_editing_score(
        original, result, mask_np, schp_labels, source_cloth_type, target_cloth_type,
    )

    # ── 6. Boundary quality ────────────────────────────────────────────
    boundary_quality = 0.5
    if mask_np is not None:
        boundary_quality = _boundary_quality_score(original, result, mask_np)

    # ── 7. Pose consistency ────────────────────────────────────────────
    pose_consistency = _pose_consistency_score(original, result, protect_np, schp_labels)

    # ── 8. Garment geometry correctness ────────────────────────────────
    geometry_correctness = _garment_geometry_score(
        result, mask_np, target_cloth_type, garment_subtype,
    )

    # ── 9. Garment leakage detection ───────────────────────────────────
    leakage_penalty = 0.0
    if source_cloth_type and mask_np is not None:
        leakage_penalty, leakage_reasons = _detect_garment_leakage(
            original, result, mask_np, source_cloth_type, schp_labels,
        )
        all_reasons.extend(leakage_reasons)

    # ── Aggregate score ────────────────────────────────────────────────
    # Base: garment-family-aware weights (must sum to 1.0)
    aggregate = (
        w["face_quality"] * face_quality
        + w["garment_quality"] * garment_quality
        + w["sharpness"] * sharpness_score_val
        + w["color_coherence"] * color_coherence
    )

    # Bonus metrics — each 0..1, weighted to contribute up to ~0.25 total
    aggregate += 0.05 * ssim_val
    aggregate += 0.10 * region_edit      # most important bonus — correct region edited
    aggregate += 0.04 * boundary_quality
    aggregate += 0.03 * pose_consistency
    aggregate += 0.03 * geometry_correctness

    # Apply leakage penalty (once)
    aggregate = max(0.0, aggregate - leakage_penalty)

    # Clamp to [0, 1]
    aggregate = max(0.0, min(1.0, aggregate))

    # Only hard-fail on severe issues
    severe_reasons = [r for r in all_reasons if any(
        kw in r for kw in ("face_severe_distortion", "garment_unchanged", "image_blurry", "face_identity_drift", "garment_leakage_severe")
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
        ssim=round(ssim_val, 4),
        region_edit=round(region_edit, 4),
        boundary_quality=round(boundary_quality, 4),
        pose_consistency=round(pose_consistency, 4),
        geometry_correctness=round(geometry_correctness, 4),
        leakage_penalty=round(leakage_penalty, 4),
        color_coherence=round(color_coherence, 4),
    )


def _detect_garment_leakage(
    original: Image.Image,
    result: Image.Image,
    mask_np: np.ndarray,
    source_cloth_type: str,
    schp_labels: np.ndarray | None = None,
) -> tuple[float, list[str]]:
    """Detect if old garment residuals persist in the output.

    Compares color histograms of the original and result within the inpaint
    region. High similarity means the old garment leaked through.

    Returns (penalty, reasons).
    """
    reasons: list[str] = []
    penalty = 0.0

    orig = np.array(original.convert("RGB"), dtype=np.uint8)
    out = np.array(result.convert("RGB"), dtype=np.uint8)
    if orig.shape != out.shape:
        out = np.array(result.convert("RGB").resize(original.size, Image.LANCZOS), dtype=np.uint8)

    h, w = orig.shape[:2]
    if mask_np.shape[:2] != (h, w):
        return penalty, reasons

    inpaint = mask_np > 127
    if not np.any(inpaint):
        return penalty, reasons

    # Compare color histograms in inpaint region using chi-squared distance
    import cv2
    orig_hist = cv2.calcHist([orig], [0, 1, 2], inpaint.astype(np.uint8), [8, 8, 8], [0, 256, 0, 256, 0, 256])
    out_hist = cv2.calcHist([out], [0, 1, 2], inpaint.astype(np.uint8), [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(orig_hist, orig_hist)
    cv2.normalize(out_hist, out_hist)
    hist_similarity = float(cv2.compareHist(orig_hist, out_hist, cv2.HISTCMP_CORREL))

    # High correlation = old garment leaked (colors are too similar)
    # Tiered penalties (mutually exclusive)
    if hist_similarity > 0.92:
        penalty += 0.15
        reasons.append(f"garment_leakage_color:{hist_similarity:.3f}")
        reasons.append("garment_leakage_severe")
    elif hist_similarity > 0.85:
        penalty += 0.10
        reasons.append(f"garment_leakage_color:{hist_similarity:.3f}")
    elif hist_similarity > 0.75:
        penalty += 0.03

    # Check for duplicate garment regions (repeated pattern = hallucination)
    # NOTE: Original check compared inpaint mask area vs garment pixel count,
    # which are fundamentally incomparable. Removed until proper output-SCHP
    # comparison is implemented.

    return penalty, reasons
