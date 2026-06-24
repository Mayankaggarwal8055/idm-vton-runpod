"""
GPU worker mask pipeline — SCHP-only binary masks.

SCHP is the single authoritative mask source. All masks are binary.
No feathering, no fusing, no hybrid strategies. The model output is final.

Design principles (quality-first):
  - Inpaint mask = SCHP clothing labels for cloth_type, aggressively dilated.
  - Protect mask = identity-critical regions only (face, hair, hands, shoes),
    NOT the inverted clothing mask (which shrinks editable area and blocks
    saree/dupatta drape over arms).
  - Draped garments (saree, dupatta, lehenga) include arm regions in inpaint
    but protect hand endpoints so mehndi / phones stay intact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("idm-vton.worker.mask")

# SCHP 20-class ATR label constants
_LABEL_BG = 0
_LABEL_HAT = 1
_LABEL_HAIR = 2
_LABEL_GLOVE = 3
_LABEL_SUNGLASSES = 4
_LABEL_UPPER_CLOTHES = 5
_LABEL_DRESS = 6
_LABEL_COAT = 7
_LABEL_SOCKS = 8
_LABEL_PANTS = 9
_LABEL_JUMPSUITS = 10
_LABEL_SCARF = 11
_LABEL_SKIRT = 12
_LABEL_FACE = 13
_LABEL_LEFT_ARM = 14
_LABEL_RIGHT_ARM = 15
_LABEL_LEFT_LEG = 16
_LABEL_RIGHT_LEG = 17
_LABEL_LEFT_SHOE = 18
_LABEL_RIGHT_SHOE = 19

# Clothing label sets per cloth_type
_DRESSES_LABELS = {
    _LABEL_UPPER_CLOTHES,
    _LABEL_DRESS,
    _LABEL_COAT,
    _LABEL_PANTS,
    _LABEL_JUMPSUITS,
    _LABEL_SKIRT,
    _LABEL_SCARF,
}
_CLOTHING_LABELS = {
    "upper_body": {_LABEL_UPPER_CLOTHES, _LABEL_COAT},
    "lower_body": {_LABEL_SOCKS, _LABEL_PANTS, _LABEL_SKIRT},
    "dresses": _DRESSES_LABELS,
    "full_body": _DRESSES_LABELS,
}

# All garment labels for cross-category mismatch detection.
# Used to detect when the person's current garment has labels that fall
# outside the target cloth_type's editable mask.
_ALL_GARMENT_LABELS = {
    _LABEL_UPPER_CLOTHES,
    _LABEL_DRESS,
    _LABEL_COAT,
    _LABEL_SOCKS,
    _LABEL_PANTS,
    _LABEL_JUMPSUITS,
    _LABEL_SCARF,
    _LABEL_SKIRT,
}

_IDENTITY_PROTECT_LABELS = {
    _LABEL_HAIR,
    _LABEL_FACE,
    _LABEL_HAT,
    _LABEL_GLOVE,
    _LABEL_SUNGLASSES,
    _LABEL_LEFT_SHOE,
    _LABEL_RIGHT_SHOE,
}

_DRAPE_ARM_LABELS = (_LABEL_LEFT_ARM, _LABEL_RIGHT_ARM)
_DRAPE_KEYWORDS = (
    "saree", "sari", "dupatta", "lehenga", "drape", "draped",
    "pallu", "shawl", "wrap", "anarkali", "ethnic",
)

# Full-body mask labels for bare-skin inpaint (Stage 0).
# Covers ALL garment labels + arms — everything that could be clothing.
_FULL_BODY_INPAINT_LABELS = {
    _LABEL_UPPER_CLOTHES,  # 5
    _LABEL_DRESS,          # 6
    _LABEL_COAT,           # 7
    _LABEL_SOCKS,          # 8
    _LABEL_PANTS,          # 9
    _LABEL_JUMPSUITS,      # 10
    _LABEL_SCARF,          # 11
    _LABEL_SKIRT,          # 12
    _LABEL_LEFT_ARM,       # 14
    _LABEL_RIGHT_ARM,      # 15
}


@dataclass(frozen=True)
class InferenceQualityReport:
    passed: bool
    identity_drift_score: float
    failure_reasons: tuple[str, ...]


def is_draped_garment(cloth_type: str, garment_subtype: str = "") -> bool:
    """True when the garment needs arm-span inpaint (saree pallu, dupatta, etc.)."""
    ct = (cloth_type or "").strip().lower()
    if ct not in ("dresses", "full_body"):
        return False
    subtype = (garment_subtype or "").strip().lower()
    if any(kw in subtype for kw in _DRAPE_KEYWORDS):
        return True
    return False


def needs_two_stage(
    schp_np: np.ndarray,
    cloth_type: str,
    uncovered_threshold: float = 0.08,
) -> bool:
    """
    Detect whether the person's current garment spans garment-label categories
    that the target cloth_type's mask would NOT cover.

    This is the root-cause check for cross-category failure.

    Example: person is wearing a saree (SCHP labels 6=DRESS, 11=SCARF) but
    target is upper_body (mask labels {5=UPPER_CLOTHES, 7=COAT}).
    The saree body (6) and pallu (11) are outside the upper_body mask,
    so they would survive the try-on → two-stage is needed.

    Returns True when uncovered garment-label area exceeds threshold
    fraction of the image.  False for same-category swaps that the
    single-stage mask already covers.
    """
    target_labels = _CLOTHING_LABELS.get(cloth_type, _CLOTHING_LABELS["dresses"])
    present = set(int(v) for v in np.unique(schp_np)) & _ALL_GARMENT_LABELS
    uncovered = present - target_labels
    if not uncovered:
        return False

    h, w = schp_np.shape
    uncovered_px = sum(int(np.sum(schp_np == lbl)) for lbl in uncovered)
    uncovered_frac = uncovered_px / max(h * w, 1)
    return uncovered_frac > uncovered_threshold


def assert_binary_mask(mask: np.ndarray, name: str = "mask") -> None:
    """Assert mask values are only {0,255} or {0,1}."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    unique = set(int(v) for v in np.unique(mask))
    allowed = [{0, 255}, {0, 1}, {0}, {255}]
    if unique not in allowed:
        raise ValueError(
            f"Non-binary mask detected: {name} has values {unique}. "
            f"Expected only {{0,255}} or {{0,1}}."
        )


def _hand_zones_from_arms(
    schp_labels: np.ndarray,
    arm_labels: tuple[int, ...] = _DRAPE_ARM_LABELS,
    hand_fraction: float = 0.38,
) -> np.ndarray:
    """
    Protect only the distal portion of each arm (hands/wrists), not the full arm.
    Enables sheer dupatta/saree drape over forearms while keeping hands intact.
    """
    h, w = schp_labels.shape
    protect = np.zeros((h, w), dtype=np.uint8)
    y_idx = np.arange(h)[:, None]

    for label in arm_labels:
        arm_mask = schp_labels == label
        if not np.any(arm_mask):
            continue
        ys = np.where(arm_mask)[0]
        y_min, y_max = int(ys.min()), int(ys.max())
        span = max(1, y_max - y_min)
        hand_y_start = y_max - int(span * hand_fraction)
        hand_zone = arm_mask & (y_idx >= hand_y_start)
        protect[hand_zone] = 255

    return protect


def _dilate_mask(mask: np.ndarray, kernel_size: int, iterations: int = 1) -> np.ndarray:
    k = max(3, kernel_size)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel, iterations=iterations)


def build_schp_inpaint_mask(
    schp_labels: np.ndarray,
    cloth_type: str,
    garment_subtype: str = "",
) -> np.ndarray:
    """
    Build binary inpaint mask from SCHP labels.
    255 = editable garment region, 0 = protected.
    """
    labels = _CLOTHING_LABELS.get(cloth_type, _CLOTHING_LABELS["dresses"])
    mask = (np.isin(schp_labels, list(labels)).astype(np.uint8) * 255)

    draped = is_draped_garment(cloth_type, garment_subtype)
    if draped:
        arm_mask = np.isin(schp_labels, list(_DRAPE_ARM_LABELS)).astype(np.uint8) * 255
        mask = np.maximum(mask, arm_mask)

    return mask


def build_full_body_inpaint_mask(schp_labels: np.ndarray) -> np.ndarray:
    """
    Full-body mask for bare-skin inpaint (Stage 0 of cross-category pipeline).

    Covers ALL garment labels + arms — everything that could be clothing.
    Unlike build_schp_inpaint_mask which is category-specific, this is
    category-agnostic: it marks every pixel that could be garment as editable.

    The protect mask for identity (face, hair, shoes) is applied separately
    in build_final_inpaint_mask.
    """
    mask = np.isin(schp_labels, list(_FULL_BODY_INPAINT_LABELS)).astype(np.uint8) * 255
    return mask


def build_schp_protect_mask(
    schp_labels: np.ndarray,
    cloth_type: str,
    garment_subtype: str = "",
    dilate_px: int = 13,
) -> np.ndarray:
    """
    Build binary protect mask from SCHP labels.
    255 = protected (identity-critical), 0 = editable.

    Uses explicit identity labels — NOT inverted clothing — so inpaint coverage
    stays large enough for full outfit replacement and draped overlays.
    """
    draped = is_draped_garment(cloth_type, garment_subtype)
    mask = np.isin(schp_labels, list(_IDENTITY_PROTECT_LABELS)).astype(np.uint8) * 255

    if cloth_type == "upper_body":
        # Block lower-body replacement when only swapping tops.
        lower_labels = {_LABEL_PANTS, _LABEL_SKIRT, _LABEL_SOCKS, _LABEL_LEFT_LEG, _LABEL_RIGHT_LEG}
        mask = np.maximum(mask, np.isin(schp_labels, list(lower_labels)).astype(np.uint8) * 255)
        # Full arms protected for upper_body (short sleeves replace arm clothing only inside torso mask).
        mask = np.maximum(mask, np.isin(schp_labels, list(_DRAPE_ARM_LABELS)).astype(np.uint8) * 255)
    elif cloth_type == "lower_body":
        upper_labels = {_LABEL_UPPER_CLOTHES, _LABEL_COAT, _LABEL_DRESS, _LABEL_SCARF}
        mask = np.maximum(mask, np.isin(schp_labels, list(upper_labels)).astype(np.uint8) * 255)
        mask = np.maximum(mask, np.isin(schp_labels, list(_DRAPE_ARM_LABELS)).astype(np.uint8) * 255)
    elif draped:
        # Drape over forearms — protect hands only.
        mask = np.maximum(mask, _hand_zones_from_arms(schp_labels))
    else:
        # Standard full-body dress: protect full arms (no drape over arms).
        mask = np.maximum(mask, np.isin(schp_labels, list(_DRAPE_ARM_LABELS)).astype(np.uint8) * 255)

    if draped:
        dilate_px = max(9, dilate_px - 4)

    mask = _dilate_mask(mask, dilate_px, iterations=1)
    return mask


def dilate_inpaint_mask(
    inpaint_mask: np.ndarray,
    cloth_type: str,
    garment_subtype: str = "",
    schp_height: int = 512,
) -> np.ndarray:
    """
    Contour-aware dilation scaled to SCHP resolution.
    Dresses/draped garments get extra expansion for full replacement.
    """
    scale = schp_height / 512.0
    draped = is_draped_garment(cloth_type, garment_subtype)

    if cloth_type in ("lower_body", "dresses", "full_body"):
        leg_ks = (max(3, int(13 * scale)), max(3, int(19 * scale)))
        leg_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, leg_ks)
        iterations = 2 if draped else 1
        return cv2.dilate(inpaint_mask, leg_k, iterations=iterations)

    mild_ks = (max(3, int(17 * scale)), max(3, int(11 * scale)))
    mild_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, mild_ks)
    return cv2.dilate(inpaint_mask, mild_k, iterations=1)


def build_final_full_body_mask(
    schp_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full-body mask pipeline for bare-skin inpaint.

    Inpaint = all garment labels + arms (full body coverage).
    Protect = identity only (face, hair, hat, gloves, sunglasses, shoes).
    Final = inpaint − protect.

    Returns (final_mask, inpaint_dilated, protect_mask).
    """
    inpaint_raw = build_full_body_inpaint_mask(schp_labels)

    # Identity-only protect: no lower body, no arm blocking — just face/hair/accessories
    protect = np.isin(schp_labels, list(_IDENTITY_PROTECT_LABELS)).astype(np.uint8) * 255

    # Dilate inpaint aggressively for full-body coverage
    scale = schp_labels.shape[0] / 512.0
    ks = (max(3, int(15 * scale)), max(3, int(19 * scale)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ks)
    inpaint_dilated = cv2.dilate(inpaint_raw, k, iterations=2)

    final = apply_protection_binary(inpaint_dilated, protect)
    return final, inpaint_dilated, protect


def build_final_inpaint_mask(
    schp_labels: np.ndarray,
    cloth_type: str,
    garment_subtype: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full mask pipeline: inpaint → dilate → subtract identity protect.
    Returns (final_mask, inpaint_dilated, protect_mask).
    """
    inpaint_raw = build_schp_inpaint_mask(schp_labels, cloth_type, garment_subtype)
    protect = build_schp_protect_mask(schp_labels, cloth_type, garment_subtype)
    inpaint_dilated = dilate_inpaint_mask(
        inpaint_raw, cloth_type, garment_subtype, schp_height=schp_labels.shape[0],
    )
    final = apply_protection_binary(inpaint_dilated, protect)
    return final, inpaint_dilated, protect


def validate_mask_integrity(mask: np.ndarray, name: str = "mask") -> None:
    """Validate mask is 2D, non-empty, binary-compatible, and non-trivial."""
    if mask.ndim != 2:
        raise ValueError(f"Mask '{name}': expected 2D, got {mask.ndim}D shape {mask.shape}")
    h, w = mask.shape
    if h < 10 or w < 10:
        raise ValueError(f"Mask '{name}': degenerate shape {mask.shape}")
    unique = set(int(v) for v in np.unique(mask))
    allowed = [{0, 255}, {0, 1}, {0}, {255}]
    if unique not in allowed:
        raise ValueError(f"Mask '{name}': non-binary values {unique}")
    nonzero = int(np.count_nonzero(mask > 127))
    total = h * w
    if nonzero == 0:
        raise ValueError(f"Mask '{name}': completely empty — no editable pixels")
    if nonzero == total:
        raise ValueError(f"Mask '{name}': completely full — no protected pixels remain")


def apply_protection_binary(inpaint_mask: np.ndarray, protect_mask: np.ndarray) -> np.ndarray:
    """Subtract protect mask from inpaint mask. Both uint8, 0 or 255."""
    assert_binary_mask(inpaint_mask, "inpaint_mask")
    assert_binary_mask(protect_mask, "protect_mask")
    inp = (inpaint_mask > 127).astype(np.int16)
    prot = (protect_mask > 127).astype(np.int16)
    result = np.clip(inp - prot, 0, 1).astype(np.uint8) * 255
    assert_binary_mask(result, "final_mask (post apply_protection_binary)")
    return result


def validate_mask_coverage(
    mask: Image.Image,
    cloth_type: str,
    min_coverage: float = 0.04,
) -> dict[str, object]:
    """Pre-inference mask sanity check."""
    mask_np = np.array(mask.convert("L"), dtype=np.uint8)
    h, w = mask_np.shape[:2]
    binary = (mask_np > 127).astype(np.uint8)
    coverage = float(np.sum(binary)) / binary.size

    if coverage < min_coverage:
        return {
            "valid": False,
            "coverage_percent": round(coverage * 100.0, 2),
            "reason": f"mask_too_small:{coverage*100:.1f}%",
        }

    if cloth_type in ("lower_body", "dresses", "full_body"):
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


def detect_inference_failures(
    original: Image.Image,
    result: Image.Image,
    inpaint_mask: Image.Image,
    protected: Image.Image | None = None,
    *,
    identity_threshold: float = 20.0,
) -> InferenceQualityReport:
    """Post-inference QA — triggers retry if identity drifted or garment unchanged."""
    orig = np.array(original.convert("RGB"), dtype=np.float32)
    out = np.array(result.convert("RGB"), dtype=np.float32)
    if orig.shape != out.shape:
        out = np.array(result.convert("RGB").resize(original.size, Image.LANCZOS), dtype=np.float32)

    mask_np = np.array(inpaint_mask.convert("L"), dtype=np.uint8)
    if mask_np.shape[:2] != orig.shape[:2]:
        mask_np = np.array(inpaint_mask.convert("L").resize(original.size, Image.NEAREST))

    reasons: list[str] = []
    h = orig.shape[0]
    face_zone_top = int(0.30 * h)

    if protected is not None:
        prot_arr = np.array(protected.convert("L"), dtype=np.uint8)
        if prot_arr.shape[:2] != orig.shape[:2]:
            prot_arr = np.array(
                protected.convert("L").resize(orig.shape[1::-1], Image.NEAREST),
                dtype=np.uint8,
            )
        upper_mask = np.zeros_like(prot_arr, dtype=bool)
        upper_mask[:face_zone_top, :] = True
        prot_mask = (prot_arr > 127) & upper_mask
        if np.any(prot_mask):
            face_diff = float(np.mean(np.abs(orig[prot_mask] - out[prot_mask])))
        else:
            face_diff = float(np.mean(np.abs(orig[:face_zone_top, :] - out[:face_zone_top, :])))
    else:
        face_diff = float(np.mean(np.abs(orig[:face_zone_top, :] - out[:face_zone_top, :])))
    identity_drift = face_diff
    if identity_drift > identity_threshold:
        reasons.append(f"identity_drift:{identity_drift:.1f}")

    inpaint_region = mask_np > 127
    if np.any(inpaint_region):
        diff_inpaint = np.mean(np.abs(orig - out), axis=2)
        unchanged = float(np.mean(diff_inpaint[inpaint_region] < 10.0))
        if unchanged > 0.45:
            reasons.append(f"original_clothing_visible:{unchanged:.2f}")

    passed = len(reasons) == 0
    return InferenceQualityReport(
        passed=passed,
        identity_drift_score=identity_drift,
        failure_reasons=tuple(reasons),
    )
