"""
GPU worker mask pipeline — SCHP-only binary masks.

SCHP is the single authoritative mask source. All masks are binary.
No feathering, no fusing, no hybrid strategies. The model output is final.

Binary enforcement:
  assert_binary_mask() verifies every mask contains only {0,255} or {0,1}
  before it reaches the diffusion model. Called automatically by
  apply_protection_binary() and at the handler level.
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
_CLOTHING_LABELS = {
    "upper_body": {_LABEL_UPPER_CLOTHES, _LABEL_COAT},
    "lower_body": {_LABEL_SOCKS, _LABEL_PANTS, _LABEL_SKIRT},
    "dresses": {_LABEL_UPPER_CLOTHES, _LABEL_DRESS, _LABEL_COAT, _LABEL_PANTS, _LABEL_JUMPSUITS, _LABEL_SKIRT, _LABEL_SCARF},
}


@dataclass(frozen=True)
class InferenceQualityReport:
    passed: bool
    identity_drift_score: float
    failure_reasons: tuple[str, ...]


def assert_binary_mask(mask: np.ndarray, name: str = "mask") -> None:
    """
    Assert that a mask is binary — values are only {0,255} or {0,1}.

    Raises ValueError if non-binary values are found. This is the
    hard enforcement point that prevents any soft/feathered mask
    from reaching the diffusion model.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    unique = set(int(v) for v in np.unique(mask))
    allowed = [{0, 255}, {0, 1}, {0}, {255}]
    if unique not in allowed:
        raise ValueError(
            f"Non-binary mask detected: {name} has values {unique}. "
            f"Expected only {{0,255}} or {{0,1}}. "
            f"This means a soft/feathered/float mask is in the pipeline."
        )


def build_schp_inpaint_mask(
    schp_labels: np.ndarray,
    cloth_type: str,
) -> np.ndarray:
    """
    Build binary inpaint mask from SCHP labels.
    255 = editable garment region, 0 = protected.
    """
    labels = _CLOTHING_LABELS.get(cloth_type, _CLOTHING_LABELS["dresses"])
    return (np.isin(schp_labels, list(labels)).astype(np.uint8) * 255)


def build_schp_protect_mask(
    schp_labels: np.ndarray,
    cloth_type: str,
    dilate_px: int = 15,
) -> np.ndarray:
    """
    Build binary protect mask from SCHP labels.
    255 = protected (identity-critical), 0 = editable.
    Dilated by dilate_px to prevent edge artifacts.
    """
    labels = _CLOTHING_LABELS.get(cloth_type, _CLOTHING_LABELS["dresses"])
    mask = (~np.isin(schp_labels, list(labels))).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


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
    """
    Subtract protect mask from inpaint mask. Both uint8, 0 or 255.
    Result is binary — no feathering, no gradients.

    Enforces binary input and output via assert_binary_mask.
    """
    assert_binary_mask(inpaint_mask, "inpaint_mask")
    assert_binary_mask(protect_mask, "protect_mask")
    inp = (inpaint_mask > 127).astype(np.uint8)
    prot = (protect_mask > 127).astype(np.uint8)
    result = np.clip(inp - prot, 0, 1).astype(np.uint8) * 255
    assert_binary_mask(result, "final_mask (post apply_protection_binary)")
    return result


def validate_mask_coverage(
    mask: Image.Image,
    cloth_type: str,
    min_coverage: float = 0.04,
) -> dict[str, object]:
    """
    Pre-inference mask sanity check.
    Returns valid=True/False, coverage_percent, reason.
    """
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


def detect_inference_failures(
    original: Image.Image,
    result: Image.Image,
    inpaint_mask: Image.Image,
    protected: Image.Image | None = None,
    *,
    identity_threshold: float = 20.0,
) -> InferenceQualityReport:
    """
    Post-inference QA — triggers retry if identity drifted or garment unchanged.
    Only checks identity drift (face/hair) and missing inpaint.
    """
    orig = np.array(original.convert("RGB"), dtype=np.float32)
    out = np.array(result.convert("RGB"), dtype=np.float32)
    if orig.shape != out.shape:
        out = np.array(result.convert("RGB").resize(original.size, Image.LANCZOS), dtype=np.float32)

    mask_np = np.array(inpaint_mask.convert("L"), dtype=np.uint8)
    if mask_np.shape[:2] != orig.shape[:2]:
        mask_np = np.array(inpaint_mask.convert("L").resize(original.size, Image.NEAREST))

    reasons: list[str] = []
    h = orig.shape[0]

    # Identity drift — restrict to face/hair region (upper 30% of image)
    # to avoid diluting the measurement with unchanged background or limbs.
    face_zone_top = int(0.30 * h)
    if protected is not None:
        prot_arr = np.array(protected.convert("L"), dtype=np.uint8)
        if prot_arr.shape[:2] != orig.shape[:2]:
            prot_arr = np.array(protected.convert("L").resize(orig.shape[1::-1], Image.NEAREST), dtype=np.uint8)
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

    # Missing inpaint — garment region barely changed
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
