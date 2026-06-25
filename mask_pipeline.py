"""
GPU worker mask pipeline — SCHP semantic masks with source+target union masks.

SCHP is the single authoritative mask source. All masks are binary.
No feathering, no fusing, no hybrid strategies. The model output is final.

Design principles (quality-first):
  - Same-category: mask = target garment labels only. Model changes texture/color
    within the source garment's shape.
  - Cross-category: mask = UNION of source + target garment labels, plus an
    adaptive dilated buffer around source labels. This ensures the model can
    fully erase the old garment while generating the new one in the correct
    body region.
  - Protect mask = identity-critical regions (face, hair, hands, shoes) plus
    non-target body regions (lower body protected when only swapping tops).
  - Draped garments (saree, dupatta, lehenga) use hand-only arm protection
    so pallu/drape can be generated over forearms.
  - Buffer dilation is adaptive: scales to garment size and body proportions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

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

# ── Garment-aware geometry taxonomy ───────────────────────────────────
# Maps garment subtypes to their geometric properties. This drives
# target-aware mask expansion: when the target garment covers different
# body regions than the source, the mask must expand to include them.
#
# expansion_down: extra pixels below the source garment's lowest point
# expansion_up: extra pixels above the source garment's highest point
# expansion_width: extra pixels on each side
# expose_arms: if True, arms are NOT protected (tank tops, sleeveless)
# protect_lower: if True, lower body is protected (upper-only swaps)
# protect_upper: if True, upper body is protected (lower-only swaps)
# body_region: which body regions the target garment covers

@dataclass(frozen=True)
class GarmentGeometry:
    expansion_down: int = 0
    expansion_up: int = 0
    expansion_width: int = 0
    expose_arms: bool = False
    protect_lower: bool = True
    protect_upper: bool = False
    body_region: str = "upper"  # upper, lower, full, draped

GARMENT_GEOMETRY: dict[str, GarmentGeometry] = {
    # ── Upper body: short / fitted ────────────────────────────────────
    "tshirt":     GarmentGeometry(body_region="upper", protect_lower=True),
    "t_shirt":    GarmentGeometry(body_region="upper", protect_lower=True),
    "polo":       GarmentGeometry(body_region="upper", protect_lower=True),
    "shirt":      GarmentGeometry(body_region="upper", protect_lower=True),
    "blouse":     GarmentGeometry(body_region="upper", protect_lower=True),
    "sweatshirt": GarmentGeometry(body_region="upper", protect_lower=True),
    "sports_jersey": GarmentGeometry(body_region="upper", protect_lower=True),
    # ── Upper body: sleeveless / exposed ──────────────────────────────
    "tank_top":   GarmentGeometry(body_region="upper", protect_lower=True, expose_arms=True),
    "crop_top":   GarmentGeometry(body_region="upper", protect_lower=True, expose_arms=True),
    "camisole":   GarmentGeometry(body_region="upper", protect_lower=True, expose_arms=True),
    "vest":       GarmentGeometry(body_region="upper", protect_lower=True, expose_arms=True),
    "corset":     GarmentGeometry(body_region="upper", protect_lower=True, expose_arms=True),
    # ── Upper body: extended / long ───────────────────────────────────
    "sweater":    GarmentGeometry(expansion_down=40, body_region="upper", protect_lower=True),
    "hoodie":     GarmentGeometry(expansion_down=40, expansion_up=30, body_region="upper", protect_lower=True),
    "jacket":     GarmentGeometry(expansion_down=80, body_region="upper", protect_lower=True),
    "blazer":     GarmentGeometry(expansion_down=80, body_region="upper", protect_lower=True),
    "coat":       GarmentGeometry(expansion_down=160, body_region="upper", protect_lower=True),
    "cardigan":   GarmentGeometry(expansion_down=80, body_region="upper", protect_lower=True),
    "leather_jacket": GarmentGeometry(expansion_down=80, body_region="upper", protect_lower=True),
    "denim_jacket":   GarmentGeometry(expansion_down=80, body_region="upper", protect_lower=True),
    # ── Upper body: wide / flowing ────────────────────────────────────
    "poncho":     GarmentGeometry(expansion_down=60, expansion_width=80, body_region="upper", protect_lower=True),
    "cape":       GarmentGeometry(expansion_down=80, expansion_width=80, body_region="upper", protect_lower=True),
    "shrug":      GarmentGeometry(expansion_width=40, body_region="upper", protect_lower=True),
    # ── Lower body ────────────────────────────────────────────────────
    "jeans":      GarmentGeometry(body_region="lower", protect_upper=True),
    "trousers":   GarmentGeometry(body_region="lower", protect_upper=True),
    "pants":      GarmentGeometry(body_region="lower", protect_upper=True),
    "shorts":     GarmentGeometry(body_region="lower", protect_upper=True),
    "skirt":      GarmentGeometry(body_region="lower", protect_upper=True),
    "mini_skirt": GarmentGeometry(body_region="lower", protect_upper=True),
    "long_skirt": GarmentGeometry(body_region="lower", protect_upper=True),
    "leggings":   GarmentGeometry(body_region="lower", protect_upper=True),
    "joggers":    GarmentGeometry(body_region="lower", protect_upper=True),
    "cargo_pants": GarmentGeometry(body_region="lower", protect_upper=True),
    "wide_leg":   GarmentGeometry(body_region="lower", protect_upper=True, expansion_width=30),
    "palazzo":    GarmentGeometry(body_region="lower", protect_upper=True, expansion_width=60),
    "dhoti_pants": GarmentGeometry(body_region="lower", protect_upper=True),
    # ── Full body: standard ───────────────────────────────────────────
    "dress":      GarmentGeometry(body_region="full"),
    "mini_dress": GarmentGeometry(body_region="full"),
    "midi_dress": GarmentGeometry(body_region="full"),
    "maxi_dress": GarmentGeometry(body_region="full"),
    "bodycon":    GarmentGeometry(body_region="full"),
    "a_line":     GarmentGeometry(body_region="full", expansion_width=20),
    "jumpsuit":   GarmentGeometry(body_region="full"),
    # ── Full body: extended ───────────────────────────────────────────
    "evening_gown": GarmentGeometry(expansion_down=60, body_region="full"),
    "ball_gown":  GarmentGeometry(expansion_down=60, expansion_width=80, body_region="full"),
    "wedding":    GarmentGeometry(expansion_down=80, expansion_width=60, body_region="full"),
    "maxi":       GarmentGeometry(body_region="full"),
    "wrap_dress": GarmentGeometry(body_region="full"),
    "off_shoulder": GarmentGeometry(expansion_up=20, body_region="full"),
    "one_shoulder": GarmentGeometry(body_region="full"),
    "strap":      GarmentGeometry(body_region="full"),
    # ── Traditional: draped ───────────────────────────────────────────
    "saree":      GarmentGeometry(body_region="draped", expansion_width=40),
    "sari":       GarmentGeometry(body_region="draped", expansion_width=40),
    "lehenga":    GarmentGeometry(body_region="draped", expansion_width=40),
    "ghagra":     GarmentGeometry(body_region="draped", expansion_width=40),
    "dupatta":    GarmentGeometry(body_region="draped", expansion_width=60),
    "shawl":      GarmentGeometry(body_region="draped", expansion_width=60),
    "anarkali":   GarmentGeometry(body_region="draped", expansion_down=40),
    "salwar_suit": GarmentGeometry(body_region="draped"),
    "kurti":      GarmentGeometry(body_region="full"),
    "kurta":      GarmentGeometry(body_region="full"),
    # ── Traditional: structured ───────────────────────────────────────
    "sherwani":   GarmentGeometry(expansion_down=80, body_region="full"),
    "abaya":      GarmentGeometry(body_region="full", expansion_width=40),
    "kaftan":     GarmentGeometry(body_region="full", expansion_width=60),
    "jalabiya":   GarmentGeometry(body_region="full", expansion_width=40),
    "kimono":     GarmentGeometry(body_region="full", expansion_width=80),
    "hanbok":     GarmentGeometry(body_region="full", expansion_width=40),
    "cheongsam":  GarmentGeometry(body_region="full"),
    "qipao":      GarmentGeometry(body_region="full"),
    "yukata":     GarmentGeometry(body_region="full", expansion_width=60),
    "dhoti":      GarmentGeometry(body_region="draped"),
    "lungi":      GarmentGeometry(body_region="draped"),
    # ── Layered (treat as outer layer) ────────────────────────────────
    "shirt_under_jacket": GarmentGeometry(expansion_down=80, body_region="upper", protect_lower=True),
    "hoodie_under_jacket": GarmentGeometry(expansion_down=80, expansion_up=30, body_region="upper", protect_lower=True),
    "dress_under_coat": GarmentGeometry(expansion_down=120, body_region="full"),
    "saree_with_shawl": GarmentGeometry(body_region="draped", expansion_width=80),
    "dupatta_over_kurti": GarmentGeometry(body_region="draped", expansion_width=60),
    "any_with_scarf": GarmentGeometry(body_region="draped", expansion_width=40),
}


def get_garment_geometry(garment_subtype: str) -> GarmentGeometry:
    """Look up geometric properties for a garment subtype.

    Falls back to cloth_type-based defaults if subtype is not in the taxonomy.
    """
    key = (garment_subtype or "").strip().lower().replace(" ", "_").replace("-", "_")
    if key in GARMENT_GEOMETRY:
        return GARMENT_GEOMETRY[key]
    # Fuzzy match: prefer longest/most-specific match
    # Direction A: taxonomy key is substring of input (e.g. "saree" in "saree_with_shawl")
    best_a = None
    best_a_len = 0
    for geo_key, geo_val in GARMENT_GEOMETRY.items():
        if key and geo_key in key and len(geo_key) > best_a_len:
            best_a = geo_val
            best_a_len = len(geo_key)
    if best_a:
        return best_a
    # Direction B: input is substring of taxonomy key (e.g. "jacket" in "bomber_jacket")
    best_b = None
    best_b_len = 0
    for geo_key, geo_val in GARMENT_GEOMETRY.items():
        if key and key in geo_key and len(geo_key) > best_b_len:
            best_b = geo_val
            best_b_len = len(geo_key)
    if best_b:
        return best_b
    return GarmentGeometry()  # conservative defaults


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


def detect_source_cloth_type(schp_np: np.ndarray) -> str:
    """Detect what the person is currently wearing from SCHP labels.

    Returns one of: "upper_body", "lower_body", "dresses", "unknown".
    This is used for source-aware mask expansion: when the target garment
    covers different body regions than the source, the mask must expand.
    """
    h, w = schp_np.shape
    total = h * w

    # Count pixels per label
    label_counts = {}
    for lbl in range(20):
        count = int(np.sum(schp_np == lbl))
        if count > 0:
            label_counts[lbl] = count

    garment_px = sum(label_counts.get(lbl, 0) for lbl in _ALL_GARMENT_LABELS)
    if garment_px == 0:
        return "unknown"

    # Check for coat/outerwear (label 7) — high confidence indicator
    coat_px = label_counts.get(_LABEL_COAT, 0)
    if coat_px / max(garment_px, 1) > 0.15:
        return "upper_body"

    # Check for dress (label 6) — covers most of body
    dress_px = label_counts.get(_LABEL_DRESS, 0)
    if dress_px / max(garment_px, 1) > 0.30:
        return "dresses"

    # Check for scarf (label 11) — indicates draped garment
    scarf_px = label_counts.get(_LABEL_SCARF, 0)
    if scarf_px / max(garment_px, 1) > 0.10:
        return "dresses"

    # Check for pants (label 9) — lower body dominant
    pants_px = label_counts.get(_LABEL_PANTS, 0)
    skirt_px = label_counts.get(_LABEL_SKIRT, 0)
    lower_px = pants_px + skirt_px + label_counts.get(_LABEL_SOCKS, 0)
    if lower_px / max(garment_px, 1) > 0.40:
        return "lower_body"

    # Check for upper clothes (label 5) — upper body dominant
    upper_px = label_counts.get(_LABEL_UPPER_CLOTHES, 0)
    if upper_px / max(garment_px, 1) > 0.40:
        return "upper_body"

    # Check for jumpsuit (label 10) — full body
    jumpsuit_px = label_counts.get(_LABEL_JUMPSUITS, 0)
    if jumpsuit_px / max(garment_px, 1) > 0.20:
        return "dresses"

    # Default: if mostly upper clothing, assume upper_body
    if upper_px >= dress_px and upper_px >= lower_px:
        return "upper_body"
    if lower_px >= upper_px and lower_px >= dress_px:
        return "lower_body"
    return "dresses"


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


# ── Garment image geometry analysis ────────────────────────────────────
@dataclass(frozen=True)
class GarmentImageInfo:
    """Geometry extracted from the target garment image."""
    bbox_area_ratio: float = 0.0
    aspect_ratio: float = 1.0
    width_ratio: float = 0.0
    height_ratio: float = 0.0
    center_y_ratio: float = 0.5
    has_sleeves: bool = False
    is_long: bool = False
    is_wide: bool = False


def analyze_garment_image(garment_img: Image.Image) -> GarmentImageInfo:
    """Extract geometry from the target garment reference image.

    Uses contour analysis on the non-white region to estimate bounding box
    coverage, aspect ratio, and sleeve/length/width hints.
    """
    arr = np.array(garment_img.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]

    is_white = np.all(arr > 240, axis=2)
    fg = (~is_white).astype(np.uint8) * 255

    if not np.any(fg):
        return GarmentImageInfo()

    ys, xs = np.where(fg > 0)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1

    bbox_w = x2 - x1
    bbox_h = y2 - y1
    bbox_area = bbox_w * bbox_h
    total_area = max(h * w, 1)

    width_ratio = bbox_w / max(w, 1)
    height_ratio = bbox_h / max(h, 1)
    aspect_ratio = bbox_w / max(bbox_h, 1)
    center_y_ratio = ((y1 + y2) / 2.0) / max(h, 1)

    left_col = int(0.15 * w)
    right_col = int(0.85 * w)
    left_fg = bool(np.any(fg[:, :left_col] > 0))
    right_fg = bool(np.any(fg[:, right_col:] > 0))
    has_sleeves = left_fg and right_fg

    return GarmentImageInfo(
        bbox_area_ratio=round(bbox_area / total_area, 4),
        aspect_ratio=round(aspect_ratio, 3),
        width_ratio=round(width_ratio, 3),
        height_ratio=round(height_ratio, 3),
        center_y_ratio=round(center_y_ratio, 3),
        has_sleeves=has_sleeves,
        is_long=height_ratio > 0.50,
        is_wide=width_ratio > 0.60,
    )


# ── Adaptive buffer dilation ────────────────────────────────────────────
def _adaptive_buffer_ks(
    schp_labels: np.ndarray,
    source_labels: set,
    garment_img_info: "GarmentImageInfo | None" = None,
) -> int:
    """Compute adaptive dilation kernel size for cross-category source buffer.

    Scales to body size, source garment coverage, and target garment geometry.
    """
    h, w = schp_labels.shape
    scale = max(1.0, h / 512.0)

    base_ks = int(12 * scale)

    if source_labels:
        source_px = sum(int(np.sum(schp_labels == lbl)) for lbl in source_labels)
        source_frac = source_px / max(h * w, 1)
        if source_frac > 0.15:
            base_ks = int(16 * scale)
        elif source_frac > 0.08:
            base_ks = int(14 * scale)
        else:
            base_ks = int(10 * scale)

    if garment_img_info:
        if garment_img_info.is_wide:
            base_ks = int(base_ks * 1.2)
        if garment_img_info.is_long:
            base_ks = int(base_ks * 1.1)

    ks = max(5, min(base_ks, int(25 * scale)))
    if ks % 2 == 0:
        ks += 1
    return ks


# ── Garment family routing ─────────────────────────────────────────────
_FAMILY_UPPER_STRUCTURED = frozenset({
    "jacket", "blazer", "coat", "leather_jacket", "denim_jacket",
    "cardigan", "windbreaker", "trench", "peacoat", "overcoat",
})
_FAMILY_UPPER_FITTED = frozenset({
    "tshirt", "t_shirt", "shirt", "polo", "blouse", "sweatshirt",
    "sports_jersey", "henley",
})
_FAMILY_UPPER_SLEEVELESS = frozenset({
    "tank_top", "crop_top", "camisole", "vest", "corset", "halter",
})
_FAMILY_UPPER_LOOSE = frozenset({
    "hoodie", "sweater", "poncho", "cape", "shrug", "pullover",
})
_FAMILY_LOWER = frozenset({
    "jeans", "trousers", "pants", "shorts", "skirt", "mini_skirt",
    "long_skirt", "leggings", "joggers", "cargo_pants", "wide_leg",
    "palazzo", "dhoti_pants", "chinos", "bermuda",
})
_FAMILY_FULL = frozenset({
    "dress", "mini_dress", "midi_dress", "maxi_dress", "bodycon",
    "a_line", "jumpsuit", "evening_gown", "ball_gown", "wedding",
    "maxi", "wrap_dress", "off_shoulder", "one_shoulder", "strap",
    "kurti", "kurta", "abaya", "kaftan", "jalabiya", "kimono",
    "hanbok", "cheongsam", "qipao", "yukata", "sherwani",
})
_FAMILY_DRAPED = frozenset({
    "saree", "sari", "lehenga", "ghagra", "dupatta", "shawl",
    "anarkali", "salwar_suit", "dhoti", "lungi",
})


def get_garment_family(garment_subtype: str) -> str:
    """Classify garment subtype into a routing family.

    Returns: "upper_structured", "upper_fitted", "upper_sleeveless",
    "upper_loose", "lower", "full", "draped", or "unknown".
    """
    key = (garment_subtype or "").strip().lower().replace(" ", "_").replace("-", "_")
    if not key:
        return "unknown"

    families = [
        ("upper_structured", _FAMILY_UPPER_STRUCTURED),
        ("upper_fitted", _FAMILY_UPPER_FITTED),
        ("upper_sleeveless", _FAMILY_UPPER_SLEEVELESS),
        ("upper_loose", _FAMILY_UPPER_LOOSE),
        ("lower", _FAMILY_LOWER),
        ("full", _FAMILY_FULL),
        ("draped", _FAMILY_DRAPED),
    ]
    for family_name, family_set in families:
        if key in family_set:
            return family_name
    for family_name, family_set in families:
        for member in family_set:
            if key in member or member in key:
                return family_name
    return "unknown"


# ── Debug artifact generation ──────────────────────────────────────────
def save_mask_debug_artifacts(
    trace_id: str,
    *,
    schp_labels: np.ndarray | None = None,
    inpaint_mask: np.ndarray | None = None,
    protect_mask: np.ndarray | None = None,
    final_mask: np.ndarray | None = None,
    person_img: Image.Image | None = None,
) -> None:
    """Save mask overlay images for debugging under /tmp/idm-vton-debug/."""
    if not trace_id:
        return
    from pathlib import Path
    debug_dir = Path("/tmp/idm-vton-debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(debug_dir / f"mask_{trace_id}")

    try:
        if schp_labels is not None:
            colors = np.zeros((*schp_labels.shape, 3), dtype=np.uint8)
            label_colors = [
                (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
                (0, 0, 128), (255, 0, 0), (255, 128, 0), (255, 255, 0),
                (128, 128, 128), (0, 255, 0), (0, 128, 128), (128, 0, 128),
                (0, 255, 255), (255, 0, 255), (128, 64, 0), (64, 128, 0),
                (0, 64, 128), (0, 128, 64), (64, 0, 128), (128, 0, 64),
            ]
            for i, c in enumerate(label_colors):
                if i < 20:
                    colors[schp_labels == i] = c
            Image.fromarray(colors).save(f"{prefix}_schp_labels.png")

        for mask_arr, suffix, clr in [
            (inpaint_mask, "inpaint", (255, 80, 80)),
            (protect_mask, "protect", (80, 80, 255)),
            (final_mask, "final", (80, 255, 80)),
        ]:
            if person_img is not None and mask_arr is not None:
                bg = person_img.convert("RGB").copy()
                if bg.size != (mask_arr.shape[1], mask_arr.shape[0]):
                    bg = bg.resize((mask_arr.shape[1], mask_arr.shape[0]), Image.LANCZOS)
                arr = np.array(bg, dtype=np.uint8)
                m = mask_arr > 127
                arr[m, 0] = (arr[m, 0].astype(np.uint16) + clr[0]) // 2
                arr[m, 1] = (arr[m, 1].astype(np.uint16) + clr[1]) // 2
                arr[m, 2] = (arr[m, 2].astype(np.uint16) + clr[2]) // 2
                contours, _ = cv2.findContours(
                    mask_arr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(arr, contours, -1, clr, 2)
                Image.fromarray(arr).save(f"{prefix}_{suffix}_overlay.png")

        logger.info("mask_debug_saved prefix=%s", prefix)
    except Exception as exc:
        logger.warning("mask_debug_save_failed error=%s", exc)


def build_schp_inpaint_mask(
    schp_labels: np.ndarray,
    cloth_type: str,
    garment_subtype: str = "",
    source_cloth_type: str = "",
    garment_img_info: "GarmentImageInfo | None" = None,
) -> np.ndarray:
    """Build binary inpaint mask from SCHP labels.

    Same-category: uses target garment labels — the model changes texture/color
    within the source garment's shape.

    Cross-category: uses UNION of source + target garment labels, plus an
    adaptive dilated buffer around source labels. This ensures the model can
    fully erase the old garment (source labels + buffer) while generating the new
    garment in the correct body region (target labels).

    The protect mask (built separately) constrains which regions actually
    change — identity (face, hair, shoes) is always protected, and
    non-target body regions are protected per garment family.

    255 = editable, 0 = protected.
    """
    target_labels = _CLOTHING_LABELS.get(cloth_type, _CLOTHING_LABELS["dresses"])
    is_cross = (
        source_cloth_type
        and source_cloth_type != cloth_type
        and source_cloth_type != "unknown"
    )

    if is_cross:
        source_labels = _CLOTHING_LABELS.get(source_cloth_type, set())
        combined = target_labels | source_labels

        # Base mask: all present garment labels (source + target)
        mask = np.isin(schp_labels, list(combined)).astype(np.uint8) * 255

        # Dilated buffer around SOURCE labels only — ensures clean erasure
        # of old garment even at edges where SCHP segmentation is imperfect.
        # Kernel size adapts to body size, source coverage, target geometry.
        source_only = np.isin(schp_labels, list(source_labels)).astype(np.uint8) * 255
        if np.any(source_only):
            ks = _adaptive_buffer_ks(schp_labels, source_labels, garment_img_info=garment_img_info)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
            source_dilated = cv2.dilate(source_only, kernel, iterations=1)
            ke = max(3, ks - 4)
            if ke % 2 == 0:
                ke += 1
            kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ke, ke))
            source_tight = cv2.erode(source_dilated, kernel_e, iterations=1)
            mask = np.maximum(mask, source_tight)

        # Include arm labels if source is draped (saree pallu over arms)
        if source_cloth_type in ("dresses", "full_body"):
            src_subtype = (garment_subtype or "").lower()
            if any(kw in src_subtype for kw in _DRAPE_KEYWORDS):
                arm_mask = np.isin(schp_labels, list(_DRAPE_ARM_LABELS)).astype(np.uint8) * 255
                mask = np.maximum(mask, arm_mask)

        # Include leg labels if source is lower or full body but target is upper
        if source_cloth_type in ("lower_body", "dresses", "full_body"):
            if cloth_type == "upper_body":
                leg_labels = {_LABEL_LEFT_LEG, _LABEL_RIGHT_LEG, _LABEL_PANTS,
                             _LABEL_SKIRT, _LABEL_SOCKS}
                leg_mask = np.isin(schp_labels, list(leg_labels)).astype(np.uint8) * 255
                mask = np.maximum(mask, leg_mask)

        # Include upper labels if source is upper but target is lower/full
        if source_cloth_type == "upper_body":
            if cloth_type in ("lower_body", "dresses", "full_body"):
                upper_labels = {_LABEL_UPPER_CLOTHES, _LABEL_COAT}
                upper_mask = np.isin(schp_labels, list(upper_labels)).astype(np.uint8) * 255
                mask = np.maximum(mask, upper_mask)
    else:
        # Same-category: target labels only
        mask = np.isin(schp_labels, list(target_labels)).astype(np.uint8) * 255

        # Include arm labels for draped targets (saree pallu, dupatta)
        if is_draped_garment(cloth_type, garment_subtype):
            arm_mask = np.isin(schp_labels, list(_DRAPE_ARM_LABELS)).astype(np.uint8) * 255
            mask = np.maximum(mask, arm_mask)

    # Always exclude identity labels from the inpaint mask
    identity_mask = np.isin(schp_labels, list(_IDENTITY_PROTECT_LABELS)).astype(np.uint8) * 255
    mask = np.where(identity_mask > 0, 0, mask).astype(np.uint8)

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

    Arm protection is garment-aware:
    - Tank tops, crop tops, vests: arms EXPOSED (not protected) so model
      can generate bare arms or sleeveless output
    - Draped garments: full arms protected (pallu generated via IP-Adapter)
    - Standard garments: full arms protected
    """
    geometry = get_garment_geometry(garment_subtype)
    draped = is_draped_garment(cloth_type, garment_subtype)
    mask = np.isin(schp_labels, list(_IDENTITY_PROTECT_LABELS)).astype(np.uint8) * 255

    if cloth_type == "upper_body":
        if not geometry.expose_arms:
            # Block lower-body replacement when only swapping tops.
            lower_labels = {_LABEL_PANTS, _LABEL_SKIRT, _LABEL_SOCKS,
                           _LABEL_LEFT_LEG, _LABEL_RIGHT_LEG}
            mask = np.maximum(mask, np.isin(schp_labels, list(lower_labels)).astype(np.uint8) * 255)
            # Full arms protected (unless garment exposes arms like tank top)
            mask = np.maximum(mask, np.isin(schp_labels, list(_DRAPE_ARM_LABELS)).astype(np.uint8) * 255)
        else:
            # Sleeveless garment: only protect lower body, NOT arms
            lower_labels = {_LABEL_PANTS, _LABEL_SKIRT, _LABEL_SOCKS,
                           _LABEL_LEFT_LEG, _LABEL_RIGHT_LEG}
            mask = np.maximum(mask, np.isin(schp_labels, list(lower_labels)).astype(np.uint8) * 255)
    elif cloth_type == "lower_body":
        upper_labels = {_LABEL_UPPER_CLOTHES, _LABEL_COAT, _LABEL_DRESS, _LABEL_SCARF}
        mask = np.maximum(mask, np.isin(schp_labels, list(upper_labels)).astype(np.uint8) * 255)
        mask = np.maximum(mask, np.isin(schp_labels, list(_DRAPE_ARM_LABELS)).astype(np.uint8) * 255)
    elif draped:
        # Draped targets: protect only HANDS (distal arm), not full forearms.
        # This lets pallu/drape be generated over forearms while keeping
        # mehndi, phones, and ring details intact.
        hand_zone = _hand_zones_from_arms(schp_labels)
        mask = np.maximum(mask, hand_zone)
    else:
        mask = np.maximum(mask, np.isin(schp_labels, list(_DRAPE_ARM_LABELS)).astype(np.uint8) * 255)

    if draped:
        dilate_px = max(11, dilate_px - 2)

    mask = _dilate_mask(mask, dilate_px, iterations=1)
    return mask


def dilate_inpaint_mask(
    inpaint_mask: np.ndarray,
    cloth_type: str,
    garment_subtype: str = "",
    schp_height: int = 512,
) -> np.ndarray:
    """Mild dilation for edge blending.

    With the full-body-silhouette architecture, the mask already covers
    the entire body.  Dilation is only needed to smooth mask boundaries
    so the diffusion model doesn't create hard-edge artifacts.  We use a
    small uniform kernel regardless of garment family.
    """
    scale = schp_height / 512.0
    ks = max(3, int(5 * scale))
    if ks % 2 == 0:
        ks += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    return cv2.dilate(inpaint_mask, kernel, iterations=1)


def build_final_inpaint_mask(
    schp_labels: np.ndarray,
    cloth_type: str,
    garment_subtype: str = "",
    source_cloth_type: str = "",
    garment_img_info: "GarmentImageInfo | None" = None,
    trace_id: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full mask pipeline: source+target labels → protect → dilate → subtract.

    For same-category: mask = target garment labels. Model changes texture/color
    within the source garment's shape.

    For cross-category: mask = UNION of source + target garment labels plus a
    dilated buffer around source labels. This ensures the model can fully erase
    the old garment while generating the new one in the correct body region.

    The protect mask constrains which regions actually change:
      - Identity (face, hair, shoes) is always protected
      - Non-target body regions are protected per garment family
        (e.g. lower body protected when only swapping tops)

    Args:
        schp_labels: SCHP label map from person image.
        cloth_type: Target garment's cloth_type.
        garment_subtype: Target garment's specific subtype (e.g. "jacket", "saree").
        source_cloth_type: Person's current garment cloth_type (for cross-category mask).
        garment_img_info: Target garment image geometry (for adaptive buffer sizing).
        trace_id: Debug trace ID for artifact generation.
    """
    # 1. Build inpaint mask — source+target labels for cross-category
    inpaint_raw = build_schp_inpaint_mask(
        schp_labels, cloth_type, garment_subtype, source_cloth_type, garment_img_info,
    )

    # 2. Build protect mask (identity regions + non-target body regions)
    protect = build_schp_protect_mask(schp_labels, cloth_type, garment_subtype)

    # 3. Mild dilation for edge blending (small kernel — just smooths boundaries)
    h, w = schp_labels.shape
    scale = max(1.0, h / 512.0)
    ks = max(3, int(5 * scale))
    if ks % 2 == 0:
        ks += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    inpaint_dilated = cv2.dilate(inpaint_raw, kernel, iterations=1)

    # 4. Apply protection (subtract identity from editable)
    final = apply_protection_binary(inpaint_dilated, protect)

    # 5. Debug artifacts
    if trace_id:
        try:
            save_mask_debug_artifacts(
                trace_id,
                schp_labels=schp_labels,
                inpaint_mask=inpaint_dilated,
                protect_mask=protect,
                final_mask=final,
            )
        except Exception:
            pass

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
