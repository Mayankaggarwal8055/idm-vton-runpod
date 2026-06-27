"""
GPU worker mask pipeline — Garment-aware difference-based editing.

SCHP is the single authoritative mask source. All masks are binary.
No feathering, no fusing, no hybrid strategies. The model output is final.

Architecture:
  1. GARMENT PROFILE — structured understanding of target garment (family,
     coverage, sleeves, drape, layering, fit). Replaces flat cloth_type.
  2. BODY REGION ANALYSIS — maps garment profile to body regions that are
     editable vs protected. Adapts per garment family.
  3. DIFFERENCE-BASED EDITING — computes editable region from source/target
     occupancy difference, not from a single mask.
  4. FAMILY-AWARE ROUTING — different garment families get different mask
     shapes, protect regions, prompts, and scoring weights.
  5. MULTI-STAGE GENERATION — cross-category and layered garments get
     erase + alignment + synthesis stages.

Design principles:
  - Same-category: mask = source garment labels (replace within source shape).
  - Cross-category: mask = target body region + source labels + buffer.
  - Protect = identity + regions target does NOT cover.
  - Buffer dilation adapts to body size, source coverage, target geometry.
  - Every garment family has its own masking profile.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("idm-vton.worker.mask")

# SCHP LIP 20-class label constants — must match the actual ONNX model output.
# Verified against parsing_api.py: label 4=upper_clothes, 7=dress, 11=face, 14/15=arms.
_LABEL_BG = 0
_LABEL_HAT = 1
_LABEL_HAIR = 2
_LABEL_SUNGLASSES = 3
_LABEL_UPPER_CLOTHES = 4
_LABEL_SKIRT = 5
_LABEL_PANTS = 6
_LABEL_DRESS = 7
_LABEL_BELT = 8
_LABEL_LEFT_SHOE = 9
_LABEL_RIGHT_SHOE = 10
_LABEL_FACE = 11
_LABEL_LEFT_LEG = 12
_LABEL_RIGHT_LEG = 13
_LABEL_LEFT_ARM = 14
_LABEL_RIGHT_ARM = 15
_LABEL_BAG = 16
_LABEL_SCARF = 17
_LABEL_NECK = 18

# Clothing label sets per cloth_type — using corrected LIP label values.
# LIP: 4=upper_clothes, 5=skirt, 6=pants, 7=dress
_DRESSES_LABELS = {
    _LABEL_UPPER_CLOTHES,
    _LABEL_DRESS,
    _LABEL_PANTS,
    _LABEL_SKIRT,
    _LABEL_SCARF,
}
_CLOTHING_LABELS = {
    "upper_body": {_LABEL_UPPER_CLOTHES},
    "lower_body": {_LABEL_PANTS, _LABEL_SKIRT},
    "dresses": _DRESSES_LABELS,
    "full_body": _DRESSES_LABELS,
}

# All garment labels for cross-category mismatch detection.
# Used to detect when the person's current garment has labels that fall
# outside the target cloth_type's editable mask.
_ALL_GARMENT_LABELS = {
    _LABEL_UPPER_CLOTHES,
    _LABEL_DRESS,
    _LABEL_PANTS,
    _LABEL_SKIRT,
    _LABEL_SCARF,
    _LABEL_BELT,
    _LABEL_BAG,
}

# Identity labels that must never be edited — face, hair, accessories, shoes.
_IDENTITY_PROTECT_LABELS = {
    _LABEL_HAIR,
    _LABEL_FACE,
    _LABEL_HAT,
    _LABEL_SUNGLASSES,
    _LABEL_LEFT_SHOE,
    _LABEL_RIGHT_SHOE,
    _LABEL_NECK,
    _LABEL_BELT,
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


# ── Body region analysis for difference-based editing ────────────────
# Maps cloth_type to the SCHP labels that are EDITABLE (inpaintable) for that
# target. The protect mask is the complement: identity + labels NOT in this set.
#
# This replaces the old hardcoded "protect lower body for upper targets" logic.
# Instead, the protect mask adapts to which body region the target garment covers.

EDITABLE_BODY_REGIONS: dict[str, set[int]] = {
    "upper_body": {
        _LABEL_UPPER_CLOTHES,
        _LABEL_LEFT_ARM,
        _LABEL_RIGHT_ARM,
    },
    "lower_body": {
        _LABEL_PANTS,
        _LABEL_SKIRT,
        _LABEL_LEFT_LEG,
        _LABEL_RIGHT_LEG,
    },
    "dresses": {
        _LABEL_UPPER_CLOTHES,
        _LABEL_DRESS,
        _LABEL_PANTS,
        _LABEL_SKIRT,
        _LABEL_SCARF,
        _LABEL_LEFT_ARM,
        _LABEL_RIGHT_ARM,
        _LABEL_LEFT_LEG,
        _LABEL_RIGHT_LEG,
    },
    "full_body": {
        _LABEL_UPPER_CLOTHES,
        _LABEL_DRESS,
        _LABEL_PANTS,
        _LABEL_SKIRT,
        _LABEL_SCARF,
        _LABEL_LEFT_ARM,
        _LABEL_RIGHT_ARM,
        _LABEL_LEFT_LEG,
        _LABEL_RIGHT_LEG,
    },
}

# For long upper garments (jacket, coat, blazer, cardigan, leather_jacket,
# denim_jacket) that extend below the waist, the editable region must include
# lower body labels too. Without this, the model can't generate the garment's
# lower portion — it gets blocked by the protect mask.
_LONG_UPPER_GARMENTS = frozenset({
    "jacket", "blazer", "coat", "cardigan", "leather_jacket", "denim_jacket",
    "trench", "peacoat", "overcoat", "windbreaker", "parka",
})


def analyze_target_body_region(
    garment_subtype: str,
    cloth_type: str,
) -> str:
    """Map target garment subtype to the cloth_type used for body region analysis.

    Returns one of: "upper_body", "lower_body", "dresses", "full_body".

    For long upper garments (jacket, coat), returns "dresses" so the editable
    region includes both upper AND lower body labels — the jacket extends
    below the waist, so the model needs freedom to generate there.

    For fitted upper garments (t-shirt, shirt), returns "upper_body" — protect
    the lower body since the garment doesn't extend past the waist.

    This is the key function that determines which body regions are editable
    for each target garment type.
    """
    key = (garment_subtype or "").strip().lower().replace(" ", "_").replace("-", "_")

    # Check if this is a long upper garment — treat as full-body coverage
    if key in _LONG_UPPER_GARMENTS:
        return "dresses"

    # Fuzzy match for long upper garments
    for long_name in _LONG_UPPER_GARMENTS:
        if key and (long_name in key or key in long_name):
            return "dresses"

    # For everything else, use the provided cloth_type
    return cloth_type


# ═══════════════════════════════════════════════════════════════════════
# GarmentProfile — structured garment understanding
# ═══════════════════════════════════════════════════════════════════════
# Every garment subtype gets a profile that drives masking, prompting,
# scoring, and routing. This replaces flat cloth_type with rich structure.

@dataclass(frozen=True)
class GarmentProfile:
    """Structured understanding of a garment's properties.

    Used to drive:
      - Mask shape and coverage (which body regions are editable)
      - Protect mask (which regions to shield)
      - Prompt construction (what to describe)
      - Scoring weights (what to optimize for)
      - Multi-stage routing (erase vs. single-stage)
    """
    # Core identity
    family: str = "unknown"         # upper_fitted, upper_structured, lower, full, draped
    cloth_type: str = "upper_body"  # upper_body, lower_body, dresses, full_body

    # Body coverage
    covers_upper: bool = True       # shoulders, chest, torso
    covers_lower: bool = False      # waist, hips, legs
    covers_arms: bool = True        # full arms
    covers_hands: bool = False      # hands/wrists
    covers_torso_full: bool = True  # full torso (vs. cropped)
    extends_below_waist: bool = False  # jacket, coat extend past waist

    # Sleeve / arm behavior
    has_sleeves: bool = True
    sleeve_length: str = "long"     # short, long, sleeveless
    expose_arms: bool = False       # tank_top, crop_top — arms should be visible

    # Fit / structure
    is_fitted: bool = True          # body-hugging
    is_structured: bool = False     # rigid shape (jacket, blazer)
    is_loose: bool = False          # flowing (kaftan, poncho)

    # Drape / flow
    is_draped: bool = False         # saree, dupatta, lehenga
    has_pallu: bool = False         # saree pallu over shoulder/arms
    has_border: bool = False        # decorative border (saree, lehenga)

    # Layering
    is_layered: bool = False        # jacket over shirt, etc.
    layer_order: int = 0            # 0=single, 1=outer, 2=inner

    # Hem behavior
    hem_type: str = "straight"      # straight, curved, asymmetric, flared

    # Special properties
    is_cropped: bool = False        # crop_top, mini_skirt
    is_voluminous: bool = False     # ball_gown, palazzo
    is_ethnic: bool = False         # saree, lehenga, kurta


# ── Garment profiles database ─────────────────────────────────────────
# Maps garment subtype → GarmentProfile. Every profile defines exactly
# which body regions the garment covers and how the mask should behave.

GARMENT_PROFILES: dict[str, GarmentProfile] = {
    # ── Upper body: fitted ─────────────────────────────────────────────
    "tshirt": GarmentProfile(
        family="upper_fitted", cloth_type="upper_body",
        covers_upper=True, covers_lower=False, covers_arms=True,
        covers_torso_full=True, extends_below_waist=False,
        has_sleeves=True, sleeve_length="short",
        is_fitted=True, is_structured=False,
    ),
    "t_shirt": GarmentProfile(
        family="upper_fitted", cloth_type="upper_body",
        covers_upper=True, covers_lower=False, covers_arms=True,
        covers_torso_full=True, extends_below_waist=False,
        has_sleeves=True, sleeve_length="short",
        is_fitted=True, is_structured=False,
    ),
    "polo": GarmentProfile(
        family="upper_fitted", cloth_type="upper_body",
        covers_upper=True, covers_lower=False, covers_arms=True,
        covers_torso_full=True, extends_below_waist=False,
        has_sleeves=True, sleeve_length="short",
        is_fitted=True, is_structured=False,
    ),
    "shirt": GarmentProfile(
        family="upper_fitted", cloth_type="upper_body",
        covers_upper=True, covers_lower=False, covers_arms=True,
        covers_torso_full=True, extends_below_waist=False,
        has_sleeves=True, sleeve_length="long",
        is_fitted=False, is_structured=False,
    ),
    "blouse": GarmentProfile(
        family="upper_fitted", cloth_type="upper_body",
        covers_upper=True, covers_lower=False, covers_arms=True,
        covers_torso_full=True, extends_below_waist=False,
        has_sleeves=True, sleeve_length="long",
        is_fitted=False, is_structured=False,
    ),
    "sweatshirt": GarmentProfile(
        family="upper_fitted", cloth_type="upper_body",
        covers_upper=True, covers_lower=False, covers_arms=True,
        covers_torso_full=True, extends_below_waist=False,
        has_sleeves=True, sleeve_length="long",
        is_fitted=False, is_structured=False, is_loose=True,
    ),
    "sports_jersey": GarmentProfile(
        family="upper_fitted", cloth_type="upper_body",
        covers_upper=True, covers_lower=False, covers_arms=True,
        covers_torso_full=True, extends_below_waist=False,
        has_sleeves=True, sleeve_length="short",
        is_fitted=False, is_structured=False, is_loose=True,
    ),

    # ── Upper body: sleeveless / exposed ───────────────────────────────
    "tank_top": GarmentProfile(
        family="upper_sleeveless", cloth_type="upper_body",
        covers_upper=True, covers_lower=False, covers_arms=False,
        covers_torso_full=True, extends_below_waist=False,
        has_sleeves=False, sleeve_length="sleeveless",
        expose_arms=True, is_fitted=True,
    ),
    "crop_top": GarmentProfile(
        family="upper_sleeveless", cloth_type="upper_body",
        covers_upper=True, covers_lower=False, covers_arms=False,
        covers_torso_full=False, extends_below_waist=False,
        has_sleeves=False, sleeve_length="sleeveless",
        expose_arms=True, is_fitted=True, is_cropped=True,
    ),
    "camisole": GarmentProfile(
        family="upper_sleeveless", cloth_type="upper_body",
        covers_upper=True, covers_lower=False, covers_arms=False,
        covers_torso_full=True, extends_below_waist=False,
        has_sleeves=False, sleeve_length="sleeveless",
        expose_arms=True, is_fitted=True,
    ),
    "vest": GarmentProfile(
        family="upper_sleeveless", cloth_type="upper_body",
        covers_upper=True, covers_lower=False, covers_arms=False,
        covers_torso_full=True, extends_below_waist=False,
        has_sleeves=False, sleeve_length="sleeveless",
        expose_arms=True, is_fitted=True,
    ),
    "corset": GarmentProfile(
        family="upper_sleeveless", cloth_type="upper_body",
        covers_upper=True, covers_lower=False, covers_arms=False,
        covers_torso_full=False, extends_below_waist=False,
        has_sleeves=False, sleeve_length="sleeveless",
        expose_arms=True, is_fitted=True, is_structured=True,
    ),

    # ── Upper body: extended / long ────────────────────────────────────
    "sweater": GarmentProfile(
        family="upper_structured", cloth_type="upper_body",
        covers_upper=True, covers_lower=False, covers_arms=True,
        covers_torso_full=True, extends_below_waist=True,
        has_sleeves=True, sleeve_length="long",
        is_fitted=False, is_loose=True,
    ),
    "hoodie": GarmentProfile(
        family="upper_structured", cloth_type="upper_body",
        covers_upper=True, covers_lower=False, covers_arms=True,
        covers_torso_full=True, extends_below_waist=True,
        has_sleeves=True, sleeve_length="long",
        is_fitted=False, is_loose=True, is_structured=False,
    ),
    "jacket": GarmentProfile(
        family="upper_structured", cloth_type="upper_body",
        covers_upper=True, covers_lower=True, covers_arms=True,
        covers_torso_full=True, extends_below_waist=True,
        has_sleeves=True, sleeve_length="long",
        is_fitted=False, is_structured=True,
    ),
    "blazer": GarmentProfile(
        family="upper_structured", cloth_type="upper_body",
        covers_upper=True, covers_lower=True, covers_arms=True,
        covers_torso_full=True, extends_below_waist=True,
        has_sleeves=True, sleeve_length="long",
        is_fitted=False, is_structured=True,
    ),
    "coat": GarmentProfile(
        family="upper_structured", cloth_type="upper_body",
        covers_upper=True, covers_lower=True, covers_arms=True,
        covers_torso_full=True, extends_below_waist=True,
        has_sleeves=True, sleeve_length="long",
        is_fitted=False, is_structured=True,
    ),
    "cardigan": GarmentProfile(
        family="upper_structured", cloth_type="upper_body",
        covers_upper=True, covers_lower=True, covers_arms=True,
        covers_torso_full=True, extends_below_waist=True,
        has_sleeves=True, sleeve_length="long",
        is_fitted=False, is_loose=True,
    ),
    "leather_jacket": GarmentProfile(
        family="upper_structured", cloth_type="upper_body",
        covers_upper=True, covers_lower=True, covers_arms=True,
        covers_torso_full=True, extends_below_waist=True,
        has_sleeves=True, sleeve_length="long",
        is_fitted=False, is_structured=True,
    ),
    "denim_jacket": GarmentProfile(
        family="upper_structured", cloth_type="upper_body",
        covers_upper=True, covers_lower=True, covers_arms=True,
        covers_torso_full=True, extends_below_waist=True,
        has_sleeves=True, sleeve_length="long",
        is_fitted=False, is_structured=True,
    ),

    # ── Upper body: wide / flowing ─────────────────────────────────────
    "poncho": GarmentProfile(
        family="upper_structured", cloth_type="upper_body",
        covers_upper=True, covers_lower=True, covers_arms=True,
        covers_torso_full=True, extends_below_waist=True,
        has_sleeves=False, sleeve_length="sleeveless",
        is_fitted=False, is_loose=True, is_voluminous=True,
    ),
    "cape": GarmentProfile(
        family="upper_structured", cloth_type="upper_body",
        covers_upper=True, covers_lower=True, covers_arms=True,
        covers_torso_full=True, extends_below_waist=True,
        has_sleeves=False, sleeve_length="sleeveless",
        is_fitted=False, is_loose=True, is_voluminous=True,
    ),

    # ── Lower body ─────────────────────────────────────────────────────
    "jeans": GarmentProfile(
        family="lower", cloth_type="lower_body",
        covers_upper=False, covers_lower=True, covers_arms=False,
        covers_hands=False, covers_torso_full=False,
        has_sleeves=False, is_fitted=True,
    ),
    "trousers": GarmentProfile(
        family="lower", cloth_type="lower_body",
        covers_upper=False, covers_lower=True, covers_arms=False,
        covers_hands=False, covers_torso_full=False,
        has_sleeves=False, is_fitted=True,
    ),
    "pants": GarmentProfile(
        family="lower", cloth_type="lower_body",
        covers_upper=False, covers_lower=True, covers_arms=False,
        covers_hands=False, covers_torso_full=False,
        has_sleeves=False, is_fitted=True,
    ),
    "shorts": GarmentProfile(
        family="lower", cloth_type="lower_body",
        covers_upper=False, covers_lower=True, covers_arms=False,
        covers_hands=False, covers_torso_full=False,
        has_sleeves=False, is_fitted=True,
    ),
    "skirt": GarmentProfile(
        family="lower", cloth_type="lower_body",
        covers_upper=False, covers_lower=True, covers_arms=False,
        covers_hands=False, covers_torso_full=False,
        has_sleeves=False, is_fitted=False,
    ),
    "mini_skirt": GarmentProfile(
        family="lower", cloth_type="lower_body",
        covers_upper=False, covers_lower=True, covers_arms=False,
        covers_hands=False, covers_torso_full=False,
        has_sleeves=False, is_fitted=False,
    ),
    "leggings": GarmentProfile(
        family="lower", cloth_type="lower_body",
        covers_upper=False, covers_lower=True, covers_arms=False,
        covers_hands=False, covers_torso_full=False,
        has_sleeves=False, is_fitted=True,
    ),
    "joggers": GarmentProfile(
        family="lower", cloth_type="lower_body",
        covers_upper=False, covers_lower=True, covers_arms=False,
        covers_hands=False, covers_torso_full=False,
        has_sleeves=False, is_fitted=False, is_loose=True,
    ),
    "wide_leg": GarmentProfile(
        family="lower", cloth_type="lower_body",
        covers_upper=False, covers_lower=True, covers_arms=False,
        covers_hands=False, covers_torso_full=False,
        has_sleeves=False, is_fitted=False, is_loose=True, is_voluminous=True,
    ),
    "palazzo": GarmentProfile(
        family="lower", cloth_type="lower_body",
        covers_upper=False, covers_lower=True, covers_arms=False,
        covers_hands=False, covers_torso_full=False,
        has_sleeves=False, is_fitted=False, is_loose=True, is_voluminous=True,
    ),

    # ── Full body ──────────────────────────────────────────────────────
    "dress": GarmentProfile(
        family="full", cloth_type="dresses",
        covers_upper=True, covers_lower=True, covers_arms=True,
        covers_torso_full=True,
        has_sleeves=True, sleeve_length="long",
        is_fitted=False,
    ),
    "mini_dress": GarmentProfile(
        family="full", cloth_type="dresses",
        covers_upper=True, covers_lower=True, covers_arms=True,
        covers_torso_full=True,
        has_sleeves=True, sleeve_length="long",
        is_fitted=False,
    ),
    "midi_dress": GarmentProfile(
        family="full", cloth_type="dresses",
        covers_upper=True, covers_lower=True, covers_arms=True,
        covers_torso_full=True,
        has_sleeves=True, sleeve_length="long",
        is_fitted=False,
    ),
    "maxi_dress": GarmentProfile(
        family="full", cloth_type="dresses",
        covers_upper=True, covers_lower=True, covers_arms=True,
        covers_torso_full=True,
        has_sleeves=True, sleeve_length="long",
        is_fitted=False, is_voluminous=True,
    ),
    "bodycon": GarmentProfile(
        family="full", cloth_type="dresses",
        covers_upper=True, covers_lower=True, covers_arms=True,
        covers_torso_full=True,
        has_sleeves=True, sleeve_length="long",
        is_fitted=True,
    ),
    "evening_gown": GarmentProfile(
        family="full", cloth_type="dresses",
        covers_upper=True, covers_lower=True, covers_arms=True,
        covers_torso_full=True,
        has_sleeves=True, sleeve_length="long",
        is_fitted=True, is_structured=True,
    ),
    "ball_gown": GarmentProfile(
        family="full", cloth_type="dresses",
        covers_upper=True, covers_lower=True, covers_arms=True,
        covers_torso_full=True,
        has_sleeves=True, sleeve_length="long",
        is_fitted=True, is_structured=True, is_voluminous=True,
    ),
    "wedding": GarmentProfile(
        family="full", cloth_type="dresses",
        covers_upper=True, covers_lower=True, covers_arms=True,
        covers_torso_full=True,
        has_sleeves=True, sleeve_length="long",
        is_fitted=True, is_structured=True,
    ),
    "jumpsuit": GarmentProfile(
        family="full", cloth_type="dresses",
        covers_upper=True, covers_lower=True, covers_arms=True,
        covers_torso_full=True,
        has_sleeves=True, sleeve_length="long",
        is_fitted=True,
    ),
    "wrap_dress": GarmentProfile(
        family="full", cloth_type="dresses",
        covers_upper=True, covers_lower=True, covers_arms=True,
        covers_torso_full=True,
        has_sleeves=True, sleeve_length="long",
        is_fitted=False, hem_type="asymmetric",
    ),

    # ── Traditional: draped ────────────────────────────────────────────
    "saree": GarmentProfile(
        family="draped", cloth_type="dresses",
        covers_upper=True, covers_lower=True, covers_arms=True,
        covers_hands=False, covers_torso_full=True,
        has_sleeves=True, sleeve_length="long",
        is_draped=True, has_pallu=True, has_border=True,
        is_ethnic=True, is_loose=True,
    ),
    "sari": GarmentProfile(
        family="draped", cloth_type="dresses",
        covers_upper=True, covers_lower=True, covers_arms=True,
        covers_hands=False, covers_torso_full=True,
        has_sleeves=True, sleeve_length="long",
        is_draped=True, has_pallu=True, has_border=True,
        is_ethnic=True, is_loose=True,
    ),
    "lehenga": GarmentProfile(
        family="draped", cloth_type="dresses",
        covers_upper=True, covers_lower=True, covers_arms=True,
        covers_torso_full=True,
        has_sleeves=True, sleeve_length="long",
        is_draped=True, has_border=True, is_ethnic=True,
    ),
    "dupatta": GarmentProfile(
        family="draped", cloth_type="dresses",
        covers_upper=True, covers_lower=True, covers_arms=True,
        covers_torso_full=True,
        has_sleeves=False, sleeve_length="sleeveless",
        is_draped=True, has_pallu=True, is_ethnic=True, is_loose=True,
    ),
    "shawl": GarmentProfile(
        family="draped", cloth_type="dresses",
        covers_upper=True, covers_lower=False, covers_arms=True,
        covers_torso_full=True,
        has_sleeves=False, sleeve_length="sleeveless",
        is_draped=True, is_ethnic=True, is_loose=True,
    ),
    "anarkali": GarmentProfile(
        family="draped", cloth_type="dresses",
        covers_upper=True, covers_lower=True, covers_arms=True,
        covers_torso_full=True,
        has_sleeves=True, sleeve_length="long",
        is_draped=True, is_ethnic=True, is_voluminous=True,
    ),
    "kimono": GarmentProfile(
        family="draped", cloth_type="dresses",
        covers_upper=True, covers_lower=True, covers_arms=True,
        covers_torso_full=True,
        has_sleeves=True, sleeve_length="long",
        is_draped=True, is_loose=True, is_voluminous=True,
    ),
    "abaya": GarmentProfile(
        family="draped", cloth_type="dresses",
        covers_upper=True, covers_lower=True, covers_arms=True,
        covers_torso_full=True,
        has_sleeves=True, sleeve_length="long",
        is_draped=True, is_loose=True, is_ethnic=True,
    ),
    "kaftan": GarmentProfile(
        family="draped", cloth_type="dresses",
        covers_upper=True, covers_lower=True, covers_arms=True,
        covers_torso_full=True,
        has_sleeves=True, sleeve_length="long",
        is_draped=True, is_loose=True, is_voluminous=True,
    ),
    "kurta": GarmentProfile(
        family="full", cloth_type="dresses",
        covers_upper=True, covers_lower=True, covers_arms=True,
        covers_torso_full=True,
        has_sleeves=True, sleeve_length="long",
        is_fitted=False, is_ethnic=True,
    ),
    "kurti": GarmentProfile(
        family="full", cloth_type="dresses",
        covers_upper=True, covers_lower=True, covers_arms=True,
        covers_torso_full=True,
        has_sleeves=True, sleeve_length="long",
        is_fitted=False, is_ethnic=True,
    ),
    "sherwani": GarmentProfile(
        family="full", cloth_type="dresses",
        covers_upper=True, covers_lower=True, covers_arms=True,
        covers_torso_full=True,
        has_sleeves=True, sleeve_length="long",
        is_fitted=False, is_structured=True, is_ethnic=True,
    ),
}


def build_garment_profile(
    garment_subtype: str,
    cloth_type: str = "",
    garment_img_info: "GarmentImageInfo | None" = None,
) -> GarmentProfile:
    """Build a GarmentProfile for the target garment.

    Looks up the subtype in GARMENT_PROFILES, falls back to cloth_type-based
    defaults, and optionally adjusts based on garment image geometry.

    This is the single entry point for garment understanding — every other
    function should use this instead of raw cloth_type.
    """
    key = (garment_subtype or "").strip().lower().replace(" ", "_").replace("-", "_")

    # Direct lookup
    if key in GARMENT_PROFILES:
        profile = GARMENT_PROFILES[key]
    else:
        # Fuzzy match — prefer longest match
        best = None
        best_len = 0
        for pkey, pval in GARMENT_PROFILES.items():
            if key and pkey in key and len(pkey) > best_len:
                best = pval
                best_len = len(pkey)
        if best is None:
            for pkey, pval in GARMENT_PROFILES.items():
                if key and key in pkey and len(pkey) > best_len:
                    best = pval
                    best_len = len(pkey)
        profile = best if best else GarmentProfile()

    # Override cloth_type from parameter if provided
    if cloth_type and cloth_type != profile.cloth_type:
        profile = GarmentProfile(
            family=profile.family,
            cloth_type=cloth_type,
            covers_upper=profile.covers_upper,
            covers_lower=profile.covers_lower,
            covers_arms=profile.covers_arms,
            covers_hands=profile.covers_hands,
            covers_torso_full=profile.covers_torso_full,
            extends_below_waist=profile.extends_below_waist,
            has_sleeves=profile.has_sleeves,
            sleeve_length=profile.sleeve_length,
            expose_arms=profile.expose_arms,
            is_fitted=profile.is_fitted,
            is_structured=profile.is_structured,
            is_loose=profile.is_loose,
            is_draped=profile.is_draped,
            has_pallu=profile.has_pallu,
            has_border=profile.has_border,
            is_layered=profile.is_layered,
            layer_order=profile.layer_order,
            hem_type=profile.hem_type,
            is_cropped=profile.is_cropped,
            is_voluminous=profile.is_voluminous,
            is_ethnic=profile.is_ethnic,
        )

    # Adjust based on garment image geometry if available
    if garment_img_info:
        if garment_img_info.is_long and not profile.extends_below_waist:
            profile = GarmentProfile(
                family=profile.family, cloth_type=profile.cloth_type,
                covers_upper=profile.covers_upper, covers_lower=profile.covers_lower,
                covers_arms=profile.covers_arms, covers_hands=profile.covers_hands,
                covers_torso_full=profile.covers_torso_full,
                extends_below_waist=True,
                has_sleeves=profile.has_sleeves, sleeve_length=profile.sleeve_length,
                expose_arms=profile.expose_arms, is_fitted=profile.is_fitted,
                is_structured=profile.is_structured, is_loose=profile.is_loose,
                is_draped=profile.is_draped, has_pallu=profile.has_pallu,
                has_border=profile.has_border, is_layered=profile.is_layered,
                layer_order=profile.layer_order, hem_type=profile.hem_type,
                is_cropped=profile.is_cropped, is_voluminous=profile.is_voluminous,
                is_ethnic=profile.is_ethnic,
            )
        if not garment_img_info.has_sleeves and profile.has_sleeves:
            profile = GarmentProfile(
                family=profile.family, cloth_type=profile.cloth_type,
                covers_upper=profile.covers_upper, covers_lower=profile.covers_lower,
                covers_arms=False, covers_hands=profile.covers_hands,
                covers_torso_full=profile.covers_torso_full,
                extends_below_waist=profile.extends_below_waist,
                has_sleeves=False, sleeve_length="sleeveless",
                expose_arms=True, is_fitted=profile.is_fitted,
                is_structured=profile.is_structured, is_loose=profile.is_loose,
                is_draped=profile.is_draped, has_pallu=profile.has_pallu,
                has_border=profile.has_border, is_layered=profile.is_layered,
                layer_order=profile.layer_order, hem_type=profile.hem_type,
                is_cropped=profile.is_cropped, is_voluminous=profile.is_voluminous,
                is_ethnic=profile.is_ethnic,
            )

    return profile


# ═══════════════════════════════════════════════════════════════════════
# Phase 4 — Multi-Stage Pipeline Routing
# ═══════════════════════════════════════════════════════════════════════
# Determines the generation pipeline path based on garment profile,
# source/target relationship, and routing rules.

@dataclass(frozen=True)
class PipelineRoute:
    """Determines the generation pipeline for a specific try-on request.

    Attributes:
        pipeline: "single" | "cross_category" | "draped" | "layered" | "structured"
        needs_erase: True for cross-category (stage 1 erases old garment)
        erase_steps: Steps for erase stage (0 if no erase)
        erase_guidance: Guidance for erase stage
        apply_steps: Steps for apply stage
        apply_guidance: Guidance for apply stage
        is_cross: True if source and target are different cloth_types
        is_draped: True if target is a draped garment
        is_structured: True if target is structured outerwear
        is_layered: True if target implies layering
        family: Garment family string for routing
    """
    pipeline: str = "single"
    needs_erase: bool = False
    erase_steps: int = 0
    erase_guidance: float = 5.5
    apply_steps: int = 50
    apply_guidance: float = 2.5
    is_cross: bool = False
    is_draped: bool = False
    is_structured: bool = False
    is_layered: bool = False
    family: str = "unknown"


def compute_pipeline_route(
    source_cloth_type: str,
    target_cloth_type: str,
    target_profile: GarmentProfile,
    schp_labels: np.ndarray | None = None,
    requested_steps: int = 50,
    requested_guidance: float | None = None,
) -> PipelineRoute:
    """Determine the optimal generation pipeline for this try-on request.

    Routing logic:
      1. Same-category, same-family → single stage (fast, stable)
      2. Same-category, different family → single stage with enhanced masking
      3. Cross-category → two-stage erase + apply
      4. Draped target → two-stage with draped-specific erase prompt
      5. Structured target (jacket/coat) → two-stage with structured erase
      6. Layered target → two-stage with layered erase
      7. Fallback → single stage (safest path)

    Environment overrides:
      ERASE_STEPS, ERASE_GUIDANCE, APPLY_STEPS, APPLY_GUIDANCE
    """
    import os

    is_cross = (
        source_cloth_type
        and source_cloth_type != target_cloth_type
        and source_cloth_type != "unknown"
    )

    family = target_profile.family
    is_draped = target_profile.is_draped
    is_structured = target_profile.is_structured
    extends_below = target_profile.extends_below_waist

    # Read env overrides
    erase_steps_env = int(os.environ.get("CROSS_CATEGORY_ERASE_STEPS", "50"))
    erase_guidance_env = float(os.environ.get("CROSS_CATEGORY_ERASE_GUIDANCE", "5.5"))
    apply_steps = requested_steps
    apply_guidance = requested_guidance if requested_guidance is not None else float(os.environ.get("IDM_VTON_GUIDANCE", "2.5"))

    # ── Routing decision tree ──────────────────────────────────────────
    if not is_cross:
        # Same-category: single stage
        # But increase steps/guidance for complex families
        if is_draped:
            apply_steps = max(apply_steps, int(os.environ.get("IDM_VTON_DRESS_STEPS", "50")))
            apply_guidance = max(apply_guidance, float(os.environ.get("IDM_VTON_DRESS_GUIDANCE", "3.1")))
            route = "draped"
        elif is_structured:
            apply_steps = max(apply_steps, 40)
            route = "structured"
        else:
            route = "single"

        return PipelineRoute(
            pipeline=route,
            needs_erase=False,
            erase_steps=0,
            erase_guidance=0.0,
            apply_steps=apply_steps,
            apply_guidance=apply_guidance,
            is_cross=False,
            is_draped=is_draped,
            is_structured=is_structured,
            is_layered=target_profile.is_layered,
            family=family,
        )

    # Cross-category: need erase stage
    if is_draped:
        route = "draped"
        erase_steps = max(erase_steps_env, 50)
        erase_guidance = max(erase_guidance_env, 5.5)
    elif is_structured:
        route = "structured"
        erase_steps = max(erase_steps_env, 40)
        erase_guidance = max(erase_guidance_env, 5.0)
    elif extends_below:
        # Long upper garment crossing into lower body
        route = "cross_category"
        erase_steps = erase_steps_env
        erase_guidance = erase_guidance_env
    else:
        route = "cross_category"
        erase_steps = erase_steps_env
        erase_guidance = erase_guidance_env

    return PipelineRoute(
        pipeline=route,
        needs_erase=True,
        erase_steps=erase_steps,
        erase_guidance=erase_guidance,
        apply_steps=apply_steps,
        apply_guidance=apply_guidance,
        is_cross=True,
        is_draped=is_draped,
        is_structured=is_structured,
        is_layered=target_profile.is_layered,
        family=family,
    )


# ═══════════════════════════════════════════════════════════════════════
# Phase 5 — Geometry-Aware Alignment
# ═══════════════════════════════════════════════════════════════════════
# Approximates garment-to-body alignment using contour analysis and
# geometric normalization. Uses only existing dependencies (cv2, numpy).

@dataclass
class AlignmentTransform:
    """Geometric alignment transform for garment-to-body mapping."""
    scale_x: float = 1.0
    scale_y: float = 1.0
    offset_x: int = 0
    offset_y: int = 0
    flip_horizontal: bool = False
    center_y_ratio: float = 0.5  # where garment center should align on body


def compute_garment_alignment(
    garment_img: Image.Image,
    target_profile: GarmentProfile,
    schp_labels: np.ndarray | None = None,
) -> AlignmentTransform:
    """Compute geometric alignment transform for garment-to-body mapping.

    Uses contour analysis on the garment image to determine:
      - How much to scale the garment to match body proportions
      - Where to position the garment vertically on the body
      - Whether to flip for correct orientation
      - Center alignment based on garment type

    This is the closest feasible alignment without a dedicated warping model.
    It ensures the garment's canonical shape is properly oriented and scaled
    before IP-Adapter conditioning.
    """
    import cv2

    arr = np.array(garment_img.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]

    # Find garment foreground (non-white region)
    # Raw garment images have white (255) backgrounds — use white threshold.
    is_white = np.all(arr > 240, axis=2)
    fg = (~is_white).astype(np.uint8) * 255

    if not np.any(fg):
        return AlignmentTransform()

    # Contour analysis
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return AlignmentTransform()

    largest = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest)

    # Vertical alignment based on garment family
    if target_profile.family == "lower":
        # Lower body: garment should align to bottom of canvas
        center_y_ratio = 0.75
    elif target_profile.cloth_type in ("dresses", "full_body"):
        # Full body: garment centered
        center_y_ratio = 0.5
    elif target_profile.extends_below_waist:
        # Long upper: garment starts at top, extends down
        center_y_ratio = 0.45
    elif target_profile.is_cropped:
        # Cropped: garment in upper third
        center_y_ratio = 0.35
    else:
        # Standard upper: garment in upper half
        center_y_ratio = 0.45

    # Scale factor: garment bbox should fill ~70% of target canvas.
    # Use uniform scaling to preserve garment aspect ratio.
    target_fill = 0.70
    scale_x = (w * target_fill) / max(cw, 1)
    scale_y = (h * target_fill) / max(ch, 1)
    uniform_scale = min(scale_x, scale_y)

    # Offset to center garment horizontally
    offset_x = (w - cw) // 2 - x
    offset_y = int(h * center_y_ratio - (y + ch // 2))

    return AlignmentTransform(
        scale_x=round(uniform_scale, 3),
        scale_y=round(uniform_scale, 3),
        offset_x=offset_x,
        offset_y=offset_y,
        flip_horizontal=False,
        center_y_ratio=round(center_y_ratio, 3),
    )


def apply_garment_alignment(
    garment_img: Image.Image,
    transform: AlignmentTransform,
    target_size: tuple[int, int] = (768, 1024),
) -> Image.Image:
    """Apply alignment transform to garment image.

    Rescales, offsets, and centers the garment on the target canvas
    based on the computed AlignmentTransform.
    """
    import cv2

    arr = np.array(garment_img.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    tw, th = target_size

    # Scale
    new_w = max(1, int(w * transform.scale_x))
    new_h = max(1, int(h * transform.scale_y))
    if new_w != w or new_h != h:
        arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Create canvas and paste — mid-gray (128) matches the preprocessing service.
    # handler.py silhouette detection uses abs(pixel-128) < 40 for background.
    canvas = np.full((th, tw, 3), 128, dtype=np.uint8)
    paste_x = max(0, min(tw - new_w, (tw - new_w) // 2 + transform.offset_x))
    paste_y = max(0, min(th - new_h, int(th * transform.center_y_ratio - new_h // 2) + transform.offset_y))

    # Clip to canvas
    src_x1 = max(0, -paste_x)
    src_y1 = max(0, -paste_y)
    dst_x1 = max(0, paste_x)
    dst_y1 = max(0, paste_y)
    copy_w = min(new_w - src_x1, tw - dst_x1)
    copy_h = min(new_h - src_y1, th - dst_y1)

    if copy_w > 0 and copy_h > 0:
        canvas[dst_y1:dst_y1 + copy_h, dst_x1:dst_x1 + copy_w] = \
            arr[src_y1:src_y1 + copy_h, src_x1:src_x1 + copy_w]

    return Image.fromarray(canvas)


def get_profile_editable_labels(profile: GarmentProfile) -> set[int]:
    """Get SCHP labels that are editable for this garment profile.

    This is the core of difference-based editing: the editable labels
    determine which body regions the inpaint mask covers.
    """
    labels: set[int] = set()

    if profile.covers_upper:
        labels |= {_LABEL_UPPER_CLOTHES}

    if profile.covers_lower:
        labels |= {_LABEL_PANTS, _LABEL_SKIRT,
                    _LABEL_LEFT_LEG, _LABEL_RIGHT_LEG}

    if profile.covers_arms or profile.expose_arms:
        labels |= {_LABEL_LEFT_ARM, _LABEL_RIGHT_ARM}

    # Full body garments cover everything
    if profile.cloth_type in ("dresses", "full_body"):
        labels |= {
            _LABEL_UPPER_CLOTHES, _LABEL_DRESS,
            _LABEL_PANTS, _LABEL_SKIRT, _LABEL_SCARF,
            _LABEL_LEFT_ARM, _LABEL_RIGHT_ARM,
            _LABEL_LEFT_LEG, _LABEL_RIGHT_LEG,
        }

    return labels


def get_profile_protect_labels(profile: GarmentProfile) -> set[int]:
    """Get SCHP labels that should be protected for this garment profile.

    Protect = identity + body regions NOT covered by the target garment.
    """
    labels: set[int] = set(_IDENTITY_PROTECT_LABELS)

    # All garment labels — using corrected LIP labels.
    all_garment = (
        {_LABEL_UPPER_CLOTHES, _LABEL_DRESS, _LABEL_PANTS,
         _LABEL_SKIRT, _LABEL_SCARF}
        | {_LABEL_LEFT_ARM, _LABEL_RIGHT_ARM}
        | {_LABEL_LEFT_LEG, _LABEL_RIGHT_LEG}
    )

    # Editable labels from profile
    editable = get_profile_editable_labels(profile)

    # Protect everything NOT in editable set
    non_target = all_garment - editable
    labels |= non_target

    # Arm handling: if garment exposes arms, don't protect arms
    if profile.expose_arms:
        labels -= {_LABEL_LEFT_ARM, _LABEL_RIGHT_ARM}

    # Drape handling: protect only hands, not full arms
    if profile.is_draped:
        labels -= {_LABEL_LEFT_ARM, _LABEL_RIGHT_ARM}
        # Will be handled by _hand_zones_from_arms in the protect mask builder

    return labels


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

    # Check for coat/outerwear (LIP label 4 = upper_clothes) — high confidence indicator
    coat_px = label_counts.get(_LABEL_UPPER_CLOTHES, 0)
    if coat_px / max(garment_px, 1) > 0.15:
        return "upper_body"

    # Check for dress (LIP label 7) — covers most of body
    dress_px = label_counts.get(_LABEL_DRESS, 0)
    if dress_px / max(garment_px, 1) > 0.30:
        return "dresses"

    # Check for scarf (LIP label 17) — indicates draped garment
    scarf_px = label_counts.get(_LABEL_SCARF, 0)
    if scarf_px / max(garment_px, 1) > 0.10:
        return "dresses"

    # Check for pants (LIP label 6) — lower body dominant
    pants_px = label_counts.get(_LABEL_PANTS, 0)
    skirt_px = label_counts.get(_LABEL_SKIRT, 0)
    lower_px = pants_px + skirt_px
    if lower_px / max(garment_px, 1) > 0.40:
        return "lower_body"

    # Check for upper clothes (LIP label 4) — upper body dominant
    upper_px = label_counts.get(_LABEL_UPPER_CLOTHES, 0)
    if upper_px / max(garment_px, 1) > 0.40:
        return "upper_body"

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

    Uses contour analysis on the non-background region to estimate bounding box
    coverage, aspect ratio, and sleeve/length/width hints.

    Background detection: pixels within 40 levels of mid-gray (128,128,128)
    are treated as background. This works because the preprocessing canvas
    uses mid-gray, and garment colors are rarely mid-gray.
    """
    arr = np.array(garment_img.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]

    # Detect mid-gray background (128,128,128 ± 40) — canvas is mid-gray
    is_bg = np.all(np.abs(arr.astype(np.int16) - 128) < 40, axis=2)
    fg = (~is_bg).astype(np.uint8) * 255

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
    profile: "GarmentProfile | None" = None,
) -> np.ndarray:
    """Build binary inpaint mask from SCHP labels using GarmentProfile.

    Uses GarmentProfile-driven difference-based editing:

    Same-category (source and target are same cloth_type):
      Mask = SOURCE garment labels. The model replaces texture/color within the
      source garment's shape — the source shape IS the edit region.

    Cross-category (source and target are different cloth_types):
      Mask = TARGET body region labels (from GarmentProfile) PLUS SOURCE garment
      labels with adaptive buffer. The GarmentProfile determines exactly which
      body regions the target covers (e.g. jacket covers upper+lower, crop_top
      covers upper only).

    255 = editable, 0 = protected.
    """
    # Build profile if not provided
    if profile is None:
        profile = build_garment_profile(garment_subtype, cloth_type, garment_img_info)

    # Get editable labels from profile
    target_labels = get_profile_editable_labels(profile)

    # Determine if this is a cross-category edit
    is_cross = (
        source_cloth_type
        and source_cloth_type != cloth_type
        and source_cloth_type != "unknown"
    )

    if is_cross:
        source_labels = _CLOTHING_LABELS.get(source_cloth_type, set())

        # Base mask: target body region labels (what the target covers)
        mask = np.isin(schp_labels, list(target_labels)).astype(np.uint8) * 255

        # Add source garment labels — ensures old garment is included
        source_present = np.isin(schp_labels, list(source_labels)).astype(np.uint8) * 255
        mask = np.maximum(mask, source_present)

        # Dilated buffer around SOURCE labels — ensures clean erasure
        if np.any(source_present):
            ks = _adaptive_buffer_ks(schp_labels, source_labels, garment_img_info=garment_img_info)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
            source_dilated = cv2.dilate(source_present, kernel, iterations=1)
            ke = max(3, ks - 4)
            if ke % 2 == 0:
                ke += 1
            kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ke, ke))
            source_tight = cv2.erode(source_dilated, kernel_e, iterations=1)
            mask = np.maximum(mask, source_tight)

        # Include arm labels if source OR target is draped
        _drape_ct = cloth_type if cloth_type in ("dresses", "full_body") else (
            source_cloth_type if source_cloth_type in ("dresses", "full_body") else ""
        )
        if _drape_ct:
            arm_mask = np.isin(schp_labels, list(_DRAPE_ARM_LABELS)).astype(np.uint8) * 255
            mask = np.maximum(mask, arm_mask)
            # For draped cross-category, also include leg labels since saree/lehenga
            # covers lower body. This ensures the mask covers pallu drape over legs.
            leg_labels = {_LABEL_LEFT_LEG, _LABEL_RIGHT_LEG}
            leg_mask = np.isin(schp_labels, list(leg_labels)).astype(np.uint8) * 255
            mask = np.maximum(mask, leg_mask)
            # Include scarf label for pallu/dupatta drape over shoulder
            scarf_mask = (schp_labels == _LABEL_SCARF).astype(np.uint8) * 255
            mask = np.maximum(mask, scarf_mask)

        # Include body regions where source overlaps but target doesn't cover
        # This ensures old garment regions that are outside target coverage
        # are still included for erasure
        source_only_labels = source_labels - target_labels
        if source_only_labels:
            source_only_mask = np.isin(schp_labels, list(source_only_labels)).astype(np.uint8) * 255
            mask = np.maximum(mask, source_only_mask)
    else:
        # Same-category: source garment labels
        source_cloth = source_cloth_type if source_cloth_type and source_cloth_type != "unknown" else cloth_type
        source_labels = _CLOTHING_LABELS.get(source_cloth, _CLOTHING_LABELS.get(cloth_type, set()))
        mask = np.isin(schp_labels, list(source_labels)).astype(np.uint8) * 255

        # Include arm + leg + scarf labels for draped targets.
        # Saree/lehenga/dupatta drape covers arms, legs, and has scarf/pallu
        # components. Without legs+scarf in the mask, the model can't generate
        # drape over legs or pallu hanging over shoulder.
        if profile.is_draped:
            _drape_extra = _DRAPE_ARM_LABELS + (_LABEL_LEFT_LEG, _LABEL_RIGHT_LEG, _LABEL_SCARF)
            drape_mask = np.isin(schp_labels, list(_drape_extra)).astype(np.uint8) * 255
            mask = np.maximum(mask, drape_mask)

        # Garment geometry expansion: if the target garment extends beyond
        # the source shape (e.g. short sleeve → long sleeve, cropped → long),
        # expand the mask to include adjacent body regions.
        geo = get_garment_geometry(garment_subtype)
        h, w = schp_labels.shape
        if geo.expansion_down > 0 or geo.expansion_up > 0 or geo.expansion_width > 0:
            # Find the bounding box of the current mask
            mask_ys, mask_xs = np.where(mask > 127)
            if len(mask_ys) > 0:
                y_min, y_max = int(mask_ys.min()), int(mask_ys.max())
                x_min, x_max = int(mask_xs.min()), int(mask_xs.max())

                # Expand vertically
                y_min_exp = max(0, y_min - geo.expansion_up)
                y_max_exp = min(h, y_max + geo.expansion_down)

                # Expand horizontally
                x_min_exp = max(0, x_min - geo.expansion_width)
                x_max_exp = min(w, x_max + geo.expansion_width)

                # Create expansion zone: pixels in expanded bbox that are
                # body labels (not background, not identity) and not already masked
                expansion_zone = np.zeros_like(mask)
                expansion_zone[y_min_exp:y_max_exp, x_min_exp:x_max_exp] = 255

                # Only include body-region pixels in the expansion
                body_labels = set(range(4, 19)) - _IDENTITY_PROTECT_LABELS
                body_mask = np.isin(schp_labels, list(body_labels)).astype(np.uint8) * 255
                expansion_body = np.minimum(expansion_zone, body_mask)

                # Add expansion to mask, but only where source wasn't already covering
                mask = np.maximum(mask, expansion_body)

                if np.any(expansion_body > 127):
                    logger.info(
                        "garment_geometry_expansion subtype=%s down=%d up=%d width=%d "
                        "expanded_px=%d",
                        garment_subtype, geo.expansion_down, geo.expansion_up,
                        geo.expansion_width, int(np.sum(expansion_body > 127)),
                    )

    # Always exclude identity labels from the inpaint mask
    identity_mask = np.isin(schp_labels, list(_IDENTITY_PROTECT_LABELS)).astype(np.uint8) * 255
    mask = np.where(identity_mask > 0, 0, mask).astype(np.uint8)

    return mask


def build_schp_protect_mask(
    schp_labels: np.ndarray,
    cloth_type: str,
    garment_subtype: str = "",
    dilate_px: int = 7,
    profile: "GarmentProfile | None" = None,
) -> np.ndarray:
    """Build binary protect mask using GarmentProfile.

    255 = protected (identity-critical), 0 = editable.

    Uses GarmentProfile-driven target-aware protection:
      1. Always protect identity labels (face, hair, shoes, hat, gloves)
      2. Protect body regions NOT covered by the target garment (from profile)
      3. Garment-aware arm protection (expose_arms, is_draped)

    The profile determines exactly which body regions the target covers,
    and the protect mask is the complement. This replaces hardcoded rules
    with per-garment profiling.
    """
    # Build profile if not provided
    if profile is None:
        profile = build_garment_profile(garment_subtype, cloth_type)

    # Get protect labels from profile
    protect_labels = get_profile_protect_labels(profile)

    # Start with identity + non-target body regions
    mask = np.isin(schp_labels, list(protect_labels)).astype(np.uint8) * 255

    # Arm protection — garment-aware
    if profile.is_draped:
        # Draped: protect only HANDS (distal arm), not full forearms
        hand_zone = _hand_zones_from_arms(schp_labels)
        mask = np.maximum(mask, hand_zone)
    elif not profile.expose_arms:
        # Standard garments: protect full arms
        mask = np.maximum(mask, np.isin(schp_labels, list(_DRAPE_ARM_LABELS)).astype(np.uint8) * 255)
    # else: sleeveless garment — arms NOT protected

    if profile.is_draped:
        dilate_px = max(5, dilate_px - 2)

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
    profile: "GarmentProfile | None" = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full mask pipeline: GarmentProfile-driven difference-based editing.

    Builds a GarmentProfile for the target garment (or uses the provided one),
    then uses it to compute:
      1. Inpaint mask — editable regions based on target coverage + source erasure
      2. Protect mask — identity + non-target body regions
      3. Final mask — inpaint minus protect

    The GarmentProfile drives everything:
      - Which body regions are editable (covers_upper, covers_lower, etc.)
      - Which regions are protected (complement of coverage)
      - Arm behavior (expose_arms, is_draped)
      - Buffer dilation (adaptive to garment geometry)

    Args:
        schp_labels: SCHP label map from person image.
        cloth_type: Target garment's cloth_type.
        garment_subtype: Target garment's specific subtype (e.g. "jacket", "saree").
        source_cloth_type: Person's current garment cloth_type.
        garment_img_info: Target garment image geometry.
        trace_id: Debug trace ID.
    """
    # 1. Build GarmentProfile (or use provided one)
    if profile is None:
        profile = build_garment_profile(garment_subtype, cloth_type, garment_img_info)

    logger.info(
        "mask_profile subtype=%s family=%s covers_upper=%s covers_lower=%s "
        "extends_below=%s expose_arms=%s is_draped=%s is_cropped=%s",
        garment_subtype, profile.family, profile.covers_upper, profile.covers_lower,
        profile.extends_below_waist, profile.expose_arms, profile.is_draped,
        profile.is_cropped,
    )

    # 2. Build inpaint mask — GarmentProfile-driven
    inpaint_raw = build_schp_inpaint_mask(
        schp_labels, cloth_type, garment_subtype, source_cloth_type,
        garment_img_info, profile=profile,
    )

    # 3. Build protect mask — GarmentProfile-driven
    protect = build_schp_protect_mask(
        schp_labels, cloth_type, garment_subtype, profile=profile,
    )

    # 4. Mild dilation for edge blending
    h, w = schp_labels.shape
    scale = max(1.0, h / 512.0)
    ks = max(3, int(5 * scale))
    if ks % 2 == 0:
        ks += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    inpaint_dilated = cv2.dilate(inpaint_raw, kernel, iterations=1)

    # 5. Apply protection (subtract identity from editable)
    final = apply_protection_binary(inpaint_dilated, protect)

    # 6. Debug artifacts
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


def feather_mask_edges(binary_mask: np.ndarray, feather_px: int = 4) -> np.ndarray:
    """Create a soft gradient at mask boundaries for smoother blending.

    Uses distance-transform-based feathering: interior pixels stay at 1.0,
    exterior pixels stay at 0.0, and only the boundary zone gets a smooth
    cosine-gradient. The gradient spans feather_px pixels on each side of
    the boundary.

    Applied AFTER all binary safety checks, just before passing the mask
    to the diffusion model.

    Args:
        binary_mask: uint8 array with values 0 or 255.
        feather_px: Half-width of the feather zone in pixels (each side).

    Returns:
        float32 array with values in [0.0, 1.0].
    """
    if feather_px < 1:
        return (binary_mask > 127).astype(np.float32)

    binary = (binary_mask > 127).astype(np.uint8)

    # Distance from each foreground pixel to the nearest background pixel
    dist_inside = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    # Distance from each background pixel to the nearest foreground pixel
    dist_outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)

    # Signed distance: positive inside, negative outside, zero at boundary
    signed_dist = dist_inside - dist_outside

    # Cosine gradient within feather zone:
    # signed_dist < -feather_px → 0.0 (exterior)
    # signed_dist > feather_px → 1.0 (interior)
    # Within [-feather_px, +feather_px] → smooth cosine ramp
    t = np.clip(signed_dist / max(feather_px, 1), -1.0, 1.0)
    # Map [-1, 1] to [0, 1] then apply cosine smoothing
    t_01 = (t + 1.0) * 0.5  # [0, 1]
    result = 0.5 * (1.0 - np.cos(np.pi * t_01))

    return result.astype(np.float32)


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


# ═══════════════════════════════════════════════════════════════════════
# Phase 7 — Complete Debug Artifact System
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DebugArtifacts:
    """Complete debug artifact collection for a single inference run."""
    trace_id: str = ""
    # Garment understanding
    garment_profile: "GarmentProfile | None" = None
    garment_img_info: "GarmentImageInfo | None" = None
    alignment_transform: "AlignmentTransform | None" = None
    pipeline_route: "PipelineRoute | None" = None
    # Body analysis
    source_cloth_type: str = ""
    target_cloth_type: str = ""
    schp_labels: "np.ndarray | None" = None
    # Masks
    inpaint_mask_np: "np.ndarray | None" = None
    protect_mask_np: "np.ndarray | None" = None
    final_mask_np: "np.ndarray | None" = None
    # Results
    raw_output: "Image.Image | None" = None
    final_output: "Image.Image | None" = None
    # Processed images
    processed_garment: "Image.Image | None" = None
    garment_silhouette_np: "np.ndarray | None" = None
    face_restoration_output: "Image.Image | None" = None
    pose_output: "Image.Image | None" = None
    # Quality
    quality_metrics: dict[str, object] | None = None
    # Timing
    timing_ms: dict[str, float] = field(default_factory=dict)
    # Scores
    candidate_scores: list[dict[str, object]] = field(default_factory=list)
    # Routing decision
    routing_decision: str = ""
    # Warnings
    warnings: list[str] = field(default_factory=list)


def save_debug_artifacts_v2(
    artifacts: DebugArtifacts,
    person_img: "Image.Image | None" = None,
    garment_img: "Image.Image | None" = None,
) -> str:
    """Save complete debug artifact collection for pipeline observability.

    Saves to /tmp/idm-vton-debug/{trace_id}/ with:
      - garment_profile.json
      - garment_img_info.json
      - alignment_transform.json
      - pipeline_route.json
      - schp_labels.png
      - inpaint_mask.png
      - protect_mask.png
      - final_mask.png
      - mask_overlay.png
      - person_input.png
      - garment_input.png
      - raw_output.png
      - final_output.png
      - timing.json
      - candidate_scores.json
      - routing_decision.json

    Returns the debug directory path.
    """
    from pathlib import Path
    import json

    if not artifacts.trace_id:
        return ""

    debug_dir = Path("/tmp/idm-vton-debug") / artifacts.trace_id
    debug_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Garment profile
        if artifacts.garment_profile:
            p = artifacts.garment_profile
            profile_data = {
                "family": p.family, "cloth_type": p.cloth_type,
                "covers_upper": p.covers_upper, "covers_lower": p.covers_lower,
                "covers_arms": p.covers_arms, "covers_hands": p.covers_hands,
                "covers_torso_full": p.covers_torso_full,
                "extends_below_waist": p.extends_below_waist,
                "has_sleeves": p.has_sleeves, "sleeve_length": p.sleeve_length,
                "expose_arms": p.expose_arms, "is_fitted": p.is_fitted,
                "is_structured": p.is_structured, "is_loose": p.is_loose,
                "is_draped": p.is_draped, "has_pallu": p.has_pallu,
                "has_border": p.has_border, "is_layered": p.is_layered,
                "is_cropped": p.is_cropped, "is_voluminous": p.is_voluminous,
                "is_ethnic": p.is_ethnic,
            }
            (debug_dir / "garment_profile.json").write_text(json.dumps(profile_data, indent=2))

        # Garment image info
        if artifacts.garment_img_info:
            g = artifacts.garment_img_info
            info_data = {
                "bbox_area_ratio": g.bbox_area_ratio, "aspect_ratio": g.aspect_ratio,
                "width_ratio": g.width_ratio, "height_ratio": g.height_ratio,
                "center_y_ratio": g.center_y_ratio, "has_sleeves": g.has_sleeves,
                "is_long": g.is_long, "is_wide": g.is_wide,
            }
            (debug_dir / "garment_img_info.json").write_text(json.dumps(info_data, indent=2))

        # Alignment transform
        if artifacts.alignment_transform:
            a = artifacts.alignment_transform
            align_data = {
                "scale_x": a.scale_x, "scale_y": a.scale_y,
                "offset_x": a.offset_x, "offset_y": a.offset_y,
                "flip_horizontal": a.flip_horizontal,
                "center_y_ratio": a.center_y_ratio,
            }
            (debug_dir / "alignment_transform.json").write_text(json.dumps(align_data, indent=2))

        # Pipeline route
        if artifacts.pipeline_route:
            r = artifacts.pipeline_route
            route_data = {
                "pipeline": r.pipeline, "needs_erase": r.needs_erase,
                "erase_steps": r.erase_steps, "erase_guidance": r.erase_guidance,
                "apply_steps": r.apply_steps, "apply_guidance": r.apply_guidance,
                "is_cross": r.is_cross, "is_draped": r.is_draped,
                "is_structured": r.is_structured, "is_layered": r.is_layered,
                "family": r.family,
            }
            (debug_dir / "pipeline_route.json").write_text(json.dumps(route_data, indent=2))

        # SCHP labels
        if artifacts.schp_labels is not None:
            colors = np.zeros((*artifacts.schp_labels.shape, 3), dtype=np.uint8)
            label_colors = [
                (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
                (0, 0, 128), (255, 0, 0), (255, 128, 0), (255, 255, 0),
                (128, 128, 128), (0, 255, 0), (0, 128, 128), (128, 0, 128),
                (0, 255, 255), (255, 0, 255), (128, 64, 0), (64, 128, 0),
                (0, 64, 128), (0, 128, 64), (64, 0, 128), (128, 0, 64),
            ]
            for i, c in enumerate(label_colors):
                if i < 20:
                    colors[artifacts.schp_labels == i] = c
            Image.fromarray(colors).save(str(debug_dir / "schp_labels.png"))

        # Masks
        for mask_arr, name in [
            (artifacts.inpaint_mask_np, "inpaint_mask"),
            (artifacts.protect_mask_np, "protect_mask"),
            (artifacts.final_mask_np, "final_mask"),
        ]:
            if mask_arr is not None:
                Image.fromarray(mask_arr, mode="L").save(str(debug_dir / f"{name}.png"))

        # Mask overlay
        if person_img is not None and artifacts.final_mask_np is not None:
            bg = person_img.convert("RGB").copy()
            if bg.size != (artifacts.final_mask_np.shape[1], artifacts.final_mask_np.shape[0]):
                bg = bg.resize((artifacts.final_mask_np.shape[1], artifacts.final_mask_np.shape[0]), Image.LANCZOS)
            arr = np.array(bg, dtype=np.uint8)
            m = artifacts.final_mask_np > 127
            arr[m, 0] = (arr[m, 0].astype(np.uint16) + 255) // 2
            arr[m, 1] = (arr[m, 1].astype(np.uint16) + 80) // 2
            arr[m, 2] = (arr[m, 2].astype(np.uint16) + 80) // 2
            Image.fromarray(arr).save(str(debug_dir / "mask_overlay.png"))

        # Person and garment inputs
        if person_img is not None:
            person_img.convert("RGB").save(str(debug_dir / "person_input.png"))
        if garment_img is not None:
            garment_img.convert("RGB").save(str(debug_dir / "garment_input.png"))

        # Processed garment (after alignment/resize)
        if artifacts.processed_garment is not None:
            artifacts.processed_garment.convert("RGB").save(str(debug_dir / "processed_garment.png"))

        # Garment silhouette
        if artifacts.garment_silhouette_np is not None:
            sil_img = Image.fromarray(
                (artifacts.garment_silhouette_np > 127).astype(np.uint8) * 255, mode="L"
            )
            sil_img.save(str(debug_dir / "garment_silhouette.png"))

        # Face restoration output
        if artifacts.face_restoration_output is not None:
            artifacts.face_restoration_output.save(str(debug_dir / "face_restoration_output.png"))

        # Pose / DensePose output
        if artifacts.pose_output is not None:
            artifacts.pose_output.convert("RGB").save(str(debug_dir / "pose_output.png"))

        # Raw and final output
        if artifacts.raw_output is not None:
            artifacts.raw_output.save(str(debug_dir / "raw_output.png"))
        if artifacts.final_output is not None:
            artifacts.final_output.save(str(debug_dir / "final_output.png"))

        # Timing
        if artifacts.timing_ms:
            (debug_dir / "timing.json").write_text(json.dumps(artifacts.timing_ms, indent=2))

        # Candidate scores
        if artifacts.candidate_scores:
            (debug_dir / "candidate_scores.json").write_text(
                json.dumps(artifacts.candidate_scores, indent=2, default=str)
            )

        # Quality metrics
        if artifacts.quality_metrics:
            (debug_dir / "quality_metrics.json").write_text(
                json.dumps(artifacts.quality_metrics, indent=2, default=str)
            )

        # Routing decision
        routing_data = {
            "decision": artifacts.routing_decision,
            "source_cloth_type": artifacts.source_cloth_type,
            "target_cloth_type": artifacts.target_cloth_type,
            "warnings": artifacts.warnings,
        }
        (debug_dir / "routing_decision.json").write_text(json.dumps(routing_data, indent=2))

        logger.info("debug_artifacts_v2_saved dir=%s", debug_dir)
        return str(debug_dir)

    except Exception as exc:
        logger.warning("debug_artifacts_v2_failed error=%s", exc)
        return ""


# ═══════════════════════════════════════════════════════════════════════
# Phase 8 — Production Safety
# ═══════════════════════════════════════════════════════════════════════

def validate_pipeline_inputs(
    schp_labels: np.ndarray | None,
    cloth_type: str,
    garment_subtype: str,
    source_cloth_type: str,
) -> list[str]:
    """Validate all pipeline inputs before processing. Returns list of warnings."""
    warnings: list[str] = []

    if schp_labels is None:
        warnings.append("schp_labels_none")
    elif schp_labels.ndim != 2:
        warnings.append(f"schp_labels_wrong_ndim:{schp_labels.ndim}")
    elif schp_labels.shape[0] < 10 or schp_labels.shape[1] < 10:
        warnings.append(f"schp_labels_degenerate:{schp_labels.shape}")

    valid_cloth_types = {"upper_body", "lower_body", "dresses", "full_body"}
    if cloth_type not in valid_cloth_types:
        warnings.append(f"invalid_cloth_type:{cloth_type}")

    if garment_subtype and len(garment_subtype) > 100:
        warnings.append(f"garment_subtype_too_long:{len(garment_subtype)}")

    valid_source_types = {"upper_body", "lower_body", "dresses", "full_body", "unknown", ""}
    if source_cloth_type not in valid_source_types:
        warnings.append(f"invalid_source_cloth_type:{source_cloth_type}")

    return warnings


def safe_build_profile(
    garment_subtype: str,
    cloth_type: str,
    garment_img_info: "GarmentImageInfo | None" = None,
) -> GarmentProfile:
    """Build garment profile with safety fallback. Never raises."""
    try:
        return build_garment_profile(garment_subtype, cloth_type, garment_img_info)
    except Exception as exc:
        logger.warning("safe_build_profile_fallback error=%s subtype=%s", exc, garment_subtype)
        return GarmentProfile(cloth_type=cloth_type)


def safe_build_mask(
    schp_labels: np.ndarray,
    cloth_type: str,
    garment_subtype: str,
    source_cloth_type: str,
    garment_img_info: "GarmentImageInfo | None" = None,
    profile: "GarmentProfile | None" = None,
    trace_id: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build masks with safety fallback. Never raises."""
    try:
        if profile is None:
            profile = safe_build_profile(garment_subtype, cloth_type, garment_img_info)

        inpaint = build_schp_inpaint_mask(
            schp_labels, cloth_type, garment_subtype, source_cloth_type,
            garment_img_info, profile=profile,
        )
        protect = build_schp_protect_mask(
            schp_labels, cloth_type, garment_subtype, profile=profile,
        )
        h, w = schp_labels.shape
        scale = max(1.0, h / 512.0)
        ks = max(3, int(5 * scale))
        if ks % 2 == 0:
            ks += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        inpaint_dilated = cv2.dilate(inpaint, kernel, iterations=1)
        final = apply_protection_binary(inpaint_dilated, protect)

        return final, inpaint_dilated, protect

    except Exception as exc:
        logger.warning("safe_build_mask_fallback error=%s trace_id=%s", exc, trace_id)
        h, w = schp_labels.shape
        # Fallback: full body mask (safest — lets model generate everything)
        fallback = np.ones((h, w), dtype=np.uint8) * 255
        identity_mask = np.isin(schp_labels, list(_IDENTITY_PROTECT_LABELS)).astype(np.uint8) * 255
        fallback = np.where(identity_mask > 0, 0, fallback).astype(np.uint8)
        protect = identity_mask
        return fallback, fallback, protect


def validate_mask_safety(
    final_mask: np.ndarray,
    inpaint_mask: np.ndarray,
    protect_mask: np.ndarray,
    cloth_type: str,
    garment_subtype: str = "",
    trace_id: str = "",
) -> list[str]:
    """Validate mask outputs for production safety. Returns list of issues."""
    issues: list[str] = []

    try:
        validate_mask_integrity(final_mask, "final_mask")
    except ValueError as e:
        issues.append(f"final_mask_invalid:{e}")

    try:
        validate_mask_integrity(inpaint_mask, "inpaint_mask")
    except ValueError as e:
        issues.append(f"inpaint_mask_invalid:{e}")

    try:
        validate_mask_integrity(protect_mask, "protect_mask")
    except ValueError as e:
        issues.append(f"protect_mask_invalid:{e}")

    # Check mask sizes match
    if final_mask.shape != inpaint_mask.shape:
        issues.append("mask_shape_mismatch:final_vs_inpaint")
    if final_mask.shape != protect_mask.shape:
        issues.append("mask_shape_mismatch:final_vs_protect")

    # Check coverage is reasonable
    final_coverage = float(np.mean(final_mask > 127))
    if final_coverage < 0.01:
        issues.append(f"final_mask_near_empty:{final_coverage:.4f}")
    elif final_coverage > 0.95:
        issues.append(f"final_mask_near_full:{final_coverage:.4f}")

    if issues:
        logger.warning("mask_safety_issues trace_id=%s issues=%s", trace_id, issues)

    return issues
