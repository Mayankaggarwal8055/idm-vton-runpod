"""
Category-Aware Semantic Masking Engine for Virtual Try-On.

Provides deterministic, category-specific mask dispatch, adaptive elliptical morphology,
clavicle/neckline triangle exposure, ethnic dupatta isolation, and anti-aliased
latent-space mask injection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("idm-vton.mask_generator")

def safe_float(val: Any, default: Any = 0.0) -> Any:
    if val is None:
        return default
    if hasattr(val, "item"):
        try:
            return float(val.item())
        except Exception:
            pass
    while isinstance(val, (list, tuple)):
        if len(val) == 0:
            return default
        val = val[0]
    if hasattr(val, "item"):
        try:
            return float(val.item())
        except Exception:
            pass
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def safe_int(val: Any, default: Any = 0) -> Any:
    if val is None:
        return default
    if hasattr(val, "item"):
        try:
            return int(val.item())
        except Exception:
            pass
    while isinstance(val, (list, tuple)):
        if len(val) == 0:
            return default
        val = val[0]
    if hasattr(val, "item"):
        try:
            return int(val.item())
        except Exception:
            pass
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


# =============================================================================
# SCHP (ATR Dataset) Canonical Label Map
# =============================================================================
LABEL_BACKGROUND = 0
LABEL_HAT = 1
LABEL_HAIR = 2
LABEL_SUNGLASSES = 3
LABEL_UPPER_CLOTHES = 4
LABEL_SKIRT = 5
LABEL_PANTS = 6
LABEL_DRESS = 7
LABEL_BELT = 8
LABEL_LEFT_SHOE = 9
LABEL_RIGHT_SHOE = 10
LABEL_FACE = 11
LABEL_LEFT_LEG = 12
LABEL_RIGHT_LEG = 13
LABEL_LEFT_ARM = 14
LABEL_RIGHT_ARM = 15
LABEL_BAG = 16
LABEL_SCARF = 17       # Scarf / Dupatta / Chunni / Stole / Pallu
LABEL_NECK = 18

ALL_CLOTHING_LABELS = frozenset({
    LABEL_UPPER_CLOTHES,
    LABEL_SKIRT,
    LABEL_PANTS,
    LABEL_DRESS,
    LABEL_BELT,
    LABEL_SCARF,
})


# =============================================================================
# Adaptive Kernel Helper
# =============================================================================
def get_adaptive_kernel(height: int, width: int, scale_h: float = 0.035, scale_w: float = 0.020) -> np.ndarray:
    """
    Construct an elliptical morphological structuring element scaled dynamically
    to image dimensions to prevent over-dilation on small frames or under-dilation on HD.
    """
    kh = max(3, int(round(height * scale_h)))
    kw = max(3, int(round(width * scale_w)))
    if kh % 2 == 0:
        kh += 1
    if kw % 2 == 0:
        kw += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh))


# =============================================================================
# Soft Mask Edge via Distance Transform
# =============================================================================
def compute_soft_mask_edge(
    binary_mask: np.ndarray,
    d_min: float = 2.0,
    d_max: float = 14.0,
) -> np.ndarray:
    """
    Generate smooth, continuous transition weights using Euclidean Distance Transform.
    
    Formula:
        M_soft = Clip((D(M) - d_min) / (d_max - d_min), 0.0, 1.0)
    """
    mask_u8 = (binary_mask > 127).astype(np.uint8)
    if not np.any(mask_u8):
        return np.zeros_like(binary_mask, dtype=np.float32)
        
    # Distance from background into mask interior
    dist_inside = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    # Distance from mask into exterior
    dist_outside = cv2.distanceTransform(1 - mask_u8, cv2.DIST_L2, 5)
    
    # Signed distance: positive inside, negative outside
    signed_dist = dist_inside - dist_outside
    
    soft_mask = np.clip((signed_dist - d_min) / (d_max - d_min), 0.0, 1.0)
    return soft_mask.astype(np.float32)

COLLARED_OR_CREW_SUBTYPES = frozenset({
    "shirt", "tshirt", "t_shirt", "t-shirt", "tee", "hoodie", "jacket",
    "blazer", "polo", "sweater", "coat", "cardigan", "sweatshirt",
})

# =============================================================================
# Anatomical Neckline Generation
# =============================================================================
def generate_anatomical_neckline(
    mask_np: np.ndarray,
    keypoints: Optional[Dict[str, Any]],
    height: int,
    width: int,
    schp: Optional[np.ndarray] = None,
    garment_subtype: str = "",
) -> np.ndarray:
    """
    Generates a natural curved neckline using elliptical arcs derived from
    shoulder and neck keypoints + 15px Euclidean distance transform soft mask
    instead of a hard polygon or sharp bounding box.

    If the target garment is collared/crewneck (shirt, t-shirt, hoodie) and the
    person is wearing a low-cut/scoop top, expands the inpaint mask upward to the
    base of the neck keypoint to give the model canvas to draw the collar.
    """
    neck_pt = None
    l_sh = None
    r_sh = None

    if keypoints:
        def _get_pt(names: List[str]) -> Optional[Tuple[int, int]]:
            for n in names:
                if n in keypoints:
                    val = keypoints[n]
                    while isinstance(val, (list, tuple)) and len(val) == 1:
                        val = val[0]
                    if isinstance(val, (list, tuple)) and len(val) >= 2:
                        x = safe_float(val[0], default=0.0)
                        y = safe_float(val[1], default=0.0)
                        if x <= 1.0 and y <= 1.0:
                            return safe_int(x * width), safe_int(y * height)
                        return safe_int(x), safe_int(y)
                    elif hasattr(val, "x") and hasattr(val, "y"):
                        x = safe_float(val.x, default=0.0)
                        y = safe_float(val.y, default=0.0)
                        return safe_int(x * width), safe_int(y * height)
            return None

        neck_pt = _get_pt(["neck", "upper_neck", "nose"])
        l_sh = _get_pt(["left_shoulder", "l_shoulder", "left_sh"])
        r_sh = _get_pt(["right_shoulder", "r_shoulder", "right_sh"])
    elif schp is not None:
        # Fallback to SCHP neck region geometry if keypoints are unavailable
        neck_mask = (schp == LABEL_NECK).astype(np.uint8)
        neck_coords = np.where(neck_mask > 0)
        if len(neck_coords[0]) > 0:
            ny_min = int(np.min(neck_coords[0]))
            ny_max = int(np.max(neck_coords[0]))
            nx_min = int(np.min(neck_coords[1]))
            nx_max = int(np.max(neck_coords[1]))
            center_x = (nx_min + nx_max) // 2
            neck_w = max(16, nx_max - nx_min)
            l_sh = (max(0, center_x - neck_w * 2), ny_max)
            r_sh = (min(width, center_x + neck_w * 2), ny_max)
            neck_pt = (center_x, (ny_min + ny_max) // 2)

    if l_sh and r_sh:
        lx, ly = l_sh
        rx, ry = r_sh
        sh_dist = abs(rx - lx)
        if sh_dist > 0:
            center_x = (lx + rx) // 2
            neck_y = neck_pt[1] if neck_pt else int((ly + ry) // 2 - sh_dist * 0.15)
            axes_x = max(10, sh_dist // 2)

            # Check if target garment is collared/crewneck
            sub = (garment_subtype or "").lower().replace("-", "_").strip()
            is_collared_or_crew = sub in COLLARED_OR_CREW_SUBTYPES or not sub

            # Check for low-cut / scoop top gap in center chest columns
            c_min = max(0, center_x - int(sh_dist * 0.15))
            c_max = min(width, center_x + int(sh_dist * 0.15))
            center_band = mask_np[:, c_min:c_max]
            rows_with_garment = np.where(center_band.any(axis=1))[0]
            garment_top_y = int(rows_with_garment[0]) if len(rows_with_garment) > 0 else height

            gap = max(0, garment_top_y - neck_y)
            if is_collared_or_crew and gap > 30:
                axes_y = max(8, int(sh_dist * 0.28), gap + 10)
            else:
                axes_y = max(8, int(sh_dist * 0.28))

            # Create elliptical arc between neck and shoulder keypoints (no hard polygons)
            neck_curve = np.zeros_like(mask_np)
            cv2.ellipse(neck_curve, (center_x, neck_y), (axes_x, axes_y), 0, 0, 180, 255, -1)

            # Apply 15px Euclidean distance transform (soft alpha gradient) at skin boundary
            soft = compute_soft_mask_edge(neck_curve, d_min=0.0, d_max=15.0)
            mask_np = np.maximum(mask_np, (soft * 255).astype(np.uint8))

    return mask_np

# =============================================================================
# Hand Protection
# =============================================================================
def protect_hands(
    schp: np.ndarray,
    keypoints: Optional[Dict[str, Any]],
    height: int,
    width: int,
    inpaint_region: Optional[np.ndarray] = None,
    buffer_px: int = 15,
) -> np.ndarray:
    """
    Extract wrist/hand keypoints, generate buffered protection zone M_hands,
    and intersect with the upper-body inpaint region to strictly enforce:
        M_agn ∩ M_hands = ∅
    Hands and crossed arms must never be included in inpainting mask.
    """
    arm_mask = np.isin(schp, [LABEL_LEFT_ARM, LABEL_RIGHT_ARM]).astype(np.uint8) * 255
    prot_mask = arm_mask.copy()

    box_size = max(35, int(width * 0.08)) + buffer_px

    if keypoints:
        def _get_pt(names: List[str]) -> Optional[Tuple[int, int]]:
            for n in names:
                if n in keypoints:
                    val = keypoints[n]
                    while isinstance(val, (list, tuple)) and len(val) == 1:
                        val = val[0]
                    if isinstance(val, (list, tuple)) and len(val) >= 2:
                        x = safe_float(val[0], default=0.0)
                        y = safe_float(val[1], default=0.0)
                        if x <= 1.0 and y <= 1.0:
                            return safe_int(x * width), safe_int(y * height)
                        return safe_int(x), safe_int(y)
                    elif hasattr(val, "x") and hasattr(val, "y"):
                        x = safe_float(val.x, default=0.0)
                        y = safe_float(val.y, default=0.0)
                        return safe_int(x * width), safe_int(y * height)
            return None

        found_wrist = False
        for wrist_names in [
            ["left_wrist", "l_wrist", "wrist_left", "left_hand"],
            ["right_wrist", "r_wrist", "wrist_right", "right_hand"],
        ]:
            wrist = _get_pt(wrist_names)
            if wrist:
                found_wrist = True
                wx, wy = wrist
                # Draw buffered circle for the hand/wrist with buffer_px
                cv2.circle(prot_mask, (wx, wy), box_size, 255, -1)

                # Include nearby arm pixels within dilated bounding box
                x1 = max(0, wx - box_size)
                y1 = max(0, wy - box_size)
                x2 = min(width, wx + box_size)
                y2 = min(height, wy + box_size)
                prot_mask[y1:y2, x1:x2] = np.maximum(prot_mask[y1:y2, x1:x2], arm_mask[y1:y2, x1:x2])

        if found_wrist and buffer_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (buffer_px * 2 + 1, buffer_px * 2 + 1))
            prot_mask = cv2.dilate(prot_mask, k, iterations=1)
    else:
        # Fallback if no keypoints: protect any arm pixels crossing or touching the torso
        prot_mask = arm_mask.copy()

    # Intersect with inpaint region + buffer to isolate M_hands if inpaint region provided
    if inpaint_region is not None and np.any(inpaint_region > 127) and keypoints:
        m_hands = cv2.bitwise_and(prot_mask, inpaint_region)
        if np.any(m_hands > 127) and buffer_px > 0:
            k_buf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (buffer_px * 2 + 1, buffer_px * 2 + 1))
            m_hands = cv2.dilate(m_hands, k_buf, iterations=1)
            prot_mask = cv2.bitwise_or(prot_mask, m_hands)

    return prot_mask

# =============================================================================
# Category Mask Dispatcher
# =============================================================================
class CategoryMaskDispatcher:
    """
    Deterministic category dispatcher generating semantic inpaint and protect masks.
    """
    
    @staticmethod
    def dispatch(
        schp_map: np.ndarray,
        category: str,
        garment_subtype: str = "",
        keypoints: Optional[Dict[str, Any]] = None,
        preserve_dupatta: bool = False,
        preserve_outer_jacket: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = schp_map.shape[:2]
        cat = category.lower().replace("-", "_").strip()
        sub = garment_subtype.lower().replace("-", "_").strip()
        
        if cat in ("upper_body", "upper", "top", "tops"):
            return CategoryMaskDispatcher._upper_body(schp_map, sub, keypoints, preserve_dupatta, preserve_outer_jacket)
        elif cat in ("lower_body", "lower", "bottom", "bottom_wear", "pants", "jeans"):
            return CategoryMaskDispatcher._lower_body(schp_map, sub, keypoints)
        elif cat in ("ethnic", "ethnic_wear", "kurta", "saree"):
            return CategoryMaskDispatcher._ethnic_wear(schp_map, sub, keypoints, preserve_dupatta)
        elif cat in ("dresses", "dress", "full_body", "full_outfit", "jumpsuit"):
            return CategoryMaskDispatcher._full_body(schp_map, sub, keypoints)
        else:
            return CategoryMaskDispatcher._upper_body(schp_map, sub, keypoints, preserve_dupatta, preserve_outer_jacket)

    @staticmethod
    def _upper_body(
        schp: np.ndarray,
        subtype: str,
        keypoints: Optional[Dict[str, Any]],
        preserve_dupatta: bool,
        preserve_outer_jacket: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = schp.shape[:2]
        
        inpaint_labels = {LABEL_UPPER_CLOTHES, LABEL_DRESS}
        
        if not preserve_dupatta:
            inpaint_labels.add(LABEL_SCARF)
            
        inpaint_mask = np.isin(schp, list(inpaint_labels)).astype(np.uint8) * 255
        
        inpaint_mask = generate_anatomical_neckline(
            inpaint_mask, keypoints, h, w, schp=schp, garment_subtype=subtype
        )
        
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 15))
        inpaint_mask = cv2.morphologyEx(inpaint_mask, cv2.MORPH_CLOSE, close_k)
        
        # Directional dilation: horizontal ellipse for sleeve coverage
        dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(w * 0.04) | 1), max(3, int(h * 0.015) | 1)))
        
        # Identify waistband boundary
        waist_y = h
        lower_body_mask = np.isin(schp, [LABEL_SKIRT, LABEL_PANTS, LABEL_BELT])
        rows = np.where(lower_body_mask.any(axis=1))[0]
        if len(rows) > 0:
            waist_y = int(rows[0])
            
        # Dilate but clip downward dilation at waistband boundary
        dilated_mask = cv2.dilate(inpaint_mask, dilate_k, iterations=1)
        if waist_y < h:
            dilated_mask[waist_y:h, :] = inpaint_mask[waist_y:h, :]
        inpaint_mask = dilated_mask

        # Extend downward by up to 50px past detected hem to catch peeking underlayer hems
        rows_inpaint = np.where(inpaint_mask.any(axis=1))[0]
        if len(rows_inpaint) > 0:
            bottom_y = int(rows_inpaint[-1])
            target_bottom = min(h, bottom_y + 50)
            if waist_y < h:
                target_bottom = min(target_bottom, waist_y)
            if target_bottom > bottom_y:
                bottom_band = inpaint_mask[max(0, bottom_y - 15):bottom_y, :]
                col_sums = np.sum(bottom_band > 127, axis=0)
                nonzero_cols = np.where(col_sums > 0)[0]
                if len(nonzero_cols) > 0:
                    left_b = max(0, int(nonzero_cols[0]) - 10)
                    right_b = min(w, int(nonzero_cols[-1]) + 10)
                    inpaint_mask[bottom_y:target_bottom, left_b:right_b] = 255
            
        protected_labels = {
            LABEL_HAT,
            LABEL_HAIR,
            LABEL_SUNGLASSES,
            LABEL_FACE,
            LABEL_LEFT_SHOE,
            LABEL_RIGHT_SHOE,
            LABEL_LEFT_LEG,
            LABEL_RIGHT_LEG,
            LABEL_BAG, # Phone occlusion / Mirror Selfie proxy
            LABEL_PANTS,
            LABEL_SKIRT,
        }
        
        if preserve_dupatta:
            protected_labels.add(LABEL_SCARF)
            
        prot_mask = np.isin(schp, list(protected_labels)).astype(np.uint8) * 255
        
        # Strict hand buffering (Solve Melted Hands on Seated/Crossed Poses)
        hand_mask = protect_hands(
            schp, keypoints, h, w, inpaint_region=inpaint_mask, buffer_px=15
        )
        prot_mask = cv2.bitwise_or(prot_mask, hand_mask)

        # Strictly subtract buffered hand regions from inpaint_mask
        inpaint_mask[hand_mask > 127] = 0
        inpaint_mask[prot_mask > 127] = 0

        return inpaint_mask, prot_mask

    @staticmethod
    def _lower_body(
        schp: np.ndarray,
        subtype: str,
        keypoints: Optional[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = schp.shape[:2]
        
        inpaint_labels = {LABEL_PANTS, LABEL_SKIRT}
        inpaint_mask = np.isin(schp, list(inpaint_labels)).astype(np.uint8) * 255
        
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 17))
        inpaint_mask = cv2.morphologyEx(inpaint_mask, cv2.MORPH_CLOSE, close_k)
        
        dilate_k = get_adaptive_kernel(h, w, scale_h=0.025, scale_w=0.018)
        inpaint_mask = cv2.dilate(inpaint_mask, dilate_k, iterations=1)
        
        # Contoured Waistband Extension (Parabolic Curve)
        waistband_curve = np.zeros_like(inpaint_mask)
        rows = np.where(inpaint_mask.any(axis=1))[0]
        if len(rows) > 0:
            mask_top = int(rows[0])
            top_band = inpaint_mask[mask_top:min(mask_top + 15, h), :]
            col_sums = np.sum(top_band > 127, axis=0)
            nonzero_cols = np.where(col_sums > 0)[0]
            if len(nonzero_cols) > 0:
                left_b = max(0, int(nonzero_cols[0]) - 10)
                right_b = min(w, int(nonzero_cols[-1]) + 10)
                band_w = right_b - left_b
                if band_w > 0:
                    x_idx = np.arange(band_w)
                    norm_x = (x_idx - band_w / 2.0) / (band_w / 2.0)
                    curve = (1.0 - 0.25 * (norm_x ** 2))
                    for col_i, col_x in enumerate(range(left_b, right_b)):
                        curr_top = max(0, mask_top - int(35 * curve[col_i]))
                        waistband_curve[curr_top:mask_top, col_x] = 255
                        
        prot_labels = {
            LABEL_BACKGROUND,
            LABEL_HAT,
            LABEL_HAIR,
            LABEL_FACE,
            LABEL_UPPER_CLOTHES,
            LABEL_SCARF,
            LABEL_NECK,
            LABEL_BAG, # Phone occlusion proxy
        }
        prot_mask = np.isin(schp, list(prot_labels)).astype(np.uint8) * 255
        
        # Allow the parabolic waistband curve to extend slightly into upper clothes
        prot_mask[waistband_curve > 0] = 0
        inpaint_mask = cv2.bitwise_or(inpaint_mask, waistband_curve)
        
        if len(rows) > 0:
            mask_top = int(rows[0])
            exclude_top = max(0, mask_top - 60)
            inpaint_mask[:exclude_top, :] = 0
            
        # Hand protection
        hand_mask = protect_hands(schp, keypoints, h, w)
        prot_mask = cv2.bitwise_or(prot_mask, hand_mask)

        inpaint_mask[prot_mask > 127] = 0
        return inpaint_mask, prot_mask

    @staticmethod
    def _ethnic_wear(
        schp: np.ndarray,
        subtype: str,
        keypoints: Optional[Dict[str, Any]],
        preserve_dupatta: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = schp.shape[:2]
        
        inpaint_labels = {LABEL_UPPER_CLOTHES, LABEL_DRESS}
        if not preserve_dupatta:
            inpaint_labels.add(LABEL_SCARF)
            
        inpaint_mask = np.isin(schp, list(inpaint_labels)).astype(np.uint8) * 255
        
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 23))
        inpaint_mask = cv2.morphologyEx(inpaint_mask, cv2.MORPH_CLOSE, close_k)
        
        dilate_k = get_adaptive_kernel(h, w, scale_h=0.035, scale_w=0.025)
        inpaint_mask = cv2.dilate(inpaint_mask, dilate_k, iterations=1)
        
        prot_labels = {
            LABEL_BACKGROUND,
            LABEL_HAT,
            LABEL_HAIR,
            LABEL_SUNGLASSES,
            LABEL_FACE,
            LABEL_LEFT_SHOE,
            LABEL_RIGHT_SHOE,
            LABEL_BAG,
        }
        if preserve_dupatta:
            prot_labels.add(LABEL_SCARF)
            
        prot_mask = np.isin(schp, list(prot_labels)).astype(np.uint8) * 255
        
        # Hand protection
        hand_mask = protect_hands(schp, keypoints, h, w)
        prot_mask = cv2.bitwise_or(prot_mask, hand_mask)

        inpaint_mask[prot_mask > 127] = 0
        return inpaint_mask, prot_mask

    @staticmethod
    def _full_body(
        schp: np.ndarray,
        subtype: str,
        keypoints: Optional[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = schp.shape[:2]
        
        inpaint_labels = {
            LABEL_UPPER_CLOTHES,
            LABEL_DRESS,
            LABEL_SKIRT,
            LABEL_PANTS,
            LABEL_BELT,
            LABEL_SCARF,
        }
        inpaint_mask = np.isin(schp, list(inpaint_labels)).astype(np.uint8) * 255
        
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 25))
        inpaint_mask = cv2.morphologyEx(inpaint_mask, cv2.MORPH_CLOSE, close_k)
        
        dilate_k = get_adaptive_kernel(h, w, scale_h=0.035, scale_w=0.025)
        inpaint_mask = cv2.dilate(inpaint_mask, dilate_k, iterations=1)
        
        bare_arm_region = np.isin(schp, [LABEL_LEFT_ARM, LABEL_RIGHT_ARM]).astype(np.uint8)
        arm_erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        bare_arm_region = cv2.erode(bare_arm_region, arm_erode_k, iterations=1)
        
        prot_labels = {
            LABEL_BACKGROUND,
            LABEL_HAT,
            LABEL_HAIR,
            LABEL_SUNGLASSES,
            LABEL_FACE,
            LABEL_LEFT_SHOE,
            LABEL_RIGHT_SHOE,
            LABEL_BAG,
        }
        prot_mask = np.isin(schp, list(prot_labels)).astype(np.uint8) * 255
        
        # Hand protection
        hand_mask = protect_hands(schp, keypoints, h, w)
        prot_mask = cv2.bitwise_or(prot_mask, hand_mask)

        inpaint_mask[prot_mask > 127] = 0
        return inpaint_mask, prot_mask


# =============================================================================
# Anti-Aliased 8x Latent Mask Downsampling
# =============================================================================
def downsample_mask_for_latent(
    mask_image: Union[Image.Image, np.ndarray],
    latent_size: Tuple[int, int] = (96, 128),  # (W/8, H/8) for (768, 1024)
) -> np.ndarray:
    if isinstance(mask_image, Image.Image):
        mask_np = np.array(mask_image.convert("L"), dtype=np.float32) / 255.0
    else:
        mask_np = mask_image.astype(np.float32)
        if mask_np.max() > 1.0:
            mask_np /= 255.0
            
    lw, lh = latent_size
    latent_mask = cv2.resize(mask_np, (lw, lh), interpolation=cv2.INTER_AREA)
    
    latent_mask = np.clip(latent_mask, 0.0, 1.0)
    return latent_mask
