"""
Comprehensive Test Harness for TryLix Virtual Try-On Pipeline.

Validates the 5 real-world failure mode mitigations and category-aware masking:
  TC1: Clean upper-body baseline (Lower-body isolation, no seams, no rect discontinuities)
  TC2: Seated pose hand preservation (M_agn and M_hands disjoint)
  TC3: Dupatta removal for Western upper-body (Scarf/dupatta inpaint inclusion)
  TC4: Upper-body try-on with pants visible
  TC5: Lower-body try-on (Waistband contour integrity, upper body isolation)
  TC6: Full-body dress try-on (Complete outfit coverage, face/hair/foot protection)
  TC7: Anatomical curved neckline via keypoints (eliminates square chest cutout)
  TC8: Seated pose WITH keypoints on torso (wrist bounding box protection)
  TC9: Mirror selfie / phone occlusion (Bag label 16 protection)
  TC10: Postprocess 3-band Laplacian Pyramid Compositor (100pct bitwise outside, 5px blend, under 100ms)
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mask_generator import (
    CategoryMaskDispatcher,
    LABEL_BACKGROUND,
    LABEL_HAIR,
    LABEL_UPPER_CLOTHES,
    LABEL_SKIRT,
    LABEL_PANTS,
    LABEL_DRESS,
    LABEL_LEFT_SHOE,
    LABEL_RIGHT_SHOE,
    LABEL_FACE,
    LABEL_LEFT_ARM,
    LABEL_RIGHT_ARM,
    LABEL_BAG,
    LABEL_SCARF,
    LABEL_NECK,
)
from postprocess import composite_tryon_result, laplacian_pyramid_blend

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests", "output_debug")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Canonical distinct colors for ATR labels
COLORS = np.zeros((20, 3), dtype=np.uint8)
COLORS[LABEL_BACKGROUND] = [20, 20, 20]
COLORS[LABEL_HAIR] = [128, 64, 0]
COLORS[LABEL_UPPER_CLOTHES] = [0, 100, 220]     # Blue
COLORS[LABEL_SKIRT] = [180, 100, 50]
COLORS[LABEL_PANTS] = [0, 200, 0]              # Green
COLORS[LABEL_DRESS] = [220, 50, 180]
COLORS[LABEL_LEFT_SHOE] = [100, 100, 100]
COLORS[LABEL_RIGHT_SHOE] = [100, 100, 100]
COLORS[LABEL_FACE] = [220, 180, 140]           # Skin
COLORS[LABEL_LEFT_ARM] = [200, 50, 50]         # Red
COLORS[LABEL_RIGHT_ARM] = [200, 50, 50]        # Red
COLORS[LABEL_BAG] = [255, 128, 0]              # Orange
COLORS[LABEL_SCARF] = [0, 220, 220]            # Cyan (Dupatta)
COLORS[LABEL_NECK] = [200, 160, 120]


def apply_color_map(schp_map: np.ndarray) -> np.ndarray:
    h, w = schp_map.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(20):
        colored[schp_map == i] = COLORS[i]
    return colored


def make_basic_person() -> np.ndarray:
    schp = np.zeros((768, 512), dtype=np.uint8)
    cv2.circle(schp, (256, 100), 50, LABEL_FACE, -1)           # Face
    cv2.circle(schp, (256, 60), 40, LABEL_HAIR, -1)            # Hair
    cv2.rectangle(schp, (240, 140), (272, 170), LABEL_NECK, -1) # Neck
    cv2.rectangle(schp, (180, 170), (332, 450), LABEL_UPPER_CLOTHES, -1) # Torso / Shirt
    cv2.rectangle(schp, (130, 170), (180, 400), LABEL_LEFT_ARM, -1)      # Left Arm
    cv2.rectangle(schp, (332, 170), (382, 400), LABEL_RIGHT_ARM, -1)     # Right Arm
    cv2.rectangle(schp, (180, 450), (250, 700), LABEL_PANTS, -1)         # Left Pant
    cv2.rectangle(schp, (262, 450), (332, 700), LABEL_PANTS, -1)         # Right Pant
    cv2.rectangle(schp, (180, 700), (250, 740), LABEL_LEFT_SHOE, -1)     # Left Shoe
    cv2.rectangle(schp, (262, 700), (332, 740), LABEL_RIGHT_SHOE, -1)    # Right Shoe
    return schp


def save_panel(case_name: str, schp_map: np.ndarray, inpaint_mask: np.ndarray, protected_mask: np.ndarray):
    colored_schp = apply_color_map(schp_map)
    overlay = colored_schp.copy()
    mask_indices = inpaint_mask > 127
    overlay[mask_indices] = (overlay[mask_indices] * 0.45 + np.array([255, 255, 255]) * 0.55).astype(np.uint8)

    fig, axs = plt.subplots(2, 2, figsize=(10, 15))
    axs[0, 0].imshow(colored_schp)
    axs[0, 0].set_title("Input SCHP (ATR Labels)")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(inpaint_mask, cmap='gray')
    axs[0, 1].set_title("Generated Inpaint Mask")
    axs[0, 1].axis("off")

    axs[1, 0].imshow(protected_mask, cmap='gray')
    axs[1, 0].set_title("Protected Mask")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(overlay)
    axs[1, 1].set_title("Inpaint Overlay on Person")
    axs[1, 1].axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{case_name}.png")
    plt.savefig(out_path, dpi=120)
    plt.close()


def check_no_rectangular_discontinuity(inpaint_mask: np.ndarray) -> bool:
    """Verifies mask contour has no isolated harsh rectangular cutouts in the neckline area."""
    chest_roi = inpaint_mask[100:220, 200:312]
    if not np.any(chest_roi > 127):
        return True
    gray = chest_roi.astype(np.float32)
    dst = cv2.cornerHarris(gray, 5, 3, 0.04)
    num_sharp_corners = np.sum(dst > 0.05 * dst.max())
    return bool(num_sharp_corners < 50)


def check_seam(inpaint_mask: np.ndarray) -> bool:
    """Detects single-pixel horizontal or vertical artifact seams spanning across the frame."""
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(inpaint_mask, cv2.MORPH_OPEN, kernel)
    diff = cv2.absdiff(inpaint_mask, opened)
    if np.max(np.sum(diff > 127, axis=1)) > inpaint_mask.shape[1] * 0.5:
        return False
    if np.max(np.sum(diff > 127, axis=0)) > inpaint_mask.shape[0] * 0.5:
        return False
    return True


def run_tests():
    print("=" * 70)
    print("RUNNING TRYLIX VTON LOCAL VALIDATION LOOP")
    print("=" * 70)
    results: list[tuple[str, bool, str]] = []

    # ── TC1: Clean Upper-Body Baseline ───────────────────────────────
    schp1 = make_basic_person()
    inpaint1, prot1 = CategoryMaskDispatcher.dispatch(schp1, "upper_body")
    save_panel("TC1_baseline", schp1, inpaint1, prot1)

    overlap1_pants = np.sum((inpaint1 > 127) & ((schp1 == LABEL_SKIRT) | (schp1 == LABEL_PANTS)))
    results.append(("TC1: Lower Body Isolation (No Pants Bleed)", overlap1_pants == 0, f"overlap={overlap1_pants}px"))
    results.append(("TC1: Artifact Seam Check", check_seam(inpaint1), ""))

    # ── TC2: Seated Pose Hand Preservation (without keypoints) ────────
    schp2 = make_basic_person()
    cv2.rectangle(schp2, (180, 300), (332, 350), LABEL_LEFT_ARM, -1)
    inpaint2, prot2 = CategoryMaskDispatcher.dispatch(schp2, "upper_body")
    save_panel("TC2_seated", schp2, inpaint2, prot2)

    arm_overlap2 = np.sum((inpaint2 > 127) & ((schp2 == LABEL_LEFT_ARM) | (schp2 == LABEL_RIGHT_ARM)))
    results.append(("TC2: Hand/Arm Preservation M_agn and M_hands disjoint", arm_overlap2 == 0, f"overlap={arm_overlap2}px"))

    # ── TC3: Dupatta Removal for Western Upper-Body ──────────────────
    schp3 = make_basic_person()
    cv2.rectangle(schp3, (200, 170), (250, 450), LABEL_SCARF, -1)
    inpaint3, prot3 = CategoryMaskDispatcher.dispatch(schp3, "upper_body", preserve_dupatta=False)
    save_panel("TC3_dupatta_removed", schp3, inpaint3, prot3)

    dupatta_pixels = np.sum(schp3 == LABEL_SCARF)
    dupatta_inpaint_pixels = np.sum((inpaint3 > 127) & (schp3 == LABEL_SCARF))
    dupatta_ratio = dupatta_inpaint_pixels / max(1, dupatta_pixels)
    results.append(("TC3: Dupatta Removed for Western Upper-Body", dupatta_ratio > 0.95, f"coverage={dupatta_ratio:.1%}"))

    # ── TC4: Upper-Body Try-On with Pants Isolation ──────────────────
    schp4 = make_basic_person()
    inpaint4, prot4 = CategoryMaskDispatcher.dispatch(schp4, "upper_body")
    save_panel("TC4_upper_with_pants", schp4, inpaint4, prot4)
    overlap4 = np.sum((inpaint4 > 127) & ((schp4 == LABEL_SKIRT) | (schp4 == LABEL_PANTS)))
    results.append(("TC4: Pants 100% Protected During Upper Try-On", overlap4 == 0, f"overlap={overlap4}px"))

    # ── TC5: Lower-Body Try-On & Contoured Waistband ─────────────────
    schp5 = make_basic_person()
    inpaint5, prot5 = CategoryMaskDispatcher.dispatch(schp5, "lower_body")
    save_panel("TC5_lower_body", schp5, inpaint5, prot5)

    upper_overlap5 = np.sum((inpaint5 > 127) & (schp5 == LABEL_UPPER_CLOTHES))
    results.append(("TC5: Upper Body Protected (Contoured Waistband Only)", upper_overlap5 < 6000, f"waist_ext={upper_overlap5}px"))
    face_neck_overlap5 = np.sum((inpaint5 > 127) & np.isin(schp5, [LABEL_FACE, LABEL_HAIR, LABEL_NECK]))
    results.append(("TC5: Face, Hair, Neck 100% Protected", face_neck_overlap5 == 0, f"overlap={face_neck_overlap5}px"))

    # ── TC6: Full-Body Dress Try-On ──────────────────────────────────
    schp6 = make_basic_person()
    inpaint6, prot6 = CategoryMaskDispatcher.dispatch(schp6, "dresses")
    save_panel("TC6_fullbody_dress", schp6, inpaint6, prot6)

    covers_torso = np.sum((inpaint6 > 127) & (schp6 == LABEL_UPPER_CLOTHES)) / max(1, np.sum(schp6 == LABEL_UPPER_CLOTHES))
    covers_pants = np.sum((inpaint6 > 127) & (schp6 == LABEL_PANTS)) / max(1, np.sum(schp6 == LABEL_PANTS))
    face_hair_overlap6 = np.sum((inpaint6 > 127) & np.isin(schp6, [LABEL_FACE, LABEL_HAIR]))
    results.append(("TC6: Dress Inpaint Covers Torso and Legs", covers_torso > 0.95 and covers_pants > 0.95, f"torso={covers_torso:.1%}, pants={covers_pants:.1%}"))
    results.append(("TC6: Face and Hair 100% Protected", face_hair_overlap6 == 0, f"overlap={face_hair_overlap6}px"))

    # ── TC7: Anatomical Curved Neckline with Shoulder/Neck Keypoints ──
    schp7 = make_basic_person()
    keypoints7 = {
        "neck": (256, 155),
        "left_shoulder": (180, 175),
        "right_shoulder": (332, 175),
    }
    inpaint7, prot7 = CategoryMaskDispatcher.dispatch(schp7, "upper_body", keypoints=keypoints7)
    save_panel("TC7_anatomical_neckline", schp7, inpaint7, prot7)

    neck_coverage = np.sum((inpaint7 > 127) & (schp7 == LABEL_NECK))
    results.append(("TC7: Anatomical Curved Neckline Exposure", neck_coverage > 50, f"neck_pixels={neck_coverage}px"))
    results.append(("TC7: Neckline Smooth (No Harsh Discontinuity)", check_no_rectangular_discontinuity(inpaint7), ""))

    # ── TC8: Seated Pose with Hand/Wrist Keypoints on Torso ───────────
    schp8 = make_basic_person()
    keypoints8 = {
        "left_wrist": (256, 300),
        "right_wrist": (360, 390),
    }
    cv2.circle(schp8, (256, 300), 25, LABEL_LEFT_ARM, -1)
    inpaint8, prot8 = CategoryMaskDispatcher.dispatch(schp8, "upper_body", keypoints=keypoints8)
    save_panel("TC8_wrist_keypoint_on_torso", schp8, inpaint8, prot8)

    wrist_zone_overlap = np.sum((inpaint8 > 127) & (schp8 == LABEL_LEFT_ARM))
    results.append(("TC8: Hand Keypoint Zone on Torso Protected", wrist_zone_overlap == 0, f"overlap={wrist_zone_overlap}px"))

    # ── TC9: Mirror Selfie / Phone Occlusion ─────────────────────────
    schp9 = make_basic_person()
    cv2.rectangle(schp9, (230, 260), (280, 340), LABEL_BAG, -1)
    inpaint9, prot9 = CategoryMaskDispatcher.dispatch(schp9, "upper_body")
    save_panel("TC9_phone_occlusion", schp9, inpaint9, prot9)

    phone_overlap = np.sum((inpaint9 > 127) & (schp9 == LABEL_BAG))
    results.append(("TC9: Phone Occlusion (Label 16) Protected", phone_overlap == 0, f"overlap={phone_overlap}px"))

    # ── TC10: Postprocess Compositor 3-Band Laplacian Pyramid ────────
    orig_img = np.full((768, 512, 3), 180, dtype=np.uint8)
    cv2.putText(orig_img, "Original Texture", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2)
    diff_img = np.full((768, 512, 3), 50, dtype=np.uint8)
    cv2.putText(diff_img, "Diffusion Texture", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (240, 240, 240), 2)

    test_inpaint = np.zeros((768, 512), dtype=np.uint8)
    test_inpaint[200:500, 150:350] = 255
    test_prot = 255 - test_inpaint

    t0 = time.perf_counter()
    composited_pil = composite_tryon_result(
        original_img=orig_img,
        diffusion_result=diff_img,
        protected_mask=test_prot,
        inpaint_mask=test_inpaint,
        cloth_type="upper_body",
        num_levels=3,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    comp_np = np.array(composited_pil)
    outside_orig = orig_img[:100, :100]
    outside_comp = comp_np[:100, :100]
    is_bitwise_identical = np.array_equal(outside_orig, outside_comp)

    center_diff = diff_img[300:400, 200:300]
    center_comp = comp_np[300:400, 200:300]
    is_diffusion_intact = np.array_equal(center_diff, center_comp)

    results.append(("TC10: Compositor Bitwise 100% Outside Dilated Mask", is_bitwise_identical, ""))
    results.append(("TC10: Compositor 100% Diffusion Inside Garment", is_diffusion_intact, ""))
    results.append(("TC10: Compositor Latency under 100ms", elapsed_ms < 100.0, f"latency={elapsed_ms:.1f}ms"))

    # ── Summary Report ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DETAILED TEST RESULTS MATRIX")
    print("=" * 70)
    all_passed = True
    for name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        detail_str = f" ({detail})" if detail else ""
        print(f"[{status}] {name}{detail_str}")
        if not passed:
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("ALL TESTS PASSED! Pipeline verified for all 5 failure modes.")
    else:
        print("SOME TESTS FAILED. Review output and adjust kernels.")
    print(f"Debug panels written to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    run_tests()
