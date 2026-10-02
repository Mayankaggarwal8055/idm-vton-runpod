"""
Generate a new inpainting mask that extends the original good mask upward
to cover the chest/cleavage area up to the neck keypoint (Y=280).

Key changes from original mask:
1. Keep ALL existing coverage (Y=321-617) - preserves full shirt length
2. Extend upward with elliptical arc from shoulders (Y=345) to neck (Y=280)
3. Fill the chest region between neck and garment top
4. Carve out hand protection zones
5. Apply soft feathering at the top boundary (15px distance transform)
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2
import os

# ──────────────────────────────────────────────────────────────────────────────
# Keypoints (768x1024 canvas)
# ──────────────────────────────────────────────────────────────────────────────
NECK_Y = 280          # upper_neck keypoint
NECK_X = 395          # center neck
SHOULDER_R_X = 265    # right shoulder X
SHOULDER_L_X = 545    # left shoulder X
SHOULDER_Y = 345      # shoulder Y
RIGHT_WRIST = (235, 380)   # finger-heart hand
LEFT_WRIST = (370, 630)    # resting on midriff

# ──────────────────────────────────────────────────────────────────────────────
# Load original mask (the one that produced the good full-shirt result)
# ──────────────────────────────────────────────────────────────────────────────
original_mask = np.array(Image.open('tests/output_debug/downloaded_mask_inspect.png'))
print(f"Original mask shape: {original_mask.shape}")
print(f"Original Y range: {np.where(np.any(original_mask > 127, axis=1))[0][[0, -1]]}")

# ──────────────────────────────────────────────────────────────────────────────
# Create new mask starting from the original
# ──────────────────────────────────────────────────────────────────────────────
new_mask = original_mask.copy()

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: Fill the chest region with an elliptical arc from neck to shoulders
# The ellipse center is at the neck, extending down to shoulders
# ──────────────────────────────────────────────────────────────────────────────

# Create the chest fill region
# We need an elliptical shape that covers from NECK_Y down to SHOULDER_Y
# and spans the full shoulder width
ellipse_center_x = NECK_X
ellipse_center_y = NECK_Y
ellipse_width = (SHOULDER_L_X - SHOULDER_R_X) + 60   # shoulder span + margin
ellipse_height = (SHOULDER_Y - NECK_Y) * 2  # full height from neck to shoulders

# Draw filled ellipse on the mask
mask_pil = Image.fromarray(new_mask)
draw = ImageDraw.Draw(mask_pil)

# Elliptical arc: top of ellipse at NECK_Y, bottom reaches below SHOULDER_Y
# bbox = [left, top, right, bottom]
left = ellipse_center_x - ellipse_width // 2
right = ellipse_center_x + ellipse_width // 2
top = NECK_Y - 5   # slightly above neck for smooth curve
bottom = NECK_Y + (SHOULDER_Y - NECK_Y) * 2 + 10  # extend below shoulders

draw.ellipse([left, top, right, bottom], fill=255)
new_mask = np.array(mask_pil)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: Also fill the rectangular gap between ellipse bottom and existing mask
# Ensure continuous coverage from the ellipse down through the original mask
# ──────────────────────────────────────────────────────────────────────────────
# Find the X extents of original mask at each row
for y in range(NECK_Y, 350):
    # At each row, find the union of ellipse and original mask coverage
    row_orig = original_mask[y, :]
    row_new = new_mask[y, :]
    # If original mask has white pixels at this row, extend to cover gap
    orig_white = np.where(row_orig > 127)[0]
    new_white = np.where(row_new > 127)[0]
    if len(orig_white) > 0 and len(new_white) > 0:
        fill_left = min(orig_white[0], new_white[0])
        fill_right = max(orig_white[-1], new_white[-1])
        new_mask[y, fill_left:fill_right+1] = 255

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: Ensure the original mask coverage is fully preserved (union)
# ──────────────────────────────────────────────────────────────────────────────
new_mask = np.maximum(new_mask, original_mask)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: Hand protection - carve out hand regions
# ──────────────────────────────────────────────────────────────────────────────
hand_mask = np.zeros_like(new_mask)

# Right hand (finger-heart gesture at 235, 380) - 65px radius
cv2.circle(hand_mask, (RIGHT_WRIST[0], RIGHT_WRIST[1]), 65, 255, -1)

# Left forearm/wrist (resting at 370, 630) - 55px radius
cv2.circle(hand_mask, (LEFT_WRIST[0], LEFT_WRIST[1]), 55, 255, -1)

# Subtract hands from mask
new_mask[hand_mask > 127] = 0

# ──────────────────────────────────────────────────────────────────────────────
# STEP 5: Apply soft feathering at the TOP boundary (15px distance transform)
# Only at the top edge, NOT the sides or bottom
# ──────────────────────────────────────────────────────────────────────────────
# Create a version with Gaussian blur just at the top boundary region
feather_zone = new_mask.copy()
# Only apply feathering to the top 20 rows of the mask
top_row = np.where(np.any(new_mask > 127, axis=1))[0]
if len(top_row) > 0:
    mask_top = top_row[0]
    feather_region_end = mask_top + 20
    
    # Create distance transform for soft edge at top
    # Invert just the top region
    top_strip = new_mask[max(0, mask_top-15):feather_region_end+5, :].copy()
    if np.any(top_strip > 0):
        dist = cv2.distanceTransform(top_strip, cv2.DIST_L2, 5)
        # Normalize distance to [0, 1] with 15px feather
        feather_px = 15
        alpha = np.clip(dist / feather_px, 0, 1)
        feathered_strip = (top_strip.astype(np.float32) * alpha).astype(np.uint8)
        new_mask[max(0, mask_top-15):feather_region_end+5, :] = feathered_strip

# ──────────────────────────────────────────────────────────────────────────────
# STEP 6: Clean up - ensure binary-ish mask (threshold at 64 to keep soft edges)
# ──────────────────────────────────────────────────────────────────────────────
# Apply slight Gaussian blur for smooth edges overall
new_mask_blurred = cv2.GaussianBlur(new_mask, (5, 5), 1.5)
# Threshold: keep pixels above 64 for soft boundary
new_mask_final = np.where(new_mask_blurred > 64, new_mask_blurred, 0).astype(np.uint8)

# ──────────────────────────────────────────────────────────────────────────────
# VERIFICATION
# ──────────────────────────────────────────────────────────────────────────────
rows_with_white = np.where(np.any(new_mask_final > 127, axis=1))[0]
coverage = np.sum(new_mask_final > 127) / new_mask_final.size * 100
print(f"\n=== NEW MASK STATS ===")
print(f"Y range: {rows_with_white[0]} to {rows_with_white[-1]}")
print(f"Coverage: {coverage:.1f}%")
print(f"Coverage < 55% check: {'PASS' if coverage < 55 else 'FAIL'}")

# Check chest coverage specifically
for y in [275, 280, 285, 290, 295, 300, 310, 320, 330, 340, 350]:
    row = new_mask_final[y, :]
    white_cols = np.where(row > 127)[0]
    if len(white_cols) > 0:
        print(f"  Y={y}: covered X={white_cols[0]}-{white_cols[-1]} ({len(white_cols)}px)")
    else:
        print(f"  Y={y}: NOT covered")

# Check hand carveouts
print(f"\nHand protection at right wrist (235,380): mask value = {new_mask_final[380, 235]}")
print(f"Hand protection at left wrist (370,630): mask value = {new_mask_final[630, 370]}")

# ──────────────────────────────────────────────────────────────────────────────
# SAVE
# ──────────────────────────────────────────────────────────────────────────────
os.makedirs('tests/output_debug', exist_ok=True)
Image.fromarray(new_mask_final).save('tests/output_debug/chest_extended_mask.png')
print(f"\nSaved: tests/output_debug/chest_extended_mask.png")

# ──────────────────────────────────────────────────────────────────────────────
# Create overlay visualization
# ──────────────────────────────────────────────────────────────────────────────
person_img = Image.open('tests/output_debug/gkkralk0kue5rjjgecn0.jpg').resize((768, 1024))
person_np = np.array(person_img)

# Create green overlay where mask > 127
overlay = person_np.copy()
mask_bool = new_mask_final > 127
overlay[mask_bool, 0] = (overlay[mask_bool, 0] * 0.4).astype(np.uint8)  # reduce red
overlay[mask_bool, 1] = np.clip(overlay[mask_bool, 1] * 0.4 + 150, 0, 255).astype(np.uint8)  # boost green
overlay[mask_bool, 2] = (overlay[mask_bool, 2] * 0.4).astype(np.uint8)  # reduce blue

# Draw hand carveout circles in red
cv2.circle(overlay, (RIGHT_WRIST[0], RIGHT_WRIST[1]), 65, (255, 0, 0), 2)
cv2.circle(overlay, (LEFT_WRIST[0], LEFT_WRIST[1]), 55, (255, 0, 0), 2)

# Draw neck keypoint
cv2.circle(overlay, (NECK_X, NECK_Y), 5, (0, 0, 255), -1)
cv2.putText(overlay, f"neck Y={NECK_Y}", (NECK_X+10, NECK_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Draw shoulder line
cv2.line(overlay, (SHOULDER_R_X, SHOULDER_Y), (SHOULDER_L_X, SHOULDER_Y), (0, 0, 255), 1)

Image.fromarray(overlay).save('tests/output_debug/chest_extended_mask_overlay.jpg', quality=95)
print("Saved: tests/output_debug/chest_extended_mask_overlay.jpg")

# Also create a side-by-side comparison of old vs new mask
comparison = np.zeros((1024, 768*2), dtype=np.uint8)
comparison[:, :768] = original_mask
comparison[:, 768:] = new_mask_final
Image.fromarray(comparison).save('tests/output_debug/mask_comparison_old_vs_new.png')
print("Saved: tests/output_debug/mask_comparison_old_vs_new.png")
