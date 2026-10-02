"""
V3: Minimal chest extension mask.

Strategy: Take the EXACT original mask that produced the good result,
and add ONLY a narrow triangle/trapezoid fill in the chest gap 
between Y=280 and Y=321. Keep everything else identical.

The key insight is that the diffusion model is very sensitive to mask shape.
We need to add the MINIMUM possible mask coverage to close the chest gap.
"""

import numpy as np
from PIL import Image
import cv2

# Load the ORIGINAL mask (the one that produced full-length shirt with buttons)
original_mask = np.array(Image.open('tests/output_debug/downloaded_mask_inspect.png'))
print(f"Original mask Y range: {np.where(np.any(original_mask > 127, axis=1))[0][[0, -1]]}")

# Keypoints
NECK_Y = 280
NECK_X = 395
SHOULDER_Y = 345
RIGHT_WRIST = (235, 380)
LEFT_WRIST = (370, 630)

new_mask = original_mask.copy()

# ──────────────────────────────────────────────────────────────────────────────
# MINIMAL STRATEGY: Add a narrow trapezoid from neck to the existing mask top.
# At Y=280 (neck): ~120px wide centered on NECK_X (tight)
# At Y=321 (mask top): match the existing mask width at that row
# Linear interpolation between them
# ──────────────────────────────────────────────────────────────────────────────

# Find what the original mask looks like at its top rows (Y=321-340)
for y in range(321, 345):
    cols = np.where(original_mask[y, :] > 127)[0]
    if len(cols) > 20:
        mask_left_at_top = cols[0]
        mask_right_at_top = cols[-1]
        merge_y = y
        print(f"  First substantial row at Y={y}: X={mask_left_at_top}-{mask_right_at_top} (width={mask_right_at_top-mask_left_at_top+1})")
        break

# Neck width at Y=280 (tight - just enough to cover collarbone/chest)
neck_left = NECK_X - 60  # 335
neck_right = NECK_X + 60  # 455 (120px total width at neck)

for y in range(NECK_Y, merge_y + 1):
    t = (y - NECK_Y) / (merge_y - NECK_Y)  # 0 at neck, 1 at merge_y
    
    left = int(neck_left + (mask_left_at_top - neck_left) * t)
    right = int(neck_right + (mask_right_at_top - neck_right) * t)
    
    # Set mask
    new_mask[y, left:right+1] = 255

# Union with original to preserve all existing coverage
new_mask = np.maximum(new_mask, original_mask)

# ──────────────────────────────────────────────────────────────────────────────
# NO hand carveout for v3 — keep the original mask's hand protection as-is
# The original mask already had proper hand avoidance
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Light 3px blur just on the new extension area for smooth transition
# ──────────────────────────────────────────────────────────────────────────────
# Only blur the top portion where we added new coverage
blur_region = new_mask[max(0, NECK_Y-3):merge_y+3, :].copy()
blur_region = cv2.GaussianBlur(blur_region, (3, 3), 0.8)
blur_region = np.where(blur_region > 100, 255, 0).astype(np.uint8)
new_mask[max(0, NECK_Y-3):merge_y+3, :] = np.maximum(
    blur_region, 
    new_mask[max(0, NECK_Y-3):merge_y+3, :]
)

# ──────────────────────────────────────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────────────────────────────────────
rows_with_white = np.where(np.any(new_mask > 127, axis=1))[0]
coverage = np.sum(new_mask > 127) / new_mask.size * 100
print(f"\n=== V3 MINIMAL MASK ===")
print(f"Y range: {rows_with_white[0]} to {rows_with_white[-1]}")
print(f"Coverage: {coverage:.1f}% (original was {np.sum(original_mask > 127) / original_mask.size * 100:.1f}%)")
print(f"Coverage < 55%: {'PASS' if coverage < 55 else 'FAIL'}")

for y in [278, 280, 290, 300, 310, 320, 330, 340, 350, 500, 617]:
    row = new_mask[y, :]
    white_cols = np.where(row > 127)[0]
    if len(white_cols) > 0:
        print(f"  Y={y}: X={white_cols[0]}-{white_cols[-1]} ({len(white_cols)}px)")
    else:
        print(f"  Y={y}: not covered")

# Save
Image.fromarray(new_mask).save('tests/output_debug/chest_extended_v3_mask.png')
print(f"\nSaved: tests/output_debug/chest_extended_v3_mask.png")

# Overlay
person_np = np.array(Image.open('tests/output_debug/gkkralk0kue5rjjgecn0.jpg').resize((768, 1024)))
overlay = person_np.copy()
mask_bool = new_mask > 127
overlay[mask_bool, 0] = (overlay[mask_bool, 0] * 0.4).astype(np.uint8)
overlay[mask_bool, 1] = np.clip(overlay[mask_bool, 1] * 0.4 + 150, 0, 255).astype(np.uint8)
overlay[mask_bool, 2] = (overlay[mask_bool, 2] * 0.4).astype(np.uint8)

# Show the new area vs original
# Draw original mask boundary in blue
orig_bool = original_mask > 127
orig_boundary = cv2.Canny(orig_bool.astype(np.uint8) * 255, 100, 200)
overlay[orig_boundary > 0] = [0, 0, 255]

# Draw new mask boundary in red
new_boundary = cv2.Canny(mask_bool.astype(np.uint8) * 255, 100, 200)
overlay[new_boundary > 0] = [255, 0, 0]

cv2.circle(overlay, (NECK_X, NECK_Y), 5, (255, 255, 0), -1)

Image.fromarray(overlay).save('tests/output_debug/chest_extended_v3_overlay.jpg', quality=95)
print("Saved: tests/output_debug/chest_extended_v3_overlay.jpg")
