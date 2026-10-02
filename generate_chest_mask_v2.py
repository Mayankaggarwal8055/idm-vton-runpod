"""
Generate a mask that simply fills in the chest gap of the ORIGINAL successful mask.
Instead of adding a new elliptical shape, we just extend the existing mask straight up
to the neck keypoint, maintaining the same width as the original mask at Y=321.
"""

import numpy as np
from PIL import Image
import cv2

# Load the ORIGINAL mask that produced the good result (full shirt with buttons)
original_mask = np.array(Image.open('tests/output_debug/downloaded_mask_inspect.png'))
print(f"Original mask Y range: {np.where(np.any(original_mask > 127, axis=1))[0][[0, -1]]}")

# Keypoints
NECK_Y = 280
NECK_X = 395
SHOULDER_R_X = 265
SHOULDER_L_X = 545
SHOULDER_Y = 345
RIGHT_WRIST = (235, 380)
LEFT_WRIST = (370, 630)

# ──────────────────────────────────────────────────────────────────────────────
# Strategy: Simply extend the original mask upward to neck keypoint.
# At each row from Y=280 to Y=321, set the mask width to interpolate between
# a narrower width at neck (centered on neck_x) and the full width at shoulders.
# ──────────────────────────────────────────────────────────────────────────────
new_mask = original_mask.copy()

# Find the original mask's X extents at the first few rows where it has coverage
# to understand the shape
for y in range(321, 350):
    cols = np.where(original_mask[y, :] > 127)[0]
    if len(cols) > 5:
        print(f"  Original at Y={y}: X={cols[0]}-{cols[-1]} (width={cols[-1]-cols[0]+1})")
        break

# At Y=345 (shoulder level), the original mask spans X=291-659
# At Y=321 (original top), it has a very narrow start
# We want to smoothly connect neck (Y=280, narrow ~150px centered at 395)
# to the original mask top (Y=325-340 where it gets wider)

# Find the original mask's width at Y=340 (a stable row)
cols_340 = np.where(original_mask[340, :] > 127)[0]
if len(cols_340) > 0:
    orig_left_340 = cols_340[0]
    orig_right_340 = cols_340[-1]
    print(f"  Original at Y=340: X={orig_left_340}-{orig_right_340}")

# Neck width at Y=280 (should be moderate - about 180px centered)
neck_half_width = 90  # 180px total width at neck

# Shoulder width at Y=345 - use the existing mask width
shoulder_left = 265  # SHOULDER_R_X
shoulder_right = 545  # SHOULDER_L_X

for y in range(NECK_Y, 350):
    # Linear interpolation from neck to shoulders
    t = (y - NECK_Y) / (SHOULDER_Y - NECK_Y)  # 0 at neck, 1 at shoulders
    # Use smooth ease-in curve for more natural shape
    t_smooth = t * t  # quadratic ease-in (narrow at top, expanding faster near shoulders)
    
    half_width = neck_half_width + (shoulder_right - shoulder_left) / 2 * t_smooth
    left = int(NECK_X - half_width)
    right = int(NECK_X + half_width)
    
    # Clip to image bounds
    left = max(0, left)
    right = min(767, right)
    
    # Set mask
    new_mask[y, left:right+1] = 255

# Ensure we keep ALL original mask pixels (union)
new_mask = np.maximum(new_mask, original_mask)

# ──────────────────────────────────────────────────────────────────────────────
# Hand protection - carve out hand regions
# ──────────────────────────────────────────────────────────────────────────────
# Right hand (finger-heart at 235, 380)
cv2.circle(new_mask, (RIGHT_WRIST[0], RIGHT_WRIST[1]), 60, 0, -1)
# Left wrist (resting at 370, 630)
cv2.circle(new_mask, (LEFT_WRIST[0], LEFT_WRIST[1]), 50, 0, -1)

# ──────────────────────────────────────────────────────────────────────────────
# Apply 3px Gaussian blur for smooth edges (subtle, don't change mask shape)
# ──────────────────────────────────────────────────────────────────────────────
new_mask = cv2.GaussianBlur(new_mask, (5, 5), 1.0)
# Threshold to keep it mostly binary but with soft edges
new_mask = np.where(new_mask > 80, 255, 0).astype(np.uint8)

# ──────────────────────────────────────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────────────────────────────────────
rows_with_white = np.where(np.any(new_mask > 127, axis=1))[0]
coverage = np.sum(new_mask > 127) / new_mask.size * 100
print(f"\n=== EXTENDED MASK STATS ===")
print(f"Y range: {rows_with_white[0]} to {rows_with_white[-1]}")
print(f"Coverage: {coverage:.1f}%")
print(f"Coverage < 55%: {'PASS' if coverage < 55 else 'FAIL'}")

for y in [278, 280, 285, 290, 300, 310, 320, 330, 340, 350, 400, 500, 600, 617]:
    row = new_mask[y, :]
    white_cols = np.where(row > 127)[0]
    if len(white_cols) > 0:
        print(f"  Y={y}: X={white_cols[0]}-{white_cols[-1]} ({len(white_cols)}px)")
    else:
        print(f"  Y={y}: not covered")

# Hand checks
print(f"\nRight hand (235,380): {new_mask[380, 235]}")
print(f"Left wrist (370,630): {new_mask[630, 370]}")

# Save
Image.fromarray(new_mask).save('tests/output_debug/chest_extended_v2_mask.png')
print(f"\nSaved: tests/output_debug/chest_extended_v2_mask.png")

# Create overlay
person_np = np.array(Image.open('tests/output_debug/gkkralk0kue5rjjgecn0.jpg').resize((768, 1024)))
overlay = person_np.copy()
mask_bool = new_mask > 127
overlay[mask_bool, 0] = (overlay[mask_bool, 0] * 0.4).astype(np.uint8)
overlay[mask_bool, 1] = np.clip(overlay[mask_bool, 1] * 0.4 + 150, 0, 255).astype(np.uint8)
overlay[mask_bool, 2] = (overlay[mask_bool, 2] * 0.4).astype(np.uint8)

# Mark keypoints
cv2.circle(overlay, (NECK_X, NECK_Y), 5, (0, 0, 255), -1)
cv2.circle(overlay, (RIGHT_WRIST[0], RIGHT_WRIST[1]), 60, (255, 0, 0), 2)
cv2.circle(overlay, (LEFT_WRIST[0], LEFT_WRIST[1]), 50, (255, 0, 0), 2)

Image.fromarray(overlay).save('tests/output_debug/chest_extended_v2_overlay.jpg', quality=95)
print("Saved: tests/output_debug/chest_extended_v2_overlay.jpg")
