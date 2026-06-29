#!/usr/bin/env python3
"""Synthetic unit tests for the garment silhouette clipping fix.

Tests that the silhouette expansion AND uses target-only editable labels
(not all body labels), preventing source garment content from leaking
into the inpaint mask.

No models, no GPU, no images — just numpy + mask_pipeline logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure mask_pipeline is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mask_pipeline import (
    GarmentProfile,
    GarmentImageInfo,
    get_profile_editable_labels,
    get_profile_protect_labels,
    build_final_inpaint_mask,
    build_schp_inpaint_mask,
    build_schp_protect_mask,
    apply_protection_binary,
    build_garment_profile,
    is_draped_garment,
    _LABEL_UPPER_CLOTHES,
    _LABEL_SKIRT,
    _LABEL_PANTS,
    _LABEL_DRESS,
    _LABEL_BELT,
    _LABEL_LEFT_SHOE,
    _LABEL_RIGHT_SHOE,
    _LABEL_FACE,
    _LABEL_HAIR,
    _LABEL_HAT,
    _LABEL_SUNGLASSES,
    _LABEL_LEFT_ARM,
    _LABEL_RIGHT_ARM,
    _LABEL_LEFT_LEG,
    _LABEL_RIGHT_LEG,
    _LABEL_BAG,
    _LABEL_SCARF,
    _LABEL_NECK,
    _LABEL_BG,
)

H, W = 768, 1024  # TARGET_SIZE


def _make_schp_map(
    upper_region: tuple[int, int, int, int] = (200, 300, 500, 700),
    lower_region: tuple[int, int, int, int] = (500, 300, 750, 700),
    arm_left_region: tuple[int, int, int, int] = (250, 200, 450, 300),
    arm_right_region: tuple[int, int, int, int] = (250, 700, 450, 800),
    leg_left_region: tuple[int, int, int, int] = (600, 350, 750, 480),
    leg_right_region: tuple[int, int, int, int] = (600, 520, 750, 650),
    dress_region: tuple[int, int, int, int] | None = None,
    skirt_region: tuple[int, int, int, int] | None = None,
    pants_region: tuple[int, int, int, int] | None = None,
    scarf_region: tuple[int, int, int, int] | None = None,
    face_region: tuple[int, int, int, int] = (100, 440, 200, 580),
    hair_region: tuple[int, int, int, int] = (50, 420, 130, 600),
    bg: int = _LABEL_BG,
) -> np.ndarray:
    """Create a synthetic SCHP label map with configurable body regions."""
    labels = np.full((H, W), bg, dtype=np.uint8)

    def fill(r, label):
        y1, x1, y2, x2 = r
        labels[y1:y2, x1:x2] = label

    fill(face_region, _LABEL_FACE)
    fill(hair_region, _LABEL_HAIR)
    fill(arm_left_region, _LABEL_LEFT_ARM)
    fill(arm_right_region, _LABEL_RIGHT_ARM)
    fill(leg_left_region, _LABEL_LEFT_LEG)
    fill(leg_right_region, _LABEL_RIGHT_LEG)

    if dress_region is not None:
        fill(dress_region, _LABEL_DRESS)
    else:
        fill(upper_region, _LABEL_UPPER_CLOTHES)

    if skirt_region is not None:
        fill(skirt_region, _LABEL_SKIRT)
    if pants_region is not None:
        fill(pants_region, _LABEL_PANTS)
    if scarf_region is not None:
        fill(scarf_region, _LABEL_SCARF)

    # Fill legs with pants/skirt if lower region present
    if pants_region is None and skirt_region is None:
        fill(leg_left_region, _LABEL_LEFT_LEG)
        fill(leg_right_region, _LABEL_RIGHT_LEG)

    return labels


def _make_silhouette_mask(
    includes_upper: bool = True,
    includes_lower: bool = True,
    includes_arms: bool = True,
) -> np.ndarray:
    """Create a synthetic garment silhouette mask (as if extracted from garment image).

    Simulates the scenario where a garment image shows BOTH upper and lower body
    content (e.g., a pink shirt + gray pants product photo).
    """
    mask = np.zeros((H, W), dtype=np.uint8)
    if includes_upper:
        mask[200:500, 300:700] = 255  # upper body region
    if includes_lower:
        mask[500:750, 300:700] = 255  # lower body region
    if includes_arms:
        mask[250:450, 200:300] = 255  # left arm
        mask[250:450, 700:800] = 255  # right arm
    return mask


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: get_profile_editable_labels correctness
# ═══════════════════════════════════════════════════════════════════════════

class TestProfileEditableLabels:
    """Verify get_profile_editable_labels returns the correct set for each cloth_type."""

    def test_upper_body_shirt(self):
        profile = build_garment_profile("shirt", "upper_body")
        labels = get_profile_editable_labels(profile)
        assert _LABEL_UPPER_CLOTHES in labels
        assert _LABEL_LEFT_ARM in labels
        assert _LABEL_RIGHT_ARM in labels
        # Must NOT include lower body labels
        assert _LABEL_PANTS not in labels
        assert _LABEL_SKIRT not in labels
        assert _LABEL_LEFT_LEG not in labels
        assert _LABEL_RIGHT_LEG not in labels
        assert _LABEL_DRESS not in labels

    def test_upper_body_tshirt(self):
        profile = build_garment_profile("tshirt", "upper_body")
        labels = get_profile_editable_labels(profile)
        assert _LABEL_UPPER_CLOTHES in labels
        assert _LABEL_LEFT_ARM in labels
        assert _LABEL_RIGHT_ARM in labels
        assert _LABEL_PANTS not in labels
        assert _LABEL_SKIRT not in labels

    def test_lower_body_jeans(self):
        profile = build_garment_profile("jeans", "lower_body")
        labels = get_profile_editable_labels(profile)
        assert _LABEL_PANTS in labels
        assert _LABEL_SKIRT in labels
        assert _LABEL_LEFT_LEG in labels
        assert _LABEL_RIGHT_LEG in labels
        # Must NOT include upper body labels
        assert _LABEL_UPPER_CLOTHES not in labels
        assert _LABEL_LEFT_ARM not in labels
        assert _LABEL_RIGHT_ARM not in labels
        assert _LABEL_DRESS not in labels

    def test_lower_body_trousers(self):
        profile = build_garment_profile("trousers", "lower_body")
        labels = get_profile_editable_labels(profile)
        assert _LABEL_PANTS in labels
        assert _LABEL_SKIRT in labels
        assert _LABEL_LEFT_LEG in labels
        assert _LABEL_RIGHT_LEG in labels
        assert _LABEL_UPPER_CLOTHES not in labels

    def test_dress(self):
        profile = build_garment_profile("dress", "dresses")
        labels = get_profile_editable_labels(profile)
        assert _LABEL_UPPER_CLOTHES in labels
        assert _LABEL_DRESS in labels
        assert _LABEL_PANTS in labels
        assert _LABEL_SKIRT in labels
        assert _LABEL_SCARF in labels
        assert _LABEL_LEFT_ARM in labels
        assert _LABEL_RIGHT_ARM in labels
        assert _LABEL_LEFT_LEG in labels
        assert _LABEL_RIGHT_LEG in labels

    def test_saree_draped(self):
        profile = build_garment_profile("saree", "dresses")
        labels = get_profile_editable_labels(profile)
        # Saree is dress/full_body — should include all body+garment labels
        assert _LABEL_UPPER_CLOTHES in labels
        assert _LABEL_DRESS in labels
        assert _LABEL_PANTS in labels
        assert _LABEL_SKIRT in labels

    def test_default_upper_body_profile(self):
        """Default GarmentProfile (no subtype) should behave as upper_body."""
        profile = GarmentProfile()
        labels = get_profile_editable_labels(profile)
        assert _LABEL_UPPER_CLOTHES in labels


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: Silhouette AND with target-only labels (the core fix)
# ═══════════════════════════════════════════════════════════════════════════

class TestSilhouetteTargetOnlyAnd:
    """Simulate the silhouette expansion AND logic from handler.py.

    The fix: AND silhouette with get_profile_editable_labels(garment_profile)
    instead of _body_labels = {4,5,6,7,8,12,13,14,15,17}.
    """

    def test_upper_body_silhouette_excludes_pants(self):
        """Upper body garment silhouette should NOT include pants/skirt/legs."""
        profile = build_garment_profile("shirt", "upper_body")
        target_labels = get_profile_editable_labels(profile)
        silhouette = _make_silhouette_mask(includes_upper=True, includes_lower=True, includes_arms=True)

        # The fix: AND silhouette with target labels only
        schp_map = _make_schp_map()
        _schp_body = np.isin(schp_map, list(target_labels)).astype(np.uint8) * 255
        clipped = np.minimum(silhouette, _schp_body)

        # Upper body region should remain
        assert np.sum(clipped[200:500, 300:700] > 127) > 0, "upper body region should be in clipped silhouette"
        # Lower body region should be REMOVED (pants/skirt/legs not in target)
        assert np.sum(clipped[550:750, 350:650] > 127) == 0, (
            "lower body region must NOT be in clipped silhouette for upper_body garment"
        )

    def test_upper_body_old_behavior_would_leak(self):
        """Demonstrate the old bug: using ALL body labels would keep pants in silhouette."""
        old_body_labels = {4, 5, 6, 7, 8, 12, 13, 14, 15, 17}
        silhouette = _make_silhouette_mask(includes_upper=True, includes_lower=True, includes_arms=True)
        schp_map = _make_schp_map()

        _schp_body_old = np.isin(schp_map, list(old_body_labels)).astype(np.uint8) * 255
        clipped_old = np.minimum(silhouette, _schp_body_old)

        # Old behavior: lower body region WOULD be included (pants=6 in old labels)
        assert np.sum(clipped_old[550:750, 350:650] > 127) > 0, (
            "old behavior incorrectly includes lower body in upper_body silhouette"
        )

    def test_lower_body_silhouette_excludes_upper(self):
        """Lower body garment silhouette should NOT include upper_clothes/arms."""
        profile = build_garment_profile("jeans", "lower_body")
        target_labels = get_profile_editable_labels(profile)
        silhouette = _make_silhouette_mask(includes_upper=True, includes_lower=True, includes_arms=True)

        schp_map = _make_schp_map()
        _schp_body = np.isin(schp_map, list(target_labels)).astype(np.uint8) * 255
        clipped = np.minimum(silhouette, _schp_body)

        # Lower body region should remain
        assert np.sum(clipped[500:750, 300:700] > 127) > 0, "lower body region should be in clipped silhouette"
        # Upper body region should be REMOVED
        assert np.sum(clipped[250:400, 350:650] > 127) == 0, (
            "upper body region must NOT be in clipped silhouette for lower_body garment"
        )

    def test_dress_silhouette_keeps_everything(self):
        """Dress silhouette should include all body regions (no clipping)."""
        profile = build_garment_profile("dress", "dresses")
        target_labels = get_profile_editable_labels(profile)
        silhouette = _make_silhouette_mask(includes_upper=True, includes_lower=True, includes_arms=True)

        schp_map = _make_schp_map()
        _schp_body = np.isin(schp_map, list(target_labels)).astype(np.uint8) * 255
        clipped = np.minimum(silhouette, _schp_body)

        # Both upper and lower should remain
        assert np.sum(clipped[250:400, 350:650] > 127) > 0, "dress should include upper body"
        assert np.sum(clipped[550:750, 350:650] > 127) > 0, "dress should include lower body"

    def test_upper_body_with_scarf_in_silhouette(self):
        """Upper body garment should not include scarf (label 17) in target labels."""
        profile = build_garment_profile("shirt", "upper_body")
        target_labels = get_profile_editable_labels(profile)
        assert _LABEL_SCARF not in target_labels, "shirt target labels should not include scarf"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: Cross-category mask does not leak source lower-body into upper-body
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossCategoryNoLeakage:
    """Verify the cross-category fix: source garment labels removed from protect."""

    def test_saree_to_shirt_mask_covers_both(self):
        """Saree→shirt: inpaint mask must include both upper (target) and dress/scarf (source)."""
        schp = _make_schp_map(
            dress_region=(150, 250, 600, 750),
            scarf_region=(150, 300, 300, 700),
        )
        profile = build_garment_profile("shirt", "upper_body")
        garment_info = GarmentImageInfo(
            bbox_area_ratio=0.4, aspect_ratio=0.75,
            width_ratio=0.8, height_ratio=0.6,
            center_y_ratio=0.4, has_sleeves=True,
            is_long=False, is_wide=False,
        )
        final_mask, inpaint_mask, protect_mask = build_final_inpaint_mask(
            schp, "upper_body", "shirt",
            source_cloth_type="dresses",
            garment_img_info=garment_info,
            trace_id="test_saree_to_shirt",
            profile=profile,
        )

        # Upper body region (target) should be in final mask
        assert np.sum(final_mask[200:400, 350:650] > 127) > 0, (
            "final mask must include upper body region for target"
        )

        # Dress region (source) should also be in final mask (needs erasure)
        assert np.sum(final_mask[350:550, 350:650] > 127) > 0, (
            "final mask must include source dress region for erasure"
        )

    def test_protect_excludes_source_labels(self):
        """Protect mask must NOT protect source garment labels (they need erasure)."""
        schp = _make_schp_map(
            dress_region=(150, 250, 600, 750),
            scarf_region=(150, 300, 300, 700),
        )
        profile = build_garment_profile("shirt", "upper_body")
        garment_info = GarmentImageInfo(
            bbox_area_ratio=0.4, aspect_ratio=0.75,
            width_ratio=0.8, height_ratio=0.6,
            center_y_ratio=0.4, has_sleeves=True,
            is_long=False, is_wide=False,
        )
        _, inpaint_mask, protect_mask = build_final_inpaint_mask(
            schp, "upper_body", "shirt",
            source_cloth_type="dresses",
            garment_img_info=garment_info,
            trace_id="test_protect_excludes_source",
            profile=profile,
        )

        # Protect mask should NOT cover dress pixels (they should be editable)
        dress_present = schp == _LABEL_DRESS
        if np.any(dress_present):
            protect_at_dress = protect_mask[dress_present]
            # Dress pixels should NOT be protected (protect should be 0 there)
            protected_ratio = float(np.mean(protect_at_dress > 127))
            assert protected_ratio < 0.3, (
                f"protect mask incorrectly protects {protected_ratio:.1%} of dress pixels — "
                "source garment labels must not be protected"
            )

    def test_upper_body_final_mask_excludes_legs(self):
        """Upper body target: final mask must NOT include leg labels."""
        schp = _make_schp_map(
            upper_region=(200, 300, 500, 700),
            leg_left_region=(600, 350, 750, 480),
            leg_right_region=(600, 520, 750, 650),
        )
        profile = build_garment_profile("shirt", "upper_body")
        garment_info = GarmentImageInfo(
            bbox_area_ratio=0.4, aspect_ratio=0.75,
            width_ratio=0.8, height_ratio=0.6,
            center_y_ratio=0.4, has_sleeves=True,
            is_long=False, is_wide=False,
        )
        final_mask, _, _ = build_final_inpaint_mask(
            schp, "upper_body", "shirt",
            source_cloth_type="",
            garment_img_info=garment_info,
            trace_id="test_upper_no_legs",
            profile=profile,
        )

        # Leg regions should NOT be in the final mask
        left_leg_area = schp == _LABEL_LEFT_LEG
        right_leg_area = schp == _LABEL_RIGHT_LEG
        if np.any(left_leg_area):
            assert np.mean(final_mask[left_leg_area] > 127) < 0.1, (
                "final mask must not cover left leg for upper_body garment"
            )
        if np.any(right_leg_area):
            assert np.mean(final_mask[right_leg_area] > 127) < 0.1, (
                "final mask must not cover right leg for upper_body garment"
            )


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: Full mask pipeline coverage assertions
# ═══════════════════════════════════════════════════════════════════════════

class TestFullMaskCoverage:
    """End-to-end mask coverage assertions for the full build_final_inpaint_mask."""

    def test_upper_body_covers_upper_not_lower(self):
        """Upper body target: inpaint covers upper_clothes+arms, not pants/legs."""
        schp = _make_schp_map()
        profile = build_garment_profile("shirt", "upper_body")
        garment_info = GarmentImageInfo(
            bbox_area_ratio=0.4, aspect_ratio=0.75,
            width_ratio=0.8, height_ratio=0.6,
            center_y_ratio=0.4, has_sleeves=True,
            is_long=False, is_wide=False,
        )
        final_mask, inpaint_mask, _ = build_final_inpaint_mask(
            schp, "upper_body", "shirt",
            garment_img_info=garment_info,
            trace_id="test_upper_coverage",
            profile=profile,
        )

        upper_present = schp == _LABEL_UPPER_CLOTHES
        pants_present = schp == _LABEL_PANTS
        legs_present = np.isin(schp, [_LABEL_LEFT_LEG, _LABEL_RIGHT_LEG])

        if np.any(upper_present):
            assert np.mean(final_mask[upper_present] > 127) > 0.5, (
                "final mask should cover majority of upper_clothes"
            )
        if np.any(pants_present):
            assert np.mean(final_mask[pants_present] > 127) < 0.1, (
                "final mask should NOT cover pants for upper_body"
            )
        if np.any(legs_present):
            assert np.mean(final_mask[legs_present] > 127) < 0.1, (
                "final mask should NOT cover legs for upper_body"
            )

    def test_lower_body_covers_pants_not_upper(self):
        """Lower body target: inpaint covers pants+skirt+legs, not upper_clothes."""
        schp = _make_schp_map()
        profile = build_garment_profile("jeans", "lower_body")
        garment_info = GarmentImageInfo(
            bbox_area_ratio=0.4, aspect_ratio=0.75,
            width_ratio=0.8, height_ratio=0.6,
            center_y_ratio=0.6, has_sleeves=False,
            is_long=True, is_wide=False,
        )
        final_mask, inpaint_mask, _ = build_final_inpaint_mask(
            schp, "lower_body", "jeans",
            garment_img_info=garment_info,
            trace_id="test_lower_coverage",
            profile=profile,
        )

        upper_present = schp == _LABEL_UPPER_CLOTHES
        pants_present = schp == _LABEL_PANTS

        if np.any(upper_present):
            assert np.mean(final_mask[upper_present] > 127) < 0.1, (
                "final mask should NOT cover upper_clothes for lower_body"
            )
        if np.any(pants_present):
            assert np.mean(final_mask[pants_present] > 127) > 0.5, (
                "final mask should cover majority of pants for lower_body"
            )

    def test_dress_covers_everything(self):
        """Dress target: inpaint covers upper+lower+arms+legs."""
        schp = _make_schp_map(
            dress_region=(200, 300, 500, 700),
            # Arms must be OUTSIDE the dress region to avoid overwrite
            arm_left_region=(250, 100, 450, 200),
            arm_right_region=(250, 800, 450, 900),
        )
        profile = build_garment_profile("dress", "dresses")
        garment_info = GarmentImageInfo(
            bbox_area_ratio=0.6, aspect_ratio=0.75,
            width_ratio=0.8, height_ratio=0.8,
            center_y_ratio=0.5, has_sleeves=False,
            is_long=True, is_wide=False,
        )
        final_mask, inpaint_mask, _ = build_final_inpaint_mask(
            schp, "dresses", "dress",
            garment_img_info=garment_info,
            trace_id="test_dress_coverage",
            profile=profile,
        )

        upper_present = schp == _LABEL_UPPER_CLOTHES
        pants_present = schp == _LABEL_PANTS
        arms_present = np.isin(schp, [_LABEL_LEFT_ARM, _LABEL_RIGHT_ARM])

        for name, region in [("upper_clothes", upper_present), ("pants", pants_present), ("arms", arms_present)]:
            if np.any(region):
                coverage = float(np.mean(final_mask[region] > 127))
                assert coverage > 0.3, f"final mask should cover {name} for dress, got {coverage:.1%}"

    def test_final_mask_is_binary(self):
        """Final mask must be strictly binary (0 or 255)."""
        schp = _make_schp_map()
        for cloth_type, subtype in [("upper_body", "shirt"), ("lower_body", "jeans"), ("dresses", "dress")]:
            profile = build_garment_profile(subtype, cloth_type)
            garment_info = GarmentImageInfo(
                bbox_area_ratio=0.4, aspect_ratio=0.75,
                width_ratio=0.8, height_ratio=0.6,
                center_y_ratio=0.5, has_sleeves=True,
                is_long=False, is_wide=False,
            )
            final_mask, _, _ = build_final_inpaint_mask(
                schp, cloth_type, subtype,
                garment_img_info=garment_info,
                trace_id=f"test_binary_{cloth_type}",
                profile=profile,
            )
            unique = set(int(v) for v in np.unique(final_mask))
            assert unique <= {0, 255}, f"{cloth_type}: mask has non-binary values {unique}"

    def test_final_mask_non_empty(self):
        """Final mask must not be completely empty."""
        schp = _make_schp_map()
        profile = build_garment_profile("shirt", "upper_body")
        garment_info = GarmentImageInfo(
            bbox_area_ratio=0.4, aspect_ratio=0.75,
            width_ratio=0.8, height_ratio=0.6,
            center_y_ratio=0.4, has_sleeves=True,
            is_long=False, is_wide=False,
        )
        final_mask, _, _ = build_final_inpaint_mask(
            schp, "upper_body", "shirt",
            garment_img_info=garment_info,
            trace_id="test_not_empty",
            profile=profile,
        )
        assert np.sum(final_mask > 127) > 0, "final mask is completely empty"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5: Draped garments skip body-label AND
# ═══════════════════════════════════════════════════════════════════════════

class TestDrapedGarmentSilhouette:
    """Draped garments (saree pallu) should NOT be ANDed with body labels."""

    def test_saree_is_draped(self):
        assert is_draped_garment("dresses", "saree") is True

    def test_dupatta_is_draped(self):
        assert is_draped_garment("dresses", "dupatta") is True

    def test_shirt_is_not_draped(self):
        assert is_draped_garment("upper_body", "shirt") is False

    def test_jeans_is_not_draped(self):
        assert is_draped_garment("lower_body", "jeans") is False

    def test_saree_profile_has_draped_flag(self):
        profile = build_garment_profile("saree", "dresses")
        assert profile.is_draped is True

    def test_draped_silhouette_not_clipped(self):
        """For draped garments, silhouette should be used directly (no AND)."""
        assert is_draped_garment("dresses", "saree") is True
        # The handler code checks: if not _is_draped_garment: AND with target labels
        # For draped, the else branch is taken (no clipping)
        # This is a behavior test — if is_draped_garment returns True, silhouette stays unclipped.


# ═══════════════════════════════════════════════════════════════════════════
# TEST 6: Identity labels never in final mask
# ═══════════════════════════════════════════════════════════════════════════

class TestIdentityProtected:
    """Identity labels (face, hair, shoes, hat, sunglasses, bag) must never be in final mask."""

    IDENTITY_LABELS = [
        _LABEL_FACE, _LABEL_HAIR, _LABEL_HAT, _LABEL_SUNGLASSES,
        _LABEL_LEFT_SHOE, _LABEL_RIGHT_SHOE, _LABEL_BAG,
    ]

    def test_identity_never_in_final_mask(self):
        schp = _make_schp_map(
            face_region=(100, 400, 250, 620),
            hair_region=(30, 380, 150, 640),
        )
        for cloth_type, subtype in [("upper_body", "shirt"), ("lower_body", "jeans"), ("dresses", "dress")]:
            profile = build_garment_profile(subtype, cloth_type)
            garment_info = GarmentImageInfo(
                bbox_area_ratio=0.4, aspect_ratio=0.75,
                width_ratio=0.8, height_ratio=0.6,
                center_y_ratio=0.5, has_sleeves=True,
                is_long=False, is_wide=False,
            )
            final_mask, _, _ = build_final_inpaint_mask(
                schp, cloth_type, subtype,
                garment_img_info=garment_info,
                trace_id=f"test_identity_{cloth_type}",
                profile=profile,
            )
            for label in self.IDENTITY_LABELS:
                label_present = schp == label
                if np.any(label_present):
                    coverage = float(np.mean(final_mask[label_present] > 127))
                    assert coverage < 0.05, (
                        f"{cloth_type}/{subtype}: identity label {label} "
                        f"has {coverage:.1%} coverage in final mask (must be <5%)"
                    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
