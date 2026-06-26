"""
P0 Diagnostic Instrumentation — Prove which conditioning path is failing.

This module provides probes for the IDM-VTON pipeline to measure:
  1. GarmentNet (fine path) activation status
  2. GarmentNet input resolution vs person latent resolution
  3. Attention-contribution magnitude (relative KV norm)
  4. Garment bounding-box-to-canvas ratio and centering offset
  5. Mask IoU between inpaint mask and garment silhouette

Usage:
    from p0_diagnostics import P0Probe
    probe = P0Probe(trace_id="test-001")
    # ... at various pipeline stages, call probe.record_*(...)
    probe.dump()  # returns dict of all findings
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("idm-vton.worker.p0")


@dataclass
class P0Probe:
    """Accumulates P0 diagnostic measurements for a single inference call."""

    trace_id: str = ""

    # P0-1: GarmentNet active
    garmentnet_active: bool = False
    garmentnet_input_shape: list[int] = field(default_factory=list)
    garmentnet_output_feature_count: int = 0

    # P0-2: Resolution comparison
    person_latent_shape: list[int] = field(default_factory=list)
    garment_input_resolution: list[int] = field(default_factory=list)
    resolution_match: bool = False

    # P0-3: Attention contribution
    attention_contribution_ratios: list[float] = field(default_factory=list)
    attention_contribution_mean: float = 0.0
    attention_contribution_min: float = 0.0
    attention_contribution_max: float = 0.0

    # P0-4: Garment bbox-to-canvas
    garment_bbox_area_ratio: float = 0.0
    garment_center_x_offset: float = 0.0
    garment_center_y_offset: float = 0.0
    garment_width_ratio: float = 0.0
    garment_height_ratio: float = 0.0

    # P0-5: Mask IoU
    mask_iou_with_silhouette: float = 0.0
    mask_coverage: float = 0.0
    silhouette_coverage: float = 0.0
    mask_is_too_loose: bool = False
    mask_is_too_tight: bool = False

    # Timing
    garmentnet_time_ms: float = 0.0
    attention_probe_time_ms: float = 0.0

    def record_garmentnet_call(
        self,
        cloth_tensor: torch.Tensor,
        output_features: list[torch.Tensor],
        elapsed_ms: float,
    ) -> None:
        """Record GarmentNet invocation details (P0-1, P0-2)."""
        self.garmentnet_active = True
        self.garmentnet_input_shape = list(cloth_tensor.shape)
        self.garmentnet_output_feature_count = len(output_features)
        self.garmentnet_time_ms = elapsed_ms

        # Log feature shapes for resolution comparison
        for i, feat in enumerate(output_features):
            logger.info(
                "P0_GARMENTNET_FEATURE idx=%d shape=%s norm=%.4f",
                i, list(feat.shape), float(feat.norm().item()),
            )

    def record_person_latent(self, latent_tensor: torch.Tensor) -> None:
        """Record person latent shape for resolution comparison (P0-2)."""
        self.person_latent_shape = list(latent_tensor.shape)

    def record_attention_contribution(
        self,
        person_hidden_norm: float,
        garment_hidden_norm: float,
        fused_output_norm: float,
        block_idx: int = 0,
        timestep: float = 0.0,
    ) -> None:
        """Record attention contribution ratio at a fusion block (P0-3).

        The ratio = garment_hidden_norm / (person_hidden_norm + garment_hidden_norm).
        Values near 0.0 = garment features have no influence.
        Values near 0.5 = equal contribution.
        Values > 0.5 = garment dominates.
        """
        total = person_hidden_norm + garment_hidden_norm
        ratio = garment_hidden_norm / max(total, 1e-8)
        self.attention_contribution_ratios.append(ratio)

        logger.info(
            "P0_ATTENTION_CONTRIB block=%d t=%.1f person_norm=%.4f "
            "garment_norm=%.4f ratio=%.4f fused_norm=%.4f",
            block_idx, timestep,
            person_hidden_norm, garment_hidden_norm,
            ratio, fused_output_norm,
        )

    def record_garment_canvas(
        self,
        garment_img: Image.Image,
        target_size: tuple[int, int] = (768, 1024),
    ) -> None:
        """Record garment bbox-to-canvas ratio and centering (P0-4)."""
        arr = np.array(garment_img.convert("RGB"), dtype=np.uint8)
        h, w = arr.shape[:2]

        # Detect foreground (non-mid-gray)
        is_bg = np.all(np.abs(arr.astype(np.int16) - 128) < 40, axis=2)
        fg = ~is_bg

        if not np.any(fg):
            return

        ys, xs = np.where(fg)
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1

        bbox_area = (x2 - x1) * (y2 - y1)
        total_area = h * w
        self.garment_bbox_area_ratio = bbox_area / max(total_area, 1)
        self.garment_center_x_offset = ((x1 + x2) / 2.0) / max(w, 1) - 0.5
        self.garment_center_y_offset = ((y1 + y2) / 2.0) / max(h, 1) - 0.5
        self.garment_width_ratio = (x2 - x1) / max(w, 1)
        self.garment_height_ratio = (y2 - y1) / max(h, 1)

        logger.info(
            "P0_GARMENT_CANVAS bbox_ratio=%.4f center_x_off=%.4f "
            "center_y_off=%.4f width_ratio=%.4f height_ratio=%.4f "
            "canvas=%dx%d",
            self.garment_bbox_area_ratio,
            self.garment_center_x_offset,
            self.garment_center_y_offset,
            self.garment_width_ratio,
            self.garment_height_ratio,
            w, h,
        )

    def record_mask_silhouette_iou(
        self,
        inpaint_mask: np.ndarray,
        garment_silhouette: np.ndarray,
    ) -> None:
        """Record mask IoU between inpaint mask and garment silhouette (P0-5)."""
        # Ensure same shape
        if inpaint_mask.shape != garment_silhouette.shape:
            from PIL import Image as PILImage
            sil_pil = PILImage.fromarray((garment_silhouette > 127).astype(np.uint8) * 255)
            sil_resized = np.array(
                sil_pil.resize(
                    (inpaint_mask.shape[1], inpaint_mask.shape[0]),
                    PILImage.NEAREST,
                )
            )
            garment_silhouette = sil_resized

        mask_bin = (inpaint_mask > 127).astype(np.bool_)
        sil_bin = (garment_silhouette > 127).astype(np.bool_)

        intersection = np.logical_and(mask_bin, sil_bin).sum()
        union = np.logical_or(mask_bin, sil_bin).sum()

        self.mask_iou_with_silhouette = float(intersection) / max(float(union), 1)
        self.mask_coverage = float(mask_bin.sum()) / mask_bin.size
        self.silhouette_coverage = float(sil_bin.sum()) / sil_bin.size

        # Loose = mask much larger than silhouette; tight = mask much smaller
        if self.mask_coverage > 0 and self.silhouette_coverage > 0:
            ratio = self.mask_coverage / self.silhouette_coverage
            self.mask_is_too_loose = ratio > 2.0
            self.mask_is_too_tight = ratio < 0.5

        logger.info(
            "P0_MASK_IOU iou=%.4f mask_cov=%.4f sil_cov=%.4f "
            "too_loose=%s too_tight=%s",
            self.mask_iou_with_silhouette,
            self.mask_coverage,
            self.silhouette_coverage,
            self.mask_is_too_loose,
            self.mask_is_too_tight,
        )

    def finalize(self) -> dict[str, Any]:
        """Compute summary statistics and return all findings."""
        if self.attention_contribution_ratios:
            self.attention_contribution_mean = float(np.mean(self.attention_contribution_ratios))
            self.attention_contribution_min = float(np.min(self.attention_contribution_ratios))
            self.attention_contribution_max = float(np.max(self.attention_contribution_ratios))

        self.resolution_match = (
            len(self.garmentnet_input_shape) >= 4
            and len(self.person_latent_shape) >= 4
            and self.garmentnet_input_shape[2] == self.person_latent_shape[2]
            and self.garmentnet_input_shape[3] == self.person_latent_shape[3]
        )

        return {
            "trace_id": self.trace_id,
            "p0_1_garmentnet_active": self.garmentnet_active,
            "p0_1_garmentnet_input_shape": self.garmentnet_input_shape,
            "p0_1_garmentnet_feature_count": self.garmentnet_output_feature_count,
            "p0_1_garmentnet_time_ms": round(self.garmentnet_time_ms, 1),
            "p0_2_person_latent_shape": self.person_latent_shape,
            "p0_2_garment_input_resolution": self.garmentnet_input_shape,
            "p0_2_resolution_match": self.resolution_match,
            "p0_3_attention_contribution_mean": round(self.attention_contribution_mean, 6),
            "p0_3_attention_contribution_min": round(self.attention_contribution_min, 6),
            "p0_3_attention_contribution_max": round(self.attention_contribution_max, 6),
            "p0_3_attention_contribution_ratios": [round(r, 6) for r in self.attention_contribution_ratios],
            "p0_3_fine_path_active": self.attention_contribution_mean > 0.01,
            "p0_3_fine_path_contribution": (
                "near_zero" if self.attention_contribution_mean < 0.01
                else "low" if self.attention_contribution_mean < 0.1
                else "moderate" if self.attention_contribution_mean < 0.3
                else "high"
            ),
            "p0_4_garment_bbox_area_ratio": round(self.garment_bbox_area_ratio, 4),
            "p0_4_garment_center_x_offset": round(self.garment_center_x_offset, 4),
            "p0_4_garment_center_y_offset": round(self.garment_center_y_offset, 4),
            "p0_4_garment_width_ratio": round(self.garment_width_ratio, 4),
            "p0_4_garment_height_ratio": round(self.garment_height_ratio, 4),
            "p0_5_mask_silhouette_iou": round(self.mask_iou_with_silhouette, 4),
            "p0_5_mask_coverage": round(self.mask_coverage, 4),
            "p0_5_silhouette_coverage": round(self.silhouette_coverage, 4),
            "p0_5_mask_too_loose": self.mask_is_too_loose,
            "p0_5_mask_too_tight": self.mask_is_too_tight,
        }

    def dump(self, output_dir: str = "/tmp/idm-vton-debug") -> str:
        """Save findings to JSON and return the path."""
        findings = self.finalize()
        out_path = Path(output_dir) / f"p0_findings_{self.trace_id}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(findings, indent=2, default=str))

        # Also log summary
        logger.info("P0_FINDINGS %s", json.dumps(findings, default=str))

        return str(out_path)
