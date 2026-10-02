"""
Automated Closed-Loop Pipeline Evaluation & Parameter Tuning Harness.

Evaluates virtual try-on pipelines against commercial fidelity metrics:
  1. S_boundary: Boundary Gradient Discontinuity Ratio (Sobel edge spike detection).
  2. S_bg / S_face: PSNR and SSIM within protected regions.
  3. S_pose: PCKh@0.5 keypoint structural retention.
  4. Latency breakdown per stage (Preprocess, Latent Prep, UNet Denoise, Decode, Postprocess).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tryon_evaluator")


@dataclass
class QualityMetrics:
    boundary_discontinuity_ratio: float
    boundary_gradient_variance: float
    face_psnr_db: float
    face_ssim: float
    background_psnr_db: float
    background_ssim: float
    pose_retention_pckh: float
    passed: bool
    failure_reasons: List[str]


@dataclass
class LatencyProfile:
    preprocessing_ms: float
    latent_prep_ms: float
    unet_denoise_ms: float
    vae_decode_ms: float
    postprocess_ms: float
    total_warm_ms: float


@dataclass
class TestCaseResult:
    case_id: str
    category: str
    metrics: QualityMetrics
    latency: LatencyProfile


# =============================================================================
# Automated Metric Computations
# =============================================================================

def compute_psnr(orig: np.ndarray, generated: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Compute PSNR strictly inside the mask region (255=evaluate, 0=ignore)."""
    orig_f = orig.astype(np.float64)
    gen_f = generated.astype(np.float64)
    
    if mask is not None:
        idx = mask > 127
        if not np.any(idx):
            return 100.0
        diff = orig_f[idx] - gen_f[idx]
    else:
        diff = orig_f - gen_f
        
    mse = np.mean(diff ** 2)
    if mse < 1e-10:
        return 100.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def compute_ssim_region(orig: np.ndarray, generated: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Compute structural similarity (SSIM) within region."""
    orig_g = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY) if orig.ndim == 3 else orig
    gen_g = cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY) if generated.ndim == 3 else generated
    
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    orig_g = orig_g.astype(np.float64)
    gen_g = gen_g.astype(np.float64)
    
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    mu1 = cv2.filter2D(orig_g, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(gen_g, -1, window)[5:-5, 5:-5]
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(orig_g ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(gen_g ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(orig_g * gen_g, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    
    if mask is not None:
        mask_cropped = mask[5:-5, 5:-5] > 127
        if not np.any(mask_cropped):
            return 1.0
        return float(np.mean(ssim_map[mask_cropped]))
    return float(np.mean(ssim_map))


def compute_boundary_discontinuity(generated: np.ndarray, inpaint_mask: np.ndarray) -> Tuple[float, float]:
    """
    Computes Boundary Gradient Discontinuity Ratio (S_boundary).
    
    Measures high-frequency Sobel gradient magnitude variance across the mask perimeter.
    A high ratio indicates unnatural step-edges, halos, or stitched seams.
    """
    gen_g = cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY) if generated.ndim == 3 else generated
    
    # Compute Sobel gradients
    sobel_x = cv2.Sobel(gen_g, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gen_g, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    
    # Extract mask perimeter ribbon (band of width 6px across the boundary)
    mask_u8 = (inpaint_mask > 127).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(mask_u8, kernel, iterations=1)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    perimeter_ribbon = (dilated - eroded) > 0
    
    # Exterior local reference region (6px outside the perimeter)
    exterior_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    exterior_region = (cv2.dilate(mask_u8, exterior_kernel, iterations=1) - dilated) > 0
    
    if not np.any(perimeter_ribbon) or not np.any(exterior_region):
        return 1.0, 0.0
        
    ribbon_grads = grad_mag[perimeter_ribbon]
    exterior_grads = grad_mag[exterior_region]
    
    mean_ribbon = float(np.mean(ribbon_grads))
    mean_exterior = float(np.mean(exterior_grads))
    variance_ribbon = float(np.var(ribbon_grads))
    
    ratio = mean_ribbon / (mean_exterior + 1e-5)
    return ratio, variance_ribbon


def compute_pckh(keypoints_orig: Dict[str, Tuple[float, float]], keypoints_gen: Dict[str, Tuple[float, float]], threshold_fraction: float = 0.5) -> float:
    """Compute PCKh score of keypoint structural retention."""
    if not keypoints_orig or not keypoints_gen:
        return 1.0
        
    # Head size calculation (distance between nose/neck or top of head)
    head_size = 50.0
    if "nose" in keypoints_orig and "neck" in keypoints_orig:
        p1 = keypoints_orig["nose"]
        p2 = keypoints_orig["neck"]
        head_size = max(20.0, math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))
        
    threshold = threshold_fraction * head_size
    matched = 0
    total = 0
    
    for k, pt_orig in keypoints_orig.items():
        if k in keypoints_gen:
            pt_gen = keypoints_gen[k]
            dist = math.sqrt((pt_orig[0] - pt_gen[0])**2 + (pt_orig[1] - pt_gen[1])**2)
            if dist <= threshold:
                matched += 1
            total += 1
            
    return (matched / total) if total > 0 else 1.0


# =============================================================================
# Benchmark Suite Runner
# =============================================================================

class PipelineEvaluator:
    """Evaluates test directory against target production thresholds."""
    
    TARGET_THRESHOLDS = {
        "face_psnr_min": 38.0,
        "face_ssim_min": 0.92,
        "bg_psnr_min": 40.0,
        "boundary_ratio_max": 2.2,
        "pose_pckh_min": 0.95,
        "total_latency_max_ms": 8000.0,
    }
    
    def evaluate_pair(
        self,
        case_id: str,
        category: str,
        original: np.ndarray,
        result: np.ndarray,
        inpaint_mask: np.ndarray,
        protected_mask: np.ndarray,
        face_mask: np.ndarray,
        latency: LatencyProfile,
        keypoints_orig: Optional[Dict[str, Tuple[float, float]]] = None,
        keypoints_gen: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> TestCaseResult:
        """Run full metric evaluation on a single try-on output."""
        bg_mask = (inpaint_mask < 128).astype(np.uint8) * 255
        
        face_psnr = compute_psnr(original, result, face_mask)
        face_ssim = compute_ssim_region(original, result, face_mask)
        
        bg_psnr = compute_psnr(original, result, bg_mask)
        bg_ssim = compute_ssim_region(original, result, bg_mask)
        
        b_ratio, b_var = compute_boundary_discontinuity(result, inpaint_mask)
        pose_pckh = compute_pckh(keypoints_orig or {}, keypoints_gen or {})
        
        failure_reasons = []
        if face_psnr < self.TARGET_THRESHOLDS["face_psnr_min"]:
            failure_reasons.append(f"Face PSNR {face_psnr:.1f} dB < {self.TARGET_THRESHOLDS['face_psnr_min']} dB")
        if face_ssim < self.TARGET_THRESHOLDS["face_ssim_min"]:
            failure_reasons.append(f"Face SSIM {face_ssim:.3f} < {self.TARGET_THRESHOLDS['face_ssim_min']}")
        if bg_psnr < self.TARGET_THRESHOLDS["bg_psnr_min"]:
            failure_reasons.append(f"Background PSNR {bg_psnr:.1f} dB < {self.TARGET_THRESHOLDS['bg_psnr_min']} dB")
        if b_ratio > self.TARGET_THRESHOLDS["boundary_ratio_max"]:
            failure_reasons.append(f"Boundary Discontinuity Ratio {b_ratio:.2f} > {self.TARGET_THRESHOLDS['boundary_ratio_max']}")
        if pose_pckh < self.TARGET_THRESHOLDS["pose_pckh_min"]:
            failure_reasons.append(f"Pose Retention {pose_pckh:.3f} < {self.TARGET_THRESHOLDS['pose_pckh_min']}")
        if latency.total_warm_ms > self.TARGET_THRESHOLDS["total_latency_max_ms"]:
            failure_reasons.append(f"Total Latency {latency.total_warm_ms:.0f} ms > {self.TARGET_THRESHOLDS['total_latency_max_ms']:.0f} ms")
            
        passed = len(failure_reasons) == 0
        
        metrics = QualityMetrics(
            boundary_discontinuity_ratio=round(b_ratio, 3),
            boundary_gradient_variance=round(b_var, 3),
            face_psnr_db=round(face_psnr, 2),
            face_ssim=round(face_ssim, 4),
            background_psnr_db=round(bg_psnr, 2),
            background_ssim=round(bg_ssim, 4),
            pose_retention_pckh=round(pose_pckh, 4),
            passed=passed,
            failure_reasons=failure_reasons,
        )
        
        return TestCaseResult(case_id=case_id, category=category, metrics=metrics, latency=latency)


def generate_benchmark_summary(results: List[TestCaseResult]) -> str:
    """Formats results into a Markdown report."""
    lines = [
        "# TryLix VTON Pipeline Verification Report",
        "",
        f"**Total Cases Evaluated:** {len(results)}",
        f"**Passed:** {sum(1 for r in results if r.metrics.passed)} / {len(results)}",
        "",
        "| Case ID | Category | Face PSNR | Bg PSNR | Boundary Ratio | Pose PCKh | Warm Latency | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in results:
        status = "[PASS]" if r.metrics.passed else f"[FAIL] ({len(r.metrics.failure_reasons)} issues)"
        lines.append(
            f"| {r.case_id} | {r.category} | {r.metrics.face_psnr_db:.1f} dB | "
            f"{r.metrics.background_psnr_db:.1f} dB | {r.metrics.boundary_discontinuity_ratio:.2f} | "
            f"{r.metrics.pose_retention_pckh:.3f} | {r.latency.total_warm_ms:.0f} ms | {status} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline benchmark.")
    parser.add_argument("--test-dir", type=str, default="", help="Directory with test cases")
    args = parser.parse_args()
    
    # Synthetic smoke test if no directory given
    evaluator = PipelineEvaluator()
    dummy_orig = np.random.randint(100, 200, (1024, 768, 3), dtype=np.uint8)
    dummy_res = dummy_orig.copy()
    dummy_inpaint = np.zeros((1024, 768), dtype=np.uint8)
    dummy_inpaint[300:700, 200:568] = 255
    dummy_face = np.zeros((1024, 768), dtype=np.uint8)
    dummy_face[50:250, 250:500] = 255
    dummy_prot = np.zeros((1024, 768), dtype=np.uint8)
    
    latency = LatencyProfile(
        preprocessing_ms=450.0,
        latent_prep_ms=120.0,
        unet_denoise_ms=4800.0,
        vae_decode_ms=350.0,
        postprocess_ms=45.0,
        total_warm_ms=5765.0,
    )
    
    res = evaluator.evaluate_pair(
        "smoke_test_01",
        "upper_body",
        dummy_orig,
        dummy_res,
        dummy_inpaint,
        dummy_prot,
        dummy_face,
        latency,
    )
    report = generate_benchmark_summary([res])
    print("\n" + report + "\n")
