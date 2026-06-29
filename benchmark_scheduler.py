"""
Benchmark DDPM@50steps vs DPM++ Karras@25steps.

This script compares the current DDPM scheduler at the default step count
against the proposed DPM++ 2M SDE Karras scheduler at a reduced step count.

Usage:
    # Run on a directory of images: expects person_*.png + garment_*.png pairs
    python benchmark_scheduler.py --image_dir /tmp/benchmark_images

    # Run on a single pair
    python benchmark_scheduler.py --person /tmp/person.png --garment /tmp/garment.png

    # Custom step counts
    python benchmark_scheduler.py --baseline_steps 50 --candidate_steps 25

Output:
    results/benchmark_{timestamp}.json   — per-image metrics
    results/benchmark_{timestamp}.csv    — aggregate table
    results/benchmark_{timestamp}.txt    — summary report
"""

import os
import sys
import json
import csv
import time
import argparse
import datetime
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("benchmark")


def compute_metrics(ref: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Compute PSNR and SSIM between two uint8 RGB images."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    metrics: dict[str, float] = {}
    metrics["psnr"] = float(
        peak_signal_noise_ratio(ref, pred, data_range=255)
    )
    metrics["ssim"] = float(
        structural_similarity(ref, pred, channel_axis=2, data_range=255)
    )
    return metrics


def load_benchmark_images(
    image_dir: str,
    person_prefix: str = "person",
    garment_prefix: str = "garment",
) -> list[tuple[str, str, str]]:
    """Load paired (person, garment, label) from a directory.

    Expects files named like:
        person_001.png, garment_001.png, ...
        person_002.jpg, garment_002.jpg, ...
    """
    p_dir = Path(image_dir)
    persons = sorted(p_dir.glob(f"{person_prefix}_*"))
    garments = sorted(p_dir.glob(f"{garment_prefix}_*"))

    pairs: list[tuple[str, str, str]] = []
    for p in persons:
        stem = p.stem.replace(f"{person_prefix}_", "")
        g = p_dir / f"{garment_prefix}_{stem}{p.suffix}"
        if not g.exists():
            g2 = p_dir / f"{garment_prefix}_{stem}.png"
            if not g2.exists():
                g2 = p_dir / f"{garment_prefix}_{stem}.jpg"
            g = g2
        if g.exists():
            pairs.append((str(p), str(g), stem))
        else:
            logger.warning("No matching garment for person=%s", p.name)
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DDPM vs DPM++ scheduler"
    )
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Directory with person_* and garment_* image pairs")
    parser.add_argument("--person", type=str, default=None,
                        help="Single person image path")
    parser.add_argument("--garment", type=str, default=None,
                        help="Single garment image path")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for results (default: results/)")
    parser.add_argument("--baseline_steps", type=int, default=50,
                        help="DDPM step count (default: 50)")
    parser.add_argument("--candidate_steps", type=int, default=25,
                        help="DPM++ step count (default: 25)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--cloth_type", type=str, default="upper_body",
                        help="Cloth type for inference (default: upper_body)")
    parser.add_argument("--garment_desc", type=str, default="garment",
                        help="Garment description prompt (default: garment)")
    parser.add_argument("--guidance", type=float, default=3.5,
                        help="Guidance scale (default: 3.5)")
    args = parser.parse_args()

    # ── Validate inputs ────────────────────────────────────────────────
    if args.image_dir:
        pairs = load_benchmark_images(args.image_dir)
        if not pairs:
            logger.error("No image pairs found in %s", args.image_dir)
            sys.exit(1)
    elif args.person and args.garment:
        pairs = [(args.person, args.garment, "single")]
    else:
        logger.error(
            "Provide either --image_dir or both --person and --garment"
        )
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load models once ───────────────────────────────────────────────
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "IDM-VTON",
        ),
    )
    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "IDM-VTON/gradio_demo",
        ),
    )

    import handler as h

    # Force DDPM for baseline
    os.environ["IDM_VTON_SCHEDULER"] = "ddpm"
    os.environ["IDM_VTON_STEPS"] = str(args.baseline_steps)

    # Reset module-level constants
    h.IDM_VTON_SCHEDULER = "ddpm"
    h.DENOISE_STEPS = args.baseline_steps

    logger.info("Loading models with DDPM scheduler...")
    h.load_models()
    logger.info("Models loaded successfully")

    # Verify scheduler is DDPM
    ddpm_type = type(h.pipe.scheduler).__name__
    logger.info("Scheduler type after load: %s", ddpm_type)

    results: list[dict[str, Any]] = []

    for person_path, garment_path, label in pairs:
        logger.info("=" * 60)
        logger.info("Benchmarking pair: %s", label)

        person_img = Image.open(person_path).convert("RGB")
        garment_img = Image.open(garment_path).convert("RGB")

        # ── Baseline: DDPM ──────────────────────────────────────────────
        logger.info(
            "Running DDPM baseline steps=%d seed=%d",
            args.baseline_steps, args.seed,
        )
        h.IDM_VTON_SCHEDULER = "ddpm"
        # Reset scheduler to DDPM (in case previous run left DPM++)
        from diffusers import DDPMScheduler
        h.pipe.scheduler = DDPMScheduler.from_config(h.pipe.scheduler.config)

        baseline_start = time.perf_counter()
        result_baseline, raw_baseline, meta_baseline = h.run_idm_vton_inference(
            person_img=person_img,
            garment_img=garment_img,
            garment_desc=args.garment_desc,
            cloth_type=args.cloth_type,
            steps=args.baseline_steps,
            seed=args.seed,
            guidance_scale=args.guidance,
            auto_crop=True,
            crop_preserve_lower=True,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        baseline_ms = (time.perf_counter() - baseline_start) * 1000

        # ── Candidate: DPM++ ────────────────────────────────────────────
        logger.info(
            "Running DPM++ candidate steps=%d seed=%d",
            args.candidate_steps, args.seed,
        )
        from diffusers import DPMSolverMultistepScheduler
        h.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            h.pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            solver_order=2,
            use_karras_sigmas=True,
        )
        candidate_start = time.perf_counter()
        result_candidate, raw_candidate, meta_candidate = h.run_idm_vton_inference(
            person_img=person_img,
            garment_img=garment_img,
            garment_desc=args.garment_desc,
            cloth_type=args.cloth_type,
            steps=args.candidate_steps,
            seed=args.seed,
            guidance_scale=args.guidance,
            auto_crop=True,
            crop_preserve_lower=True,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        candidate_ms = (time.perf_counter() - candidate_start) * 1000

        # ── Compute metrics ────────────────────────────────────────────
        ref_np = np.array(result_baseline, dtype=np.uint8)
        pred_np = np.array(result_candidate, dtype=np.uint8)
        metrics = compute_metrics(ref_np, pred_np)

        entry: dict[str, Any] = {
            "label": label,
            "baseline_scheduler": "ddpm",
            "baseline_steps": args.baseline_steps,
            "baseline_latency_ms": round(baseline_ms, 1),
            "candidate_scheduler": "dpmpp_sde_karras",
            "candidate_steps": args.candidate_steps,
            "candidate_latency_ms": round(candidate_ms, 1),
            "latency_savings_ms": round(baseline_ms - candidate_ms, 1),
            "latency_savings_pct": round(
                (baseline_ms - candidate_ms) / baseline_ms * 100, 1
            ),
            "psnr": round(metrics["psnr"], 4),
            "ssim": round(metrics["ssim"], 4),
        }

        # Save output images
        out_dir = Path(args.output_dir)
        result_baseline.save(str(out_dir / f"baseline_ddpm_{label}.png"))
        result_candidate.save(str(out_dir / f"candidate_dpmpp_{label}.png"))

        results.append(entry)
        logger.info(
            "Results: PSNR=%.2f SSIM=%.4f "
            "DDPM=%.0fms DPM++=%.0fms savings=%.0fms (%.0f%%)",
            metrics["psnr"], metrics["ssim"],
            baseline_ms, candidate_ms,
            baseline_ms - candidate_ms,
            (baseline_ms - candidate_ms) / baseline_ms * 100,
        )

    # ── Write results ──────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.output_dir, f"benchmark_{ts}.json")
    csv_path = os.path.join(args.output_dir, f"benchmark_{ts}.csv")
    txt_path = os.path.join(args.output_dir, f"benchmark_{ts}.txt")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "label", "baseline_scheduler", "baseline_steps",
            "baseline_latency_ms", "candidate_scheduler",
            "candidate_steps", "candidate_latency_ms",
            "latency_savings_ms", "latency_savings_pct",
            "psnr", "ssim",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    # ── Summary ────────────────────────────────────────────────────────
    avg_psnr = np.mean([r["psnr"] for r in results])
    avg_ssim = np.mean([r["ssim"] for r in results])
    avg_baseline_ms = np.mean([r["baseline_latency_ms"] for r in results])
    avg_candidate_ms = np.mean([r["candidate_latency_ms"] for r in results])
    avg_savings_pct = np.mean([r["latency_savings_pct"] for r in results])

    summary = f"""Benchmark Summary
=================
Date: {ts}
Images: {len(results)}

Baseline: DDPM @ {args.baseline_steps} steps
Candidate: DPM++ 2M SDE Karras @ {args.candidate_steps} steps

Average Latency:
  DDPM:    {avg_baseline_ms:.0f} ms
  DPM++:   {avg_candidate_ms:.0f} ms
  Savings: {avg_baseline_ms - avg_candidate_ms:.0f} ms ({avg_savings_pct:.1f}%)

Average Quality:
  PSNR: {avg_psnr:.2f} dB
  SSIM: {avg_ssim:.4f}

Per-Image Results:
"""
    for r in results:
        summary += (
            f"  {r['label']}: PSNR={r['psnr']:.2f} SSIM={r['ssim']:.4f} "
            f"DDPM={r['baseline_latency_ms']:.0f}ms "
            f"DPM++={r['candidate_latency_ms']:.0f}ms "
            f"({r['latency_savings_pct']:.0f}% savings)\n"
        )

    summary += f"""
Results saved to:
  {json_path}
  {csv_path}
  {txt_path}
"""

    print(summary)

    with open(txt_path, "w") as f:
        f.write(summary)

    logger.info("Benchmark complete — results saved to %s", json_path)


if __name__ == "__main__":
    import torch
    main()
