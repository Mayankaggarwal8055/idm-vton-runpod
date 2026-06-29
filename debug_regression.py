#!/usr/bin/env python3
"""RunPod debug script: saves 4 intermediate artifacts per regression pair.

Outputs to /tmp/idm-vton-debug-regression/{trace_id}/:
  - 01_inpaint_mask.png
  - 02_raw_diffusion.png
  - 03_body_preserve.png
  - 04_final_composited.png
  - metadata.json (cloth_type, source_type, garment_subtype, timings)

Usage:
  python debug_regression.py --manifest manifest.json
  python debug_regression.py --person <url> --garment <url> --cloth-type upper_body

Manifest format (JSON array):
  [
    {
      "person_url": "https://...",
      "garment_url": "https://...",
      "cloth_type": "upper_body",
      "garment_subtype": "shirt",
      "source_cloth_type": "dresses",
      "trace_id": "reg_upper_001"
    },
    ...
  ]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("debug_regression")

# ── Output directory ─────────────────────────────────────────────────────
OUTPUT_BASE = Path("/tmp/idm-vton-debug-regression")


def run_single_pair(
    person_url: str,
    garment_url: str,
    cloth_type: str,
    garment_subtype: str,
    source_cloth_type: str,
    trace_id: str,
    steps: int = 50,
    guidance_scale: float = 3.5,
    seed: int = 42,
) -> dict:
    """Run one person-garment pair and save 4 intermediate artifacts."""
    from handler import (
        run_inference,
        download_image,
        TARGET_SIZE,
    )
    from mask_pipeline import DebugArtifacts

    pair_start = time.perf_counter()
    out_dir = OUTPUT_BASE / trace_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Download images ──────────────────────────────────────────────────
    logger.info("downloading person=%s garment=%s trace_id=%s", person_url, garment_url, trace_id)
    download_start = time.perf_counter()
    person_img = download_image(person_url)
    garment_img = download_image(garment_url)
    download_ms = (time.perf_counter() - download_start) * 1000

    # ── Save inputs ──────────────────────────────────────────────────────
    person_img.convert("RGB").save(str(out_dir / "person_input.png"))
    garment_img.convert("RGB").save(str(out_dir / "garment_input.png"))

    # ── Run inference ────────────────────────────────────────────────────
    # Build job_input dict matching what run_inference expects
    job_input = {
        "person_image_url": person_url,
        "garment_image_url": garment_url,
        "cloth_type": cloth_type,
        "garment_subtype": garment_subtype,
        "source_cloth_type": source_cloth_type,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "trace_id": trace_id,
    }

    logger.info(
        "running_inference cloth_type=%s subtype=%s source=%s steps=%d guidance=%.2f trace_id=%s",
        cloth_type, garment_subtype, source_cloth_type, steps, guidance_scale, trace_id,
    )
    inference_start = time.perf_counter()
    result = run_inference(job_input, trace_id)
    inference_ms = (time.perf_counter() - inference_start) * 1000

    total_ms = (time.perf_counter() - pair_start) * 1000

    # ── Read back the 4 intermediates from the debug artifacts dir ────────
    # run_inference saves to /tmp/idm-vton-debug/{trace_id}/ via save_debug_artifacts_v2
    debug_source = Path("/tmp/idm-vton-debug") / trace_id

    artifacts_saved = []
    for src_name, dst_name in [
        ("inpaint_mask.png", "01_inpaint_mask.png"),
        ("raw_output.png", "02_raw_diffusion.png"),
        ("body_preserve_output.png", "03_body_preserve.png"),
        ("final_output.png", "04_final_composited.png"),
        ("mask_overlay.png", "05_mask_overlay.png"),
        ("schp_labels.png", "06_schp_labels.png"),
        ("garment_silhouette.png", "07_garment_silhouette.png"),
        ("protect_mask.png", "08_protect_mask.png"),
        ("final_mask.png", "09_final_mask.png"),
    ]:
        src_path = debug_source / src_name
        dst_path = out_dir / dst_name
        if src_path.exists():
            import shutil
            shutil.copy2(str(src_path), str(dst_path))
            artifacts_saved.append(dst_name)
            logger.info("  saved %s", dst_name)
        else:
            logger.warning("  MISSING %s (source: %s)", dst_name, src_path)

    # ── Save metadata ────────────────────────────────────────────────────
    metadata = {
        "trace_id": trace_id,
        "cloth_type": cloth_type,
        "garment_subtype": garment_subtype,
        "source_cloth_type": source_cloth_type,
        "person_url": person_url,
        "garment_url": garment_url,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "download_ms": round(download_ms, 1),
        "inference_ms": round(inference_ms, 1),
        "total_ms": round(total_ms, 1),
        "artifacts_saved": artifacts_saved,
        "result_status": result.get("status"),
        "result_url": result.get("result_url"),
        "mask_coverage_percent": result.get("mask_coverage_percent"),
        "pipeline_route": result.get("pipeline_route"),
        "source_cloth_type_detected": result.get("source_cloth_type"),
        "is_cross_category": result.get("is_cross_category"),
        "is_draped": result.get("is_draped"),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info("metadata saved to %s", out_dir / "metadata.json")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="RunPod debug: save 4 intermediates per regression pair")
    parser.add_argument("--manifest", type=str, help="Path to JSON manifest of regression pairs")
    parser.add_argument("--person", type=str, help="Single person image URL")
    parser.add_argument("--garment", type=str, help="Single garment image URL")
    parser.add_argument("--cloth-type", type=str, default="upper_body", help="Cloth type")
    parser.add_argument("--garment-subtype", type=str, default="", help="Garment subtype")
    parser.add_argument("--source-cloth-type", type=str, default="", help="Source cloth type")
    parser.add_argument("--trace-id", type=str, default="debug_single", help="Trace ID")
    parser.add_argument("--steps", type=int, default=50, help="Diffusion steps")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        with open(args.manifest) as f:
            manifest = json.load(f)
        logger.info("loaded manifest with %d pairs", len(manifest))
        results = []
        for i, pair in enumerate(manifest):
            logger.info("=" * 60)
            logger.info("PAIR %d/%d: trace_id=%s", i + 1, len(manifest), pair.get("trace_id", f"reg_{i:03d}"))
            logger.info("=" * 60)
            meta = run_single_pair(
                person_url=pair["person_url"],
                garment_url=pair["garment_url"],
                cloth_type=pair.get("cloth_type", "upper_body"),
                garment_subtype=pair.get("garment_subtype", ""),
                source_cloth_type=pair.get("source_cloth_type", ""),
                trace_id=pair.get("trace_id", f"reg_{i:03d}"),
                steps=pair.get("steps", args.steps),
                guidance_scale=pair.get("guidance_scale", args.guidance),
                seed=pair.get("seed", args.seed),
            )
            results.append(meta)

        # Summary
        summary_path = OUTPUT_BASE / "summary.json"
        summary_path.write_text(json.dumps(results, indent=2))
        logger.info("summary saved to %s", summary_path)

    elif args.person and args.garment:
        meta = run_single_pair(
            person_url=args.person,
            garment_url=args.garment,
            cloth_type=args.cloth_type,
            garment_subtype=args.garment_subtype,
            source_cloth_type=args.source_cloth_type,
            trace_id=args.trace_id,
            steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
        )
        logger.info("done: %s", json.dumps(meta, indent=2))

    else:
        parser.error("Either --manifest or --person + --garment is required")


if __name__ == "__main__":
    main()
