"""
EXPERIMENTAL: Test bare-skin inpaint hypothesis.

This script tests whether IDM-VTON can reconstruct a believable body
underneath a garment using only the visible person image as reference.

Test cases:
  A. Saree → bare-skin stage only
  B. Dress → bare-skin stage only
  C. T-shirt → bare-skin stage only

For each test, saves:
  - mask: full-body inpaint mask
  - raw: raw pipeline output
  - final: final output (same as raw for this test)

Run on RunPod GPU instance:
  python test_bare_skin.py

Output directory: /tmp/bare-skin-experiment/
"""

from __future__ import annotations

import io
import os
import sys
import time
import logging
import traceback
from pathlib import Path

import requests
import numpy as np
import torch
from PIL import Image

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bare-skin-test")

# ── Constants ────────────────────────────────────────────────────────
TARGET_SIZE = (768, 1024)
TARGET_W, TARGET_H = TARGET_SIZE

IDM_VTON_DIR = os.environ.get("IDM_VTON_DIR", "/workspace/IDM-VTON")
IDM_VTON_MODEL = os.environ.get("IDM_VTON_MODEL", "/workspace/models/yisol/IDM-VTON")
DENSEPOSE_WEIGHTS = os.environ.get(
    "DENSEPOSE_WEIGHTS",
    "/workspace/IDM-VTON/ckpt/densepose/model_final_162be9.pkl",
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16

# ── Test images ──────────────────────────────────────────────────────
# Using publicly available garment images for testing.
# These are product shots that show the garment clearly.
TEST_IMAGES = {
    "saree": {
        "url": "https://images.unsplash.com/photo-1610030469983-98e550d6193c?w=800",
        "desc": "purple silk saree with gold border",
        "cloth_type": "dresses",
    },
    "dress": {
        "url": "https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=800",
        "desc": "blue floral summer dress",
        "cloth_type": "dresses",
    },
    "tshirt": {
        "url": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=800",
        "desc": "white cotton t-shirt",
        "cloth_type": "upper_body",
    },
}


def download_image(url: str, timeout: int = 60) -> Image.Image:
    """Download image from URL."""
    resp = requests.get(url, timeout=timeout, stream=True)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def load_models():
    """Load all required models (same as handler.py)."""
    global pipe, parsing_model, openpose_model
    global densepose_predictor, densepose_cfg, tensor_transform

    logger.info("Loading models...")

    # Add paths
    if IDM_VTON_DIR not in sys.path:
        sys.path.insert(0, IDM_VTON_DIR)
    gradio_demo_dir = os.path.join(IDM_VTON_DIR, "gradio_demo")
    if gradio_demo_dir not in sys.path:
        sys.path.insert(0, gradio_demo_dir)

    from torchvision import transforms
    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # Import custom modules
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
    from src.unet_hacked_tryon import UNet2DConditionModel as UNet2DConditionModel_tryon
    from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline

    from transformers import (
        CLIPImageProcessor,
        CLIPVisionModelWithProjection,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        AutoTokenizer,
    )
    from diffusers import DDPMScheduler, AutoencoderKL

    # Load components
    unet = UNet2DConditionModel_tryon.from_pretrained(
        IDM_VTON_MODEL, subfolder="unet", torch_dtype=TORCH_DTYPE,
    ).requires_grad_(False)

    tokenizer_one = AutoTokenizer.from_pretrained(
        IDM_VTON_MODEL, subfolder="tokenizer", use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        IDM_VTON_MODEL, subfolder="tokenizer_2", use_fast=False,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        IDM_VTON_MODEL, subfolder="scheduler",
    )
    text_encoder_one = CLIPTextModel.from_pretrained(
        IDM_VTON_MODEL, subfolder="text_encoder", torch_dtype=TORCH_DTYPE,
    ).requires_grad_(False)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        IDM_VTON_MODEL, subfolder="text_encoder_2", torch_dtype=TORCH_DTYPE,
    ).requires_grad_(False)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        IDM_VTON_MODEL, subfolder="image_encoder", torch_dtype=TORCH_DTYPE,
    ).requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(
        IDM_VTON_MODEL, subfolder="vae", torch_dtype=TORCH_DTYPE,
    ).requires_grad_(False)
    unet_encoder = UNet2DConditionModel_ref.from_pretrained(
        IDM_VTON_MODEL, subfolder="unet_encoder", torch_dtype=TORCH_DTYPE,
    ).requires_grad_(False)

    pipe = TryonPipeline.from_pretrained(
        IDM_VTON_MODEL,
        unet=unet, vae=vae,
        feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one, text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one, tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler, image_encoder=image_encoder,
        torch_dtype=TORCH_DTYPE,
    )
    pipe.unet_encoder = unet_encoder
    pipe = pipe.to(DEVICE)

    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("xformers enabled")
        except Exception:
            pass

    # Parsing model
    from preprocess.humanparsing.run_parsing import Parsing
    parsing_model = Parsing(0)

    # OpenPose
    from preprocess.openpose.run_openpose import OpenPose
    openpose_model = OpenPose(0)

    # DensePose
    from detectron2.config import get_cfg
    from densepose import add_densepose_config
    from detectron2.engine.defaults import DefaultPredictor

    densepose_cfg = get_cfg()
    add_densepose_config(densepose_cfg)
    config_path = os.path.join(IDM_VTON_DIR, "configs", "densepose_rcnn_R_50_FPN_s1x.yaml")
    densepose_cfg.merge_from_file(config_path)
    densepose_cfg.MODEL.WEIGHTS = DENSEPOSE_WEIGHTS
    densepose_cfg.MODEL.DEVICE = DEVICE
    densepose_cfg.freeze()
    densepose_predictor = DefaultPredictor(densepose_cfg)

    logger.info("All models loaded")


def run_bare_skin_inpaint(
    person_img: Image.Image,
    steps: int = 40,
    seed: int = 42,
    guidance_scale: float = 4.5,
    trace_id: str = "",
) -> tuple[Image.Image, dict[str, object]]:
    """
    Run bare-skin inpaint on a person image.

    Uses:
    - Full-body mask (all garment labels + arms)
    - Person image as IP-Adapter reference (not a garment)
    - Person image as cloth tensor (not a garment)
    - Minimal prompt (no face/body terms)
    """
    from mask_pipeline import (
        build_final_full_body_mask,
        assert_binary_mask,
        validate_mask_integrity,
        validate_mask_coverage,
    )

    human_img = person_img.convert("RGB").resize(TARGET_SIZE)

    # ── Build full-body mask ──────────────────────────────────────────
    model_parse, _ = parsing_model(human_img)
    schp_np = np.array(model_parse) if not isinstance(model_parse, np.ndarray) else model_parse
    if isinstance(model_parse, torch.Tensor):
        schp_np = model_parse.cpu().numpy()
    if schp_np.ndim == 3:
        schp_np = schp_np.squeeze(0)
    schp_np = schp_np.astype(np.uint8)

    final_mask_np, inpaint_mask_np, protect_mask_np = build_final_full_body_mask(schp_np)
    assert_binary_mask(final_mask_np, "bare_skin_final_mask")
    validate_mask_integrity(final_mask_np, "bare_skin_final_mask")

    final_mask = Image.fromarray(final_mask_np, mode="L")
    if final_mask.size != TARGET_SIZE:
        final_mask = final_mask.resize(TARGET_SIZE, Image.LANCZOS)
        final_mask = final_mask.point(lambda x: 255 if x > 127 else 0)

    mask_v = validate_mask_coverage(final_mask, "dresses")
    logger.info(
        "mask coverage=%.1f%% valid=%s",
        mask_v["coverage_percent"], mask_v["valid"],
    )

    # ── DensePose ─────────────────────────────────────────────────────
    import cv2
    from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation

    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    with torch.no_grad():
        densepose_pred = densepose_predictor(human_img_arg)
        if "instances" not in densepose_pred or len(densepose_pred["instances"]) == 0:
            pose_img = Image.new("RGB", TARGET_SIZE, (128, 128, 128))
        else:
            densepose_outputs = densepose_pred["instances"]
            from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer
            from densepose.vis.extractor import create_extractor
            vis = DensePoseResultsFineSegmentationVisualizer(cfg=densepose_cfg)
            extractor = create_extractor(vis)
            data = extractor(densepose_outputs)
            gray_img = cv2.cvtColor(human_img_arg, cv2.COLOR_BGR2GRAY)
            gray_img = np.tile(gray_img[:, :, np.newaxis], [1, 1, 3])
            pose_img = vis.visualize(gray_img, data)
            pose_img = pose_img[:, :, ::-1]
            pose_img = Image.fromarray(pose_img).resize(TARGET_SIZE)

    # ── KEY: Person as IP-Adapter reference ───────────────────────────
    ip_adapter_ref = human_img

    # ── KEY: Person as cloth tensor ───────────────────────────────────
    cloth_tensor_src = human_img

    # ── Minimal prompt ────────────────────────────────────────────────
    prompt = "person wearing no clothing, bare skin, photorealistic"
    negative_prompt = (
        "worst quality, low quality, deformed, distorted, disfigured, "
        "bad anatomy, bad proportions, extra limbs, missing limbs, "
        "ugly, blurry, watermark, signature, text, logo, "
        "smooth plastic, airbrushed, cg render, "
        "original clothing, old garment, previous outfit, "
        "residual fabric, double garment, layered clothing"
    )

    # ── Encode prompts ────────────────────────────────────────────────
    with torch.inference_mode():
        with torch.cuda.amp.autocast(dtype=TORCH_DTYPE):
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
                prompt, num_images_per_prompt=1,
                do_classifier_free_guidance=True, negative_prompt=negative_prompt,
            )
            prompt_c = "a photo of a person, high quality"
            prompt_embeds_c, _, _, _ = pipe.encode_prompt(
                prompt_c, num_images_per_prompt=1,
                do_classifier_free_guidance=False, negative_prompt=negative_prompt,
            )

    # ── Prepare tensors ───────────────────────────────────────────────
    pose_tensor = tensor_transform(pose_img).unsqueeze(0).to(DEVICE, TORCH_DTYPE)
    cloth_tensor = tensor_transform(cloth_tensor_src).unsqueeze(0).to(DEVICE, TORCH_DTYPE)
    generator = torch.Generator(DEVICE).manual_seed(seed)

    # ── Run pipeline ──────────────────────────────────────────────────
    with torch.inference_mode():
        with torch.cuda.amp.autocast(dtype=TORCH_DTYPE):
            logger.info("Running inference: steps=%d guidance=%.2f seed=%d", steps, guidance_scale, seed)
            pipe_output = pipe(
                prompt_embeds=prompt_embeds.to(DEVICE, TORCH_DTYPE),
                negative_prompt_embeds=negative_prompt_embeds.to(DEVICE, TORCH_DTYPE),
                pooled_prompt_embeds=pooled_prompt_embeds.to(DEVICE, TORCH_DTYPE),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(DEVICE, TORCH_DTYPE),
                num_inference_steps=steps,
                generator=generator,
                strength=1.0,
                pose_img=pose_tensor.to(DEVICE, TORCH_DTYPE),
                text_embeds_cloth=prompt_embeds_c.to(DEVICE, TORCH_DTYPE),
                cloth=cloth_tensor.to(DEVICE, TORCH_DTYPE),
                mask_image=final_mask,
                image=human_img,
                height=TARGET_H,
                width=TARGET_W,
                ip_adapter_image=ip_adapter_ref,
                guidance_scale=guidance_scale,
            )
            images = pipe_output[0]
            if not images:
                raise RuntimeError("Pipeline returned empty images")

    raw_output = images[0].copy()

    mask_meta = {
        "coverage_percent": mask_v["coverage_percent"],
        "coverage_valid": mask_v["valid"],
        "schp_labels": schp_np,
        "final_mask_np": final_mask_np,
    }

    return raw_output, mask_meta


def main():
    """Run all 3 test cases."""
    output_dir = Path("/tmp/bare-skin-experiment")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BARE-SKIN INPAINT EXPERIMENT")
    logger.info("=" * 60)

    # Load models
    load_models()

    results = {}

    for test_name, test_config in TEST_IMAGES.items():
        logger.info("=" * 60)
        logger.info("TEST: %s", test_name.upper())
        logger.info("Description: %s", test_config["desc"])
        logger.info("=" * 60)

        try:
            # Download person image
            logger.info("Downloading image: %s", test_config["url"])
            person_img = download_image(test_config["url"])
            logger.info("Downloaded: %dx%d", person_img.size[0], person_img.size[1])

            # Save original
            person_img.save(str(output_dir / f"{test_name}_original.png"))

            # Run bare-skin inpaint
            t0 = time.perf_counter()
            raw_output, mask_meta = run_bare_skin_inpaint(
                person_img=person_img,
                steps=40,
                seed=42,
                guidance_scale=4.5,
                trace_id=test_name,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # Save outputs
            mask_np = mask_meta["final_mask_np"]
            Image.fromarray(mask_np, mode="L").save(
                str(output_dir / f"{test_name}_mask.png")
            )
            raw_output.save(str(output_dir / f"{test_name}_raw.png"))
            raw_output.save(str(output_dir / f"{test_name}_final.png"))

            # Log results
            logger.info("TEST %s COMPLETE: %.0fms", test_name.upper(), elapsed_ms)
            logger.info("  mask coverage: %.1f%%", mask_meta["coverage_percent"])

            results[test_name] = {
                "status": "success",
                "elapsed_ms": elapsed_ms,
                "coverage": mask_meta["coverage_percent"],
            }

        except Exception as exc:
            logger.error("TEST %s FAILED: %s", test_name.upper(), exc)
            traceback.print_exc()
            results[test_name] = {
                "status": "failed",
                "error": str(exc),
            }

    # ── Summary ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    for name, result in results.items():
        status = result["status"]
        if status == "success":
            logger.info(
                "  %s: SUCCESS (%.0fms, coverage=%.1f%%)",
                name, result["elapsed_ms"], result["coverage"],
            )
        else:
            logger.info("  %s: FAILED (%s)", name, result.get("error", "unknown"))

    logger.info("")
    logger.info("Output directory: %s", output_dir)
    logger.info("Files saved:")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            logger.info("  %s (%d bytes)", f.name, f.stat().st_size)

    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
