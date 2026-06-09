"""
RunPod Serverless Handler — IDM-VTON Virtual Try-On Worker
===========================================================

ARCHITECTURE:
  This worker runs IDM-VTON (Improving Diffusion Model for Virtual Try-On)
  as a RunPod serverless endpoint. All preprocessing (human parsing,
  OpenPose, DensePose, mask generation) and inference happens inside
  the worker on the GPU.

  Frontend -> Next.js -> RunPod worker -> Cloudinary -> Frontend

  The worker receives:
    - person_image_url: Cloudinary URL of the user photo
    - garment_image_url: Cloudinary URL of the garment photo
    - garment_desc: Text description of the garment ("blue cotton hoodie")
    - cloth_type: "upper_body", "lower_body", or "dresses"
    - steps: Number of inference steps (default 30)
    - seed: Random seed for reproducibility

  The worker returns:
    - result_url: Cloudinary URL of the generated try-on image

PERFORMANCE OPTIMIZATIONS:
  - All models loaded once at cold start, reused across requests
  - torch.inference_mode() for inference
  - Persistent HTTP session for image downloads
  - Cloudinary direct upload (no base64 round-trip)
  - Per-stage timing logs
  - No intermediate file I/O unless necessary
"""

from __future__ import annotations

import io
import os
import sys
import time
import logging
import random
import threading
import traceback
from typing import Any

import runpod
import requests
import numpy as np
import torch
from PIL import Image
import cloudinary
import cloudinary.uploader
from requests.adapters import HTTPAdapter

# ── Logging ────────────────────────────────────────────────────────────────

logger = logging.getLogger("idm-vton.worker")
_handler_configured = False


def _ensure_logging():
    global _handler_configured
    if not _handler_configured:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        _handler_configured = True


# ── Constants & Env ───────────────────────────────────────────────────────

TARGET_SIZE = (768, 1024)
TARGET_W, TARGET_H = TARGET_SIZE
CLOUDINARY_FOLDER = os.environ.get("CLOUDINARY_FOLDER", "trylix/tryon/results")

IDM_VTON_MODEL = os.environ.get("IDM_VTON_MODEL", "yisol/IDM-VTON")
DENOISE_STEPS = int(os.environ.get("IDM_VTON_STEPS", "30"))
GUIDANCE_SCALE = float(os.environ.get("IDM_VTON_GUIDANCE", "2.0"))
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ── Global state (loaded once at cold start) ──────────────────────────────

pipe = None
parsing_model = None
openpose_model = None
densepose_predictor = None
densepose_cfg = None
tensor_transform = None
get_mask_location_fn = None
_WARM = threading.Event()
_STARTUP_TIME = time.perf_counter()
_REUSE_COUNT: int = 0

# ── HTTP Session (persistent connection pool) ─────────────────────────────

_SESSION: requests.Session | None = None
_SESSION_LOCK = threading.Lock()


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    with _SESSION_LOCK:
        if _SESSION is not None:
            return _SESSION
        session = requests.Session()
        session.headers.update({
            "User-Agent": "TryLix-Worker/1.0",
            "Accept": "image/webp,image/jpeg,image/png,*/*",
        })
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=2)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _SESSION = session
        logger.info("http_session_created pool_maxsize=8")
        return session


# ── Cloudinary Upload ─────────────────────────────────────────────────────

def _configure_cloudinary() -> bool:
    cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME")
    api_key = os.environ.get("CLOUDINARY_API_KEY")
    api_secret = os.environ.get("CLOUDINARY_API_SECRET")
    if not all([cloud_name, api_key, api_secret]):
        logger.warning("Cloudinary not configured - cannot upload results")
        return False
    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret,
        secure=True,
    )
    return True


def _upload_to_cloudinary(image: Image.Image, job_id: str) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    buffer.seek(0)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            result = cloudinary.uploader.upload(
                buffer,
                folder=CLOUDINARY_FOLDER,
                public_id=f"result_{job_id}",
                resource_type="image",
                overwrite=False,
            )
            url = str(result["secure_url"])
            logger.info("cloudinary_upload_complete result_url=%s", url)
            return url
        except Exception as exc:
            last_error = exc
            logger.warning("cloudinary_upload_failed attempt=%s error=%s", attempt + 1, exc)
            if attempt < 2:
                buffer.seek(0)
                time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"Cloudinary upload failed after 3 attempts: {last_error}")


# ── Image Download ────────────────────────────────────────────────────────

def download_image(url: str, timeout: int = 60) -> Image.Image:
    session = _get_session()
    resp = session.get(url, timeout=timeout, stream=True)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


# ── Model Loading ─────────────────────────────────────────────────────────

def load_models():
    """Load all IDM-VTON models at cold start.

    Loads:
      - IDM-VTON pipeline (SDXL-based, from HuggingFace)
      - Human parsing model (ONNX-based)
      - OpenPose model
      - DensePose predictor (detectron2)
    """
    global pipe, parsing_model, openpose_model
    global densepose_predictor, densepose_cfg, tensor_transform, get_mask_location_fn

    if pipe is not None:
        return

    load_start = time.perf_counter()

    # ── Add IDM-VTON src to path ──────────────────────────────────────
    idm_dir = os.environ.get("IDM_VTON_DIR", "/workspace/IDM-VTON")
    if idm_dir not in sys.path:
        sys.path.insert(0, idm_dir)
    gradio_demo_dir = os.path.join(idm_dir, "gradio_demo")
    if gradio_demo_dir not in sys.path:
        sys.path.insert(0, gradio_demo_dir)

    from torchvision import transforms
    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # ── Load IDM-VTON custom UNets ────────────────────────────────────
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
    from src.unet_hacked_tryon import UNet2DConditionModel as UNet2DConditionModel_tryon
    from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
    from transformers import (
        CLIPImageProcessor, CLIPVisionModelWithProjection,
        CLIPTextModel, CLIPTextModelWithProjection, AutoTokenizer,
    )
    from diffusers import DDPMScheduler, AutoencoderKL

    logger.info("Loading IDM-VTON model components from %s ...", IDM_VTON_MODEL)

    # Load UNet
    unet = UNet2DConditionModel_tryon.from_pretrained(
        IDM_VTON_MODEL, subfolder="unet", torch_dtype=torch.float16,
    )
    unet.requires_grad_(False)

    # Load tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        IDM_VTON_MODEL, subfolder="tokenizer", use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        IDM_VTON_MODEL, subfolder="tokenizer_2", use_fast=False,
    )

    # Load scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(IDM_VTON_MODEL, subfolder="scheduler")

    # Load text encoders
    text_encoder_one = CLIPTextModel.from_pretrained(
        IDM_VTON_MODEL, subfolder="text_encoder", torch_dtype=torch.float16,
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        IDM_VTON_MODEL, subfolder="text_encoder_2", torch_dtype=torch.float16,
    )

    # Load image encoder
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        IDM_VTON_MODEL, subfolder="image_encoder", torch_dtype=torch.float16,
    )

    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        IDM_VTON_MODEL, subfolder="vae", torch_dtype=torch.float16,
    )

    # Load UNet encoder (garment feature extractor)
    UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
        IDM_VTON_MODEL, subfolder="unet_encoder", torch_dtype=torch.float16,
    )
    UNet_Encoder.requires_grad_(False)

    # Build the pipeline
    pipe = TryonPipeline.from_pretrained(
        IDM_VTON_MODEL,
        unet=unet,
        vae=vae,
        feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
    )
    pipe.unet_encoder = UNet_Encoder

    logger.info("IDM-VTON pipeline loaded")

    # Load human parsing model
    from preprocess.humanparsing.run_parsing import Parsing
    parsing_model = Parsing(0)

    # Load OpenPose model
    from preprocess.openpose.run_openpose import OpenPose
    openpose_model = OpenPose(0)

    logger.info("Parsing and OpenPose models loaded")

    # Load DensePose predictor
    from detectron2.config import get_cfg
    from densepose import add_densepose_config
    from detectron2.engine.defaults import DefaultPredictor

    densepose_cfg = get_cfg()
    add_densepose_config(densepose_cfg)
    config_path = os.path.join(
        idm_dir, "configs", "densepose_rcnn_R_50_FPN_s1x.yaml"
    )
    densepose_cfg.merge_from_file(config_path)

    densepose_weights = os.environ.get(
        "DENSEPOSE_WEIGHTS",
        "/workspace/models/densepose/model_final_162be9.pkl",
    )
    densepose_cfg.MODEL.WEIGHTS = densepose_weights
    densepose_cfg.MODEL.DEVICE = "cuda"
    densepose_cfg.freeze()

    densepose_predictor = DefaultPredictor(densepose_cfg)
    logger.info("DensePose predictor loaded")

    # Import mask utility (used by inference pipeline)
    from utils_mask import get_mask_location as _get_mask_location
    get_mask_location_fn = _get_mask_location

    load_ms = (time.perf_counter() - load_start) * 1000
    logger.info("models_ready model_load_ms=%.0f", load_ms)


# ── Warmup ────────────────────────────────────────────────────────────────

def warmup():
    """Initialize all models and warm GPU.

    Runs ONCE at cold start. Thread-safe via _WARM event.
    """
    global _REUSE_COUNT
    if _WARM.is_set():
        return

    logger.info("=" * 60)
    logger.info("COLD START BEGIN")
    logger.info("=" * 60)

    load_models()

    # GPU warm-up
    try:
        torch.cuda.synchronize()
        logger.info("gpu_warmup_ready=True")
    except Exception as exc:
        logger.warning("gpu_warmup_skipped error=%s", exc)

    cloudinary_ok = _configure_cloudinary()

    startup_total_ms = (time.perf_counter() - _STARTUP_TIME) * 1000
    logger.info("=" * 60)
    logger.info("COLD START COMPLETE")
    logger.info("  startup_total_ms=%.0f", startup_total_ms)
    logger.info("  cloudinary_configured=%s", cloudinary_ok)
    logger.info("=" * 60)

    _WARM.set()
    _REUSE_COUNT = 0


# ── IDM-VTON Inference ────────────────────────────────────────────────────

def run_idm_vton_inference(
    person_img: Image.Image,
    garment_img: Image.Image,
    garment_desc: str,
    cloth_type: str,
    steps: int = 30,
    seed: int = 42,
    auto_crop: bool = True,
) -> Image.Image:
    """Run IDM-VTON inference on preprocessed inputs.

    Args:
        person_img: Person image (RGB, any size — will be resized to 768x1024).
        garment_img: Garment image (RGB, any size — will be resized to 768x1024).
        garment_desc: Text description of the garment.
        cloth_type: 'upper_body', 'lower_body', or 'dresses'.
        steps: Number of denoising steps.
        seed: Random seed.
        auto_crop: Whether to auto-crop person image to 3:4 aspect ratio.

    Returns:
        PIL Image of the generated try-on result.
    """
    global pipe, parsing_model, openpose_model
    global densepose_predictor, densepose_cfg, tensor_transform, get_mask_location_fn

    import cv2

    device = DEVICE

    # Move models to device
    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)

    # ── Resize inputs ─────────────────────────────────────────────────
    garm_img = garment_img.convert("RGB").resize(TARGET_SIZE)
    human_img_orig = person_img.convert("RGB")

    # ── Auto-crop and resize ────────────────────────────────────────────
    width, height = human_img_orig.size
    left, top, crop_size = 0.0, 0.0, None

    if auto_crop:
        # Center crop to 3:4 aspect ratio
        target_width = int(min(width, height * (TARGET_W / TARGET_H)))
        target_height = int(min(height, width * (TARGET_H / TARGET_W)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize(TARGET_SIZE)
    else:
        human_img = human_img_orig.resize(TARGET_SIZE)

    # ── Generate mask via human parsing + OpenPose ────────────────────
    keypoints = openpose_model(human_img.resize((384, 512)))
    model_parse, _ = parsing_model(human_img.resize((384, 512)))
    mask, _ = get_mask_location_fn('hd', cloth_type, model_parse, keypoints)
    mask = mask.resize(TARGET_SIZE)

    # ── DensePose ─────────────────────────────────────────────────────
    from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation

    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    # Run DensePose
    with torch.no_grad():
        densepose_outputs = densepose_predictor(human_img_arg)["instances"]

    # Generate DensePose visualization (segmentation map)
    from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer
    from densepose.vis.extractor import create_extractor

    vis = DensePoseResultsFineSegmentationVisualizer(cfg=densepose_cfg)
    extractor = create_extractor(vis)
    data = extractor(densepose_outputs)

    # Create a grayscale image for the pose map
    gray_img = cv2.cvtColor(human_img_arg, cv2.COLOR_BGR2GRAY)
    gray_img = np.tile(gray_img[:, :, np.newaxis], [1, 1, 3])
    pose_img = vis.visualize(gray_img, data)
    pose_img = pose_img[:, :, ::-1]  # BGR -> RGB
    pose_img = Image.fromarray(pose_img).resize(TARGET_SIZE)

    # ── Encode prompts ───────────────────────────────────────────────
    # IDM-VTON uses two prompts: one for the person (class prompt) and
    # one for the garment (caption prompt). The garment description is
    # critical for preserving fabric texture and pattern details.
    prompt = "model is wearing " + garment_desc
    negative_prompt = (
        "monochrome, lowres, bad anatomy, worst quality, low quality, "
        "deformed, distorted, disfigured, bad proportions, "
        "extra limbs, missing limbs, cloned head, body out of frame, "
        "poorly drawn face, mutation, mutated, extra fingers, "
        "ugly, blurry, watermark, signature, text, logo"
    )

    with torch.inference_mode():
        with torch.cuda.amp.autocast():
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
                pipe.encode_prompt(
                    prompt,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )

            # Garment caption (no CFG — single forward pass)
            prompt_c = "a photo of " + garment_desc
            prompt_embeds_c, _, _, _ = pipe.encode_prompt(
                prompt_c,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=negative_prompt,
            )

    # ── Prepare tensors ───────────────────────────────────────────────
    pose_tensor = tensor_transform(pose_img).unsqueeze(0).to(device, torch.float16)
    garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(device, torch.float16)

    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None

    # ── Inference ─────────────────────────────────────────────────────
    with torch.inference_mode():
        with torch.cuda.amp.autocast():
            images = pipe(
                prompt_embeds=prompt_embeds.to(device, torch.float16),
                negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                num_inference_steps=steps,
                generator=generator,
                strength=1.0,
                pose_img=pose_tensor.to(device, torch.float16),
                text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                cloth=garm_tensor.to(device, torch.float16),
                mask_image=mask,
                image=human_img,
                height=TARGET_H,
                width=TARGET_W,
                ip_adapter_image=garm_img.resize(TARGET_SIZE),
                guidance_scale=GUIDANCE_SCALE,
            )[0]

    # ── Post-process: paste back if cropped ───────────────────────────
    if auto_crop and crop_size is not None:
        out_img = images.resize(crop_size)
        final_img = human_img_orig.copy()
        final_img.paste(out_img, (int(left), int(top)))
        return final_img
    else:
        return images[0]


# ── Inference (per job) ──────────────────────────────────────────────────

def run_inference(job_input: dict[str, Any], job_id: str) -> dict[str, Any]:
    """Run IDM-VTON inference.

    Inputs:
      - person_image_url: URL of the user's photo
      - garment_image_url: URL of the garment image
      - garment_desc: Text description of the garment
      - cloth_type: 'upper_body', 'lower_body', or 'dresses'
      - steps: Number of inference steps (default 30)
      - seed: Random seed (default random)

    Returns:
      - status: 'success' or 'error'
      - result_url: Cloudinary URL of the result
      - timings in ms
    """
    job_start = time.perf_counter()

    person_url = job_input.get("person_image_url") or job_input.get("person_image")
    garment_url = job_input.get("garment_image_url") or job_input.get("garment_image")
    garment_desc = job_input.get("garment_desc") or job_input.get("garment_description") or "garment"
    cloth_type = job_input.get("cloth_type", "upper_body")
    steps = int(job_input.get("steps", DENOISE_STEPS))
    seed = int(job_input.get("seed", random.randint(0, 2**31 - 1)))

    if not person_url or not garment_url:
        raise ValueError("Missing required inputs: person_image_url and garment_image_url")

    logger.info(
        "inference_start cloth_type=%s steps=%s garment_desc=%s",
        cloth_type, steps, garment_desc,
    )

    # Normalize cloth_type to IDM-VTON format
    cloth_type_map = {
        "upper": "upper_body", "upper_body": "upper_body",
        "lower": "lower_body", "lower_body": "lower_body",
        "dress": "dresses", "dresses": "dresses", "overall": "dresses",
    }
    vton_type = cloth_type_map.get(cloth_type, "upper_body")

    # ── Download images ──────────────────────────────────────────────
    download_start = time.perf_counter()
    person_img = download_image(person_url)
    garment_img = download_image(garment_url)
    download_ms = (time.perf_counter() - download_start) * 1000

    logger.info("images_downloaded person_size=%s garment_size=%s", person_img.size, garment_img.size)

    # ── Run IDM-VTON inference ───────────────────────────────────────
    inference_start = time.perf_counter()

    result = run_idm_vton_inference(
        person_img=person_img,
        garment_img=garment_img,
        garment_desc=garment_desc,
        cloth_type=vton_type,
        steps=steps,
        seed=seed,
        auto_crop=True,
    )

    torch.cuda.synchronize()
    inference_ms = (time.perf_counter() - inference_start) * 1000

    # ── Upload result ────────────────────────────────────────────────
    upload_start = time.perf_counter()
    result_url = _upload_to_cloudinary(result, job_id)
    upload_ms = (time.perf_counter() - upload_start) * 1000

    total_ms = (time.perf_counter() - job_start) * 1000

    logger.info(
        "job_complete total_ms=%.0f download_ms=%.0f inference_ms=%.0f upload_ms=%.0f",
        total_ms, download_ms, inference_ms, upload_ms,
    )

    return {
        "status": "success",
        "result_url": result_url,
        "cloth_type_used": vton_type,
        "steps_used": steps,
        "seed": seed,
        "inference_ms": round(inference_ms, 2),
        "upload_ms": round(upload_ms, 2),
        "download_ms": round(download_ms, 2),
        "total_ms": round(total_ms, 2),
    }


# ── RunPod Handler ────────────────────────────────────────────────────────

def handler(job: dict[str, Any]) -> dict[str, Any]:
    """RunPod serverless handler entry point."""
    job_start = time.time()

    if not _WARM.is_set():
        warmup()
        cold_start = True
    else:
        cold_start = False

    global _REUSE_COUNT
    _REUSE_COUNT += 1

    logger.info(
        "handler_invoked cold_start=%s reuse_count=%s job_id=%s",
        cold_start, _REUSE_COUNT, job.get("id", "unknown"),
    )

    user_input = job.get("input", {})
    job_id = str(job.get("id", "unknown"))

    try:
        output = run_inference(user_input, job_id)
        output["cold_start"] = cold_start
        return output
    except Exception as exc:
        total_ms = (time.time() - job_start) * 1000
        logger.error("job_failed total_ms=%.0f error=%s", total_ms, exc, exc_info=True)
        return {
            "status": "error",
            "error": str(exc),
            "error_code": "worker_inference_failed",
            "total_ms": round(total_ms, 2),
            "cold_start": cold_start,
        }


# ── Startup ───────────────────────────────────────────────────────────────

_ensure_logging()
logger.info("=" * 60)
logger.info("IDM-VTON Worker v1.0.0 — loading")
logger.info("target_size=%s", TARGET_SIZE)
logger.info("device=%s", DEVICE)
logger.info("gpu_available=%s", torch.cuda.is_available())
if torch.cuda.is_available():
    dev = torch.cuda.get_device_properties(0)
    logger.info("gpu_device=%s", dev.name)
    logger.info("vram_total_gb=%.1f", dev.total_memory / (1024**3))
logger.info("=" * 60)


if __name__ == "__main__":
    try:
        if not os.environ.get("RUNPOD_WARMUP_DISABLE"):
            warmup()
        runpod.serverless.start({"handler": handler})
    except Exception:
        logger.error("Worker startup failed")
        traceback.print_exc()
        sys.stdout.flush()
        raise
