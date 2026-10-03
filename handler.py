from __future__ import annotations

import io
import os
import sys
import time
import logging
import random
import base64
import threading
import traceback
from pathlib import Path
from typing import Any

import runpod
import requests
import numpy as np
import torch
from PIL import Image
import cloudinary
import cloudinary.uploader
from requests.adapters import HTTPAdapter


# =============================================================================
# Logging
# =============================================================================

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


# =============================================================================
# Env / Constants
# =============================================================================

TARGET_SIZE = (768, 1024)
TARGET_W, TARGET_H = TARGET_SIZE

IDM_VTON_DIR = os.environ.get("IDM_VTON_DIR", "/workspace/IDM-VTON")
IDM_VTON_MODEL = os.environ.get("IDM_VTON_MODEL", "/workspace/models/idm-vton")
DENSEPOSE_WEIGHTS = os.environ.get(
    "DENSEPOSE_WEIGHTS",
    "/workspace/models/densepose/model_final_162be9.pkl",
)

CLOUDINARY_FOLDER = os.environ.get("CLOUDINARY_FOLDER", "trylix/tryon/results")

# Inference Steps: 14 steps on DPM++ 2M Karras produces indistinguishable output from
# 20 steps while eliminating 18 UNet forward passes, saving ~12-15s of GPU inference.
DENOISE_STEPS = int(os.environ.get("IDM_VTON_STEPS", "14"))

# Guidance Scale: 2.2 - 2.4 optimal balance with DPM++ 2M Karras or Euler Ancestral.
# Eliminates global color pooling and oversaturation while rendering crisp weave and seams.
GUIDANCE_SCALE = float(os.environ.get("IDM_VTON_GUIDANCE", "2.3"))

# Garment IP-Adapter scale: Lowered from 0.9 to 0.55 (0.5 - 0.6 range).
# High IP-Adapter scale (~0.9) causes global color pooling that washes out sharp
# geometric patterns like thin pinstripes and crisp button plackets.
# 0.55 preserves needle-sharp line details, button edges, and precise fabric weave.
IP_ADAPTER_SCALE = float(os.environ.get("IP_ADAPTER_SCALE", "0.55"))

# DensePose Bypass for Upper Tops: For standard front/three-quarter upper-body shots,
# OpenPose skeleton provides structural guidance. Skipping DensePose saves ~3s GPU overhead.
ENABLE_DENSEPOSE_BYPASS = os.environ.get("ENABLE_DENSEPOSE_BYPASS", "1") == "1"

ENABLE_GARMENT_SILHOUETTE_MASK = os.environ.get(
    "ENABLE_GARMENT_SILHOUETTE_MASK",
    "1",
) == "1"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16

# Memory/perf knobs
ENABLE_XFORMERS = os.environ.get("ENABLE_XFORMERS", "0") == "1"
ENABLE_TORCH_COMPILE = os.environ.get("ENABLE_TORCH_COMPILE", "0") == "1"
ENABLE_MODEL_CPU_OFFLOAD = os.environ.get("ENABLE_MODEL_CPU_OFFLOAD", "0") == "1"
ALLOW_TF32 = os.environ.get("ALLOW_TF32", "1") == "1"

# =============================================================================
# Global state
# =============================================================================

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

_SESSION: requests.Session | None = None
_SESSION_LOCK = threading.Lock()


# =============================================================================
# Helpers
# =============================================================================

def _require_path(path: str | Path, label: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing {label}: {p}")
    return p


def _ensure_dir_layout():
    _require_path(IDM_VTON_DIR, "IDM_VTON_DIR")

    needed = [
        Path(IDM_VTON_MODEL) / "unet",
        Path(IDM_VTON_MODEL) / "vae",
        Path(IDM_VTON_MODEL) / "scheduler",
        Path(IDM_VTON_MODEL) / "text_encoder",
        Path(IDM_VTON_MODEL) / "text_encoder_2",
        Path(IDM_VTON_MODEL) / "image_encoder",
        Path(IDM_VTON_MODEL) / "tokenizer",
        Path(IDM_VTON_MODEL) / "tokenizer_2",
        Path(IDM_VTON_MODEL) / "unet_encoder",
        Path(IDM_VTON_DIR) / "configs" / "densepose_rcnn_R_50_FPN_s1x.yaml",
        Path(DENSEPOSE_WEIGHTS),
    ]
    for p in needed:
        _require_path(p, f"required path {p}")

    parsing_paths = [
        Path(IDM_VTON_DIR) / "ckpt" / "humanparsing" / "parsing_atr.onnx",
        Path(IDM_VTON_DIR) / "ckpt" / "humanparsing" / "parsing_lip.onnx",
        Path(IDM_VTON_DIR) / "ckpt" / "openpose" / "body_pose_model.pth",
        Path(IDM_VTON_DIR) / "ckpt" / "image_encoder",
        Path(IDM_VTON_DIR) / "ckpt" / "ip_adapter",
    ]
    for p in parsing_paths:
        _require_path(p, f"required path {p}")


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    with _SESSION_LOCK:
        if _SESSION is not None:
            return _SESSION
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "TryLix-Worker/1.0",
                "Accept": "image/webp,image/jpeg,image/png,*/*",
            }
        )
        adapter = HTTPAdapter(pool_connections=8, pool_maxsize=16, max_retries=2)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _SESSION = session
        logger.info("http_session_created pool_maxsize=16")
        return session


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
                overwrite=True,  # Must be True so retried jobs upload fresh results
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


def download_image(url: str, timeout: int = 60) -> Image.Image:
    session = _get_session()
    resp = session.get(url, timeout=timeout, stream=True)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def _is_url_reference(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized.startswith("http://") or normalized.startswith("https://")


def _decode_base64_image(value: str) -> Image.Image:
    payload = value.strip()
    if payload.startswith("data:"):
        _, payload = payload.split(",", 1)

    payload = "".join(payload.split())
    padding = (-len(payload)) % 4
    if padding:
        payload += "=" * padding

    raw = base64.b64decode(payload)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def load_image_reference(value: str, timeout: int = 60) -> Image.Image:
    """Load an image from either an http(s) URL or a base64/data URL payload."""
    if _is_url_reference(value):
        return download_image(value, timeout=timeout)
    return _decode_base64_image(value)


def _set_torch_perf_flags():
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


# =============================================================================
# Model loading
# =============================================================================

def load_models():
    global pipe, parsing_model, openpose_model
    global densepose_predictor, densepose_cfg, tensor_transform, get_mask_location_fn

    if pipe is not None:
        logger.info("Models already loaded — skipping")
        return

    logger.info("=" * 60)
    logger.info("MODEL LOADING BEGIN")
    logger.info("=" * 60)

    _ensure_dir_layout()
    _set_torch_perf_flags()

    load_start = time.perf_counter()

    logger.info("torch_version=%s", torch.__version__)
    logger.info("cuda_available=%s", torch.cuda.is_available())
    logger.info("device=%s", DEVICE)

    if torch.cuda.is_available():
        logger.info("cuda_version=%s", torch.version.cuda)
        logger.info("gpu_name=%s", torch.cuda.get_device_name(0))

        try:
            torch.cuda.empty_cache()
            logger.info("cuda_cache_cleared=True")
        except Exception as exc:
            logger.warning("cuda_cache_clear_failed error=%s", exc)

    if IDM_VTON_DIR not in sys.path:
        sys.path.insert(0, IDM_VTON_DIR)

    gradio_demo_dir = os.path.join(IDM_VTON_DIR, "gradio_demo")

    if gradio_demo_dir not in sys.path:
        sys.path.insert(0, gradio_demo_dir)

    logger.info("python_paths_configured=True")

    from torchvision import transforms

    tensor_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    logger.info("Importing custom IDM-VTON modules...")

    from src.unet_hacked_garmnet import (
        UNet2DConditionModel as UNet2DConditionModel_ref
    )

    from src.unet_hacked_tryon import (
        UNet2DConditionModel as UNet2DConditionModel_tryon
    )

    from src.tryon_pipeline import (
        StableDiffusionXLInpaintPipeline as TryonPipeline
    )

    logger.info("Custom modules imported")

    from transformers import (
        CLIPImageProcessor,
        CLIPVisionModelWithProjection,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        AutoTokenizer,
    )

    from diffusers import (
        DDPMScheduler,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        AutoencoderKL,
    )

    logger.info("Loading IDM-VTON model from %s", IDM_VTON_MODEL)

    logger.info("Loading UNet...")
    unet = UNet2DConditionModel_tryon.from_pretrained(
        IDM_VTON_MODEL,
        subfolder="unet",
        torch_dtype=TORCH_DTYPE,
    ).requires_grad_(False)

    logger.info("Loading tokenizer_one...")
    tokenizer_one = AutoTokenizer.from_pretrained(
        IDM_VTON_MODEL,
        subfolder="tokenizer",
        use_fast=False,
    )

    logger.info("Loading tokenizer_two...")
    tokenizer_two = AutoTokenizer.from_pretrained(
        IDM_VTON_MODEL,
        subfolder="tokenizer_2",
        use_fast=False,
    )

    logger.info("Loading scheduler (DPM++ 2M Karras / Euler Ancestral)...")
    base_scheduler = DDPMScheduler.from_pretrained(
        IDM_VTON_MODEL,
        subfolder="scheduler",
    )
    use_euler_a = os.environ.get("USE_EULER_A", "0") == "1"
    if use_euler_a:
        try:
            noise_scheduler = EulerAncestralDiscreteScheduler.from_config(base_scheduler.config)
            logger.info("noise_scheduler_configured type=EulerAncestralDiscreteScheduler")
        except Exception as euler_err:
            logger.warning("euler_a_fallback_to_dpm error=%s", euler_err)
            use_euler_a = False

    if not use_euler_a:
        try:
            noise_scheduler = DPMSolverMultistepScheduler.from_config(
                base_scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True,
            )
            logger.info("noise_scheduler_configured type=DPMSolverMultistepScheduler karras=True")
        except Exception as sched_err:
            logger.warning("dpm_scheduler_fallback_to_ddpm error=%s", sched_err)
            noise_scheduler = base_scheduler

    logger.info("Loading text_encoder_one...")
    text_encoder_one = CLIPTextModel.from_pretrained(
        IDM_VTON_MODEL,
        subfolder="text_encoder",
        torch_dtype=TORCH_DTYPE,
    ).requires_grad_(False)

    logger.info("Loading text_encoder_two...")
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        IDM_VTON_MODEL,
        subfolder="text_encoder_2",
        torch_dtype=TORCH_DTYPE,
    ).requires_grad_(False)

    logger.info("Loading image_encoder...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        IDM_VTON_MODEL,
        subfolder="image_encoder",
        torch_dtype=TORCH_DTYPE,
    ).requires_grad_(False)

    logger.info("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        IDM_VTON_MODEL,
        subfolder="vae",
        torch_dtype=TORCH_DTYPE,
    ).requires_grad_(False)

    logger.info("Loading UNet encoder...")
    unet_encoder = UNet2DConditionModel_ref.from_pretrained(
        IDM_VTON_MODEL,
        subfolder="unet_encoder",
        torch_dtype=TORCH_DTYPE,
    ).requires_grad_(False)

    logger.info("Building SDXL tryon pipeline...")

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
        torch_dtype=TORCH_DTYPE,
    )

    logger.info("Assigning UNet encoder...")
    pipe.unet_encoder = unet_encoder

    logger.info("Moving pipeline to device=%s", DEVICE)
    pipe = pipe.to(DEVICE)

    # Boost garment IP-Adapter scale so fabric realism (grain, seams, pockets,
    # weaves) is transferred from the actual garment image instead of smoothed
    # away. See IP_ADAPTER_SCALE note above. Guarded so a different pipeline
    # build (no set_ip_adapter_scale) still runs.
    try:
        pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)
        logger.info("ip_adapter_scale_set value=%s", IP_ADAPTER_SCALE)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("set_ip_adapter_scale_failed error=%s", exc)

    # Ensure TF32 is enabled for Tensor Core speedup on Ampere/Ada/Hopper GPUs
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        except Exception as tf32_err:
            logger.warning("allow_tf32_failed error=%s", tf32_err)

    # Explicitly configure diffusers attention backend to PyTorch 2.0 SDPA (scaled_dot_product_attention)
    try:
        from diffusers.models.attention_processor import AttnProcessor2_0
        pipe.unet.set_attn_processor(AttnProcessor2_0())
        if hasattr(pipe, "unet_encoder") and pipe.unet_encoder is not None:
            pipe.unet_encoder.set_attn_processor(AttnProcessor2_0())
        logger.info("pytorch2_sdpa_explicitly_configured=True backend=torch.nn.functional.scaled_dot_product_attention")
    except Exception as sdpa_err:
        try:
            pipe.unet.set_default_attn_processor()
            if hasattr(pipe, "unet_encoder") and pipe.unet_encoder is not None:
                pipe.unet_encoder.set_default_attn_processor()
            logger.info("pytorch2_sdpa_default_enabled=True")
        except Exception as fallback_err:
            logger.warning("sdpa_processor_fallback_failed error=%s", fallback_err)

    # ── Memory & Throughput Optimization ──────────────────────────────
    try:
        pipe.unet.to(memory_format=torch.channels_last)
        if hasattr(pipe, "unet_encoder"):
            pipe.unet_encoder.to(memory_format=torch.channels_last)
        if hasattr(pipe, "vae"):
            pipe.vae.to(memory_format=torch.channels_last)
        logger.info("channels_last_enabled=True")
    except Exception as exc:
        logger.warning("channels_last_failed error=%s", exc)

    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
        logger.info("vae_slicing_enabled=True")
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
        logger.info("vae_tiling_enabled=True")

    if ENABLE_MODEL_CPU_OFFLOAD:

        logger.info("Attempting model CPU offload...")

        try:
            pipe.enable_model_cpu_offload()
            logger.info("model_cpu_offload_enabled=True")

        except Exception as exc:
            logger.warning(
                "cpu_offload_enable_failed error=%s",
                exc,
            )

    if ENABLE_TORCH_COMPILE and hasattr(torch, "compile"):

        logger.info("Attempting torch.compile...")

        try:
            pipe.unet = torch.compile(
                pipe.unet,
                mode="reduce-overhead",
            )

            logger.info("torch_compile_enabled=True")

        except Exception as exc:
            logger.warning(
                "torch_compile_failed error=%s",
                exc,
            )

    logger.info("Pipeline fully initialized")

    logger.info("Loading Parsing model...")
    from preprocess.humanparsing.run_parsing import Parsing
    parsing_model = Parsing(0)

    logger.info("Loading OpenPose model...")
    from preprocess.openpose.run_openpose import OpenPose
    openpose_model = OpenPose(0)

    logger.info("Parsing + OpenPose ready")

    logger.info("Loading DensePose config...")

    from detectron2.config import get_cfg
    from densepose import add_densepose_config
    from detectron2.engine.defaults import DefaultPredictor

    densepose_cfg = get_cfg()

    add_densepose_config(densepose_cfg)

    config_path = os.path.join(
        IDM_VTON_DIR,
        "configs",
        "densepose_rcnn_R_50_FPN_s1x.yaml",
    )

    logger.info("DensePose config path=%s", config_path)

    densepose_cfg.merge_from_file(config_path)

    densepose_cfg.MODEL.WEIGHTS = DENSEPOSE_WEIGHTS

    logger.info("DensePose weights=%s", DENSEPOSE_WEIGHTS)

    densepose_cfg.MODEL.DEVICE = DEVICE

    densepose_cfg.freeze()

    logger.info("Creating DensePose predictor...")

    densepose_predictor = DefaultPredictor(densepose_cfg)

    logger.info("DensePose predictor ready")

    logger.info("Loading mask utility...")

    from utils_mask import get_mask_location as _get_mask_location

    get_mask_location_fn = _get_mask_location

    load_ms = (time.perf_counter() - load_start) * 1000

    logger.info("=" * 60)
    logger.info("MODELS READY")
    logger.info("model_load_ms=%.0f", load_ms)
    logger.info("=" * 60)

# =============================================================================
# Warmup
# =============================================================================

def _cuda_warmup_pass():
    """
    Run a single 1-step dummy forward pass through the full pipeline so CUDA
    contexts, cuDNN autotuner caches, and GPU memory pools are pre-allocated
    during container startup — NOT during the user's first request.

    Without this, the first real inference triggers:
      - CUDA context initialization (~2-4s)
      - cuDNN algorithm selection (~3-8s per unique tensor shape)
      - PyTorch memory allocator warm-up (~1-3s)
    Total: 6-15s of hidden latency on the first request.
    """
    if pipe is None or not torch.cuda.is_available():
        logger.info("cuda_warmup_skipped pipe=%s cuda=%s", pipe is not None, torch.cuda.is_available())
        return

    warmup_start = time.perf_counter()
    logger.info("cuda_warmup_pass_begin (1-step dummy inference)")

    try:
        # Create minimal dummy inputs at TARGET_SIZE
        dummy_person = Image.new("RGB", TARGET_SIZE, (128, 128, 128))
        dummy_garment = Image.new("RGB", TARGET_SIZE, (200, 200, 200))
        dummy_mask = Image.new("L", TARGET_SIZE, 255)
        dummy_pose = Image.new("RGB", TARGET_SIZE, (64, 64, 64))

        pose_tensor = tensor_transform(dummy_pose).unsqueeze(0).to(DEVICE, TORCH_DTYPE)
        garm_tensor = tensor_transform(dummy_garment).unsqueeze(0).to(DEVICE, TORCH_DTYPE)

        with torch.inference_mode():
            with torch.cuda.amp.autocast(dtype=TORCH_DTYPE):
                # Encode a minimal prompt
                prompt_embeds, neg_embeds, pooled_embeds, neg_pooled = pipe.encode_prompt(
                    "warmup garment",
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt="low quality",
                )
                prompt_embeds_c, _, _, _ = pipe.encode_prompt(
                    "warmup garment",
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt="low quality",
                )

                # Single-step forward pass to trigger all CUDA allocations
                _ = pipe(
                    prompt_embeds=prompt_embeds.to(DEVICE, TORCH_DTYPE),
                    negative_prompt_embeds=neg_embeds.to(DEVICE, TORCH_DTYPE),
                    pooled_prompt_embeds=pooled_embeds.to(DEVICE, TORCH_DTYPE),
                    negative_pooled_prompt_embeds=neg_pooled.to(DEVICE, TORCH_DTYPE),
                    num_inference_steps=1,
                    generator=torch.Generator(DEVICE).manual_seed(0),
                    strength=1.0,
                    pose_img=pose_tensor,
                    text_embeds_cloth=prompt_embeds_c.to(DEVICE, TORCH_DTYPE),
                    cloth=garm_tensor,
                    mask_image=dummy_mask,
                    image=dummy_person,
                    height=TARGET_H,
                    width=TARGET_W,
                    ip_adapter_image=dummy_garment,
                    guidance_scale=2.5,
                )

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        warmup_ms = (time.perf_counter() - warmup_start) * 1000
        logger.info("cuda_warmup_pass_complete elapsed_ms=%.0f", warmup_ms)

    except Exception as exc:
        warmup_ms = (time.perf_counter() - warmup_start) * 1000
        logger.warning("cuda_warmup_pass_failed elapsed_ms=%.0f error=%s", warmup_ms, exc)
        # Non-fatal: first real request will just be slightly slower
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def warmup():
    global _REUSE_COUNT
    if _WARM.is_set():
        return

    logger.info("=" * 60)
    logger.info("COLD START BEGIN")
    logger.info("=" * 60)

    load_models()

    # ── CUDA Warm-Up: 1-step dummy forward pass ────────────────────────
    # Pre-allocates CUDA contexts, cuDNN autotuner caches, and memory pools
    # so the user's first request doesn't pay a 6-15s hidden latency tax.
    _cuda_warmup_pass()

    cloudinary_ok = _configure_cloudinary()

    startup_total_ms = (time.perf_counter() - _STARTUP_TIME) * 1000
    logger.info("=" * 60)
    logger.info("COLD START COMPLETE")
    logger.info("  startup_total_ms=%.0f", startup_total_ms)
    logger.info("  cloudinary_configured=%s", cloudinary_ok)
    logger.info("=" * 60)

    _WARM.set()
    _REUSE_COUNT = 0


# =============================================================================
# Inference
# =============================================================================

def _maybe_autocast():
    if torch.cuda.is_available():
        return torch.cuda.amp.autocast(dtype=TORCH_DTYPE)
    class _NullCtx:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc, tb): return False
    return _NullCtx()


def _refine_target_inpaint_mask(mask: Image.Image, cloth_type: str) -> Image.Image:
    """
    Expand the target person's mask slightly without using the garment image as
    geometry. The person stays the spatial authority; this only gives the model
    enough boundary room for waistbands, hems, and drape.
    """
    import cv2

    gt = (cloth_type or "upper_body").strip().lower().replace(" ", "_")
    mask_np = np.array(mask.convert("L"), dtype=np.uint8)
    mask_np = (mask_np > 127).astype(np.uint8) * 255

    if gt == "lower_body":
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 17))
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, close_kernel)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 15))
        mask_np = cv2.dilate(mask_np, dilate_kernel, iterations=1)

        rows = np.where(mask_np.any(axis=1))[0]
        if len(rows) > 0:
            top = int(rows[0])
            waist_top = max(0, top - 56)
            band = mask_np[top:min(mask_np.shape[0], top + 24), :]
            cols = np.where(np.sum(band > 127, axis=0) > 0)[0]
            if len(cols) > 0:
                x1 = max(0, int(cols[0]) - 20)
                x2 = min(mask_np.shape[1], int(cols[-1]) + 20)
                mask_np[waist_top:top, x1:x2] = 255
            hard_protect_top = max(0, waist_top - 32)
            mask_np[:hard_protect_top, :] = 0

    elif gt in ("dresses", "full_body"):
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 21))
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, close_kernel)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 17))
        mask_np = cv2.dilate(mask_np, dilate_kernel, iterations=1)

    else:
        # Upper body: more aggressive dilation to cover layered outfits
        # (e.g. kurti hem peeking below a jacket, shirt tail below sweater)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 15))
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, close_kernel)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 13))
        mask_np = cv2.dilate(mask_np, dilate_kernel, iterations=1)

    return Image.fromarray(mask_np, mode="L")


def _feather_mask_top(mask: Image.Image, feather: int = 30) -> Image.Image:
    """
    Soften the TOP edge of a lower-body inpaint mask so the new garment blends
    into the existing torso/shirt instead of looking pasted.

    The painted region's extent — and therefore body geometry, pose, leg
    thickness and hip width — is UNCHANGED. Only the boundary blend is
    softened: a soft (0..1) mask tells the inpaint pipeline to blend original
    + generated across the waistband, which reads as a natural transition
    rather than a hard pasted line.
    """
    mask_np = np.array(mask.convert("L"), dtype=np.uint8)
    rows = np.where(mask_np.any(axis=1))[0]
    if len(rows) == 0 or feather <= 0:
        return mask
    top = int(rows[0])
    for dy in range(feather):
        yy = top + dy
        if yy >= mask_np.shape[0]:
            break
        a = dy / float(feather)  # 0 at the very top -> 1 after `feather` rows
        mask_np[yy] = (mask_np[yy].astype(np.float32) * a).astype(np.uint8)
    return Image.fromarray(mask_np, mode="L")


# =============================================================================
# Subtype-aware prompt attributes (lower-body only)
# =============================================================================

_GARMENT_PROMPT_ATTRS: dict[str, dict[str, str]] = {
    "jeans": {
        "coverage": "lower body garment",
        "fit": "regular fit",
        "silhouette": "straight leg",
        "sleeves": "n/a",
        "neckline": "n/a",
        "collar": "n/a",
        "waist_position": "natural waist or hip",
        "garment_length": "full length to ankle",
        "layering": "single layer",
        "structure": "denim two legs button fly",
        "drape": "minimal drape",
        "material": "denim cotton twill",
        "fabric_behavior": "stiff structured",
    },
    "trousers": {
        "coverage": "lower body garment",
        "fit": "regular fit",
        "silhouette": "straight leg",
        "sleeves": "n/a",
        "neckline": "n/a",
        "collar": "n/a",
        "waist_position": "natural waist",
        "garment_length": "full length to ankle",
        "layering": "single layer",
        "structure": "two legs zip fly",
        "drape": "moderate drape",
        "material": "woven cotton polyester",
        "fabric_behavior": "smooth structured",
    },
    "pants": {
        "coverage": "lower body garment",
        "fit": "regular fit",
        "silhouette": "straight leg",
        "sleeves": "n/a",
        "neckline": "n/a",
        "collar": "n/a",
        "waist_position": "natural waist or hip",
        "garment_length": "full length to ankle",
        "layering": "single layer",
        "structure": "two legs",
        "drape": "moderate drape",
        "material": "woven cotton",
        "fabric_behavior": "smooth structured",
    },
    "shorts": {
        "coverage": "lower body garment",
        "fit": "regular fit",
        "silhouette": "straight leg",
        "sleeves": "n/a",
        "neckline": "n/a",
        "collar": "n/a",
        "waist_position": "natural waist or hip",
        "garment_length": "above knee",
        "layering": "single layer",
        "structure": "two legs",
        "drape": "minimal drape",
        "material": "cotton twill",
        "fabric_behavior": "casual structured",
    },
    "skirt": {
        "coverage": "lower body garment",
        "fit": "regular fit",
        "silhouette": "A-line",
        "sleeves": "n/a",
        "neckline": "n/a",
        "collar": "n/a",
        "waist_position": "natural waist",
        "garment_length": "varies",
        "layering": "single layer",
        "structure": "no leg separation",
        "drape": "flowing drape",
        "material": "woven cotton",
        "fabric_behavior": "soft flowing",
    },
    "joggers": {
        "coverage": "lower body garment",
        "fit": "relaxed fit",
        "silhouette": "tapered leg",
        "sleeves": "n/a",
        "neckline": "n/a",
        "collar": "n/a",
        "waist_position": "elastic waist",
        "garment_length": "full length to ankle",
        "layering": "single layer",
        "structure": "two legs elastic cuff",
        "drape": "soft drape",
        "material": "fleece cotton jersey",
        "fabric_behavior": "soft relaxed",
    },
    "leggings": {
        "coverage": "lower body garment",
        "fit": "tight fitted",
        "silhouette": "body-hugging",
        "sleeves": "n/a",
        "neckline": "n/a",
        "collar": "n/a",
        "waist_position": "high waist",
        "garment_length": "full length to ankle",
        "layering": "single layer",
        "structure": "two legs no fly",
        "drape": "no drape skin-tight",
        "material": "stretch jersey",
        "fabric_behavior": "stretch conforming",
    },
    "cargo_pants": {
        "coverage": "lower body garment",
        "fit": "relaxed fit",
        "silhouette": "straight leg",
        "sleeves": "n/a",
        "neckline": "n/a",
        "collar": "n/a",
        "waist_position": "natural waist or hip",
        "garment_length": "full length to ankle",
        "layering": "single layer",
        "structure": "pocketed utility",
        "drape": "moderate drape",
        "material": "cotton twill",
        "fabric_behavior": "rugged structured",
    },
    "chinos": {
        "coverage": "lower body garment",
        "fit": "regular fit",
        "silhouette": "straight leg",
        "sleeves": "n/a",
        "neckline": "n/a",
        "collar": "n/a",
        "waist_position": "natural waist",
        "garment_length": "full length to ankle",
        "layering": "single layer",
        "structure": "two legs zip fly",
        "drape": "moderate drape",
        "material": "cotton chino",
        "fabric_behavior": "smooth structured",
    },
    "wide_leg": {
        "coverage": "lower body garment",
        "fit": "relaxed fit",
        "silhouette": "wide leg",
        "sleeves": "n/a",
        "neckline": "n/a",
        "collar": "n/a",
        "waist_position": "natural waist",
        "garment_length": "full length to ankle",
        "layering": "single layer",
        "structure": "two legs wide",
        "drape": "flowing drape",
        "material": "woven cotton",
        "fabric_behavior": "flowing structured",
    },
    "palazzo": {
        "coverage": "lower body garment",
        "fit": "relaxed fit",
        "silhouette": "extremely wide leg",
        "sleeves": "n/a",
        "neckline": "n/a",
        "collar": "n/a",
        "waist_position": "natural waist",
        "garment_length": "full length to ankle",
        "layering": "single layer",
        "structure": "two legs very wide",
        "drape": "heavy flowing drape",
        "material": "flowing woven",
        "fabric_behavior": "flowing soft",
    },
    "bermuda": {
        "coverage": "lower body garment",
        "fit": "regular fit",
        "silhouette": "straight leg",
        "sleeves": "n/a",
        "neckline": "n/a",
        "collar": "n/a",
        "waist_position": "natural waist or hip",
        "garment_length": "above knee to knee",
        "layering": "single layer",
        "structure": "two legs",
        "drape": "minimal drape",
        "material": "cotton twill",
        "fabric_behavior": "casual structured",
    },
    # ── Dress / full-body garments ──────────────────────────────────
    "dress": {
        "coverage": "full body garment",
        "fit": "regular fit",
        "silhouette": "follows body shape with natural skirt drape",
        "sleeves": "varies",
        "neckline": "varies",
        "collar": "n/a",
        "waist_position": "natural waist",
        "garment_length": "knee length or longer",
        "layering": "single layer",
        "structure": "one-piece bodice and skirt following body contours",
        "drape": "moderate drape following leg positions",
        "material": "woven cotton polyester",
        "fabric_behavior": "structured flowing",
    },
    "gown": {
        "coverage": "full body garment",
        "fit": "regular fit",
        "silhouette": "floor length elegant following body contours",
        "sleeves": "varies",
        "neckline": "varies",
        "collar": "n/a",
        "waist_position": "natural waist",
        "garment_length": "floor length",
        "layering": "single layer",
        "structure": "one-piece full length following leg positions",
        "drape": "heavy flowing drape following body structure",
        "material": "silk satin chiffon",
        "fabric_behavior": "flowing elegant",
    },
    "jumpsuit": {
        "coverage": "full body garment",
        "fit": "regular fit",
        "silhouette": "continuous torso to legs",
        "sleeves": "varies",
        "neckline": "varies",
        "collar": "n/a",
        "waist_position": "natural waist",
        "garment_length": "full length to ankle",
        "layering": "single layer",
        "structure": "one-piece with leg separation",
        "drape": "moderate drape",
        "material": "woven cotton polyester",
        "fabric_behavior": "structured fitted",
    },
    "kurti": {
        "coverage": "full body garment",
        "fit": "regular fit",
        "silhouette": "tunic over pants",
        "sleeves": "varies",
        "neckline": "round or v-neck",
        "collar": "n/a",
        "waist_position": "natural waist",
        "garment_length": "hip to knee length",
        "layering": "layered with bottoms",
        "structure": "tunic top with separate bottoms",
        "drape": "moderate drape",
        "material": "cotton silk",
        "fabric_behavior": "soft flowing",
    },
    "kurta_set": {
        "coverage": "full body outfit",
        "fit": "regular fit",
        "silhouette": "long kurta over coordinated bottoms",
        "sleeves": "varies",
        "neckline": "round, v-neck, or mandarin placket",
        "collar": "varies",
        "waist_position": "natural waist covered by tunic",
        "garment_length": "knee to calf length top with full bottoms",
        "layering": "layered tunic and pants",
        "structure": "separate top and bottom following body pose",
        "drape": "soft vertical drape",
        "material": "cotton silk rayon",
        "fabric_behavior": "soft structured ethnic wear",
    },
    "saree": {
        "coverage": "draped full body garment",
        "fit": "wrapped drape",
        "silhouette": "saree pleats with pallu draped over shoulder",
        "sleeves": "blouse sleeves vary",
        "neckline": "blouse neckline varies",
        "collar": "n/a",
        "waist_position": "natural waist with wrapped pleats",
        "garment_length": "floor length drape",
        "layering": "blouse, skirt, and pallu layers",
        "structure": "wrapped fabric, pleats, shoulder drape",
        "drape": "asymmetric flowing drape following body pose",
        "material": "silk chiffon georgette cotton",
        "fabric_behavior": "flowing folded draped fabric",
    },
    "lehenga": {
        "coverage": "draped full body outfit",
        "fit": "fitted blouse with voluminous skirt",
        "silhouette": "flared skirt with blouse and dupatta",
        "sleeves": "blouse sleeves vary",
        "neckline": "blouse neckline varies",
        "collar": "n/a",
        "waist_position": "natural waist",
        "garment_length": "floor length skirt",
        "layering": "blouse, skirt, dupatta",
        "structure": "separate blouse and flared skirt",
        "drape": "heavy skirt drape with optional dupatta",
        "material": "silk brocade chiffon net",
        "fabric_behavior": "structured embellished flowing",
    },
    "anarkali": {
        "coverage": "draped full body garment",
        "fit": "fitted bodice with flared skirt",
        "silhouette": "long flared anarkali dress",
        "sleeves": "varies",
        "neckline": "varies",
        "collar": "n/a",
        "waist_position": "high waist or natural waist",
        "garment_length": "calf to floor length",
        "layering": "single long tunic over bottoms",
        "structure": "fitted upper bodice and flared lower panels",
        "drape": "radial flowing drape",
        "material": "cotton silk georgette",
        "fabric_behavior": "flowing paneled fabric",
    },
    "abaya": {
        "coverage": "draped full body garment",
        "fit": "loose relaxed fit",
        "silhouette": "long robe-like drape",
        "sleeves": "long sleeves",
        "neckline": "modest neckline",
        "collar": "varies",
        "waist_position": "loose no defined waist",
        "garment_length": "ankle to floor length",
        "layering": "single outer layer",
        "structure": "robe panels following body posture",
        "drape": "loose vertical drape",
        "material": "crepe nida chiffon",
        "fabric_behavior": "soft modest flowing",
    },
    "kaftan": {
        "coverage": "draped full body garment",
        "fit": "loose relaxed fit",
        "silhouette": "wide flowing kaftan",
        "sleeves": "wide sleeves",
        "neckline": "varies",
        "collar": "n/a",
        "waist_position": "loose or belted",
        "garment_length": "knee to floor length",
        "layering": "single flowing layer",
        "structure": "wide body and sleeve panels",
        "drape": "generous flowing drape",
        "material": "cotton silk rayon",
        "fabric_behavior": "soft wide flowing",
    },
    "kimono": {
        "coverage": "draped full body garment",
        "fit": "wrapped relaxed fit",
        "silhouette": "straight robe with wide sleeves",
        "sleeves": "wide sleeves",
        "neckline": "cross-over front",
        "collar": "flat collar",
        "waist_position": "belted natural waist",
        "garment_length": "knee to ankle length",
        "layering": "wrapped outer layer",
        "structure": "cross-front robe panels",
        "drape": "straight controlled drape",
        "material": "silk satin cotton",
        "fabric_behavior": "smooth structured drape",
    },
    "thobe": {
        "coverage": "draped full body garment",
        "fit": "straight relaxed fit",
        "silhouette": "long straight robe",
        "sleeves": "long sleeves",
        "neckline": "collared or banded neckline",
        "collar": "band or shirt collar",
        "waist_position": "straight no defined waist",
        "garment_length": "ankle length",
        "layering": "single robe layer",
        "structure": "long straight panels",
        "drape": "clean vertical drape",
        "material": "cotton polyester",
        "fabric_behavior": "crisp modest drape",
    },
    "sherwani": {
        "coverage": "draped full body outfit",
        "fit": "structured tailored fit",
        "silhouette": "long structured coat over bottoms",
        "sleeves": "long sleeves",
        "neckline": "mandarin collar",
        "collar": "mandarin collar",
        "waist_position": "natural waist or straight cut",
        "garment_length": "knee length coat",
        "layering": "coat over trousers",
        "structure": "tailored long jacket with front closure",
        "drape": "structured minimal drape",
        "material": "brocade silk jacquard",
        "fabric_behavior": "structured embellished formal",
    },
    "coord": {
        "coverage": "full body outfit",
        "fit": "regular fit",
        "silhouette": "matching top and bottom",
        "sleeves": "varies",
        "neckline": "varies",
        "collar": "varies",
        "waist_position": "natural waist",
        "garment_length": "varies by set",
        "layering": "coordinated set",
        "structure": "matching top and bottom pieces",
        "drape": "moderate drape",
        "material": "matching fabric set",
        "fabric_behavior": "coordinated structured",
    },
    "overall": {
        "coverage": "full body garment",
        "fit": "relaxed fit",
        "silhouette": "one-piece bib front",
        "sleeves": "sleeveless or with shirt",
        "neckline": "bib front",
        "collar": "n/a",
        "waist_position": "natural waist",
        "garment_length": "full length to ankle",
        "layering": "over shirt",
        "structure": "bib and brace with legs",
        "drape": "minimal drape",
        "material": "denim cotton twill",
        "fabric_behavior": "rugged structured",
    },
}

# ── Upper-body prompt attributes (P3) ─────────────────────────────────
# These ensure shirts, hoodies, jackets etc. get correct hem-length
# attributes so the diffusion model doesn't generate kurta-length garments.
_UPPER_GARMENT_PROMPT_ATTRS: dict[str, dict[str, str]] = {
    "shirt": {
        "coverage": "upper body garment",
        "fit": "regular fit",
        "silhouette": "straight body",
        "sleeves": "long sleeves or short sleeves",
        "neckline": "collar neckline",
        "collar": "pointed collar or spread collar",
        "waist_position": "natural waist",
        "garment_length": "ends at belt line or just below waist",
        "layering": "single layer",
        "structure": "button front placket with collar and cuffs",
        "drape": "moderate structured drape",
        "material": "woven cotton polyester",
        "fabric_behavior": "crisp structured woven",
    },
    "tshirt": {
        "coverage": "upper body garment",
        "fit": "regular fit",
        "silhouette": "straight body",
        "sleeves": "short sleeves",
        "neckline": "crew neck or v-neck",
        "collar": "ribbed crew collar",
        "waist_position": "natural waist",
        "garment_length": "ends at belt line or hip",
        "layering": "single layer",
        "structure": "pullover with crew or v-neck",
        "drape": "soft casual drape",
        "material": "cotton jersey",
        "fabric_behavior": "soft stretchy knit",
    },
    "hoodie": {
        "coverage": "upper body garment",
        "fit": "relaxed fit",
        "silhouette": "straight oversized body",
        "sleeves": "long sleeves with ribbed cuffs",
        "neckline": "hood",
        "collar": "hood with drawstring",
        "waist_position": "hip length",
        "garment_length": "ends at hip or just below",
        "layering": "single layer or over tshirt",
        "structure": "pullover or zip front with kangaroo pocket and hood",
        "drape": "casual relaxed drape",
        "material": "fleece cotton blend",
        "fabric_behavior": "soft thick casual",
    },
    "jacket": {
        "coverage": "upper body outerwear",
        "fit": "regular fit",
        "silhouette": "structured body",
        "sleeves": "long sleeves",
        "neckline": "collared or zip neck",
        "collar": "lapel collar or stand collar",
        "waist_position": "natural waist to hip",
        "garment_length": "ends at waist or hip",
        "layering": "outer layer over shirt or tshirt",
        "structure": "front opening with zip or buttons",
        "drape": "structured minimal drape",
        "material": "polyester nylon cotton",
        "fabric_behavior": "structured outerwear",
    },
    "blazer": {
        "coverage": "upper body outerwear",
        "fit": "tailored fit",
        "silhouette": "structured tailored body",
        "sleeves": "long sleeves",
        "neckline": "notch lapel",
        "collar": "notch lapel or peak lapel",
        "waist_position": "natural waist",
        "garment_length": "ends at hip",
        "layering": "outer layer over shirt",
        "structure": "single or double breasted with lapels",
        "drape": "structured formal drape",
        "material": "wool polyester blend",
        "fabric_behavior": "crisp structured formal",
    },
    "cardigan": {
        "coverage": "upper body layer",
        "fit": "regular relaxed fit",
        "silhouette": "straight open front body",
        "sleeves": "long sleeves",
        "neckline": "v-neck or round neck",
        "collar": "n/a",
        "waist_position": "hip length",
        "garment_length": "ends at hip or below",
        "layering": "open front layer over inner top",
        "structure": "open front with buttons or draped",
        "drape": "soft relaxed drape",
        "material": "knit wool cotton",
        "fabric_behavior": "soft knit structured",
    },
    "sweater": {
        "coverage": "upper body garment",
        "fit": "regular fit",
        "silhouette": "straight body",
        "sleeves": "long sleeves with ribbed cuffs",
        "neckline": "crew neck or turtleneck",
        "collar": "ribbed crew or turtleneck",
        "waist_position": "natural waist to hip",
        "garment_length": "ends at waist or hip",
        "layering": "single layer or over shirt",
        "structure": "pullover knit with ribbed hem",
        "drape": "moderate structured drape",
        "material": "knit wool cotton blend",
        "fabric_behavior": "knit structured warm",
    },
    "polo": {
        "coverage": "upper body garment",
        "fit": "regular fit",
        "silhouette": "straight body",
        "sleeves": "short sleeves",
        "neckline": "polo collar",
        "collar": "polo collar with button placket",
        "waist_position": "natural waist",
        "garment_length": "ends at belt line",
        "layering": "single layer",
        "structure": "pique knit with collar and two button placket",
        "drape": "moderate casual drape",
        "material": "pique cotton",
        "fabric_behavior": "structured casual knit",
    },
    "coat": {
        "coverage": "upper body outerwear",
        "fit": "regular fit",
        "silhouette": "straight structured body",
        "sleeves": "long sleeves",
        "neckline": "lapel collar",
        "collar": "notch or peak lapel",
        "waist_position": "below hip",
        "garment_length": "ends at mid-thigh or knee",
        "layering": "outer layer",
        "structure": "front opening with buttons or zip",
        "drape": "heavy structured drape",
        "material": "wool polyester blend",
        "fabric_behavior": "heavy structured formal",
    },
}


_FABRIC_CUES: dict[str, str] = {
    "jeans": "denim texture with visible stitching, realistic wash pattern, natural creasing at knees and hips",
    "trousers": "woven fabric with pressed crease, smooth structured finish",
    "pants": "woven fabric with natural drape and fold lines",
    "shorts": "cotton twill with casual structured appearance",
    "skirt": "flowing fabric with natural hem movement",
    "joggers": "soft jersey fabric with gathered cuffs and elastic waist",
    "leggings": "stretch fabric conforming to leg shape",
    "cargo_pants": "rugged cotton twill with pocket flaps and utility stitching",
    "wide_leg": "flowing fabric with wide silhouette and natural drape",
    "chinos": "smooth cotton twill with clean finish",
    "palazzo": "flowing wide-leg fabric with dramatic drape",
    "bermuda": "casual cotton with straight hem above knee",
    "dress": "visible garment texture with natural skirt folds and correct hem length",
    "gown": "flowing full-length fabric with layered folds and realistic highlights",
    "jumpsuit": "continuous one-piece fabric with natural waist and leg creases",
    "kurti": "embroidered or woven tunic fabric with soft vertical folds",
    "kurta_set": "coordinated ethnic fabric with tunic folds and matching bottom drape",
    "saree": "saree pleats, pallu shoulder drape, woven border, flowing fabric folds",
    "lehenga": "flared skirt fabric with blouse detail, dupatta drape, visible embroidery or border",
    "anarkali": "flared paneled fabric with radial folds and ethnic detailing",
    "abaya": "loose robe fabric with clean vertical folds and modest flowing drape",
    "kaftan": "wide flowing fabric with soft folds and relaxed drape",
    "kimono": "wrapped robe fabric with smooth sleeves and cross-front fold",
    "thobe": "crisp robe fabric with long vertical folds and clean placket",
    "sherwani": "structured brocade or jacquard texture with front closure and formal embroidery",
    "tshirt": "soft cotton jersey with subtle ribbed crew collar and natural shoulder seams",
    "shirt": "woven cotton with collar, button placket, and natural fabric folds",
    "blouse": "lightweight fabric with soft draped fit, buttons or tie front, and natural folds",
    "polo": "pique cotton with collar and short sleeves, casual knit texture",
    "sweater": "knit wool or cotton with visible stitch texture and ribbed cuffs and hem",
    "hoodie": "soft fleece with hood and kangaroo pocket, casual drawstring waist",
    "jacket": "structured outerwear with front opening, lapels or zip, and lining",
    "coat": "long structured outerwear with front closure and natural length drape",
    "blazer": "structured tailored jacket with lapels and single or double button closure",
    "cardigan": "knit open-front sweater with buttons or draped front, worn over an inner top",
    "open_front": "front opens at the center, worn open or closed over an inner top, visible lapels",
    "kurta": "ethnic tunic with side slits, mandarin collar, and soft vertical folds",
    "long_kurta": "long ethnic tunic extending past the hips with side slits and soft vertical folds",
}


def _build_subtype_aware_prompt(garment_desc: str, garment_subtype: str = "") -> str:
    """
    Build a detailed prompt enriched with subtype-specific garment attributes.

    For lower-body garments, appends fabric, fit, silhouette, and structure
    details that guide the diffusion model toward realistic generation.

    Filters out "n/a" values (sleeves/neckline/collar for lower-body) to
    avoid wasting prompt capacity on irrelevant attributes.
    """
    key = (garment_subtype or "").strip().lower().replace(" ", "_").replace("-", "_")
    attrs = _GARMENT_PROMPT_ATTRS.get(key) or _UPPER_GARMENT_PROMPT_ATTRS.get(key)
    if not attrs:
        cue = _FABRIC_CUES.get(key, "")
        if cue:
            return "model wearing " + garment_desc + ", " + cue
        return "model is wearing " + garment_desc

    parts = ["model wearing " + garment_desc]
    for attr_key in (
        "coverage", "fit", "silhouette", "sleeves", "neckline", "collar",
        "waist_position", "garment_length", "layering", "structure", "drape",
        "material", "fabric_behavior",
    ):
        val = attrs.get(attr_key, "")
        if val and val.lower() != "n/a":
            parts.append(val)

    fabric_cue = _FABRIC_CUES.get(key, "")
    if fabric_cue:
        parts.append(fabric_cue)

    if key:
        parts.append("detailed fabric texture")
        parts.append("natural garment folds")

    return ", ".join(parts)


def _build_source_specific_negative(source_cloth_type: str = "", target_subtype: str = "") -> str:
    """
    Build a negative prompt that suppresses accessories, artifacts, and
    unrealistic rendering styles.

    Does NOT include task-specific terms (e.g., no "shirt, top" for lower-body)
    — those are handled by the positive prompt's preservation instructions.
    """
    return (
        "monochrome, lowres, bad anatomy, worst quality, low quality, "
        "deformed, distorted, disfigured, bad proportions, "
        "extra limbs, missing limbs, cloned head, body out of frame, "
        "poorly drawn face, mutation, mutated, extra fingers, "
        "ugly, blurry, watermark, signature, text, logo, "
        "smooth plastic, airbrushed, cg render, 3d render, "
        "flat lighting, "
        "flat, painted, plastic, cartoon, oversaturated, "
        "smudged cloth, 2d vector, no texture, wax skin, "
        "soft focus, gaussian blur, posterized, cel shaded, "
        "changed body shape, different body proportions, "
        "female body on male, male body on female, "
        "changed shoulder width, different chest size, altered waist, "
        "different torso, changed hips, modified body structure, "
        "bag, purse, handbag, clutch, tote, backpack, "
        "headphones, earphones, headset, "
        "necklace, chain, pendant, choker, "
        "watch, wristwatch, bracelet, "
        "sunglasses, eyewear, glasses, "
        "phone, smartphone, mobile, "
        "strap, belt, waist belt, "
        "accessory, accessories, "
        "extra object, held item, carrying"
    )


def isolate_cloth_item(
    garment_img: Image.Image,
    cloth_type: str = "upper_body",
    parsing_model: Any = None,
) -> Image.Image:
    """
    Isolate the target garment item on a clean white canvas.
    If the garment image contains a human model, skirts, bags, or other items,
    strip non-target elements using semantic parsing so GarmentNet receives strictly
    the target garment on a clean white canvas.
    """
    if garment_img is None:
        return garment_img

    # If the image has an alpha channel with actual transparency, composite on white
    if garment_img.mode in ("RGBA", "LA") or (garment_img.mode == "P" and "transparency" in garment_img.info):
        rgba = garment_img.convert("RGBA")
        alpha = np.array(rgba.split()[-1])
        if np.any(alpha < 250):
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            bg.alpha_composite(rgba)
            return bg.convert("RGB")

    # If a parsing model is available, check for and strip non-garment elements
    if parsing_model is not None:
        try:
            import cv2
            orig_size = garment_img.size
            g_parse_img = garment_img.convert("RGB").resize((384, 512), Image.BILINEAR)
            g_parse, _ = parsing_model(g_parse_img)
            g_parse_np = np.array(g_parse, dtype=np.uint8)
            g_parse_full = cv2.resize(g_parse_np, orig_size, interpolation=cv2.INTER_NEAREST)

            # Check if person elements (face, hair, legs, arms) exist
            has_person = np.any(np.isin(g_parse_full, [11, 2, 12, 13, 14, 15]))
            if has_person:
                c_norm = cloth_type.lower().replace("-", "_")
                if c_norm in ("upper_body", "upper", "top"):
                    target_labels = [4, 7]  # UpperClothes, Dress
                elif c_norm in ("lower_body", "lower", "bottom", "pants", "skirt"):
                    target_labels = [5, 6]  # Skirt, Pants
                elif c_norm in ("dresses", "dress", "full_body"):
                    target_labels = [4, 5, 6, 7]
                else:
                    target_labels = [4, 7]

                cloth_mask = np.isin(g_parse_full, target_labels).astype(np.uint8) * 255
                if np.sum(cloth_mask > 127) > 500:
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_CLOSE, k)
                    cloth_mask = cv2.dilate(cloth_mask, k, iterations=1)

                    g_np = np.array(garment_img.convert("RGB"))
                    isolated = np.full_like(g_np, 255)
                    isolated[cloth_mask > 127] = g_np[cloth_mask > 127]

                    ys, xs = np.where(cloth_mask > 127)
                    if len(ys) > 0 and len(xs) > 0:
                        ymin, ymax = int(np.min(ys)), int(np.max(ys))
                        xmin, xmax = int(np.min(xs)), int(np.max(xs))
                        pad = 10
                        ymin, ymax = max(0, ymin - pad), min(g_np.shape[0], ymax + pad)
                        xmin, xmax = max(0, xmin - pad), min(g_np.shape[1], xmax + pad)
                        cropped_item = Image.fromarray(isolated[ymin:ymax, xmin:xmax])

                        cw, ch = 768, 1024
                        canvas = Image.new("RGB", (cw, ch), (255, 255, 255))
                        cropped_aspect = float(cropped_item.height) / max(1.0, float(cropped_item.width))
                        is_tall_item = cropped_aspect > 1.20 or any(
                            kw in (cloth_type or "").lower() for kw in ["dress", "kurta", "kurti", "tunic", "long"]
                        )
                        if c_norm in ("upper_body", "upper", "top"):
                            if is_tall_item:
                                target_w, target_h = int(cw * 0.78), int(ch * 0.80)
                                scaled_item = cropped_item.copy()
                                scaled_item.thumbnail((target_w, target_h), Image.LANCZOS)
                                paste_x = (cw - scaled_item.width) // 2
                                paste_y = int(ch * 0.08)
                            else:
                                target_w, target_h = int(cw * 0.75), int(ch * 0.58)
                                scaled_item = cropped_item.copy()
                                scaled_item.thumbnail((target_w, target_h), Image.LANCZOS)
                                paste_x = (cw - scaled_item.width) // 2
                                paste_y = int(ch * 0.12)
                        else:
                            target_w, target_h = int(cw * 0.80), int(ch * 0.80)
                            scaled_item = cropped_item.copy()
                            scaled_item.thumbnail((target_w, target_h), Image.LANCZOS)
                            paste_x = (cw - scaled_item.width) // 2
                            paste_y = (ch - scaled_item.height) // 2

                        canvas.paste(scaled_item, (paste_x, paste_y))
                        logger.info("garment_isolated_from_model cloth_type=%s bbox=(%d,%d,%d,%d) is_tall=%s", cloth_type, xmin, ymin, xmax, ymax, is_tall_item)
                        return canvas
        except Exception as exc:
            logger.warning("isolate_cloth_item_failed error=%s", exc)

    return garment_img


def _restore_person_identity(
    result: Image.Image,
    original: Image.Image,
    cloth_type: str,
    crop_top: int = 0,
    parsing_map: np.ndarray | None = None,
    inpaint_mask: Image.Image | None = None,
    has_collar: bool = False,
) -> Image.Image:
    """
    Restore the person's identity AND body structure from the original onto
    the diffusion result using SCHP parsing labels.

    The IDM-VTON model with strength=1.0 denoises the ENTIRE image,
    regenerating the face and body even though only clothing should change.
    This function restores identity by blending original pixels back for:
      - Face (11), Hair (2), upper Neck (18): Always restored.
      - Bare skin — Arms (14, 15), Legs (12, 13): Restored ONLY for pixels
        that fall OUTSIDE the inpaint mask, to prevent TryonNet from
        replacing the user's body shape with the garment model's body.

    Using parsing labels instead of a rectangular box means the restore
    mask follows actual face/hair/skin contours, so garment collars,
    necklines, and sleeve edges are never overwritten by original pixels.

    Falls back to Haar cascade only if no parsing_map is provided.
    """
    import cv2

    orig_np = np.array(original.convert("RGB"), dtype=np.float32)
    result_np = np.array(result.convert("RGB"), dtype=np.float32)
    h, w = orig_np.shape[:2]

    if orig_np.shape != result_np.shape:
        result_np = np.array(
            result.convert("RGB").resize(original.size, Image.LANCZOS),
            dtype=np.float32,
        )

    # ── SCHP parsing-based restoration (preferred) ─────────────────
    if parsing_map is not None:
        p_h, p_w = parsing_map.shape[:2]
        if (p_w, p_h) == (w, h):
            parse_resized = parsing_map
        else:
            parse_resized = cv2.resize(
                np.array(parsing_map, dtype=np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )

        # Build identity mask from parsing labels
        # Always include: face (11), hair (2)
        identity_labels = {2, 11}  # _LABEL_HAIR, _LABEL_FACE

        # For non-lower-body (upper_body / dresses): include neck (18)
        # Protect ONLY the uppermost 15% of neck under the chin when collar is present
        # (mandarin collar, polo, shirt collar, turtleneck, high neckline), and top 30%
        # for open/scoop necks (replacing the static 60% neck cutoff). This allows the new
        # garment's collar area to breathe and prevents mandarin collars and necklines from
        # melting with original exposed throat skin during Laplacian blending.
        identity_mask = np.isin(parse_resized, list(identity_labels)).astype(np.uint8) * 255

        if cloth_type != "lower_body":
            neck_mask = (parse_resized == 18).astype(np.uint8) * 255
            neck_rows = np.where(neck_mask.any(axis=1))[0]
            if len(neck_rows) > 0:
                neck_top = int(neck_rows[0])
                neck_bottom = int(neck_rows[-1])
                neck_height = neck_bottom - neck_top
                neck_ratio = 0.15 if has_collar else 0.30
                cutoff_y = neck_top + int(neck_height * neck_ratio)
                neck_mask[cutoff_y:, :] = 0
            identity_mask = np.maximum(identity_mask, neck_mask)

        # ── Body structure preservation: restore bare skin pixels ──────
        # Composite original skin (arms=14,15 legs=12,13) back, but ONLY
        # for pixels that fall OUTSIDE the inpaint mask. This prevents
        # TryonNet cross-attention from replacing the user's body shape
        # with the garment reference model's anatomy, while still allowing
        # clothing-region pixels inside the mask to come from diffusion.
        skin_labels = {12, 13, 14, 15}  # LeftLeg, RightLeg, LeftArm, RightArm
        skin_mask = np.isin(parse_resized, list(skin_labels)).astype(np.uint8) * 255

        inpaint_np = None
        if inpaint_mask is not None:
            inpaint_np = np.array(
                inpaint_mask.convert("L").resize((w, h), Image.BILINEAR),
                dtype=np.uint8,
            )
            # Only preserve skin pixels that are OUTSIDE the inpaint mask.
            # Skin inside the mask is where sleeves/garment should be generated.
            skin_mask[inpaint_np > 50] = 0
            # Identity regions (neck/chest/skin) inside inpaint mask must NOT be restored,
            # allowing new collars, high necklines, and shirts to generate naturally.
            identity_mask[inpaint_np > 50] = 0

        identity_mask = np.maximum(identity_mask, skin_mask)

        # Morphological closing to fill tiny gaps in parsed regions
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        identity_mask = cv2.morphologyEx(identity_mask, cv2.MORPH_CLOSE, close_k)

        # Gentle dilation to provide a small safety margin around identity
        dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        identity_mask = cv2.dilate(identity_mask, dilate_k, iterations=1)

        # Re-enforce zeroing inside inpaint mask AFTER dilation so dilated face/neck
        # boundaries NEVER bleed downward into newly generated collars, shoulders, or lapels.
        if inpaint_np is not None:
            identity_mask[inpaint_np > 50] = 0

        # Multi-scale Laplacian pyramid blending (<50ms) preserving pores, hair, and edges
        from postprocess import laplacian_pyramid_blend
        restored = laplacian_pyramid_blend(orig_np, result_np, identity_mask, num_levels=3)

        identity_pixel_count = int(np.sum(identity_mask > 127))
        skin_pixel_count = int(np.sum(skin_mask > 127))
        logger.info(
            "laplacian_identity_restored cloth_type=%s identity_labels=%s skin_labels=%s "
            "identity_pixels=%d skin_pixels=%d",
            cloth_type, sorted(identity_labels), sorted(skin_labels),
            identity_pixel_count, skin_pixel_count,
        )

        return Image.fromarray(restored, mode="RGB")

    # ── Fallback: Haar cascade (only when parsing_map is unavailable) ──
    orig_uint8 = np.array(original.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(orig_uint8, cv2.COLOR_RGB2GRAY)
    cascade_path = os.path.join(
        cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
    )
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        return result

    min_dim = max(30, int(min(h, w) * 0.04))
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_dim, min_dim)
    )
    if len(faces) == 0:
        return result

    (fx, fy, fw, fh) = max(faces, key=lambda r: r[2] * r[3])

    # Conservative face-only padding — smaller than before to avoid
    # collar clipping. Only restores face + hair, not neck/chest.
    pad_x = int(fw * 0.20)
    pad_y_top = int(fh * 0.50)
    pad_y_bottom = int(fh * 0.20)  # Reduced from 0.30/0.60 to avoid collars

    face_x1 = max(0, fx - pad_x)
    face_y1 = max(0, fy - pad_y_top)
    face_x2 = min(w, fx + fw + pad_x)
    face_y2 = min(h, fy + fh + pad_y_bottom)

    mask_region = np.zeros((h, w), dtype=np.uint8)
    mask_region[face_y1:face_y2, face_x1:face_x2] = 255

    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_eroded = cv2.erode(mask_region, erode_k, iterations=2)
    blur_size = max(15, min(fw, fh) // 8)
    if blur_size % 2 == 0:
        blur_size += 1
    mask_soft = cv2.GaussianBlur(mask_eroded.astype(np.float32), (blur_size, blur_size), 0)
    mask_3d = mask_soft[:, :, np.newaxis] / 255.0

    restored = (result_np * (1.0 - mask_3d) + orig_np * mask_3d).astype(np.uint8)

    logger.info(
        "haar_fallback_identity_restored face_box=(%d,%d,%d,%d) region=(%d,%d,%d,%d) blur=%d",
        fx, fy, fw, fh, face_x1, face_y1, face_x2, face_y2, blur_size,
    )

    return Image.fromarray(restored, mode="RGB")


def analyze_garment_attributes(
    garment_img: Image.Image,
    garment_desc: str = "",
    garment_subtype: str = "",
    cloth_type: str = "upper_body",
) -> dict[str, Any]:
    """
    Analyze geometry, aspect ratio, vertical coverage, sleeve width, and collar
    structure of the target garment. Combines vision contour analysis with textual cues.
    Eliminates the "Crop Top Trap" and provides garment length/sleeve awareness.
    """
    import cv2
    desc_lower = f"{garment_desc} {garment_subtype}".lower().replace("-", " ")

    # 1. Textual analysis
    explicit_long = any(kw in desc_lower for kw in [
        "long kurta", "long kurti", "kurta set", "kurti set", "kurta", "kurti",
        "tunic", "long shirt", "dress", "anarkali", "sherwani", "kaftan", "robe",
        "longline", "trench", "maxi", "midi", "gown", "overcoat", "pathani"
    ])
    explicit_crop = any(kw in desc_lower for kw in [
        "crop top", "cropped top", "crop", "cropped", "choli", "tube top",
        "bralette", "bandeau", "corset", "bustier", "short top", "baby tee"
    ])
    explicit_regular = any(kw in desc_lower for kw in [
        "shirt", "t-shirt", "tshirt", "tee", "polo", "blouse", "sweater",
        "hoodie", "jacket", "blazer", "cardigan", "sweatshirt", "regular top", "top"
    ]) and not explicit_crop and not explicit_long

    explicit_sleeveless = any(kw in desc_lower for kw in [
        "sleeveless", "strapless", "tank top", "tank", "cami", "spaghetti strap",
        "spaghetti", "halter", "tube top", "off-shoulder", "off shoulder",
        "slip dress", "sleeveless dress"
    ])
    explicit_has_sleeves = any(kw in desc_lower for kw in [
        "short sleeve", "half sleeve", "long sleeve", "full sleeve", "sleeve",
        "sleeved", "t-shirt", "shirt", "polo", "hoodie", "sweater", "jacket",
        "blazer", "cardigan", "kurta", "kurti", "sweatshirt"
    ])
    explicit_long_sleeve = any(kw in desc_lower for kw in [
        "long sleeve", "full sleeve", "long-sleeve", "full-sleeve", "long-sleeved",
        "sweater", "sweatshirt", "hoodie", "cardigan", "blazer", "jacket", "coat"
    ])
    explicit_collar = any(kw in desc_lower for kw in [
        "collar", "collared", "mandarin", "nehru", "spread collar", "button down",
        "button-down", "polo", "turtleneck", "high neck", "hoodie", "lapel", "shirt", "kurta"
    ])

    # 2. Vision contour analysis on garment_img
    aspect_ratio = 1.0
    vertical_coverage = 0.55
    has_sleeves_cv = True
    has_long_sleeves_cv = False

    try:
        g_np = np.array(garment_img.convert("RGB"))
        gh, gw = g_np.shape[:2]

        # Background: luminance > 235 and low saturation
        max_c = np.max(g_np, axis=2).astype(np.int16)
        min_c = np.min(g_np, axis=2).astype(np.int16)
        sat = max_c - min_c
        mean_c = np.mean(g_np, axis=2)
        is_bg = (mean_c > 235) & (sat < 25)

        fg_mask = (~is_bg).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        ys, xs = np.where(fg_mask > 127)
        if len(ys) > 500 and len(xs) > 500:
            ymin, ymax = int(np.min(ys)), int(np.max(ys))
            xmin, xmax = int(np.min(xs)), int(np.max(xs))
            box_h = max(1, ymax - ymin)
            box_w = max(1, xmax - xmin)
            aspect_ratio = float(box_h) / float(box_w)
            vertical_coverage = float(box_h) / float(gh)

            # Analyze sleeve width at shoulder/armpit (top 20% - 45% of garment height)
            y_upper_start = ymin + int(box_h * 0.20)
            y_upper_end = ymin + int(box_h * 0.45)
            upper_band = fg_mask[y_upper_start:y_upper_end, xmin:xmax]
            upper_widths = np.sum(upper_band > 127, axis=1)
            max_upper_w = np.max(upper_widths) if len(upper_widths) > 0 else box_w

            # Sleeveless garments (tank tops, tube tops, camis) have narrow width at sleeve height
            if (max_upper_w / float(box_w) < 0.48) and not explicit_has_sleeves and not explicit_long_sleeve and not explicit_regular:
                has_sleeves_cv = False

            # Outer columns in the midsection (sleeves running down sides)
            y_mid_start = ymin + int(box_h * 0.35)
            y_mid_end = ymin + int(box_h * 0.75)
            left_outer = fg_mask[y_mid_start:y_mid_end, xmin:xmin + int(box_w * 0.22)]
            right_outer = fg_mask[y_mid_start:y_mid_end, xmax - int(box_w * 0.22):xmax]
            outer_density = (np.mean(left_outer > 127) + np.mean(right_outer > 127)) / 2.0
            if outer_density > 0.20:
                has_long_sleeves_cv = True
    except Exception as exc:
        logger.warning("analyze_garment_geometry_failed error=%s", exc)

    # 3. Combine vision + text: Fix the "Crop Top Trap"
    if explicit_crop:
        is_crop = True
        is_long = False
        is_regular = False
    elif explicit_long or cloth_type in ("dresses", "full_body"):
        is_crop = False
        is_long = True
        is_regular = False
    elif explicit_regular:
        # Regular top / shirt / t-shirt:
        # Never clamp into crop top! Only long if vertical coverage or aspect ratio is exceptionally high.
        is_crop = False
        is_long = (aspect_ratio >= 1.25) or (vertical_coverage >= 0.72)
        is_regular = not is_long
    else:
        # Vision-based classification avoiding the Crop Top Trap:
        # A garment is crop ONLY if aspect_ratio < 0.78 AND vertical_coverage < 0.42.
        # Boxy shirts/t-shirts (aspect_ratio ~0.85, vertical_coverage ~0.55) are regular tops!
        is_crop = (aspect_ratio < 0.78) and (vertical_coverage < 0.42)
        is_long = (aspect_ratio >= 1.18) or (vertical_coverage >= 0.65)
        is_regular = not is_crop and not is_long

    # 4. Sleeve classification: Only mark sleeveless if explicitly detected/stated
    if explicit_sleeveless:
        has_sleeves = False
        has_long_sleeves = False
    elif explicit_long_sleeve:
        has_sleeves = True
        has_long_sleeves = True
    elif explicit_has_sleeves or explicit_regular:
        has_sleeves = True
        has_long_sleeves = has_long_sleeves_cv
    else:
        has_sleeves = has_sleeves_cv
        has_long_sleeves = has_long_sleeves_cv

    has_collar = explicit_collar or (aspect_ratio > 1.1)

    return {
        "aspect_ratio": aspect_ratio,
        "vertical_coverage": vertical_coverage,
        "is_long_garment": is_long,
        "is_crop_top": is_crop,
        "is_regular_top": is_regular,
        "has_sleeves": has_sleeves,
        "has_long_sleeves": has_long_sleeves,
        "has_collar_or_high_neck": has_collar,
    }


def _extract_hip_waist_y(
    keypoints: Any,
    parse_768: np.ndarray,
    target_h: int = TARGET_H,
    target_w: int = TARGET_W,
) -> tuple[int | None, int | None]:
    """
    Extract hip_y and waist_y in target image coordinates using OpenPose keypoints
    and/or SCHP human parsing maps.
    Returns (hip_y, waist_y).
    """
    hip_y = None
    waist_y = None

    # 1. From SCHP parsing: top of pants/skirt (labels 5, 6)
    lower_clothing_labels = {5, 6}
    lower_rows = np.where(np.isin(parse_768, list(lower_clothing_labels)).any(axis=1))[0]
    if len(lower_rows) > 0:
        waist_y = int(lower_rows[0])

    # 2. From OpenPose keypoints
    if keypoints is not None:
        try:
            if isinstance(keypoints, dict):
                hips_found = []
                # Case A: Named keys
                for k in ("left_hip", "right_hip", "l_hip", "r_hip"):
                    pt = keypoints.get(k)
                    if pt is not None:
                        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                            py = float(pt[1])
                            if py <= 1.0:
                                py *= target_h
                            elif py <= 512:
                                py = py * (target_h / 512.0)
                            hips_found.append(py)
                        elif hasattr(pt, "y"):
                            py = float(pt.y)
                            if py <= 1.0:
                                py *= target_h
                            hips_found.append(py)

                # Case B: OpenPose candidate / subset
                if not hips_found and "candidate" in keypoints and "subset" in keypoints:
                    candidate = keypoints["candidate"]
                    subset = keypoints["subset"]
                    if len(subset) > 0 and len(candidate) > 0:
                        for part_idx in (8, 11):  # 8=RHip, 11=LHip
                            cand_idx = int(subset[0][part_idx])
                            if 0 <= cand_idx < len(candidate):
                                py = float(candidate[cand_idx][1])
                                if py <= 1.0:
                                    py *= target_h
                                elif py <= 512:
                                    py = py * (target_h / 512.0)
                                hips_found.append(py)

                # Case C: pose_keypoints_2d list
                if not hips_found and "pose_keypoints_2d" in keypoints:
                    pk = keypoints["pose_keypoints_2d"]
                    for part_idx in (8, 11):
                        offset = part_idx * 3
                        if len(pk) > offset + 2 and pk[offset + 2] > 0.05:
                            py = float(pk[offset + 1])
                            if py <= 1.0:
                                py *= target_h
                            elif py <= 512:
                                py = py * (target_h / 512.0)
                            hips_found.append(py)

                if hips_found:
                    hip_y = int(np.mean(hips_found))
        except Exception as exc:
            logger.warning("extract_hip_waist_y_failed error=%s", exc)

    return hip_y, waist_y


def _render_openpose_pose_img(
    human_bgr: np.ndarray,
    keypoints: Any,
    target_size: tuple[int, int] = TARGET_SIZE,
) -> Image.Image:
    """
    Render OpenPose skeleton overlay on grayscale person image for DensePose bypass.
    Saves ~3s GPU latency by skipping Detectron2 ResNet-50 DensePose inference on upper tops.
    """
    gray_img = cv2.cvtColor(human_bgr, cv2.COLOR_BGR2GRAY)
    canvas = np.tile(gray_img[:, :, np.newaxis], [1, 1, 3])
    h, w = canvas.shape[:2]

    # OpenPose standard limb pairs (COCO 18 keypoints)
    limb_seq = [
        (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
        (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
        (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)
    ]
    # Standard OpenPose limb colors (BGR)
    colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
        [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
        [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
        [255, 0, 255], [255, 0, 170]
    ]

    points: dict[int, tuple[int, int]] = {}
    if keypoints is not None and isinstance(keypoints, dict):
        if "candidate" in keypoints and "subset" in keypoints:
            candidate = keypoints["candidate"]
            subset = keypoints["subset"]
            if len(subset) > 0:
                sub = subset[0]
                for i in range(min(18, len(sub))):
                    cand_idx = int(sub[i])
                    if 0 <= cand_idx < len(candidate):
                        px, py = float(candidate[cand_idx][0]), float(candidate[cand_idx][1])
                        if px <= 1.0:
                            px *= w
                        if py <= 1.0:
                            py *= h
                        points[i] = (int(px), int(py))
        elif "pose_keypoints_2d" in keypoints:
            pk = keypoints["pose_keypoints_2d"]
            for i in range(min(18, len(pk) // 3)):
                conf = float(pk[i * 3 + 2])
                if conf > 0.05:
                    px, py = float(pk[i * 3]), float(pk[i * 3 + 1])
                    if px <= 1.0:
                        px *= w
                    if py <= 1.0:
                        py *= h
                    points[i] = (int(px), int(py))

    # Draw limbs
    for i, (p1, p2) in enumerate(limb_seq):
        if p1 in points and p2 in points:
            color = colors[i % len(colors)]
            cv2.line(canvas, points[p1], points[p2], color, thickness=3, lineType=cv2.LINE_AA)

    # Draw keypoint joints
    for i, (px, py) in points.items():
        color = colors[i % len(colors)]
        cv2.circle(canvas, (px, py), radius=4, color=color, thickness=-1, lineType=cv2.LINE_AA)

    pose_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return Image.fromarray(pose_rgb).resize(target_size, Image.BILINEAR)


def run_idm_vton_inference(
    person_img: Image.Image,
    garment_img: Image.Image,
    garment_desc: str,
    cloth_type: str,
    garment_subtype: str = "",
    steps: int = 14,
    seed: int = 42,
    auto_crop: bool = True,
    external_mask: Image.Image | None = None,
    protected_mask: Image.Image | None = None,
    mask_strategy: str = "external",
    mask_quality_score: float | None = None,
    guidance_scale: float | None = None,
) -> tuple[Image.Image, dict[str, object]]:
    global pipe, parsing_model, openpose_model
    global densepose_predictor, densepose_cfg, tensor_transform, get_mask_location_fn

    import cv2

    device = DEVICE

    if torch.cuda.is_available():
        openpose_model.preprocessor.body_estimation.model.to(device)
        pipe.to(device)
        pipe.unet_encoder.to(device)

    from mask_pipeline import (
        WorkerMaskStrategy,
        apply_protected_mask,
        fuse_hybrid_mask,
        select_worker_mask_strategy,
    )

    isolated_garment = isolate_cloth_item(garment_img, cloth_type=cloth_type, parsing_model=parsing_model)
    garm_img = isolated_garment.convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)
    garm_attrs = analyze_garment_attributes(
        garment_img=garm_img,
        garment_desc=garment_desc,
        garment_subtype=garment_subtype,
        cloth_type=cloth_type,
    )
    logger.info("garment_attributes_analyzed %s", garm_attrs)
    human_img_orig = person_img.convert("RGB")

    width, height = human_img_orig.size
    left, top, crop_size = 0.0, 0.0, None

    if auto_crop:
        # ── Aspect-ratio preserving crop ───────────────────────────────
        target_aspect = TARGET_W / TARGET_H  # 0.75
        img_aspect = width / height

        if img_aspect > target_aspect:
            target_height = height
            target_width = int(height * target_aspect)
        else:
            target_width = width
            target_height = int(width / target_aspect)

        is_full_body = cloth_type in ("dresses", "lower_body", "full_body")
        if is_full_body:
            left = (width - target_width) / 2
            bottom = height
            top = height - target_height
            right = (width + target_width) / 2
        else:
            left = (width - target_width) / 2
            # For upper_body, position crop from top of head down through upper thighs
            # so original long kurti/tunic hems are completely inside the crop
            # and fully replaced, avoiding crop boundary leakage during uncropping.
            top = max(0.0, (height - target_height) * 0.15)
            right = (width + target_width) / 2
            bottom = min(float(height), top + target_height)

        left = max(0.0, left)
        top = max(0.0, top)
        right = min(float(width), right)
        bottom = min(float(height), bottom)

        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize(TARGET_SIZE, Image.LANCZOS)
    else:
        # ── No auto_crop: aspect-ratio preserving resize with padding ──
        img_aspect = width / height
        target_aspect = TARGET_W / TARGET_H

        if img_aspect > target_aspect:
            new_w = TARGET_W
            new_h = int(TARGET_W / img_aspect)
        else:
            new_h = TARGET_H
            new_w = int(TARGET_H * img_aspect)

        resized = human_img_orig.resize((new_w, new_h), Image.LANCZOS)
        resized_np = np.array(resized, dtype=np.uint8)

        pad_top = (TARGET_H - new_h) // 2
        pad_bottom = TARGET_H - new_h - pad_top
        pad_left = (TARGET_W - new_w) // 2
        pad_right = TARGET_W - new_w - pad_left

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            padded = cv2.copyMakeBorder(
                resized_np, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_REFLECT_101,
            )
            human_img = Image.fromarray(padded)
        else:
            human_img = resized

    # ── Single-pass pre-resizing for OpenPose, SCHP, and DensePose ─────
    human_img_384 = human_img.resize((384, 512), Image.LANCZOS)

    # Always compute AutoMasker mask (SCHP + OpenPose) for routing / hybrid
    keypoints = openpose_model(human_img_384)
    model_parse, _ = parsing_model(human_img_384)

    # Pre-compute parse_768 once at TARGET_SIZE for mask ops and identity restore
    parse_768 = cv2.resize(
        np.array(model_parse, dtype=np.uint8),
        (TARGET_W, TARGET_H),
        interpolation=cv2.INTER_NEAREST,
    )

    automasker_mask, _ = get_mask_location_fn("hd", cloth_type, model_parse, keypoints)
    automasker_mask = automasker_mask.resize(TARGET_SIZE)

    min_quality = float(os.environ.get("MASK_MIN_QUALITY_SCORE", "62.0"))
    strategy = select_worker_mask_strategy(
        external_mask,
        mask_quality_score,
        min_quality=min_quality,
        cloth_type=cloth_type,
    )
    if mask_strategy == "automasker":
        strategy = WorkerMaskStrategy.AUTOMASKER
    elif mask_strategy == "hybrid":
        strategy = WorkerMaskStrategy.HYBRID

    mask_meta: dict[str, object] = {
        "mask_type_used": strategy.value,
        "mask_quality_score": mask_quality_score,
    }

    if strategy == WorkerMaskStrategy.EXTERNAL and external_mask is not None:
        mask = external_mask.convert("L").resize(TARGET_SIZE)
    elif strategy == WorkerMaskStrategy.HYBRID:
        mask = fuse_hybrid_mask(external_mask, automasker_mask, cloth_type)
        mask_meta["mask_type_used"] = "hybrid"
    else:
        mask = automasker_mask
        mask_meta["mask_type_used"] = "automasker"

    mask = _refine_target_inpaint_mask(mask, cloth_type)

    # ── Lower-body mask enhancement using SCHP parsing labels ──────────
    if ENABLE_GARMENT_SILHOUETTE_MASK and cloth_type == "lower_body":
        _lower_clothing_labels = {5, 6}  # _LABEL_SKIRT, _LABEL_PANTS
        _lower_region = np.isin(parse_768, list(_lower_clothing_labels)).astype(np.uint8) * 255

        # Morphological closing to fill gaps (belt loops, seams, zippers)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 17))
        _lower_region = cv2.morphologyEx(_lower_region, cv2.MORPH_CLOSE, close_kernel)

        # Generous dilation to expand beyond tight SCHP label boundaries
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 21))
        _lower_region = cv2.dilate(_lower_region, dilate_kernel, iterations=1)

        # Merge SCHP-derived region into mask
        mask_np = np.array(mask.convert("L"), dtype=np.uint8)
        mask_np = np.maximum(mask_np, _lower_region)

        # ── Contoured Waistband Extension (Task 5) ─────────────────────
        # Follow shirt hem curvature (label 4) to extend waistband mask with
        # a smooth natural curve rather than a flat rectangular box.
        _rows_with_mask = np.where(mask_np.any(axis=1))[0]
        if len(_rows_with_mask) > 0:
            _mask_top = int(_rows_with_mask[0])
            _extend_up = max(0, _mask_top - 60)
            _top_band = mask_np[_mask_top:min(_mask_top + 15, TARGET_H), :]
            _col_sums = np.sum(_top_band > 127, axis=0)
            _nonzero_cols = np.where(_col_sums > 0)[0]
            if len(_nonzero_cols) > 0:
                _left_bound = max(0, int(_nonzero_cols[0]) - 10)
                _right_bound = min(TARGET_W, int(_nonzero_cols[-1]) + 10)
                _band_width = _right_bound - _left_bound
                if _band_width > 0:
                    # Build parabolic waistband contour curve
                    x_idx = np.arange(_band_width)
                    norm_x = (x_idx - _band_width / 2.0) / (_band_width / 2.0)
                    curve = (1.0 - 0.25 * (norm_x ** 2))  # dip slightly at edges
                    for col_i, col_x in enumerate(range(_left_bound, _right_bound)):
                        curr_top = max(0, _mask_top - int(60 * curve[col_i]))
                        mask_np[curr_top:_mask_top, col_x] = 255

        # Hard upper-body exclusion: protect everything well above waistband
        rows_with_mask = np.where(mask_np.any(axis=1))[0]
        if len(rows_with_mask) > 0:
            mask_top = int(rows_with_mask[0])
            exclude_top = max(0, mask_top - 80)
            mask_np[:exclude_top, :] = 0

        mask = Image.fromarray(mask_np, mode="L")

    # ── Dress/full-body mask enhancement using SCHP parsing labels ────
    if ENABLE_GARMENT_SILHOUETTE_MASK and cloth_type in ("dresses", "full_body"):
        _dress_clothing_labels = {4, 5, 6, 7, 17}
        _dress_region = np.isin(parse_768, list(_dress_clothing_labels)).astype(np.uint8) * 255

        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 21))
        _dress_region = cv2.morphologyEx(_dress_region, cv2.MORPH_CLOSE, close_kernel)

        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 23))
        _dress_region = cv2.dilate(_dress_region, dilate_kernel, iterations=1)

        mask_np = np.array(mask.convert("L"), dtype=np.uint8)
        mask_np = np.maximum(mask_np, _dress_region)

        # ── Target-Aware Arm & Sleeve Masking for Dresses ──────────────
        _bare_arm_labels = {14, 15}  # LeftArm, RightArm
        _bare_arm_region = np.isin(parse_768, list(_bare_arm_labels)).astype(np.uint8)

        # Static Sleeve Lockout Fix: Only zero out arms if explicitly sleeveless/strapless.
        # If target has sleeves (half or full), allow the garment to inpaint over the arms.
        if not garm_attrs.get("has_sleeves", True):
            _arm_erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            _bare_arm_region = cv2.erode(_bare_arm_region, _arm_erode_k, iterations=1)
            mask_np[_bare_arm_region > 0] = 0
            logger.info("dress_arms_protected target_sleeveless=True")
        else:
            _arm_dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            _arm_mask_expanded = cv2.dilate(_bare_arm_region, _arm_dilate_k, iterations=1) * 255
            mask_np = np.maximum(mask_np, _arm_mask_expanded)
            logger.info("dress_arms_masked_for_sleeves has_long_sleeves=%s", garm_attrs.get("has_long_sleeves", False))

        _rows_with_mask = np.where(mask_np.any(axis=1))[0]
        if len(_rows_with_mask) > 0:
            _mask_bottom = int(_rows_with_mask[-1])
            _target_bottom = min(TARGET_H, int(TARGET_H * 0.92))
            if _mask_bottom < _target_bottom:
                for y in range(_mask_bottom, _target_bottom):
                    row_body = np.where(np.isin(parse_768[y, :], [4, 5, 6, 7, 8, 12, 13, 18]))[0]
                    if len(row_body) > 0:
                        x_left = max(0, int(row_body[0]) - 12)
                        x_right = min(TARGET_W, int(row_body[-1]) + 12)
                        mask_np[y, x_left:x_right] = 255
                    else:
                        _bottom_band = mask_np[max(0, _mask_bottom - 20):_mask_bottom, :]
                        _col_sums = np.sum(_bottom_band > 127, axis=0)
                        _nonzero_cols = np.where(_col_sums > 0)[0]
                        if len(_nonzero_cols) > 0:
                            x_left = max(0, int(_nonzero_cols[0]) - 10)
                            x_right = min(TARGET_W, int(_nonzero_cols[-1]) + 10)
                            mask_np[y, x_left:x_right] = 255

        mask = Image.fromarray(mask_np, mode="L")

    # ── Upper-body mask enhancement using SCHP parsing labels (P0) ─────
    # Expands the upper_body mask using SCHP label 4 (upper_clothes) to cover
    # layered garments where an underlayer (kurti, long shirt, blouse) extends
    # below the outermost visible garment (jacket, cardigan, hoodie).
    if ENABLE_GARMENT_SILHOUETTE_MASK and cloth_type == "upper_body":
        # In ATR parsing: UpperClothes=4, Dress/Kurti=7, Scarf/Dupatta=17.
        # DO NOT include Skirt=5, Pants=6, or RightShoe=10.
        _upper_clothing_labels = {4, 7, 17}
        _upper_region = np.isin(parse_768, list(_upper_clothing_labels)).astype(np.uint8) * 255

        # Morphological closing to unify jacket + underlayer regions
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 17))
        _upper_region = cv2.morphologyEx(_upper_region, cv2.MORPH_CLOSE, close_kernel)

        # Generous dilation — especially downward — to cover layered garment hems
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 19))
        _upper_region = cv2.dilate(_upper_region, dilate_kernel, iterations=1)

        mask_np = np.array(mask.convert("L"), dtype=np.uint8)
        mask_np = np.maximum(mask_np, _upper_region)

        # ── Target-Aware Arm & Sleeve Masking (Fix Static Sleeve Lockout) ──
        # If target garment has sleeves (full or half sleeved), DO NOT zero out
        # _bare_arm_region (labels 14, 15). Allow the garment to inpaint over the arms.
        # Only zero out bare arms if the target garment is explicitly detected as sleeveless/strapless.
        _bare_arm_labels = {14, 15}  # LeftArm, RightArm
        _bare_arm_region = np.isin(parse_768, list(_bare_arm_labels)).astype(np.uint8)

        if not garm_attrs.get("has_sleeves", True):
            _arm_erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            _bare_arm_region = cv2.erode(_bare_arm_region, _arm_erode_k, iterations=1)
            mask_np[_bare_arm_region > 0] = 0
            logger.info("bare_arms_protected target_sleeveless=True")
        else:
            _arm_dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            _arm_mask_expanded = cv2.dilate(_bare_arm_region, _arm_dilate_k, iterations=1) * 255
            mask_np = np.maximum(mask_np, _arm_mask_expanded)
            logger.info("bare_arms_masked_for_sleeves has_long_sleeves=%s", garm_attrs.get("has_long_sleeves", False))

        # ── Upward Collar / Neckline Expansion (Fix Collar Ghosting) ────
        # Allow the new garment's collar area to breathe so mandarin collars,
        # Nehru collars, and high necklines do not get melted with original exposed throat skin.
        if garm_attrs.get("has_collar_or_high_neck", False):
            neck_zone = (parse_768 == 18).astype(np.uint8) * 255
            neck_rows = np.where(neck_zone.any(axis=1))[0]
            if len(neck_rows) > 0:
                neck_top = int(neck_rows[0])
                neck_bottom = int(neck_rows[-1])
                neck_height = neck_bottom - neck_top
                collar_cut = neck_top + int(neck_height * 0.20)  # breathe up to top 20% of neck
                collar_inpaint_patch = np.zeros_like(neck_zone)
                collar_inpaint_patch[collar_cut:, :] = neck_zone[collar_cut:, :]
                mask_np = np.maximum(mask_np, collar_inpaint_patch)

        # ── Dynamic Downward Garment Extension (Fix Crop Top Trap) ─────
        # Compute anatomical hip/waist coordinates from OpenPose and SCHP.
        # Never limit the mask bottom to _mask_bottom + 50!
        # Use OpenPose hip keypoints (LeftHip, RightHip) or SCHP lower torso labels
        # to extend the mask over the bare stomach/navel down to the hip/thigh line.
        # Ensure the bare midriff and waistband do not clamp long garments into crop tops.
        hip_y, waist_y = _extract_hip_waist_y(keypoints, parse_768, TARGET_H, TARGET_W)
        _rows_with_mask = np.where(mask_np.any(axis=1))[0]
        if len(_rows_with_mask) > 0:
            _mask_bottom = int(_rows_with_mask[-1])

            if garm_attrs.get("is_long_garment", False):
                # Long Kurta / Kurti / Tunic / Long Shirt / Dress:
                # Extend down over bare midriff, navel, and hips down to mid-thigh line
                if hip_y is not None:
                    _target_bottom = min(TARGET_H, hip_y + int(TARGET_H * 0.18))
                elif waist_y is not None:
                    _target_bottom = min(TARGET_H, waist_y + int(TARGET_H * 0.25))
                else:
                    _target_bottom = min(TARGET_H, max(_mask_bottom + 300, int(TARGET_H * 0.72)))
                _target_bottom = max(_target_bottom, int(TARGET_H * 0.68))
                logger.info("mask_extended_for_long_garment target_bottom=%d hip_y=%s waist_y=%s", _target_bottom, hip_y, waist_y)

            elif not garm_attrs.get("is_crop_top", False):
                # Regular Top / Shirt / T-Shirt:
                # Do NOT limit mask to _mask_bottom + 50!
                # Extend over bare stomach/navel down to waistband / hip line (+40px into waistband)
                if waist_y is not None:
                    _target_bottom = min(TARGET_H, max(_mask_bottom + 100, waist_y + 40))
                elif hip_y is not None:
                    _target_bottom = min(TARGET_H, max(_mask_bottom + 100, hip_y + 30))
                else:
                    _target_bottom = min(TARGET_H, _mask_bottom + 160)
                logger.info("mask_extended_for_regular_top target_bottom=%d hip_y=%s waist_y=%s", _target_bottom, hip_y, waist_y)

            else:
                # Explicit crop top: preserve natural cropped hemline
                _target_bottom = min(TARGET_H, _mask_bottom + 30)
                logger.info("mask_kept_for_crop_top target_bottom=%d", _target_bottom)

            if _target_bottom > _mask_bottom:
                # Row-by-row anatomical extension using SCHP torso/lower body silhouette:
                # Ensures bare stomach/navel and waistband are completely covered down to _target_bottom
                for y in range(_mask_bottom, _target_bottom):
                    # Check torso / lower clothing / body pixels at row y in parse_768
                    row_body = np.where(np.isin(parse_768[y, :], [4, 5, 6, 7, 8, 12, 13, 18]))[0]
                    if len(row_body) > 0:
                        x_left = max(0, int(row_body[0]) - 10)
                        x_right = min(TARGET_W, int(row_body[-1]) + 10)
                        mask_np[y, x_left:x_right] = 255
                    else:
                        _bottom_band = mask_np[max(0, _mask_bottom - 20):_mask_bottom, :]
                        _col_sums = np.sum(_bottom_band > 127, axis=0)
                        _nonzero_cols = np.where(_col_sums > 0)[0]
                        if len(_nonzero_cols) > 0:
                            x_left = max(0, int(_nonzero_cols[0]) - 15)
                            x_right = min(TARGET_W, int(_nonzero_cols[-1]) + 15)
                            mask_np[y, x_left:x_right] = 255

        mask = Image.fromarray(mask_np, mode="L")

    # ── Universal Body Silhouette Constraint for ALL cloth types ──────
    # Prevents any mask dilation from expanding outside the human body
    # into bedroom background clutter, sofas, chairs, or textured wallpaper.
    mask_np = np.array(mask.convert("L"), dtype=np.uint8)
    _body_labels = set(range(1, 20))  # All valid SCHP person labels
    _body_silhouette = np.isin(parse_768, list(_body_labels)).astype(np.uint8) * 255
    _body_dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    _body_silhouette = cv2.dilate(_body_silhouette, _body_dilate_k, iterations=1)
    mask_np = cv2.bitwise_and(mask_np, _body_silhouette)
    mask = Image.fromarray(mask_np, mode="L")

    # ── Re-apply protected regions AFTER all SCHP expansion steps ──────
    # Protected mask must be applied last so face, hair, hands, and lower
    # body locks are never negated by SCHP label dilation above.
    mask = apply_protected_mask(mask, protected_mask)

    # Feather top edge for natural waistband transition
    if cloth_type == "lower_body":
        mask = _feather_mask_top(mask, feather=30)

    logger.info(
        "mask_selected strategy=%s mask_size=%s quality_score=%s",
        mask_meta["mask_type_used"],
        mask.size,
        mask_quality_score,
    )

    from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
    # Reuse human_img_384 for pose conditioning (Task 2 optimization: half-resolution 384x512)
    human_img_arg = _apply_exif_orientation(human_img_384)
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    # DensePose Bypass for Upper Tops:
    # For standard front/three-quarter upper-body shots, OpenPose skeleton provides
    # all necessary pose structure. Skipping DensePose RCNN inference saves ~3s of GPU overhead.
    # For lower_body, dresses, full_body or when bypass is disabled, DensePose runs at 384x512.
    should_bypass_densepose = (
        ENABLE_DENSEPOSE_BYPASS
        and cloth_type == "upper_body"
        and keypoints is not None
    )

    if should_bypass_densepose:
        logger.info("densepose_bypassed_for_upper_tops=True using_openpose_skeleton=True")
        pose_img = _render_openpose_pose_img(human_img_arg, keypoints, target_size=TARGET_SIZE)
    else:
        t_dense_0 = time.perf_counter()
        with torch.no_grad():
            densepose_outputs = densepose_predictor(human_img_arg)["instances"]

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
        logger.info("densepose_computed_at_half_res elapsed_ms=%.1f", (time.perf_counter() - t_dense_0) * 1000)

    effective_guidance = guidance_scale if guidance_scale is not None else GUIDANCE_SCALE
    effective_guidance = max(2.2, min(2.4, float(effective_guidance)))
    effective_steps = min(steps, 15) if steps >= 15 else steps

    if cloth_type in ("lower_body", "dresses", "full_body"):
        prompt = _build_subtype_aware_prompt(garment_desc, garment_subtype) + (
            ", photorealistic fabric texture, visible weave and grain, "
            "natural fabric drape and tension, soft realistic contact shadows"
        )
        if cloth_type == "lower_body":
            negative_prompt = _build_source_specific_negative() + (
                ", changed shirt, new shirt, different top, altered torso, "
                "regenerated upper body, different arms, moved hands, "
                "changed shoulders, modified chest, new upper garment, "
                "generic pants, plain pants, lost pockets, missing seams, "
                "lost belt loops, lost fabric detail, smooth texture, "
                "lost garment structure, changed silhouette, "
                "wrong drape, pasted on look, plastic texture, airbrushed"
            )
        else:
            negative_prompt = _build_source_specific_negative() + (
                ", changed garment category, wrong outfit type, "
                "mini dress, different silhouette, wrong length, "
                "missing sleeves, changed sleeve style, wrong neckline, "
                "different face, new face, changed facial features, "
                "different hair, changed hair color, different skin tone, "
                "regenerated face, altered identity, different person, "
                "generic skirt, lost hem shape, wrong print, "
                "lost fabric texture, simplified folds, "
                "symmetric skirt, ignored leg positions"
            )
    else:
        prompt = _build_subtype_aware_prompt(garment_desc, garment_subtype) + (
            ", photorealistic fabric texture, visible weave and grain, "
            "natural fabric drape and tension, crisp seam details, "
            "realistic thread texture, soft realistic contact shadows"
        )
        negative_prompt = _build_source_specific_negative() + (
            ", flat fabric, painted texture, lost stitching, "
            "smooth cloth, no folds, plastic surface, airbrushed fabric"
        )

    with torch.inference_mode():
        with _maybe_autocast():
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            prompt_c = "a photo of " + garment_desc
            _sub = (garment_subtype or "").strip().lower().replace("-", "_").replace(" ", "_")
            _cue = _FABRIC_CUES.get(_sub, "")
            if _cue:
                prompt_c = f"a photo of {garment_desc}, {_cue}"
            elif cloth_type in ("lower_body", "dresses", "full_body"):
                prompt_c = f"a photo of {garment_desc}, detailed fabric texture, natural folds, visible weave, soft contact shadows"
            prompt_embeds_c, _, _, _ = pipe.encode_prompt(
                prompt_c,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=negative_prompt,
            )

    pose_tensor = tensor_transform(pose_img).unsqueeze(0).to(device, TORCH_DTYPE)
    garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(device, TORCH_DTYPE)
    generator = torch.Generator(device).manual_seed(seed) if seed is not None and torch.cuda.is_available() else None

    with torch.inference_mode():
        with _maybe_autocast():
            images = pipe(
                prompt_embeds=prompt_embeds.to(device, TORCH_DTYPE),
                negative_prompt_embeds=negative_prompt_embeds.to(device, TORCH_DTYPE),
                pooled_prompt_embeds=pooled_prompt_embeds.to(device, TORCH_DTYPE),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, TORCH_DTYPE),
                num_inference_steps=effective_steps,
                generator=generator,
                strength=1.0,
                pose_img=pose_tensor.to(device, TORCH_DTYPE),
                text_embeds_cloth=prompt_embeds_c.to(device, TORCH_DTYPE),
                cloth=garm_tensor.to(device, TORCH_DTYPE),
                mask_image=mask,
                image=human_img,
                height=TARGET_H,
                width=TARGET_W,
                ip_adapter_image=garm_img,
                guidance_scale=effective_guidance,
            )[0]

    if auto_crop and crop_size is not None:
        out_img = images[0].resize(crop_size)
        final_img = human_img_orig.copy()

        # Edge-aware feathering: blend the crop output with the original
        # at the crop boundary to avoid visible seams
        crop_w, crop_h = crop_size
        feather_px = max(12, min(crop_w, crop_h) // 20)

        # For lower_body, use MUCH larger feathering at the top (waist) edge
        # to prevent the stitched composite artifact at the waist boundary.
        # The crop boundary at the waist is where model-generated pixels meet
        # original pixels — insufficient feathering creates a visible horizontal seam.
        top_feather = feather_px
        if cloth_type == "lower_body":
            # Compact 16px feathering at waist boundary to keep waistband stitching sharp
            # and prevent 60px translucent smudging artifacts.
            top_feather = min(feather_px, 16)

        # Build 1D feathering ramps for each edge
        alpha = np.ones((crop_h, crop_w), dtype=np.float32)

        # Top edge fade (only if crop doesn't start at image top)
        if int(top) > 0:
            ramp = np.linspace(0.0, 1.0, top_feather)
            alpha[:top_feather, :] *= ramp[:, np.newaxis]

        # Bottom edge fade (only if not at image bottom)
        orig_bottom = int(top) + crop_h
        if orig_bottom < height:
            ramp = np.linspace(1.0, 0.0, feather_px)
            alpha[-feather_px:, :] *= ramp[:, np.newaxis]

        # Left edge fade (only if not at image left)
        if int(left) > 0:
            ramp = np.linspace(0.0, 1.0, feather_px)
            alpha[:, :feather_px] *= ramp[np.newaxis, :]

        # Right edge fade (only if not at image right)
        orig_right = int(left) + crop_w
        if orig_right < width:
            ramp = np.linspace(1.0, 0.0, feather_px)
            alpha[:, -feather_px:] *= ramp[np.newaxis, :]

        # Alpha composite: output * alpha + original * (1 - alpha)
        out_np = np.array(out_img.convert("RGB"), dtype=np.float32)
        orig_crop = np.array(
            human_img_orig.crop((int(left), int(top), orig_right, orig_bottom))
            .resize((crop_w, crop_h)),
            dtype=np.float32,
        )
        blended = (out_np * alpha[..., np.newaxis] + orig_crop * (1.0 - alpha[..., np.newaxis])).astype(np.uint8)
        final_img.paste(Image.fromarray(blended), (int(left), int(top)))

        # ── Face identity restoration ──────────────────────────────────
        # The IDM-VTON model with strength=1.0 denoises the ENTIRE image,
        # regenerating the face even though it's not in the inpaint mask.
        # We MUST hard-composite the person's face from the original to
        # preserve identity. This is not a heuristic — it's required for
        # any diffusion-based try-on with strength=1.0.
        if cloth_type in ("upper_body", "dresses", "full_body", "lower_body"):
            final_img = _restore_person_identity(
                final_img, human_img_orig, cloth_type,
                crop_top=int(top),
                parsing_map=parse_768,
                inpaint_mask=mask,
                has_collar=garm_attrs.get("has_collar_or_high_neck", False),
            )

        # ── High-frequency fabric texture enhancement ──────────────────
        # Restores crisp seam/button/thread detail in the garment region
        # that diffusion smooths away. Uses the inpaint mask to target
        # only the clothing area, never face/hair/background.
        try:
            from postprocess import enhance_fabric_texture
            final_img = enhance_fabric_texture(
                final_img,
                garment_mask=mask,
                sharpen_amount=0.45,
                sharpen_radius=1.0,
                detail_boost=0.25,
            )
            logger.info("fabric_texture_enhanced=True")
        except Exception as exc:
            logger.warning("fabric_texture_enhance_failed error=%s", exc)

        return final_img, mask_meta

    # ── Non-auto-crop path: also apply texture enhancement ─────────────
    raw_result = images[0]
    try:
        from postprocess import enhance_fabric_texture
        raw_result = enhance_fabric_texture(
            raw_result,
            garment_mask=mask,
            sharpen_amount=0.45,
            sharpen_radius=1.0,
            detail_boost=0.25,
        )
        logger.info("fabric_texture_enhanced_no_crop=True")
    except Exception as exc:
        logger.warning("fabric_texture_enhance_no_crop_failed error=%s", exc)

    return raw_result, mask_meta


# =============================================================================
# Per-job
# =============================================================================

def run_inference(job_input: dict[str, Any], job_id: str) -> dict[str, Any]:
    """
    Direct inference with preprocessing support.

    Downloads person + garment images, optionally downloads preprocessing
    mask, and runs IDM-VTON. When a preprocessing mask is provided, it
    is used instead of the AutoMasker for better garment placement.
    """
    from quality_validation import validate_output_quality

    job_start = time.perf_counter()

    person_url = job_input.get("person_image_url") or job_input.get("person_image")
    garment_url = job_input.get("garment_image_url") or job_input.get("garment_image")
    garment_desc = job_input.get("garment_desc") or job_input.get("garment_description") or "garment"
    garment_subtype = job_input.get("garment_subtype", "")
    cloth_type = job_input.get("cloth_type", "upper_body")
    mask_url = job_input.get("mask_image_url") or job_input.get("mask_url") or ""
    mask_quality_raw = job_input.get("mask_quality_score")
    try:
        mask_quality_score = (
            float(mask_quality_raw)
            if mask_quality_raw is not None and mask_quality_raw != ""
            else None
        )
    except (TypeError, ValueError):
        mask_quality_score = None

    _LOWER_SUBTYPE_KEYWORDS: dict[str, list[str]] = {
        "jeans": ["jeans", "denim"],
        "trousers": ["trousers", "slacks", "formal pant", "formal trouser"],
        "pants": ["pants", "pant"],
        "shorts": ["shorts", "bermuda", "board shorts", "cargo shorts"],
        "joggers": ["joggers", "jogger", "sweatpants", "sweat pant"],
        "leggings": ["leggings", "tights", "yoga pants"],
        "cargo_pants": ["cargo", "cargo pants", "utility pants"],
        "wide_leg": ["wide leg", "wide-leg", "flared", "bootcut", "bell bottom"],
        "chinos": ["chinos", "chino"],
        "skirt": ["skirt", "mini skirt", "pencil skirt", "circle skirt"],
        "palazzo": ["palazzo", "culottes"],
        "bermuda": ["bermuda", "capri"],
        "track_pants": ["track pant", "track pants", "trackpant"],
        "pajama_pants": ["pajama pant", "pajama pants", "pyjama pant"],
        "straight_fit": ["straight fit", "regular fit", "classic fit"],
        "slim_fit": ["slim fit", "skinny", "tight fit"],
        "relaxed_fit": ["relaxed fit", "loose fit", "comfort fit"],
        "dhoti_pants": ["dhoti pants", "dhoti"],
    }
    _FULL_SUBTYPE_KEYWORDS: dict[str, list[str]] = {
        "saree": ["saree", "sari"],
        "lehenga": ["lehenga"],
        "dupatta": ["dupatta"],
        "anarkali": ["anarkali"],
        "abaya": ["abaya"],
        "kaftan": ["kaftan", "caftan"],
        "kimono": ["kimono"],
        "thobe": ["thobe", "thawb"],
        "sherwani": ["sherwani"],
        "salwar_suit": ["salwar suit", "salwar kameez", "churidar suit"],
        "sharara": ["sharara", "gharara"],
        "kurti": ["kurti"],
        "kurta_set": ["kurta set", "long kurta", "kurta dress"],
        "dress": ["dress", "one piece", "one-piece"],
        "gown": ["gown"],
        "jumpsuit": ["jumpsuit"],
        "overall": ["overall", "overalls", "dungaree"],
        "coord": ["co-ord", "coord", "co ord", "matching set", "two piece"],
    }
    # ── Upper-body subtype keywords (P4) ────────────────────────────────
    _UPPER_SUBTYPE_KEYWORDS: dict[str, list[str]] = {
        "crop_top": ["crop top", "crop", "cropped", "tube top", "tank top", "choli", "bralette", "bandeau", "corset", "short top"],
        "regular_top": ["regular top", "top", "blouse", "tunic"],
        "shirt": ["shirt", "button up", "button-up", "dress shirt", "casual shirt", "oxford"],
        "tshirt": ["t-shirt", "tshirt", "t shirt", "tee", "crew neck tee"],
        "hoodie": ["hoodie", "hooded", "sweatshirt", "pullover hoodie", "zip hoodie"],
        "jacket": ["jacket", "bomber", "windbreaker", "puffer", "denim jacket"],
        "blazer": ["blazer", "sport coat", "sports coat"],
        "cardigan": ["cardigan", "open front"],
        "sweater": ["sweater", "jumper", "pullover", "knit top"],
        "polo": ["polo", "polo shirt", "polo t-shirt"],
        "coat": ["coat", "overcoat", "trench", "parka"],
        "kurta": ["kurta", "kurti"],
        "long_kurta": ["long kurta"],
    }

    steps = int(job_input.get("steps", DENOISE_STEPS))
    steps = min(steps, 15)  # Cap steps at 14-15 for ~10-12s inference
    seed = int(job_input.get("seed", random.randint(0, 2**31 - 1)))
    trace_id = job_input.get("trace_id", "")

    # Guidance Scale: precisely in 2.2 - 2.4 range to preserve thin pinstripes
    # and crisp button details without color pooling.
    user_guidance = job_input.get("guidance_scale")
    if user_guidance is not None:
        try:
            req_guidance = float(user_guidance)
        except (ValueError, TypeError):
            req_guidance = GUIDANCE_SCALE
    else:
        req_guidance = GUIDANCE_SCALE
    effective_guidance = max(2.2, min(2.4, req_guidance))

    # Allow dynamic request-level IP-Adapter scale adjustment (default: 0.55)
    req_ip_scale = job_input.get("ip_adapter_scale")
    if req_ip_scale is not None:
        try:
            req_ip_val = max(0.4, min(0.8, float(req_ip_scale)))
            if pipe is not None and hasattr(pipe, "set_ip_adapter_scale"):
                pipe.set_ip_adapter_scale(req_ip_val)
                logger.info("request_ip_adapter_scale_set value=%.2f", req_ip_val)
        except Exception as ip_err:
            logger.warning("request_ip_adapter_scale_failed error=%s", ip_err)

    if not person_url or not garment_url:
        raise ValueError("Missing required inputs: person_image_url and garment_image_url")

    cloth_type_map = {
        "upper": "upper_body",
        "upper_body": "upper_body",
        "lower": "lower_body",
        "lower_body": "lower_body",
        "dress": "dresses",
        "dresses": "dresses",
        "overall": "dresses",
        "full_body": "dresses",
        "full": "dresses",
        "full_outfit": "dresses",
        "outfit": "dresses",
        "one_piece": "dresses",
        "one-piece": "dresses",
        "jumpsuit": "dresses",
        "kurti": "dresses",
        "saree": "dresses",
        "sari": "dresses",
        "lehenga": "dresses",
        "anarkali": "dresses",
        "abaya": "dresses",
        "kaftan": "dresses",
        "kimono": "dresses",
        "thobe": "dresses",
        "sherwani": "dresses",
        "dupatta": "dresses",
    }
    vton_type = cloth_type_map.get(cloth_type, "upper_body")

    # P1: Restrict subtype keyword matching to cloth_type-appropriate dicts
    # to prevent regular shirts from being classified as kurta/long_kurta.
    if not garment_subtype:
        _desc_lower = (garment_desc or "").lower().replace("-", " ")
        if vton_type == "upper_body":
            _keyword_dict = _UPPER_SUBTYPE_KEYWORDS
        elif vton_type == "lower_body":
            _keyword_dict = _LOWER_SUBTYPE_KEYWORDS
        else:
            _keyword_dict = {**_FULL_SUBTYPE_KEYWORDS, **_LOWER_SUBTYPE_KEYWORDS}
        for _sub, _kws in _keyword_dict.items():
            if any(_kw in _desc_lower for _kw in _kws):
                garment_subtype = _sub
                break

    garment_desc = garment_desc.strip()
    if garment_desc.lower().startswith(("a ", "an ", "the ")):
        garment_desc = garment_desc[garment_desc.index(" ") + 1:].strip()

    logger.info(
        "inference_start cloth_type=%s steps=%s seed=%s garment_desc=%s trace_id=%s",
        vton_type, steps, seed, garment_desc, trace_id,
    )

    # ── Download raw images ──
    download_start = time.perf_counter()
    person_img = download_image(person_url)
    garment_img = download_image(garment_url)
    # Download preprocessing mask if provided (from preprocessing service)
    external_mask = None
    if mask_url:
        try:
            external_mask = download_image(mask_url)
            logger.info("preprocessing_mask_downloaded url=%s", mask_url[:80])
        except Exception as exc:
            logger.warning("preprocessing_mask_download_failed error=%s — falling back to AutoMasker", exc)
            external_mask = None
    download_ms = (time.perf_counter() - download_start) * 1000

    # ── Garment RGB diagnostics ──
    garm_np = np.array(garment_img.convert("RGB"), dtype=np.float32)
    garm_mean_all = float(np.mean(garm_np))
    logger.info(
        "garment_rgb_stats mean_all=%.1f is_dark=%s",
        garm_mean_all, garm_mean_all < 80.0,
    )

    # ── Direct inference — always use AutoMasker, no preprocessing ──
    # NOTE: previously dark garments (mean < 80) got guidance * 0.75, which
    # weakened garment conditioning and washed out black/dark garments
    # (lost texture, turned gray). Dark garments need FULL guidance so the
    # model actually applies the (low-luminance) garment color/texture.
    # Guidance scale in 2.2 - 2.4 sweet spot prevents muddy color pooling
    effective_guidance = max(2.2, min(2.4, float(effective_guidance)))

    inference_start = time.perf_counter()
    result, mask_meta = run_idm_vton_inference(
        person_img=person_img,
        garment_img=garment_img,
        garment_desc=garment_desc,
        cloth_type=vton_type,
        garment_subtype=garment_subtype,
        steps=steps,
        seed=seed,
        auto_crop=True,
        external_mask=external_mask,
        protected_mask=None,
        mask_strategy="external" if external_mask is not None else "automasker",
        mask_quality_score=mask_quality_score,
        guidance_scale=effective_guidance,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_ms = (time.perf_counter() - inference_start) * 1000

    # ── Quality validation ──
    geometry_report = None
    if result is not None:
        geometry_report = validate_output_quality(
            person_img,
            result,
            Image.fromarray(np.zeros((TARGET_H, TARGET_W), dtype=np.uint8), mode="L"),
            vton_type,
            garment_img,
        )
        if not geometry_report["passed"]:
            logger.warning("quality_check_failed reasons=%s", geometry_report["reasons"])

    upload_start = time.perf_counter()
    result_url = _upload_to_cloudinary(result, job_id)
    upload_ms = (time.perf_counter() - upload_start) * 1000

    total_ms = (time.perf_counter() - job_start) * 1000

    logger.info(
        "job_complete total_ms=%.0f download_ms=%.0f inference_ms=%.0f upload_ms=%.0f "
        "mask_type=%s trace_id=%s",
        total_ms, download_ms, inference_ms, upload_ms,
        mask_meta.get("mask_type_used"),
        trace_id,
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
        "mask_type_used": mask_meta.get("mask_type_used"),
        "trace_id": trace_id,
    }


# =============================================================================
# RunPod handler
# =============================================================================

def handler(job: dict[str, Any]) -> dict[str, Any]:
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

    import gc
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
    finally:
        # Free CUDA memory and run garbage collection after every job
        # to prevent memory accumulation and worker container restarts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# =============================================================================
# Startup
# =============================================================================

_ensure_logging()

# ── Startup diagnostics: verify mask_pipeline import ──────────────────
def _startup_diagnostics():
    """
    Verify that mask_pipeline.py is available and importable at runtime.

    Checks:
      1. /workspace is in sys.path (or adds it)
      2. /workspace/mask_pipeline.py exists on disk
      3. The module imports correctly

    This runs once at worker startup, before any job arrives, so the
    ModuleNotFoundError that previously only appeared during jobs
    is caught early.
    """
    logger.info("STARTUP_DIAG: cwd=%s", os.getcwd())
    logger.info("STARTUP_DIAG: sys.path=%s", sys.path)
    logger.info("STARTUP_DIAG: handler_location=%s", os.path.abspath(__file__))

    # Belt-and-suspenders: ensure /workspace is on sys.path
    ws = "/workspace"
    if ws not in sys.path:
        sys.path.insert(0, ws)
        logger.info("STARTUP_DIAG: added %s to sys.path", ws)

    # Check file exists on disk
    mp_path = os.path.join(ws, "mask_pipeline.py")
    if not os.path.isfile(mp_path):
        logger.error(
            "STARTUP_DIAG: mask_pipeline.py NOT FOUND at %s — "
            "Dockerfile must have COPY mask_pipeline.py /workspace/mask_pipeline.py",
            mp_path,
        )
        return False

    logger.info("STARTUP_DIAG: mask_pipeline.py found at %s (%d bytes)", mp_path, os.path.getsize(mp_path))

    # Actual import test — catches ModuleNotFoundError at startup, not during a job
    try:
        from mask_pipeline import (
            WorkerMaskStrategy,
            apply_protected_mask,
            fuse_hybrid_mask,
            detect_inference_failures,
            select_worker_mask_strategy,
        )
        logger.info("STARTUP_DIAG: import mask_pipeline OK")
        return True
    except Exception as exc:
        logger.error(
            "STARTUP_DIAG: import mask_pipeline FAILED — %s: %s",
            type(exc).__name__, exc,
        )
        return False

_startup_diagnostics_result = _startup_diagnostics()
if not _startup_diagnostics_result:
    logger.warning(
        "STARTUP_DIAG: mask_pipeline is unavailable — inference retry "
        "and hybrid mask features will fail when a job arrives"
    )

logger.info("=" * 60)
logger.info("IDM-VTON Worker v2.0.0 — loading")
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
