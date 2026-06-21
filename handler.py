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
IDM_VTON_MODEL = os.environ.get("IDM_VTON_MODEL", "/workspace/models/yisol/IDM-VTON")
DENSEPOSE_WEIGHTS = os.environ.get(
    "DENSEPOSE_WEIGHTS",
    "/workspace/IDM-VTON/ckpt/densepose/model_final_162be9.pkl",
)

CLOUDINARY_FOLDER = os.environ.get("CLOUDINARY_FOLDER", "trylix/tryon/results")

DENOISE_STEPS = int(os.environ.get("IDM_VTON_STEPS", "45"))
GUIDANCE_SCALE = float(os.environ.get("IDM_VTON_GUIDANCE", "2.75"))

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16

# Memory/perf knobs
ENABLE_XFORMERS = os.environ.get("ENABLE_XFORMERS", "1") == "1"
ENABLE_TORCH_COMPILE = os.environ.get("ENABLE_TORCH_COMPILE", "0") == "1"
ENABLE_MODEL_CPU_OFFLOAD = os.environ.get("ENABLE_MODEL_CPU_OFFLOAD", "0") == "1"
ALLOW_TF32 = os.environ.get("ALLOW_TF32", "1") == "1"

# Multi-candidate generation — generate N candidates with different
# seeds on the first mask strategy, pick the one with the best aggregate
# quality score.  Set to 1 for original single-candidate behaviour.
MULTI_CANDIDATE_COUNT = int(os.environ.get("MULTI_CANDIDATE_COUNT", "4"))

# Candidate diversity — vary guidance scale and denoising steps across
# candidates so the scoring system can choose from structurally different
# outputs rather than pure seed-noise variants.  Set to 0 to use the same
# parameters for all candidates (original behaviour).
CANDIDATE_GUIDANCE_VARY = os.environ.get("CANDIDATE_GUIDANCE_VARY", "1") == "1"
CANDIDATE_STEPS_VARY = os.environ.get("CANDIDATE_STEPS_VARY", "1") == "1"

# Post-processing feature flags
#   ENABLE_FACE_RESTORATION=1   — GFPGAN or OpenCV face enhancement
#   ENABLE_FACE_COMPOSITE=1     — composite original face/accessories back (default: on)
#   ENABLE_SEAMLESS_CLONE=1     — Poisson edge blending on garment boundary (default: off)
#   ENABLE_SKIN_TONE_CORRECTION=1 — per-channel skin color correction (default: off)
#   ENABLE_DEBUG_IMAGES=1       — save 5-stage debug images to /tmp/trylix_debug/
# All are read at runtime (not startup), so they can be toggled per-job via env injection.

# Concurrency — single GPU SDXL typically needs ~16-20 GB VRAM.
# On a 24 GB card keep max_workers=1; set to 2+ only if the GPU
# (or multi-GPU setup) has enough headroom for concurrent passes.
MAX_WORKERS = int(os.environ.get("RUNPOD_MAX_WORKERS", "1"))

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
_REUSE_LOCK = threading.Lock()

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
        # Model weight subdirectories (baked into the image via Layer 5)
        Path(IDM_VTON_MODEL) / "unet",
        Path(IDM_VTON_MODEL) / "vae",
        Path(IDM_VTON_MODEL) / "scheduler",
        Path(IDM_VTON_MODEL) / "tokenizer",
        Path(IDM_VTON_MODEL) / "tokenizer_2",
        Path(IDM_VTON_MODEL) / "image_encoder",
        Path(IDM_VTON_MODEL) / "text_encoder",
        Path(IDM_VTON_MODEL) / "text_encoder_2",
        Path(IDM_VTON_MODEL) / "unet_encoder",
        # Lightweight build-time assets
        Path(IDM_VTON_DIR) / "configs" / "densepose_rcnn_R_50_FPN_s1x.yaml",
        Path(DENSEPOSE_WEIGHTS),
    ]
    for p in needed:
        _require_path(p, f"required path {p}")

    parsing_paths = [
        Path(IDM_VTON_DIR) / "ckpt" / "humanparsing" / "parsing_atr.onnx",
        Path(IDM_VTON_DIR) / "ckpt" / "humanparsing" / "parsing_lip.onnx",
        Path(IDM_VTON_DIR) / "ckpt" / "openpose" / "body_pose_model.pth",
    ]
    for p in parsing_paths:
        _require_path(p, f"required path {p}")

    optional_paths = [
        ("ckpt/image_encoder", Path(IDM_VTON_DIR) / "ckpt" / "image_encoder"),
        ("ckpt/ip_adapter", Path(IDM_VTON_DIR) / "ckpt" / "ip_adapter"),
    ]
    for label, p in optional_paths:
        if not p.exists():
            logger.warning("Optional path %s not found — IP-Adapter features may be degraded", label)
        else:
            logger.info("Optional path %s OK", label)


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
    image.save(buffer, format="JPEG", quality=98, optimize=True, subsampling=0)
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
            torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
            torch.backends.cudnn.allow_tf32 = ALLOW_TF32
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

    # ── Startup diagnostics: disk, symlinks, weight inventory ────────────
    import shutil
    total, used, free = shutil.disk_usage("/workspace")
    logger.info("DISK: total_gb=%.1f used_gb=%.1f free_gb=%.1f",
        total / (1024**3), used / (1024**3), free / (1024**3))

    logger.info("MODEL_PATH=%s", IDM_VTON_MODEL)
    logger.info("MODEL_EXISTS=%s", os.path.isdir(IDM_VTON_MODEL))
    if os.path.isdir(IDM_VTON_MODEL):
        try:
            logger.info("MODEL_CONTENTS=%s", sorted(os.listdir(IDM_VTON_MODEL)))
        except Exception:
            pass

        # Per-subfolder weight file inventory
        weight_extensions = (".bin", ".safetensors", ".pt", ".pth")
        subfolders = ["unet", "vae", "scheduler", "tokenizer", "tokenizer_2",
                      "image_encoder", "text_encoder", "text_encoder_2", "unet_encoder"]
        for sub in subfolders:
            subpath = os.path.join(IDM_VTON_MODEL, sub)
            if not os.path.isdir(subpath):
                logger.warning("MODEL_SUBFOLDER_MISSING sub=%s", sub)
                continue
            files = [f for f in os.listdir(subpath) if f.endswith(weight_extensions)]
            if not files:
                logger.warning("MODEL_SUBFOLDER_EMPTY sub=%s path=%s", sub, subpath)
            for fname in files:
                fpath = os.path.join(subpath, fname)
                try:
                    size_mb = os.path.getsize(fpath) / 1024 / 1024
                    is_link = os.path.islink(fpath)
                    link_info = " SYMLINK" if is_link else ""
                    logger.info("WEIGHT: %s/%s size_mb=%.1f%s", sub, fname, size_mb, link_info)
                except OSError as e:
                    logger.error("WEIGHT_ERROR: %s/%s — %s", sub, fname, e)
    else:
        logger.warning("MODEL_DIR_DOES_NOT_EXIST — from_pretrained will trigger snapshot_download")

    # ── snapshot_download monkey-patch for diagnostics ──────────────────
    import huggingface_hub
    _original_snapshot = huggingface_hub.snapshot_download
    def _diagnostic_snapshot(*args, **kwargs):
        logger.warning("SNAPSHOT_DOWNLOAD_TRIGGERED args=%s kwargs=%s", args, kwargs)
        total2, used2, free2 = shutil.disk_usage("/workspace")
        logger.warning("SNAPSHOT_DOWNLOAD_DISK pre: total_gb=%.1f used_gb=%.1f free_gb=%.1f",
            total2 / (1024**3), used2 / (1024**3), free2 / (1024**3))
        result = _original_snapshot(*args, **kwargs)
        total3, used3, free3 = shutil.disk_usage("/workspace")
        logger.warning("SNAPSHOT_DOWNLOAD_DISK post: total_gb=%.1f used_gb=%.1f free_gb=%.1f",
            total3 / (1024**3), used3 / (1024**3), free3 / (1024**3))
        return result
    huggingface_hub.snapshot_download = _diagnostic_snapshot

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

    logger.info("Loading scheduler...")
    noise_scheduler = DDPMScheduler.from_pretrained(
        IDM_VTON_MODEL,
        subfolder="scheduler",
    )

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

    if ENABLE_XFORMERS:

        logger.info("Attempting xformers enable...")

        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("xformers_enabled=True")

        except Exception as exc:
            logger.warning(
                "xformers_enable_failed error=%s",
                exc,
            )

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

def warmup():
    global _REUSE_COUNT
    if _WARM.is_set():
        return

    logger.info("=" * 60)
    logger.info("COLD START BEGIN")
    logger.info("=" * 60)

    load_models()

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info("gpu_warmup_ready=True")
    except Exception as exc:
        logger.warning("gpu_warmup_skipped error=%s", exc)

    # ── Dummy inference (single step) to compile CUDA kernels and warm memory ──
    try:
        logger.info("warmup_inference_start")
        wt0 = time.perf_counter()
        dummy_person = Image.new("RGB", TARGET_SIZE, (128, 128, 128))
        dummy_garment = Image.new("RGB", TARGET_SIZE, (200, 100, 50))
        _, _, _ = run_idm_vton_inference(
            person_img=dummy_person,
            garment_img=dummy_garment,
            garment_desc="shirt",
            cloth_type="upper_body",
            steps=1,
            seed=42,
            auto_crop=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wt1 = time.perf_counter()
        logger.info("warmup_inference_complete elapsed_ms=%.0f", (wt1 - wt0) * 1000)
    except Exception as exc:
        logger.warning("warmup_inference_skipped error=%s", exc)

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


def run_idm_vton_inference(
    person_img: Image.Image,
    garment_img: Image.Image,
    garment_desc: str,
    cloth_type: str,
    garment_subtype: str = "",
    steps: int = 30,
    seed: int = 42,
    auto_crop: bool = True,
    external_mask: Image.Image | None = None,
    protected_mask: Image.Image | None = None,
    mask_strategy: str = "external",
    mask_quality_score: float | None = None,
    guidance_scale: float | None = None,
    crop_preserve_lower: bool = True,
) -> tuple[Image.Image, dict[str, object]]:
    global pipe, parsing_model, openpose_model
    global densepose_predictor, densepose_cfg, tensor_transform, get_mask_location_fn

    import cv2
    import numpy as np

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
        validate_mask_coverage,
    )

    # Preserve garment aspect ratio — only pad to target, never stretch
    garm_img = garment_img.convert("RGB")
    if garm_img.size != TARGET_SIZE:
        gw, gh = garm_img.size
        scale = min(TARGET_W / gw, TARGET_H / gh)
        nw = max(1, int(gw * scale))
        nh = max(1, int(gh * scale))
        garm_resized = garm_img.resize((nw, nh), Image.LANCZOS)
        garm_canvas = Image.new("RGB", TARGET_SIZE, (255, 255, 255))
        garm_canvas.paste(garm_resized, ((TARGET_W - nw) // 2, (TARGET_H - nh) // 2))
        garm_img = garm_canvas
    else:
        garm_img = garm_img.convert("RGB")
    human_img_orig = person_img.convert("RGB")

    width, height = human_img_orig.size
    left, top, crop_size = 0.0, 0.0, None

    if auto_crop:
        target_width = int(min(width, height * (TARGET_W / TARGET_H)))
        target_height = int(min(height, width * (TARGET_H / TARGET_W)))

        # Determine crop anchor based on cloth_type:
        #   - lower_body / dresses: BOTTOM-anchored to preserve legs/feet
        #   - upper_body / default: CENTER-anchored (original behavior)
        #
        # This is critical: the preprocessing service already pads the image
        # correctly for the garment type (bottom-anchor for dresses/lower_body).
        # A center crop would undo that work and cut off legs.
        is_full_body = cloth_type in ("dresses", "lower_body", "full_body")
        if crop_preserve_lower and is_full_body:
            # Bottom-anchored crop: keep the bottom portion, sacrifice top
            left = (width - target_width) / 2
            bottom = height
            top = height - target_height
            right = (width + target_width) / 2
            logger.info(
                "auto_crop_bottom_anchored cloth_type=%s "
                "target=%dx%d image=%dx%d crop_top=%d crop_bottom=%d",
                cloth_type, target_width, target_height,
                width, height, top, bottom,
            )
        else:
            # Center-anchored crop (original)
            left = (width - target_width) / 2
            top = (height - target_height) / 2
            right = (width + target_width) / 2
            bottom = (height + target_height) / 2
            logger.info(
                "auto_crop_center_anchored cloth_type=%s "
                "target=%dx%d image=%dx%d",
                cloth_type, target_width, target_height, width, height,
            )

        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize(TARGET_SIZE)
    else:
        human_img = human_img_orig.resize(TARGET_SIZE)

    # Align protected_mask with auto-crop — the mask was computed from the
    # full-image resize, but the inference image was cropped then resized.
    # Without alignment, face protection is applied at wrong coordinates,
    # exposing face boundary pixels to diffusion modification.
    if auto_crop and crop_size is not None and protected_mask is not None:
        t_left = int(left * TARGET_W / width)
        t_top = int(top * TARGET_H / height)
        t_right = int(right * TARGET_W / width)
        t_bottom = int(bottom * TARGET_H / height)
        prot_cropped = protected_mask.crop((t_left, t_top, t_right, t_bottom))
        protected_mask = prot_cropped.resize(TARGET_SIZE, Image.NEAREST)

    # Always compute AutoMasker mask (SCHP + OpenPose) for routing / hybrid
    keypoints = openpose_model(human_img.resize((384, 512)))
    model_parse, _ = parsing_model(human_img.resize((384, 512)))
    automasker_mask, _ = get_mask_location_fn("hd", cloth_type, model_parse, keypoints)
    automasker_mask = automasker_mask.resize(TARGET_SIZE)

    # ── Universal AutoMasker mask enlargement ──────────────────────────
    # CRITICAL FINDING from research: "The mask area needs to be as large
    # as possible. The garment mask shouldn't just frame the garment itself
    # but needs to leave enough drawing space for different garment
    # replacements." (CatVTON training notes)
    #
    # ATR parsing (SCHP) clips tightly around detected body regions, which
    # leaves no margin for the new garment to be rendered. The diffusion
    # model needs "drawing space" — extra mask area around the body —
    # especially for lower-body, full-body, and oversized clothing.
    #
    # This applies to ALL garment types with type-specific intensity:
    #   lower_body/dresses: strong dilation on leg zone (25px × 3)
    #   upper_body:         gentle dilation on whole mask (19px × 2)
    #   full_body:          strongest dilation for full outfits (25px × 3)
    # Increased intensity vs. previous: better drawing space for realistic
    # wrinkles, folds, and fabric draping at garment edges.
    auto_np = np.array(automasker_mask, dtype=np.uint8)
    if cloth_type in ("lower_body", "dresses", "full_body"):
        h_auto = auto_np.shape[0]
        lower_zone = auto_np[h_auto * 3 // 5:, :]
        leg_cov = float(np.mean(lower_zone > 127))
        leg_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        auto_np = cv2.dilate(auto_np, leg_k, iterations=3)
        logger.info(
            "automasker_boost lower_body leg_coverage=%.2f iterations=3",
            leg_cov,
        )
    else:
        cov = float(np.mean(auto_np > 127))
        mild_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        auto_np = cv2.dilate(auto_np, mild_k, iterations=2)
        logger.info(
            "automasker_boost upper_body coverage=%.2f iterations=2",
            cov,
        )
    automasker_mask = Image.fromarray(auto_np, mode="L")

    min_quality = float(os.environ.get("MASK_MIN_QUALITY_SCORE", "62.0"))
    strategy = select_worker_mask_strategy(
        external_mask,
        mask_quality_score,
        min_quality=min_quality,
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

    # Snapshot the pre-protection mask so the outer retry loop can use
    # it for multi-candidate quality scoring (texture / artifact analysis).
    mask_meta["inpaint_mask_pil"] = mask.copy()

    mask = apply_protected_mask(mask, protected_mask)
    logger.info(
        "mask_selected strategy=%s mask_size=%s quality_score=%s",
        mask_meta["mask_type_used"],
        mask.size,
        mask_quality_score,
    )

    # ── Pre-inference mask validation ──────────────────────────────────
    # Catch pathological masks before they waste ~8s of GPU inference.
    mask_v = validate_mask_coverage(mask, cloth_type, protected_mask=protected_mask)
    logger.info(
        "mask_coverage coverage=%.1f%% valid=%s cloth_type=%s",
        mask_v["coverage_percent"], mask_v["valid"], cloth_type,
    )
    if not mask_v["valid"]:
        logger.warning(
            "pre_inference_mask_invalid reason=%s coverage=%.1f%% cloth_type=%s",
            mask_v["reason"], mask_v["coverage_percent"], cloth_type,
        )

    from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    with torch.no_grad():
        densepose_pred = densepose_predictor(human_img_arg)
        if "instances" not in densepose_pred or len(densepose_pred["instances"]) == 0:
            logger.warning(
                "densepose_no_instances_fallback image_shape=%s cloth_type=%s",
                human_img_arg.shape, cloth_type,
            )
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

    effective_guidance = guidance_scale if guidance_scale is not None else GUIDANCE_SCALE

    _SUBTYPE_FABRIC = {
        "jeans": "denim fabric, realistic denim texture, natural denim folds, faded whiskers, authentic seams, coin pocket, rivet details, worn fabric texture",
        "hoodie": "cotton fleece fabric, relaxed fit, natural wrinkles, soft folds, ribbed cuffs, drawstring hood, pouch pocket seams",
        "sweatshirt": "cotton fleece fabric, relaxed fit, natural wrinkles, soft folds, ribbed hem and cuffs",
        "t-shirt": "cotton jersey fabric, natural fit, soft folds, realistic drape, ribbed neckline seam, hem stitch detail",
        "shirt": "woven cotton fabric, tailored fit, sharp creases, natural wrinkles, button placket seam, collar stand, cuffed sleeves",
        "blazer": "structured woven fabric, tailored fit, sharp seams, natural folds, notch lapel, chest pocket seam, vent detail",
        "jacket": "structured fabric, fitted shoulders, realistic seams, natural folds, zipper detail, pocket flaps, hem band",
        "sweater": "knit fabric, relaxed fit, knit texture, natural wrinkles, ribbed crew neck, cable knit detail, hem ribbing",
        "cardigan": "knit fabric, relaxed fit, knit texture, natural wrinkles, button placket, ribbed edges",
        "blouse": "flowing fabric, natural drape, soft wrinkles, realistic folds, collar detail, pleated shoulder seam",
        "top": "soft fabric, natural fit, realistic folds, soft drape, neckline seam, hem detail",
        "vest": "structured fabric, fitted, sharp seams, natural folds, armhole binding, button front seam",
        "kurta": "flowing fabric, straight cut, natural drape, soft folds, neckline embroidery detail, side slits",
        "pants": "woven fabric, natural fabric folds, realistic wrinkles, tailored crease, belt loop seams, hem stitch, pocket outline",
        "trousers": "woven fabric, tailored fit, sharp crease, natural wrinkles, pleated front seam, hem detail",
        "shorts": "cotton fabric, relaxed fit, soft folds, natural wrinkles, hem seam, pocket stitching",
        "skirt": "flowing fabric, natural drape, realistic folds, soft wrinkles, waistband seam, hemline detail",
        "dress": "flowing fabric, natural drape, realistic wrinkles, soft folds, waist seam, neckline binding, hem detail",
        "gown": "flowing fabric, floor-length drape, elegant folds, natural wrinkles, bodice seam, skirt gathers, hem stitch",
        "jumpsuit": "structured fabric, tailored fit, sharp seams, natural folds, waist seam detail, leg hem, pocket stitching",
        "saree": "draped silk fabric, flowing silhouette, realistic saree folds, natural pleats, decorative border detail, pallu drape, fabric sheen",
        "lehenga": "structured waistband, flowing skirt, realistic pleats, natural folds, embroidered border detail, waist seam",
        "tracksuit": "cotton fabric, sporty fit, soft folds, natural wrinkles, ribbed cuffs, drawstring waistband, leg zip detail",
        "co-ord": "matching fabric, coordinated fit, natural folds, realistic wrinkles, waist seam detail, hem finishing",
    }
    fabric_desc = _SUBTYPE_FABRIC.get(garment_subtype, "structured fabric, natural folds, realistic texture, sharp details, visible seams, natural shadows, fabric grain visible")
    prompt = "model is wearing " + garment_desc + ", " + fabric_desc + ", no accessories"
    negative_prompt = (
        "monochrome, lowres, bad anatomy, worst quality, low quality, "
        "deformed, distorted, disfigured, bad proportions, "
        "extra limbs, missing limbs, cloned head, body out of frame, "
        "poorly drawn face, mutation, mutated, extra fingers, "
        "ugly, blurry, watermark, signature, text, logo, "
        "smooth plastic, airbrushed, cg render, 3d render, "
        "flat lighting, "
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

    with torch.inference_mode():
        with _maybe_autocast():
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            prompt_c = "a photo of " + garment_desc + ", " + fabric_desc
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
            pipe_output = pipe(
                prompt_embeds=prompt_embeds.to(device, TORCH_DTYPE),
                negative_prompt_embeds=negative_prompt_embeds.to(device, TORCH_DTYPE),
                pooled_prompt_embeds=pooled_prompt_embeds.to(device, TORCH_DTYPE),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, TORCH_DTYPE),
                num_inference_steps=steps,
                generator=generator,
                strength=1.0,
                pose_img=pose_tensor.to(device, TORCH_DTYPE),
                text_embeds_cloth=prompt_embeds_c.to(device, TORCH_DTYPE),
                cloth=garm_tensor.to(device, TORCH_DTYPE),
                mask_image=mask,
                image=human_img,
                height=TARGET_H,
                width=TARGET_W,
                ip_adapter_image=garm_img.resize(TARGET_SIZE),
                guidance_scale=effective_guidance,
            )
            images = pipe_output[0]
            if not images:
                logger.error("pipeline_returned_empty_images")
                raise RuntimeError("Pipeline returned empty images list — inference produced no output")

    raw_output = images[0].copy()

    if auto_crop and crop_size is not None:
        out_img = images[0].resize(crop_size)
        final_img = human_img_orig.copy()
        final_img.paste(out_img, (int(left), int(top)))
        return final_img, raw_output, mask_meta

    return images[0], raw_output, mask_meta


# =============================================================================
# Per-job
# =============================================================================

def _validate_person_quality(
    person_img: Image.Image,
    min_side: int = 300,
) -> tuple[bool, str]:
    """Quick quality check on the downloaded person image.

    Returns (valid, reason). Runs before expensive GPU inference to catch
    inputs that would produce garbage outputs regardless of model quality.
    """
    w, h = person_img.size
    if w < min_side or h < min_side:
        return False, f"image_too_small:{w}x{h}<{min_side}"
    arr = np.array(person_img.convert("L"), dtype=np.float32)
    std = float(np.std(arr))
    if std < 1.0:
        return False, f"blank_image:std={std:.2f}"
    return True, ""


def run_inference(job_input: dict[str, Any], job_id: str) -> dict[str, Any]:
    from mask_pipeline import (
        WorkerMaskStrategy,
        compute_aggregate_quality_score,
        detect_inference_failures,
    )

    job_start = time.perf_counter()

    person_url = job_input.get("person_image_url") or job_input.get("person_image")
    garment_url = job_input.get("garment_image_url") or job_input.get("garment_image")
    mask_image_ref = job_input.get("mask_image") or job_input.get("mask_image_url")
    protected_ref = job_input.get("protected_mask") or job_input.get("protected_mask_url")
    garment_desc = job_input.get("garment_desc") or job_input.get("garment_description") or "garment"
    garment_subtype = job_input.get("garment_subtype") or ""
    cloth_type = job_input.get("cloth_type", "upper_body")
    steps = int(job_input.get("steps", DENOISE_STEPS))
    seed = int(job_input.get("seed", random.randint(0, 2**31 - 1)))
    mask_quality_score = job_input.get("mask_quality_score")
    if mask_quality_score is not None:
        mask_quality_score = float(mask_quality_score)
    mask_strategy_hint = str(job_input.get("mask_strategy", "auto"))
    trace_id = job_input.get("trace_id", "")
    # Forward preprocessing warnings through the output so the frontend
    # can display them to the user (body completeness, face, etc.)
    input_warnings = job_input.get("warnings", [])
    if not isinstance(input_warnings, list):
        input_warnings = []
    max_retries = int(os.environ.get("MASK_WORKER_MAX_RETRIES", "2"))
    retry_enabled = os.environ.get("MASK_WORKER_RETRY", "1") == "1"

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
    }
    vton_type = cloth_type_map.get(cloth_type, "upper_body")

    import numpy as np

    # Color-preserving garment description — include color from preprocessing
    garment_desc = garment_desc.strip()
    if garment_desc.lower().startswith(("a ", "an ", "the ")):
        garment_desc = garment_desc[garment_desc.index(" ") + 1:].strip()

    logger.info(
        "inference_start cloth_type=%s steps=%s seed=%s garment_desc=%s trace_id=%s",
        vton_type, steps, seed, garment_desc, trace_id,
    )

    # ── Garment RGB diagnostics (after download below) ────────────────

    download_start = time.perf_counter()
    person_img = download_image(person_url)
    garment_img = download_image(garment_url)

    # ── Pre-inference quality check ────────────────────────────────────
    q_ok, q_reason = _validate_person_quality(person_img)
    if not q_ok:
        raise ValueError(f"Person image rejected: {q_reason}")
    logger.info("person_image_quality_ok size=%s std=%.2f", person_img.size,
                float(np.std(np.array(person_img.convert("L"), dtype=np.float32))))

    external_mask = None
    if mask_image_ref:
        try:
            external_mask_img = load_image_reference(str(mask_image_ref))
            external_mask = external_mask_img.convert("L").resize(TARGET_SIZE)
            logger.info(
                "external_mask_loaded source=%s mask_size=%s quality_score=%s",
                "url" if _is_url_reference(str(mask_image_ref)) else "base64",
                external_mask.size,
                mask_quality_score,
            )
        except Exception as exc:
            logger.warning(
                "external_mask_load_failed error=%s falling_back_to_automasker",
                exc,
            )

    protected_mask = None
    if protected_ref:
        try:
            protected_mask = load_image_reference(str(protected_ref)).convert("L")
            logger.info("protected_mask_loaded size=%s", protected_mask.size)
        except Exception as exc:
            logger.warning("protected_mask_load_failed error=%s", exc)

    download_ms = (time.perf_counter() - download_start) * 1000

    # ── Garment foreground area check ──────────────────────────────────
    # If the garment image is mostly white/background (e.g. a product shot
    # placed on a white canvas with too much padding), the model doesn't
    # have enough garment pixels to render correctly. Log the ratio for
    # monitoring — severe cases could be addressed by fallback.
    garm_check = np.array(garment_img.convert("RGB"), dtype=np.uint8)
    non_white = (
        (garm_check[:, :, 0] < 240)
        | (garm_check[:, :, 1] < 240)
        | (garm_check[:, :, 2] < 240)
    )
    garm_foreground_ratio = float(np.mean(non_white))
    logger.info(
        "garment_foreground_ratio=%.3f cloth_type=%s trace_id=%s",
        garm_foreground_ratio, vton_type, trace_id,
    )
    if garm_foreground_ratio < 0.10:
        logger.warning(
            "garment_very_small_on_canvas ratio=%.3f cloth_type=%s trace_id=%s",
            garm_foreground_ratio, vton_type, trace_id,
        )

    # ── Garment RGB diagnostics (must run AFTER garment_img is downloaded) ─
    garm_np = np.array(garment_img.convert("RGB"), dtype=np.float32)
    garm_mean_r = float(np.mean(garm_np[:, :, 0]))
    garm_mean_g = float(np.mean(garm_np[:, :, 1]))
    garm_mean_b = float(np.mean(garm_np[:, :, 2]))
    garm_mean_all = (garm_mean_r + garm_mean_g + garm_mean_b) / 3.0
    garm_is_dark = garm_mean_all < 80.0
    logger.info(
        "garment_rgb_stats mean_r=%.1f mean_g=%.1f mean_b=%.1f mean_all=%.1f is_dark=%s",
        garm_mean_r, garm_mean_g, garm_mean_b, garm_mean_all, garm_is_dark,
    )

    # ── Garment dimension logging (fidelity audit) ──────────────────────
    garm_w, garm_h = garment_img.size
    garm_aspect = garm_w / max(garm_h, 1)
    target_aspect = TARGET_W / TARGET_H
    logger.info(
        "garment_dimensions size=%dx%d aspect=%.4f target_aspect=%.4f cloth_type=%s trace_id=%s",
        garm_w, garm_h, garm_aspect, target_aspect, vton_type, trace_id,
    )

    # Retry strategies — mask_strategy_hint overrides the first attempt
    retry_strategies: list[str] = []
    if mask_strategy_hint == "hybrid" and external_mask:
        retry_strategies = ["hybrid", "automasker"]
    elif mask_strategy_hint == "automasker":
        retry_strategies = ["automasker"]
    else:
        if external_mask:
            retry_strategies.append("external")
        retry_strategies.append("automasker")
        if external_mask:
            retry_strategies.append("hybrid")
    if not retry_enabled:
        retry_strategies = retry_strategies[:1]

    inference_start = time.perf_counter()
    result: Image.Image | None = None
    mask_meta: dict[str, object] = {}
    quality_report = None
    retry_count = 0
    failure_reasons: list[str] = []
    last_inpaint_mask = external_mask

    # Dark garment handling — per-channel guidance adjustment.
    # Dark garments (mean < 80) need HIGHER guidance so the model
    # preserves their color against the diffusion prior's tendency
    # toward mid-gray. Low guidance on dark garments washes them out.
    effective_guidance = GUIDANCE_SCALE
    if garm_mean_all < 80.0:
        effective_guidance = GUIDANCE_SCALE * 1.15
        logger.info(
            "dark_garment_detected mean_r=%.1f mean_g=%.1f mean_b=%.1f "
            "increasing_guidance from %.1f to %.1f",
            garm_mean_r, garm_mean_g, garm_mean_b,
            GUIDANCE_SCALE, effective_guidance,
        )

    for attempt_idx, strategy in enumerate(retry_strategies[: max_retries + 1]):
        retry_count = attempt_idx
        crop_preserve = True
        if vton_type in ("lower_body", "dresses") and attempt_idx > 0:
            crop_preserve = False
            logger.info(
                "retry_crop_variant center_crop attempt=%s cloth_type=%s",
                attempt_idx, vton_type,
            )

        # ── Multi-candidate generation ──────────────────────────────────
        # On the first attempt, generate N candidates with different seeds
        # and pick the one with the best aggregate quality score.  Retry
        # attempts use a single candidate to minimise cost.
        num_candidates = max(1, MULTI_CANDIDATE_COUNT) if attempt_idx == 0 else 1

        candidates: list[dict] = []
        for c in range(num_candidates):
            c_seed = seed + c * 1000 + attempt_idx

            # ── Per-candidate guidance scale ────────────────────────────
            # Vary guidance across candidates to explore the adherence-vs-creativity
            # trade-off. Lower guidance = more natural wrinkles, folds, shadows.
            # Higher guidance = sharper edges, better color adherence.
            # 4-candidate spread: (0.90, 0.97, 1.03, 1.10) for wider diversity.
            if num_candidates > 1 and CANDIDATE_GUIDANCE_VARY:
                if num_candidates == 4:
                    gmult = (0.90, 0.97, 1.03, 1.10)[c]
                elif num_candidates == 3:
                    gmult = (0.93, 1.0, 1.07)[c]
                elif num_candidates == 2:
                    gmult = (0.93, 1.07)[c]
                else:
                    gmult = 1.0
                c_guidance = effective_guidance * gmult
            else:
                c_guidance = effective_guidance

            # ── Per-candidate denoising steps ───────────────────────────
            # Vary steps to give the scoring system a choice between
            # faster/coarser and slower/finer outputs. More steps = more
            # detail in wrinkles, folds, fabric texture.
            if num_candidates > 1 and CANDIDATE_STEPS_VARY:
                if num_candidates == 4:
                    sdel = (-8, -3, 3, 8)[c]
                elif num_candidates == 3:
                    sdel = (-5, 0, 5)[c]
                elif num_candidates == 2:
                    sdel = (-5, 5)[c]
                else:
                    sdel = 0
                c_steps = max(25, min(60, steps + sdel))
            else:
                c_steps = steps

            c_result, c_raw, c_meta = run_idm_vton_inference(
                person_img=person_img,
                garment_img=garment_img,
                garment_desc=garment_desc,
                garment_subtype=garment_subtype,
                cloth_type=vton_type,
                steps=c_steps,
                seed=c_seed,
                auto_crop=True,
                external_mask=external_mask,
                protected_mask=protected_mask,
                mask_strategy=strategy,
                mask_quality_score=mask_quality_score,
                guidance_scale=c_guidance,
                crop_preserve_lower=crop_preserve,
            )

            c_inpaint = c_meta.get("inpaint_mask_pil") or external_mask
            if c_inpaint is not None:
                c_qa_report = detect_inference_failures(
                    person_img.resize(TARGET_SIZE),
                    c_result.resize(TARGET_SIZE) if c_result.size != TARGET_SIZE else c_result,
                    c_inpaint,
                    protected_mask,
                    garment_ref=garment_img,
                )
                c_agg = compute_aggregate_quality_score(
                    c_qa_report,
                    c_result,
                    c_inpaint,
                    garment_img,
                )
            else:
                c_qa_report = None
                c_agg = {"aggregate_score": 50.0}

            candidates.append({
                "agg_score": c_agg["aggregate_score"],
                "result": c_result,
                "raw_output": c_raw,
                "meta": c_meta,
                "seed": c_seed,
                "steps": c_steps,
                "guidance": c_guidance,
                "qa_report": c_qa_report,
                "agg": c_agg,
            })

        # Select the best candidate
        candidates.sort(key=lambda x: x["agg_score"], reverse=True)
        best = candidates[0]
        result = best["result"]
        raw_output = best["raw_output"]
        mask_meta = best["meta"]
        best_seed = best["seed"]
        best_steps: int = best["steps"]
        best_guidance: float = best["guidance"]
        quality_report = best["qa_report"]
        best_agg = best["agg"]

        # Clean up internal-only key before it leaks to serialisation
        mask_meta.pop("inpaint_mask_pil", None)

        last_inpaint_mask = external_mask if strategy == "external" else None

        logger.info(
            "attempt_%d strategy=%s candidates=%d best_seed=%d "
            "best_guidance=%.3f best_steps=%d "
            "aggregate=%.1f identity=%.1f color=%.1f texture=%.1f artifact=%.1f "
            "trace_id=%s",
            attempt_idx, strategy, num_candidates, best_seed,
            best_guidance, best_steps,
            best_agg.get("aggregate_score", -1),
            best_agg.get("identity_score", -1),
            best_agg.get("color_fidelity_score", -1),
            best_agg.get("texture_score", -1),
            best_agg.get("artifact_score", -1),
            trace_id,
        )

        if attempt_idx == 0 and num_candidates > 1:
            spread = best_agg.get("aggregate_score", 0) - candidates[-1]["agg_score"]
            params = ", ".join(
                f"c{i}:s={c['seed']},g={c.get('guidance', 0):.2f},st={c.get('steps', 0)}"
                for i, c in enumerate(candidates)
            )
            logger.info(
                "multi_candidate_summary candidates=%d best=%.1f worst=%.1f "
                "spread=%.1f params=[%s]",
                num_candidates,
                best_agg.get("aggregate_score", -1),
                candidates[-1]["agg_score"],
                spread,
                params,
            )

        # ── Retry decision (same logic as before) ──────────────────────
        if not retry_enabled or attempt_idx >= len(retry_strategies) - 1:
            break

        inpaint_for_qa = external_mask if strategy == "external" else None
        if inpaint_for_qa is None:
            break

        if quality_report is None:
            quality_report = detect_inference_failures(
                person_img.resize(TARGET_SIZE),
                result.resize(TARGET_SIZE) if result.size != TARGET_SIZE else result,
                inpaint_for_qa,
                protected_mask,
                garment_ref=garment_img,
            )

        color_passed = True
        if quality_report.color_fidelity_score < 50.0:
            color_passed = False
            logger.warning(
                "color_fidelity_low attempt=%s score=%.1f garm_mean=%.1f retrying",
                attempt_idx, quality_report.color_fidelity_score, garm_mean_all,
            )

        if quality_report.passed and color_passed:
            break

        failure_reasons = list(quality_report.failure_reasons)
        if not color_passed:
            failure_reasons.append(f"color_fidelity:{quality_report.color_fidelity_score:.0f}")
        logger.warning(
            "inference_qa_failed attempt=%s strategy=%s reasons=%s retrying",
            attempt_idx,
            strategy,
            failure_reasons,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_ms = (time.perf_counter() - inference_start) * 1000

    # ── Color fidelity: computed via detect_inference_failures with garment_ref ──
    # CRITICAL: We pass garment_img as garment_ref so the metric compares the
    #           SOURCE GARMENT against the OUTPUT, not the original person's
    #           clothing against the output (which would be backwards).
    result_color_fidelity = quality_report.color_fidelity_score if quality_report else None
    result_color_drift = quality_report.color_drift_mean_rgb if quality_report else None
    if result_color_fidelity is None and result is not None and external_mask is not None:
        qa = detect_inference_failures(
            person_img.resize(TARGET_SIZE),
            result.resize(TARGET_SIZE) if result.size != TARGET_SIZE else result,
            external_mask,
            protected_mask,
            garment_ref=garment_img,
        )
        result_color_fidelity = qa.color_fidelity_score
        result_color_drift = qa.color_drift_mean_rgb

    # ── Post-processing pipeline ────────────────────────────────────────
    # Each stage is independently disableable and logs its own timing.
    from post_processing import (
        apply_face_composite,
        apply_seamless_clone,
        apply_skin_tone_correction,
    )
    pp_meta: dict[str, object] = {}

    face_meta: dict[str, object] = {}

    # ── Face bbox detection for skin-tone correction ───────────────────
    import cv2
    # Detect face bounding box from the ORIGINAL person image so that
    # skin-tone correction samples from the actual face region (which was
    # composited from the original), not the diffusion-modified neck area.
    # When face_bbox is provided, gain ≈ 1.0 (face matches original),
    # correction is skipped, and face color is preserved intact.
    person_cv = cv2.cvtColor(np.array(person_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    face_cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if not face_cascade.empty():
        gray = cv2.cvtColor(person_cv, cv2.COLOR_BGR2GRAY)
        h_i, w_i = gray.shape
        min_dim = max(30, int(min(w_i, h_i) * 0.04))
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(min_dim, min_dim))
        if len(faces) > 0:
            (fx, fy, fw, fh) = max(faces, key=lambda r: r[2] * r[3])
            face_bbox = (fx, fy, fx + fw, fy + fh)
            logger.info("skin_tone_face_bbox_detected bbox=(%d,%d,%d,%d)", fx, fy, fx + fw, fy + fh)
        else:
            face_bbox = None
    else:
        face_bbox = None

    # Stage A: Face composite — preserve original face/accessory pixels
    fc_start = time.perf_counter()
    result = apply_face_composite(result, person_img, protected_mask)
    fc_ms = (time.perf_counter() - fc_start) * 1000
    pp_meta["face_composite_ms"] = round(fc_ms, 1)

    # Stage B: Seamless clone — Poisson edge blending
    sc_start = time.perf_counter()
    seam_mask = last_inpaint_mask if last_inpaint_mask is not None else external_mask
    if seam_mask is not None:
        result = apply_seamless_clone(result, person_img, seam_mask)
    sc_ms = (time.perf_counter() - sc_start) * 1000
    pp_meta["seamless_clone_ms"] = round(sc_ms, 1)

    # Stage C: Skin tone correction
    st_start = time.perf_counter()
    result = apply_skin_tone_correction(result, person_img, face_bbox=face_bbox)
    st_ms = (time.perf_counter() - st_start) * 1000
    pp_meta["skin_tone_ms"] = round(st_ms, 1)

    # Stage D: Face restoration (GFPGAN or OpenCV enhancement) — runs LAST
    # so it enhances the composited original face rather than the
    # diffusion-generated face that will be overwritten by Stage A.
    if os.environ.get("ENABLE_FACE_RESTORATION", "1") == "1":
        from face_restoration import enhance_face
        face_start = time.perf_counter()
        result, face_meta = enhance_face(
            result,
            person_original=person_img,
            trace_id=trace_id,
        )
        face_ms = (time.perf_counter() - face_start) * 1000
        if face_meta.get("face_detected") is True:
            logger.info(
                "face_restoration_completed method=%s time_ms=%.0f trace_id=%s",
                face_meta.get("restoration_method", "unknown"),
                face_ms,
                trace_id,
            )
        else:
            logger.info(
                "face_restoration_skipped reason=no_face_detected time_ms=%.0f trace_id=%s",
                face_ms,
                trace_id,
            )
    pp_meta["face_restoration"] = face_meta

    # ── Debug: save all pipeline images ────────────────────────────────────
    if os.environ.get("ENABLE_DEBUG_IMAGES") == "1":
        debug_dir = f"/tmp/trylix_debug/{job_id}"
        try:
            os.makedirs(debug_dir, exist_ok=True)
            person_img.convert("RGB").save(f"{debug_dir}/01_person.jpg", quality=90)
            garment_img.convert("RGB").save(f"{debug_dir}/02_garment.jpg", quality=90)
            if external_mask is not None:
                external_mask.convert("L").save(f"{debug_dir}/03_external_mask.png")
            else:
                logger.warning("debug_no_external_mask trace_id=%s", trace_id)
            if protected_mask is not None:
                protected_mask.convert("L").save(f"{debug_dir}/04_protected_mask.png")
            raw_output.convert("RGB").save(f"{debug_dir}/06_raw_output.jpg", quality=95)
            result.convert("RGB").save(f"{debug_dir}/07_final_output.jpg", quality=95)
            logger.info(
                "debug_images_saved dir=%s files=6 trace_id=%s",
                debug_dir, trace_id,
            )
        except Exception as exc:
            logger.warning("debug_images_save_failed dir=%s error=%s trace_id=%s", debug_dir, exc, trace_id)

    # ── Upload ───────────────────────────────────────────────────────────
    upload_start = time.perf_counter()
    result_url = _upload_to_cloudinary(result, job_id)
    upload_ms = (time.perf_counter() - upload_start) * 1000

    total_ms = (time.perf_counter() - job_start) * 1000

    logger.info(
        "job_complete total_ms=%.0f download_ms=%.0f inference_ms=%.0f upload_ms=%.0f "
        "mask_type=%s retry_count=%s trace_id=%s",
        total_ms, download_ms, inference_ms, upload_ms,
        mask_meta.get("mask_type_used"),
        retry_count,
        trace_id,
    )

    return {
        "status": "success",
        "result_url": result_url,
        "cloth_type_used": vton_type,
        "steps_used": best_steps,
        "seed": best_seed,
        "inference_ms": round(inference_ms, 2),
        "upload_ms": round(upload_ms, 2),
        "download_ms": round(download_ms, 2),
        "total_ms": round(total_ms, 2),
        "mask_type_used": mask_meta.get("mask_type_used"),
        "mask_quality_score": mask_quality_score,
        "retry_count": retry_count,
        "failure_reasons": failure_reasons or None,
        "identity_drift_score": (
            quality_report.identity_drift_score if quality_report else None
        ),
        "color_fidelity_score": result_color_fidelity,
        "color_drift_mean_rgb": result_color_drift,
        "garment_mean_rgb": round(garm_mean_all, 1),
        "guidance_scale_used": round(best_guidance, 2),
        "pp_meta": pp_meta,
        "trace_id": trace_id,
        "warnings": input_warnings,
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
    with _REUSE_LOCK:
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
        logger.info("Starting RunPod serverless with max_workers=%s", MAX_WORKERS)
        runpod.serverless.start({"handler": handler})
    except Exception:
        logger.error("Worker startup failed")
        traceback.print_exc()
        sys.stdout.flush()
        raise
