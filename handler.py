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

# Post-inference quality recovery — face restoration module.
# enhanace_face() is called in run_inference() after inference completes.
# It applies mild sharpening to the face region (or GFPGAN if available)
# and blends back with Gaussian feathering. No original pixels are pasted
# over generated clothing — only the face region is touched.
_FACE_RESTORATION_AVAILABLE = True
try:
    from face_restoration import enhance_face as _do_enhance_face
except ImportError:
    _FACE_RESTORATION_AVAILABLE = False
    _do_enhance_face = None

# Post-inference quality validation and candidate scoring
_QUALITY_VALIDATION_AVAILABLE = True
try:
    from quality_validation import score_candidate as _score_candidate
except ImportError:
    _QUALITY_VALIDATION_AVAILABLE = False
    _score_candidate = None

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

# Retry / candidate scoring thresholds
MULTI_CANDIDATE_COUNT = int(os.environ.get("MULTI_CANDIDATE_COUNT", "1"))
CANDIDATE_MIN_SCORE = float(os.environ.get("CANDIDATE_MIN_SCORE", "0.55"))
CANDIDATE_GUIDANCE_VARY = os.environ.get("CANDIDATE_GUIDANCE_VARY", "1") == "1"
CANDIDATE_STEPS_VARY = os.environ.get("CANDIDATE_STEPS_VARY", "1") == "1"
RETRY_GUIDANCE_BOOST = float(os.environ.get("RETRY_GUIDANCE_BOOST", "0.15"))
RETRY_STEPS_BOOST = int(os.environ.get("RETRY_STEPS_BOOST", "5"))
FACE_RESTORATION_DEFAULT = os.environ.get("ENABLE_FACE_RESTORATION", "0") == "1"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16

# Memory/perf knobs
ENABLE_XFORMERS = os.environ.get("ENABLE_XFORMERS", "1") == "1"
ENABLE_TORCH_COMPILE = os.environ.get("ENABLE_TORCH_COMPILE", "0") == "1"
ENABLE_MODEL_CPU_OFFLOAD = os.environ.get("ENABLE_MODEL_CPU_OFFLOAD", "0") == "1"
ALLOW_TF32 = os.environ.get("ALLOW_TF32", "1") == "1"

# Post-processing:
#   face_restoration.py — opt-in only (ENABLE_FACE_RESTORATION=1, default off).
#                         Sharpens diffusion output face in-place; does not paste
#                         original person pixels (avoids halos / identity clash).

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
        build_final_inpaint_mask,
        assert_binary_mask,
        is_draped_garment,
        validate_mask_coverage,
        validate_mask_integrity,
        detect_inference_failures,
    )

    # Trust preprocessing canvas when garment is already at target resolution.
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
    already_target = human_img_orig.size == TARGET_SIZE

    if auto_crop and not already_target:
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
    elif already_target:
        human_img = human_img_orig.copy()
        logger.info("auto_crop_skipped image_already_target_size=%s", TARGET_SIZE)
    else:
        human_img = human_img_orig.resize(TARGET_SIZE)

    # SCHP is the single authoritative mask source.
    keypoints = openpose_model(human_img.resize((384, 512)))
    # SCHP at full TARGET_SIZE resolution so mask boundaries are native-res,
    # not interpolated from 384x512. The ONNX models internally affine-warp
    # to 512x512, so the compute cost is identical — only the output label
    # map resolution increases (1024x768 vs 512x384).
    model_parse, _ = parsing_model(human_img)
    # Build binary masks from SCHP labels
    schp_np = np.array(model_parse) if not isinstance(model_parse, np.ndarray) else model_parse
    if isinstance(model_parse, torch.Tensor):
        schp_np = model_parse.cpu().numpy()
    if schp_np.ndim == 3:
        schp_np = schp_np.squeeze(0)
    schp_np = schp_np.astype(np.uint8)

    final_mask_np, inpaint_mask_np, protect_mask_np = build_final_inpaint_mask(
        schp_np, cloth_type, garment_subtype,
    )
    draped = is_draped_garment(cloth_type, garment_subtype)
    assert_binary_mask(final_mask_np, "final_mask before inference")
    validate_mask_integrity(final_mask_np, "final_mask")
    # Smooth upscale: LANCZOS + threshold to 255/0 gives anti-aliased
    # mask boundaries instead of the jagged pixel blocks from NEAREST.
    final_mask = Image.fromarray(final_mask_np, mode="L")
    if final_mask.size != TARGET_SIZE:
        final_mask = final_mask.resize(TARGET_SIZE, Image.LANCZOS)
        final_mask = final_mask.point(lambda x: 255 if x > 127 else 0)
        assert_binary_mask(np.array(final_mask, dtype=np.uint8), "final_mask after resize")

    mask_meta: dict[str, object] = {
        "mask_type_used": "automasker",
        "coverage_valid": None,
        "coverage_percent": None,
        "schp_labels": schp_np,
        "protect_mask_np": protect_mask_np,
        "inpaint_mask_np": inpaint_mask_np,
        "final_mask_np": final_mask_np,
        "is_draped_garment": draped,
        "garment_subtype": garment_subtype,
    }

    # ── Pre-inference mask validation ──────────────────────────────────
    mask_v = validate_mask_coverage(final_mask, cloth_type)
    mask_meta["coverage_valid"] = mask_v["valid"]
    mask_meta["coverage_percent"] = mask_v["coverage_percent"]
    logger.info(
        "mask_coverage coverage=%.1f%% valid=%s cloth_type=%s",
        mask_v["coverage_percent"], mask_v["valid"], cloth_type,
    )
    if not mask_v["valid"]:
        logger.warning(
            "pre_inference_mask_invalid reason=%s coverage=%.1f%% cloth_type=%s",
            mask_v["reason"], mask_v["coverage_percent"], cloth_type,
        )

    mask = final_mask

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
        "dupatta": "sheer draped fabric, flowing dupatta, soft translucent textile, natural folds, pallu drape, delicate fabric sheen",
        "lehenga": "structured waistband, flowing skirt, realistic pleats, natural folds, embroidered border detail, waist seam",
        "tracksuit": "cotton fabric, sporty fit, soft folds, natural wrinkles, ribbed cuffs, drawstring waistband, leg zip detail",
        "co-ord": "matching fabric, coordinated fit, natural folds, realistic wrinkles, waist seam detail, hem finishing",
    }
    fabric_desc = _SUBTYPE_FABRIC.get(garment_subtype, "structured fabric, natural folds, realistic texture, sharp details, visible seams, natural shadows, fabric grain visible")
    prompt = (
        "model is wearing " + garment_desc + ", " + fabric_desc
        + ", photorealistic, sharp focus, fashion photography, "
        + "soft studio lighting, high quality, detailed fabric texture, "
        + "natural skin, professional photo, no accessories, "
        + "detailed face, natural facial features, symmetric face, "
        + "natural body proportions, natural hands, natural fingers"
    )
    negative_prompt = (
        "monochrome, lowres, bad anatomy, worst quality, low quality, "
        "deformed, distorted, disfigured, bad proportions, "
        "extra limbs, missing limbs, cloned head, body out of frame, "
        "poorly drawn face, mutation, mutated, extra fingers, "
        "ugly, blurry, watermark, signature, text, logo, "
        "smooth plastic, airbrushed, cg render, 3d render, "
        "flat lighting, "
        "watercolor, smudged, washed out, overlaid, transparent, "
        "see-through, ghost, double exposure, translucent, "
        "poorly sewn, unfinished edges, loose threads, "
        "fabric tearing, fabric distortion, fabric wrinkling, "
        "pattern mismatch, texture seam, visible seam, "
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
            # ── Input shape verification ──────────────────────────────────
            _person_sz = human_img.size
            _garment_sz = garm_img.size
            _mask_sz = mask.size
            _pose_sz = pose_img.size
            _ip_sz = garm_img.resize(TARGET_SIZE).size
            _all_ok = (
                _person_sz == TARGET_SIZE
                and _garment_sz == TARGET_SIZE
                and _mask_sz == TARGET_SIZE
                and _pose_sz == TARGET_SIZE
                and _ip_sz == TARGET_SIZE
            )
            if not _all_ok:
                logger.error(
                    "SHAPE_MISMATCH person=%s garment=%s mask=%s pose=%s "
                    "ip_adapter=%s expected=%s",
                    _person_sz, _garment_sz, _mask_sz, _pose_sz, _ip_sz, TARGET_SIZE,
                )
            logger.info(
                "INPUT_SHAPES person=%s garment=%s mask=%s pose=%s "
                "openpose=384x512 ip_adapter=%s all_ok=%s",
                _person_sz, _garment_sz, _mask_sz, _pose_sz, _ip_sz, _all_ok,
            )
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
        is_draped_garment,
        validate_mask_coverage,
        InferenceQualityReport,
    )

    job_start = time.perf_counter()

    person_url = job_input.get("person_image_url") or job_input.get("person_image")
    garment_url = job_input.get("garment_image_url") or job_input.get("garment_image")
    garment_desc = job_input.get("garment_desc") or job_input.get("garment_description") or "garment"
    garment_subtype = job_input.get("garment_subtype") or ""
    cloth_type = job_input.get("cloth_type", "upper_body")
    steps = int(job_input.get("steps", DENOISE_STEPS))
    seed = int(job_input.get("seed", random.randint(0, 2**31 - 1)))
    trace_id = job_input.get("trace_id", "")
    # Forward preprocessing warnings through the output so the frontend
    # can display them to the user (body completeness, face, etc.)
    input_warnings = job_input.get("warnings", [])
    if not isinstance(input_warnings, list):
        input_warnings = []
    max_retries = int(os.environ.get("MASK_WORKER_MAX_RETRIES", "0"))
    retry_enabled = os.environ.get("MASK_WORKER_RETRY", "0") == "1"

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
        "draped": "dresses",
        "saree": "dresses",
        "sari": "dresses",
        "dupatta": "dresses",
        "lehenga": "dresses",
        "lehanga": "dresses",
        "anarkali": "dresses",
        "ethnic": "dresses",
        "gown": "dresses",
        "gowns": "dresses",
        "jumpsuit": "dresses",
        "jumpsuits": "dresses",
        "kurta": "dresses",
        "kurti": "dresses",
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

    inference_start = time.perf_counter()
    result: Image.Image | None = None
    raw_output: Image.Image | None = None
    mask_meta: dict[str, object] = {}
    quality_report = None
    retry_count = 0
    failure_reasons: list[str] = []
    best_candidate_score: float = -1.0

    # Dark garment + draped/full-body: stronger diffusion for full replacement.
    effective_guidance = GUIDANCE_SCALE
    effective_steps = steps
    draped_request = is_draped_garment(vton_type, garment_subtype)
    if draped_request or vton_type in ("dresses", "full_body"):
        effective_steps = max(steps, int(os.environ.get("IDM_VTON_DRESS_STEPS", "50")))
        effective_guidance = max(effective_guidance, float(os.environ.get("IDM_VTON_DRESS_GUIDANCE", "3.1")))
    if garm_mean_all < 80.0:
        effective_guidance = GUIDANCE_SCALE * 1.15
        logger.info(
            "dark_garment_detected mean_r=%.1f mean_g=%.1f mean_b=%.1f "
            "increasing_guidance from %.1f to %.1f",
            garm_mean_r, garm_mean_g, garm_mean_b,
            GUIDANCE_SCALE, effective_guidance,
        )

    min_candidate_score = CANDIDATE_MIN_SCORE
    candidate_count = max(1, MULTI_CANDIDATE_COUNT)
    max_retry_rounds = max(0, max_retries) if retry_enabled else 0

    retry_round = 0
    while retry_round <= max_retry_rounds:
        candidates: list[tuple[Image.Image, Image.Image | None, dict[str, object], object | None]] = []
        for ci in range(candidate_count):
            c_seed = seed + retry_round * 10000 + ci * 1000

            # Base params for this retry round (escalates guidance/steps on retry)
            c_guidance = effective_guidance * max(1.0, 1.0 + RETRY_GUIDANCE_BOOST * retry_round)
            c_steps = effective_steps + RETRY_STEPS_BOOST * retry_round

            # Candidate diversity: vary guidance/steps around the retry-round base
            if CANDIDATE_GUIDANCE_VARY and candidate_count > 1:
                variation = 1.0 + 0.08 * (ci - (candidate_count - 1) / 2.0)
                c_guidance *= variation
            if CANDIDATE_STEPS_VARY and candidate_count > 1:
                c_steps += int(5 * (ci - (candidate_count - 1) / 2.0))
                c_steps = max(10, c_steps)

            c_result, c_raw, c_meta = run_idm_vton_inference(
                person_img=person_img,
                garment_img=garment_img,
                garment_desc=garment_desc,
                garment_subtype=garment_subtype,
                cloth_type=vton_type,
                steps=c_steps,
                seed=c_seed,
                auto_crop=True,
                guidance_scale=c_guidance,
                crop_preserve_lower=True,
            )

            # Validate + score candidate
            c_final_mask_np = c_meta.get("final_mask_np")
            c_protect_np = c_meta.get("protect_mask_np")
            c_schp_labels = c_meta.get("schp_labels")
            c_vresult = None
            if _QUALITY_VALIDATION_AVAILABLE and _score_candidate is not None:
                c_vresult = _score_candidate(
                    person_img.resize(TARGET_SIZE) if person_img.size != TARGET_SIZE else person_img,
                    c_result.resize(TARGET_SIZE) if c_result.size != TARGET_SIZE else c_result,
                    garment_img,
                    mask_np=c_final_mask_np,
                    protect_np=c_protect_np,
                    schp_labels=c_schp_labels,
                )
                logger.info(
                    "candidate_ci=%d score=%.4f face=%.4f garment=%.4f "
                    "sharpness=%.2f drift=%.1f replacement=%.4f trace_id=%s",
                    ci,
                    c_vresult.score, c_vresult.face_quality, c_vresult.garment_quality,
                    c_vresult.sharpness, c_vresult.identity_drift,
                    c_vresult.garment_replacement, trace_id,
                )
            candidates.append((c_result, c_raw, c_meta, c_vresult))

        # ── Pick best candidate ────────────────────────────────────────
        scored = [(r, ro, m, v) for r, ro, m, v in candidates if v is not None]
        if scored:
            best = max(scored, key=lambda x: x[3].score)
            result, raw_output, mask_meta, vresult = best
            quality_report = InferenceQualityReport(
                passed=vresult.passed,
                identity_drift_score=vresult.identity_drift,
                failure_reasons=tuple(vresult.failure_reasons),
            )
            best_candidate_score = vresult.score
            failure_reasons = vresult.failure_reasons
        elif candidates:
            # Fallback: first candidate if scoring unavailable
            result, raw_output, mask_meta = candidates[0][:3]
            vresult = None
            quality_report = InferenceQualityReport(
                passed=True, identity_drift_score=0.0, failure_reasons=(),
            )
            best_candidate_score = 0.0
        else:
            raise RuntimeError("No candidates generated — inference produced no output")

        retry_count = retry_round

        # ── Mask coverage validation (for output metadata) ─────────────
        final_mask_np = mask_meta.get("final_mask_np")
        protect_mask_np = mask_meta.get("protect_mask_np")
        if final_mask_np is not None:
            final_mask_img = Image.fromarray(final_mask_np, mode="L")
            if final_mask_img.size != TARGET_SIZE:
                final_mask_img = final_mask_img.resize(TARGET_SIZE, Image.NEAREST)
            coverage = validate_mask_coverage(final_mask_img, vton_type)
            mask_meta["coverage_valid"] = coverage.get("valid")
            mask_meta["coverage_percent"] = coverage.get("coverage_percent")

        logger.info(
            "candidate_selection round=%d candidates=%d best_score=%.4f "
            "passed=%s trace_id=%s",
            retry_round, len(candidates), best_candidate_score,
            quality_report.passed if quality_report else False,
            trace_id,
        )

        # ── Decide whether to retry ───────────────────────────────────
        if vresult is None or (vresult.passed and vresult.score >= min_candidate_score):
            break
        if retry_round >= max_retry_rounds:
            break

        logger.warning(
            "retry_round=%d score=%.4f reasons=%s next_guidance=%.2f next_steps=%d",
            retry_round, best_candidate_score, failure_reasons,
            effective_guidance * (1.0 + RETRY_GUIDANCE_BOOST * (retry_round + 1)),
            effective_steps + RETRY_STEPS_BOOST * (retry_round + 1),
        )
        retry_round += 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_ms = (time.perf_counter() - inference_start) * 1000

    # ── Face restoration — mild enhancement, no identity overwrite ────────
    # Enahnces the face region in the diffusion output using mild sharpening
    # or GFPGAN.  Sources the face from the original person image and never
    # pastes original pixels over generated clothing.
    face_restore_enabled = os.environ.get("ENABLE_FACE_RESTORATION", "1") == "1"
    if (
        face_restore_enabled
        and result is not None
        and _FACE_RESTORATION_AVAILABLE
        and _do_enhance_face is not None
    ):
        person_ref = person_img
        if person_ref.size != result.size:
            person_ref = person_ref.resize(result.size, Image.LANCZOS)
        result, face_meta_out = _do_enhance_face(result, person_original=person_ref)
        logger.info(
            "face_restoration_applied face_detected=%s trace_id=%s",
            face_meta_out.get("face_detected", "unknown"), trace_id,
        )
    else:
        logger.info("face_restoration_skipped available=%s trace_id=%s",
                     _FACE_RESTORATION_AVAILABLE, trace_id)

    # ── Upload ───────────────────────────────────────────────────────────
    upload_start = time.perf_counter()
    result_url = _upload_to_cloudinary(result, job_id)
    upload_ms = (time.perf_counter() - upload_start) * 1000

    total_ms = (time.perf_counter() - job_start) * 1000

    logger.info(
        "job_complete total_ms=%.0f download_ms=%.0f inference_ms=%.0f upload_ms=%.0f "
        "retry_count=%s trace_id=%s",
        total_ms, download_ms, inference_ms, upload_ms,
        retry_count,
        trace_id,
    )

    return {
        "status": "success",
        "result_url": result_url,
        "cloth_type_used": vton_type,
        "inference_ms": round(inference_ms, 2),
        "upload_ms": round(upload_ms, 2),
        "download_ms": round(download_ms, 2),
        "total_ms": round(total_ms, 2),
        "mask_coverage_percent": mask_meta.get("coverage_percent"),
        "mask_coverage_valid": mask_meta.get("coverage_valid"),
        "retry_count": retry_count,
        "failure_reasons": failure_reasons or None,
        "identity_drift_score": (
            quality_report.identity_drift_score if quality_report else None
        ),
        "garment_mean_rgb": round(garm_mean_all, 1),
        "guidance_scale_used": round(effective_guidance, 2),
        "best_candidate_score": round(best_candidate_score, 4),
        "candidate_count": candidate_count,
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
            build_schp_inpaint_mask,
            build_schp_protect_mask,
            apply_protection_binary,
            assert_binary_mask,
            validate_mask_coverage,
            detect_inference_failures,
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
