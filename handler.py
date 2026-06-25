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
from PIL import Image, ImageDraw, ImageFilter
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

DENOISE_STEPS = int(os.environ.get("IDM_VTON_STEPS", "50"))
GUIDANCE_SCALE = float(os.environ.get("IDM_VTON_GUIDANCE", "2.5"))

# Cross-category two-stage pipeline constants
NEUTRAL_GARMENT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "neutral_garment.png"
)
CROSS_CATEGORY_ERASE_STEPS = int(os.environ.get("CROSS_CATEGORY_ERASE_STEPS", "20"))
CROSS_CATEGORY_ERASE_GUIDANCE = float(os.environ.get("CROSS_CATEGORY_ERASE_GUIDANCE", "3.5"))

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
# Cross-category two-stage pipeline
# =============================================================================


def _generate_neutral_garment(target_size: tuple[int, int] = TARGET_SIZE) -> Image.Image:
    """Generate a plain neutral undergarment reference image for erase stage.

    Creates a simple beige/cream tank top + shorts silhouette on white canvas
    using only PIL primitives.  The image is deliberately low-detail — the
    IP-Adapter receives a colour-and-shape reference with no texture, so the
    text prompt (high guidance) dominates the erase generation.

    Result is cached to disk so it only renders once per worker lifetime.
    """
    cached = Path(NEUTRAL_GARMENT_PATH)
    if cached.exists():
        return Image.open(cached).convert("RGB")

    w, h = target_size
    cx = w // 2  # 384

    canvas = Image.new("RGB", target_size, (255, 255, 255))
    pixels = canvas.load()

    # Neutral beige/cream — visually reads as "plain undergarment"
    nr, ng, nb = 226, 210, 190

    for y in range(h):
        for x in range(w):
            dx = abs(x - cx)
            # Tank top: ~shoulder to waist
            if 90 <= y <= 500:
                # Shoulders wider, tapers to waist
                half_w = int(200 - (y - 90) * 0.15)
                if dx <= half_w:
                    pixels[x, y] = (nr, ng, nb)
            # Shorts: ~hips to mid-thigh
            if 470 <= y <= 760:
                half_w = 155
                if dx <= half_w:
                    pixels[x, y] = (nr, ng, nb)

    # Soften edges so IP-Adapter doesn't try to reproduce hard polygon boundaries
    result = canvas.filter(ImageFilter.GaussianBlur(radius=6))

    try:
        cached.parent.mkdir(parents=True, exist_ok=True)
        result.save(cached, format="PNG")
        logger.info("neutral_garment_cached path=%s", cached)
    except Exception as exc:
        logger.warning("neutral_garment_cache_failed error=%s", exc)

    return result


def detect_cross_category(
    person_img: Image.Image,
    vton_type: str,
    trace_id: str = "",
) -> bool:
    """Check whether two-stage inference is needed for this person+garment pair.

    Runs a quick SCHP parse on the person image at target resolution, then
    compares the detected garment labels against the target cloth_type's
    editable label set.

    Returns True when significant garment-label area falls outside the target
    mask — i.e. the old garment would survive a single-stage try-on.
    """
    from mask_pipeline import needs_two_stage

    try:
        if parsing_model is None:
            logger.info("cross_category_skip parsing_model_not_loaded trace_id=%s", trace_id)
            return False

        det_img = person_img.resize(TARGET_SIZE)
        det_parse, _ = parsing_model(det_img)
        det_schp = np.array(det_parse, dtype=np.uint8)
        if det_schp.ndim == 3:
            det_schp = det_schp.squeeze(0)
        det_schp = det_schp.astype(np.uint8)

        needed = needs_two_stage(det_schp, vton_type)

        present_labels = sorted(int(x) for x in np.unique(det_schp) if x > 0)
        logger.info(
            "cross_category_detection needed=%s vton_type=%s "
            "person_garment_labels=%s trace_id=%s",
            needed, vton_type, present_labels, trace_id,
        )
        return needed
    except Exception as exc:
        logger.warning("cross_category_detection_failed error=%s trace_id=%s", exc, trace_id)
        return False


def run_cross_category_inference(
    person_img: Image.Image,
    garment_img: Image.Image,
    garment_desc: str,
    cloth_type: str,
    garment_subtype: str = "",
    steps: int = 30,
    seed: int = 42,
    guidance_scale: float | None = None,
    trace_id: str = "",
    source_cloth_type: str = "",
) -> tuple[Image.Image, Image.Image | None, dict[str, object]]:
    """Two-stage cross-category try-on.

    Stage 1 — Erase:  broad full-body inpaint that removes the old garment
                       and replaces it with a neutral plain underlayer.
                       Uses 30 steps + guidance 4.0 for complete erasing.
                       The positive prompt is overridden to remove face/body
                       terms that cause identity drift during erase.

    Stage 2 — Apply:   normal try-on inference using the actual target garment
                       on the Stage-1 erased body.  Standard steps/guidance.

    Returns (final_result, raw_output, mask_meta_from_stage2).
    """
    erase_steps = int(os.environ.get("CROSS_CATEGORY_ERASE_STEPS", "50"))
    erase_guidance = float(os.environ.get("CROSS_CATEGORY_ERASE_GUIDANCE", "5.5"))

    logger.info(
        "cross_category_stage1_erase_start cloth_type=%s "
        "steps=%d guidance=%.2f trace_id=%s",
        cloth_type, erase_steps,
        erase_guidance, trace_id,
    )

    # ── Stage 1: erase old garment ────────────────────────────────────
    neutral = _generate_neutral_garment()
    stage1_desc = (
        "plain beige seamless tank top and shorts, "
        "simple neutral undergarments, solid fabric, "
        "minimal basic clothing"
    )
    stage1_positive = (
        "model wearing plain simple beige tank top and shorts, "
        "solid neutral undergarments"
    )
    stage1_negative = (
        "saree, drape, pallu, dupatta, scarf, shawl, lehenga, "
        "pattern, print, design, logo, text, embroidery, lace, trim, "
        "original clothing, old garment, previous outfit, residual fabric, "
        "detailed fabric, textured, woven, striped, plaid, floral, "
        "worst quality, low quality, deformed, distorted, "
        "bad anatomy, bad proportions, extra limbs, missing limbs, "
        "ugly, blurry, watermark, signature, "
        "smooth plastic, airbrushed, cg render, "
        "recoloring, double garment, layered clothing"
    )

    erased_person, erased_raw, stage1_meta = run_idm_vton_inference(
        person_img=person_img,
        garment_img=neutral,
        garment_desc=stage1_desc,
        cloth_type="dresses",
        garment_subtype="saree",  # triggers draped mode → arm labels in mask
        steps=erase_steps,
        seed=seed,
        guidance_scale=erase_guidance,
        auto_crop=True,
        crop_preserve_lower=True,
        override_prompt=stage1_positive,
        override_negative_prompt=stage1_negative,
        source_cloth_type=source_cloth_type,
    )

    logger.info(
        "cross_category_stage1_complete trace_id=%s "
        "erased_person_size=%s",
        trace_id, erased_person.size,
    )

    # ── Stage 2: apply target garment ─────────────────────────────────
    # Use standard guidance on erased body — avoid over-guiding which causes identity drift
    stage2_guidance = guidance_scale if guidance_scale is not None else GUIDANCE_SCALE

    result, raw_output, mask_meta = run_idm_vton_inference(
        person_img=erased_person,
        garment_img=garment_img,
        garment_desc=garment_desc,
        cloth_type=cloth_type,
        garment_subtype=garment_subtype,
        steps=steps,
        seed=seed + 1,  # different seed for diversity from erase stage
        guidance_scale=stage2_guidance,
        auto_crop=True,
        crop_preserve_lower=True,
        source_cloth_type=source_cloth_type,
    )

    logger.info(
        "cross_category_stage2_complete trace_id=%s",
        trace_id,
    )

    # ── Debug saves ───────────────────────────────────────────────────
    if trace_id:
        _debug_dir = Path("/tmp/idm-vton-debug")
        _debug_dir.mkdir(parents=True, exist_ok=True)
        try:
            erased_person.save(str(_debug_dir / f"cross_cat_stage1_person_{trace_id}.png"))
            if erased_raw is not None:
                erased_raw.save(str(_debug_dir / f"cross_cat_stage1_raw_{trace_id}.png"))
            s1_final = stage1_meta.get("final_mask_np")
            if s1_final is not None:
                Image.fromarray(s1_final, mode="L").save(
                    str(_debug_dir / f"cross_cat_stage1_mask_{trace_id}.png")
                )
            if raw_output is not None:
                raw_output.save(str(_debug_dir / f"cross_cat_stage2_raw_{trace_id}.png"))
            s2_final = mask_meta.get("final_mask_np")
            if s2_final is not None:
                Image.fromarray(s2_final, mode="L").save(
                    str(_debug_dir / f"cross_cat_stage2_mask_{trace_id}.png")
                )
            result.save(str(_debug_dir / f"cross_cat_final_{trace_id}.png"))
            logger.info("cross_category_debug_saved trace_id=%s", trace_id)
        except Exception as exc:
            logger.warning("cross_category_debug_save_failed error=%s trace_id=%s", exc, trace_id)

    return result, raw_output, mask_meta


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


# ── Subtype-aware prompt construction ────────────────────────────────
# Different garment families need different prompt guidance to help the
# model generate the correct geometry, fit, and structure.

_GARMENT_PROMPT_ATTRS: dict[str, dict[str, str]] = {
    # Upper body: short / fitted
    "tshirt":     {"fit": "fitted", "sleeves": "short sleeves", "structure": "casual"},
    "t_shirt":    {"fit": "fitted", "sleeves": "short sleeves", "structure": "casual"},
    "polo":       {"fit": "fitted", "sleeves": "short sleeves", "structure": "collar with buttons"},
    "shirt":      {"fit": "regular fit", "sleeves": "long sleeves", "structure": "collar with button front"},
    "blouse":     {"fit": "regular fit", "sleeves": "long sleeves", "structure": "feminine collar"},
    "sweatshirt": {"fit": "relaxed fit", "sleeves": "long sleeves", "structure": "crew neck pullover"},
    "sports_jersey": {"fit": "loose fit", "sleeves": "short sleeves", "structure": "athletic"},
    # Upper body: sleeveless
    "tank_top":   {"fit": "fitted", "sleeves": "sleeveless", "structure": "thin straps"},
    "crop_top":   {"fit": "fitted", "sleeves": "short or sleeveless", "structure": "cropped at midriff"},
    "camisole":   {"fit": "fitted", "sleeves": "sleeveless", "structure": "thin straps"},
    "vest":       {"fit": "fitted", "sleeves": "sleeveless", "structure": "open front"},
    "corset":     {"fit": "tight fitted", "sleeves": "sleeveless", "structure": "boned structured"},
    # Upper body: extended
    "sweater":    {"fit": "relaxed fit", "sleeves": "long sleeves", "structure": "knit pullover"},
    "hoodie":     {"fit": "relaxed fit", "sleeves": "long sleeves", "structure": "hood with front pocket"},
    "jacket":     {"fit": "structured fit", "sleeves": "long sleeves", "structure": "zip or button front, collar, extends below waist"},
    "blazer":     {"fit": "structured fit", "sleeves": "long sleeves", "structure": "notched lapels, button front, extends below waist"},
    "coat":       {"fit": "structured fit", "sleeves": "long sleeves", "structure": "long to knees, button front, collar"},
    "cardigan":   {"fit": "relaxed fit", "sleeves": "long sleeves", "structure": "open front knit"},
    "leather_jacket": {"fit": "structured fit", "sleeves": "long sleeves", "structure": "zip front, leather texture"},
    "denim_jacket": {"fit": "structured fit", "sleeves": "long sleeves", "structure": "button front, denim texture"},
    # Upper body: wide
    "poncho":     {"fit": "loose draped", "sleeves": "sleeveless", "structure": "drapes over shoulders, wide"},
    "cape":       {"fit": "loose draped", "sleeves": "sleeveless", "structure": "drapes over shoulders, open front"},
    "shrug":      {"fit": "fitted", "sleeves": "short sleeves", "structure": "bolero style, cropped"},
    # Lower body
    "jeans":      {"fit": "regular fit", "structure": "denim, two legs, button fly"},
    "trousers":   {"fit": "regular fit", "structure": "formal, creased, two legs"},
    "pants":      {"fit": "regular fit", "structure": "casual, two legs"},
    "shorts":     {"fit": "regular fit", "structure": "above knee, two legs"},
    "skirt":      {"fit": "regular fit", "structure": "no leg separation"},
    "leggings":   {"fit": "tight fitted", "structure": "stretchy, body-hugging"},
    "joggers":    {"fit": "relaxed fit", "structure": "elastic waist, tapered leg"},
    "wide_leg":   {"fit": "loose wide", "structure": "wide from hip to hem"},
    "palazzo":    {"fit": "very loose wide", "structure": "very wide flowing leg"},
    "dhoti_pants": {"fit": "draped loose", "structure": "wrapped, pleated"},
    # Full body
    "dress":      {"fit": "regular fit", "structure": "one piece"},
    "mini_dress": {"fit": "regular fit", "structure": "above knee"},
    "midi_dress": {"fit": "regular fit", "structure": "below knee"},
    "maxi_dress": {"fit": "regular fit", "structure": "ankle length"},
    "bodycon":    {"fit": "tight fitted", "structure": "body-hugging, stretchy"},
    "a_line":     {"fit": "fitted bodice", "structure": "flared from waist"},
    "jumpsuit":   {"fit": "regular fit", "structure": "one piece with pants"},
    "evening_gown": {"fit": "elegant fitted", "structure": "floor length, formal"},
    "ball_gown":  {"fit": "fitted bodice", "structure": "voluminous skirt, floor length"},
    "wedding":    {"fit": "elegant fitted", "structure": "white, floor length, formal"},
    "wrap_dress": {"fit": "regular fit", "structure": "wrap closure, tied at waist"},
    "off_shoulder": {"fit": "regular fit", "structure": "exposed shoulders, neckline below shoulders"},
    "one_shoulder": {"fit": "regular fit", "structure": "one shoulder strap"},
    "strap":      {"fit": "fitted", "structure": "thin straps"},
    # Traditional
    "saree":      {"fit": "draped", "structure": "pallu over shoulder, wrapped around body"},
    "sari":       {"fit": "draped", "structure": "pallu over shoulder, wrapped around body"},
    "lehenga":    {"fit": "fitted bodice", "structure": "flared skirt with dupatta"},
    "anarkali":   {"fit": "fitted bodice", "structure": "flared from waist, long"},
    "salwar_suit": {"fit": "regular fit", "structure": "tunic with pants"},
    "kurti":      {"fit": "regular fit", "structure": "tunic to hips"},
    "kurta":      {"fit": "regular fit", "structure": "tunic to hips, collar"},
    "sherwani":   {"fit": "structured fit", "structure": "long to knees, formal, embroidered"},
    "abaya":      {"fit": "loose flowing", "structure": "full body, loose, modest"},
    "kaftan":     {"fit": "loose flowing", "structure": "tunic, wide sleeves, loose"},
    "kimono":     {"fit": "loose draped", "structure": "wide sleeves, wrap front, to ankles"},
    "hanbok":     {"fit": "fitted bodice", "structure": "jacket with high waist skirt"},
    "cheongsam":  {"fit": "tight fitted", "structure": "mandarin collar, side slit"},
    "qipao":      {"fit": "tight fitted", "structure": "mandarin collar, side slit"},
    "dhoti":      {"fit": "draped loose", "structure": "wrapped around waist and legs"},
    "lungi":      {"fit": "draped loose", "structure": "wrapped around waist"},
}

# Source garment-specific negative terms to prevent source residual
_SOURCE_NEGATIVES: dict[str, list[str]] = {
    "saree":   ["saree drape", "pallu", "arm drape", "wrapped fabric", "blouse sleeves"],
    "sari":    ["saree drape", "pallu", "arm drape", "wrapped fabric", "blouse sleeves"],
    "lehenga": ["lehenga skirt", "dupatta drape", "embroidered border"],
    "dupatta": ["dupatta drape", "shawl drape", "arm drape"],
    "jacket":  ["lapels", "zipper", "button placket", "cuff"],
    "blazer":  ["notched lapels", "button front", "structured shoulder"],
    "coat":    ["long coat hem", "coat collar", "coat buttons"],
    "hoodie":  ["hood", "kangaroo pocket", "drawstring"],
    "shirt":   ["collar", "button placket", "cuff"],
    "kimono":  ["wide sleeves", "wrap closure", "obi belt"],
    "abaya":   ["loose flowing fabric", "wide sleeves"],
    "kaftan":  ["wide sleeves", "loose tunic"],
    "sherwani": ["embroidery", "long hem", "mandarin collar"],
}


def _build_subtype_aware_prompt(garment_desc: str, garment_subtype: str = "") -> str:
    """Build a prompt that describes the target garment's fit, sleeves, and structure."""
    key = (garment_subtype or "").strip().lower().replace(" ", "_").replace("-", "_")
    attrs = _GARMENT_PROMPT_ATTRS.get(key, {})
    if not attrs:
        # Fuzzy match — prefer longest/most-specific match
        best_len = 0
        for geo_key, geo_val in _GARMENT_PROMPT_ATTRS.items():
            if key and geo_key in key and len(geo_key) > best_len:
                attrs = geo_val
                best_len = len(geo_key)
        if not attrs:
            best_len = 0
            for geo_key, geo_val in _GARMENT_PROMPT_ATTRS.items():
                if key and key in geo_key and len(geo_key) > best_len:
                    attrs = geo_val
                    best_len = len(geo_key)

    parts = ["model wearing " + garment_desc]
    if "fit" in attrs:
        parts.append(attrs["fit"])
    if "sleeves" in attrs:
        parts.append(attrs["sleeves"])
    if "structure" in attrs:
        parts.append(attrs["structure"])
    parts.append("detailed fabric texture, natural garment folds")
    return ", ".join(parts)


def _build_source_specific_negative(
    source_cloth_type: str = "",
    target_subtype: str = "",
) -> str:
    """Build negative prompt that suppresses source garment residual."""
    base = (
        "worst quality, low quality, "
        "ugly, blurry, watermark, signature, text, logo"
    )

    # Add source-specific negatives to prevent old garment from bleeding through
    key = (source_cloth_type or "").strip().lower().replace(" ", "_").replace("-", "_")
    source_terms = _SOURCE_NEGATIVES.get(key, [])
    if not source_terms:
        # Fuzzy match — prefer longest/most-specific match
        best_len = 0
        for src_key, src_terms in _SOURCE_NEGATIVES.items():
            if key and src_key in key and len(src_key) > best_len:
                source_terms = src_terms
                best_len = len(src_key)

    # Also add generic garment-transition negatives
    transition_terms = ["original clothing", "old garment", "residual fabric"]

    all_terms = source_terms + transition_terms
    if all_terms:
        base += ", " + ", ".join(all_terms)

    return base


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
    override_prompt: str | None = None,
    override_negative_prompt: str | None = None,
    source_cloth_type: str = "",
) -> tuple[Image.Image, Image.Image, dict[str, object]]:
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
        schp_np, cloth_type, garment_subtype, source_cloth_type=source_cloth_type,
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

    if override_prompt is not None:
        prompt = override_prompt
    else:
        prompt = _build_subtype_aware_prompt(garment_desc, garment_subtype)

    if override_negative_prompt is not None:
        negative_prompt = override_negative_prompt
    else:
        negative_prompt = _build_source_specific_negative(source_cloth_type, garment_subtype)

    with torch.inference_mode():
        with _maybe_autocast():
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            prompt_c = "a photo of " + garment_desc + ", detailed fabric texture, natural folds"
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
        detect_source_cloth_type,
    )

    job_start = time.perf_counter()

    person_url = job_input.get("person_image_url") or job_input.get("person_image")
    garment_url = job_input.get("garment_image_url") or job_input.get("garment_image")
    garment_desc = job_input.get("garment_desc") or job_input.get("garment_description") or "garment"
    garment_subtype = job_input.get("garment_subtype") or ""
    cloth_type = job_input.get("cloth_type", "upper_body")
    steps = int(job_input.get("steps", DENOISE_STEPS))
    seed = int(job_input.get("seed", random.randint(0, 2**31 - 1)))
    guidance_scale_input = job_input.get("guidance_scale")
    guidance_scale = float(guidance_scale_input) if guidance_scale_input is not None else None
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

    result: Image.Image | None = None
    raw_output: Image.Image | None = None
    mask_meta: dict[str, object] = {}
    quality_report = None
    retry_count = 0
    failure_reasons: list[str] = []
    best_candidate_score: float = -1.0

    # ── Detect source garment type from SCHP ────────────────────────────
    # Run SCHP on person image to determine what the person is currently wearing.
    # This enables target-aware mask expansion: when target covers different
    # body regions than source, the mask expands to include them.
    source_cloth_type = ""
    try:
        det_img = person_img.convert("RGB").resize(TARGET_SIZE)
        det_parse, _ = parsing_model(det_img)
        det_schp = np.array(det_parse) if not isinstance(det_parse, np.ndarray) else det_parse
        if isinstance(det_parse, torch.Tensor):
            det_schp = det_parse.cpu().numpy()
        if det_schp.ndim == 3:
            det_schp = det_schp.squeeze(0)
        det_schp = det_schp.astype(np.uint8)
        source_cloth_type = detect_source_cloth_type(det_schp)
        logger.info(
            "source_cloth_type_detected source=%s target=%s trace_id=%s",
            source_cloth_type, vton_type, trace_id,
        )
    except Exception as exc:
        logger.warning("source_cloth_type_detection_failed error=%s trace_id=%s", exc, trace_id)
        source_cloth_type = ""

    # ── Cross-category detection —───────────────────────────────────────
    # If the person's current garment covers label categories outside the
    # target cloth_type's editable mask, two-stage erase+apply is needed.
    # Same-category swaps (shirt→shirt, dress→dress) skip this path entirely.
    cross_category_needed = detect_cross_category(person_img, vton_type, trace_id)

    if cross_category_needed:
        logger.info(
            "cross_category_routing_two_stage vton_type=%s "
            "person_img_size=%s garment_desc=%s trace_id=%s",
            vton_type, person_img.size, garment_desc, trace_id,
        )
        inference_start = time.perf_counter()
        result, raw_output, mask_meta = run_cross_category_inference(
            person_img=person_img,
            garment_img=garment_img,
            garment_desc=garment_desc,
            cloth_type=vton_type,
            garment_subtype=garment_subtype,
            steps=steps,
            seed=seed,
            guidance_scale=guidance_scale,
            trace_id=trace_id,
            source_cloth_type=source_cloth_type,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - inference_start) * 1000

        # Score cross-category output with garment-aware weights
        quality_report = InferenceQualityReport(
            passed=True, identity_drift_score=0.0, failure_reasons=(),
        )
        retry_count = 0
        best_candidate_score = 0.0
        candidate_count = 1
        failure_reasons = []
        effective_guidance = guidance_scale if guidance_scale is not None else GUIDANCE_SCALE
        draped_request = False

        if _QUALITY_VALIDATION_AVAILABLE and _score_candidate is not None:
            try:
                cc_vresult = _score_candidate(
                    person_img.resize(TARGET_SIZE) if person_img.size != TARGET_SIZE else person_img,
                    result.resize(TARGET_SIZE) if result.size != TARGET_SIZE else result,
                    garment_img,
                    mask_np=mask_meta.get("final_mask_np"),
                    protect_np=mask_meta.get("protect_mask_np"),
                    schp_labels=mask_meta.get("schp_labels"),
                    garment_subtype=garment_subtype,
                )
                best_candidate_score = cc_vresult.score
                quality_report = InferenceQualityReport(
                    passed=cc_vresult.identity_drift < 50.0,
                    identity_drift_score=cc_vresult.identity_drift,
                    failure_reasons=tuple(cc_vresult.failure_reasons),
                )
                logger.info(
                    "cross_category_scored score=%.4f face=%.4f garment=%.4f "
                    "drift=%.1f replacement=%.4f trace_id=%s",
                    cc_vresult.score, cc_vresult.face_quality, cc_vresult.garment_quality,
                    cc_vresult.identity_drift, cc_vresult.garment_replacement, trace_id,
                )
            except Exception as e:
                logger.warning("cross_category_scoring_failed error=%s trace_id=%s", e, trace_id)

        logger.info(
            "cross_category_inference_complete inference_ms=%.0f trace_id=%s",
            inference_ms, trace_id,
        )
    else:
        logger.info(
            "cross_category_not_needed single_stage vton_type=%s trace_id=%s",
            vton_type, trace_id,
        )

    # ── Single-stage path ─────────────────────────────────────────────────
    if not cross_category_needed:
        inference_start = time.perf_counter()

        # Guidance/step tuning — use frontend preset if provided, else env default
        effective_guidance = guidance_scale if guidance_scale is not None else GUIDANCE_SCALE
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
                    source_cloth_type=source_cloth_type,
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
                        garment_subtype=garment_subtype,
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

    # ── DEBUG SAVE: raw output before face restoration ─────────────────────
    _debug_dir = Path("/tmp/idm-vton-debug")
    _debug_dir.mkdir(parents=True, exist_ok=True)
    if raw_output is not None:
        raw_output.save(str(_debug_dir / f"raw_output_before_face_restoration_{trace_id}.png"))
    else:
        logger.warning("debug_save_raw_output raw_output_is_None trace_id=%s", trace_id)

    # ── Face restoration — mild enhancement, no identity overwrite ────────
    # Enahnces the face region in the diffusion output using mild sharpening
    # or GFPGAN.  Sources the face from the original person image and never
    # pastes original pixels over generated clothing.
    face_restore_enabled = os.environ.get("ENABLE_FACE_RESTORATION", "0") == "1"
    logger.info("ENABLE_FACE_RESTORATION=%s trace_id=%s", face_restore_enabled, trace_id)
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
        # ── DEBUG SAVE: output after face restoration ────────────────────
        result.save(str(_debug_dir / f"output_after_face_restoration_{trace_id}.png"))
    else:
        logger.info("face_restoration_skipped available=%s trace_id=%s",
                     _FACE_RESTORATION_AVAILABLE, trace_id)
        # ── DEBUG SAVE: output with face restoration skipped ─────────────
        if result is not None:
            result.save(str(_debug_dir / f"output_after_face_restoration_{trace_id}.png"))

    # ── DEBUG SAVE: final returned output ─────────────────────────────────
    if result is not None:
        result.save(str(_debug_dir / f"final_returned_output_{trace_id}.png"))

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
