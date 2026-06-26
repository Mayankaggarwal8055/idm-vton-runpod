# syntax=docker/dockerfile:1.7

# =============================================================================
# IDM-VTON RunPod Inference Image
# =============================================================================

FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel AS build

# =============================================================================
# Environment
# =============================================================================

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    PYTHONPATH=/workspace \
    IDM_VTON_DIR=/workspace/IDM-VTON \
    IDM_VTON_MODEL=/workspace/models/yisol/IDM-VTON \
    DENSEPOSE_WEIGHTS=/workspace/IDM-VTON/ckpt/densepose/model_final_162be9.pkl \
    CLOUDINARY_FOLDER=trylix/tryon/results \
    ENABLE_XFORMERS=0 \
    ALLOW_TF32=1

WORKDIR /workspace

# =============================================================================
# Layer 1 — OS dependencies
# =============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Layer 2 — Python dependencies
# =============================================================================

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        diffusers==0.25.1 \
        transformers==4.41.1 \
        accelerate==0.25.0 \
        peft==0.11.1 \
        safetensors==0.4.3 \
        tokenizers==0.19.1 \
        huggingface_hub==0.25.2 \
        einops==0.7.0 \
        scipy==1.10.1 \
        scikit-image==0.21.0 \
        opencv-python-headless==4.7.0.72 \
        Pillow==9.4.0 \
        onnxruntime-gpu==1.16.2 \
        av==12.3.0 \
        "protobuf<5" \
        fvcore \
        cloudpickle \
        omegaconf \
        pycocotools \
        tqdm \
        requests==2.32.3 \
        runpod==1.9.1 \
        cloudinary==1.41.0

# =============================================================================
# Layer 3 — Detectron2
# =============================================================================

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        "detectron2@git+https://github.com/facebookresearch/detectron2.git@02b5c4e295e990042a714712c21dc79b731e8833"

# =============================================================================
# Layer 4 — Clone IDM-VTON repo + download ALL binary checkpoints
# =============================================================================

# 4a — Clone the repository, download OpenPose checkpoint, and verify
# The GIT_REVISION build arg acts as a cache buster — pass the latest commit SHA
# at build time to invalidate this layer when the repo changes.
#   docker build --build-arg GIT_REVISION=$(git rev-parse HEAD) ...
ARG GIT_REVISION=unknown
RUN git lfs install && \
    echo "Cloning IDM-VTON at revision: ${GIT_REVISION}" && \
    git clone --depth 1 https://github.com/Mayankaggarwal8055/IDM-VTON.git $IDM_VTON_DIR && \
    mkdir -p $IDM_VTON_DIR/ckpt/openpose/ckpts && \
    curl -L \
        -o $IDM_VTON_DIR/ckpt/openpose/ckpts/body_pose_model.pth \
        https://huggingface.co/spaces/yisol/IDM-VTON/resolve/main/ckpt/openpose/ckpts/body_pose_model.pth && \
    ln -sf \
        $IDM_VTON_DIR/ckpt/openpose/ckpts/body_pose_model.pth \
        $IDM_VTON_DIR/ckpt/openpose/body_pose_model.pth && \
    python -c "import os,sys; p='$IDM_VTON_DIR/ckpt/openpose/ckpts/body_pose_model.pth'; sz=os.path.getsize(p)/1024/1024; print(f'body_pose_model.pth = {sz:.1f} MB'); sys.exit(0 if sz>=10 else 1)"

# 4b — Download ONNX humanparsing models (bypasses git LFS issues)
RUN python - <<'PY'
from huggingface_hub import hf_hub_download
import os, shutil, sys

dest = os.environ["IDM_VTON_DIR"] + "/ckpt/humanparsing"
os.makedirs(dest, exist_ok=True)

for fname in ["parsing_atr.onnx", "parsing_lip.onnx"]:
    print(f"Downloading {fname}...")
    cached = hf_hub_download(
        repo_id="levihsu/OOTDiffusion",
        filename=f"checkpoints/humanparsing/{fname}",
        cache_dir="/tmp/hf_onnx_cache",
    )
    final = os.path.join(dest, fname)
    shutil.copy2(cached, final)
    size_mb = os.path.getsize(final) / 1024 / 1024
    if size_mb < 50:
        print(f"FATAL: {fname} is {size_mb:.2f} MB — still a git LFS pointer", flush=True)
        sys.exit(1)
    print(f"  OK: {fname} = {size_mb:.1f} MB")

shutil.rmtree("/tmp/hf_onnx_cache", ignore_errors=True)
print("humanparsing ONNX downloads complete")
PY

# 4c — Download DensePose checkpoint (model_final_162be9.pkl)
RUN python3 - <<'PY'
from huggingface_hub import hf_hub_download
import os, shutil, sys

dest = os.environ["IDM_VTON_DIR"] + "/ckpt/densepose"
os.makedirs(dest, exist_ok=True)

fname = "model_final_162be9.pkl"
print(f"Downloading {fname}...")
cached = hf_hub_download(
    repo_id="yisol/IDM-VTON",
    filename=f"densepose/{fname}",
    cache_dir="/tmp/hf_densepose_cache",
)
final = os.path.join(dest, fname)
shutil.copy2(cached, final)

size_mb = os.path.getsize(final) / 1024 / 1024
print(f"{fname} = {size_mb:.1f} MB")
if size_mb < 100:
    sys.exit("DensePose checkpoint looks truncated or is a pointer file")

shutil.rmtree("/tmp/hf_densepose_cache", ignore_errors=True)
print("DensePose download complete")
PY

# =============================================================================
# Layer 5 — Download full IDM-VTON SDXL weights + cleanup + verify
# =============================================================================

RUN python - <<'PY'
from huggingface_hub import snapshot_download
import os, sys, shutil

# --- Download ---
target_dir = "/workspace/models/yisol/IDM-VTON"
os.makedirs(target_dir, exist_ok=True)

print("Downloading IDM-VTON weights...")

snapshot_download(
    repo_id="yisol/IDM-VTON",
    local_dir=target_dir,
    local_dir_use_symlinks=False,
)

print("Download complete")

# --- Cleanup HuggingFace cache (~7-10 GB) to avoid image bloat ---
import shutil as _shutil
_shutil.rmtree("/root/.cache/huggingface", ignore_errors=True)
print("HuggingFace cache cleaned")

# --- Verify required weight files exist (not just directories) ---
target = target_dir

size_checked = {
    "unet/diffusion_pytorch_model.bin": 1000,
    "vae/diffusion_pytorch_model.safetensors": 50,
}
config_files = [
    "scheduler/scheduler_config.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer_2/tokenizer_config.json",
    "image_encoder/config.json",
    "text_encoder/config.json",
    "text_encoder_2/config.json",
    "unet_encoder/config.json",
]

all_ok = True

for rel_path, min_mb in size_checked.items():
    full = os.path.join(target, rel_path)
    if not os.path.isfile(full):
        print(f"FATAL: missing required file: {full}")
        all_ok = False
        continue
    if os.path.islink(full):
        print(f"FATAL: {full} is a symlink — local_dir_use_symlinks did not work")
        all_ok = False
        continue
    size_mb = os.path.getsize(full) / 1024 / 1024
    if size_mb < min_mb:
        print(f"FATAL: {rel_path} too small ({size_mb:.2f} MB < {min_mb} MB) — truncated or pointer file")
        all_ok = False
        continue
    print(f"  OK: {rel_path} = {size_mb:.1f} MB")

for rel_path in config_files:
    full = os.path.join(target, rel_path)
    if not os.path.isfile(full):
        print(f"FATAL: missing required config: {full}")
        all_ok = False
    else:
        size_kb = os.path.getsize(full) / 1024
        print(f"  OK: {rel_path} = {size_kb:.1f} KB")

for sub in ["unet", "vae", "scheduler", "tokenizer", "tokenizer_2",
            "image_encoder", "text_encoder", "text_encoder_2", "unet_encoder"]:
    path = os.path.join(target, sub)
    if not os.path.isdir(path):
        print(f"FATAL: missing required subdirectory: {path}")
        all_ok = False
    else:
        print(f"  OK: {sub}/")

total, used, free = shutil.disk_usage("/workspace")
print(f"DISK: total_gb={total / (1024**3):.1f} used_gb={used / (1024**3):.1f} free_gb={free / (1024**3):.1f}")

if not all_ok:
    sys.exit(1)
print("All model weight files verified — Layer 5 complete")
PY

# =============================================================================
# Layer 6 — Build validation (IDM-VTON pipeline)
# =============================================================================

RUN python - <<'PY'
import os
import sys

root = os.environ["IDM_VTON_DIR"]
demo = os.path.join(root, "gradio_demo")

sys.path.insert(0, root)
sys.path.insert(0, demo)

required_files = {
    os.path.join(root, "ckpt/humanparsing/parsing_atr.onnx"): 50,
    os.path.join(root, "ckpt/humanparsing/parsing_lip.onnx"): 50,
    os.path.join(root, "ckpt/openpose/ckpts/body_pose_model.pth"): 10,
    os.path.join(root, "ckpt/densepose/model_final_162be9.pkl"): 100,
}
for path, min_mb in required_files.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    size_mb = os.path.getsize(path) / 1024 / 1024
    if size_mb < min_mb:
        raise RuntimeError(
            f"File too small ({size_mb:.2f} MB < {min_mb} MB) — "
            f"likely corrupted: {path}"
        )
    print(f"Size OK: {os.path.basename(path)} = {size_mb:.1f} MB")

import diffusers, transformers, torch, cv2, onnxruntime, detectron2
print(f"Core imports OK (diffusers={diffusers.__version__} torch={torch.__version__})")

from src.tryon_pipeline import StableDiffusionXLInpaintPipeline
print("Pipeline import OK")

from utils_mask import get_mask_location
print("Mask utils import OK")

from preprocess.openpose.run_openpose import OpenPose
print("OpenPose import OK")

from densepose import add_densepose_config
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
print("DensePose + Detectron2 imports OK")

print("All validation passed")
PY

# =============================================================================
# Copy RunPod handler + mask pipeline
# =============================================================================

COPY handler.py /workspace/handler.py
COPY mask_pipeline.py /workspace/mask_pipeline.py
COPY quality_validation.py /workspace/quality_validation.py
COPY post_processing.py /workspace/post_processing.py
COPY face_restoration.py /workspace/face_restoration.py
COPY p0_diagnostics.py /workspace/p0_diagnostics.py

# =============================================================================
# Layer 7 — Validate worker module (mask_pipeline.py)
# =============================================================================
# NOTE: This runs AFTER the COPY layer above, so /workspace/mask_pipeline.py exists.

RUN python - <<'PY'
import os
import sys

# Ensure /workspace is on the path — this is the target for COPY mask_pipeline.py
_ws = "/workspace"
sys.path.insert(0, _ws)

_mp = os.path.join(_ws, "mask_pipeline.py")

if not os.path.exists(_mp):
    raise FileNotFoundError(
        f"MASK_PIPELINE NOT FOUND at {_mp}. "
        "This should never happen because the COPY layer above places it there."
    )

print(f"mask_pipeline.py exists at {_mp} ({os.path.getsize(_mp)} bytes)")

try:
    from mask_pipeline import (
        assert_binary_mask,
        build_schp_inpaint_mask,
        build_schp_protect_mask,
        apply_protection_binary,
        validate_mask_coverage,
        detect_inference_failures,
    )
    print("import mask_pipeline OK")
    print(f"  assert_binary_mask: {callable(assert_binary_mask)}")
    print(f"  build_schp_inpaint_mask: {callable(build_schp_inpaint_mask)}")
    print(f"  build_schp_protect_mask: {callable(build_schp_protect_mask)}")
    print(f"  apply_protection_binary: {callable(apply_protection_binary)}")
    print(f"  validate_mask_coverage: {callable(validate_mask_coverage)}")
    print(f"  detect_inference_failures: {callable(detect_inference_failures)}")
except Exception as exc:
    raise RuntimeError(f"Failed to import mask_pipeline: {exc}") from exc

try:
    from quality_validation import score_candidate
    print(f"import quality_validation OK — score_candidate: {callable(score_candidate)}")
except Exception as exc:
    raise RuntimeError(f"Failed to import quality_validation: {exc}") from exc

print("Mask pipeline validation passed")
PY

# =============================================================================
# Runtime
# =============================================================================

WORKDIR $IDM_VTON_DIR

CMD ["python", "-u", "/workspace/handler.py"]
