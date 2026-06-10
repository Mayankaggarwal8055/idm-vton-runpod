# syntax=docker/dockerfile:1.4
# =============================================================================
# Optimized Dockerfile for IDM-VTON inference on RunPod
#
# Build strategy:
#   1. OS packages (rarely changes)
#   2. Core Python packages (stable, cache-friendly)
#   3. xformers (needs torch from base image)
#   4. Detectron2 from source (long build, cached separately)
#   5. Clone repo with LFS assets (all ckpt files are in the repo)
#   6. Pre-download only the HF model (yisol/IDM-VTON)
#   7. Build-time validation + copy handler
# =============================================================================

FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel AS build

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    IDM_VTON_DIR=/workspace/IDM-VTON \
    # Point DensePose to the repo's ckpt/ — no separate download needed
    DENSEPOSE_WEIGHTS=/workspace/IDM-VTON/ckpt/densepose/model_final_162be9.pkl \
    # Where to download the HF model
    IDM_VTON_MODEL=/workspace/models/yisol/IDM-VTON \
    CLOUDINARY_FOLDER=trylix/tryon/results \
    # Performance defaults
    ENABLE_XFORMERS=1 \
    ALLOW_TF32=1

WORKDIR /workspace

# ── Layer 1: OS dependencies ──────────────────────────────────────────────
# git-lfs is needed to pull the LFS checkpoint files from the repo.
# libgl1/libglib2.0-0 are required by OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Layer 2: Core Python packages ─────────────────────────────────────────
# torch & torchvision are pre-installed in the base image (via conda).
# We only install what's missing or needs specific versions.
# iopath and yacs are intentionally NOT pinned here — detectron2 resolves
# them transitively in Layer 4 with its own version constraints.
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

# ── Layer 3: Install xformers for memory-efficient attention ──────────────
# xformers reduces VRAM usage by ~40% during inference.
# Must be installed AFTER torch (which is in the base image).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "xformers==0.0.20"

# ── Layer 4: Build detectron2 from source ─────────────────────────────────
# The repo bundles detectron2 under gradio_demo/detectron2/ but the
# pre-compiled .so is for Python 3.9 only — it won't load under Python 3.10.
# Building from source is required. This layer is cached independently.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        "detectron2@git+https://github.com/facebookresearch/detectron2.git@main"

# ── Layer 5: Clone IDM-VTON repo (includes all ckpt assets via LFS) ──────
# All checkpoints (DensePose, human parsing ONNX, OpenPose, IP-Adapter,
# image_encoder) are stored in the repo with Git LFS, so cloning pulls them.
RUN git lfs install && \
    git clone --depth 1 https://github.com/Mayankaggarwal8055/IDM-VTON.git $IDM_VTON_DIR && \
    cd $IDM_VTON_DIR && \
    git lfs pull && \
    # Fix OpenPose path: the handler expects body_pose_model.pth at
    # ckpt/openpose/ but the repo stores it at ckpt/openpose/ckpts/.
    # Create a symlink so the handler's _ensure_dir_layout() check passes.
    ln -s $IDM_VTON_DIR/ckpt/openpose/ckpts/body_pose_model.pth $IDM_VTON_DIR/ckpt/openpose/body_pose_model.pth

# ── Layer 6: Pre-download IDM-VTON model weights from HuggingFace ─────────
# This is the ~7GB SDXL-based model NOT stored in the repo.
RUN python - <<'PY'
import sys

sys.path.insert(0, "/workspace/IDM-VTON")
import diffusers, transformers, torch, cv2, onnxruntime, detectron2
print(f"Core imports OK (diffusers={diffusers.__version__} torch={torch.__version__})")

from src.tryon_pipeline import StableDiffusionXLInpaintPipeline
print("Pipeline import OK")

sys.path.insert(0, "/workspace/IDM-VTON/gradio_demo")
from utils_mask import get_mask_location
print("Mask utils import OK")

from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
print("Parsing + OpenPose imports OK")

from densepose import add_densepose_config
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
print("DensePose + Detectron2 imports OK")
PY

# ── Layer 7: Build-time validation + RunPod handler ───────────────────────
RUN python - <<'PY'
import os
import sys

root = os.environ["IDM_VTON_DIR"]
demo = os.path.join(root, "gradio_demo")

sys.path.insert(0, root)
sys.path.insert(0, demo)

import diffusers, transformers, torch, cv2, onnxruntime, detectron2, xformers
print(f'Core imports OK (diffusers={diffusers.__version__} torch={torch.__version__})')

from src.tryon_pipeline import StableDiffusionXLInpaintPipeline
print('Pipeline import OK')

from utils_mask import get_mask_location
print('Mask utils import OK')

from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
print('Parsing + OpenPose imports OK')

from densepose import add_densepose_config
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
print('DensePose + Detectron2 imports OK')
PY

# Copy the RunPod handler
COPY handler.py /workspace/handler.py

WORKDIR $IDM_VTON_DIR
CMD ["python", "-u", "/workspace/handler.py"]
