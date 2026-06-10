# syntax=docker/dockerfile:1.4

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
        "detectron2@git+https://github.com/facebookresearch/detectron2.git@main"

# =============================================================================
# Layer 4 — Clone IDM-VTON repo + LFS assets
# =============================================================================

RUN git lfs install && \
    git clone --depth 1 https://github.com/Mayankaggarwal8055/IDM-VTON.git $IDM_VTON_DIR && \
    cd $IDM_VTON_DIR && \
    git lfs pull && \
    mkdir -p $IDM_VTON_DIR/ckpt/openpose && \
    ln -sf \
        $IDM_VTON_DIR/ckpt/openpose/ckpts/body_pose_model.pth \
        $IDM_VTON_DIR/ckpt/openpose/body_pose_model.pth

# =============================================================================
# Layer 5 — Download full IDM-VTON SDXL weights
# =============================================================================

RUN python - <<'PY'
from huggingface_hub import snapshot_download

target_dir = "/workspace/models/yisol/IDM-VTON"

print("Downloading IDM-VTON weights...")

snapshot_download(
    repo_id="yisol/IDM-VTON",
    local_dir=target_dir,
    local_dir_use_symlinks=False,
)

print("Download complete")
print("Saved to:", target_dir)
PY

# =============================================================================
# Layer 6 — Build validation
# =============================================================================

RUN python - <<'PY'
import os
import sys

root = os.environ["IDM_VTON_DIR"]
demo = os.path.join(root, "gradio_demo")

sys.path.insert(0, root)
sys.path.insert(0, demo)

import diffusers
import transformers
import torch
import cv2
import onnxruntime
import detectron2

print(
    f"Core imports OK "
    f"(diffusers={diffusers.__version__} "
    f"torch={torch.__version__})"
)

from src.tryon_pipeline import StableDiffusionXLInpaintPipeline
print("Pipeline import OK")

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

# =============================================================================
# Copy RunPod handler
# =============================================================================

COPY handler.py /workspace/handler.py

# =============================================================================
# Runtime
# =============================================================================

WORKDIR $IDM_VTON_DIR

CMD ["python", "-u", "/workspace/handler.py"]
