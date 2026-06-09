# syntax=docker/dockerfile:1.4
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel AS build

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    MODELS_DIR=/workspace/models \
    IDM_VTON_DIR=/workspace/IDM-VTON

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl git-lfs \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    build-essential gcc g++ python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Clone IDM-VTON repository and download LFS assets ───────────────────
RUN git clone --depth 1 https://github.com/Mayankaggarwal8055/IDM-VTON.git /workspace/IDM-VTON
    && cd /workspace/IDM-VTON \
    && git lfs pull
WORKDIR /workspace/IDM-VTON

# ── Install Python dependencies ───────────────────────────────────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        "accelerate==0.25.0" \
        "diffusers==0.25.0" \
        "transformers==4.36.2" \
        "torchmetrics==1.2.1" \
        "tqdm==4.66.1" \
        "einops==0.7.0" \
        "bitsandbytes==0.39.0" \
        "scipy==1.11.1" \
        "opencv-python-headless==4.10.0.84" \
        "Pillow==10.3.0" \
        "fvcore==0.1.5.post20221221" \
        "cloudpickle==3.0.0" \
        "omegaconf==2.3.0" \
        "pycocotools==2.0.8" \
        "av==12.3.0" \
        "basicsr==1.4.2" \
        "onnxruntime-gpu==1.16.3" \
        "runpod==1.7.13" \
        "cloudinary==1.41.0" \
        "requests==2.31.0" \
        "huggingface-hub==0.25.0"

# Install detectron2 (needs building from source for CUDA 11.8)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    "detectron2@git+https://github.com/facebookresearch/detectron2.git@main"

# ── Pre-download model checkpoints ────────────────────────────────────────
RUN python - <<'PY'
import os
import urllib.request
import shutil

models_dir = "/workspace/models"

# DensePose weights
densepose_path = os.path.join(models_dir, "densepose", "model_final_162be9.pkl")
os.makedirs(os.path.dirname(densepose_path), exist_ok=True)
if not os.path.exists(densepose_path):
    print("Downloading DensePose weights...")
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl",
        densepose_path,
    )
    print("DensePose weights downloaded")

# Create checkpoint directories and placeholders for parsing/openpose
# The actual downloads happen at warmup via the IDM-VTON library
idm_ckpt = "/workspace/IDM-VTON/ckpt"
os.makedirs(f"{idm_ckpt}/densepose", exist_ok=True)
os.makedirs(f"{idm_ckpt}/humanparsing", exist_ok=True)
os.makedirs(f"{idm_ckpt}/openpose/ckpts", exist_ok=True)

# Symlink DensePose weights
if os.path.exists(densepose_path):
    target = f"{idm_ckpt}/densepose/model_final_162be9.pkl"
    if not os.path.exists(target):
        os.symlink(densepose_path, target)
        print("DensePose symlink created")

print("All model weights pre-downloaded")
PY

# Pre-download the IDM-VTON model weights from HuggingFace using huggingface_hub
RUN python - <<'PY'
from huggingface_hub import snapshot_download
import os

model_id = "yisol/IDM-VTON"
print(f"Pre-downloading {model_id}...")
snapshot_download(
    model_id,
    cache_dir="/root/.cache/huggingface",
    local_dir_use_symlinks=False,
)
print("IDM-VTON model weights pre-downloaded")
PY

# ── Build-time validation ─────────────────────────────────────────────────
RUN python -c "import diffusers, transformers, torch, cv2, detectron2, onnxruntime; print('Base imports OK')"
RUN python -c "import sys; sys.path.insert(0, '/workspace/IDM-VTON'); from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as T; print('Pipeline import OK')"
RUN python -c "import sys; sys.path.insert(0, '/workspace/IDM-VTON/gradio_demo'); from utils_mask import get_mask_location; print('Mask utils import OK')"

# Copy the RunPod handler
COPY handler.py /workspace/IDM-VTON/gradio_demo/handler.py

WORKDIR /workspace/IDM-VTON/gradio_demo
CMD ["python", "-u", "/workspace/IDM-VTON/gradio_demo/handler.py"]
