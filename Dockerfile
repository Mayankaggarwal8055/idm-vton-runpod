# syntax=docker/dockerfile:1.4
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel AS build

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    MODELS_DIR=/workspace/models \
    IDM_VTON_DIR=/workspace/IDM-VTON \
    HF_HOME=/root/.cache/huggingface

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl git-lfs \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    build-essential gcc g++ python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/Mayankaggarwal8055/IDM-VTON.git /workspace/IDM-VTON && \
    cd /workspace/IDM-VTON && \
    git lfs pull
WORKDIR /workspace/IDM-VTON

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        "runpod==1.9.1" \
        "requests==2.32.3" \
        "numpy==1.24.4" \
        "scipy==1.10.1" \
        "scikit-image==0.21.0" \
        "opencv-python-headless==4.7.0.72" \
        "Pillow==9.4.0" \
        "matplotlib==3.7.4" \
        "tqdm==4.64.1" \
        "diffusers==0.25.1" \
        "transformers==4.41.1" \
        "huggingface_hub==0.25.2" \
        "accelerate==0.31.0" \
        "peft==0.11.1" \
        "einops==0.7.0" \
        "safetensors==0.4.3" \
        "tokenizers==0.19.1" \
        "onnxruntime-gpu==1.16.2" \
        "omegaconf==2.3.0" \
        "cloudpickle==3.0.0" \
        "pycocotools==2.0.7" \
        "fvcore==0.1.5.post20221221" \
        "iopath==0.1.10" \
        "yacs==0.1.8" \
        "basicsr==1.4.2" \
        "av==12.3.0" \
        "cloudinary==1.40.0" \
        "protobuf<5"

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "detectron2@git+https://github.com/facebookresearch/detectron2.git@main"

RUN python - <<'PY'
import os, urllib.request
models_dir = "/workspace/models"
densepose_path = os.path.join(models_dir, "densepose", "model_final_162be9.pkl")
os.makedirs(os.path.dirname(densepose_path), exist_ok=True)
if not os.path.exists(densepose_path):
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl",
        densepose_path,
    )
PY

RUN python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="yisol/IDM-VTON",
    cache_dir="/root/.cache/huggingface",
    local_dir="/workspace/IDM-VTON-model-cache",
    local_dir_use_symlinks=False,
)
PY

COPY handler.py /workspace/IDM-VTON/gradio_demo/handler.py
WORKDIR /workspace/IDM-VTON/gradio_demo
CMD ["python", "-u", "/workspace/IDM-VTON/gradio_demo/handler.py"]