# ---- Base image: CUDA runtime for GPU inference ----
FROM nvidia/cuda:12.2.2-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps: Python, pip, ffmpeg (often useful with librosa), git (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv ffmpeg git \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip setuptools wheel

# Python packages (install GPU build of onnxruntime)
# Equivalent to:
# !pip install -q museval resampy
# !pip uninstall -q -y onnxruntime
# !pip install -q -U onnxruntime-gpu
RUN python3 -m pip install \
        numpy \
        soundfile \
        librosa \
        resampy \
        museval \
        onnxruntime-gpu

RUN pip install \
    fastapi==0.115.4 uvicorn[standard]==0.32.0 

# Optional: working directory for your code
WORKDIR /app
# If you have project files, uncomment the next line to copy them in:
COPY . /app

EXPOSE 8003 
# Quick sanity check on container start (prints available ORT providers)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003"]
