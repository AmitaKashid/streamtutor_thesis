FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /workspace/llmtutor

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    cmake \
    ninja-build \
    git \
    ffmpeg \
    sox \
    libsox-dev \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m pip install --upgrade pip setuptools wheel

COPY requirements.txt .

RUN python3.11 -m pip install --no-cache-dir \
    llama-cpp-python==0.3.4 \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3.11", "app/mic_stt_llm_tts.py", "--help"]