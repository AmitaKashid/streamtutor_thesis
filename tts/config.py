"""
Global configuration for the smart tutor system.

This module centralizes configuration values so they can be reused
across the codebase and easily tuned for experiments.
"""

from __future__ import annotations

import os
import torch


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

# chatterbox = local in-process batch/chunked TTS
# chatterbox_stream = local HTTP streaming server around Chatterbox
TTS_BACKEND: str = os.getenv("TTS_BACKEND", "chatterbox")


# ---------------------------------------------------------------------------
# Chatterbox in-process configuration
# ---------------------------------------------------------------------------

TTS_DEVICE: str = os.getenv("TTS_DEVICE", get_default_device())
TTS_TEMPERATURE: float = float(os.getenv("TTS_TEMPERATURE", "0.8"))
TTS_EXAGGERATION: float = float(os.getenv("TTS_EXAGGERATION", "0.45"))
TTS_CFG_WEIGHT: float = float(os.getenv("TTS_CFG_WEIGHT", "0.45"))
TTS_REPETITION_PENALTY: float = float(os.getenv("TTS_REPETITION_PENALTY", "1.1"))
TTS_TOP_P: float = float(os.getenv("TTS_TOP_P", "1.0"))
TTS_MIN_P: float = float(os.getenv("TTS_MIN_P", "0.05"))
TTS_OUTPUT_DIR: str = os.getenv("TTS_OUTPUT_DIR", "output")


# ---------------------------------------------------------------------------
# Chatterbox HTTP streaming server configuration
# Based on the Chatterbox Streaming API documentation:
# - POST /v1/audio/speech/stream for direct WAV streaming
# - optional upload endpoint for voice cloning
# - recommended defaults: balanced + sentence + chunk size 200 + buffer 3
# ---------------------------------------------------------------------------

CHATTERBOX_STREAM_BASE_URL: str = os.getenv("CHATTERBOX_STREAM_BASE_URL", "http://localhost:4123")
CHATTERBOX_STREAM_ENDPOINT: str = os.getenv("CHATTERBOX_STREAM_ENDPOINT", "/v1/audio/speech/stream")
CHATTERBOX_STREAM_UPLOAD_ENDPOINT: str = os.getenv(
    "CHATTERBOX_STREAM_UPLOAD_ENDPOINT",
    "/v1/audio/speech/stream/upload",
)

CHATTERBOX_STREAM_TEMPERATURE: float = float(os.getenv("CHATTERBOX_STREAM_TEMPERATURE", str(TTS_TEMPERATURE)))
CHATTERBOX_STREAM_EXAGGERATION: float = float(os.getenv("CHATTERBOX_STREAM_EXAGGERATION", str(TTS_EXAGGERATION)))
CHATTERBOX_STREAM_CFG_WEIGHT: float = float(os.getenv("CHATTERBOX_STREAM_CFG_WEIGHT", str(TTS_CFG_WEIGHT)))

CHATTERBOX_STREAM_CHUNK_SIZE: int = int(os.getenv("CHATTERBOX_STREAM_CHUNK_SIZE", "200"))
CHATTERBOX_STREAM_STRATEGY: str = os.getenv("CHATTERBOX_STREAM_STRATEGY", "sentence")
CHATTERBOX_STREAM_BUFFER_SIZE: int = int(os.getenv("CHATTERBOX_STREAM_BUFFER_SIZE", "3"))
CHATTERBOX_STREAM_QUALITY: str = os.getenv("CHATTERBOX_STREAM_QUALITY", "balanced")

CHATTERBOX_STREAM_CONNECT_TIMEOUT_S: float = float(os.getenv("CHATTERBOX_STREAM_CONNECT_TIMEOUT_S", "10"))
CHATTERBOX_STREAM_READ_TIMEOUT_S: float = float(os.getenv("CHATTERBOX_STREAM_READ_TIMEOUT_S", "120"))
CHATTERBOX_STREAM_REQUEST_CHUNK_BYTES: int = int(os.getenv("CHATTERBOX_STREAM_REQUEST_CHUNK_BYTES", "4096"))



KOKORO_VOICE = "af_heart"
KOKORO_LANG_CODE = "a"
KOKORO_SPEED = 0.9
KOKORO_SAMPLE_RATE = 24000
KOKORO_STREAM_AUDIO_CHUNK_MS = 120

# ---------------------------------------------------------------------------
# Qwen3-TTS configuration
# CustomVoice model with 9 built-in preset speakers
# ---------------------------------------------------------------------------

QWEN3_TTS_MODEL: str = os.getenv(
    "QWEN3_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
)
QWEN3_TTS_SPEAKER: str = os.getenv("QWEN3_TTS_SPEAKER", "Aiden")
QWEN3_TTS_LANGUAGE: str = os.getenv("QWEN3_TTS_LANGUAGE", "en")
QWEN3_TTS_SPEED: float = float(os.getenv("QWEN3_TTS_SPEED", "1.0"))
QWEN3_TTS_SAMPLE_RATE: int = int(os.getenv("QWEN3_TTS_SAMPLE_RATE", "24000"))
QWEN3_TTS_STREAM_AUDIO_CHUNK_MS: int = int(os.getenv("QWEN3_TTS_STREAM_AUDIO_CHUNK_MS", "120"))

# ---------------------------------------------------------------------------
# Voxtral self-hosted server configuration
# Open-weights / preset-voice backend
# ---------------------------------------------------------------------------

VOXTRAL_BASE_URL: str = os.getenv("VOXTRAL_BASE_URL", "http://127.0.0.1:8091")
VOXTRAL_ENDPOINT: str = os.getenv("VOXTRAL_ENDPOINT", "/v1/audio/speech")
VOXTRAL_MODEL: str = os.getenv("VOXTRAL_MODEL", "mistralai/Voxtral-4B-TTS-2603")
VOXTRAL_VOICE: str = os.getenv("VOXTRAL_VOICE", "casual_male")
VOXTRAL_RESPONSE_FORMAT: str = os.getenv("VOXTRAL_RESPONSE_FORMAT", "wav")
VOXTRAL_CONNECT_TIMEOUT_S: float = float(os.getenv("VOXTRAL_CONNECT_TIMEOUT_S", "10"))
VOXTRAL_READ_TIMEOUT_S: float = float(os.getenv("VOXTRAL_READ_TIMEOUT_S", "180"))
VOXTRAL_SAMPLE_RATE_HINT: int = int(os.getenv("VOXTRAL_SAMPLE_RATE_HINT", "24000"))