from __future__ import annotations

import base64
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

CKPT_DIR = ROOT / "ckpt" / "Mega-ASR"

MODEL_PATH = CKPT_DIR / "Qwen3-ASR-1.7B"
LORA_DIR = CKPT_DIR / "mega-asr-merged"
ROUTER_CHECKPOINT = CKPT_DIR / "audio_quality_router" / "best_acc_model.safetensors"

DEVICE_MAP = os.getenv("MEGA_ASR_DEVICE_MAP", "cuda:0")
ROUTING_ENABLED = os.getenv("MEGA_ASR_ROUTING", "true").lower() in {"1", "true", "yes"}
QUALITY_THRESHOLD = float(os.getenv("MEGA_ASR_THRESHOLD", "0.5"))

# Safer for 8 GB GPUs.
KEEP_DELTA_ON_GPU = os.getenv("MEGA_ASR_KEEP_DELTA_ON_GPU", "false").lower() in {"1", "true", "yes"}

LANGUAGE_MAP = {
    "en": "English",
    "english": "English",
    "de": "German",
    "german": "German",
    "fr": "French",
    "french": "French",
    "es": "Spanish",
    "spanish": "Spanish",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "it": "Italian",
    "italian": "Italian",
    "ru": "Russian",
    "russian": "Russian",
    "ja": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "korean": "Korean",
    "zh": "Chinese",
    "chinese": "Chinese",
    "hi": "Hindi",
    "hindi": "Hindi",
    "ar": "Arabic",
    "arabic": "Arabic",
}


def normalize_language(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None

    key = str(language).strip().lower()
    if not key:
        return None

    return LANGUAGE_MAP.get(key, str(language).strip())

app = FastAPI(title="Mega-ASR STT Server")

model = None


class TranscribeRequest(BaseModel):
    audio_base64: str
    sample_rate: int = 16000
    language: Optional[str] = "en"
    context: Optional[str] = None
    return_route: bool = True


class TranscribeResponse(BaseModel):
    text: str
    language: Optional[str] = None
    engine: str = "mega-asr"
    latency_s: float
    audio_duration_s: float
    rtf: float
    use_lora: Optional[bool] = None
    degraded_prob: Optional[float] = None
    route_source: Optional[str] = None


def pcm16_base64_to_float32(audio_base64: str) -> np.ndarray:
    raw = base64.b64decode(audio_base64)
    pcm16 = np.frombuffer(raw, dtype=np.int16)
    return np.clip(pcm16.astype(np.float32) / 32768.0, -1.0, 1.0)


def normalize_text_result(value) -> str:
    if value is None:
        return ""

    if isinstance(value, list):
        return " ".join(str(x).strip() for x in value if str(x).strip()).strip()

    return str(value).strip()


@app.on_event("startup")
def load_model():
    global model

    print("[MEGA-ASR] Loading model")
    print(f"[MEGA-ASR] model_path        : {MODEL_PATH}")
    print(f"[MEGA-ASR] lora_dir          : {LORA_DIR}")
    print(f"[MEGA-ASR] router_checkpoint : {ROUTER_CHECKPOINT}")
    print(f"[MEGA-ASR] device_map        : {DEVICE_MAP}")
    print(f"[MEGA-ASR] routing_enabled   : {ROUTING_ENABLED}")
    print(f"[MEGA-ASR] keep_delta_on_gpu : {KEEP_DELTA_ON_GPU}")

    if DEVICE_MAP.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")

    from MegaASR.model.megaASR import MegaASR

    model = MegaASR(
        model_path=MODEL_PATH,
        lora_dir=LORA_DIR,
        router_checkpoint=ROUTER_CHECKPOINT,
        routing_enabled=ROUTING_ENABLED,
        quality_threshold=QUALITY_THRESHOLD,
        device_map=DEVICE_MAP,
        keep_delta_on_gpu=KEEP_DELTA_ON_GPU,
        backend="transformers",
    )

    print("[MEGA-ASR] Ready.")


@app.get("/health")
def health():
    return {
        "ok": model is not None,
        "engine": "mega-asr",
        "backend": "transformers",
        "device_map": DEVICE_MAP,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "routing_enabled": ROUTING_ENABLED,
    }


@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe(req: TranscribeRequest):
    if model is None:
        raise RuntimeError("Mega-ASR model is not loaded.")

    audio = pcm16_base64_to_float32(req.audio_base64)
    duration_s = len(audio) / max(1, int(req.sample_rate))

    if duration_s < 0.15:
        return TranscribeResponse(
            text="",
            language=mega_language,
            latency_s=0.0,
            audio_duration_s=duration_s,
            rtf=0.0,
        )

    if req.sample_rate != 16000:
        raise ValueError("Mega-ASR server expects 16 kHz audio from the tutor pipeline.")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    try:
        sf.write(tmp_path, audio, 16000)

        t0 = time.perf_counter()

        mega_language = normalize_language(req.language)

        try:
            result = model.infer(
                tmp_path,
                language=mega_language,
                return_route=True,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Mega-ASR inference failed: {type(exc).__name__}: {exc}",
            )

        latency_s = time.perf_counter() - t0
        rtf = latency_s / max(duration_s, 1e-6)

        if isinstance(result, dict):
            text = normalize_text_result(result.get("text"))
            use_lora = result.get("use_lora")
            degraded_prob = result.get("degraded_prob")
            route_source = result.get("route_source")
        else:
            text = normalize_text_result(result)
            use_lora = None
            degraded_prob = None
            route_source = None

        print(
            f"[MEGA-ASR] transcribed | dur={duration_s:.2f}s | "
            f"lat={latency_s:.3f}s | rtf={rtf:.3f} | "
            f"use_lora={use_lora} | text={text!r}"
        )

        return TranscribeResponse(
            text=text,
            language=mega_language,
            latency_s=latency_s,
            audio_duration_s=duration_s,
            rtf=rtf,
            use_lora=use_lora,
            degraded_prob=degraded_prob,
            route_source=route_source,
        )

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass