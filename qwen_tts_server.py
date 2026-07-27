from __future__ import annotations

import base64
import io
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI
from pydantic import BaseModel


MODEL_NAME = os.getenv(
    "QWEN3_TTS_MODEL",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
)

DEFAULT_SPEAKER = os.getenv("QWEN3_TTS_SPEAKER", "Aiden")
DEFAULT_LANGUAGE = os.getenv("QWEN3_TTS_LANGUAGE", "English")


LANGUAGE_MAP = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}


def ensure_sox_on_path() -> None:
    if shutil.which("sox"):
        return

    candidates = [
        r"C:\Program Files (x86)\sox-14-4-2",
        r"C:\Program Files\sox-14-4-2",
        r"C:\sox-14-4-2",
    ]

    for folder in candidates:
        exe = Path(folder) / "sox.exe"
        if exe.exists():
            os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
            print(f"[QWEN SERVER] Added SoX to PATH: {folder}")
            return

    print("[QWEN SERVER WARNING] SoX not found on PATH.")


class SynthesizeRequest(BaseModel):
    text: str
    speaker: Optional[str] = None
    language: Optional[str] = None


class SynthesizeResponse(BaseModel):
    sample_rate: int
    audio_base64: str
    duration_s: float
    synth_time_s: float
    speaker: str
    language: str


app = FastAPI(title="Qwen3 TTS Server")

model = None


@app.on_event("startup")
def load_model():
    global model

    ensure_sox_on_path()

    print("[QWEN SERVER] Loading Qwen3-TTS")
    print(f"[QWEN SERVER] model: {MODEL_NAME}")

    from qwen_tts import Qwen3TTSModel

    if torch.cuda.is_available():
        print(f"[QWEN SERVER] CUDA: {torch.cuda.get_device_name(0)}")
        device_map = "cuda:0"

        # Start with float32 for quality/stability diagnosis.
        # Once clean, you can test torch.float16.
        dtype = torch.float32
    else:
        print("[QWEN SERVER WARNING] CUDA not available. CPU will be slow.")
        device_map = "cpu"
        dtype = torch.float32

    model = Qwen3TTSModel.from_pretrained(
        MODEL_NAME,
        device_map=device_map,
        dtype=dtype,
    )

    print("[QWEN SERVER] Model loaded.")


@app.get("/health")
def health():
    return {
        "ok": model is not None,
        "model": MODEL_NAME,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.post("/synthesize", response_model=SynthesizeResponse)
def synthesize(req: SynthesizeRequest):
    if model is None:
        raise RuntimeError("Qwen model is not loaded yet.")

    text = (req.text or "").strip()
    if not text:
        raise ValueError("Empty text.")

    speaker = req.speaker or DEFAULT_SPEAKER
    language = req.language or DEFAULT_LANGUAGE
    language = LANGUAGE_MAP.get(language, language)

    print(f"[QWEN SERVER] synth start | chars={len(text)} | speaker={speaker} | language={language}")

    t0 = time.perf_counter()

    with torch.inference_mode():
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
        )

    synth_time_s = time.perf_counter() - t0

    wav = wavs[0] if isinstance(wavs, (list, tuple)) else wavs

    if torch.is_tensor(wav):
        wav_np = wav.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        wav_np = np.asarray(wav, dtype=np.float32)

    wav_np = np.squeeze(wav_np).astype(np.float32, copy=False)

    if wav_np.ndim != 1:
        raise RuntimeError(f"Expected mono waveform, got shape={wav_np.shape}")

    wav_np = np.nan_to_num(wav_np, nan=0.0, posinf=0.0, neginf=0.0)

    max_abs = float(np.max(np.abs(wav_np))) if wav_np.size else 0.0
    duration_s = len(wav_np) / max(1, int(sr))

    print(
        f"[QWEN SERVER] synth done | sec={synth_time_s:.3f} | "
        f"duration={duration_s:.3f}s | sr={sr} | max_abs={max_abs:.4f}"
    )

    # Do not loudness-normalize. Only clip safety.
    wav_np = np.clip(wav_np, -1.0, 1.0)

    buf = io.BytesIO()
    sf.write(buf, wav_np, int(sr), format="WAV")
    audio_base64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return SynthesizeResponse(
        sample_rate=int(sr),
        audio_base64=audio_base64,
        duration_s=duration_s,
        synth_time_s=synth_time_s,
        speaker=speaker,
        language=language,
    )