from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_qwen3_tts import FasterQwen3TTS


MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_SPEAKER = "aiden"
DEFAULT_LANGUAGE = "English"
DEFAULT_CHUNK_SIZE = 4

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

SPEAKER_MAP = {
    "Aiden": "aiden",
    "Ryan": "ryan",
    "Vivian": "vivian",
    "Serena": "serena",
    "Dylan": "dylan",
    "Eric": "eric",
    "Ono_Anna": "ono_anna",
    "Sohee": "sohee",
    "Uncle_Fu": "uncle_fu",
    "aiden": "aiden",
    "ryan": "ryan",
    "vivian": "vivian",
    "serena": "serena",
    "dylan": "dylan",
    "eric": "eric",
    "ono_anna": "ono_anna",
    "sohee": "sohee",
    "uncle_fu": "uncle_fu",
}

app = FastAPI(title="Qwen3 Streaming TTS Server")

model: Optional[FasterQwen3TTS] = None


def to_pcm16_base64(audio: np.ndarray) -> str:
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)
    return base64.b64encode(pcm16.tobytes()).decode("ascii")


@app.on_event("startup")
def load_model() -> None:
    global model

    print("[QWEN STREAM] Loading faster-qwen3-tts")
    print(f"[QWEN STREAM] model: {MODEL_NAME}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for faster-qwen3-tts streaming mode.")

    print(f"[QWEN STREAM] CUDA GPU: {torch.cuda.get_device_name(0)}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = FasterQwen3TTS.from_pretrained(
        MODEL_NAME,
        device="cuda",
        dtype=torch.bfloat16,
    )

    print("[QWEN STREAM] Running warm-up...")
    model.warmup(prefill_len=100)
    print("[QWEN STREAM] Ready.")


@app.get("/health")
def health():
    return {
        "ok": model is not None,
        "model": MODEL_NAME,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda": torch.cuda.is_available(),
        "mode": "streaming",
    }


@app.websocket("/stream")
async def stream_tts(ws: WebSocket):
    await ws.accept()

    if model is None:
        await ws.send_text(json.dumps({
            "type": "error",
            "message": "Qwen stream model is not loaded.",
        }))
        await ws.close()
        return

    try:
        while True:
            raw = await ws.receive_text()
            req = json.loads(raw)

            if req.get("type") == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
                continue

            if req.get("type") != "synthesize":
                await ws.send_text(json.dumps({
                    "type": "error",
                    "message": f"Unknown message type: {req.get('type')}",
                }))
                continue

            text = (req.get("text") or "").strip()
            if not text:
                await ws.send_text(json.dumps({
                    "type": "error",
                    "message": "Empty text.",
                }))
                continue

            speaker = SPEAKER_MAP.get(req.get("speaker") or DEFAULT_SPEAKER, DEFAULT_SPEAKER)
            language = LANGUAGE_MAP.get(req.get("language") or DEFAULT_LANGUAGE, req.get("language") or DEFAULT_LANGUAGE)
            chunk_size = int(req.get("chunk_size") or DEFAULT_CHUNK_SIZE)

            print(
                f"[QWEN STREAM] synth start | chars={len(text)} | "
                f"speaker={speaker} | language={language} | chunk_size={chunk_size}"
            )

            start_t = time.perf_counter()
            first_audio_t = None
            chunk_count = 0
            total_samples = 0
            sr_seen = 24000

            await ws.send_text(json.dumps({
                "type": "audio_start",
                "sample_rate": 24000,
                "text": text,
            }))

            for audio_chunk, sr, timing in model.generate_custom_voice_streaming(
                text=text,
                speaker=speaker,
                language=language,
                chunk_size=chunk_size,
                max_new_tokens=512,
                temperature=0.9,
                top_k=50,
                top_p=1.0,
                do_sample=True,
                repetition_penalty=1.05,
            ):
                if first_audio_t is None:
                    first_audio_t = time.perf_counter()

                chunk_count += 1
                sr_seen = int(sr)

                audio_np = np.asarray(audio_chunk, dtype=np.float32).squeeze()
                total_samples += int(audio_np.size)

                await ws.send_text(json.dumps({
                    "type": "audio_chunk",
                    "sample_rate": sr_seen,
                    "pcm16_base64": to_pcm16_base64(audio_np),
                    "chunk_index": chunk_count,
                    "timing": timing,
                }))

                # Yield control to the event loop so chunks are sent immediately.
                await asyncio.sleep(0)

            total_s = time.perf_counter() - start_t
            audio_s = total_samples / max(1, sr_seen)
            ttfa_s = (first_audio_t - start_t) if first_audio_t else None
            rtf = audio_s / total_s if total_s > 0 else 0.0

            print(
                f"[QWEN STREAM] synth done | chunks={chunk_count} | "
                f"ttfa={ttfa_s:.3f}s | audio={audio_s:.3f}s | total={total_s:.3f}s | rtf={rtf:.2f}"
            )

            await ws.send_text(json.dumps({
                "type": "audio_done",
                "sample_rate": sr_seen,
                "chunks": chunk_count,
                "ttfa_s": ttfa_s,
                "audio_s": audio_s,
                "total_s": total_s,
                "rtf": rtf,
            }))

    except WebSocketDisconnect:
        print("[QWEN STREAM] client disconnected")