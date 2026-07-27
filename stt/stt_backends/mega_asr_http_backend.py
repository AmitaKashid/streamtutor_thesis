from __future__ import annotations

import base64
from typing import Optional

import numpy as np
import requests

from stt.stt_types import TranscriptResult, Segment
from .base import STTBackend


class MegaASRHTTPBackend(STTBackend):
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8095",
        timeout_s: float = 120.0,
    ):
        self.engine = "mega-asr-http"
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

        r = requests.get(f"{self.base_url}/health", timeout=5)
        r.raise_for_status()
        data = r.json()

        if not data.get("ok"):
            raise RuntimeError(f"Mega-ASR server is not ready: {data}")

        print(f"[MegaASRHTTPBackend] Connected: {data}")

    def _audio_to_pcm16_base64(self, audio: np.ndarray) -> str:
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1.0, 1.0)

        pcm16 = (audio * 32767.0).astype(np.int16)
        return base64.b64encode(pcm16.tobytes()).decode("ascii")

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: Optional[str] = None,
        context: Optional[str] = None,
        return_words: bool = False,
    ) -> TranscriptResult:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        audio = np.asarray(audio, dtype=np.float32).squeeze()
        duration_s = len(audio) / max(sample_rate, 1)

        if duration_s < 0.15:
            return TranscriptResult(
                text="",
                segments=[],
                language=language,
                engine=self.engine,
            )

        payload = {
            "audio_base64": self._audio_to_pcm16_base64(audio),
            "sample_rate": int(sample_rate),
            "language": language or "en",
            "context": context,
            "return_route": True,
        }

        r = requests.post(
            f"{self.base_url}/transcribe",
            json=payload,
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        data = r.json()

        text = (data.get("text") or "").strip()

        return TranscriptResult(
            text=text,
            segments=[
                Segment(
                    start=0.0,
                    end=duration_s,
                    text=text,
                    words=None,
                )
            ] if text else [],
            language=data.get("language") or language,
            engine=self.engine,
        )