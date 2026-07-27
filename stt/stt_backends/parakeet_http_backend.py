from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import soundfile as sf

from stt.stt_types import TranscriptResult, Segment
from .base import STTBackend


class ParakeetHTTPBackend(STTBackend):
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8096",
        timeout_s: float = 120.0,
        include_timestamps: bool = False,
        should_chunk: bool = False,
    ):
        self.engine = "parakeet-http"
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.include_timestamps = include_timestamps
        self.should_chunk = should_chunk

        r = requests.get(f"{self.base_url}/healthz", timeout=5)
        r.raise_for_status()
        data = r.json()

        if data.get("status") != "ok":
            raise RuntimeError(f"Parakeet server is not ready: {data}")

        print(f"[ParakeetHTTPBackend] Connected: {data}")

    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: Optional[str] = None,
        context: Optional[str] = None,
        return_words: bool = False,
    ) -> TranscriptResult:
        audio = self._prepare_audio(audio)
        duration_s = len(audio) / max(sample_rate, 1)

        if duration_s < 0.15:
            return TranscriptResult(
                text="",
                segments=[],
                language=language or "en",
                engine=self.engine,
            )

        if sample_rate != 16000:
            raise ValueError(
                f"ParakeetHTTPBackend expects 16 kHz audio from the tutor pipeline; got {sample_rate}."
            )

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = Path(f.name)

            sf.write(str(tmp_path), audio, 16000, subtype="PCM_16")

            with open(tmp_path, "rb") as f:
                files = {"file": ("utterance.wav", f, "audio/wav")}
                data = {
                    "include_timestamps": "true" if (return_words or self.include_timestamps) else "false",
                    "should_chunk": "true" if self.should_chunk else "false",
                }
                r = requests.post(
                    f"{self.base_url}/transcribe",
                    files=files,
                    data=data,
                    timeout=self.timeout_s,
                )

            r.raise_for_status()
            payload = r.json()
            text = (payload.get("text") or "").strip()

            return TranscriptResult(
                text=text,
                segments=[
                    Segment(start=0.0, end=duration_s, text=text, words=None)
                ] if text else [],
                language=language or "en",
                engine=self.engine,
            )

        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass