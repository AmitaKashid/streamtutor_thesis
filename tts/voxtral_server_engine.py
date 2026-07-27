from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import httpx
import numpy as np
import soundfile as sf
import torch

from tts.base_engine import BaseTTSEngine
from tts.config import (
    VOXTRAL_BASE_URL,
    VOXTRAL_ENDPOINT,
    VOXTRAL_MODEL,
    VOXTRAL_VOICE,
    VOXTRAL_RESPONSE_FORMAT,
    VOXTRAL_CONNECT_TIMEOUT_S,
    VOXTRAL_READ_TIMEOUT_S,
    VOXTRAL_SAMPLE_RATE_HINT,
)


@dataclass
class VoxtralServerConfig:
    base_url: str = VOXTRAL_BASE_URL
    endpoint: str = VOXTRAL_ENDPOINT
    model: str = VOXTRAL_MODEL
    voice: str = VOXTRAL_VOICE
    response_format: str = VOXTRAL_RESPONSE_FORMAT
    connect_timeout_s: float = VOXTRAL_CONNECT_TIMEOUT_S
    read_timeout_s: float = VOXTRAL_READ_TIMEOUT_S
    sample_rate_hint: int = VOXTRAL_SAMPLE_RATE_HINT


class VoxtralServerEngine(BaseTTSEngine):
    """
    Self-hosted Voxtral HTTP backend.

    Designed for your thesis setup:
    - mic mode
    - stream-chunked LLM output
    - per-chunk synthesize()
    - preset voice, no voice prompt, no API key
    """

    def __init__(self, cfg: Optional[VoxtralServerConfig] = None) -> None:
        self.cfg = cfg or VoxtralServerConfig()
        self._client: Optional[httpx.Client] = None

    def load(self) -> None:
        if self._client is not None:
            return

        timeout = httpx.Timeout(
            connect=self.cfg.connect_timeout_s,
            read=self.cfg.read_timeout_s,
            write=self.cfg.read_timeout_s,
            pool=self.cfg.connect_timeout_s,
        )
        self._client = httpx.Client(timeout=timeout)

    def supports_streaming_input(self) -> bool:
        # For your thesis flow, stream-chunked uses synthesize() chunk by chunk.
        return False

    def _url(self) -> str:
        return f"{self.cfg.base_url.rstrip('/')}{self.cfg.endpoint}"

    def _save_wav(self, save_to: Path, wav_t: torch.Tensor, sr: int) -> None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        wav_np = wav_t.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        sf.write(str(save_to), wav_np, sr)

    def synthesize(
        self,
        text: str,
        voice_prompt: Optional[Path] = None,
        config: Optional[VoxtralServerConfig] = None,
        save_to: Optional[Path] = None,
    ) -> Tuple[torch.Tensor, int]:
        del voice_prompt  # not used for preset-voice self-hosted Voxtral

        if self._client is None:
            self.load()

        cfg = config or self.cfg
        text = (text or "").strip()
        if not text:
            raise ValueError("Empty text passed to VoxtralServerEngine.synthesize().")

        payload = {
            "input": text,
            "model": cfg.model,
            "response_format": cfg.response_format,
            "voice": cfg.voice,
        }

        resp = self._client.post(self._url(), json=payload)
        resp.raise_for_status()

        audio_bytes = resp.content
        wav_np, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        if wav_np.ndim == 2:
            wav_np = wav_np.mean(axis=1)

        if not isinstance(sr, int) or sr <= 0:
            sr = cfg.sample_rate_hint

        wav_t = torch.from_numpy(wav_np.astype(np.float32, copy=False)).unsqueeze(0)

        if save_to is not None:
            self._save_wav(save_to, wav_t, sr)

        return wav_t, sr