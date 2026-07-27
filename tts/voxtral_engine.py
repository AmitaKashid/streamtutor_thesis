from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

from mistralai.client import Mistral

from tts.base_engine import BaseTTSEngine, StreamingTextChunk
from tts.config import (
    MISTRAL_API_KEY,
    VOXTRAL_MODEL,
    VOXTRAL_RESPONSE_FORMAT,
    VOXTRAL_LANGUAGE,
    VOXTRAL_VOICE_ID,
    VOXTRAL_SAMPLE_RATE,
    VOXTRAL_SPEED,
)


@dataclass
class VoxtralConfig:
    model: str = VOXTRAL_MODEL
    response_format: str = VOXTRAL_RESPONSE_FORMAT
    language: str = VOXTRAL_LANGUAGE
    voice_id: str = VOXTRAL_VOICE_ID
    sample_rate: int = VOXTRAL_SAMPLE_RATE
    speed: float = VOXTRAL_SPEED


class VoxtralEngine(BaseTTSEngine):
    """
    Mistral-hosted Voxtral TTS backend.

    Current implementation:
    - synthesize(): one-shot API TTS
    - stream_text(): chunk-wise pseudo-streaming by synthesizing flushed chunks
    """

    def __init__(self, api_key: Optional[str] = None, cfg: Optional[VoxtralConfig] = None) -> None:
        self.api_key = api_key or MISTRAL_API_KEY
        self.cfg = cfg or VoxtralConfig()
        self._client: Optional[Mistral] = None
        self.last_stream_tts_proc_total: float = 0.0

    def load(self) -> None:
        if self._client is not None:
            return

        if not self.api_key:
            raise RuntimeError(
                "MISTRAL_API_KEY is not set. Set it in your environment before using Voxtral."
            )

        self._client = Mistral(api_key=self.api_key)

    def supports_streaming_input(self) -> bool:
        return True

    def _save_wav(self, save_to: Path, wav_t: torch.Tensor, sr: int) -> None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        wav_np = wav_t.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        sf.write(str(save_to), wav_np, sr)

    def _encode_ref_audio(self, voice_prompt: Path) -> str:
        p = Path(voice_prompt)
        if not p.exists():
            raise FileNotFoundError(f"Voice prompt not found: {p}")
        return base64.b64encode(p.read_bytes()).decode("utf-8")

    def synthesize(
        self,
        text: str,
        voice_prompt: Optional[Path] = None,
        config: Optional[VoxtralConfig] = None,
        save_to: Optional[Path] = None,
    ) -> Tuple[torch.Tensor, int]:
        if self._client is None:
            self.load()

        cfg = config or self.cfg
        text = (text or "").strip()
        if not text:
            raise ValueError("Empty text passed to VoxtralEngine.synthesize().")

        kwargs = {
            "model": cfg.model,
            "input": text,
            "response_format": cfg.response_format,
        }

        # Mistral requires either voice_id or ref_audio
        if cfg.voice_id:
            kwargs["voice_id"] = cfg.voice_id
        elif voice_prompt is not None:
            kwargs["ref_audio"] = self._encode_ref_audio(voice_prompt)
        else:
            raise RuntimeError(
                "Voxtral requires either VOXTRAL_VOICE_ID or --voice_prompt. "
                "No voice was provided."
            )

        response = self._client.audio.speech.complete(**kwargs)

        if not hasattr(response, "audio_data") or not response.audio_data:
            raise RuntimeError("Voxtral API returned no audio_data.")

        audio_bytes = base64.b64decode(response.audio_data)

        wav_np, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if wav_np.ndim == 2:
            wav_np = wav_np.mean(axis=1)

        wav_t = torch.from_numpy(wav_np.astype(np.float32, copy=False)).unsqueeze(0)

        if save_to is not None:
            self._save_wav(save_to, wav_t, sr)

        return wav_t, sr

    def stream_text(
        self,
        text_chunks: Iterable[StreamingTextChunk],
        voice_prompt: Optional[Path] = None,
        config: Optional[VoxtralConfig] = None,
    ) -> Iterator[tuple[np.ndarray, int]]:
        cfg = config or self.cfg
        buffer = ""
        self.last_stream_tts_proc_total = 0.0

        from time import perf_counter

        for ev in text_chunks:
            if ev.text:
                buffer += ev.text

            if not (ev.flush or ev.is_final):
                continue

            text = buffer.strip()
            buffer = ""

            if not text:
                continue

            t0 = perf_counter()
            wav_t, sr = self.synthesize(
                text=text,
                voice_prompt=voice_prompt,
                config=cfg,
                save_to=None,
            )
            self.last_stream_tts_proc_total += perf_counter() - t0

            wav_np = wav_t.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

            # ~120ms pieces for your existing audio queue/playback behavior
            chunk_samples = max(1, int(sr * 0.12))
            for start in range(0, len(wav_np), chunk_samples):
                piece = wav_np[start:start + chunk_samples]
                if piece.size == 0:
                    continue
                yield piece[:, None], sr