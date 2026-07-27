from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

from kokoro import KPipeline

from tts.base_engine import BaseTTSEngine, StreamingTextChunk


@dataclass
class KokoroConfig:
    voice: str = "af_heart"
    lang_code: str = "a"   # American English in official Kokoro examples
    speed: float = 1.0
    sample_rate: int = 24000
    stream_audio_chunk_ms: int = 120


class KokoroEngine(BaseTTSEngine):
    """
    Local Kokoro TTS engine.

    Two modes:
    - synthesize(): one-shot text -> full waveform
    - stream_text(): flush-level incremental TTS from StreamingTextChunk events
    """

    def __init__(self, cfg: Optional[KokoroConfig] = None) -> None:
        self.cfg = cfg or KokoroConfig()
        self._pipeline: Optional[KPipeline] = None

    def load(self) -> None:
        if self._pipeline is not None:
            return

        # Official Kokoro uses KPipeline(lang_code=...)
        self._pipeline = KPipeline(lang_code=self.cfg.lang_code)

        # Warm-up to reduce first real inference delay
        warmup_text = "Hello."
        try:
            gen = self._pipeline(
                warmup_text,
                voice=self.cfg.voice,
                speed=self.cfg.speed,
            )
            for _gs, _ps, _audio in gen:
                break
        except Exception:
            # warmup failure should not kill app startup
            pass

    def supports_streaming_input(self) -> bool:
        return True

    def _audio_to_numpy(self, audio) -> np.ndarray:
        if isinstance(audio, np.ndarray):
            arr = audio.astype(np.float32, copy=False)
        elif torch.is_tensor(audio):
            arr = audio.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            arr = np.asarray(audio, dtype=np.float32)

        # normalize to 1D mono
        if arr.ndim == 2:
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[1] == 1:
                arr = arr[:, 0]
            else:
                arr = arr.mean(axis=1)

        return arr

    def _save_wav(self, save_to: Path, wav_np: np.ndarray, sr: int) -> None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(save_to), wav_np, sr)

    def synthesize(
        self,
        text: str,
        voice_prompt: Optional[Path] = None,
        config=None,
        save_to: Optional[Path] = None,
    ) -> Tuple[torch.Tensor, int]:
        if self._pipeline is None:
            self.load()

        cfg = config or self.cfg
        text = (text or "").strip()
        if not text:
            raise ValueError("Empty text passed to KokoroEngine.synthesize()")

        # Kokoro official examples expose a generator from pipeline(...)
        gen = self._pipeline(
            text,
            voice=cfg.voice,
            speed=cfg.speed,
        )

        audio_parts: list[np.ndarray] = []
        for _gs, _ps, audio in gen:
            arr = self._audio_to_numpy(audio)
            if arr.size > 0:
                audio_parts.append(arr)

        if not audio_parts:
            raise RuntimeError("Kokoro returned no audio.")

        wav_np = np.concatenate(audio_parts, axis=0).astype(np.float32, copy=False)
        sr = getattr(cfg, "sample_rate", 24000)

        if save_to is not None:
            self._save_wav(save_to, wav_np, sr)

        # Return torch tensor in same convention as your other engines: [1, T]
        wav_t = torch.from_numpy(wav_np).unsqueeze(0)
        return wav_t, sr

    def stream_text(
        self,
        text_chunks: Iterable[StreamingTextChunk],
        voice_prompt: Optional[Path] = None,
        config=None,
    ) -> Iterator[tuple[np.ndarray, int]]:
        """
        Local incremental TTS.

        Strategy:
        - accumulate text from StreamingTextChunk events
        - synthesize when flush/final=True
        - emit smaller PCM chunks for playback buffering
        """
        cfg = config or self.cfg
        buf = ""

        for ev in text_chunks:
            if ev.text:
                buf += ev.text

            if not (ev.flush or ev.is_final):
                continue

            text = buf.strip()
            buf = ""
            if not text:
                continue

            wav_t, sr = self.synthesize(
                text=text,
                voice_prompt=voice_prompt,
                config=cfg,
                save_to=None,
            )

            wav_np = wav_t.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

            chunk_samples = max(1, int(sr * cfg.stream_audio_chunk_ms / 1000))
            for start in range(0, len(wav_np), chunk_samples):
                piece = wav_np[start:start + chunk_samples]
                if piece.size == 0:
                    continue
                # AudioOutput expects [T, C]
                yield piece[:, None], sr