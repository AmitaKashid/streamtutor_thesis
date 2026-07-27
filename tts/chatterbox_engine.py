"""
Low-level Chatterbox TTS engine wrapper.

This module encapsulates direct interaction with the Chatterbox model.
It exposes a simple, synchronous API:
    text (and optional voice prompt) -> waveform tensor + sample rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch

try:
    import torchaudio
except Exception:
    torchaudio = None

try:
    import soundfile as sf
except Exception:
    sf = None

from chatterbox.tts_turbo import ChatterboxTurboTTS
import logging

from tts.base_engine import BaseTTSEngine
from tts.config import (
    TTS_DEVICE,
    TTS_TEMPERATURE,
    TTS_EXAGGERATION,
    TTS_CFG_WEIGHT,
    TTS_REPETITION_PENALTY,
    TTS_TOP_P,
    TTS_MIN_P,
)


@dataclass
class TTSConfig:
    temperature: float = TTS_TEMPERATURE
    exaggeration: float = TTS_EXAGGERATION
    cfg_weight: float = TTS_CFG_WEIGHT
    repetition_penalty: float = TTS_REPETITION_PENALTY
    top_p: float = TTS_TOP_P
    min_p: float = TTS_MIN_P


class ChatterboxEngine(BaseTTSEngine):
    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or TTS_DEVICE
        self._model: Optional[ChatterboxTurboTTS] = None

    def load(self) -> None:
        if self._model is not None:
            return

        logging.getLogger("chatterbox.tts_turbo").setLevel(logging.ERROR)
        logging.getLogger("chatterbox").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("diffusers").setLevel(logging.ERROR)

        print(f"[ChatterboxEngine] Loading model on device: {self.device}")
        self._model = ChatterboxTurboTTS.from_pretrained(device=self.device)
        _ = self._model.generate(text="Hello.", temperature=TTS_TEMPERATURE)
        print("[ChatterboxEngine] Model loaded successfully.")

    @property
    def sample_rate(self) -> int:
        if self._model is None:
            raise RuntimeError("Chatterbox model has not been loaded yet.")
        return self._model.sr

    def _save_wav(self, save_to: Path, wav: torch.Tensor, sr: int) -> None:
        save_to.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(wav, torch.Tensor):
            wav_t = wav.detach().cpu()
            if wav_t.dim() == 1:
                wav_t = wav_t.unsqueeze(0)
        else:
            wav_t = torch.as_tensor(wav).detach().cpu()
            if wav_t.dim() == 1:
                wav_t = wav_t.unsqueeze(0)

        if torchaudio is not None:
            torchaudio.save(str(save_to), wav_t, sample_rate=sr)
            return

        if sf is None:
            raise RuntimeError(
                "Cannot save audio: torchaudio is not installed and soundfile is not installed. "
                "Install one of them: pip install soundfile or install torchaudio."
            )

        wav_np = wav_t.transpose(0, 1).numpy()
        sf.write(str(save_to), wav_np, sr)

    def synthesize(
        self,
        text: str,
        voice_prompt: Optional[Path] = None,
        config: Optional[TTSConfig] = None,
        save_to: Optional[Path] = None,
    ) -> Tuple[torch.Tensor, int]:
        if self._model is None:
            self.load()

        cfg = config or TTSConfig()
        wav = self._model.generate(
            text=text,
            audio_prompt_path=str(voice_prompt) if voice_prompt else None,
            temperature=cfg.temperature,
            exaggeration=cfg.exaggeration,
            cfg_weight=cfg.cfg_weight,
            repetition_penalty=cfg.repetition_penalty,
            top_p=cfg.top_p,
            min_p=cfg.min_p,
        )
        sr = self._model.sr

        if save_to is not None:
            self._save_wav(save_to, wav, sr)
            print(f"[ChatterboxEngine] Saved synthesized audio to {save_to}")

        return wav, sr
