# tts/chatterbox_engine.py
"""
Low-level Chatterbox TTS engine wrapper.

This module encapsulates direct interaction with the Chatterbox model.
It exposes a simple, synchronous API:
    text (and optional voice prompt) → waveform tensor + sample rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch

# torchaudio is optional (often painful to install on Windows)
try:
    import torchaudio  # optional
except Exception:
    torchaudio = None

# soundfile is a reliable fallback for saving wav
try:
    import soundfile as sf  # pip install soundfile
except Exception:
    sf = None

# old
#from chatterbox.tts import ChatterboxTTS

# new
from chatterbox.tts_turbo import ChatterboxTurboTTS
import logging
from tts.config import (
    TTS_DEVICE,
    TTS_TEMPERATURE,
    TTS_EXAGGERATION,
    TTS_CFG_WEIGHT,
    TTS_REPETITION_PENALTY,
    TTS_TOP_P,
    TTS_MIN_P,
    TTS_OUTPUT_DIR,
)


@dataclass
class TTSConfig:
    """Configuration parameters for a single TTS synthesis call."""
    temperature: float = TTS_TEMPERATURE
    exaggeration: float = TTS_EXAGGERATION
    cfg_weight: float = TTS_CFG_WEIGHT
    repetition_penalty: float = TTS_REPETITION_PENALTY
    top_p: float = TTS_TOP_P
    min_p: float = TTS_MIN_P


class ChatterboxEngine:
    """
    Chatterbox-based TTS engine.

    Responsibilities:
      - Load and hold the Chatterbox model in memory.
      - Provide a method to synthesize waveform tensors from text.
      - Optionally save generated audio to disk.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or TTS_DEVICE
        self._model: Optional[ChatterboxTurboTTS] = None

    def load(self) -> None:
        """Load model once (lazy)."""
        if self._model is not None:
            return

        logging.getLogger("chatterbox.tts_turbo").setLevel(logging.ERROR)
        logging.getLogger("chatterbox").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("diffusers").setLevel(logging.ERROR)

        print(f"[ChatterboxEngine] Loading model on device: {self.device}")
        self._model = ChatterboxTurboTTS.from_pretrained(device=self.device)

        # warm-up to reduce first real chunk latency
        
        _ = self._model.generate(
                text="Hello.",
                temperature=TTS_TEMPERATURE,
            )

        print("[ChatterboxEngine] Model loaded successfully.")

    @property
    def sample_rate(self) -> int:
        if self._model is None:
            raise RuntimeError("Chatterbox model has not been loaded yet.")
        return self._model.sr

    def _save_wav(self, save_to: Path, wav: torch.Tensor, sr: int) -> None:
        """Save wav to disk using torchaudio if available, else soundfile fallback."""
        save_to.parent.mkdir(parents=True, exist_ok=True)

        # Ensure shape: [channels, time]
        if isinstance(wav, torch.Tensor):
            wav_t = wav.detach().cpu()
            if wav_t.dim() == 1:
                wav_t = wav_t.unsqueeze(0)
        else:
            # unexpected, but handle defensively
            wav_t = torch.as_tensor(wav).detach().cpu()
            if wav_t.dim() == 1:
                wav_t = wav_t.unsqueeze(0)

        if torchaudio is not None:
            torchaudio.save(str(save_to), wav_t, sample_rate=sr)
            return

        if sf is None:
            raise RuntimeError(
                "Cannot save audio: torchaudio is not installed and soundfile is not installed. "
                "Install one of them: pip install soundfile  (recommended) or install torchaudio."
            )

        # soundfile expects shape [time, channels]
        wav_np = wav_t.transpose(0, 1).numpy()
        sf.write(str(save_to), wav_np, sr)

    def synthesize(
        self,
        text: str,
        voice_prompt: Optional[Path] = None,
        config: Optional[TTSConfig] = None,
        save_to: Optional[Path] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Synthesize speech for the given text and optionally save audio.

        Returns:
            wav: torch.Tensor [channels, time]
            sr:  int
        """
        # Auto-load so the caller doesn't need to remember it
        if self._model is None:
            self.load()

        cfg = config or TTSConfig()

        
        wav = self._model.generate(
                text=text,
                audio_prompt_path=str(voice_prompt) if voice_prompt else None,
                temperature=cfg.temperature,
            )

        sr = self._model.sr

        if save_to is not None:
            self._save_wav(save_to, wav, sr)
            print(f"[ChatterboxEngine] Saved synthesized audio to {save_to}")

        return wav, sr
