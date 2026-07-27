"""
Qwen3-TTS engine — wraps the `qwen-tts` package (Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice).

Supports:
- 9 built-in preset speakers (Vivian, Serena, Ryan, Aiden, Ono_Anna, Sohee, Uncle_Fu, Dylan, Eric)
- 10 languages (en, zh, ja, ko, de, fr, ru, pt, es, it)
- Flush-level streaming via stream_text() — same contract as KokoroEngine
- No sox dependency; audio decode done via soundfile / torchaudio
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Iterable, Iterator, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import os
import shutil
from tts.base_engine import BaseTTSEngine, StreamingTextChunk


# ---------------------------------------------------------------------------
# Speaker and language catalogs
# ---------------------------------------------------------------------------

QWEN3_TTS_SPEAKERS: List[str] = [
    "Vivian",
    "Serena",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
    "Uncle_Fu",
    "Dylan",
    "Eric",
]

QWEN3_TTS_LANGUAGES: List[str] = [
    "en",
    "zh",
    "ja",
    "ko",
    "de",
    "fr",
    "ru",
    "pt",
    "es",
    "it",
]
QWEN3_LANGUAGE_NAME_MAP = {
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
    """
    Qwen3-TTS invokes SoX through subprocess. On Windows, VS Code / PowerShell
    may not inherit the same PATH where sox works in cmd.exe.
    """
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
            print(f"[Qwen3TTS] Added SoX to PATH: {folder}")
            return

    print("[Qwen3TTS WARNING] SoX not found on PATH. Qwen3-TTS may fail during audio processing.")

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class Qwen3TTSConfig:
    model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    speaker: str = "Vivian"
    language: str = "en"
    speed: float = 1.0
    sample_rate: int = 24000
    stream_audio_chunk_ms: int = 120


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class Qwen3TTSEngine(BaseTTSEngine):
    """
    Qwen3-TTS local engine using the official `qwen-tts` Python package.

    Two modes:
    - synthesize(): one-shot text -> full waveform
    - stream_text(): flush-level incremental TTS from StreamingTextChunk events
    """

    def __init__(self, cfg: Optional[Qwen3TTSConfig] = None) -> None:
        self.cfg = cfg or Qwen3TTSConfig()
        self._model = None
        self.last_stream_tts_proc_total: float = 0.0

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        if self._model is not None:
            return

        print(f"[Qwen3TTS] Loading model: {self.cfg.model_name}")

        ensure_sox_on_path()

        try:
            from qwen_tts import Qwen3TTSModel  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "qwen-tts package not found. Install it with: pip install qwen-tts"
            ) from exc

        load_kwargs = {}

        if torch.cuda.is_available():
            load_kwargs["device_map"] = "cuda:0"

            # Use float16 first on RTX 4070. It is usually safer for local inference
            # than bfloat16 when the generated audio sounds unstable/distorted.
            load_kwargs["dtype"] = torch.float32

            print(f"[Qwen3TTS] CUDA available: {torch.cuda.get_device_name(0)}")
            print("[Qwen3TTS] Loading on cuda:0 with float16")
        else:
            load_kwargs["device_map"] = "cpu"
            load_kwargs["dtype"] = torch.float32
            print("[Qwen3TTS WARNING] CUDA is not available. Qwen3-TTS will be very slow on CPU.")

        self._model = Qwen3TTSModel.from_pretrained(
            self.cfg.model_name,
            **load_kwargs,
        )
        print(f"[Qwen3TTS] Model loaded. Speaker: {self.cfg.speaker}, Language: {self.cfg.language}")

        print("[Qwen3TTS] Warm-up skipped for realtime WebSocket mode.")
        #try:
        #    self._run_inference("Hello.", self.cfg.speaker, self.cfg.language)
        #    print("[Qwen3TTS] Warm-up complete.")
        # except Exception as exc:
        #    print(f"[Qwen3TTS] Warm-up failed (non-fatal): {type(exc).__name__}: {exc}")
    # ------------------------------------------------------------------
    # Internal inference helper
    # ------------------------------------------------------------------

    def _run_inference(self, text: str, speaker: str, language: str) -> Tuple[np.ndarray, int]:
        """
        Run Qwen3-TTS CustomVoice inference and return (wav_np_float32, sample_rate).

        qwen-tts==0.1.1 does NOT expose model.generate().
        For Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice, the correct API is:
            model.generate_custom_voice(...)
        """
        if self._model is None:
            raise RuntimeError("Qwen3TTS model is not loaded.")

        text = (text or "").strip()
        if not text:
            raise ValueError("Empty text passed to Qwen3TTS inference.")

        qwen_language = QWEN3_LANGUAGE_NAME_MAP.get(language, language)
        instruct = ""

        if not hasattr(self._model, "generate_custom_voice"):
            available = [name for name in dir(self._model) if name.startswith("generate")]
            raise RuntimeError(
                "Loaded Qwen3TTSModel does not expose generate_custom_voice(). "
                f"Available generation methods: {available}"
            )

        with torch.inference_mode():
            wavs, sr = self._model.generate_custom_voice(
                text=text,
                language=qwen_language,
                speaker=speaker,
            )

        # qwen-tts returns wavs as a list/array batch. For one input, take first item.
        if isinstance(wavs, (list, tuple)):
            if not wavs:
                raise RuntimeError("Qwen3TTS returned an empty wav list.")
            wav = wavs[0]
        else:
            wav = wavs

        if torch.is_tensor(wav):
            wav_np = wav.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            wav_np = np.asarray(wav, dtype=np.float32)

        # Normalise to mono [T]
        if wav_np.ndim == 2:
            if wav_np.shape[0] == 1:
                wav_np = wav_np[0]
            elif wav_np.shape[1] == 1:
                wav_np = wav_np[:, 0]
            else:
                wav_np = wav_np.mean(axis=1)

        wav_np = np.squeeze(wav_np).astype(np.float32, copy=False)

        if wav_np.size == 0:
            raise RuntimeError("Qwen3TTS returned empty audio.")

        # If the package ever returns int-style PCM values, normalize instead of clipping.
        wav_np = np.nan_to_num(wav_np, nan=0.0, posinf=0.0, neginf=0.0)

        max_abs = float(np.max(np.abs(wav_np)))
        print(f"[Qwen3TTS] raw audio stats | max_abs={max_abs:.4f} | sr={sr} | samples={len(wav_np)}")

        # Qwen normally returns float waveform already. Do not loudness-normalize it.
        # Only scale if it is clearly integer-style PCM.
        if max_abs > 10.0:
            wav_np = wav_np / max_abs

        wav_np = np.clip(wav_np, -1.0, 1.0)
        return wav_np.astype(np.float32, copy=False), int(sr)

    # ------------------------------------------------------------------
    # BaseTTSEngine API
    # ------------------------------------------------------------------

    def supports_streaming_input(self) -> bool:
        return False

    def synthesize(
        self,
        text: str,
        voice_prompt: Optional[Path] = None,
        config=None,
        save_to: Optional[Path] = None,
    ) -> Tuple[torch.Tensor, int]:
        if self._model is None:
            self.load()

        cfg = config or self.cfg
        text = (text or "").strip()
        if not text:
            raise ValueError("Empty text passed to Qwen3TTSEngine.synthesize()")

        speaker = getattr(cfg, "speaker", self.cfg.speaker)
        language = getattr(cfg, "language", self.cfg.language)

        t0 = perf_counter()
        print(f"[Qwen3TTS] synth start | chars={len(text)} | speaker={speaker} | lang={language}")
        wav_np, sr = self._run_inference(text, speaker, language)
        duration_s = len(wav_np) / max(1, sr)

        if duration_s > 20.0:
            print(
                f"[Qwen3TTS WARNING] Abnormally long audio: "
                f"text_chars={len(text)}, duration={duration_s:.2f}s. Trimming to 20s."
            )
            wav_np = wav_np[: int(sr * 20.0)]
        dt = perf_counter() - t0
        print(f"[Qwen3TTS] synth done  | chars={len(text)} | sec={dt:.3f} | samples={len(wav_np)} | sr={sr}")

        if save_to is not None:
            save_to = Path(save_to)
            save_to.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(save_to), wav_np, sr)

        wav_t = torch.from_numpy(wav_np).unsqueeze(0)  # [1, T]
        return wav_t, sr

    def stream_text(
        self,
        text_chunks: Iterable[StreamingTextChunk],
        voice_prompt: Optional[Path] = None,
        config=None,
    ) -> Iterator[tuple[np.ndarray, int]]:
        cfg = config or self.cfg
        buf = ""
        self.last_stream_tts_proc_total = 0.0

        chunk_ms = getattr(cfg, "stream_audio_chunk_ms", 120)

        for ev in text_chunks:
            if ev.text:
                buf += ev.text

            if not (ev.flush or ev.is_final):
                continue

            text = buf.strip()
            buf = ""
            if not text:
                continue

            t0 = perf_counter()
            wav_t, sr = self.synthesize(text=text, voice_prompt=voice_prompt, config=cfg)
            self.last_stream_tts_proc_total += perf_counter() - t0

            wav_np = wav_t.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

            chunk_samples = max(1, int(sr * chunk_ms / 1000))
            for start in range(0, len(wav_np), chunk_samples):
                piece = wav_np[start : start + chunk_samples]
                if piece.size == 0:
                    continue
                yield piece[:, None], sr
