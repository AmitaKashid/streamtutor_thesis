from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import requests
import soundfile as sf
import torch

from tts.base_engine import BaseTTSEngine


@dataclass
class Qwen3HTTPConfig:
    base_url: str = "http://127.0.0.1:8093"
    speaker: str = "Aiden"
    language: str = "en"
    timeout_s: float = 180.0


class Qwen3HTTPEngine(BaseTTSEngine):
    def __init__(self, cfg: Optional[Qwen3HTTPConfig] = None) -> None:
        self.cfg = cfg or Qwen3HTTPConfig()

    def load(self) -> None:
        url = f"{self.cfg.base_url.rstrip('/')}/health"
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            data = r.json()
            if not data.get("ok"):
                raise RuntimeError(f"Qwen server not ready: {data}")
            print(f"[Qwen3HTTP] Connected to Qwen server: {data}")
        except Exception as exc:
            raise RuntimeError(
                f"Qwen3 HTTP server is not available at {url}. "
                f"Start it with: uvicorn qwen_tts_server:app --host 127.0.0.1 --port 8093"
            ) from exc

    def supports_streaming_input(self) -> bool:
        # This server returns full WAV per request.
        return False

    def synthesize(
        self,
        text: str,
        voice_prompt: Optional[Path] = None,
        config=None,
        save_to: Optional[Path] = None,
    ) -> Tuple[torch.Tensor, int]:
        cfg = config or self.cfg

        text = (text or "").strip()
        if not text:
            raise ValueError("Empty text passed to Qwen3HTTPEngine.synthesize().")

        speaker = getattr(cfg, "speaker", self.cfg.speaker)
        language = getattr(cfg, "language", self.cfg.language)

        url = f"{self.cfg.base_url.rstrip('/')}/synthesize"

        payload = {
            "text": text,
            "speaker": speaker,
            "language": language,
        }

        print(f"[Qwen3HTTP] synth request | chars={len(text)} | speaker={speaker} | lang={language}")

        r = requests.post(url, json=payload, timeout=self.cfg.timeout_s)
        r.raise_for_status()
        data = r.json()

        wav_bytes = base64.b64decode(data["audio_base64"])
        wav_np, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")

        if wav_np.ndim == 2:
            if wav_np.shape[1] == 1:
                wav_np = wav_np[:, 0]
            else:
                wav_np = wav_np.mean(axis=1)

        wav_np = np.asarray(wav_np, dtype=np.float32)
        wav_np = np.clip(wav_np, -1.0, 1.0)

        if save_to is not None:
            save_to = Path(save_to)
            save_to.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(save_to), wav_np, int(sr))

        print(
            f"[Qwen3HTTP] synth response | "
            f"server_synth={data.get('synth_time_s'):.3f}s | "
            f"duration={data.get('duration_s'):.3f}s | sr={sr}"
        )

        return torch.from_numpy(wav_np).unsqueeze(0), int(sr)