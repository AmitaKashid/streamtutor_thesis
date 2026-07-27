# stt_backends/__init__.py
from __future__ import annotations
from typing import Any, Dict

from .base import STTBackend
from .faster_whisper_backend import FasterWhisperBackend
from .mega_asr_http_backend import MegaASRHTTPBackend
from .parakeet_http_backend import ParakeetHTTPBackend


def create_backend(name: str, config: Dict[str, Any]) -> STTBackend:
    n = name.strip().lower()

    if n in ("faster-whisper", "faster_whisper", "whisper"):
        return FasterWhisperBackend(**config)

    if n in {"mega-asr", "mega_asr", "mega-asr-http"}:
        return MegaASRHTTPBackend(
            base_url=config.get("base_url", "http://127.0.0.1:8095"),
            timeout_s=config.get("timeout_s", 120.0),
        )

    if n in {"parakeet", "parakeet-http", "parakeet_http"}:
        return ParakeetHTTPBackend(
            base_url=config.get("base_url", "http://127.0.0.1:8096"),
            timeout_s=config.get("timeout_s", 120.0),
            include_timestamps=config.get("include_timestamps", False),
            should_chunk=config.get("should_chunk", False),
        )

    raise ValueError(f"Unknown backend '{name}'. Available: faster-whisper")
