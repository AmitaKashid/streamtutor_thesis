from __future__ import annotations

import io
import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional
from urllib.parse import urljoin

import numpy as np
import requests

from tts.base_engine import BaseTTSEngine, StreamingTextChunk
from tts.config import (
    CHATTERBOX_STREAM_BASE_URL,
    CHATTERBOX_STREAM_BUFFER_SIZE,
    CHATTERBOX_STREAM_CFG_WEIGHT,
    CHATTERBOX_STREAM_CHUNK_SIZE,
    CHATTERBOX_STREAM_CONNECT_TIMEOUT_S,
    CHATTERBOX_STREAM_ENDPOINT,
    CHATTERBOX_STREAM_EXAGGERATION,
    CHATTERBOX_STREAM_QUALITY,
    CHATTERBOX_STREAM_READ_TIMEOUT_S,
    CHATTERBOX_STREAM_REQUEST_CHUNK_BYTES,
    CHATTERBOX_STREAM_STRATEGY,
    CHATTERBOX_STREAM_TEMPERATURE,
    CHATTERBOX_STREAM_UPLOAD_ENDPOINT,
)


@dataclass
class ChatterboxHTTPStreamingConfig:
    base_url: str = CHATTERBOX_STREAM_BASE_URL
    endpoint: str = CHATTERBOX_STREAM_ENDPOINT
    upload_endpoint: str = CHATTERBOX_STREAM_UPLOAD_ENDPOINT
    temperature: float = CHATTERBOX_STREAM_TEMPERATURE
    exaggeration: float = CHATTERBOX_STREAM_EXAGGERATION
    cfg_weight: float = CHATTERBOX_STREAM_CFG_WEIGHT
    streaming_chunk_size: int = CHATTERBOX_STREAM_CHUNK_SIZE
    streaming_strategy: str = CHATTERBOX_STREAM_STRATEGY
    streaming_buffer_size: int = CHATTERBOX_STREAM_BUFFER_SIZE
    streaming_quality: str = CHATTERBOX_STREAM_QUALITY
    connect_timeout_s: float = CHATTERBOX_STREAM_CONNECT_TIMEOUT_S
    read_timeout_s: float = CHATTERBOX_STREAM_READ_TIMEOUT_S
    request_chunk_bytes: int = CHATTERBOX_STREAM_REQUEST_CHUNK_BYTES

    @property
    def timeout(self) -> tuple[float, float]:
        return (self.connect_timeout_s, self.read_timeout_s)


class ChatterboxHTTPStreamingEngine(BaseTTSEngine):
    """
    Streaming client for the local Chatterbox HTTP streaming server.

    Important note:
    The server documented by the user streams audio progressively for a submitted
    text request, but it is not a bidirectional text-input stream. In this app we
    therefore map each flushed LLM text chunk to one streaming HTTP request and
    play the returned WAV stream progressively. This is the closest professional
    fit to the documented server contract while preserving low-latency playback.
    """

    def __init__(self, cfg: Optional[ChatterboxHTTPStreamingConfig] = None) -> None:
        self.cfg = cfg or ChatterboxHTTPStreamingConfig()
        self._session: Optional[requests.Session] = None

    def load(self) -> None:
        if self._session is None:
            self._session = requests.Session()

    def supports_streaming_input(self) -> bool:
        return True

    def synthesize(self, text: str, voice_prompt: Optional[Path] = None, config=None, save_to: Optional[Path] = None):
        raise RuntimeError(
            "ChatterboxHTTPStreamingEngine is for streaming playback. Use stream_text() instead."
        )

    def _endpoint_url(self, voice_prompt: Optional[Path]) -> str:
        endpoint = self.cfg.upload_endpoint if voice_prompt else self.cfg.endpoint
        return urljoin(self.cfg.base_url.rstrip("/") + "/", endpoint.lstrip("/"))

    def _base_payload(self) -> dict:
        return {
            "temperature": self.cfg.temperature,
            "exaggeration": self.cfg.exaggeration,
            "cfg_weight": self.cfg.cfg_weight,
            "streaming_chunk_size": self.cfg.streaming_chunk_size,
            "streaming_strategy": self.cfg.streaming_strategy,
            "streaming_buffer_size": self.cfg.streaming_buffer_size,
            "streaming_quality": self.cfg.streaming_quality,
        }

    def _open_stream(self, text: str, voice_prompt: Optional[Path]):
        assert self._session is not None
        url = self._endpoint_url(voice_prompt)
        payload = {"input": text, **self._base_payload()}

        if voice_prompt is None:
            return self._session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                stream=True,
                timeout=self.cfg.timeout,
            )

        form_data = {k: str(v) for k, v in payload.items()}
        files = {"voice_file": (voice_prompt.name, voice_prompt.open("rb"), "audio/wav")}
        try:
            response = self._session.post(
                url,
                data=form_data,
                files=files,
                stream=True,
                timeout=self.cfg.timeout,
            )
        finally:
            files["voice_file"][1].close()
        return response

    @staticmethod
    def _parse_wav_header(header_bytes: bytes) -> tuple[int, int, np.dtype]:
        if len(header_bytes) < 44:
            raise RuntimeError("Streaming response ended before a full WAV header was received.")

        with wave.open(io.BytesIO(header_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()

        if sample_width == 1:
            dtype = np.int8
        elif sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            raise RuntimeError(f"Unsupported WAV sample width from streaming server: {sample_width}")

        return sample_rate, channels, dtype

    @staticmethod
    def _pcm_to_float32(raw: bytes, channels: int, dtype: np.dtype) -> np.ndarray:
        pcm = np.frombuffer(raw, dtype=dtype)
        if pcm.size == 0:
            return np.empty((0, channels), dtype=np.float32)

        if dtype == np.int8:
            scale = 128.0
        elif dtype == np.int16:
            scale = 32768.0
        elif dtype == np.int32:
            scale = 2147483648.0
        else:
            raise RuntimeError(f"Unsupported PCM dtype: {dtype}")

        pcm = pcm.astype(np.float32) / scale

        if channels > 1:
            pcm = pcm.reshape(-1, channels)
        else:
            pcm = pcm.reshape(-1, 1)
        return pcm

    def _stream_single_text(
        self,
        text: str,
        voice_prompt: Optional[Path] = None,
    ) -> Iterator[tuple[np.ndarray, int]]:
        if not text.strip():
            return

        response = self._open_stream(text=text, voice_prompt=voice_prompt)
        with response:
            response.raise_for_status()

            header = response.raw.read(44)
            sample_rate, channels, dtype = self._parse_wav_header(header)

            while True:
                chunk = response.raw.read(self.cfg.request_chunk_bytes)
                if not chunk:
                    break
                pcm = self._pcm_to_float32(chunk, channels=channels, dtype=dtype)
                if pcm.size == 0:
                    continue
                yield pcm, sample_rate

    def stream_text(
        self,
        text_chunks: Iterable[StreamingTextChunk],
        voice_prompt: Optional[Path] = None,
        config=None,
    ) -> Iterator[tuple[np.ndarray, int]]:
        self.load()
        del config

        for item in text_chunks:
            text = (item.text or "").strip()
            if not text and not item.is_final:
                continue

            if text:
                yield from self._stream_single_text(text=text, voice_prompt=voice_prompt)

            if item.is_final:
                break


# Backward-compatible alias so existing imports do not break.
ElevenLabsStreamingEngine = ChatterboxHTTPStreamingEngine
