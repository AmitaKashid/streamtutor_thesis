from __future__ import annotations

import base64
import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

import numpy as np
import requests
import torch
import websocket
from websocket import WebSocketTimeoutException

from tts.base_engine import BaseTTSEngine, StreamingTextChunk


@dataclass
class Qwen3StreamingConfig:
    health_url: str = "http://127.0.0.1:8094/health"
    stream_url: str = "ws://127.0.0.1:8094/stream"
    speaker: str = "Aiden"
    language: str = "en"
    timeout_s: float = 180.0
    chunk_size: int = 4


class Qwen3StreamingEngine(BaseTTSEngine):
    def __init__(self, cfg: Optional[Qwen3StreamingConfig] = None) -> None:
        self.cfg = cfg or Qwen3StreamingConfig()
        self._cancel_event = None
        self._session = threading.local()

    @contextmanager
    def streaming_session(self):
        """Reuse one socket for ordered segments in this synthesis thread."""
        if getattr(self._session, "active", False):
            raise RuntimeError("Nested Qwen streaming sessions are not supported")
        self._session.active = True
        self._session.socket = None
        try:
            yield
        finally:
            sock = self._session.socket
            self._session.socket = None
            self._session.active = False
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

    def set_cancel_event(self, event) -> None:
        self._cancel_event = event

    def _cancelled(self) -> bool:
        return self._cancel_event is not None and self._cancel_event.is_set()

    def load(self) -> None:
        try:
            r = requests.get(self.cfg.health_url, timeout=5)
            r.raise_for_status()
            data = r.json()
            if not data.get("ok"):
                raise RuntimeError(f"Qwen stream server not ready: {data}")
            print(f"[Qwen3Streaming] Connected: {data}")
        except Exception as exc:
            raise RuntimeError(
                f"Qwen streaming server is not available at {self.cfg.health_url}. "
                f"Start it with: uvicorn qwen_stream_server:app --host 127.0.0.1 --port 8094"
            ) from exc

    def supports_streaming_input(self) -> bool:
        return True

    def _decode_pcm16(self, b64: str) -> np.ndarray:
        raw = base64.b64decode(b64)
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return np.clip(pcm, -1.0, 1.0).astype(np.float32, copy=False)

    def _stream_one_text(self, text: str, cfg) -> Iterator[tuple[np.ndarray, int]]:
        text = (text or "").strip()
        if not text:
            return

        if self._cancelled():
            return

        speaker = getattr(cfg, "speaker", self.cfg.speaker)
        language = getattr(cfg, "language", self.cfg.language)
        chunk_size = getattr(cfg, "chunk_size", self.cfg.chunk_size)

        persistent = getattr(self._session, "active", False)
        ws = getattr(self._session, "socket", None) if persistent else None
        if ws is None:
            ws = websocket.create_connection(self.cfg.stream_url, timeout=5)
            ws.settimeout(0.2)
            if persistent:
                self._session.socket = ws
        idle_since = time.monotonic()
        idle_timeout = max(1.0, float(getattr(cfg, "timeout_s", self.cfg.timeout_s)))
        completed = False

        try:
            ws.send(json.dumps({
                "type": "synthesize",
                "text": text,
                "speaker": speaker,
                "language": language,
                "chunk_size": chunk_size,
            }))

            while True:
                if self._cancelled():
                    print("[Qwen3Streaming] cancelled; closing stream socket")
                    break

                try:
                    raw = ws.recv()
                except WebSocketTimeoutException:
                    if time.monotonic() - idle_since >= idle_timeout:
                        raise TimeoutError(f"Qwen TTS sent no data for {idle_timeout:.1f} seconds")
                    continue

                if not raw:
                    raise RuntimeError("Qwen TTS disconnected before completing the speech segment")
                idle_since = time.monotonic()
                msg = json.loads(raw)
                msg_type = msg.get("type")

                if msg_type == "audio_start":
                    continue

                if msg_type == "audio_chunk":
                    if self._cancelled():
                        break

                    sr = int(msg.get("sample_rate") or 24000)
                    audio = self._decode_pcm16(msg["pcm16_base64"])
                    if audio.size:
                        yield audio[:, None], sr
                    continue

                if msg_type == "audio_done":
                    print(
                        f"[Qwen3Streaming] done | "
                        f"chunks={msg.get('chunks')} | "
                        f"ttfa={msg.get('ttfa_s')} | "
                        f"rtf={msg.get('rtf')}"
                    )
                    completed = True
                    break

                if msg_type == "error":
                    raise RuntimeError(msg.get("message", "Unknown Qwen streaming server error."))

        finally:
            # A failed, abandoned, or interrupted stream must never be reused.
            if not persistent or not completed:
                if persistent:
                    self._session.socket = None
                try:
                    ws.close()
                except Exception:
                    pass

    def stream_text(
        self,
        text_chunks: Iterable[StreamingTextChunk],
        voice_prompt: Optional[Path] = None,
        config=None,
    ) -> Iterator[tuple[np.ndarray, int]]:
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

            for audio, sr in self._stream_one_text(text, cfg):
                yield audio, sr

    def synthesize(
        self,
        text: str,
        voice_prompt: Optional[Path] = None,
        config=None,
        save_to: Optional[Path] = None,
    ) -> Tuple[torch.Tensor, int]:
        cfg = config or self.cfg

        pieces = []
        sr_final = 24000

        for audio, sr in self._stream_one_text(text, cfg):
            sr_final = sr
            pieces.append(audio[:, 0])

        if not pieces:
            raise RuntimeError("Qwen streaming server returned no audio.")

        wav_np = np.concatenate(pieces).astype(np.float32, copy=False)
        return torch.from_numpy(wav_np).unsqueeze(0), sr_final
