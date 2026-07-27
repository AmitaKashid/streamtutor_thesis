# tts/tts_queue.py

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import torch

try:
    import sounddevice as sd
except Exception:
    sd = None

from tts.chatterbox_engine import ChatterboxEngine, TTSConfig


@dataclass
class TTSRequest:
    text: str
    request_id: str
    voice_prompt: Optional[Path] = None
    config: Optional[TTSConfig] = None
    save_to: Optional[Path] = None
    play_audio: bool = False


def _default_playback(wav: torch.Tensor, sr: int) -> None:
    if sd is None:
        print("[TTSQueue] sounddevice not available.")
        return

    wav_cpu = wav.detach().cpu()
    if wav_cpu.dim() == 2:
        wav_np = wav_cpu.transpose(0, 1).numpy()
    else:
        wav_np = wav_cpu.numpy()

    sd.play(wav_np, sr)
    sd.wait()


class TTSQueue:
    def __init__(
        self,
        engine: ChatterboxEngine,
        playback_fn: Optional[Callable[[torch.Tensor, int], None]] = None,
    ) -> None:
        self.engine = engine
        self.playback_fn = playback_fn or _default_playback

        self._queue: "queue.Queue[Optional[TTSRequest]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print("[TTSQueue] Worker thread started.")

    def stop(self) -> None:
        self._stop_evt.set()
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=5)
        print("[TTSQueue] Worker thread stopped.")

    def enqueue(self, req: TTSRequest) -> None:
        self._queue.put(req)

    def join(self) -> None:
        self._queue.join()

    def _worker(self) -> None:
        try:
            self.engine.load()
        except Exception as e:
            print(f"[TTSQueue] Engine load failed: {e}")

        while not self._stop_evt.is_set():
            req = self._queue.get()
            if req is None:
                self._queue.task_done()
                break

            try:
                # ---- Measure TTS processing latency ----
                t0 = time.perf_counter()

                wav, sr = self.engine.synthesize(
                    text=req.text,
                    voice_prompt=req.voice_prompt,
                    config=req.config,
                    save_to=req.save_to,
                )

                tts_proc_time = time.perf_counter() - t0

                print(f"[TTS Proc Latency] {tts_proc_time:.3f}s (playback excluded)")

                if req.play_audio:
                    self.playback_fn(wav, sr)

            except Exception as e:
                print(f"[TTSQueue] Error processing request {req.request_id}: {e}")

            finally:
                self._queue.task_done()