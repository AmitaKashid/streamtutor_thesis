from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
except Exception:
    sd = None


@dataclass
class AudioOutputConfig:
    blocksize: int = 2048
    queue_max_chunks: int = 16


class AudioOutput:
    """
    Interruptible continuous audio output stream with queue.
    Chunks must be float32 numpy arrays shaped [T, C].
    """

    def __init__(self, sr: int, channels: int, cfg: Optional[AudioOutputConfig] = None) -> None:
        if sd is None:
            raise RuntimeError("sounddevice not installed/available; cannot do playback.")

        self.sr = sr
        self.channels = channels
        self.cfg = cfg or AudioOutputConfig()

        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=self.cfg.queue_max_chunks)
        self._cur: np.ndarray = np.zeros((0, channels), dtype=np.float32)

        self._stop = threading.Event()
        self._drained = threading.Event()
        self._drained.set()
        self._lock = threading.Lock()

        self._stream = sd.OutputStream(
            samplerate=self.sr,
            channels=self.channels,
            dtype="float32",
            blocksize=self.cfg.blocksize,
            callback=self._callback,
        )

    def start(self) -> None:
        self._stop.clear()
        self._drained.clear()
        self._stream.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._stream.stop()
        finally:
            self._stream.close()

    def enqueue(self, chunk: np.ndarray) -> None:
        if chunk.size == 0:
            return
        if chunk.ndim == 1:
            chunk = chunk[:, None]
        if chunk.shape[1] != self.channels:
            raise ValueError(f"channels mismatch: got {chunk.shape[1]} expected {self.channels}")
        self._drained.clear()
        self._q.put(chunk, block=True)

    def wait_done(self) -> None:
        self._drained.wait()

    def interrupt(self) -> None:
        """
        Stop current playback immediately and flush queued audio.
        """
        with self._lock:
            self._cur = np.zeros((0, self.channels), dtype=np.float32)
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
            self._drained.set()

    def _callback(self, outdata, frames, time_info, status):
        if self._stop.is_set():
            outdata[:] = 0
            return

        with self._lock:
            while len(self._cur) < frames:
                try:
                    nxt = self._q.get_nowait()
                    self._cur = np.concatenate([self._cur, nxt], axis=0)
                except queue.Empty:
                    break

            if len(self._cur) >= frames:
                out = self._cur[:frames]
                self._cur = self._cur[frames:]
                outdata[:] = out
            else:
                if len(self._cur) > 0:
                    outdata[: len(self._cur)] = self._cur
                    outdata[len(self._cur) :] = 0
                    self._cur = np.zeros((0, self.channels), dtype=np.float32)
                else:
                    outdata[:] = 0

            if self._q.empty() and len(self._cur) == 0:
                self._drained.set()