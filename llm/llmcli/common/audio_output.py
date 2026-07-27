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

@dataclass
class QueuedAudioChunk:
    audio: np.ndarray
    on_start: Optional[callable] = None
    on_end: Optional[callable] = None
    started: bool = False

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

        self._q: "queue.Queue[QueuedAudioChunk]" = queue.Queue(maxsize=self.cfg.queue_max_chunks)
        self._cur_chunk: Optional[QueuedAudioChunk] = None
        self._cur_offset: int = 0

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

    def enqueue(self, chunk: np.ndarray, on_start=None, on_end=None) -> None:
        if chunk.size == 0:
            return
        if chunk.ndim == 1:
            chunk = chunk[:, None]
        if chunk.shape[1] != self.channels:
            raise ValueError(f"channels mismatch: got {chunk.shape[1]} expected {self.channels}")

        self._drained.clear()
        self._q.put(
            QueuedAudioChunk(
                audio=chunk,
                on_start=on_start,
                on_end=on_end,
                started=False,
            ),
            block=True,
        )

    def wait_done(self) -> None:
        self._drained.wait()

    def interrupt(self) -> None:
        """
        Stop current playback immediately and flush queued audio.
        """
        with self._lock:
            self._cur_chunk = None
            self._cur_offset = 0
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

        outdata[:] = 0
        written = 0

        with self._lock:
            while written < frames:
                if self._cur_chunk is None:
                    try:
                        self._cur_chunk = self._q.get_nowait()
                        self._cur_offset = 0
                    except queue.Empty:
                        break

                chunk = self._cur_chunk
                if chunk is None:
                    break

                if not chunk.started:
                    chunk.started = True
                    if chunk.on_start is not None:
                        try:
                            chunk.on_start()
                        except Exception:
                            pass

                audio = chunk.audio
                remaining_chunk = len(audio) - self._cur_offset
                remaining_out = frames - written
                take = min(remaining_chunk, remaining_out)

                outdata[written:written + take] = audio[self._cur_offset:self._cur_offset + take]
                written += take
                self._cur_offset += take

                if self._cur_offset >= len(audio):
                    if chunk.on_end is not None:
                        try:
                            chunk.on_end()
                        except Exception:
                            pass
                    self._cur_chunk = None
                    self._cur_offset = 0

            if self._q.empty() and self._cur_chunk is None:
                self._drained.set()