from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class CrossfadeConfig:
    crossfade_ms: float = 50.0  # 30–80ms is usually safe
    enabled: bool = True


class AudioCrossfader:
    """
    Applies a short crossfade between consecutive audio chunks to reduce clicks/gaps.
    Works on float32 numpy arrays shaped [T, C].
    """

    def __init__(self, sr: int, channels: int, cfg: CrossfadeConfig | None = None) -> None:
        self.sr = sr
        self.channels = channels
        self.cfg = cfg or CrossfadeConfig()
        self._tail: np.ndarray | None = None

    def process(self, chunk: np.ndarray) -> np.ndarray:
        if not self.cfg.enabled:
            return chunk

        if chunk.ndim == 1:
            chunk = chunk[:, None]
        if chunk.shape[1] != self.channels:
            raise ValueError(f"channels mismatch: got {chunk.shape[1]} expected {self.channels}")

        n = int(self.sr * (self.cfg.crossfade_ms / 1000.0))
        if n <= 0:
            return chunk

        if self._tail is None:
            # store tail and pass chunk as-is
            if len(chunk) > n:
                self._tail = chunk[-n:].copy()
                return chunk[:-n]
            else:
                # too short; store all as tail
                self._tail = chunk.copy()
                return np.zeros((0, self.channels), dtype=np.float32)

        # Crossfade tail with head
        head = chunk[:n] if len(chunk) >= n else chunk
        tail = self._tail if len(self._tail) == n else self._tail[-n:]

        m = min(len(head), len(tail), n)
        if m <= 0:
            return chunk

        fade_out = np.linspace(1.0, 0.0, m, dtype=np.float32)[:, None]
        fade_in = 1.0 - fade_out

        mixed = tail[:m] * fade_out + head[:m] * fade_in

        rest = chunk[m:]
        out = np.concatenate([mixed, rest], axis=0)

        # update tail for next time
        if len(out) > n:
            self._tail = out[-n:].copy()
            return out[:-n]
        else:
            self._tail = out.copy()
            return np.zeros((0, self.channels), dtype=np.float32)

    def flush(self) -> np.ndarray:
        if self._tail is None:
            return np.zeros((0, self.channels), dtype=np.float32)
        tail = self._tail
        self._tail = None
        return tail
