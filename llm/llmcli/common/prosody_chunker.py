from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Iterator


@dataclass
class ProsodyChunkerConfig:
    first_target_chars: int = 80
    later_target_chars: int = 180
    min_sentence_chars: int = 45
    min_comma_chars: int = 60
    max_chars: int = 260
    max_wait_s: float = 0.5


class ProsodyChunker:
    STRONG_PUNCT = {".", "?", "!"}
    SOFT_PUNCT = {",", ";", ":"}

    def __init__(self, cfg: ProsodyChunkerConfig):
        self.cfg = cfg
        self._buf = ""
        self._started = False
        self._last_flush_t = time.perf_counter()

    def _target_chars(self) -> int:
        return self.cfg.later_target_chars if self._started else self.cfg.first_target_chars

    def _flush(self) -> str:
        text = self._buf.strip()
        self._buf = ""
        self._started = True
        self._last_flush_t = time.perf_counter()
        return text

    def _should_flush_on_punct(self, ch: str) -> bool:
        buf_len = len(self._buf.strip())

        if ch in self.STRONG_PUNCT:
            return buf_len >= self.cfg.min_sentence_chars

        if ch in self.SOFT_PUNCT:
            return buf_len >= self.cfg.min_comma_chars

        return False

    def _should_force_flush(self) -> bool:
        buf_len = len(self._buf.strip())
        age = time.perf_counter() - self._last_flush_t

        if buf_len >= self.cfg.max_chars:
            return True

        if buf_len >= self._target_chars() and age >= self.cfg.max_wait_s:
            return True

        return False

    def feed(self, pieces: Iterable[str]) -> Iterator[str]:
        for piece in pieces:
            if not piece:
                continue

            for ch in piece:
                self._buf += ch

                if self._should_flush_on_punct(ch):
                    text = self._flush()
                    if text:
                        yield text
                    continue

                if self._should_force_flush():
                    candidate = self._buf.strip()
                    split_idx = candidate.rfind(" ")
                    if split_idx > 20:
                        out = candidate[:split_idx].strip()
                        rest = candidate[split_idx + 1:].strip()
                        self._buf = rest
                        self._started = True
                        self._last_flush_t = time.perf_counter()
                        if out:
                            yield out
                    else:
                        text = self._flush()
                        if text:
                            yield text

    def flush(self) -> str:
        return self._flush() if self._buf.strip() else ""