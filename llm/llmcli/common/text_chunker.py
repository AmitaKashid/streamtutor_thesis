from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional


# sentence ending punctuation at end-of-buffer (no need for trailing whitespace)
_SENT_END_RE = re.compile(r"[.!?]+(?:\s|$)")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class ChunkerConfig:
    # adaptive chunk sizes: early chunks smaller for fast TTFA
    first_max_chars: int = 80
    mid_max_chars: int = 140
    later_max_chars: int = 220

    # flush if no sentence boundary arrives for a while
    max_wait_s: float = 0.9

    # also flush if buffer gets too big even without sentence boundary
    hard_max_chars: int = 320

    # --- NEW: gating to prevent micro-chunks ---
    # minimum chars required to emit on the timer (prevents "one word" chunks)
    min_chars_timer: int = 60

    # minimum chars required to emit even if sentence boundary appears
    # (prevents tiny chunks like "Yes." or "Okay." unless you really want them)
    min_chars_sentence: int = 30

    # keep text clean
    normalize_whitespace: bool = True


class TextChunker:
    """
    Convert a stream of text pieces (tokens/deltas) into speakable chunks.

    Improvements over the previous version:
    - More robust sentence boundary detection (doesn't require trailing whitespace).
    - Gating thresholds to avoid micro-chunks on timer flush.
    - Adaptive chunk sizes remain (small early, larger later).
    """

    def __init__(self, cfg: Optional[ChunkerConfig] = None) -> None:
        self.cfg = cfg or ChunkerConfig()
        self.buf = ""
        self.last_flush_t = time.perf_counter()
        self.chunk_index = 0

    def _max_chars_now(self) -> int:
        if self.chunk_index == 0:
            return self.cfg.first_max_chars
        if self.chunk_index < 3:
            return self.cfg.mid_max_chars
        return self.cfg.later_max_chars

    def _clean(self, s: str) -> str:
        if not self.cfg.normalize_whitespace:
            return s.strip()
        s = _WHITESPACE_RE.sub(" ", s)
        return s.strip()

    def _find_last_sentence_end(self, s: str) -> int:
        """
        Return an index (exclusive) of the last sentence-ending boundary we can cut at.
        We cut at the end of the *last* match of punctuation+optional space/end.
        If none found, return -1.
        """
        last_end = -1
        for m in _SENT_END_RE.finditer(s):
            last_end = m.end()
        return last_end

    def feed(self, pieces: Iterable[str]) -> Iterator[str]:
        for piece in pieces:
            if not piece:
                continue

            self.buf += piece
            now = time.perf_counter()

            # Safety: hard cap split (never let buffer grow unbounded)
            if len(self.buf) >= self.cfg.hard_max_chars:
                cut = self.cfg.hard_max_chars
                chunk = self._clean(self.buf[:cut])
                self.buf = self.buf[cut:]
                if chunk:
                    self.chunk_index += 1
                    self.last_flush_t = now
                    yield chunk
                continue

            # Prefer sentence boundary (best prosody), but only if big enough
            last_sent_end = self._find_last_sentence_end(self.buf)
            if last_sent_end != -1:
                candidate = self._clean(self.buf[:last_sent_end])
                # Keep remainder in buffer
                remainder = self.buf[last_sent_end:]
                # Emit only if candidate is sufficiently long
                if len(candidate) >= self.cfg.min_chars_sentence:
                    self.buf = remainder
                    self.chunk_index += 1
                    self.last_flush_t = now
                    yield candidate
                    continue

            # Flush if buffer big enough (adaptive size)
            max_chars = self._max_chars_now()
            if len(self.buf) >= max_chars:
                chunk = self._clean(self.buf)
                self.buf = ""
                if chunk:
                    self.chunk_index += 1
                    self.last_flush_t = now
                    yield chunk
                continue

            # Timer flush (responsiveness) — gated to prevent micro-chunks
            if (now - self.last_flush_t) >= self.cfg.max_wait_s:
                cleaned = self._clean(self.buf)
                if len(cleaned) >= self.cfg.min_chars_timer:
                    self.buf = ""
                    self.chunk_index += 1
                    self.last_flush_t = now
                    yield cleaned
                # else: keep buffering until we have enough text

    def flush(self) -> Optional[str]:
        """
        Flush remaining buffer at end of stream.
        This returns whatever is left (even if short), because stream ended.
        """
        chunk = self._clean(self.buf)
        self.buf = ""
        if chunk:
            self.chunk_index += 1
            self.last_flush_t = time.perf_counter()
            return chunk
        return None