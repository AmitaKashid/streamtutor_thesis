from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Iterable, Iterator


@dataclass
class ProsodyChunkerConfig:
    first_target_chars: int = 80
    later_target_chars: int = 180
    min_sentence_chars: int = 45
    min_comma_chars: int = 60
    first_min_sentence_chars: int | None = None
    first_min_comma_chars: int | None = None
    second_target_chars: int | None = None
    second_max_chars: int | None = None
    max_chars: int = 260
    max_wait_s: float = 0.5


class ProsodyChunker:
    STRONG_PUNCT = {".", "?", "!"}
    SOFT_PUNCT = {",", ";", ":"}
    # Common abbreviations and dotted initials do not end a spoken sentence.
    # A list alone cannot disambiguate every abbreviation; this avoids common
    # false boundaries without changing the text sent to the model or browser.
    _abbreviation = re.compile(r"(?:^|\s)(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|fig|eq)\.$", re.IGNORECASE)
    _initials = re.compile(r"(?:^|\s)(?:[a-z]\.){2,}$", re.IGNORECASE)
    _closing_punctuation = re.compile(r"([.!?,;:])[\"'\)\]\}*]*$")

    def __init__(self, cfg: ProsodyChunkerConfig):
        self.cfg = cfg
        self._buf = ""
        self._started = False
        self._segment_count = 0
        self._last_flush_t = time.perf_counter()

    def _target_chars(self) -> int:
        if self._segment_count == 1 and self.cfg.second_target_chars is not None:
            return self.cfg.second_target_chars
        return self.cfg.later_target_chars if self._started else self.cfg.first_target_chars

    def _flush(self) -> str:
        text = self._buf.strip()
        self._buf = ""
        self._started = True
        self._segment_count += 1
        self._last_flush_t = time.perf_counter()
        return text

    def _ends_with_abbreviation(self) -> bool:
        ending = self._buf.rstrip().rstrip("\"')]}*")
        return bool(self._abbreviation.search(ending) or self._initials.search(ending))

    def _should_flush_on_punct(self, ch: str) -> bool:
        buf_len = len(self._buf.strip())

        if ch == "." and self._ends_with_abbreviation():
            return False

        if ch in self.STRONG_PUNCT:
            threshold = self.cfg.min_sentence_chars
            if not self._started and self.cfg.first_min_sentence_chars is not None:
                threshold = self.cfg.first_min_sentence_chars
            return buf_len >= threshold

        if ch in self.SOFT_PUNCT:
            threshold = self.cfg.min_comma_chars
            if not self._started and self.cfg.first_min_comma_chars is not None:
                threshold = self.cfg.first_min_comma_chars
            return buf_len >= threshold

        return False

    def _should_force_flush(self) -> bool:
        buf_len = len(self._buf.strip())
        age = time.perf_counter() - self._last_flush_t

        max_chars = (self.cfg.second_max_chars if self._segment_count == 1
                     and self.cfg.second_max_chars is not None else self.cfg.max_chars)
        if buf_len >= max_chars:
            return True

        if buf_len >= self._target_chars() and age >= self.cfg.max_wait_s:
            return True

        return False

    def feed(self, pieces: Iterable[str]) -> Iterator[str]:
        """Yield completed phrases, retaining partial words across model events.

        Punctuation is considered when the following whitespace arrives. The
        final phrase is emitted by flush() when model generation ends.
        """
        for piece in pieces:
            if not piece:
                continue

            for ch in piece:
                self._buf += ch

                # Wait for a word boundary before considering punctuation.
                # A period seen in isolation may be inside 3.14 or a URL;
                # the next token can arrive in a separate model event.
                if not ch.isspace():
                    continue
                ending = self._closing_punctuation.search(self._buf.rstrip())
                if ending and ending.group(1) == "." and self._ends_with_abbreviation():
                    continue
                if ending and self._should_flush_on_punct(ending.group(1)):
                    text = self._flush()
                    if text:
                        yield text
                    continue

                if self._should_force_flush():
                    # Never split a word, number, or URL merely because the
                    # timer or character target was reached mid-token.
                    text = self._flush()
                    if text:
                        yield text

    def flush(self) -> str:
        return self._flush() if self._buf.strip() else ""
