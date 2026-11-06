from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class GenerationResult:
    text: str
    total_seconds: float
    prompt_tokens: int
    new_tokens: int
    tokens_per_second: float
    raw_text: Optional[str] = None
    response_protocol: Optional[str] = None
    protocol_error: Optional[str] = None


class LLMProvider:
    def generate(self, user_text: str) -> GenerationResult:
        raise NotImplementedError

    def stream(self, user_text: str) -> Iterator[str]:
        """
        Yield incremental text chunks (not necessarily single tokens).
        Implementations should yield small strings as the model generates.
        """
        raise NotImplementedError
