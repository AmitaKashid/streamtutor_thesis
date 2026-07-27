from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass
class GenerationResult:
    text: str
    total_seconds: float
    prompt_tokens: int
    new_tokens: int
    tokens_per_second: float


class LLMProvider:
    def generate(self, user_text: str) -> GenerationResult:
        raise NotImplementedError

    def stream(self, user_text: str) -> Iterator[str]:
        """
        Yield incremental text chunks (not necessarily single tokens).
        Implementations should yield small strings as the model generates.
        """
        raise NotImplementedError
