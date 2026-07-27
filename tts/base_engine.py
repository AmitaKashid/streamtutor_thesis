from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

import numpy as np
import torch


@dataclass
class StreamingTextChunk:
    text: str
    flush: bool = False
    is_final: bool = False


class BaseTTSEngine(ABC):
    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice_prompt: Optional[Path] = None,
        config=None,
        save_to: Optional[Path] = None,
    ) -> Tuple[torch.Tensor, int]:
        raise NotImplementedError

    def supports_streaming_input(self) -> bool:
        return False

    def stream_text(
        self,
        text_chunks: Iterable[StreamingTextChunk],
        voice_prompt: Optional[Path] = None,
        config=None,
    ) -> Iterator[tuple[np.ndarray, int]]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support streaming input")
