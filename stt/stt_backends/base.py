# stt_backends/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from stt.stt_types import TranscriptResult


class STTBackend(ABC):
    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,           # float32 mono
        sample_rate: int,
        language: Optional[str] = None,
        context: Optional[str] = None,
        return_words: bool = False,
    ) -> TranscriptResult:
        """Convert audio to text. Must be implemented by every backend."""
        raise NotImplementedError
