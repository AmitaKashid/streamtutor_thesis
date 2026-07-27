# stt_types.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Word:
    start: float
    end: float
    word: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: Optional[List[Word]] = None


@dataclass
class TranscriptResult:
    text: str
    segments: List[Segment]
    language: Optional[str] = None
    engine: Optional[str] = None
