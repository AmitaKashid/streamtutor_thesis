"""Minimal shared result types expected by the attached STT backends.

The uploaded ``stt_types(1).py`` file was empty.  Use this file only if the
corresponding file in the real project is also empty or missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


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
    words: Optional[list[Word]] = None


@dataclass
class TranscriptResult:
    text: str
    segments: list[Segment] = field(default_factory=list)
    language: Optional[str] = None
    engine: Optional[str] = None
