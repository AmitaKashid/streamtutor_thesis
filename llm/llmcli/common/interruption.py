# common/interruption.py
from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class InterruptionState:
    """
    Shared state between capture thread and response thread.
    """
    response_active: threading.Event = field(default_factory=threading.Event)
    cancel_requested: threading.Event = field(default_factory=threading.Event)

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _last_spoken_sentence: str = ""
    _last_user_query: str = ""

    def start_response(self) -> None:
        self.cancel_requested.clear()
        self.response_active.set()

    def finish_response(self) -> None:
        self.response_active.clear()
        self.cancel_requested.clear()

    def request_cancel(self) -> None:
        self.cancel_requested.set()

    def should_cancel(self) -> bool:
        return self.cancel_requested.is_set()

    def set_last_spoken_sentence(self, sentence: str) -> None:
        with self._lock:
            self._last_spoken_sentence = sentence.strip()

    def get_last_spoken_sentence(self) -> str:
        with self._lock:
            return self._last_spoken_sentence

    def set_last_user_query(self, query: str) -> None:
        with self._lock:
            self._last_user_query = query.strip()

    def get_last_user_query(self) -> str:
        with self._lock:
            return self._last_user_query


def build_interruption_query(last_spoken_sentence: str, new_user_utterance: str) -> str:
    """
    Build the interruption-aware user query that gets sent to the LLM.
    """
    if last_spoken_sentence:
        return (
            "The user interrupted you while you were saying:\n"
            f"\"{last_spoken_sentence}\"\n\n"
            "The user then interrupted with:\n"
            f"\"{new_user_utterance}\"\n\n"
            "Respond directly to the interruption, continue helpfully, and adapt your explanation."
        )
    return new_user_utterance