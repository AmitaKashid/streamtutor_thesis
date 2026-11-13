"""Validate llama-server thinking streams before forwarding final-answer text.

Validate the answer channel and enforce the requested reasoning mode.
"""
from __future__ import annotations


class ThinkingStreamError(RuntimeError):
    """The server did not provide a safely separable completed answer."""


class ReasoningNotObserved(ThinkingStreamError):
    """The model returned content without a separate reasoning field."""


class ThinkingStream:
    # Retain enough trailing characters to detect tags split across SSE events.
    _tags = ("<think>", "</think>", "<|think|>", "<|channel>thought", "<channel|>", "<|channel|>analysis", "<|channel|>final")
    _hold = max(map(len, _tags)) - 1

    def __init__(self, require_reasoning: bool = True) -> None:
        self.require_reasoning = require_reasoning
        self.saw_reasoning = False
        self.finished = False
        self.buffer = ""
        self.answer_seen = False

    def feed(self, event: dict) -> str:
        if self.finished:
            raise ThinkingStreamError("Event received after completion")
        if event.get("error"):
            raise ThinkingStreamError("Model server returned an error")
        choices = event.get("choices", [])
        if not isinstance(choices, list):
            raise ThinkingStreamError("Invalid model-server response")
        if not choices:
            return ""
        choice = choices[0]
        delta = choice.get("delta") or {}
        reasoning = delta.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ThinkingStreamError("Invalid reasoning field")
        if reasoning:
            if not self.require_reasoning:
                raise ThinkingStreamError("Server emitted reasoning with thinking disabled")
            self.saw_reasoning = True
        content = delta.get("content")
        if content is not None and not isinstance(content, str):
            raise ThinkingStreamError("Invalid content field")
        if content:
            if self.require_reasoning and not self.saw_reasoning:
                raise ReasoningNotObserved("Reasoning separation was not verified")
            self.answer_seen = True
            self.buffer += content
        if any(tag in self.buffer for tag in self._tags):
            raise ThinkingStreamError("Unexpected reasoning or channel marker in answer")
        reason = choice.get("finish_reason")
        if reason is not None:
            if reason != "stop":
                raise ThinkingStreamError(f"Model did not finish normally: {reason}")
            if not self.answer_seen:
                raise ThinkingStreamError("Model produced no final answer")
            self.finished = True
            result, self.buffer = self.buffer, ""
            return result
        if len(self.buffer) <= self._hold:
            return ""
        result, self.buffer = self.buffer[:-self._hold], self.buffer[-self._hold:]
        return result

    def end(self) -> None:
        if not self.finished:
            raise ThinkingStreamError("Model stream ended before completion")
