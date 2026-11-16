"""Independent acoustic gate for microphone transcripts.

This is a speech detector, not a guarantee that the recognized words are true.
The input is retained between the first and last detected speech with padding,
so normal pauses are not joined together.
"""
from __future__ import annotations

import numpy as np


class SpeechEvidenceError(RuntimeError):
    """Speech evidence could not be checked, so no transcript is published."""


def speech_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray | None:
    """Return the speech region, or None if Silero detects no speech.

    The application already applies WebRTC VAD; this independent detector
    helps reject noises that trigger WebRTC but should not be decoded by ASR.
    """
    if sample_rate != 16000:
        raise SpeechEvidenceError("Speech verification expects 16 kHz audio")
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim != 1 or not np.all(np.isfinite(data)):
        raise SpeechEvidenceError("Invalid audio for speech verification")
    if not data.size or not np.any(data):
        return None
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        regions = get_speech_timestamps(
            data,
            VadOptions(threshold=0.5, min_speech_duration_ms=120,
                       min_silence_duration_ms=160, speech_pad_ms=0),
            sampling_rate=sample_rate,
        )
    except Exception as exc:
        raise SpeechEvidenceError("Speech verification is unavailable") from exc
    if not regions:
        return None
    # Keep the intervening audio and a small margin for quiet initial/final
    # phonemes. Do not concatenate detected segments across natural pauses.
    padding = int(sample_rate * 0.2)
    start = max(0, int(regions[0]["start"]) - padding)
    end = min(len(data), int(regions[-1]["end"]) + padding)
    if start >= end:
        raise SpeechEvidenceError("Speech verification returned invalid timestamps")
    return data[start:end]


def warm_speech_evidence() -> None:
    """Initialize the bundled VAD off the request path."""
    try:
        from faster_whisper.vad import get_vad_model
        get_vad_model()(np.zeros(512, dtype=np.float32))
        print("[SPEECH EVIDENCE] Silero ready", flush=True)
    except Exception as exc:
        # The request path retries and fails closed with a user-visible error.
        print(f"[SPEECH EVIDENCE] Warmup unavailable: {type(exc).__name__}: {exc}", flush=True)
