# audio_frames.py
from __future__ import annotations
import numpy as np
from typing import Iterable
from stt.vad_segmenter import FRAME_SAMPLES, FRAME_MS

def float32_to_frames_int16(audio: np.ndarray) -> Iterable[bytes]:
    """
    audio: float32 mono [-1,1] at 16kHz
    yields: frame bytes (int16) sized FRAME_SAMPLES
    Pads the last frame with zeros so no audio tail is lost.
    """
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)

    n = len(pcm16)
    full_frames = n // FRAME_SAMPLES

    # yield full frames
    for i in range(full_frames):
        start = i * FRAME_SAMPLES
        yield pcm16[start:start + FRAME_SAMPLES].tobytes()

    # pad leftover
    rem = n - full_frames * FRAME_SAMPLES
    if rem > 0:
        last = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        last[:rem] = pcm16[full_frames * FRAME_SAMPLES:]
        yield last.tobytes()

def frame_index_to_ts(i: int) -> float:
    return (i * FRAME_MS) / 1000.0
