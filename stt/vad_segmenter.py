# vad_segmenter.py
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Optional, List
import time

import numpy as np
import webrtcvad

# Standardize: internal pipeline expects 16kHz mono
TARGET_SR = 16000
CHANNELS = 1

# WebRTC VAD supports 10/20/30 ms frames
FRAME_MS = 20
FRAME_SAMPLES = int(TARGET_SR * FRAME_MS / 1000)

# Endpointing knobs
MAX_SILENCE_MS = 800       # end utterance after trailing silence
MAX_UTTERANCE_MS = 30000   # hard stop
PRE_ROLL_MS = 250          # keep some audio before speech start


@dataclass
class Utterance:
    audio: np.ndarray       # int16 mono
    t_start: float
    t_end: float
    index: int
    voiced_ms: int = 0


class VADSegmenter:
    """Turn fixed 20ms frames into utterances using WebRTC VAD, with pre-roll."""
    def __init__(self, aggressiveness: int = 2, pre_roll_ms: int = PRE_ROLL_MS):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.pre = deque(maxlen=max(1, pre_roll_ms // FRAME_MS))  # frames of int16
        self.buffer: List[np.ndarray] = []
        self.in_speech = False
        self.speech_start_ts: Optional[float] = None
        self.trailing_silence_ms = 0
        self.voiced_ms = 0
        self.ms_in_buffer = 0
        self.utt_index = 0

    def process_frame(self, frame_int16: bytes, frame_ts: float) -> Optional[Utterance]:
        is_speech = self.vad.is_speech(frame_int16, TARGET_SR)

        if not self.in_speech and not is_speech:
            # collect pre-roll while idle
            self.pre.append(np.frombuffer(frame_int16, dtype=np.int16))

        if is_speech:
            self.voiced_ms += FRAME_MS
            self.trailing_silence_ms = 0
            if not self.in_speech:
                # speech start
                self.in_speech = True
                self.speech_start_ts = frame_ts - (len(self.pre) * FRAME_MS) / 1000.0
                self.buffer = list(self.pre) + [np.frombuffer(frame_int16, dtype=np.int16)]
                self.ms_in_buffer = (len(self.pre) + 1) * FRAME_MS
                self.pre.clear()
            else:
                self.buffer.append(np.frombuffer(frame_int16, dtype=np.int16))
                self.ms_in_buffer += FRAME_MS
        else:
            if self.in_speech:
                # keep trailing silence inside the utterance buffer
                self.trailing_silence_ms += FRAME_MS
                self.buffer.append(np.frombuffer(frame_int16, dtype=np.int16))
                self.ms_in_buffer += FRAME_MS

        time_limit_hit = self.ms_in_buffer >= MAX_UTTERANCE_MS
        silence_hit = self.in_speech and self.trailing_silence_ms >= MAX_SILENCE_MS

        if self.in_speech and (silence_hit or time_limit_hit):
            audio = np.concatenate(self.buffer, axis=0) if self.buffer else np.array([], dtype=np.int16)
            t_start = self.speech_start_ts if self.speech_start_ts is not None else 0.0
            t_end = frame_ts + FRAME_MS / 1000.0

            voiced_ms = self.voiced_ms
            # reset for next utterance
            self.buffer = []
            self.ms_in_buffer = 0
            self.in_speech = False
            self.trailing_silence_ms = 0
            self.voiced_ms = 0
            self.speech_start_ts = None

            self.utt_index += 1
            return Utterance(audio=audio, t_start=t_start, t_end=t_end, index=self.utt_index, voiced_ms=voiced_ms)

        return None

    def flush(self) -> Optional[Utterance]:
        if not self.buffer:
            return None
        audio = np.concatenate(self.buffer, axis=0)
        voiced_ms = self.voiced_ms
        self.buffer = []
        self.ms_in_buffer = 0
        self.in_speech = False
        self.trailing_silence_ms = 0
        self.voiced_ms = 0
        self.speech_start_ts = None
        self.utt_index += 1
        return Utterance(audio=audio, t_start=0.0, t_end=0.0, index=self.utt_index, voiced_ms=voiced_ms)

    def reset(self):
        """
        Reset internal VAD state.

        This is used when the microphone is temporarily muted
        (e.g., during TTS playback) to avoid emitting
        partial or stale utterances after unmuting.
        """
        self.pre.clear()
        self.buffer = []
        self.ms_in_buffer = 0
        self.in_speech = False
        self.trailing_silence_ms = 0
        self.voiced_ms = 0
        self.speech_start_ts = None
