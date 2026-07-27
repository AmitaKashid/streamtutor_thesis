# audio_capture.py
from __future__ import annotations
from typing import Optional
import sys
import time
import numpy as np
import sounddevice as sd

from stt.vad_segmenter import TARGET_SR, CHANNELS, FRAME_MS, FRAME_SAMPLES



def log_input_device(mic: Optional[int]) -> None:
    """Best-effort printing of the chosen input device name."""
    try:
        dev_index = mic
        if dev_index is None:
            # sounddevice chooses default if None
            info = sd.query_devices(None, "input")
        else:
            info = sd.query_devices(dev_index, "input")
        print(f"[INFO] Using input device: '{info['name']}'")
    except Exception:
        print("[WARN] Could not query input device; continuing.")


def start_input_stream(q_frames, mic: Optional[int] = None) -> sd.InputStream:
    """Return a configured InputStream that pushes (frame_bytes, timestamp) into q_frames."""
    if mic is not None:
        sd.default.device = (mic, None)  # (input, output)

    sd.default.channels = CHANNELS
    log_input_device(mic)

    def sd_callback(indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)

        mono = indata[:, 0] if indata.ndim > 1 else indata

        # float32 [-1,1] -> int16 PCM
        pcm16 = np.clip(mono, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)

        base_ts = float(getattr(time_info, "inputBufferAdcTime", time.monotonic()))

        # Slice into 20ms frames
        for i, start in enumerate(range(0, len(pcm16) - FRAME_SAMPLES + 1, FRAME_SAMPLES)):
            ts = base_ts + (i * FRAME_MS) / 1000.0
            q_frames.put((pcm16[start:start + FRAME_SAMPLES].tobytes(), ts))

    return sd.InputStream(
        callback=sd_callback,
        samplerate=TARGET_SR,
        channels=CHANNELS,
        dtype="float32",
        blocksize=0,
    )
