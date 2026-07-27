# audio_file.py
from __future__ import annotations
from typing import Tuple
import numpy as np

def load_audio_mono_16k(path: str) -> Tuple[np.ndarray, int]:
    """
    Load a WAV file and return (audio_float32_mono, sample_rate=16000).

    Cluster-friendly: uses scipy (no libsndfile dependency).
    Input should be a WAV file (PCM or float WAV).
    """
    from scipy.io import wavfile

    sr, audio = wavfile.read(path)  # audio can be int16/int32/float32, mono or stereo

    # Convert to float32 in [-1, 1]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        # uint8 WAV is 0..255 with 128 as "zero"
        audio = (audio.astype(np.float32) - 128.0) / 128.0
    else:
        # float32/float64 already typically in [-1,1], just cast
        audio = audio.astype(np.float32)

    # Stereo -> mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != 16000:
        import resampy
        audio = resampy.resample(audio, sr, 16000).astype(np.float32)
        sr = 16000

    # Safety clamp
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    return audio, sr
