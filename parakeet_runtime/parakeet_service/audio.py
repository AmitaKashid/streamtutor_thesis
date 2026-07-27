from __future__ import annotations

from pathlib import Path
from typing import Tuple, List
import tempfile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from fastapi import BackgroundTasks, HTTPException, status

from .config import TARGET_SR, logger


SUPPORTED_EXTS: List[str] = [".wav", ".flac", ".mp3", ".ogg", ".opus"]


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32, copy=False)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(audio, -1.0, 1.0)


def _resample_if_needed(audio: np.ndarray, sr_orig: int, sr_target: int) -> np.ndarray:
    if sr_orig == sr_target:
        return audio.astype(np.float32, copy=False)

    from math import gcd

    g = gcd(sr_orig, sr_target)
    up = sr_target // g
    down = sr_orig // g

    out = resample_poly(audio, up, down)
    return out.astype(np.float32, copy=False)


def ensure_mono_16k(src: Path) -> Tuple[Path, Path]:
    """
    Downmix and resample to mono 16 kHz without torchaudio.load/TorchCodec.
    This avoids the Windows TorchCodec/FFmpeg DLL issue.
    """
    if src.suffix.lower() not in SUPPORTED_EXTS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type {src.suffix}",
        )

    try:
        audio, sr = sf.read(str(src), dtype="float32", always_2d=False)
        audio = _to_mono_float32(audio)
        audio = _resample_if_needed(audio, int(sr), int(TARGET_SR))

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        dst = Path(tmp.name)
        tmp.close()

        sf.write(str(dst), audio, int(TARGET_SR), subtype="PCM_16")
        return src, dst

    except Exception as e:
        logger.exception("Audio conversion failed")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Audio conversion failed: {e}",
        ) from e


def schedule_cleanup(tasks: BackgroundTasks, *paths: Path) -> None:
    for p in paths:
        tasks.add_task(p.unlink, missing_ok=True)