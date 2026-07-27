# config.py
"""
Global configuration for the smart tutor system.

This module centralizes configuration values so they can be reused
across the codebase and easily tuned for experiments.
"""

from __future__ import annotations

import torch


def get_default_device() -> str:
    """
    Select the best available device for neural models.

    Priority order:
      1. CUDA (NVIDIA GPU)
      2. MPS (Apple Silicon)
      3. CPU

    Returns
    -------
    str
        One of "cuda", "mps", or "cpu".
    """
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# TTS configuration
# ---------------------------------------------------------------------------

# Device for running the TTS model.
TTS_DEVICE: str = get_default_device()

# Default TTS sampling parameters.
TTS_TEMPERATURE: float = 0.8  # 0.6–0.8 is stable/natural
TTS_EXAGGERATION: float = 0.45  # 0–1, controls emotional intensity
TTS_CFG_WEIGHT: float = 0.45  # 0.3–0.6 usually good for naturalness

# Repetition / sampling controls to avoid weird loops.
TTS_REPETITION_PENALTY: float = 1.1
TTS_TOP_P: float = 1.0
TTS_MIN_P: float = 0.05

# Directory for saving generated WAV files (e.g., for logging or offline analysis).
TTS_OUTPUT_DIR: str = "output"
