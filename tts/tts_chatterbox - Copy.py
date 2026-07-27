"""
tts_chatterbox.py

Professional, reproducible text-to-speech pipeline using Chatterbox TTS.
Designed for research and thesis use.

Run:
    python tts_chatterbox.py --text "Hello students" --out output.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio
from chatterbox.tts import ChatterboxTTS


def best_device() -> str:
    """Return the best available device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synthesize(
    text: str,
    out_path: Path,
    *,
    audio_prompt: Path | None = None,
    temperature: float = 0.8,
    exaggeration: float = 0.45,
    cfg_weight: float = 0.45,
    repetition_penalty: float = 1.1,
    top_p: float = 1.0,
    min_p: float = 0.05,
) -> None:
    """
    Convert text to speech using Chatterbox and save as WAV.

    Parameters
    ----------
    text : str
        Input text to convert to speech.
    out_path : Path
        Output WAV path.
    audio_prompt : Path | None
        Optional voice-cloning prompt WAV.
    temperature : float
        Controls randomness and naturalness.
    exaggeration : float
        Emotional intensity (0–1).
    cfg_weight : float
        Guidance strength (controls adherence to text).
    """
    device = best_device()
    print(f"[INFO] Using device: {device}")

    model = ChatterboxTTS.from_pretrained(device=device)

    wav = model.generate(
        text,
        audio_prompt_path=str(audio_prompt) if audio_prompt else None,
        temperature=temperature,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        min_p=min_p,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), wav, sample_rate=model.sr)

    print(f"[INFO] Audio saved to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chatterbox TTS text-to-speech generator")

    parser.add_argument(
        "--text",
        type=str,
        default=(
            "Hello! I am your AI tutor. "
            "Today we will explore a new learning topic together."
        ),
        help="Text to synthesize."
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output") / "tutor.wav",
        help="Output WAV file path."
    )

    parser.add_argument(
        "--voice_prompt",
        type=Path,
        default=None,
        help="Optional reference audio for voice cloning."
    )

    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--exaggeration", type=float, default=0.45)
    parser.add_argument("--cfg_weight", type=float, default=0.45)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    synthesize(
        text=args.text,
        out_path=args.out,
        audio_prompt=args.voice_prompt,
        temperature=args.temperature,
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
    )


if __name__ == "__main__":
    main()
