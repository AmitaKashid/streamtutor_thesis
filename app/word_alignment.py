"""Align known speech text to synthesized audio with the existing Whisper model.

The timing values are model estimates, not ground-truth word boundaries.  This
module never invents evenly spaced timestamps if alignment fails.
"""
from __future__ import annotations

import math
import re
from math import gcd

import numpy as np
from scipy.signal import resample_poly


class WordAlignmentError(RuntimeError):
    """The generated audio and text could not be aligned reliably."""


def bind_alignment_to_text(text: str, alignment: list[dict], duration: float) -> list[dict]:
    """Map aligner tokens back to the exact input text without changing words."""
    if not text or not alignment or not math.isfinite(duration) or duration <= 0:
        raise WordAlignmentError("Audio or word alignment is empty")
    position = 0
    pending_prefix = ''
    words: list[dict] = []
    last_start = 0.0
    for item in alignment:
        piece = item.get('word')
        if not isinstance(piece, str) or not piece:
            continue
        start, end = item.get('start'), item.get('end')
        if not isinstance(start, (int, float, np.number)) or not isinstance(end, (int, float, np.number)):
            raise WordAlignmentError("A word has no numeric timing")
        start, end = float(start), float(end)
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start or start + .001 < last_start or end > duration + .1:
            raise WordAlignmentError("Word timestamps are not ordered within the audio")
        start = min(duration, max(last_start, start))
        end = min(duration, max(start, end))
        last_start = start
        # Tokenizers may normalize whitespace, but must preserve every other
        # character. Attach the source's actual whitespace to its next token.
        begin = position
        for part in re.findall(r'\s+|\S+', piece):
            if part.isspace():
                while position < len(text) and text[position].isspace():
                    position += 1
            else:
                while position < len(text) and text[position].isspace():
                    position += 1
                if not text.startswith(part, position):
                    raise WordAlignmentError("Aligner tokens do not match the synthesis text")
                position += len(part)
        display = text[begin:position]
        if not display:
            continue
        if any(char.isalnum() for char in display):
            words.append({'text': pending_prefix + display, 'start': start, 'end': end})
            pending_prefix = ''
        elif words:
            words[-1]['text'] += display
        else:
            pending_prefix += display
    if text[position:].strip():
        raise WordAlignmentError("Alignment omitted part of the synthesis text")
    if not words:
        raise WordAlignmentError("No spoken words were aligned")
    words[-1]['text'] += text[position:] + pending_prefix
    if ''.join(item['text'] for item in words) != text:
        raise WordAlignmentError("Alignment did not preserve the original text")
    return words


def align_with_whisper(whisper_model, text: str, audio: np.ndarray, sample_rate: int, language: str = 'en') -> list[dict]:
    """Force-align text using faster-whisper 1.2's existing model and tokenizer.

    A complete short synthesized segment is required before alignment. The
    caller is responsible for serializing use of a shared Whisper model.
    """
    from faster_whisper.audio import pad_or_trim
    from faster_whisper.tokenizer import Tokenizer

    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.ndim != 1 or not samples.size or sample_rate <= 0 or not np.isfinite(samples).all():
        raise WordAlignmentError("Invalid synthesized audio")
    duration = len(samples) / sample_rate
    if duration > 29.5:
        raise WordAlignmentError("Synthesized segment is too long for word alignment")
    source_text = text.strip()
    if not source_text:
        raise WordAlignmentError("Cannot align empty text")
    if sample_rate != 16000:
        divisor = gcd(int(sample_rate), 16000)
        samples = resample_poly(samples, 16000 // divisor, int(sample_rate) // divisor).astype(np.float32)
    extractor = whisper_model.feature_extractor
    features = extractor(samples)
    num_frames = min(features.shape[-1] - 1, extractor.nb_max_frames)
    if num_frames <= 0:
        raise WordAlignmentError("Synthesized audio has no alignable frames")
    tokenizer = Tokenizer(whisper_model.hf_tokenizer, whisper_model.model.is_multilingual,
                          task='transcribe', language=language)
    tokens = tokenizer.encode(source_text)
    if not tokens or len(tokens) >= whisper_model.max_length:
        raise WordAlignmentError("Text is too long for word alignment")
    encoded = whisper_model.encode(pad_or_trim(features[:, :num_frames]))
    result = whisper_model.find_alignment(tokenizer, [tokens], encoded, num_frames)
    if not result or not result[0]:
        raise WordAlignmentError("Whisper returned no word alignment")
    words = bind_alignment_to_text(source_text, result[0], duration)
    # Preserve the space separating this segment from the following segment.
    words[-1]['text'] += ' '
    return words
