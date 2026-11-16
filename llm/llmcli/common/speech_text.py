"""Convert a streamed text chunk to a speech-friendly approximation.

The UI receives the original text; this is only for the TTS input. Chunks may
end inside markup, so delimiter characters are removed independently.
"""
import re


def speech_text(text: str) -> str:
    text = re.sub(r'!?\[([^\]]+)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'(?m)^\s{0,3}#{1,6}\s*', '', text)
    text = re.sub(r'(?m)^\s*(?:[-+]\s+|\d+[.)]\s+|>\s*)', '', text)
    text = re.sub(r'(?<=\d)\s*\*\s*(?=\d)', ' times ', text)
    text = text.replace('*', '').replace('`', '').replace('~', '')
    text = re.sub(r'(?<!\w)_+|_+(?!\w)', '', text)
    return re.sub(r'\s+', ' ', text).strip()
