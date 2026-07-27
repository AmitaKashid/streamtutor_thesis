import base64
import os
from pathlib import Path

from mistralai.client import Mistral

API_KEY = os.environ["MISTRAL_API_KEY"]
AUDIO_PATH = Path("sample.wav")   # one-time sample, just to create the saved voice
VOICE_NAME = "thesis_voice"

if not AUDIO_PATH.exists():
    raise FileNotFoundError(f"Audio file not found: {AUDIO_PATH}")

client = Mistral(api_key=API_KEY)

sample_audio_b64 = base64.b64encode(AUDIO_PATH.read_bytes()).decode("utf-8")

voice = client.audio.voices.create(
    name=VOICE_NAME,
    sample_audio=sample_audio_b64,
    sample_filename=AUDIO_PATH.name,
    languages=["en"],
)

print(f"Created voice: {voice.id}")
print(f"Name: {voice.name}")
print(f"Languages: {voice.languages}")
print(f"\nSet this in your environment:\nVOXTRAL_VOICE_ID={voice.id}")