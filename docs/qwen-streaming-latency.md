# Qwen streaming latency and captions

## What changes

For browser speech with Qwen selected, each PCM packet is forwarded as the
existing Qwen server produces it. Playback is no longer gated on a full segment
and Whisper alignment. Qwen remains the TTS engine; the Kokoro and other
non-Qwen routes retain their previous behavior.

Optional English word tracking runs on CPU from copies of the audio. It is
best effort: its estimates can be late or wrong, especially for domain terms,
abbreviations, numbers and unexpected pronunciations. Timestamps are mapped to
the actual audio schedule and output clock. Late words catch up; if tracking
cannot confidently complete a segment, its remaining text appears when that
segment finishes. Audio never waits for this worker. If the model is not
installed, has not finished loading, fails to start, or the language is not
English, captions fall back to showing the full segment when its audio starts.
That fallback can display the segment ahead of its spoken words.

This is not a guarantee of exact word synchronization or real-time synthesis.
The recognizer is a general English acoustic model and must be evaluated on the
actual Qwen voices and Data Mining vocabulary. A true alignment-capable TTS
engine could provide better timing if it exposes reliable word timestamps.

## One-time installation, Windows PowerShell

From the repository root, with internet access on the Windows computer:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-streaming-captions.txt
.\.venv\Scripts\python.exe scripts\setup_streaming_captions.py
```

The second command downloads the pinned quantized Wav2Vec2 model (about 95 MB)
and vocabulary, verifies SHA-256, and performs a CPU inference test. Model
files are stored in the ignored `runtime/word-alignment/` directory. The
frontend API loads and warms the model asynchronously during startup. Its
terminal prints `[CAPTIONS] CPU acoustic word tracker ready` when ready. If it
is unavailable the audio still works, with segment captions. Restart the API
after installing. Qwen's separate virtual environment needs no new dependency.

The default caption mode is `acoustic`. To switch modes, set the variable in
the PowerShell window **before** starting the stack; restarted child processes
inherit it:

```powershell
$env:LLMTUTOR_QWEN_CAPTIONS = 'segment'   # immediate audio, segment captions
$env:LLMTUTOR_QWEN_CAPTIONS = 'acoustic'  # immediate audio, optional CPU words
$env:LLMTUTOR_QWEN_CAPTIONS = 'whisper'   # old buffered path, useful for A/B
```

Set one mode at a time and restart `frontend_api` after changing it. `whisper`
restores the latency-producing full-segment alignment path. The existing
launcher starts the Qwen server, API and frontend. Keep Qwen selected as the
TTS backend in the app and hard-reload the browser after an update.

## Measure on the target machine

Compare `whisper` and `acoustic` on the **same machine and same LLM**, after
warm-up. Keep the voice, Qwen chunk size, STT engine, RAG, thinking settings,
conversation context and microphone procedure the same. Use a fixed set of
questions; run several warm-up turns, then at least 20 measured turns per mode
if practical. Alternate or randomize mode order to reduce time-dependent bias.
Record the app's `First audio` and TTS first-chunk metrics and the number and
duration of `[AUDIO UNDERRUN]` messages in the browser console. Report median
and p95, and note sample counts. Capture quality and interruption failures as
well as speed; a low first-packet time does not guarantee smooth speech.

The app's **First audio** is measured at the backend, from pipeline start after
STT until the first PCM is emitted to the browser. It is not speech-end-to-ear
latency and does not include endpoint silence, STT, browser buffering or speaker
latency. Do not add overlapping timings or compare values with different start
points. For the real user-perceived delay, use a synchronized microphone/output
loopback recording or external video/audio measurement.

The following isolated probe measures just the Qwen server. Run it from another
PowerShell window while port 8094 is healthy:

```powershell
.\.venv\Scripts\python.exe scripts\probe_qwen_stream.py --runs 10 --warmup 1 --chunk-size 4
```

It reports request-to-first-PCM, audio duration, wall time, packet count and
maximum interval between packets. `audio/wall` above 1 means the server
generated more audio seconds than wall seconds, but one aggregate ratio cannot
rule out packet stalls or browser underruns. This probe bypasses the LLM,
StreamTutor pipeline, captioning and browser, so it cannot show the improvement
from removing StreamTutor's alignment wait. It can help identify a remaining
Qwen bottleneck. The current server's chunk size is 4; do not assume a smaller
value is faster or smoother without a controlled quality and underrun test.

Useful source references:

- Faster Qwen3-TTS streaming and chunk-size tradeoffs: https://github.com/andimarafioti/faster-qwen3-tts
- The pinned acoustic model and checksums: https://huggingface.co/elabora/wav2vec2-base-960h-onnx
- Provider-aligned versus sentence-level captions: https://docs.livekit.io/agents/multimodality/text/
