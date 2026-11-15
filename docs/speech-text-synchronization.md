# Word-level speech and text synchronization

For browser speech output, the backend now aligns the known TTS input with the synthesized audio using the project's existing faster-whisper model. It sends estimated word start/end times together with the audio. The browser schedules each word against the Web Audio output clock; it does not reveal the whole segment when playback starts. With speech disabled, the browser still displays incoming LLM text deltas immediately.

## Important trade-off

The current Qwen3 streaming server emits PCM packets without word timestamps. To obtain audio-based word estimates, the backend buffers each complete TTS segment and runs Whisper forced alignment before sending it for playback. Thus **speech no longer begins on the first Qwen audio packet**: segment synthesis and alignment add to first-audio latency. LLM generation continues concurrently, and later segments can be generated while earlier audio plays. Do not describe this as zero-latency audio streaming.

Whisper timing is estimated from audio and the requested transcript. It is not ground truth and cannot guarantee that the TTS spoke every word correctly. Browser rendering, timer scheduling, and output-device buffering also affect perceived synchronization. The player uses `AudioContext.getOutputTimestamp()` where available and falls back to the context clock. Real-device tests, audio loopback, and manually annotated word boundaries are required before making an accuracy claim.

## Delivery and failure semantics

- Each audio message with spoken text has an ordered `word_timings` array. Entries contain the exact display text and a start/end time in seconds relative to that audio buffer. Whitespace and punctuation are preserved.
- The backend console reports a segment count and character length for browser speech rather than printing the answer ahead of playback. The browser transcript is the synchronized display. The native command-line audio mode has not been changed to word-level synchronization.
- The web backend loads faster-whisper if needed, reuses that instance, and uses the session's STT inference lock. This adds inference work and may contend with microphone transcription and other GPU workloads. The lock does not coordinate other browser sessions that share the model. `wordAlignmentTime` in the final turn metrics includes preprocessing, waiting for the lock and inference. `firstAudio` is a backend emission timestamp, not an audible-device timestamp. The `ttsTotal` fallback for streaming engines may include alignment and waiting.
- If timing is missing, invalid, out of order, or does not match the requested text, the turn fails rather than silently presenting an invented word schedule or reverting to an entire segment. Non-spoken text is queued in order.
- Interruption clears queued audio and word timers. An already-running GPU inference or blocked network call may take time to return; browser playback should stop without waiting for it. Successful audio delivery is acknowledged only after the stream ends, audio output drains, and scheduled words have been revealed.
- Backend and frontend must be updated together; the browser rejects spoken text without word timings. Text-only mode and turn identity remain unchanged.

## Validation

From the repository root:

```bash
python -m unittest discover -s tests/thinking_models -v
node --test frontend/tests/*.test.mjs
```

The automated tests use synthetic audio/timestamps and mocked model calls; they do not measure alignment accuracy, real browser/device latency, or interference with live STT/TTS models.

For acceptance, use the real TTS and audio device:

1. Ask a short question with speech on. Confirm individual words appear with speech, including the first and final words and punctuation.
2. Test longer answers, pauses, numbers, abbreviations, repeated words, accented names, and a question with no punctuation. Check for missing, repeated, early, or delayed words.
3. Stop during a segment and ask a new question; verify no old words or audio appear, and test voice barge-in as well as the stop button.
4. Test speech off, switch voices/backends, and test a suspended browser audio context and a slow network. Compare perceived first-audio latency with the segment-level baseline on the same hardware, prompts, and model configuration.
5. For thesis reporting, record multiple runs and use reference annotations or audio loopback to measure word-onset error (median and tail), first-audio latency, failures, hardware, workload, and synchronization clock. Do not interpret alignment-model timestamps as ground truth.
