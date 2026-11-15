# Speech and text synchronization

The browser reveals each tutor text segment when the corresponding audio segment starts in the Web Audio playback queue. Generating text and synthesizing audio can run ahead, but queued speech does not appear in the transcript until its turn to play. Text-only output continues to stream as the language model produces it.

## Delivery contract

- Every tutor turn has a `turn_id` and a fixed `delivery_mode`, captured at turn start. Changing the speech switch applies to a subsequent turn.
- For speech, the first `audio_chunk` of each synthesized text segment includes its `text`; later audio packets for that segment have no text. An `audio_text` event carries an ordered segment without speakable characters. The browser queues these events alongside audio and displays each segment once.
- The server emits `audio_done` after it has sent the stream. The browser reports `browser_audio_finished` only after that event and after all received audio has finished playing. The response remains active until generation and playback have both ended.
- Stop, barge-in, disconnect, or playback failure clears pending audio and text. Turn identifiers and queue generations prevent callbacks or packets from an earlier turn revealing text in a newer one. An interrupted turn is reported as interrupted rather than successfully delivered.
- In text mode, turn-tagged `llm_chunk` events appear immediately, and `text_complete` still determines text-delivery acknowledgement.

The API continues to emit `llm_chunk` for compatibility with older clients, but the updated browser ignores it for audio turns. Deploy the updated backend and frontend together: an older backend does not attach the segment text required by the updated browser.

## Timing and limitations

This is **segment-level synchronization**, not word-by-word alignment. The engines do not supply word timestamps, so a whole synthesized segment appears when its first browser audio buffer is started. Segments may contain more than one sentence, and browser scheduling, rendering, and audio-device buffering can introduce a small delay between the display update and audible sound. The change does not measure that physical delay or modify the existing latency metrics.

The browser requests audio permission in a user gesture when speech is enabled. If the audio context remains suspended, spoken text is held until playback can start. A playback error interrupts the turn and reports an error rather than marking it delivered.

## Validation

Run the deterministic checks from the repository root:

```bash
python -m unittest discover -s tests/thinking_models -v
node --test frontend/tests/*.test.mjs
cd frontend && npm run build
```

Also test with the actual browser, selected TTS service, and audio device:

1. Enable **Read aloud** and ask a question long enough to produce several segments. Confirm that the next segment is not visible while an earlier one is playing and that the final segment appears.
2. Stop once during a segment and once while later segments are queued. Confirm playback stops, no queued text appears afterward, and the next question starts a clean turn.
3. Turn **Read aloud** off and verify text appears incrementally without waiting for audio. Turn it back on for another question.
4. Repeat with the speech backends used in the experiment, including a streaming backend if applicable. Check browser audio permissions and repeat after reconnecting. Real device and end-to-end latency must be measured separately from the deterministic tests.
