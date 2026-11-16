# Live LLM streaming in StreamTutor

## Scope

The LLM generates and transmits incremental **text deltas**. A delta is a piece of text from the server; it need not be exactly one model token. TTS is a separate stage: the currently configured engines accept a text segment per synthesis request, and Qwen3/Chatterbox Stream can return audio progressively for that request. These endpoints do not expose a bidirectional token-input TTS stream in this integration.

The application sends each received answer delta as a turn-tagged `llm_chunk` immediately. With Read aloud **off**, the browser displays these deltas as they arrive. With Read aloud **on**, the browser retains the existing speech synchronization: it reveals words according to audio-derived alignment timestamps and the browser playback clock. Therefore the visible transcript is intentionally behind the LLM when audio playback is behind; this does not mean that model generation is non-streaming.

## Current model routes

| Tutor model | Default local tunnel | Serving protocol | Streaming request |
| --- | --- | --- | --- |
| Llama 3 8B Q4 | 8000 | `remote_llm_server.py`, llama-cpp-python | `/stream` returns newline-delimited JSON from `create_chat_completion(stream=True)` |
| Gemma 4 26B Q3 | 8007 | llama-server | `/v1/chat/completions`, `stream: true` (SSE) |
| Qwen 3.5 9B Q6 | 8008 | llama-server | `/v1/chat/completions`, `stream: true` (SSE) |
| Gemma 4 E4B | 8005 | llama-server | `/v1/chat/completions`, `stream: true` (SSE) |
| Bonsai 2 9B | 8006 | llama-server | `/v1/chat/completions`, `stream: true` (SSE) |

`llama3_8b_q5` is a compatibility alias to port 8000, not evidence of a separate Q5 deployment. These entries describe the checked-in configuration; validate the actual running endpoints before claiming experimental results.

## Implementation

- The streaming HTTP readers use available network bytes rather than waiting for a fixed-size response read. Complete NDJSON/SSE lines are decoded as UTF-8; transport fragmentation may split a line or a Unicode character. An adapter lacking `read1` falls back to smaller Requests reads.
- The thinking parser holds only a suffix that could form a prohibited reasoning/channel marker. It still rejects split markers, unexpected reasoning, incomplete streams, and non-stop completion. Thinking and final-answer timings remain separate.
- In streaming-input mode, the pipeline reads the LLM stream while a worker synthesizes queued segments and a separate worker aligns completed audio in order. The text queue is bounded to 32 segments and the audio alignment queue to two; if downstream processing falls behind, backpressure pauses upstream consumption instead of allowing unbounded memory growth.
- In streaming-input mode, a stop or LLM failure cancels queued synthesis; a TTS failure fails the turn and cancels model consumption when the provider next observes the cancellation event. In chunked-synthesis mode, an LLM failure discards pending segments, but an in-flight synthesis call may still need to finish. A blocked network read may also take until its timeout. `audio_done` and judge delivery acknowledgment are not sent as successful on failure.
- The final answer is still accumulated for memory and evaluation. TTS segmentation remains necessary for the current synthesis APIs, and the browser receives word-timed audio segments. The current word alignment step buffers a full speech segment and adds latency; see `speech-text-synchronization.md`.

The default `ttsTotal` fallback for streaming engines can include time waiting for model text and streaming/network activity; it is not a clean GPU synthesis-only measurement. `firstAudio` in the backend is an audio-emission timestamp, not proof of audible speaker onset. Compare latency only with the same hardware, settings, workload, and clearly defined measurement boundaries.

## Deterministic checks

From the repository root:

```bash
python -m unittest discover -s tests/thinking_models -v
node --test frontend/tests/*.test.mjs
```

These tests include delayed live HTTP responses for both protocols, UTF-8 split across network writes, bounded-queue interruption, LLM/TTS overlap, streaming failures, and the synchronized browser queue. They use simulated model and speech responses; they do not verify an HPC model or physical audio device.

## Probe the real model servers

Start the intended model servers and tunnels first. From the repository root, run with the project Python environment:

```powershell
& C:\Users\shaik\Desktop\llmtutor\.venv\Scripts\python.exe .\scripts\probe_llm_stream.py --model all
```

This sends **one actual generation request to each of the five models**, sequentially, using a short question and thinking disabled. It prints each received answer delta and client-observed timing. For the four llama-server routes, the probe checks `/v1/models` against the configured GGUF filename before issuing the request. The legacy Llama endpoint does not expose verified model identity through this probe. Override a single model's URL if your tunnel differs:

```powershell
& C:\Users\shaik\Desktop\llmtutor\.venv\Scripts\python.exe .\scripts\probe_llm_stream.py --model gemma4_e4b_thinking --url http://127.0.0.1:8005 --thinking
```

Multiple deltas observed over time are evidence of incremental delivery to this client. A single delta is inconclusive; multiple deltas arriving together also do not rule out proxy buffering. Model loading, prompt prefill, network buffering, reasoning, TTS synthesis, and audio playback are separate sources of latency. Repeat a representative question with Read aloud both on and off and check interruptions; use proper timestamps and repeated trials for thesis claims.
