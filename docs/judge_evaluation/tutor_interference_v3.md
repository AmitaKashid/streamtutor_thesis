# Tutor interference measurement

The runner replays the 30 checked-in `data/benchmark_questions.jsonl`
questions in file order with one unmeasured warm-up per condition. Each
question uses a fresh WebSocket and fixed Llama-3, Kokoro and RAG settings.
This measures the post-transcription tutor pipeline. It does not measure STT,
microphone capture, or when audio becomes audible on a user's speakers.

Dry-run (no GPU, service, or WebSocket package needed):

```text
python -m judge.run_tutor_interference_v3 run --mode disabled --run-id tutor-interference-plan-v3 --dry-run
```

For a live run, use a clean committed checkout, the `websocket-client` Python
package, a distinct run ID per condition, and add
`--ws-url ws://127.0.0.1:8001/ws` in place of `--dry-run`.
Restart the frontend process between conditions; it reads these environment
variables at startup:

- Disabled: `LLMTUTOR_RUNTIME_JUDGE=0`.
- Same GPU / isolated GPU: `LLMTUTOR_RUNTIME_JUDGE=1`, set
  `LLMTUTOR_JUDGE_URL` to the appropriate endpoint and
  `LLMTUTOR_JUDGE_OUTPUT` to a fresh runtime results path per condition.
- Optional unavailable-at-startup check: use an intentionally unavailable
  endpoint with judging enabled, preserve the startup healthcheck log, and
  select `--mode unavailable`. This does not test a mid-run failure.

Hold the tutor GPU, tutor backend, endpoints for other components, and client
location constant. Record physical GPU UUIDs, Slurm IDs, the judge startup
log, resource samples for both services, and the runtime judge results.
Use the same runtime judge artifact and configuration for both enabled
conditions. Do not confuse this runtime evaluator with the offline scalar
quality benchmark. The mode is declared, not detected. Queue acceptance
alone does not prove contention: confirm judge inference overlapped later
tutor work before interpreting a same-GPU run as an interference test.

```text
python -m judge.run_tutor_interference_v3 compare DISABLED_DIR SAME_GPU_DIR ISOLATED_GPU_DIR
```

The runner retains raw per-turn measurements and failed turns. Comparison
refuses incomplete or invalid cohorts rather than dropping failures. It
reports median and p95 tutor timing and the latency change relative to the
disabled baseline. Judge enqueue time covers preparation of submission
arguments and the synchronous submission call, not the evaluation itself.
The summary does not certify topology, actual overlap, peak VRAM or outage
resilience and does not issue an overall deployment pass.
