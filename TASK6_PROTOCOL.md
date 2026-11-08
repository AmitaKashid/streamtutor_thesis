# Task 6 — StreamTutor runtime reliability and latency validation

**Scope:** quantitative comparison of tutor behavior with scalar FARE judging disabled, asynchronously enabled, and unavailable at startup. FARE scalar-v3 rubric and model configuration stay frozen. Do **not** mix these timings with judge-selection calibration measurements or claim that test references are human-verified gold.

## Measurement definitions

- **Tutor backend total (s):** pipeline's existing `End-to-end total` log. This is not browser playback completion.
- **LLM TTFT (s):** the pipeline's request-to-first-token metric. This is not end-user time to first audio.
- **First audio from pipeline (s):** existing pipeline metric, not independently observed acoustic onset.
- **Backend turn start to browser ACK (s):** elapsed monotonic time from server emission of `tutor_turn_started` to server receipt of `browser_audio_finished` for the same `turn_id`. Includes browser playback, event loop/network transit; NOT from microphone utterance onset and NOT physically verified listening.
- **Browser drain-after-audio-done (s):** time from server emission of `audio_done` to receipt of successful browser ACK. This tail is omitted by backend totals and matters for user experience.
- **Judge attempt (s):** evaluator wall time from SQLite `attempts.elapsed_s` (contains model inference and adapter parsing; excludes waiting in queue).
- **Judge time-to-final (s):** elapsed between SQLite `evaluations.created_at` and `updated_at` for valid evaluations (includes queue wait and any retries). It is measured per dimension, **not** per tutor turn.
- **Coverage:** counts of valid, skipped, invalid, error, queued, running, missing-reference, interruption, queue-full, and duplicate/contradictory ACKs.

## Installation (run once after confirming a clean Git state)

1. Copy `task6_telemetry.py`, `task6_instrument.py`, `task6_benchmark.py`, `test_task6_benchmark.py` into repository root.
2. In CMD `python task6_instrument.py install` (creates backup; refuses unknown code layout).
3. `python task6_instrument.py check` then `python -m unittest test_task6_benchmark -v` and the existing 28 Task 5 tests. Keep judge URL at `8004` in existing `task5_acceptance.py`; the Task 6 runner changes the URL only inside its own process.
4. No GPU job or SSH tunnel changes. Keep Llama at `127.0.0.1:8000`, FARE at `127.0.0.1:8004`, Next.js at `localhost:3000`.

## Controlled runs (CMD, one backend process at a time)

Use a **fresh browser conversation for each trial** (reload the page or reset conversation; do not let previous context accumulate). Disable the microphone for the paired baseline so VAD/barging does not confound timings. Hold Llama/Q4, FARE/Q4, STT, Kokoro, RAG `top_k=2`, teaching style, hardware, question, concurrency, and frontend configuration fixed. Warm model(s) before recording. Stop any prior backend process using port 8001.

Three baseline conditions:

```cmd
.\.venv\Scripts\python.exe task6_benchmark.py serve off --id off01
.\.venv\Scripts\python.exe task6_benchmark.py serve on --id on01
.\.venv\Scripts\python.exe task6_benchmark.py serve unavailable --id fail01
.\.venv\Scripts\python.exe task6_benchmark.py report
```

For each run, submit **`What is entropy?` exactly five times**, once per fresh conversation; let all audio finish. After the final turn in the ON condition, check `task6_benchmark.py report` and make sure `queued=0` and `running=0` before stopping the backend, so scores have finished processing. For UNAVAILABLE condition, a startup log indicating adapter health-check disablement is expected, and the tutor must still answer; it does not test mid-session judge failure.

For a stronger comparative result, collect counterbalanced repetitions: `off01 → on01 → on02 → off02`, plus `fail01`, using distinct run IDs. Report observed sample sizes and descriptive p50/p95 separately for each mode. Five turns is **only an engineering acceptance minimum**, not sufficient for precise statistical conclusions. A thesis-grade performance result should have a pre-registered larger sample and uncertainty estimate (e.g., paired bootstrap for ordered, independently reinitialized trials), with environmental drift controlled.

Run **two separate functional scenarios** outside the latency baseline: (1) deliberate interruption during actual browser audio, requiring 7 interrupted skips and no valid evaluation for that turn, (2) deliberately duplicate/delayed success ACK through an automated protocol test, verifying no double-scoring. The telemetry report flags duplicate, contradictory, and premature browser acknowledgements. Do not introduce deliberate interruption into normal-latency sessions.

## Acceptance gates

- Tests pass and Task 5 hashes remain unchanged.
- OFF: at least 5 completed browser ACKs, tutor latency samples, no new scalar-v3 evaluations.
- ON: at least 5 completed ACKs and 7 valid dimension evaluations **per delivered eligible entropy answer**, subject to missing reference and explicit skip rules; zero false-valid interrupted turns, zero invalid duplicate ACKs, zero unknown errors; queue drained.
- UNAVAILABLE: tutor completes playback despite FARE health failure; no new scalar scores incorrectly marked valid.
- Latency report has both `backend_total_p50_s` and `browser_completion_p50_s` populated. Do not describe them interchangeably.
- Evidence stored in `judge_outputs/task6/` (session logs, timing-only JSONL, manifest, per-run records, CSV). It may contain transcripts in **server logs** and unredacted judge `evaluations` exports—treat those files as private.

## Git and research hygiene

Commit only source (`task6_*.py`, tests, protocol, and the small `frontend_api.py` telemetry hook), not `judge_outputs/task6/`, SQLite, SSH details, backups or any private transcript. Always inspect `git diff` before staging. The engineering reference in `task5_acceptance.py` is **not** independently rated gold. The backend SHA256 of `build_scalar_prompt_v3` must be matched to frozen experiment provenance before claiming judge-selection equivalence in thesis results.

**Known limitations:** browser ACK timestamps are received on the server, not acquired from an acoustic playback sensor; SQLite `updated_at-created_at` is per-dimension evaluation turnaround; runtime judge is deliberately skipped for questions absent from exact reference registry; backend logs may contain user answers and must be handled accordingly. Startup-failure resilience is tested; live after-start judge crash is an independent future resilience case.
