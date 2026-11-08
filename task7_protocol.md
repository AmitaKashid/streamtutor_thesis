# Task 7 (proposed) — Expanded runtime benchmark: preregistration and readiness

## Status and provenance
Task 6 engineering pilot is complete (OFF/ON/UNAVAILABLE, five fresh-session text questions each). Task 7 is **proposed** here as the next milestone because no previously approved Task 7 definition was available. This document is a protocol, not live evidence or a claim of completion.

## Research question
Does asynchronous, delivery-gated scalar FARE evaluation materially change time-to-first-audio and backend response latency of the same StreamTutor configuration, without reducing valid evaluation coverage or creating browser-acknowledgment errors?

## Hard prerequisite: verified reference coverage
The Task 5 runtime adapter matches *exact question text* to a verified reference registry. The current engineering registry provides only variants of `What is entropy?`. All other planned questions will be skipped with `reference_unavailable` unless their reference answers are independently verified and added to a separate benchmark fixture. No invented answers or frozen-v3 benchmark leakage. **Do not run ON workload until reference-coverage preflight is 100%.**

## Workloads and design
- 12 text-input questions: three each from definition, why/intuition, procedure, and application. Fix their exact wording and validate course coverage; see `task7_workload.json` (illustrative; not yet validated against slide corpus).
- Four replicates per question per condition (OFF and ON): **96** measured text turns across eight ABBA-order slots per question. Alternate condition first across questions to mitigate order effects; use clean conversational context for each turn.
- Warm the tutor and judge before the first recorded session. Record hardware nodes, GPU allocation, quantization, tokenizer, decoding parameters, TTS voice/rate, retrieval settings, model hashes, frontend and browser versions, git commit, and all SSH tunnel endpoints (without secrets).
- Never run concurrent tutor trials. Use a fresh conversation per measured trial and keep the mic disabled for this text benchmark.
- Treat ambiguous pronoun/follow-up turns as a **separate two-turn contextual workload**, not independent prompts. Similarly, benchmark speech STT separately with fixed WAV inputs and matched transcripts.
- Each trial should yield an actual audio-playback completion event; do not infer browser completion from backend `audio_done`.
- Run an additional deliberate interruption scenario and post-initialization judge-drop test separately from normal latency trials; record whether valid scores were inappropriately generated.

## Outcomes and statistical analysis
- Primary: tutor backend time to first audio. Secondary: LLM TTFT, backend response completion, server-observed browser ACK, browser queue drain, per-dimension judge inference time, time to final score, success coverage, and protocol anomalies.
- Compare OFF and ON per exact prompt using differences of per-prompt mean or median across replicates; report raw per-run data and both mean/median, p50/p95, and uncertainty intervals. The schedule is stratified by question; do not treat seven evaluations from one turn as seven independent observations.
- Use a question-cluster bootstrap (resample prompts, preserving their replicates) for uncertainty around the OFF-versus-ON contrast; present sensitivity analyses for output length and GPU scheduling drift. An observed near-zero difference is **not** evidence of equivalence without a prespecified equivalence margin and adequate power.
- Record generated text length or tokens and synthesized audio duration; differing answer lengths can distort playback and backend-total comparisons.
- Mark all incomplete, interrupted, missing-reference, and judge-error cases; no silent exclusion.

## Acceptance gates
1. Frozen rubric SHA256 `b4dbaa6569f410eb5199982fd546aefc0c59bb2786e5eb951b71dfec04b47ed8` remains unchanged and prompt hash matches archived model-selection evidence (not just current source).
2. Approved workload is fixed, reference registry has verified references for all 12 exact strings, and every workload item has a provenance/verification record.
3. Measurement runner preserves session/turn ID correlation, and all 96 planned trials either have complete records or explicit failure classifications.
4. ON score coverage is reported at per-turn and per-dimension levels, excluding dimensions correctly deemed not applicable; judge queue drained and false-valid interrupted count zero.
5. Output-length and hardware/drift covariates recorded; result limitations and uncertainty documented.
6. Results archived separately from frozen v3 judge-selection benchmark; only reviewed source/protocol code committed to Git.

## Execution boundaries
Do not launch the 96-trial workload until gates 1–2 pass. Existing `task6_benchmark.py` supports manual browser turns but has no enforced question scheduling or automatic per-prompt ON/OFF pairing; extend/instrument the runner before declaring the above schedule executed. The present `task7_plan.py` creates a deterministic, auditable schedule and refuses uncovered references; it does not automate browser interaction or infer successful trials.
