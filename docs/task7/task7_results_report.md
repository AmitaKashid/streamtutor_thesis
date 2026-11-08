# Task 7 — FARE Runtime Integration: Experimental Results and Evidence Audit

**Experiment date:** 8 November 2026  
**Source:** `task7_evidence_20261108.zip` (eight block event logs, eight trial result files, frozen schedule/manifests, SQLite judge spool)  
**Status:** Executed and accepted. **Inference:** exploratory repeated-measures engineering pilot, not a population-generalizable efficacy study.

## Executive conclusion

The runtime ON/OFF study completed **88/88 planned tutor turns**, with **44 OFF and 44 ON**, and generated **308/308 valid FARE 20B Q4 dimension scores** for ON turns. The judge evaluation records were created at the browser audio-completion acknowledgment, and the last judge result finished a median of **5.95 s** afterward. Therefore the observed judge scoring was **outside the critical path of audible answer delivery** for these ON trials.

First-audio responsiveness was essentially identical: **median 0.603 s (OFF) versus 0.600 s (ON)**; the mean within-question-and-repetition ON − OFF difference was **+0.0004 s**. A 95% bootstrap confidence interval constructed by resampling **11 question-level average differences** was **[−0.053, +0.045] s**. This is consistent with no material change in this pilot, **not** proof of a zero or equivalently small effect. A noninferiority/equivalence claim would require a prespecified latency margin and an adequately powered design.

The median elapsed time to **complete audio playback** was **78.317 s OFF** and **70.570 s ON**. This measure is heavily influenced by response content/length, speech duration, and potential experiment-order or hardware variation. It cannot be interpreted as evidence that judging accelerates tutoring.

## 1. Design and validity

- **Question units:** 11 distinct, standalone questions from the frozen v3 course benchmark.
- **Repetitions:** 4 per condition, per question.
- **Total:** 11 × 4 × 2 = **88 trials**, across eight blocks.
- **Block order:** `b01_off`, `b02_on`, `b03_on`, `b04_off`, `b05_on`, `b06_off`, `b07_off`, `b08_on`.
- **Models:** `llama3_8b_q4` tutor; `fare_20b_q4` judge during ON; Kokoro TTS; slide retrieval enabled in the established tutor configuration.
- **Stimulus:** Browser-typed questions. This experiment **does not benchmark microphone capture or STT**.
- **Observations:** Browser-derived playback acknowledgments; backend monotonic event timestamps; judge SQLite rows linked by `turn_id`.
- **Data integrity:** Frozen manifest hashes agree across blocks; 88 recorded results and each matching turn has start, first audio chunk, `audio_done`, and `browser_audio_finished` events. Earlier unsuccessful setup attempts are archived separately, not counted in the 88 successes. `b01_off.events.jsonl` includes 12 turns because an additional earlier attempt is logged; valid measurements are joined to the 11 accepted trial IDs.
- **Correlated measurements:** Four repetitions for the same question are *not* four independent questions. Confidence intervals below resample the **11 question-level averages**, not all 44 pairs independently.

## 2. Performance results

| Metric | OFF (44 trials) | ON (44 trials) | Interpretation |
|---|---:|---:|---|
| First audio, median | 0.603 s | 0.600 s | Initial audible responsiveness |
| First audio, p95 | 0.811 s | 0.779 s | Descriptive tail estimate; small n |
| Backend `audio_done`, median from turn start | 3.029 s | 2.641 s | Audio-generation endpoint, not total audible duration |
| Browser playback completion, median | 78.317 s | 70.570 s | Includes spoken playback and content-length effects |
| Browser playback completion, mean | 76.456 s | 71.650 s | Descriptive only |
| Browser playback completion, p95 | 103.118 s | 101.078 s | Approximate empirical percentile |
| Audio chunks, mean | 11.841 | 11.000 | Content-length proxy, not token count |

### Paired question-level comparisons

| Endpoint | Mean ON − OFF | 95% question-bootstrap interval | Two-sided question-level paired t-test (exploratory) |
|---|---:|---:|---:|
| First audio | +0.0004 s | [−0.053, +0.045] s | p = 0.989 |
| Backend `audio_done` | −0.170 s | [−0.365, +0.002] s | p = 0.116 |
| Complete playback | −4.805 s | [−10.180, +0.247] s | p = 0.118 |

Bootstrap method: deterministic seed `20261108`, 30,000 resamples of 11 question-level average paired differences, using percentile 2.5% and 97.5% limits. Pairing was by `(question_id, repetition)`; block order was not randomized within each condition. These analyses are **exploratory** and not adjusted for multiple outcomes.

**Critical measurement distinction:** `browser_ack_s` spans backend tutor turn start to browser audio playback completion and matches the event timestamp difference to within 0.00005 seconds; it is **not** LLM time-to-first-token or time to first audible answer. `turn_to_first_audio_s` is the interval from tutor-turn start to the first logged backend `audio_chunk`, not independently verified speaker-output time. More granular LLM/TTS and device-level attribution would require instrumentation not present in the uploaded archive.

## 3. Judge reliability, scores, and asynchronous processing

The SQLite database contains 371 total evaluation rows (including older/non-Task-7 records). **Only rows whose `payload.turn_id` matches one of the 44 accepted ON turns were included.** Exactly **308** such rows were found, representing seven dimensions for each ON answer. All 308 have `status='valid'`, score in 1–5, and **one execution attempt**. All have `outcome='delivered'`, `delivery_completed=true`, and judge ID `fare_20b_q4`. The 21 `skipped` rows in the full database are **outside the selected Task 7 ON-turn population** and must not be counted against its reliability.

| Dimension | Mean score / 5 | Median |
|---|---:|---:|
| Correctness | 4.68 | 5.0 |
| Clarity | 4.39 | 4.0 |
| Completeness | 4.11 | 4.0 |
| Pedagogy | 4.41 | 4.5 |
| Learner appropriateness | 4.27 | 4.5 |
| Conversational context | 4.18 | 4.0 |
| Groundedness | 4.39 | 5.0 |

**Interpretation of score levels:** these are the *judge model's own assessments*, not independently validated human scores or externally established answer correctness. With no human reference ratings, they cannot establish criterion validity or judge-human agreement.

Timing: median per-dimension `updated_at − created_at` was **3.265 s** (mean **3.507 s**, p95 **6.419 s**). The median time from browser playback acknowledgment to completion of the **last** score for a turn was **5.95 s** (mean **6.22 s**). All 44 ON turns had their evaluation rows created at the playback acknowledgment (to the timestamp precision available) and had the last score recorded afterward. The judge was asynchronous relative to **user-perceived playback completion**, by construction in this runtime protocol. Per-dimension elapsed values overlap due to concurrent work; do not sum them as a per-turn overhead.

## 4. Threats to validity

1. **Small set of unique questions.** Eleven distinct questions limit coverage and statistical precision. The 88 trials are repeated measures, not 88 unique contexts.
2. **Order and drift.** Conditions were run in eight fixed, partially balanced blocks rather than randomized paired interleaving. The observed block median playback times vary considerably; cache effects, generated-answer variation, machine load, and temporal drift cannot be isolated.
3. **Infrastructure changes.** GPU allocations and SSH connectivity were restored during the session. Tutor and judge jobs may have shared a GPU node. Archived failures were excluded from acceptance results and should be included in an operational incident record rather than concealed.
4. **Different spoken-answer lengths.** ON generated roughly 0.84 fewer audio chunks on average. This variation can strongly affect playback completion latency. Response text, token count, audio duration and speech rate were not jointly controlled in this comparison.
5. **No audio acquisition.** Browser-typed input bypassed live STT; these results do not characterize full voice-turn latency including capture and transcription.
6. **Telemetry timing.** First `audio_chunk` is a backend event, not device-level audio onset. Exact audible onset needs client-side audio playback timing or loopback measurement.
7. **No independently established judge accuracy.** All scores passed formatting/contract validation, but validity of their *substantive judgments* was not tested against humans in this archive.
8. **No predefined equivalence margin.** A nonsignificant difference is not proof of absence of latency cost. Future tests should prespecify an acceptable overhead margin and target confidence interval width.
9. **Selection bias and generalization.** This is a selected, standalone 11-question workload, not the full v3 benchmark and not conversation-rich/ambiguous ASR-error testing.

## 5. Evidence-grounded conclusion

For the chosen standalone Data Mining tutoring questions, the browser reliably completed all scheduled speech responses under both conditions, and FARE produced a valid seven-dimension score set for all 44 judged turns. Since evaluations were queued at playback completion and finished afterward, **the implemented judge path did not block audible response delivery in this recorded experiment**. The paired first-audio measurements showed no detectable ON penalty (mean difference +0.0004 s; question-bootstrap 95% CI [−0.053, +0.045] s), but the dataset does not justify a formal claim of equivalence or broader latency guarantees. Playback completion time varied with generated speech content and should not be interpreted as a judge-performance improvement.

**Task 7 status:** experimental execution and integrity checks complete. This report is a finished descriptive and exploratory analysis; final project checkpoint should include versioned code and manifests, the CSV analysis tables, provenance of GPU allocations, and a privacy review before publishing the original SQLite evidence.

## Reproducibility artifacts

- `task7_per_trial_analysis.csv`: 88 accepted turns, condition, question, repetition, monotonic event intervals, playback ACK, chunk counts and IDs.
- `task7_judge_evaluations.csv`: 308 accepted ON-turn dimension scores, statuses and per-dimension processing elapsed times.
- `task7_evidence_20261108.zip`: source evidence; preserve unchanged for auditability.

The analysis uses parsed event data directly, pairs trials by question and repetition, and filters SQLite records by *accepted ON-turn IDs*. It does not impute missing observations. All planned accepted observations are present.
