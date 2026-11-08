# Task 8 — Runtime and Deployment Qualification Gate

**Decision ID:** ST-JUDGE-RUNTIME-2026-11-08
**Status:** Qualified for the observed research-pilot configuration only; not production-certified
**Scope:** FARE 20B Q4_K_M as asynchronous scalar evaluator of delivered answers in StreamTutor
**Companion records:** `docs/judge_evaluation/task8/judge_selection_decision_final.md`, `docs/task7/task7_results_report.md`, `docs/judge_evaluation/day4_runtime_evaluation_contract_v1.md`, `docs/judge_evaluation/gpu_resources_v3.md`, `docs/judge_evaluation/tutor_interference_v3.md`

## 1. Executive deployment decision

**PASS — observed research-pilot functionality:** The recorded Task 7 system successfully delivered 88/88 browser-driven, text-initiated tutor turns (44 judge OFF, 44 ON). For 44 accepted judge-ON turns, FARE wrote 308/308 valid, one-attempt, seven-dimensional evaluations. The accepted evaluation records were created at the browser audio-playback completion acknowledgment, and the last score completed a median of 5.95 seconds later. This verifies the observed delivery-triggered asynchronous boundary for these trials.

**NOT VERIFIED — production deployment or general hardware compatibility:** Neither A30 24 GB startup and memory fit nor dedicated-GPU isolation, worst-case co-residency, sustained concurrent load, full live STT latency, mid-run judge outage resilience, and human-calibrated scoring accuracy has been certified. An A40 success is not an A30 success. Judge validity as an output-contract property is not evidence of judgment accuracy.

**Operating decision:** Retain FARE as a *provisional* research-primary judge, behind a nonblocking asynchronous interface, with a clearly labeled fallback candidate (CompassJudger 2 7B). Do not silently substitute the fallback without a versioned model/configuration record and separate acceptance run. Do not describe this as production-ready.

## 2. Configuration and source identity

| Attribute | Observed or declared configuration | Evidence / caution |
|---|---|---|
| Tutor model | `llama3_8b_q4` | Task 7 report and runtime logs; tutor is not the judge |
| Judge model | `fare_20b_q4`, GGUF Q4_K_M | Frozen v3 comparison and Task 7 SQLite identity |
| Speech synthesis | Kokoro | Task 7 report |
| Retrieval | Slide RAG enabled | Task 7 report; retrieval relevance not independently assessed in this run |
| Benchmark stimuli | Eleven distinct browser-typed standalone questions, four repetitions per condition | No live microphone/STT benchmarking |
| Conditions and order | `b01_off`, `b02_on`, `b03_on`, `b04_off`, `b05_on`, `b06_off`, `b07_off`, `b08_on` | Fixed partially balanced blocks, not randomized paired interleaving |
| Service endpoints in Windows setup | Tutor `127.0.0.1:8000`; judge `127.0.0.1:8004`; application `127.0.0.1:8001` | SSH-forwarded to allocated HPC nodes; local ports alone do not certify topology |
| HPC jobs observed | Tutor job `507645` on `gpu04`; replacement judge job `507671` on `gpu04`; previous judge job `507595` on `gpu05` | User-captured Slurm records; node sharing is **not proof** of physical GPU sharing |
| Raw evidence archive | `task7_evidence_20261108.zip` | SHA-256 `47426b73fad70fdc6635159d2e89d33b290219a8a72d975acf74581f8fc0b20f` |
| Reproducible analysis | `task7_reproduce.py` and `docs/judge_evaluation/task8/task8_reproduce.md` | Local replay passed 88/88, 308/308 against matching archive hash |
| Code baseline | `82e5a9898cf1826803594731cc842aba73973738` at initial Task 8 evidence manifest; later reproduction checkpoint `12baca5` | Record final handoff commit separately; do not mistake the archive's earlier trial-code commit for the final documentation revision |

**Hardware attribution boundary:** Historical candidate screening used declared NVIDIA A40 48 GB resources. Task 7 user logs identify shared **node** `gpu04`, but do not include verified per-process physical GPU UUID mapping, per-process VRAM footprint, or an isolated-vs-shared topology experiment. Consequently, GPU contention cannot be ruled in or out from these records alone.

## 3. Functional and timing gates

| Gate | Verdict | Evidence | Scope/qualification |
|---|---|---|---|
| Browser delivery completion | **PASS** | 88/88 accepted trials contain completed browser acknowledgments and audio completion events | One selected repeated-question pilot, with setup failures excluded and archived |
| Condition isolation | **PASS for corrected runner and accepted blocks** | Runtime block-identity endpoint checks block ID, ON/OFF condition, schedule hash, registry hash and telemetry file; guard regression tests passed | Earlier wrong-backend incident motivated correction; not a claim of fault-free early execution |
| Asynchronous judge eligibility | **PASS for observed ON turns** | Judge rows created at playback acknowledgment, completed afterward | Does not prove noninterference under arbitrary load or mid-playback disruption |
| Judge scalar response contract | **PASS for accepted Task 7 turns** | 308/308 `valid`, score 1–5, seven dimensions, one recorded attempt per evaluation | Valid format does not establish semantic correctness of scores |
| Median first backend audio-chunk event | **Observed, no gate threshold defined** | OFF 0.603 s vs ON 0.600 s; paired mean ON−OFF +0.0004 s | Backend `audio_chunk` event, not speaker audible-onset measurement |
| First-audio effect precision | **NOT VERIFIED as equivalence/noninferiority** | Question-bootstrap 95% CI [−0.053,+0.045] s | No prespecified equivalence margin or power target; interval includes positive overhead |
| Complete browser playback | **Observed, not judge speed benefit** | Median OFF 78.317 s vs ON 70.570 s | Depends on answer length and speech duration; nonrandomized block effects |
| Judge completion delay after playback | **Observed** | Median 5.95 s until last of seven scores completed | This occurs after user-facing delivery; may matter for delayed analytics and queue capacity |
| Unavailable judge at startup | **Limited engineering pilot** | Earlier Task 6 unavailable-at-startup condition continued tutor operation | Not equivalent to recovering from a judge crash mid-run |
| Queue-full, crash recovery, interruption, high concurrency | **NOT VERIFIED by Task 7** | Contract contains intended behavior and unit tests | Require fault-injection and sustained load before production claims |
| Source-grounded judge accuracy / expert agreement | **NOT VERIFIED** | No independent human raters; synthetic provisional labels | Requires separate blinded human validation |

**Timing terminology:** `turn_to_first_audio_s` is turn start to the first *backend* `audio_chunk` event. `browser_ack_s` is turn start to browser playback completion. Neither directly measures microphone end-of-utterance to first sound from the output device. The 88-trial study bypassed STT by submitting text in the browser.

## 4. Hardware qualification matrix

| Deployment target | Status | Justification / next gate |
|---|---|---|
| A40 48 GB, historical candidate screening | **PASS for recorded candidate loading and bench execution** | Frozen v3 judge benchmark declares A40 Q4_K_M runs and candidate evidence logs show A40 activity. Does not imply generic deployment certification. |
| A40-class Slurm-served runtime, observed Task 7 | **PASS for serving and functional pilot** | Tutor and FARE endpoints served real HTTP requests, pilot completed; exact physical device isolation not captured. |
| A30 24 GB with FARE 20B Q4_K_M | **NOT VERIFIED** | No successful A30 allocation + complete judge startup + memory sampling + genuine inference record. |
| Shared physical GPU tutor/judge | **NOT VERIFIED** | Shared Slurm node reported, but GPU UUID mapping and overlapping load not proven; cannot infer interference. |
| Separate physical GPU tutor/judge | **NOT VERIFIED as controlled comparison** | No complete matched isolation-vs-co-residence dataset presented. |
| Production multitenant deployment | **NOT VERIFIED** | No capacity, error budget, persistence/restart or privacy qualification under realistic concurrency. |

For a future A30 gate, require an actual A30 job, immutable model/quantization/configuration identifiers, successful health plus genuine seven-dimension inference, physical GPU UUID-to-process mapping, loaded-idle and in-run memory sampling (including minimum free VRAM), startup/error logs and fixed-case runtime latency. Use `docs/judge_evaluation/gpu_resources_v3.md` for sampling requirements. If any essential observation is missing, the verdict stays **NOT VERIFIED**, rather than PASS or FAIL.

## 5. Operational incidents and engineering controls

| Incident observed in Task 7 execution | Diagnosed cause | Corrective/containment measure |
|---|---|---|
| Tutor endpoint unreachable while an SSH listener existed | No active tutor server behind the tunnel until job `507645` launched | Verify remote GPU job and remote `/health` before local tunnel; keep tunnel open |
| Judge endpoint unavailable after previously passing health | Judge Slurm job `507595` reached its four-hour time limit on 2026-11-08 at 15:03 local cluster time | Record job expiration; launch replacement job (`507671`) and establish fresh tunnel to assigned node |
| ON question spoken but ON telemetry empty | The `b01_off` FastAPI process still owned port 8001 during a supposed `b02_on` run | Block-aware live endpoint `/task7/block`; runner refuses a mismatched condition/manifest/telemetry target before submitting any question |
| Runner blocked reuse after an infrastructure-failed submission | Fail-closed attempt tracking prevented silent retry | Archive invalid attempts and resume only against a verified condition-matched manifest |
| ON audit briefly reported queued/running evaluations | Judge work outlasted browser speech | Wait for terminal states while retaining service; audit exact turn IDs before switching blocks |

**Operational procedure:** Before each block, record (a) scheduler job ID, node and wall-time remaining, (b) remote model/health, (c) local tunnel health, (d) `/task7/block` identity, (e) schedule/registry hashes. At block completion, verify 11 unique browser completions; for ON blocks, verify exactly 77 distinct valid evaluation records before stopping the judge worker. Preserve error and setup attempts as a separate operational cohort; do not silently insert them into accepted latency statistics.

## 6. Promotion requirements and reversal triggers

**Pilot continuation — permitted:** Use the current FARE asynchronous scalar judge for research evaluation of completed turns on the already tested environment, with explicit experimental scope, monitoring and recorded model identity.

**Hold production promotion** until independent human validity, explicit data privacy/retention review, crash/restart durability, queue backpressure, mid-run outages, multiuser load, and device-level latency definitions are tested. Hardware-specific production targets must pass on actual target hardware. No combined “production ready” verdict is warranted now.

**Reconsider FARE selection** if targeted expert review finds systematic missed critical errors or harmful borderline leniency, if output contract validity falls below the chosen acceptance threshold in a new representative workload, or if an actual target GPU fails memory/latency qualification. CompassJudger remains an *engineering fallback candidate*, not a transparent automatic replacement. The Day 4 post hoc thresholds and rationale are detailed in `docs/judge_evaluation/day4_judge_selection_decision_v3.md` and the final Task 8 decision record.

## 7. Reproducibility and evidence references

- Candidate selection and hardware declarations: `docs/judge_evaluation/day4_comparison_v3.md`, `docs/judge_evaluation/day4_judge_selection_decision_v3.md`, `docs/judge_evaluation/evidence_archive_v3.md`.
- Runtime contract and interference protocol: `docs/judge_evaluation/day4_runtime_evaluation_contract_v1.md`, `docs/judge_evaluation/tutor_interference_v3.md`.
- Task 7 experimental measurements and validity limitations: `docs/task7/task7_results_report.md`.
- Raw Task 7 frozen evidence: SHA-256 above, archived separately; not a repository-tracked artifact.
- Reproduction: `task7_reproduce.py task7_evidence_20261108.zip --out task7_reproduced`; requires the same evidence archive and project dependencies.
- Task 8 manifest: `docs/judge_evaluation/task8/task8_evidence_manifest.json`.

**Document limitation:** This gate reviews the currently available records; it is not an independent replication of physical GPU residency, resource utilization or scheduler allocation constraints. Missing hardware provenance remains explicitly unverified.
