# StreamTutor — Final Judge-Selection Decision Record

**Decision ID:** ST-JUDGE-2026-11-08
**Status:** Finalized *provisional engineering selection*; independent scientific validity **not established**
**Research context:** Master's thesis, spoken Data Mining tutor with retrieval-augmented generation and asynchronous answer evaluation
**Primary scalar judge:** `fare_20b_q4` — FARE 20B, GGUF Q4_K_M
**Engineering fallback candidate:** `compassjudger_2_7b_q4` — CompassJudger 2 7B, GGUF Q4_K_M
**Decision scope:** Frozen v3 comparison and Task 7 runtime evidence, **not** a claim of universal superiority or production certification.

## 1. Decision and rationale

Retain **FARE 20B Q4_K_M** as StreamTutor's **provisional primary asynchronous scalar evaluator**. Retain **CompassJudger 2 7B Q4_K_M** as the documented alternative if operational reliability, resource envelope, or independently validated rating quality favors it. Do not switch models silently: a switch changes evaluator provenance and requires a newly identified, versioned evaluation run.

The v3 decision uses an **explicitly post hoc** engineering screen, established after review of the candidate results. It requires overall output validity ≥99.5%, weak three-level response-quality ordering ≥95%, and bidirectionally usable contrast-pair coverage ≥90%. For passing models, the preference prioritizes fewer high scores on synthetically critical *correctness* and *groundedness* cases, then greater discrimination on the targeted stress subset as a descriptive tie-breaker. The criteria are transparent but **not preregistered**; different priorities or thresholds could produce a different choice.

The decisive engineering evidence is that FARE rated **0/72** synthetically critical correctness/groundedness cases at ≥4, while Compass rated **1/72**, and FARE had the larger excellent-to-critical rating gap in the selection set (**3.179** versus **2.250**) and targeted stress subset (**3.282** versus **2.231**). These proxies support selecting a candidate for further use; they do **not** establish correctness against independent expert labels. Compass's **zero invalid outputs** and lower per-request latency make it a credible alternative.

## 2. Frozen comparison and evidence boundary

The four-model benchmark contains **4,300 recorded judgments**: four models × (132 calibration + 792 selection + 123 targeted stress + 28 pairwise contrasts) = 1,075 per candidate. The selection ratings derive from **36 underlying question clusters**, not 792 independent questions. All candidates declared A40 48 GB, Q4_K_M, context 4096, maximum 256 new tokens, temperature 0, top-p 1, seed 0. The analysis retained invalid outputs as invalid; no missing score was coerced to a numeric rating.

| Candidate | Valid / 1,075 | Selection weak ordering | Critical correctness/groundedness ≥4 (72 cases) | Usable pairwise comparisons | Selection request p95 | Selection excellent−critical gap |
|---|---:|---:|---:|---:|---:|---:|
| FARE 20B | **1,073** | 245/251 (97.6%) | **0** | 14/14 | 1.45 s | **3.179** |
| CompassJudger 2 7B | **1,075** | 246/252 (97.6%) | 1 | 14/14 | **1.31 s** | 2.250 |
| GLM 32B | **1,075** | 235/252 (93.3%) | 6 | 14/14 | 3.16 s | 2.960 |
| Prometheus 7B v2 | 1,060 | 197/252 (78.2%) | 6 | 3/14 | 2.29 s | 1.817 |

**Interpretation controls:** p95 is **per-request serving latency**, not spoken-turn latency. Weak ordering means excellent ≥ borderline ≥ critical; it is a coarse ordinal discrimination check, not accuracy. All provisional source preferences in the pairwise contrast set used source ID `a`, although the displayed position was swapped; positional consistency under this setup does not prove source-label-balanced absence of bias. The targeted stress subset does not consist of paired one-factor interventions, so its summary is not a causal robustness estimate.

The paired, question-clustered bootstrap estimate of FARE minus GLM selection-set separation is **+0.219**, 95% interval **−0.024 to +0.480**; this interval includes zero. **No significant or definitive FARE-over-GLM advantage** is asserted on that metric. The apparent model ordering must not be treated as human-calibrated accuracy.

## 3. Case-level reasoning and unresolved calibration

Selected cases from the archived decision record show why score tables alone are inadequate:

| Case | Evidence pattern | Decision significance |
|---|---|---|
| `conv-05-t01-critical::conversational_context` | Incorrect claim that ID3 chooses the *smallest* information gain; FARE 1, GLM 5, Compass 3 | FARE detected a substantive error that GLM feedback incorrectly accepted; dimension overlaps factual correctness |
| `conv-04-t03-critical::conversational_context` | Incorrect Gini computation yields 0 rather than 0.375; FARE 1, GLM 5, Compass 2 | Important numerical-error detection case; not a clean pure-context test |
| `conv-08-t02-critical::conversational_context` | Incorrect K-means claim attributing changing clusters to changing data; FARE 2, GLM 5 | Suggests a GLM miss on a severe conceptual claim |
| `conv-01-t01-borderline::correctness` | Omits labeled training and fixed-model inference distinctions; FARE and GLM 5, Compass 3 | FARE may be permissive toward incomplete answers |
| `conv-15-t01-borderline::correctness` | “Good beats a baseline; best beats candidates” omits equal protocol and uncertainty; FARE 5, GLM/Compass 3 | Concrete calibration concern for borderline explanations |
| `conv-04-t02-borderline::groundedness` | Entropy intuition omits child-size weighting; FARE 4, GLM 3, Compass 5 | Disagreement may reflect completeness judgment rather than a clear error |
| `conv-05-t02-borderline::conversational_context` | Threshold split explanation omits threshold construction; FARE/Compass 3, GLM 5 | Context dimension is not well isolated by this example |

These cases are **expert-review priorities**, not independently adjudicated proof that any one judge is correct. Some nominal conversational-context examples are standalone or confound memory with correctness. Targeted dimension-specific human validation is required.

## 4. Runtime validation and what it does *not* prove

The Task 7 pilot used a frozen **11-question workload**, four repetitions per condition (44 FARE OFF and 44 FARE ON), with browser-driven answer playback. All **88/88** scheduled browser trials completed, and all **308/308** ON-condition scalar evaluations (seven dimensions for 44 delivered answers) reached valid status. No additional browser trials are required to close this pilot.

| Measured quantity | OFF | ON | Meaning |
|---|---:|---:|---|
| Median first-audio latency | 0.603 s | 0.600 s | No observed material median regression in this workload |
| Median browser playback completion | 78.317 s | 70.570 s | Heavily influenced by generated response/speech length; not evidence that FARE accelerated generation |

The paired mean ON−OFF first-audio difference was approximately **+0.0004 s**; the question-cluster bootstrap 95% interval was approximately **−0.053 to +0.045 s**. This is **not an equivalence test** and cannot establish zero overhead. In the accepted ON runs, the judge evaluations were created after the browser playback acknowledgment, supporting the claim that FARE was **outside the observed student-facing playback critical path**. Median time from playback acknowledgment to completion of all seven judge evaluations was approximately **5.95 s**. This delay is a downstream observability cost, not a delay imposed on the current spoken answer.

**Qualification:** The Task 7 workload was typed, so it did **not** measure microphone capture, VAD endpointing or STT. The 88 observations were repeated measurements on 11 distinct questions. Hardware scheduling, process restarts, possible resource sharing and changing answer lengths remain validity threats. Completion and scalar-format validity **do not** establish judge accuracy.

## 5. Deployment and operating envelope

- **A40:** Real FARE and tutor inference deployment was demonstrated on the university HPC environment; exact run-specific job/node/configuration provenance belongs in the deployment evidence manifest. A documented production availability SLA is **not established**.
- **A30 24 GB:** **NOT VERIFIED**. Neither FARE's nominal quantization nor a successful A40 allocation establishes A30 memory fit, initialization or acceptable serving latency. A separate genuine A30 smoke and memory-pressure test is required if that target is operationally necessary.
- **Co-location:** Task 7 infrastructure placed both serving jobs on `gpu04` at one point; this establishes neither exclusive devices nor absence of contention. Document scheduler GPU IDs, memory pressure and interference before making dedicated-hardware generalizations.
- **Slurm lifecycle:** A prior FARE job ended at the four-hour time limit. A running SSH tunnel does not guarantee its remote model server remains alive. Check the remote allocation, HTTP health and service identity for each block.
- **Condition identity:** The first ON pilot attempted to use a still-running OFF-configured FastAPI process. The corrected server exposes `/task7/block`; the runner checks block ID, ON/OFF condition, schedule and registry hashes, and telemetry target *before* sending a question. Preserve the failed preflight/incorrect-condition attempts separately, excluded from accepted results.

The decision is a **qualified research-engineering deployment**, not a claim of production certification or untested hardware compatibility.

## 6. Reversal and promotion criteria

**Reconsider FARE in favor of CompassJudger (or another candidate) if any of the following occurs:**

1. **Human validation:** blinded, independently adjudicated course-expert ratings reveal materially worse correctness/groundedness discrimination, more severe critical-error false negatives, or materially poorer calibration than Compass on a new held-out set.
2. **Operational validity:** on a new held-out runtime sample FARE fails the predeclared ≥99.5% valid-output screen, exhibits repeated unrecoverable parser errors, or systematically misses critical rubric cases.
3. **Resource viability:** FARE fails the actual target GPU memory/startup/latency gate or its resource contention degrades student-facing latency beyond a **predefined** acceptable margin; the margin must be declared before confirmatory testing, not selected retrospectively.
4. **Failure containment:** judge unavailability, slow requests, queue overload, or interruption blocks student-facing response delivery, or admitted evaluations are lost/duplicated in a way that violates the versioned runtime contract.
5. **Bias and generalization:** source-identity-balanced pairwise or conversation-dependent evaluation reveals systematic position/source bias or failure modes not captured by the original frozen benchmark.

**Promotion from provisional to scientifically validated judge selection requires** a new untouched, label-balanced validation set, independent expert raters, an adjudication protocol, item-level error analysis, and predeclared decision criteria. No human agreement or judge accuracy value is imputed from synthetic labels.

## 7. Traceability and frozen inputs

Primary version-controlled evidence:

- `docs/judge_evaluation/day4_comparison_v3.md` — four-model results, bootstrap comparisons and disagreement interpretation.
- `docs/judge_evaluation/day4_judge_selection_decision_v3.md` — initial post hoc selection rule, critical-case review and fallback.
- `docs/judge_evaluation/day4_runtime_evaluation_contract_v1.md` — delivered-answer eligibility, asynchronous boundary, identity/idempotency and failure semantics.
- `docs/task7/task7_results_report.md` — browser-level latency and runtime experimental findings.
- `docs/judge_evaluation/task8/task8_evidence_manifest.json` — archive and source provenance.
- `docs/judge_evaluation/task8/task8_reproduce.md` and `task7_reproduce.py` — reproducible Task 7 analysis.
- `data/judge_evaluation/streamtutor_judge_dataset_v3/` — frozen rubric, questions, synthetic response variants, schemas and source manifest.

Frozen Task 7 archive: `task7_evidence_20261108.zip`, **SHA-256** `47426b73fad70fdc6635159d2e89d33b290219a8a72d975acf74581f8fc0b20f`. Archive is intentionally retained outside ordinary Git source commits. The verified Task 7 reproduction checkpoint was committed as `12baca5` to branch `judge-selection-sprint`.

**Known evidence qualifications:** The Prometheus calibration manifest retains placeholder run ID `NEW_JOB_ID`; reconcile provenance through a separate record without editing frozen source evidence. The runtime contract's original document predates Task 7 integration and must be read with the later integration records; its historical status line is not evidence that integration remains absent.

## 8. Final disposition

**APPROVED — PROVISIONAL ENGINEERING USE** of FARE 20B Q4_K_M for post-delivery asynchronous scalar evaluation in the tested StreamTutor research configuration. **CompassJudger 2 7B Q4_K_M remains the engineering fallback candidate.**

**NOT APPROVED / NOT VERIFIED:** independently validated judge accuracy; A30 compatibility; production operational readiness; statistical equivalence of FARE ON/OFF latency; comprehensive voice-pipeline latency including STT; generalization beyond the frozen workload.

This decision closes the candidate-selection reasoning for the Day 4 handoff. The remaining Task 8 records must qualify deployment, prescribe independent human validation, and archive the reproducibility and risk register without changing the original benchmark evidence.
