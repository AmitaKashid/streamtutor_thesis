# Day 4 — Provisional LLM-as-judge selection decision (v3)

**Date:** 2026-11-08  
**Status:** Provisional engineering selection; **not independent scientific validation**.  
**Primary scalar judge:** `fare_20b_q4` (FARE 20B, Q4_K_M).  
**Fallback scalar judge:** `compassjudger_2_7b_q4` (CompassJudger 2 7B, Q4_K_M).  
**No permanent model exclusion:** GLM 32B remains a benchmark comparator; Prometheus 7B remains a scalar comparator, but its frozen pairwise format is unreliable.

## Decision boundary and purpose

The purpose of the judge is to assess completed answers from a voice-based, RAG-grounded Data Mining tutor, using a rubric including correctness, groundedness, conversational context, pedagogy, clarity, completeness, and learner appropriateness. The judge is an *asynchronous evaluation service* and must not be placed on the student-facing STT → retrieval → LLM → streaming TTS critical path. The selected judge is **not** the tutor generation model.

The frozen v3 dataset has four models × (132 calibration + 792 selection + 123 stress/robustness + 28 A/B contrasts) = **4,300 recorded judgments**. Candidate manifests share data/rubric hashes and declared A40 48 GB, Q4_K_M, 4096-token context, 256 output tokens, temperature 0, top-p 1, seed 0. Most scores are on synthetic response variants with *provisional* quality labels; **36 underlying selection question clusters**, not 792 independent questions.

## Decision rule — explicitly post hoc

The rule was defined after examining the benchmark, so it is a *transparent engineering decision*, **not a preregistered hypothesis test**. First require overall output validity ≥99.5%, weak response-quality ordering ≥95%, and bidirectionally usable contrast-pair coverage ≥90%. Among passing candidates, prioritize fewer scores ≥4 on synthetically **critical correctness and groundedness** answers, then greater separation on the independent targeted stress subset. The latter is a descriptive tie-breaker, not a human-calibrated quality metric. This screen selects FARE with Compass as fallback. Altering thresholds could alter the ranking; do not present it as a uniquely determined scientific winner.

| Model | Overall valid | Selection weak-order consistency | Critical correctness/groundedness rated ≥4 (72 cases) | Contrast usable pairs | Selection p95 request latency |
|---|---:|---:|---:|---:|---:|
| GLM 32B | 1,075/1,075 | 235/252 (93.3%) | 6/72 | 14/14 | 3.16 s |
| CompassJudger | 1,075/1,075 | 246/252 (97.6%) | 1/72 | 14/14 | 1.31 s |
| **FARE 20B** | **1,073/1,075** | **245/251 (97.6%)** | **0/72** | **14/14** | **1.45 s (reported attempt p95)** |
| Prometheus 7B | 1,060/1,075 | 197/252 (78.2%) | 6/72 | 3/14 | 2.29 s |

Selection p95 uses the benchmark run summary per-attempt metric. It is **not full turn latency or time to first audio**. FARE selection p95 from the original summary is 1.448 s; other rounded metrics are from the same run-level evidence. The analysis tool also exports a near-equivalent nearest-rank metric, which should not be substituted for exact run-summary p95. Invalid outputs stay invalid (never coerced to scores).

### Additional evidence

- Selection excellent−critical mean separation: FARE **3.179**, GLM **2.960**, Compass **2.250**, Prometheus **1.817**. A paired question-cluster bootstrap of FARE minus GLM yields **+0.219**, 95% percentile CI **−0.024 to +0.480**, which includes zero; do not claim a statistically established advantage over GLM on this metric.
- Selection weak-order consistency: FARE **245/251**, Compass **246/252**. Both order synthetic quality levels similarly despite different absolute score calibration.
- Targeted stress-set excellent−critical separation: FARE **3.282**, GLM **2.333**, Compass **2.231**, Prometheus **2.410**. Stress subsets are not controlled matched perturbations, so these values do not demonstrate causal robustness or invariance.
- FARE has **two invalid scalar outcomes**, one calibration and one selection; Compass and GLM have none. If an evaluation service must maximize completeness above critical-error detection, Compass is the reasonable first alternative. Use explicit retry/invalid handling, never invent missing scores.
- All 14 contrast provisional preferred source IDs are `a` even though display positions were swapped, and preferences were AI-annotated. The 14/14 position consistency of GLM/Compass/FARE is **not proof of general freedom from position/source bias**.

## Evidence inspection — cases requiring caution

Review based on the raw frozen question/answer/reference/feedback records, not the score table alone:

1. `conv-05-t01-critical::conversational_context`: The answer claims ID3 chooses the **smallest** information gain, instead of the maximum. FARE scored 1 and explains the factual mistake, while GLM scored 5 and describes the wrong answer as correct. Compass scored 3. This is a concrete severe GLM feedback failure, although the *context dimension* is confounded with factual accuracy.
2. `conv-04-t03-critical::conversational_context`: The answer computes Gini as `1 - (0.75 + 0.25)^2 = 0` instead of `1 - (0.75² + 0.25²) = 0.375`. FARE scored 1 and identified the error; GLM scored 5 and claimed the calculation was correct. Compass scored 2.
3. `conv-08-t02-critical::conversational_context`: An answer asserts that the *data change randomly on each K-means run*; the referenced explanation is about initialization and local optima. FARE scored 2; GLM scored 5. The raw answer also makes an unsupported claim that identical initial centroids must yield different clusters.
4. `conv-01-t01-borderline::correctness`: The answer distinguishes learning patterns and predicting labels but omits that training uses **labeled** examples and inference uses a fixed model. FARE and GLM scored 5; Compass scored 3. FARE's rating can be interpreted as lenient on omitted nuance, not necessarily factually incorrect; this is a calibration concern.
5. `conv-15-t01-borderline::correctness`: “Good beats a baseline; best beats candidates” omits equal-protocol comparison and uncertainty. FARE scored 5; GLM and Compass scored 3. This is additional evidence of FARE's permissiveness toward incomplete borderline responses.
6. `conv-04-t02-borderline::groundedness`: The entropy explanation captures the main intuition but omits child-size weighting. FARE scored 4, GLM 3, Compass 5; this disagreement is about degree of completeness, not a clear-cut factual mistake.
7. `conv-05-t02-borderline::conversational_context`: The answer supplies threshold-based binary splits but omits candidate threshold construction. FARE and Compass scored 3, GLM 5. The question alone does not cleanly isolate memory of conversational context.

**Methodological hazard:** Several examples labeled `conversational_context` are standalone or conflate correctness with memory use. Their disagreement demonstrates real evaluation inconsistency, but does not prove one judge is better at turn-level context tracking. The independent validation set should isolate each rubric dimension, including whether prior context is necessary to interpret the query.

## Unresolved issues / conditions for promotion

1. **Human validity:** Obtain blinded reviews from a knowledgeable course instructor or independent expert(s), stratified across dimensions and cases where the judges disagree. Estimate agreement and adjudicate disagreements; synthetic categories must not be called ground truth. Without this, the claim is *provisional engineering selection* only.
2. **Dataset leakage / post hoc rule:** The selection, case inspection and thresholds used the same frozen v3 set. Use a new untouched, source-label-balanced validation set for confirmatory comparisons. Do not retrofit the frozen test set or report the rule as preregistered.
3. **Judge runtime interface:** Validate the selected model's scalar contract and parse behavior on the actual runtime judge service. The older `LocalLLMJudge` code cannot be assumed to implement v3 just because benchmark code does.
4. **Latency isolation:** Prove async evaluation cannot block STT, streaming LLM, TTS, cancellation, or first audio, and record the actual judge service overhead, retry path and queue backpressure independently.
5. **Hardware:** A40 results do not prove A30 compatibility or GPU co-residency with the tutor stack. Treat the A30 allocation/smoke test as a separate gate and use Compass if FARE exceeds the available memory/runtime envelope.
6. **Provenance:** The Prometheus calibration manifest retains `NEW_JOB_ID` in the run ID. Preserve it and separately reconcile the scheduler record without mutating original evidence.

## Decision, operational recommendation, and stopping rule

**Select FARE 20B provisionally as the primary scalar evaluator for further integration testing**, not as a final validated winner; configure CompassJudger as a tested fallback candidate, not an automatic substitution without a versioned evaluation run. No further frozen v3 benchmarks are needed for this decision. The next Day 4 task is to define an **async runtime evaluation contract** with stable turn IDs, finalized answer semantics, captured RAG evidence, retries/invalid outputs, and non-interference testing. Escalate to independent validation before publishing judge-accuracy claims. If FARE fails runtime format/memory/latency isolation gates, document the gate failure and provisionally switch the integration candidate to Compass; **do not change the frozen v3 evidence to rationalize the switch**.

## Reproduction

From repo root with `day4_task2_frozen_inputs.zip` present locally:

```cmd
.\.venv\Scripts\python.exe -m unittest judge.test_select_provisional_judge_v3 -v
.\.venv\Scripts\python.exe -m judge.select_provisional_judge_v3 --input-zip day4_task2_frozen_inputs.zip --output-dir judge_outputs\day4_selection
```

The decision runner writes `provisional_decision_metrics.json` and `critical_risk_summary.csv`. The unmodified frozen input archive is excluded from source-control commits.
