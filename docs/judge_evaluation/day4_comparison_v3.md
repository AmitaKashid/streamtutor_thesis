# Day 4 — Four-model judge comparison (v3)

**Status:** Task 2 offline analysis complete; scientific selection remains provisional. 8 November 2026.

## Design and integrity

The analysis consumes 16 immutable run manifests and 4,300 raw evaluation records (four models × calibration 132 + selection 792 + robustness 123 + pairwise 28). Dataset, rubric, task-plan and Git hashes agree across candidate runs. All 17 invalid judgments remain included in validity calculations; scalar scores from invalid outputs are absent rather than coerced to 1. One Prometheus calibration run retains the original placeholder run ID (`NEW_JOB_ID`); provenance should be resolved without altering source evidence.

All candidates used declared A40 48 GB, Q4_K_M, context 4096, 256 new tokens, temperature 0, top-p 1 and seed 0. Results are tied to this *specific* prompt/output contract and hardware setup; comparisons must not be extrapolated to another serving configuration.

## Selection-set discrimination

| Judge | Valid selection scores | Mean excellent−critical | Question-clustered 95% percentile CI | Weak quality order | Strict quality order |
|---|---:|---:|---:|---:|---:|
| GLM 32B | 792/792 | 2.960 | 2.675–3.202 | 93.25% | 44.05% |
| CompassJudger 2 7B | 792/792 | 2.250 | 2.020–2.464 | 97.62% | 44.44% |
| FARE 20B | 791/792 | 3.179 | 3.052–3.302 | 97.61% | 47.01% |
| Prometheus 7B v2 | 792/792 | 1.817 | 1.492–2.147 | 78.17% | 28.17% |

The 252 expected question-by-dimension triples per judge arise from **36 underlying question clusters**. FARE has 251 complete triples because one score is invalid. Weak ordering means excellent ≥ borderline ≥ critical; strict means excellent > borderline > critical. The gap is an *ordinal-rating discrimination proxy*, not a calibrated accuracy measure. CIs are percentile cluster bootstraps with 2,000 replicates, seed 20261108. There is no legitimate claim of human-level accuracy or a probabilistic confidence level on model correctness.

**Paired differences:** On 251 matched triples, FARE minus GLM in mean excellent–critical gap is +0.219 (paired cluster bootstrap 95% interval −0.024 to +0.480). This interval crosses zero. FARE minus Compass is +0.928 (interval +0.700 to +1.179). Compass versus FARE weak-order differences are effectively zero: their *relative ordering* is similar despite different numerical scales.

**Important calibration warning:** FARE assigns high scores to some borderline answers (mean correctness 4.31; mean groundedness 4.39), while rating critical answers sharply lower. This *might* be appropriate under the rubric, or it could indicate leniency for subtle defects. Only targeted expert adjudication can settle that question. GLM's unusually high conversational-context scores on some critical responses also merit inspection. Review the high-disagreement rows before model selection.

## Targeted robustness stress set

This split contains 123 dimension evaluations per model and **39 quality triples**, across selected natural spoken queries, context-dependent turns, missing/partial/irrelevant retrieval evidence, and ASR-affected follow-ups. It does **not** isolate one perturbation at a time on matched original cases, so call it a *stress subset*, not evidence of invariance to controlled perturbations.

| Judge | Mean excellent−critical gap | Weak ordered triples | Strict ordered triples |
|---|---:|---:|---:|
| GLM | 2.333 | 89.74% | 41.03% |
| Compass | 2.231 | 97.44% | 33.33% |
| FARE | 3.282 | 94.87% | 46.15% |
| Prometheus | 2.410 | 89.74% | 25.64% |

All 123 robustness outputs are valid for each model after bounded retries. Compare these stress-set results descriptively: the selection and robustness question mixes differ, so differences between their aggregate means are not causal perturbation effects.

## Pairwise contrast and output reliability

| Judge | Valid / 28 | Bidirectional usable / 14 | Position-consistent among usable | Provisional preference matched among usable |
|---|---:|---:|---:|---:|
| GLM | 28 | 14 | 14 | 14 |
| Compass | 28 | 14 | 14 | 14 |
| FARE | 28 | 14 | 14 | 14 |
| Prometheus | 13 | 3 | 3 | 2 |

All 14 provisional preferred **source identities** are `a`. Each example was presented as both candidate 1 and candidate 2, allowing a position-swap consistency test, but because all sources have the same preferred identity, the set does not establish label-balanced generalization. Reference preferences were AI-generated, not independently human adjudicated. Prometheus's 3/3 consistency is conditioned on a heavily selected set of valid pairs and is not comparable to 14/14 coverage.

Overall 4,300 outputs include 17 invalid: FARE 2 (calibration and selection), Prometheus 15 (contrasts), GLM 0, Compass 0. The documented run-attempt medians/p95 measure service requests, not end-to-end tutor speech latency. Retries and incomplete pairwise judgments are included in the separate reliability metrics.

## Paired model disagreement

On selection ratings GLM and FARE have 791 jointly valid cases: exact agreement 61.82%, mean absolute score difference 0.536. GLM and Compass have 792 pairs with exact agreement 47.85% and mean absolute difference 0.658. This difference justifies item-level review, especially correctness, groundedness and conversational context. The `salient_disagreements.csv` table is a *triage queue*, not a human-rated error catalog; without independent evidence, scoring divergence cannot be assigned to a guilty model.

## Interpretation, bias, and next decision

1. **FARE:** largest category gap and robust stress separation, but two invalid scalar records and possible borderline over-scoring. Candidate for provisional scalar deployment after critical-case audit.
2. **CompassJudger:** zero invalid outputs, stable scalar order, and best observed A40 per-request latency; reasonable engineering fallback or preferred option if stricter reliability constraints dominate.
3. **GLM:** reliable outputs and strong separation, at substantially higher latency, with some suspiciously high context scores in critical cases.
4. **Prometheus:** retains scientific value as a scalar judge; unreliable for this 256-token pairwise output contract (15/28 invalid); no grounds to call its semantic competence intrinsically poor.

**Do not pick the overall judge from this report alone.** Task 3 must audit high-impact disagreements, apply a preregistered or explicitly post hoc selection rule with clear weight/priority choices, and label the winner provisional pending independent human assessment. The A30 gate and live tutor-interference experiments are independent tasks.

## Reproduction

From the repository root after copying the two Python files into `judge/` and placing the immutable ZIP locally:

```cmd
.\.venv\Scripts\python.exe -m unittest judge.test_analyze_judge_selection_v3 -v
.\.venv\Scripts\python.exe -m judge.analyze_judge_selection_v3 --input-zip day4_task2_frozen_inputs.zip --output-dir judge_outputs\day4_comparison
```

Requires NumPy and pandas. The analysis is fail-closed for missing/duplicate IDs, wrong per-split counts, mismatching shared input hashes, dataset/rubric hashes, task plans, and declared model configurations. The source ZIP is a local audit input; keep large raw predictions outside normal source commits unless repository policy explicitly allows archived research data.

### Output contract

`aligned_predictions.csv`, `run_metrics.csv`, `selection_triples.csv`, `selection_bootstrap.csv`, `selection_paired_bootstrap.csv`, `robustness_triples.csv`, `paired_agreement.csv`, `dimension_summary.csv`, `contrast_summary.csv`, `salient_disagreements.csv`, `analysis_manifest.json`.
