# Task 8 — Independent Human Validation Protocol for StreamTutor's LLM-as-Judge

**Protocol ID:** ST-JUDGE-HV-2026-11-08-v1
**Status:** Proposed, not executed; no independent human ratings collected
**Engineering selection under validation:** FARE 20B Q4_K_M (`fare_20b_q4`), with CompassJudger 2 7B Q4_K_M (`compassjudger_2_7b_q4`) as an engineering fallback
**Measurement target:** Agreement of each judge's *dimension-specific educational-quality judgments* with blinded course-competent human assessments. This is not a study of educational learning gains.

## 1. Scope and separation of evidence

The completed four-model comparison used 4,300 recorded outputs across calibration, selection, robustness and pairwise-contrast conditions. The provisional selection rule was specified **after** inspecting those results. The Task 7 88-turn ON/OFF pilot demonstrated an asynchronous runtime property and produced 308 valid scalar evaluations; output validity is not semantic accuracy. Neither evidence stream contains independent human reference ratings. This protocol addresses that missing validity gate.

Use two explicitly separated arms:

- **Arm A — retrospective audit (diagnostic, not confirmatory):** blinded humans rate a stratified sample of the *existing frozen v3 cases*, including high judge-disagreement records, ambiguous follow-ups, borderline answers and apparent critical mistakes. Its findings can diagnose weaknesses and support thesis discussion, but cannot provide a fresh unbiased validation of the post hoc judge selection.
- **Arm B — untouched confirmatory holdout (required before claiming independent comparative validation):** author and freeze new case/conversation IDs and original tutor-generated answers after locking the present protocol and selected model configurations. Neither candidate-selection analysis nor human adjudication of Arm A may be used to tune the judges, cutoffs or rubric before the Arm B evaluation. Keep source/answer provenance, source-page checks and a blinded selection process in a versioned manifest. If Arm B is infeasible, report **Arm A only** as diagnostic evidence and retain provisional model-selection status.

The existing v3 dataset (`data/judge_evaluation/streamtutor_judge_dataset_v3/`) contains 16 conversations × 3 turns = 48 question turns and 144 synthetic response variants, plus 14 contrast pairs. It also contains provisional AI-generated quality labels. Those labels, judge scores, model names, earlier disagreement rankings and generated variant names (`excellent`, `borderline`, `critical`) must **never appear in a human rating interface**. A random opaque item ID should map back to these IDs only in a locked key.

## 2. Sampling and feasibility

### Arm A — diagnostic audit

Proposed target: **60 response items**, independently rated by two course-competent assessors, selected before showing assessor ratings. Allocation: 20 high-scoring model-disagreement cases, 15 critical-error candidates, 15 borderline/completeness cases and 10 context-/retrieval-dependent cases. These are mutually exclusive selection strata assigned by priority; document duplicate eligibility and the final allocation. Sample entire associated context when needed. Balance course topics, answer-length buckets and synthetic variant identities where feasible; explicitly report any shortfall. Treat the 60-item target as an operational recommendation, **not** an achieved sample or statistical power guarantee.

### Arm B — independent confirmatory holdout

Proposed minimum planning target: **at least 60 new response items** spanning **at least 20 new conversation clusters**, with at least two independent raters per item. Include naturally generated tutor answers (not only intentionally degraded examples), spoken/ASR-like utterances, history-dependent follow-ups, relevant/irrelevant/missing retrieval, formula or calculation answers, and unsupported claims. Include failures and abstentions; do not select only fluent delivered answers. Avoid conversation, answer and source-page near-duplicates with the v3 selection set, and publish the deduplication rules in advance. Oversample important rare failures only with explicit strata; performance estimates are conditional on this constructed evaluation distribution.

A meaningful precision/power target **cannot be asserted from 60 items**. Before execution, simulate or analytically estimate uncertainty using the expected dimension score prevalence, judge–human correlation and conversation clustering. If intervals remain too wide to separate models, increase clusters or report the result as inconclusive. If only one expert is available, report single-rater exploratory outcomes; do not compute inter-rater reliability or claim consensus.

## 3. Raters, training and conflict-of-interest controls

Recruit **at least two** individuals competent in the taught Data Mining material, ideally a course instructor or qualified teaching/research staff member and an independent second rater. Record expertise and relationship to the system, not unnecessary identifying information. Raters should not author or grade the candidate judge outputs and should not consult model predictions. A third adjudicator is preferable; if only two raters are available, prespecify a consensus meeting after independent ratings and retain both originals.

Training uses `rubric_v3.json`, `annotation_manual.md` and the 35 `rubric_anchor_examples.jsonl` anchors, with a short practice batch **outside both scored arms**. Clarify score anchors, evidence applicability, critical-error codes, conversation references and the distinction between retrieval quality and answer quality. Keep practice feedback out of validation results. Freeze the rater instructions and rubric before scored annotation; document any later amendment and do not silently relabel prior outcomes.

## 4. Blinded two-stage assessment

**Stage 1 — retrieval (answer hidden).** Show the learner utterance, necessary turn history, expected retrieval mode, and exactly the retrieved slide excerpts/citations exposed to the tutor at response generation. Rate `retrieval_relevance` from 1–5. Use `N/A` **only when retrieval was intentionally disabled**. Missing retrieval when expected scores **1**, following `rubric_v3.json`. A retrieved snippet should be rated for sufficiency and relevance, not for the quality of the eventual tutor answer.

**Stage 2 — response.** Reveal one anonymized tutor response, learner level, requested teaching style, history, retrieval evidence and the *course-grounded* reference answer or source material. Rate **correctness, groundedness, completeness, clarity, pedagogy, learner appropriateness, and conversational context** separately on the frozen ordinal scale 1–5. Groundedness is `N/A` when **no usable evidence** was supplied, even if retrieval was expected; do not equate absence of evidence with an unsupported factual answer. Record critical-error codes, rater confidence and a short evidence-based note for scores 1, 2 and 5. Permit `SOURCE_CONFLICT`/`UNRATEABLE` flags with documented reasons; do not force uncertain evidence into a numerical label.

Use the rubric's anchors: **1** critical failure, **2** major weaknesses, **3** mixed/adequate with meaningful issues, **4** good with minor limitations, **5** excellent on that dimension. Do not calculate a general-purpose weighted average score. For context dependence, display precisely the prior turns actually available to the tutor; do not infer context that was never supplied. For spoken clarity, assess how comprehensible the *available text* would be when spoken; no claim about acoustic intelligibility is possible without audio.

Randomize anonymized item order within logistical constraints, counterbalance response-variant order, and keep necessary conversation histories together so references remain interpretable. Raters must work independently until the individual rating files are locked. Store only pseudonymous identifiers; follow the university's applicable approval, consent and data retention requirements before collecting real student interactions.

## 5. Adjudication and source review

Resolve `SOURCE_CONFLICT` against the frozen course PDFs before a case can be used as a ground-truth comparison; log source document/page and the resolution. For numerical disagreements greater than one point, disagreements on critical-error presence, and disputed applicability, request adjudication. Retain `(rater_1, rater_2, adjudicated)` as distinct columns: **never overwrite raw ratings**. Do not let an adjudicator see candidate judge outputs. Predefine whether the reference is the adjudicated score or a consensus score and use the same rule for all models.

If no adjudication is possible, report both judge–rater comparisons separately and sensitivity to which rater is treated as reference; do not manufacture a consensus label.

## 6. Statistical analysis plan — lock before Arm B

**Primary endpoint (proposed):** macro-average of **per-dimension quadratic-weighted Cohen's kappa** between the candidate judge and independently adjudicated human scores, across the seven answer dimensions with sufficient class variation and applicable observations. Equal weight per eligible dimension; explicitly report excluded/undefined dimensions. When kappa is undefined because human scores have no variation, do not impute 0 or 1. A macro-average across a very small number of eligible dimensions is not a robust single-number selection rule: expose its coverage.

**Secondary endpoints:** per-dimension weighted kappa, mean absolute error on 1–5 scores, exact agreement, within-one-point agreement, critical-error detection precision/recall (with definition of an actionable error), judge-output validity/timeout/retry rate, and agreement on the `revise` decision defined by the frozen rubric (`critical error`, `correctness <= 2`, applicable `groundedness <= 2`, or context score `<= 2` for context-dependent turns). Report confusion matrices and false acceptances separately, with special emphasis on harmful-to-learning errors. **Retrieval relevance** is a separate retrieval-pipeline endpoint; do not merge it into the seven-dimension answer-quality macro score.

**Human–human reliability:** raw agreement and per-dimension quadratic-weighted kappa between independent raters *before adjudication*, accompanied by score distributions and uncertainty. This is essential context for interpreting judge–human results. Calculate judge–human metrics for **both FARE and Compass on exactly the same eligible items**; add GLM/Prometheus only if fixed outputs/configurations are available, and label comparisons exploratory if not prespecified.

**Uncertainty and dependence:** 10,000 paired bootstrap resamples of whole **conversation clusters**, stratified by prespecified sampling stratum where feasible, deterministic registered seed. Sample each conversation once per replicate with all of its turn/response variants and all model ratings; produce percentile 95% intervals for each estimable metric and for paired FARE–Compass differences. If a replicate produces undefined kappa, report its frequency and revisit the estimator or sample size rather than dropping it invisibly. Do not treat 144 variants or seven rubric dimensions as independent questions.

**Multiplicity and interpretation:** the primary endpoint must be fixed in the Arm B manifest before seeing human labels; report other dimensions descriptively and avoid interpreting isolated unadjusted p-values as conclusive. A zero-crossing paired CI is inconclusive rather than proof of equivalence. An observed advantage does not establish general validity beyond the course, sampled scenarios, evaluator configuration or rating rubric.

## 7. Promotion and reversal criteria (proposed, not yet registered)

Before starting Arm B, approve and freeze numerical tolerances after feasibility/power review. The following are *decision requirements*, not claimed achieved results:

1. **Critical-error safety:** document false-acceptance counts and confidence intervals. No automatic promotion based on mean kappa if one model systematically approves harmful misconceptions or fabricated evidence.
2. **Comparative quality:** prefer FARE only if the paired human-alignment estimates and intervals provide a practically meaningful advantage or acceptable noninferiority relative to Compass on the prespecified endpoints. An illustrative 0.05 weighted-kappa difference appears in the existing thesis methodology, but it is **not** a statistically justified margin until explicitly approved and powered.
3. **Operational feasibility:** retain scalar-output validity, bounded latency, deployment memory fit, and nonblocking tutor behavior as *separate* engineering gates. A model that wins human alignment but cannot run reliably within available hardware is not automatically deployable.
4. **Inconclusive outcomes:** retain **provisional engineering selection** if human agreement is weak or model comparison intervals are wide. Consider a larger untouched holdout, rubric revision followed by a new freeze, or the faster Compass fallback rather than rewriting the original v3 selection evidence.

Any model switch requires a new versioned decision record and provenance of model weights, quantization, prompt, parser, rubric, hardware, and evaluation date. Do not modify the frozen original selection manifests.

## 8. Data products and reproducibility contract

Version before annotation: `human_validation_protocol.md`, source/case manifest with item hashes, dataset split and near-duplicate exclusions, randomization seed, rater instructions, blank two-stage annotation forms, and a locked model-output configuration list. Preserve the frozen course-source manifest and PDF page references separately.

After annotation: pseudonymized raw rating files, rater training completion log, applicability/source-conflict ledger, adjudication decisions, scorer/metric implementation and tests, paired model predictions, bootstrap seed and summary tables. Hash the immutable inputs and outputs; provide a rerun command. Restrict raw student information and human annotator identities; publish only non-sensitive aggregates when appropriate.

The existing `human_annotation_template.csv` is useful for Arm A but is **not sufficient without modification** for a confirmatory, blinded two-stage study: its `response_id` exposes synthetic quality suffixes, and it combines retrieval and response ratings on one sheet. Create opaque randomized IDs, distinct retrieval/response forms, timestamps/version fields, and a protected ID mapping before using it with actual raters.

## 9. Acceptance checklist and honest reporting

This *protocol document* is complete when its design, source version, sample plan, forms, rater workflow, statistics and decision conditions have been reviewed and committed. **The human-validation experiment is not complete** until the independent rating and analysis records exist. Distinguish these clearly in any thesis abstract or presentation.

**Approved reporting language before collecting human labels:** "FARE 20B Q4_K_M was selected provisionally using a frozen synthetic judge-comparison dataset and demonstrated asynchronous runtime feasibility in a controlled speech-output pilot. Independent expert calibration and comparative validity remain to be established."

**Disallowed before validation:** "FARE agrees with human experts," "FARE is the most accurate educational judge," "our 308 valid ratings prove the tutor answers were correct," or "the 88-turn pilot proves no end-user latency cost under all hardware configurations."

## Source documents (repository-relative)

- `data/judge_evaluation/streamtutor_judge_dataset_v3/rubric_v3.json`
- `data/judge_evaluation/streamtutor_judge_dataset_v3/annotation_manual.md`
- `data/judge_evaluation/streamtutor_judge_dataset_v3/human_annotation_template.csv`
- `data/judge_evaluation/streamtutor_judge_dataset_v3/methodology_for_thesis.md`
- `docs/judge_evaluation/day4_comparison_v3.md`
- `docs/judge_evaluation/day4_judge_selection_decision_v3.md`
- `docs/task7/task7_results_report.md`
