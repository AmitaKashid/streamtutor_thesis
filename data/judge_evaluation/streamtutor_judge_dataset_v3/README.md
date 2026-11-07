# StreamTutor LLM-as-Judge Evaluation Dataset v3.0.0

This archive is a course-grounded meta-evaluation dataset for selecting and validating an LLM judge for StreamTutor, a conversational voice tutor for Data Mining.

## Contents

- `questions.jsonl`: 48 natural multi-turn question turns in 16 three-turn conversations.
- `responses.jsonl`: 144 answer-evaluation cases (excellent, borderline, and critical variants for every turn).
- `controlled_contrasts.jsonl`: 14 minimally controlled answer pairs that isolate one rubric dimension at a time.
- `rubric_anchor_examples.jsonl`: 35 balanced 1--5 examples for rater training (seven dimensions by five scores), excluded from selection metrics.
- `rubric_v3.json`: dimension-specific five-point scoring anchors and critical-error policy.
- `*_schema_v3.json`: JSON Schemas for questions, scalar-response cases, and controlled contrasts.
- `human_annotation_template.csv`: blank sheet for future independent human annotation.
- `annotation_manual.md`: blinded two-stage rating instructions and adjudication rules.
- `dataset_statistics.csv`: auditable coverage counts.
- `validation_report.json`: structural, split, label-range, and source-integrity checks.
- `source_manifest.json`: slide-deck filenames and SHA-256 hashes; the PDFs are intentionally not redistributed.
- `methodology_for_thesis.md`: research basis, construction protocol, statistical rationale, and thesis-ready limitations.
- `references.bib`: research references used in the design.

## Experimental units

The unit scored by a judge is one candidate response, not one question. The 48 question turns produce 144 scored response cases. Whole conversations are assigned to exactly one split to prevent history leakage:

- calibration: 18 response cases;
- frozen selection: 108 response cases;
- robustness: 18 response cases.

The answer variants are controlled counterfactuals. They hold the question, history, source evidence, learner level, and requested teaching style fixed while varying response quality. This supports discrimination testing and reduces confounding.

The 14 additional contrast pairs are diagnostic rather than part of the ordinal kappa calculation. Each pair isolates one of seven rubric dimensions. Run each pair in both answer orders; a preference is counted only when it is invariant to the order reversal.

The 35 anchor examples cover every 1--5 score for every answer-quality dimension. They are for calibration and rater training only. They must not be added to the frozen selection set or used to improve a candidate after selection results are inspected.

## Annotation status

The included labels are **provisional AI-generated expert annotations**. They are not independent human ratings and must not be reported as human ground truth or human-judge agreement. They allow engineering, pilot analysis, rubric debugging, and rater training while human raters are unavailable.

For confirmatory judge selection, obtain at least two independent human ratings for the 108 frozen selection cases if feasible. If resources are constrained, double-rate a preregistered stratified subset of at least 60 selection cases, report percentage agreement and quadratic-weighted kappa with bootstrap confidence intervals, and clearly label the remaining cases as singly rated or AI-provisional.

## Required use discipline

1. Tune prompts only on the calibration split.
2. Do not inspect candidate-model scores on the selection split before freezing the prompt and decision rule.
3. Report every invalid output, retry, timeout, and missing result.
4. Compare models per rubric dimension; do not invent a weighted overall score after observing results.
5. Keep retrieval relevance separate from tutor-answer quality.
6. Preserve raw judge output and all provenance.
