# Human annotation manual

## Purpose and status

This manual governs future human annotation of the frozen StreamTutor judge-selection set. The labels already shipped in `responses.jsonl` are AI-generated provisional labels; annotators must not see them.

## What a rater receives

For each item, show the learner level, requested teaching style, necessary conversation history, current user utterance, retrieved context, reference answer, and one anonymized candidate response. Hide the candidate model, controlled-quality label, provisional scores, and all other raters' decisions.

## Two-stage rating workflow

1. **Retrieval screen:** view the question, history, and retrieved context without the candidate response. Rate retrieval relevance from 1 to 5. Mark retrieval intentionally disabled only when the item explicitly says so.
2. **Response screen:** reveal the candidate response. Rate correctness, groundedness, completeness, clarity, pedagogy, learner appropriateness, and conversational context independently. Then mark critical errors, confidence, and notes.

This separation reduces the chance that a fluent answer inflates the retrieval score or that bad retrieval is incorrectly blamed on the tutor.

## Dimension rules

- **Correctness:** compare claims, reasoning, formulas, and calculations with the reference and course source. Do not reward style here.
- **Groundedness:** judge support from the supplied usable evidence. Use N/A when evidence is missing or unusable; do not assign a low groundedness score merely because retrieval failed.
- **Completeness:** ask whether this turn's essential requirements are covered. Do not reward unrelated detail.
- **Clarity:** judge coherence, concision, ambiguity, and spoken comprehensibility while holding factual quality separate.
- **Pedagogy:** judge learning support and the requested style, not merely whether the final fact is present.
- **Learner appropriateness:** judge terminology, depth, pace, and assumed prerequisites for the stated learner.
- **Conversational context:** judge resolution of history, pronouns, ellipsis, correction, and slide deixis. For standalone turns, score whether the response creates no unnecessary context failure.
- **Retrieval relevance:** this is a pipeline score, not an answer-quality score.

## Scale discipline

- 1 = critical failure;
- 2 = major weaknesses requiring substantial correction;
- 3 = mixed but usable, with a meaningful error or omission;
- 4 = good, with only a minor limitation;
- 5 = no meaningful defect on that dimension.

Use the 35 cases in `rubric_anchor_examples.jsonl` during training. They intentionally cover every score for every answer-quality dimension. They are not selection cases and must never be included in agreement or model-ranking calculations.

## Critical errors and evidence

Select every applicable critical-error code from `rubric_v3.json`. A score of 1, 2, or 5 requires a brief evidence-based note. Quote only the shortest phrase needed. Do not infer hidden intentions or repair the answer mentally.

## Independence, order, and adjudication

Rate independently and in randomized order. Keep all turns from a conversation within one annotation batch but randomize response variants. Two raters should score the frozen selection set. Calculate raw agreement and quadratic-weighted kappa per dimension with conversation-level bootstrap intervals. Flag disagreements greater than one point for adjudication, preserve both original ratings, and record a separate adjudicated label.

## Conflict rule

If the reference answer and supplied slide evidence appear inconsistent, do not guess. Mark the item for source review and exclude it from confirmatory analysis until resolved and versioned.
