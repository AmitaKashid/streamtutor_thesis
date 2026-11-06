# Judge Quality Evaluation Protocol v2

**Version:** 2.0  
**Freeze date:** 2026-11-06  
**Project:** StreamTutor  
**Status:** Pre-registered before protocol-v2 model outputs are inspected

## 1. Purpose

Protocol v1 measured zero-shot compatibility with StreamTutor's complete
multi-dimensional JSON contract. It remains the authoritative deployment-
compatibility experiment.

Protocol v2 measures a different property: agreement between judge models and
human educational-quality ratings.

The two protocols must not be combined into a single metric:

- Protocol v1: integration and structured-output compatibility
- Protocol v2: underlying judge quality
- Final selection: quality first, followed by deployability and performance

All protocol-v1 results and raw outputs remain unchanged.

## 2. Research Questions

RQ1. Which judge agrees most closely with human ratings of Data Mining tutor
answers?

RQ2. Which judge detects critical educational errors most reliably?

RQ3. Does the low-active-parameter MoE provide a better quality-speed trade-off
than dense or specialist judge models?

RQ4. Do judge scores change because of answer length, tutor-model family,
irrelevant context or prompt injection?

RQ5. Can the highest-quality judge be deployed without materially increasing
StreamTutor's interactive latency?

## 3. Candidate Configurations

The frozen candidate set is:

1. `glm4_32b_0414_q4`
   - GLM-4-32B-0414
   - Q4_K_M
   - dense general-purpose incumbent

2. `fare_20b_q4`
   - Salesforce FARE-20B
   - Q4_K_M
   - sparse MoE
   - approximately 3.6B active parameters

3. `compassjudger_2_7b_q4`
   - CompassJudger-2-7B-Instruct
   - Q4_K_M
   - Qwen-based specialist judge

4. `prometheus_7b_v2_q4`
   - Prometheus-7B-v2.0
   - Q4_K_M
   - Mistral-based specialist evaluator

No candidate may be added or removed after protocol-v2 results are inspected.

## 4. Separation from Protocol v1

A protocol-v1 format failure does not automatically exclude a model from the
protocol-v2 quality comparison.

Protocol v2 uses a smaller scalar output interface to avoid conflating
educational judgement with the ability to generate one large JSON object.

Protocol-v1 failures remain deployment evidence and must be considered during
final model selection.

## 5. Evaluation Data

### 5.1 Supervisor-demonstration pilot

The pilot contains 20 previously unseen tutor responses derived from 10 Data
Mining questions.

Each question contributes:

1. one substantially correct response; and
2. one incorrect, incomplete, unsupported or pedagogically weak response.

The pilot must not reuse the protocol-v1 smoke cases verbatim.

The pilot covers:

- definitions;
- conceptual comparisons;
- algorithm explanations;
- numerical reasoning;
- interpretation;
- misconceptions;
- course-context grounding;
- learner-level adaptation;
- pedagogical explanation.

Pilot results are preliminary and are not presented as final thesis validation.

### 5.2 Final thesis selection set

The final selection set contains 60 responses derived from 30 Data Mining
questions, with the same correct/degraded pairing design.

The final set is frozen before final model outputs are inspected.

## 6. Human Reference Ratings

The primary human rater scores every response independently before seeing model
scores.

A second human rater scores:

- at least 10 of 20 pilot responses; and
- at least 20 of 60 final responses.

Human ratings use the same written 1-5 anchors as the judge models.

Quadratic weighted kappa is reported between human raters.

Disagreements greater than one scale point are reviewed and adjudicated before
creating the final reference label.

If only one human rater is available for the pilot, the results are explicitly
labelled preliminary.

## 7. Evaluation Dimensions

Each answer is evaluated independently on:

1. correctness;
2. groundedness;
3. completeness;
4. clarity;
5. pedagogy;
6. learner appropriateness.

Retrieval relevance is evaluated separately.

Groundedness is not evaluated when no context is supplied.

Retrieval relevance is not evaluated when no retrieved evidence is supplied.

## 8. Scalar Evaluation Task

Each model evaluates exactly one dimension per request.

Every request contains:

- student question;
- tutor answer;
- retrieved context, when applicable;
- reference answer or reference facts;
- learner level;
- teaching style;
- one rubric dimension;
- written score anchors from 1 to 5.

All semantic instructions are provided in the user message.

Model-specific chat templates may be used because they are part of the model's
documented inference interface.

No candidate receives model-specific rubric wording, additional examples or
hidden guidance.

## 9. Output Contract

The required output is:

```text
Feedback: <concise evidence-based explanation>
[RESULT] <integer from 1 to 5>
```

The text above is the canonical model-facing serialization request.

### 9.1 Accepted serialization grammar (protocol 2.1)

For parsing, the literal `Feedback:` prefix is optional because it carries no
semantic information. A response is valid only when all of these conditions
hold:

1. it contains non-empty feedback;
2. it contains exactly one `[RESULT]` marker;
3. the marker is followed by one integer from 1 through 5;
4. the score is the final response content;
5. it contains no Markdown fence; and
6. no score is inferred, clamped, repaired or supplied by the evaluator.

The raw output is retained unchanged. Parsed records identify whether the
model used `prefixed_feedback` or `unprefixed_feedback` serialization.

## 10. Pre-evaluation Amendment Log

### Amendment 2.1 — 2026-11-06

An integration-only Prometheus probe, performed before comparative pilot
evaluation and before human-agreement results were inspected, returned
non-empty feedback and an explicit terminal `[RESULT] 3` marker but omitted
the semantically redundant `Feedback:` prefix. Protocol 2.1 therefore accepts
both prefixed and unprefixed feedback while retaining every substantive
constraint on score extraction.

The canonical prompt remains version 2.0 and continues to request the prefix.
Historical protocol-2.0 records are not reclassified or overwritten. The
amendment applies only to newly generated protocol-2.1 records and is applied
identically to every candidate model.
