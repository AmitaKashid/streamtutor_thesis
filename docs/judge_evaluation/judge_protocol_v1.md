\# Judge Model Selection Protocol v1



Status: Frozen before benchmark execution  

Date: 2026-11-05  

Project: Real-Time Voice Tutor for Data Mining  

Protocol version: 1.0



\## 1. Objective



Select the most suitable locally deployable LLM-as-judge for evaluating

educational tutor answers while satisfying hardware, reliability, quality,

and latency constraints.



The selected model is the best deployable model among the tested candidates

under this protocol. The experiment does not claim to identify the universally

best judge model.



\## 2. Research Questions



RQ1. Which candidate loads and runs reliably on an NVIDIA A30 GPU?



RQ2. Which candidate agrees most closely with human educational-quality

ratings?



RQ3. Does a mixture-of-experts evaluator with fewer active parameters provide

a better speed-quality trade-off than dense judge models?



RQ4. Does overlap between the tutor-model family and judge-model family

systematically affect ratings?



RQ5. Does asynchronous judge execution measurably affect tutor response

latency or reliability?



\## 3. Candidate Models



\### Incumbent baseline



\- Model key: glm4\_32b\_0414\_q4

\- Model: GLM-4-32B-0414 Q4\_K\_M

\- Backend: llama.cpp

\- Current hardware: NVIDIA A100 80 GB

\- Purpose: existing general-purpose judge baseline



\### Specialist candidate A



\- Checkpoint: prometheus-eval/prometheus-7b-v2.0

\- Model family: Mistral

\- Size: 7B

\- Purpose: specialist evaluator trained for absolute and pairwise assessment

\- Target hardware: NVIDIA A30



\### Specialist candidate B



\- Checkpoint: opencompass/CompassJudger-2-7B-Instruct

\- Model family: Qwen2.5

\- Size class: 7B

\- Purpose: general-purpose specialist judge

\- Target hardware: NVIDIA A30



\### Low-active-parameter MoE candidate



\- Checkpoint: Salesforce/FARE-20B

\- Base model: gpt-oss-20b

\- Total parameters: approximately 20B

\- Active parameters: approximately 3.6B

\- Purpose: test the speed-quality benefit of sparse activation

\- Target hardware: NVIDIA A30

\- Admission limit: two hours for acquisition and compatibility testing



If FARE-20B cannot load, execute its documented prompt format, or produce a

parseable output on the available A30 environment within two hours, it is

recorded as a compatibility failure. It is not replaced after benchmark

results have been observed.



\## 4. Hardware and Runtime Conditions



Primary deployment target:



\- GPU: NVIDIA A30, 24 GB

\- One judge server per GPU allocation

\- Fixed model configuration per experiment

\- No concurrent workload on the judge GPU

\- Warm-up requests excluded from reported latency



Incumbent reference:



\- GPU: NVIDIA A100, 80 GB

\- Model: GLM-4-32B-0414 Q4\_K\_M



For every run, record:



\- GPU model

\- GPU memory capacity

\- peak allocated GPU memory

\- model checkpoint

\- model revision, when available

\- quantization

\- inference backend and version

\- prompt version

\- schema version

\- generation parameters

\- Git commit

\- SLURM job ID



\## 5. Evaluation Data



\### 5.1 Engineering smoke set



A fixed set of 20 cases will test integration and failure handling.



It must include:



\- fully correct answer

\- factually incorrect answer

\- partially correct answer

\- incomplete answer

\- unsupported claim

\- irrelevant retrieved context

\- missing retrieved context

\- empty answer

\- excessively long answer

\- unclear answer

\- unsuitable learner-level answer

\- answer containing prompt injection

\- answer containing JSON-like text

\- interrupted tutor response

\- duplicate callback attempt

\- unavailable judge server

\- timeout

\- malformed judge output

\- out-of-range score

\- missing required output field



\### 5.2 Model-selection set



The selection set contains 60 responses derived from 30 Data Mining questions.



Each question contributes:



1\. one normal tutor-generated response; and

2\. one incorrect, incomplete, unsupported, or otherwise degraded response.



The set must cover:



\- definitions

\- conceptual comparisons

\- algorithm explanations

\- calculation or reasoning questions

\- interpretation questions

\- misconceptions

\- questions requiring retrieved course context



Tutor-model families must be balanced where practically possible.



The cases and human labels are frozen before candidate results are inspected.



\## 6. Human Reference Ratings



One primary human rater scores all selection cases.



A second rater independently scores at least 20 cases without seeing:



\- the first rater's scores;

\- judge-model identity; or

\- judge-model output.



Inter-rater agreement is reported using quadratic weighted kappa.



Disagreements greater than one scale point are reviewed and adjudicated before

the final reference labels are created.



\## 7. Evaluation Dimensions



Tutor-answer quality is scored separately on:



1\. correctness;

2\. groundedness;

3\. completeness;

4\. clarity;

5\. pedagogy; and

6\. learner appropriateness.



Retrieval relevance is evaluated separately and is not included as an

answer-quality dimension.



Groundedness is marked not applicable when no retrieved evidence is supplied.

It must not be assigned an invented numerical score in that situation.



\## 8. Reliability Metrics



For each candidate, report:



\- first-attempt valid-JSON rate;

\- valid-JSON rate after one retry;

\- schema-valid rate;

\- request success rate;

\- timeout rate;

\- duplicate-result rate; and

\- missing-result rate.



A model receives no more than one retry following an invalid output.



The original invalid output and retry output are both retained.



Invalid responses must not be silently repaired, clamped, or completed with

default scores.



\## 9. Quality Metrics



Primary quality metric:



\- macro-average quadratic weighted kappa between judge and human scores across

&#x20; applicable rubric dimensions.



Secondary quality metrics:



\- per-dimension quadratic weighted kappa;

\- mean absolute error;

\- exact agreement rate;

\- agreement within one scale point;

\- critical-error recall;

\- critical-error precision;

\- verdict accuracy; and

\- bias-test score differences.



BLEU and BERTScore may be reported as answer-similarity diagnostics, but they

are not used to validate the judge or select the winning model.



\## 10. Performance Metrics



Measure after warm-up:



\- time to first judge token, where available;

\- total judge latency;

\- generated tokens;

\- output tokens per second;

\- peak GPU memory;

\- tutor latency with judging disabled;

\- tutor latency with asynchronous judging enabled; and

\- judge enqueue overhead.



Report median, p95, minimum, maximum, and failed-request count.



\## 11. Hard Admission Gates



A candidate advances to quality comparison only if it satisfies all applicable

conditions:



1\. loads on the target A30;

2\. completes all 20 smoke-test requests;

3\. produces valid JSON on at least 19 of 20 first attempts;

4\. produces valid JSON on all cases after at most one retry;

5\. generates no duplicate result for a turn;

6\. does not crash or block the tutor when unavailable;

7\. achieves median generation throughput of at least 10 output tokens/second;

8\. has p95 judge latency no greater than 25 seconds; and

9\. records all required provenance and performance fields.



Failure of a gate is recorded as an experimental result rather than hidden.



\## 12. Bias and Robustness Tests



The following tests are planned:



\- tutor-family overlap;

\- answer-length bias;

\- verbosity preference;

\- prompt-injection resistance;

\- irrelevant-context robustness;

\- missing-context handling;

\- unsupported-claim detection;

\- malformed-output handling; and

\- repeated-run consistency.



For repeated-run consistency, a fixed subset is judged three times using the

same deterministic generation configuration.



\## 13. Model-Selection Rule



Selection is lexicographic rather than an arbitrary weighted sum.



1\. Exclude candidates that fail a hard admission gate.

2\. Select the surviving candidate with the highest macro quadratic weighted

&#x20;  kappa against human ratings.

3\. If the kappa difference is at most 0.05 or uncertainty intervals overlap,

&#x20;  prefer the candidate with higher critical-error recall.

4\. If quality remains practically tied, prefer lower p95 latency.

5\. If p95 latency differs by less than 10%, prefer lower peak GPU memory.

6\. If no candidate passes all gates, retain the incumbent baseline and report

&#x20;  that no tested A30 candidate met the deployment requirements.



This rule must not be changed after candidate results are inspected.



\## 14. Tutor-Latency Non-Interference Rule



Judge execution must be asynchronous and isolated from the user-facing answer

path.



The judge passes the non-interference test when:



\- judge enqueue overhead has p95 below 50 milliseconds;

\- tutor operation continues when the judge server is unavailable; and

\- enabling the judge increases median tutor response latency by no more than

&#x20; 5% relative to the judge-disabled condition.



\## 15. Known Limitations



\- The selection set is limited to the Data Mining tutoring domain.

\- Human ratings may contain disagreement and subjective variation.

\- Quantized and full-precision versions are not equivalent configurations.

\- A model failing on A30 may still be viable on larger hardware.

\- Results may depend on inference backend and prompt formatting.

\- Thirty questions cannot represent every educational interaction.

\- The four-day schedule permits a controlled pilot, not a definitive

&#x20; large-scale judge benchmark.



\## 16. Reproducibility Rule



Any post-freeze change to the dataset, rubric, schema, prompt, candidate list,

generation configuration, metric, threshold, or selection rule requires a new

protocol version and a written justification.

