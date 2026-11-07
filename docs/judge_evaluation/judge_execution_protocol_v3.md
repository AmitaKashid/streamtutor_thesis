# StreamTutor Judge Execution Protocol v3

**Version:** 3.0

**Freeze date:** 2026-11-07

**Status:** Frozen before any benchmark-v3 calibration, selection, robustness,
or controlled-contrast output is inspected

## 1. Purpose and scope

This document freezes the execution conditions for the StreamTutor judge
benchmark v3. It does not alter or erase protocol-v1 compatibility results or
protocol-v2 scalar-interface development evidence.

Two earlier correct/incorrect scalar integration probes were used only to
verify transport and parsing. They are excluded from calibration, model
selection, robustness analysis and all reported agreement estimates.

Protocol v3 separates three questions that must not be collapsed:

1. **Compatibility:** can the artifact load and communicate through the
   required application interface?
2. **Judge quality:** does the model distinguish educational answer quality
   under the frozen scalar rubric?
3. **Deployment performance:** does the same model configuration satisfy the
   latency, reliability and memory gates on the target hardware?

## 2. Reconciliation with earlier protocols

Protocol v1.1 froze a compatibility configuration of 4096 context tokens,
360 output tokens, temperature 0.2 and top-p 0.9 on an NVIDIA A30. The existing
Day-2 screening records include exploratory A40 runs and non-identical
generation settings. Those records remain valid compatibility evidence, but
they are not treated as a controlled comparative latency benchmark and are
not used to rank candidates by speed.

Benchmark v3 uses the scalar interface because protocol v1 demonstrated that
large multi-dimensional JSON generation confounds educational judgement with
format-following ability. The v3 configuration below is frozen independently
for the v3 research question. Protocol-v1 decisions remain reported as
separate deployment evidence.

## 3. Frozen candidate set

The candidate set is unchanged:

- `glm4_32b_0414_q4`
- `fare_20b_q4`
- `compassjudger_2_7b_q4`
- `prometheus_7b_v2_q4`

Every model uses its committed Q4_K_M GGUF artifact and the artifact revision
and SHA-256 recorded in the acquisition evidence. No artifact substitution is
permitted within a run series.

## 4. Frozen benchmark-v3 inference configuration

All quality, controlled-contrast and comparative performance requests use:

- backend: `llama-cpp-python 0.3.20`;
- context length: 4096 tokens;
- maximum generated tokens: 256;
- temperature: 0.0;
- top-p: 1.0;
- seed: 0;
- batch size: one request;
- CPU threads: 12;
- GPU layers: -1;
- llama.cpp batch size: 512;
- llama.cpp micro-batch size: 512;
- flash attention: enabled;
- K/Q/V offload: enabled;
- registry system prompt: disabled;
- model-native chat template: retained;
- at most one format-only retry;
- no silent output repair, score clamping or default completion.

All semantic instructions and rubric anchors are supplied in the user message.
Disabling registry system prompts prevents candidate-specific hidden semantic
guidance. A model's native chat template is retained because it is part of its
documented inference interface.

## 5. Hardware separation

### 5.1 Quality comparison

Quality calibration, selection, robustness and controlled contrasts may run
on one NVIDIA A40 48 GB per model server. All candidates in a reported
comparison must use the same GPU class and have no concurrent workload on the
judge GPU. Hardware is recorded but quality scores are not interpreted as
A30 deployment evidence.

### 5.2 Deployment gate

The deployment target remains one NVIDIA A30 24 GB. A model cannot be declared
A30-deployable until its frozen v3 configuration is executed on an A30.
Exploratory A40 success is not a substitute for this gate.

The deployment run must demonstrate:

- complete GPU-layer offload;
- no unintended CPU tensor offload;
- at least 2048 MiB minimum free GPU memory;
- no unrecovered crash or hang;
- median decode throughput of at least 10 output tokens/second;
- p95 completion latency no greater than 25 seconds;
- all measured evaluations valid after at most one retry.

Queue waiting time is recorded separately and excluded from model latency.

## 6. Quality execution stages

### 6.1 Calibration

Run the complete frozen calibration split for every candidate. Calibration is
used to verify the interface, score coverage and analysis pipeline. It must
not be used to modify candidate-specific prompts, rubric wording or labels.
Only global implementation defects may be corrected, and any correction
requires a new protocol version and rerunning every affected candidate.

### 6.2 Model selection

After every candidate completes calibration under an identical configuration,
run the complete selection split. Truncating the selection split is forbidden.
Model identifiers remain blinded during human annotation where human ratings
are available.

Provisional AI-expert annotations may support engineering analysis, but they
must never be described as independent human ground truth. Confirmatory thesis
claims require human ratings; otherwise agreement results are explicitly
reported as provisional.

### 6.3 Robustness

Run the complete robustness split without changing inference settings. Report
prompt-injection, missing-context, irrelevant-context and conversational-
context results separately rather than combining them into one opaque score.

### 6.4 Controlled contrasts

Run every frozen contrast twice: once as A/B and once as B/A. A contrast is
usable only when both outputs are valid and the selected underlying answer is
identical after position reversal. Report position consistency before
provisional expected-preference accuracy.

## 7. Comparative performance benchmark

The latency benchmark uses 30 deterministic scalar evaluation tasks drawn
from the calibration split:

- four tasks from each of the seven answer-quality dimensions; and
- two retrieval-relevance tasks.

Within each dimension, tasks are selected by sorted frozen evaluation ID. The
30-task list and its SHA-256 are written before model requests begin.

For each candidate:

1. execute five unmeasured warm-up requests;
2. execute the same 30 tasks three times;
3. retain the same task and repetition order;
4. record every attempt, including invalid outputs and retries;
5. write request-level results before computing aggregates.

Report attempt-level and evaluation-level latency separately. Report time to
first token only when the backend measures it directly; never infer it from
total completion time. Report median, p95, minimum and maximum completion
latency, median decode throughput, first-attempt validity, validity after one
retry, timeout count and failed-request count.

## 8. Tutor non-interference experiment

Use an identical frozen tutor request set under three conditions:

1. judge disabled;
2. judge active on the tutor GPU;
3. judge isolated on a separate GPU/server.

Run one warm-up followed by at least 30 measured tutor turns per condition in
the same order. Record tutor TTFT, first-audio latency, total tutor generation
time, total turn latency, TTS failures, judge enqueue overhead and peak GPU
memory.

The judge passes non-interference when:

- judge enqueue-overhead p95 is below 50 milliseconds;
- the tutor remains operational when the judge server is unavailable; and
- median tutor response latency increases by no more than 5% relative to the
  disabled condition.

Same-GPU measurements characterize interference; the isolated condition is
the intended deployment topology when interference is material.

## 9. Provenance and resumability

Every run records:

- dataset and rubric checksums;
- exact task-plan checksum;
- model identifier, artifact, quantization and response protocol;
- inference backend and generation configuration;
- hardware and GPU-memory samples;
- Git commit and SLURM job ID;
- timestamps excluding queue delay;
- raw and decoded model output;
- validation errors, retries and generation errors.

Resume is permitted only when the immutable manifest matches exactly.
Completed evaluation IDs are never generated twice. Partial runs are retained
and labelled incomplete rather than silently discarded.

## 10. Decision boundary

Compatibility, quality, performance and non-interference remain separately
reported. No weighted composite score is introduced. Final selection follows
the pre-registered lexicographic rule and clearly distinguishes provisional
AI-reference analysis from human-reference validation.
