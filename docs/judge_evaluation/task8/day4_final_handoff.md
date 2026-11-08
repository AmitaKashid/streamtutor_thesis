# StreamTutor — Day 4 Judge-Selection Sprint: Final Engineering and Research Handoff

**Handoff ID:** ST-JUDGE-D4-2026-11-08
**Status:** Research-pilot engineering handoff; **not** independent judge-validity or production certification
**Branch:** `judge-selection-sprint`
**Verified Task 8 reproducibility commit:** `12baca5` (source state at that checkpoint; later document commits must be identified separately)
**Scope:** Selection of the asynchronous tutor-answer evaluator, frozen candidate comparison, runtime evidence, operational constraints and follow-on scientific validation.

## 1. Executive decision

**Provisional primary evaluator:** FARE 20B, GGUF Q4_K_M, runtime ID `fare_20b_q4`.
**Documented fallback candidate:** CompassJudger 2 7B, GGUF Q4_K_M, runtime ID `compassjudger_2_7b_q4`.
**Alternatives retained as comparators:** GLM 32B and Prometheus 7B v2.

This is an **engineering selection**, not a claim that FARE agrees most closely with human assessors. The candidate-screening rule was **post hoc**, the benchmark quality categories are synthetic/provisional, and independent human judgments have not yet been obtained. FARE is to remain an asynchronous observer of completed delivered tutor answers, **never a dependency of the learner-facing STT → retrieval → LLM → TTS playback path**. Model substitution must be versioned and requalified; fallback is not implicit.

**Authoritative decision document:** [`judge_selection_decision_final.md`](judge_selection_decision_final.md). Do not reinterpret the model-selection proxy scores as accuracy against human ground truth.

## 2. Evidence register and claim traceability

| Claim / decision | Authoritative evidence | Finding and limit |
|---|---|---|
| Candidate comparison completed | [`../day4_comparison_v3.md`](../day4_comparison_v3.md), [`../day4_judge_selection_decision_v3.md`](../day4_judge_selection_decision_v3.md) | Four judges, 4,300 records. Thirty-six underlying question clusters in the selection split; 17 invalid outputs retained in reliability denominators. |
| Provisional FARE choice | [`judge_selection_decision_final.md`](judge_selection_decision_final.md) | 1,073/1,075 valid overall; 0/72 high (≥4) scores on synthetically critical correctness/groundedness cases; greater excellent–critical separation than Compass. Post hoc selection; not human validated. |
| Compass remains plausible fallback | Same decision record | 1,075/1,075 valid overall, smaller per-request latency, and differing calibration. Not a silently interchangeable runtime model. |
| FARE runtime works on tested configuration | [`../../task7/task7_results_report.md`](../../task7/task7_results_report.md); [`runtime_deployment_gate.md`](runtime_deployment_gate.md) | 88/88 typed-input browser trials (44 OFF, 44 ON); 308/308 valid FARE dimension evaluations for ON turns. No live microphone/STT inference. |
| Observed judge asynchronous boundary | Task 7 report | Judge evaluation created at browser playback-completion acknowledgment, last of seven scores completed median 5.95 seconds later. This does **not** establish global non-interference under outages or concurrent use. |
| First-audio pilot latency | Task 7 report | Backend first-audio-chunk medians OFF 0.603 s, ON 0.600 s; paired mean ON−OFF +0.0004 s and clustered 95% bootstrap interval [−0.053,+0.045] s. No equivalence, device-onset, or zero-overhead claim. |
| Offline reproduction of accepted Task 7 results | [`task8_reproduce.md`](task8_reproduce.md), [`task8_evidence_manifest.json`](task8_evidence_manifest.json), root `task7_reproduce.py` | User-executed replay passed 88/88 and 308/308 against archive SHA-256. Clean-checkout verification remains outstanding at the time of this handoff draft. |
| Deployment qualification boundary | [`runtime_deployment_gate.md`](runtime_deployment_gate.md) | Observed A40-hosted research pilot accepted; A30 deployment, physical GPU isolation, sustained contention, production durability and privacy controls not certified. |
| Human validation planned, not performed | [`human_validation_protocol.md`](human_validation_protocol.md) | Two blinded expert-rater design with diagnostic existing-case arm and independent untouched confirmatory arm; **no human scores yet**. |

The original [`../day4_runtime_evaluation_contract_v1.md`](../day4_runtime_evaluation_contract_v1.md) is a **pre-integration interface contract**: its original status note that it was not yet integrated reflects the document's historical creation stage. Later Task 5–7 code and observations establish the subsequently tested integration. Preserve both stages; neither document should be retrospectively rewritten as if it had observed the other stage's outcomes.

## 3. Reproduction package and provenance

- **Frozen archive:** `task7_evidence_20261108.zip` (kept outside Git; obtain from designated research archive).
- **Archive SHA-256:** `47426b73fad70fdc6635159d2e89d33b290219a8a72d975acf74581f8fc0b20f`.
- **Manifest source revision:** `82e5a9898cf1826803594731cc842aba73973738` (pre-reproduction checkpoint).
- **Tracked dependency repair:** `task5_acceptance.py` added in `82e5a98`.
- **Tracked reproducibility checkpoint:** `12baca5`, adding `task7_reproduce.py`, evidence manifest and reproduction instructions.
- **Frozen schedule SHA-256:** `ab55fe73a9531d692f316c68e025b6d668b13a5839a6962d4ddb121921f6e016`.
- **Frozen reference registry SHA-256:** `d15f4593451f35542144ba60905686026ee68aa311bfa3b192c7c3df0de64155`.
- **Observed local verification:** Python compilation and 14 relevant Task 7 unit tests passed; Task 7 archive replay passed 88/88 and 308/308 (operator logs). This is **not** a clean-checkout pass.

From the repository root, with a Python environment and a separately supplied archive:

```cmd
certutil -hashfile task7_evidence_20261108.zip SHA256
.\.venv\Scripts\python.exe task7_reproduce.py task7_evidence_20261108.zip --out task7_reproduced
.\.venv\Scripts\python.exe -m unittest test_task7_v3_workload test_task7_execution task7_guard_tests -v
```

Outputs in `task7_reproduced/` are generated data; do not commit them as independent sources. The archive contains failed setup attempts and older SQLite rows; the replay must filter solely the **88 accepted trial IDs** and **44 accepted judge-ON turn IDs**, rather than counting everything in the SQLite spool. The original Task 7 report used 30,000 question-cluster bootstrap replicates; the later reproduction script uses a documented 20,000-replicate implementation. Expect matching primary descriptive results, not necessarily identical confidence-interval endpoints. Preserve both method specifications.

### Clean-checkout acceptance — pending execution

After all Task 8 documents have been committed and pushed, verify in a **separate fresh clone** (not the dirty original tree):

```cmd
cd /d C:\Users\shaik\Desktop
git clone --branch judge-selection-sprint --single-branch https://github.com/AmitaKashid/streamtutor_thesis.git streamtutor_task8_verify
cd /d C:\Users\shaik\Desktop\streamtutor_task8_verify
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m unittest test_task7_v3_workload test_task7_execution task7_guard_tests -v
```

Then copy **only** the original ZIP from a trusted local location into the fresh clone, rerun `task7_reproduce.py`, and verify its hash. The targeted tests and analysis runner require standard-library Python; a *full live tutor launch* additionally requires model artifacts, speech dependencies, configured backends, HPC allocations/tunnels and a separate deployment test. Record exact commands, interpreter version, full git SHA and test outcomes in the final release note; do not mark this gate PASS until actually run.

## 4. Runtime and deployment qualification summary

**PASS (narrow):** Browser-typed tutor response delivery and asynchronous post-playback FARE scoring for the 88-trial pilot. The accepted log set contains seven valid evaluation dimension results per ON turn. This does not measure microphone capture, ASR, physical speaker first sound, concurrency saturation or long-term availability.

**NOT VERIFIED:** A30 24 GB deployment, physical device/UUID attribution and GPU co-residency, peak resource contention, robustness to live judge outages/queue saturation, recovery across crashes and Slurm expiration, retention/consent for real student data, and independent judgment validity. The prior A40 smoke tests under different output formats and the later v3 scalar benchmark must not be merged into a single undifferentiated schema-validity figure.

**Operational incidents to retain:** HPC job time limit / resource queuing; SSH-tunnel and host changes; initial erroneous ON block against an OFF backend; recovery and explicit block-identity guard. Failed setup attempts are archived but excluded from the planned 88 accepted observations. See the deployment gate for evidence boundaries.

## 5. Open-risk register

| Priority | Risk | Required owner/action | Exit evidence |
|---|---|---|---|
| P0, scientific | Model scores may disagree with qualified human assessors, particularly subtle borderline omissions and conversational context | Conduct blinded human review and untouched validation holdout following `human_validation_protocol.md` | Expert agreement, paired model comparison, uncertainty intervals and documented adjudication |
| P1, deployability | A30 memory and performance unknown | Allocate actual A30 and collect startup, GPU UUID, VRAM, inference and schema traces **when A30 deployment is a requirement** | A30-specific PASS/FAIL evidence; no extrapolation from A40 |
| P1, operations | Queue overload, service crash or judge outage during later turns not comprehensively qualified | Run fault-injection and durable-restart regression under future operational gate | Tutor remains responsive, no missing/duplicate persisted evaluation identities, auditable skipped/error outcomes |
| P1, performance | Condition ordering, answer length and resource contention confound observed timing | Prespecified repeated randomized/matched test, monitor actual audio onset and node/GPU resource IDs | Confidence interval against declared noninferiority margin |
| P1, privacy | Raw tutor content and SQLite payloads can contain sensitive learner text | Define consent, retention, redaction, access control and deletion procedure before real-student data collection | Approved study/data-handling plan |
| P2, provenance | Historical Prometheus run ID contains `NEW_JOB_ID` placeholder | Reconcile external Slurm record without altering immutable manifest | Separate provenance appendix tied to original artifact hash |
| P2, replication | Local run reproduced but a fresh clone may reveal undeclared dependency | Perform clean-checkout gate and register results | Fresh-clone test and archive replay PASS |

## 6. Handoff outcomes and next research stage

**Accepted now:** A defensible provisional FARE-vs-Compass engineering selection, versioned scalar evaluation architecture, observed A40 research pilot, 88-trial Task 7 result report, replayable Task 7 analysis, and a proposed blinded human-validation design.

**Not accepted now:** A claim that FARE is the human-best judge, a formal latency-equivalence claim, an A30 deployment pass, production-level resilience, or a full microphone-to-speaker latency characterization.

**Immediate closeout:** Complete clean-checkout acceptance, verify all links and hashes, then create and push the final Task 8 documentation checkpoint. Capture final full commit SHA; amend this document only if a result or link actually needs correction.

**Next thesis workstream (separate from Task 8):** A slide-grounded retrieval evaluation, comparing extraction, slide/section chunking, dense and hybrid retrieval, source traceability, Recall@k/nDCG@k/MRR and retrieval latency against a judged question-to-slide relevance set. Follow with voice-pipeline studies and human-rated educational answer quality. Do not bundle these experiments into Task 8 merely to inflate the scope of the judge-selection sprint.

---

**Final status rule:** Mark Task 8 *COMPLETE* only when the decision, deployment gate, human-validation protocol and handoff are committed; the evidence manifest/replay have been verified; the clean-checkout test has run; and the final push is confirmed. External scientific and hardware gates can remain **NOT VERIFIED**, provided they are explicitly carried into the risk register rather than represented as sprint failures or successes.
