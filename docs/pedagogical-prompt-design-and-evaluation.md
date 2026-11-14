# Pedagogical prompts for StreamTutor

Version: `pedagogy-v1.0.0`. Status: implemented candidate; model behavior, pedagogical effectiveness, learning gains, safety and latency remain unverified. Scope: the existing Socratic, inquiry, adaptive and collaborative modes. The user's memorandum is a design input, not evidence that its claims were implemented or validated.

## 1. Decision and runtime boundaries

The same shared factual, history, transcription and voice guidance wraps one style-specific method in every mode. The concise source of truth is `prompts/common.py` and the four `prompts/{socratic,inquiry,adaptive,collaborative}.py` modules. `SYSTEM_PROMPT`, `get_style_prompt`, `compose_prompt`, style identifiers, UI selection, RAG selection and streaming APIs are retained. The common policy comes first, the teaching method follows, and the voice rule is last. The existing registry then appends `[CONTEXT]`, `[USER]` and `[ASSISTANT]`. History is inside the current user-query text. The historic wrapper's instruction to answer directly was changed to respect the selected approach.

**Important role distinction:** `[SYSTEM]` is a literal marker in the composed prompt; it does not create an API system role. In the Qwen thinking HTTP provider the compiled teaching prompt is sent inside the user message after a separate short system message. Provider architectures differ. Treat retrieved material as data, but do not regard this wording or the literal markers as a proven prompt-injection security boundary. Avoid untested claims about exact hierarchy or placement compliance.

**No new learner-state engine:** the application does not supply the memorandum's `{level}`, `{mastery}`, `{recent_errors}` or `{hints_used}` profile; it also does not parse `[STATE: ...]` or `[END]` before streaming speech. The candidate therefore uses only available conversation text for adaptive behavior and does not generate control tags. It cannot establish mastery, track a formal hint ladder, identify multiple speakers, enforce a five-phase state machine, or promise consistent behavior after history truncation. Those require separately designed and tested orchestration.

**Retrieval:** when material is provided, the instructions request grounding in it and an honest statement when it does not support the answer. When no material is supplied (including RAG off), they allow a cautious general explanation explicitly distinguished from course material. Student-provided examples are not attributed to the slides. Retrieved sources can be irrelevant or incomplete, and this policy needs explicit missing/irrelevant/contradictory evidence tests. A prompt cannot prove factual correctness. The UI may separately show source metadata.

**Voice:** the 70-word target, plain speech and at-most-one-question rule are soft instructions for every mode, not a deterministic length or formatting guarantee. A direct request for an explanation and a stuck student are handled without an endless questioning loop. Evaluate deviations; do not silently truncate speech, strip valid mathematics, or insert unimplemented state control.

## 2. Style intent and constraints

| Mode | Observable intended move | Escalation / scope |
| --- | --- | --- |
| Socratic | Use a focused question to elicit a useful inference, without prematurely supplying the next solution step in problem-solving cases. | Give a smaller hint when stuck; provide a brief explanation on an explicit request or when continued help is not working. |
| Inquiry | Invite a prediction, consider a small example or evidence, and help the student form a conclusion when appropriate. | Take one step at a time; do not present invented outcomes as observations; explain directly if requested. |
| Adaptive | Change explanation, hint or extension based on evidence in the visible student messages. | Do not invent a level, mastery score, attempt or hint count; this is conversational adaptation, not a validated ITS learner model. |
| Collaborative | Acknowledge an actual student idea, offer a perspective and invite evaluation when useful. | Correct false claims respectfully; do not assume a group, speaker identities, or earlier shared decisions. |

There is no independent scaffold or hint mode, neutral-baseline UI mode, group-detection mode, or model-generated learner state in this change. A versioned snapshot of the four prior prompts in `data/prompt_evaluation/baseline_v0.json` supports comparison without claiming it is a pedagogically neutral baseline.

## 3. Prompt length and placement

A 350–450-word prompt is a proposal in the supplied memorandum, not a universal optimum or a measured threshold. The candidate is deliberately shorter than that proposal but longer than the previous prompts. `len(text.split())` and character counts below describe only the style instruction, not the complete model request, model tokens, generated answer, or time:

| Mode | Prior words / characters | Candidate words / characters |
| --- | ---: | ---: |
| Socratic | 47 / 288 | 284 / 1,752 |
| Inquiry | 105 / 747 | 288 / 1,804 |
| Adaptive | 59 / 400 | 283 / 1,776 |
| Collaborative | 49 / 364 | 280 / 1,803 |

The candidate methods differ slightly in length. Common text controls some confounders but does not alone prove that any outcome difference is caused by pedagogy. Record full compiled prompt and its hash, actual prompt tokens, cache tokens, prefill, TTFT, decode and output length; the provider-level system text, chat template, history and RAG all contribute. Five historical turns are not a token budget. Version a prompt set and freeze it before evaluation; compare results only with recorded template and provider. `scripts/measure_teaching_prompt_tokens.py` uses Qwen's llama-server template and tokenizer on one fixed synthetic question/RAG pair. It makes no generation requests and does not measure response quality, real-turn latency, provider-specific budgets or behavior under other models. Compare its token count against an actual generation event before treating it as exact for that server configuration.

Instruction order is an experimental design choice: placing shared role/grounding first makes it stable, teaching method follows, and voice guidance is repeated in the same position across variants. The actual current question and any RAG follow these blocks. No claim is made that first or last is universally optimal, that extra text always increases latency proportionally, or that one format works best for every model. Model-specific chat templates, caching, KV state, hardware and prompt length can all affect timings. Prior context-study and cache-probe results used older prompt text and hashes; new runs using this version are a new experimental condition, not directly pooled with those results.

## 4. Evaluation protocol

### Unit of evaluation and frozen conditions

1. Evaluate one model and deployment first. Record reported model identity, model hash where available, runtime/version, node/load, slot/configuration, context window, prompt version and full prompt hash. Comparing models under matched hardware is a separate task.
2. Compare four candidate modes and the four matching prior prompts on paired, identical cases. The old Socratic and other old prompts differ substantially, so this measures prior-versus-new change per mode, not a neutral pedagogical baseline. If a neutral baseline is required, design and preregister one separately; it is not currently a runtime mode.
3. Fix the question, full prior conversation, retrieval evidence, teaching context, history limit, RAG state, system/provider template, thinking setting, generation parameters and output cap per matched comparison. Randomize condition order with a saved seed, log failures and finish reasons, and avoid concurrent traffic for latency runs. Keep learning-gain evaluation separate from live tutoring telemetry.
4. Use topic/sequence-level train-development-held-out separation, not random turns of the same conversation. Draft `data/prompt_evaluation/scenarios_draft_v1.jsonl` contains 16 synthetic cases, is not reviewed and must not be reported as a representative dataset. Human reviewers should verify reference evidence, ambiguous cases, style-appropriate responses and any RAG leakage before freezing; expand using course materials and realistic independent student dialogues. Do not use personal learner data without appropriate consent and access controls.
5. For turns with history, preserve identical history snapshots. A purely frozen transcript measures a single response, not how each style shapes later dialogue; evaluate multi-turn trajectories separately. Prompts must not be scored on how close an intentionally guided answer is to a full reference answer alone.
6. Run repeated generations where stochastic, and report the number of unique scenarios separately from the number of outputs. Avoid treating repeated outputs and variants of one conversation as independent students. Record server prefill/decode, prompt/output/cached tokens, answer TTFT, total pipeline and incomplete/error rates where available. Report p50, distribution and p95 only when sample size supports it; label small pilots descriptive. Do not sum overlapping latency components.

### Shared quality and reliability criteria

Grade eligible responses independently on a preregistered 1–5 anchored rubric: 1 = substantially fails, 3 = mixed or partly useful, 5 = accurate, appropriate and actionable. Use `N/A` where a dimension cannot be judged, record why, and report its eligible denominator. Capture concrete error annotations alongside scores.

| Criterion | What to inspect | Suggested rate / measure |
| --- | --- | --- |
| Correctness | Domain facts, calculations and correction of misconceptions; did not endorse a false premise. | Human 1–5; material error rate = responses with a material error / eligible responses. |
| Groundedness and evidence use | Supplied slide support; accurately separate student-provided facts; no invented evidence; admit when course evidence is insufficient. | Supported-claim rate and unsupported-claim rate; human 1–5. Count claims and responses separately. |
| Completeness and relevance | Addresses the student's intent at the appropriate instructional step; avoids evasion on explicit explanation requests. | Human 1–5; direct-request fulfillment / eligible requests. A pedagogical hint is not automatically incomplete. |
| Clarity and voice usability | Spoken intelligibility, concise wording, terminology and a manageable question; check actual synthesized audio in a separate TTS-on subset. | Human 1–5; audible formatting artifacts and pronunciation issues / eligible outputs; word count distribution and over-70-word share, excluding explicit long requests. |
| Pedagogical support and scaffolding | Moves the learner forward; guidance matches need without unnecessary answer leakage or unproductive withholding. | Human 1–5; useful-hint rate; premature-solution leakage / eligible problem-solving turns; repeated-stall rate. |
| Misconception handling | Identifies or helps surface the mistake and gives specific, respectful feedback without false agreement. | Human 1–5; false agreement / incorrect-claim cases; targeted feedback rate. |
| Context and uncertainty | Uses visible history correctly; requests missing details; does not claim truncated information was never provided. | Correct recall, appropriate clarification, false denial, invented-history rates, each over its applicable cases. |
| ASR robustness | Interprets plausible transcription mistakes from context; asks a concise clarification when ambiguous. | Correct interpretation / resolvable cases; appropriate clarification / ambiguous cases. |
| Behavior and scope | Respects student intent, avoids fabricated group members and inappropriate or repetitive redirection. | Unsupported-persona, refusal and unnecessary-clarification rates; human review. |
| Robustness to untrusted retrieval | Tests irrelevant or adversarial instructions placed in retrieved material. | Instruction-following failure / attack cases; report as behavioral testing, not proof of security. |
| Runtime and completion | Completed valid response, measured prompt/cache/output tokens and timing, TTS errors, cancellation and reliability. | Incomplete/error rates, medians and, at adequate scale, tail values; include sample counts and uncertainty. |

### Mode fidelity and multi-turn criteria

Use separate mode-specific ratings on eligible scenarios rather than assuming that “more questions” is better in every mode:

- **Socratic:** appropriate probing question; student opportunity to reason; avoid premature solution leakage in an exercise; hint or direct explanation when requested/stuck; no repeated question loop. Measure leakage only on cases where withholding is appropriate.
- **Inquiry:** prediction before outcome where the task supports it, meaningful evidence/test, link between observation and conclusion, and no invented outcome. Do not grade a one-turn reply as a failed full cycle solely because the conversation has not progressed.
- **Adaptive:** appropriate response changes between matched beginner/advanced or correct/confused cases when that information is actually visible; escalation for repeated difficulty; no invented profile or mastery. This is behavior sensitivity, not proof of a validated student model.
- **Collaborative:** builds on the student's actual contribution, adds a useful perspective, invites evaluation when useful and disagrees with an incorrect claim respectfully; does not invent participants. Do not require a question or role swap every turn.
- **Dialogue-level:** turns to resolve the learner's request, repeated unproductive loops, coherent progression, appropriate escalation and student agency; count interruptions and unresolved outcomes separately. Only an appropriate pre/post learning evaluation with participants can support claims about learning gains.

### Human review, automated judges and interpretation

Have trained independent reviewers blind to prompt version, style where feasible, model and latency annotate a prespecified double-rated subset. Report agreement by criterion (for ordinal ratings e.g. quadratic-weighted kappa where appropriate, with prevalence/degenerate cases and sample counts) and resolve disagreements with a documented process. Use the existing FARE judge only after checking its current operational version and actual rubric mapping; do not assume it implements these mode-fidelity criteria. Calibrate it against human labels, report judge–human agreement by dimension, inspect disagreements and order/verbosity bias, and never treat the judge as ground truth. BLEU/BERTScore can be descriptive supplemental metrics but can penalize valid guided questions because they are not full reference answers. Human learning outcomes need separate participants, consent, a preregistered intervention, pre/post or transfer tests, and an appropriate comparator.

## 5. How to measure prompt tokens on the user's Qwen server

With the stack already running and Qwen endpoint confirmed:

```cmd
.venv\Scripts\python.exe scripts\measure_teaching_prompt_tokens.py --url http://127.0.0.1:8008 --output prompt_token_counts_qwen_v1.json
```

The script uses `/v1/models`, `/apply-template` and `/tokenize`; it makes no completion request. Output is created exclusively and will not overwrite an existing file. The template assumes the Qwen thinking HTTP provider's fixed system message, one synthetic user question, the same synthetic RAG evidence for all modes, no history, thinking off, and the server's current tokenizer. It reports instruction word/character counts, full rendered request token counts and prompt hashes for old and new versions. The endpoint-reported model name is not a cryptographic model hash. Run representative generation requests and compare server prompt-token counts before relying on exact agreement. Use full real requests and server counters for actual impact and separately measure cache-on/off latency and output quality.

## 6. Primary research used

1. Kargupta et al. (2024), *Instruct, Not Assist*: Socratic code debugging; its results do not establish efficacy for StreamTutor's Data Mining task. https://arxiv.org/abs/2406.11709
2. Borchers and Shou (2025), *Can Large Language Models Match Tutoring System Adaptivity?* A prompt-variation framework and limits of inferred adaptivity. https://arxiv.org/abs/2504.05570
3. Macina et al. (2023), *MathDial*: teacher–student dialogues and the trade-off between instructional guidance and prematurely revealing solutions. https://aclanthology.org/2023.findings-emnlp.372/
4. Maurya et al. (2025), *Unifying AI Tutor Evaluation / MRBench*: eight pedagogical dimensions and limitations of LLM judges; focused on mathematical mistake remediation, requiring adaptation to Data Mining and voice. https://aclanthology.org/2025.naacl-long.57/
5. Sclar et al. (2023), *Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design*: response sensitivity to format across evaluated models and tasks, not a universal order rule. https://arxiv.org/abs/2310.11324
6. Liu et al. (2023), *Lost in the Middle*: relevant-information position affects performance in evaluated long-context tasks, not an optimal prompt length for StreamTutor. https://arxiv.org/abs/2307.03172
