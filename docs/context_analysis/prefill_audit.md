# StreamTutor context and inference measurements

Audit baseline: `feature/thinking-budget-selector` at `16b99110` (2026-11-13). These are code-derived settings, not verification of a running Slurm job. User-local changes and uploaded cluster scripts may differ.

## Prompt construction

For the four direct llama-server models, the client sends a chat-template system message ("You are a concise AI tutor for data mining...") and one user message. The latter contains the teaching-style instructions, optional formatted retrieval context, optional flattened previous user/assistant turns, and the current question. The model's chat template adds its own formatting and tokens. The legacy Llama path uses a model-registry system message and a single composed user message. The application's displayed `promptChars` excludes the provider system message and template. `conversationChars` describes formatted history, not tokens.

Web defaults: RAG on, top-k 2, conversation on, max 5 previous completed turns. `0` means unlimited (web accepts 0–50). RAG configuration has a 6,000-character maximum, not a token limit; a full retrieved block that does not fit is omitted. Retrieval uses only the current question, so context-dependent follow-ups may retrieve unrelated material. A changing RAG block, its similarity score, or a shifting/renumbered history can change the prompt prefix before most history and reduce cache reuse. This is a testable hypothesis; actual `cache_n` decides.

The direct model Slurm script configures `-c 8192 --parallel 1` and the client requests `max_tokens=8192`; this is a generation ceiling, not 8,192 tokens reserved in addition to the prompt. Remaining context, thinking, output and server behavior determine what fits. The legacy Python server code defaults to 2,048 context tokens and 360 output tokens for Llama 3 absent overrides. Verify the running job's actual command, logs, model identity, Slurm node, and environment. Do not interpret the model's maximum trained context as the configured serving context. There is no model-token-aware per-turn budget or reserved answer allowance in this snapshot.

## Max turns and prefill

`max_turns` controls which completed prior turns are in the next prompt. It does not change the model context window or the number of tokens prefetched per forward pass. The previous code trimmed stored turns only when a new turn was added, so reducing the limit did not affect the next prompt. The patch applies the limit when reading too. Previously discarded turns are not recovered when the limit is later increased. `max_chars` is set to unlimited for web sessions.

Prefill processes the complete tokenized prompt and produces the initial state used for generation. It is not an answerability check and cannot independently decide whether to forget dialogue. For example, "Why is that?" may require the previous answer. In the four llama-server deployments, the b11556 source defaults `cache_prompt` to true and can reuse a matching prompt prefix. Reuse is not guaranteed and depends on the actual slot, server version, earlier requests and exact prefix; measure `cache_n`. Identical-looking history can tokenize or template differently between models.

## Measurement semantics

On completed calls to a compatible llama-server, the new table fields use its final streaming `timings` object:

- Prompt tokens = `prompt_n` + `cache_n`, including the formatted chat template.
- Cached tokens = `cache_n`; processed prompt tokens = `prompt_n`.
- Prefill = `prompt_ms / 1000`, server-side processing of uncached prompt tokens.
- Output tokens = `predicted_n`; Decode = `predicted_ms / 1000`; Decode tok/s = server `predicted_per_second`.

Decode counters include reasoning and answer generation when thinking is on. Prefill is not client answer TTFT: queueing, network, sampling, reasoning and client buffering differ. Durations and rates are server definitions, not a GPU profiler. Missing stats remain unavailable; interrupted or unsuccessful streams are not marked completed. The current legacy Llama Python `/stream` endpoint does not supply these server counters, so the new columns show `—` for that path. Do not substitute character counts or client TTFT for Llama prefill. A separately tested protocol/runtime change is necessary there.

## Opt-in synthetic probe

Start the normal stack, then in Windows CMD run on one verified direct llama-server tunnel:

```cmd
.venv\Scripts\python.exe scripts\context_diagnostics\probe.py --url http://127.0.0.1:8008 --model-label qwen3_5_9b_q6 --output context_probe_qwen.jsonl --trials 3
```

Use 8007 (Gemma 26B), 8008 (Qwen), 8005 (Gemma E4B), or 8006 (Bonsai); check the launcher confirms endpoint identity. Port 8000 uses a different legacy API and is not supported by this probe. The run sends real requests and alters the server's cache state. Run on an authorised allocated node, keep the server free of other traffic, record exact hardware and job settings, and obey the cluster's restrictions on personal data. It does not start or cancel jobs.

The probe uses synthetic history of 0, 1, 3, or 5 turns, a fixed synthetic RAG passage, thinking off, greedy sampling, and 128 output tokens. It randomises the condition order, records cold (`cache_prompt=false`), priming calls, identical warm prompt, changed question, and a sliding history. The JSONL includes hashes and lengths, not prompt or answer text. Each condition costs 4–6 calls; `--trials 3` costs 66 calls. It refuses to overwrite output. `finish_reason=length`, unexpected reasoning, absent timing fields, HTTP errors, and unexpected interruptions must be examined; a nonzero exit indicates a failed stream or absent required measurements. Priming results and the synthetic single passage are not representative of actual student traffic. The model label is user-provided, not an identity check.

Compare per-condition distributions (individual values and median initially; p95 only with enough samples) of `serverPromptTokens`, `serverCachedTokens`, prefill, decode, and client first-output delay. Warm identical prompts are an upper-bound cache scenario, not normal dialogue. Repeat on the same assigned GPU type and isolated load, record runtime version, quantization, context size, slot count, model and configuration, warmup, seed and cache policy. Treat speed as a measurement, not evidence of answer quality.

## Next evaluation and design

Build a held-out dialogue set with standalone and context-dependent follow-ups, topic changes, long history, interruption and retrieval failure. Compare no history, 1/3/5 turns, and later token-budgeted recent turns, with the same prompts, RAG sources and hardware. Measure contextual correctness, faithfulness, citations, missing references, incomplete answers, actual prompt/output tokens, cached tokens, prefill, decode and end-to-end time. Use human-scored examples and calibrate any separate judge; pre-register acceptance criteria and uncertainty before choosing policies. Score paired answers independently; avoid treating synthetic prompts as a quality benchmark.

Do not silently drop history just because prefill is expensive. Once baseline data exists, evaluate a token-aware budget reserving answer space and prioritising system/current question/relevant retrieval plus recent complete turns; a history-relevance gate or query rewrite adds latency and can drop a needed referent, so compare against a conservative always-recent baseline. Reordering prompts for stable prefixes can change model behavior and cache reuse; A/B test quality and latency before rollout. A summary is lossy and should preserve provenance or allow recovery, not silently replace needed student context.

References: [llama.cpp b11556 server API](https://github.com/ggml-org/llama.cpp/blob/b11556/tools/server/README.md), [vLLM automatic prefix caching](https://docs.vllm.ai/en/v0.15.0/features/automatic_prefix_caching/), [Lost in the Middle](https://arxiv.org/abs/2307.03172), [Conversational retrieval and rewriting](https://aclanthology.org/2024.findings-emnlp.622/).
