from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple
from contextlib import contextmanager
import time
import math

import llama_cpp as llama_package
from llama_cpp import Llama, llama_chat_format, llama_cpp

from .base import LLMProvider, GenerationResult
from .response_protocols import (
    ResponseProtocolError,
    decode_response_protocol,
)
from ..common.latency import Timer


OBSERVED_LLAMA_CPP_VERSIONS = frozenset({'0.3.4', '0.3.20'})


@contextmanager
def _observe_llama3_stream(llm, enabled):
    """Observe actual tokens for the reviewed llama-cpp-python versions.

    The server serializes every operation on this model. The temporary hooks
    observe its existing generator, never a second inference. Unknown versions
    deliberately retain unavailable metrics instead of guessing. Prompt eval
    is timed with a GPU synchronization; decode is time inside generator steps
    after the first sampled token, excluding terminal tokens and network waits.
    """
    result = {}
    version = getattr(llama_package, '__version__', None)
    model = getattr(llm, '_model', None)
    if version == '0.3.4':
        eog = getattr(llama_cpp, 'llama_token_is_eog', None)
        vocabulary = getattr(model, 'model', None)
    elif version == '0.3.20':
        eog = getattr(llama_cpp, 'llama_vocab_is_eog', None)
        vocabulary = getattr(model, 'vocab', None)
    else:
        eog = vocabulary = None
    instance_dict = getattr(llm, '__dict__', None)
    original_generate = getattr(llm, 'generate', None)
    original_eval = getattr(llm, 'eval', None)
    if (not enabled or version not in OBSERVED_LLAMA_CPP_VERSIONS or
            not callable(eog) or vocabulary is None or not isinstance(instance_dict, dict) or
            not callable(original_generate) or not callable(original_eval) or
            getattr(llm, 'draft_model', None) is not None):
        yield result
        return

    prior = {name: instance_dict.get(name) for name in ('generate', 'eval')}
    existed = {name: name in instance_dict for name in prior}
    ctx = getattr(getattr(llm, '_ctx', None), 'ctx', None)
    synchronize = getattr(llama_cpp, 'llama_synchronize', None)
    state = {'prompt': None, 'first_eval': True, 'sampled': 0,
             'output': 0, 'decode_steps': 0, 'decode_seconds': 0.0,
             'eog_valid': True}

    def observed_eval(tokens, *args, **kwargs):
        first = state['first_eval'] and state['prompt'] is not None
        if first:
            state['first_eval'] = False
            try:
                count = len(tokens)
                if 0 < count <= state['prompt']:
                    result['serverPromptTokens'] = state['prompt']
                    result['serverProcessedPromptTokens'] = count
                    result['serverCachedTokens'] = state['prompt'] - count
            except (TypeError, ValueError):
                pass
        start = time.perf_counter() if first else None
        output = original_eval(tokens, *args, **kwargs)
        if first and start is not None and ctx is not None and callable(synchronize):
            try:
                synchronize(ctx)
                result['serverPrefillSeconds'] = time.perf_counter() - start
            except (TypeError, ValueError, RuntimeError):
                pass
        return output

    def observed_generate(tokens, *args, **kwargs):
        try:
            state['prompt'] = len(tokens)
        except TypeError:
            state['prompt'] = None
        source = iter(original_generate(tokens, *args, **kwargs))
        try:
            while True:
                start = time.perf_counter()
                try:
                    token = next(source)
                except StopIteration:
                    return
                elapsed = time.perf_counter() - start
                try:
                    is_end = bool(eog(vocabulary, token))
                except (TypeError, ValueError, RuntimeError):
                    state['eog_valid'] = False
                    is_end = True
                if not is_end:
                    state['output'] += 1
                    if state['sampled'] > 0:
                        state['decode_steps'] += 1
                        state['decode_seconds'] += elapsed
                state['sampled'] += 1
                yield token
        finally:
            close = getattr(source, 'close', None)
            if callable(close):
                close()

    try:
        llm.eval = observed_eval
        llm.generate = observed_generate
    except (AttributeError, TypeError):
        # Restore any attribute installed before a failed assignment.
        for name in prior:
            if existed[name]:
                setattr(llm, name, prior[name])
            elif name in instance_dict:
                delattr(llm, name)
        yield result
        return

    try:
        yield result
        if state['prompt'] is not None and state['eog_valid']:
            result['serverOutputTokens'] = state['output']
            if state['decode_steps']:
                result['serverDecodeSeconds'] = state['decode_seconds']
                if state['decode_seconds'] > 0:
                    result['serverDecodeTokensPerSecond'] = (
                        state['decode_steps'] / state['decode_seconds']
                    )
            result['serverMetricsSource'] = 'llama_cpp_python_observed_v1'
    finally:
        for name in prior:
            if existed[name]:
                setattr(llm, name, prior[name])
            elif name in instance_dict:
                delattr(llm, name)


@dataclass
class LlamaCppProvider(LLMProvider):
    model_path: str
    chat_format: Optional[str]
    chat_template_kwargs: Optional[Dict[str, Any]] = None
    n_ctx: int = 4096
    n_threads: int = 8
    n_threads_batch: int = 8
    n_gpu_layers: int = -1
    n_batch: int = 512
    n_ubatch: int = 512
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    system_prompt: Optional[str] = None
    seed: int = 0
    verbose: bool = True
    flash_attn: bool = True
    offload_kqv: bool = True
    response_protocol: Optional[str] = None

    def __post_init__(self) -> None:
        print("\n[LLM LOAD CONFIG]")
        print(f"  model_path      : {self.model_path}")
        print(f"  chat_format     : {self.chat_format}")
        print(f"  template_kwargs : {self.chat_template_kwargs}")
        print(f"  response_protocol: {self.response_protocol}")
        print(f"  n_ctx           : {self.n_ctx}")
        print(f"  n_threads       : {self.n_threads}")
        print(f"  n_threads_batch : {self.n_threads_batch}")
        print(f"  n_gpu_layers    : {self.n_gpu_layers}")
        print(f"  n_batch         : {self.n_batch}")
        print(f"  n_ubatch        : {self.n_ubatch}")
        print(f"  flash_attn      : {self.flash_attn}")
        print(f"  offload_kqv     : {self.offload_kqv}")
        print(f"  max_new_tokens  : {self.max_new_tokens}")

        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_threads_batch=self.n_threads_batch,
            n_gpu_layers=self.n_gpu_layers,
            n_batch=self.n_batch,
            n_ubatch=self.n_ubatch,
            chat_format=self.chat_format,
            seed=self.seed,
            verbose=self.verbose,
            flash_attn=self.flash_attn,
            offload_kqv=self.offload_kqv,
        )


        if self.chat_template_kwargs:
            base_chat_handler = (
                self.llm.chat_handler
                or self.llm._chat_handlers.get(self.llm.chat_format)
                or llama_chat_format.get_chat_completion_handler(
                    self.llm.chat_format
                )
            )

            template_kwargs = dict(self.chat_template_kwargs)

            def chat_handler_with_kwargs(*args, **kwargs):
                return base_chat_handler(
                    *args,
                    **{**template_kwargs, **kwargs},
                )

            self.llm.chat_handler = chat_handler_with_kwargs

    def _messages(self, user_text: str):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_text})
        return messages

    def generate(self, user_text: str) -> GenerationResult:
        messages = self._messages(user_text)

        with Timer() as t:
            out = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stream=False,
            )

        raw_text = out["choices"][0]["message"]["content"].strip()
        protocol_error = None
        try:
            text = decode_response_protocol(raw_text, self.response_protocol)
        except ResponseProtocolError as exc:
            # Return the original response and a structured decoding error so
            # the judge can retain evidence and apply its one-retry policy.
            text = ""
            protocol_error = f"{type(exc).__name__}: {exc}"

        usage = out.get("usage", {}) or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        new_tokens = int(usage.get("completion_tokens", 0))

        total_seconds = t.seconds
        tps = (new_tokens / total_seconds) if total_seconds > 0 and new_tokens > 0 else 0.0

        print("\n[LLM GENERATE METRICS]")
        print(f"  prompt_tokens : {prompt_tokens}")
        print(f"  new_tokens    : {new_tokens}")
        print(f"  total_seconds : {total_seconds:.3f}")
        print(f"  tok_per_sec   : {tps:.3f}")

        return GenerationResult(
            text=text,
            total_seconds=total_seconds,
            prompt_tokens=prompt_tokens,
            new_tokens=new_tokens,
            tokens_per_second=tps,
            raw_text=raw_text if self.response_protocol else None,
            response_protocol=self.response_protocol,
            protocol_error=protocol_error,
        )

    def stream_events(self, user_text: str):
        """Emit deltas and native finish reason; timings are optional native counters.

        Calls using the same model must be serialized by the server. The native
        counters measure batched and single-token evaluation, which are not
        necessarily equivalent to prompt prefill and output decoding.
        """
        ctx = getattr(getattr(self.llm, '_ctx', None), 'ctx', None)
        perf_available = False
        if ctx is not None:
            try:
                llama_cpp.llama_perf_context_reset(ctx)
                perf_available = True
            except (AttributeError, TypeError, RuntimeError):
                pass
        finish_reason = None
        usage = None
        with _observe_llama3_stream(self.llm, getattr(self, 'chat_format', None) == 'llama-3') as observed:
            stream = self.llm.create_chat_completion(
                messages=self._messages(user_text), max_tokens=self.max_new_tokens,
                temperature=self.temperature, top_p=self.top_p, stream=True,
            )
            try:
                for event in stream:
                    if isinstance(event.get('usage'), dict):
                        usage = event['usage']
                    for choice in event.get('choices') or []:
                        finish_reason = choice.get('finish_reason') or finish_reason
                        piece = (choice.get('delta') or {}).get('content')
                        if piece:
                            yield {'delta': piece}
            finally:
                close = getattr(stream, 'close', None)
                if callable(close):
                    close()
        metrics = dict(observed)
        if usage:
            for source, target in (('prompt_tokens', 'serverPromptTokens'),
                                   ('completion_tokens', 'serverOutputTokens')):
                value = usage.get(source)
                if type(value) is int and value >= 0:
                    metrics[target] = value
        if perf_available:
            try:
                perf = llama_cpp.llama_perf_context(ctx)
                if perf.n_p_eval > 0 and math.isfinite(perf.t_p_eval_ms) and perf.t_p_eval_ms >= 0:
                    metrics['legacyBatchedEvalSeconds'] = perf.t_p_eval_ms / 1000.0
                    metrics['legacyBatchedEvalTokens'] = perf.n_p_eval
                if perf.n_eval > 0 and math.isfinite(perf.t_eval_ms) and perf.t_eval_ms >= 0:
                    metrics['legacySingleEvalSeconds'] = perf.t_eval_ms / 1000.0
                    metrics['legacySingleEvalCalls'] = perf.n_eval
            except (AttributeError, TypeError, RuntimeError):
                pass
        yield {'done': True, 'finish_reason': finish_reason, 'metrics': metrics}

    def stream(self, user_text: str) -> Iterator[str]:
        for event in self.stream_events(user_text):
            if event.get('delta'):
                yield event['delta']

    def stream_with_timing(self, user_text: str) -> Iterator[Tuple[str, dict]]:
        messages = self._messages(user_text)

        # Try to estimate prompt tokens before generation
        prompt_tokens_est = None
        try:
            prompt_tokens_est = len(
                self.llm.tokenize(
                    self.llm.tokenize(b"") if False else str(messages).encode("utf-8"),
                    add_bos=True,
                )
            )
        except Exception:
            prompt_tokens_est = None

        request_start = time.perf_counter()
        first_token_time = None
        emitted_chars = 0

        if prompt_tokens_est is not None:
            print("\n[LLM REQUEST]")
            print(f"  estimated_prompt_tokens : {prompt_tokens_est}")

        stream = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stream=True,
        )

        for event in stream:
            try:
                delta = event["choices"][0].get("delta", {})
                piece = delta.get("content")
                if not piece:
                    continue

                emitted_chars += len(piece)

                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                    print("\n[LLM FIRST TOKEN]")
                    print(f"  ttft_s : {first_token_time - request_start:.3f}")

                yield piece, {
                    "request_start_s": request_start,
                    "first_token_s": first_token_time,
                    "first_token_latency_s": first_token_time - request_start,
                    "estimated_prompt_tokens": prompt_tokens_est,
                    "emitted_chars": emitted_chars,
                }
            except Exception:
                continue
