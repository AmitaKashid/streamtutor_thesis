from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple
import time

from llama_cpp import Llama, llama_chat_format

from .base import LLMProvider, GenerationResult
from ..common.latency import Timer


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

    def __post_init__(self) -> None:
        print("\n[LLM LOAD CONFIG]")
        print(f"  model_path      : {self.model_path}")
        print(f"  chat_format     : {self.chat_format}")
        print(f"  template_kwargs : {self.chat_template_kwargs}")
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

        text = out["choices"][0]["message"]["content"].strip()

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
        )

    def stream(self, user_text: str) -> Iterator[str]:
        messages = self._messages(user_text)

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
                if piece:
                    yield piece
            except Exception:
                continue

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