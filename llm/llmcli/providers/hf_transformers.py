from __future__ import annotations

from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from .base import LLMProvider, GenerationResult
from ..common.device import resolve_device, resolve_dtype
from ..common.latency import Timer




@dataclass
class HFTransformersProvider(LLMProvider):
    model_id: str
    device: str = "auto"
    dtype: str = "auto"
    trust_remote_code: bool = False
    use_chat_template: bool = True
    system_prompt: str | None = None
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    seed: int = 0
   


    def __post_init__(self):
        self._device = resolve_device(self.device)
        self._dtype = resolve_dtype(self.dtype, self._device)

        if self.seed:
            set_seed(self.seed)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            use_fast=True,
            trust_remote_code=self.trust_remote_code,
        )

        # For portability, avoid fancy device_map until you need multi-GPU.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self._dtype,
            trust_remote_code=self.trust_remote_code,
        )

        self.model.to(self._device)
        p = next(self.model.parameters())
        print(">>> resolved device:", self._device)
        print(">>> model param device:", p.device)
        print(">>> model param dtype:", p.dtype)
        self.model.eval()

        # Some tokenizers don’t have pad token set
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_input(self, user_text: str) -> str:
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": user_text})
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Generic fallback prompt format
        if self.system_prompt:
            return f"System: {self.system_prompt}\nUser: {user_text}\nAssistant:"
        return f"User: {user_text}\nAssistant:"

    @torch.inference_mode()
    def generate(self, user_text: str) -> GenerationResult:
        prompt = self._build_input(user_text)
        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self._device)
        attn_mask = enc.get("attention_mask", None)
        if attn_mask is not None:
            attn_mask = attn_mask.to(self._device)

        prompt_tokens = int(input_ids.shape[-1])

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0),
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        with Timer() as t:
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                **gen_kwargs,
            )

        # Decode only newly generated tokens (more stable for chat templates)
        new_tokens = int(out.shape[-1] - input_ids.shape[-1])
        gen_ids = out[0, -new_tokens:] if new_tokens > 0 else out[0, :]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        total_seconds = t.seconds
        tps = (new_tokens / total_seconds) if total_seconds > 0 else 0.0

        return GenerationResult(
            text=text.strip(),
            total_seconds=total_seconds,
            prompt_tokens=prompt_tokens,
            new_tokens=new_tokens,
            tokens_per_second=tps,
        )
