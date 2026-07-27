from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal

Backend = Literal["llamacpp", "hf"]

@dataclass(frozen=True)
class ModelSpec:
    backend: Backend

    # HF
    model_id: Optional[str] = None
    use_chat_template: bool = True
    trust_remote_code: bool = False

    # llama.cpp (GGUF)
    # llama.cpp (GGUF)
    gguf_path: Optional[str] = None
    chat_format: Optional[str] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None

    # shared
    system_prompt: Optional[str] = None

_REGISTRY: Dict[str, ModelSpec] = {}

def register(key: str, spec: ModelSpec) -> None:
    _REGISTRY[key] = spec

def get_model_spec(key: str) -> ModelSpec:
    if key not in _REGISTRY:
        raise KeyError(f"Unknown model key '{key}'. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[key]

from . import llama3_gguf, qwen_gguf, ministral_gguf, gemma_gguf, glm_gguf  # noqa