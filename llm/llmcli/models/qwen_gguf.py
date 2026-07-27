from pathlib import Path

from .registry import ModelSpec, register


ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "models"


register(
    "qwen3_5_9b_q6",
    ModelSpec(
        backend="llamacpp",
        gguf_path=str(MODELS / "Qwen_Qwen3.5-9B-Q6_K.gguf"),

        # Use the chat template embedded inside the Qwen GGUF.
        chat_format=None,
        chat_template_kwargs={"enable_thinking": False},

        # Keep this consistent with the existing Llama registration.
        system_prompt="You are a helpful assistant.",
    ),
)