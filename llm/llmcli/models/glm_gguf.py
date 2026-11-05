from pathlib import Path

from .registry import ModelSpec, register


ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "models"


register(
    "glm4_32b_0414_q4",
    ModelSpec(
        backend="llamacpp",
        gguf_path=str(
            MODELS / "glm-4-32b-0414-q4_k_m.gguf"
        ),

        # Use the chat template embedded in the GGUF.
        chat_format=None,

        system_prompt=(
            "You are a strict educational answer evaluator. "
            "Return only valid JSON matching the requested schema."
        ),
    ),
)