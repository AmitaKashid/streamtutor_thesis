from pathlib import Path

from .registry import ModelSpec, register


ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "models"


register(
    "fare_20b_q4",
    ModelSpec(
        backend="llamacpp",
        gguf_path=str(
            MODELS / "FARE-20B.Q4_K_M.gguf"
        ),

        # Use the chat template embedded in the GGUF.
        chat_format=None,
        response_protocol="harmony",

        system_prompt=(
            "You are a strict educational answer evaluator. "
            "Return only valid JSON matching the requested schema."
        ),
    ),
)
