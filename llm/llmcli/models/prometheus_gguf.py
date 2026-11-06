from pathlib import Path

from .registry import ModelSpec, register


ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "models"


register(
    "prometheus_7b_v2_q4",
    ModelSpec(
        backend="llamacpp",
        gguf_path=str(
            MODELS / "prometheus-7b-v2.0.Q4_K_M.gguf"
        ),
        # First test the template embedded in the official GGUF.
        chat_format=None,
        response_protocol=None,
        system_prompt=(
            "You are a strict educational answer evaluator. "
            "Follow the supplied rubric and return only valid JSON "
            "matching the requested schema."
        ),
    ),
)