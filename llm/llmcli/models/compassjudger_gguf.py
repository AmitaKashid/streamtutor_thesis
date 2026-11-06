from pathlib import Path

from .registry import ModelSpec, register


ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "models"


register(
    "compassjudger_2_7b_q4",
    ModelSpec(
        backend="llamacpp",
        gguf_path=str(
            MODELS / "CompassJudger-2-7B-Instruct.Q4_K_M.gguf"
        ),
        # Use the chat template embedded in the GGUF.
        chat_format=None,
        response_protocol=None,
        system_prompt=(
            "You are a strict educational answer evaluator. "
            "Follow the supplied rubric and return only valid JSON "
            "matching the requested schema."
        ),
    ),
)