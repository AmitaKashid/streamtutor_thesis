from pathlib import Path
from .registry import register, ModelSpec

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "models"

register(
    "ministral_14b_q5",
    ModelSpec(
        backend="llamacpp",
        gguf_path=str(
            MODELS / "Ministral-3-14B-Instruct-2512-Q5_K_M.gguf"
        ),
        chat_format="mistral",
        system_prompt=(
            "You are a concise AI tutor for data mining. "
            "Use provided context. "
            "Teach clearly and briefly."
        ),
    ),
)