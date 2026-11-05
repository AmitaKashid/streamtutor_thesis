from pathlib import Path
from .registry import register, ModelSpec

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "models"

register(
    "gemma4_26b_q3",
    ModelSpec(
        backend="llamacpp",
        gguf_path=str(MODELS / "gemma-4-26B-A4B-it-UD-Q3_K_XL.gguf"),
        chat_format=None,  # use embedded GGUF chat template if available
        system_prompt=(
            "You are a concise AI tutor for data mining. "
            "Use the provided course context when available. "
            "Answer clearly and briefly. "
            "Do not expose hidden reasoning."
        ),
    ),
)