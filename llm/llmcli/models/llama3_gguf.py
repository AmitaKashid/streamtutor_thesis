from pathlib import Path
from .registry import register, ModelSpec
import os

# File path:
#   llmtutor/llm/llmcli/models/llama3_gguf.py
# Repo root should be:
#   llmtutor/
ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "models"

register(
    "llama3_8b_q4",
    ModelSpec(
        backend="llamacpp",
        gguf_path=str(MODELS / "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"),
        #gguf_path=f"/tmp/{os.environ['USER']}/llmtutor_models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        chat_format="llama-3",
        system_prompt="You are a helpful assistant.",
    ),
)

register(
    "llama3_8b_q5",
    ModelSpec(
        backend="llamacpp",
        gguf_path=str(MODELS / "Meta-Llama-3-8B-Instruct.Q5_K_M.gguf"),
        #gguf_path=f"/tmp/{os.environ['USER']}/llmtutor_models/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf",
        chat_format="llama-3",
        system_prompt="You are a helpful assistant.",
    ),
)

register(
    "llama3_8b_q6",
    ModelSpec(
        backend="llamacpp",
        gguf_path=str(MODELS / "Meta-Llama-3-8B-Instruct.Q6_K.gguf"),
        chat_format="llama-3",
        system_prompt="You are a helpful assistant.",
    ),
)
