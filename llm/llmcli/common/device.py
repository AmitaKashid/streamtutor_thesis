import torch

def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device

def resolve_dtype(dtype: str, device: str):
    # Safe defaults:
    # - CPU: float32
    # - CUDA: float16 (or bfloat16 if you prefer and your GPU supports it)
    if dtype == "auto":
        return torch.float32 if device == "cpu" else torch.float16
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]
