import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="model key, e.g. llama3, qwen2_5, generic")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", default="llamacpp", choices=["llamacpp", "hf"])
    return p.parse_args()
