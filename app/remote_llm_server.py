from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm.llmcli.models.registry import get_model_spec
from llm.llmcli.providers.llamacpp import LlamaCppProvider


app = FastAPI(title="Remote LLM Server")

LLM = None


def build_llm(model_key: str) -> LlamaCppProvider:
    spec = get_model_spec(model_key)

    k = model_key.lower()

    temp = 0.2
    top_p = 0.9
    n_ctx = 2048

    if "qwen" in k:
        temp = 0.6
        top_p = 0.95

    elif "ministral" in k:
        temp = 0.3
        top_p = 0.9
        n_ctx = 4096

    elif "gemma" in k:
        temp = 1.0
        top_p = 0.95
        n_ctx = 4096

    return LlamaCppProvider(
        model_path=spec.gguf_path,
        chat_format=spec.chat_format,
        chat_template_kwargs=spec.chat_template_kwargs,
        system_prompt=spec.system_prompt,
        n_ctx=n_ctx,
        n_threads=12,
        n_gpu_layers=-1,
        n_batch=512,
        n_ubatch=512,
        flash_attn=True,
        offload_kqv=True,
        max_new_tokens=360,
        temperature=temp,
        top_p=top_p,
        seed=0,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
async def generate(request: Request):
    global LLM
    body = await request.json()
    prompt = (body.get("prompt") or "").strip()

    t0 = time.perf_counter()
    res = LLM.generate(prompt)
    total_s = time.perf_counter() - t0

    return JSONResponse(
        {
            "text": res.text,
            "prompt_tokens": res.prompt_tokens,
            "new_tokens": res.new_tokens,
            "total_seconds": total_s,
        }
    )


@app.post("/stream")
async def stream(request: Request):
    global LLM
    body = await request.json()
    prompt = (body.get("prompt") or "").strip()

    def event_iter():
        try:
            for piece in LLM.stream(prompt):
                yield json.dumps({"delta": piece}) + "\n"
        finally:
            yield json.dumps({"done": True}) + "\n"

    return StreamingResponse(event_iter(), media_type="application/x-ndjson")


def main():
    global LLM

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    print("[SERVER] Loading model...")
    LLM = build_llm(args.model)
    print("[SERVER] Model loaded.")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()