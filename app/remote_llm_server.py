from __future__ import annotations

import argparse
import json
import sys
import time
import threading
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm.llmcli.models.registry import get_model_spec
from llm.llmcli.providers.llamacpp import LlamaCppProvider


app = FastAPI(title="Remote LLM Server")

LLM = None
LLM_LOCK = threading.Lock()


def build_llm(
    model_key: str,
    *,
    n_ctx_override: int | None = None,
    max_new_tokens_override: int | None = None,
    temperature_override: float | None = None,
    top_p_override: float | None = None,
    disable_system_prompt: bool = False,
) -> LlamaCppProvider:
    spec = get_model_spec(model_key)

    k = model_key.lower()

    temp = 0.2
    top_p = 0.9
    n_ctx = 2048
    max_new_tokens = 360

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

    elif k in {
        "compassjudger_2_7b_q4",
        "prometheus_7b_v2_q4",
    }:
        # Frozen judge-selection configuration.
        temp = 0.0
        top_p = 1.0
        n_ctx = 4096

    elif "glm" in k:
        n_ctx = 4096

    if n_ctx_override is not None:
        if n_ctx_override <= 0:
            raise ValueError("n_ctx_override must be positive")
        n_ctx = n_ctx_override

    if max_new_tokens_override is not None:
        if max_new_tokens_override <= 0:
            raise ValueError(
                "max_new_tokens_override must be positive"
            )
        max_new_tokens = max_new_tokens_override

    if temperature_override is not None:
        if temperature_override < 0:
            raise ValueError(
                "temperature_override must be non-negative"
            )
        temp = temperature_override

    if top_p_override is not None:
        if not 0 < top_p_override <= 1:
            raise ValueError(
                "top_p_override must be greater than 0 and at most 1"
            )
        top_p = top_p_override

    system_prompt = (
        None if disable_system_prompt else spec.system_prompt
    )

    return LlamaCppProvider(
        model_path=spec.gguf_path,
        chat_format=spec.chat_format,
        chat_template_kwargs=spec.chat_template_kwargs,
        response_protocol=spec.response_protocol,
        system_prompt=system_prompt,
        n_ctx=n_ctx,
        n_threads=12,
        n_gpu_layers=-1,
        n_batch=512,
        n_ubatch=512,
        flash_attn=True,
        offload_kqv=True,
        max_new_tokens=max_new_tokens,
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
    def run_locked():
        with LLM_LOCK:
            return LLM.generate(prompt)

    res = await run_in_threadpool(run_locked)
    total_s = time.perf_counter() - t0

    return JSONResponse(
        {
            "text": res.text,
            "raw_text": res.raw_text,
            "response_protocol": res.response_protocol,
            "protocol_error": res.protocol_error,
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
        # Native counters are process-wide for the model context. Serialize the
        # entire request so another generation cannot mix its timing values.
        with LLM_LOCK:
            try:
                for event in LLM.stream_events(prompt):
                    yield json.dumps(event) + "\n"
            except Exception as exc:
                yield json.dumps({"error": {"type": type(exc).__name__}}) + "\n"

    return StreamingResponse(event_iter(), media_type="application/x-ndjson")


def main():
    global LLM

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--n-ctx",
        type=int,
        default=None,
        help="Optional runtime context-size override",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional maximum generation-length override",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional generation-temperature override",
    )
    p.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Optional nucleus-sampling override",
    )
    p.add_argument(
        "--disable-system-prompt",
        action="store_true",
        help=(
            "Disable the model registry system prompt so the complete "
            "experimental instructions can be supplied in the user prompt"
        ),
    )
    args = p.parse_args()

    print("[SERVER] Loading model...")
    LLM = build_llm(
        args.model,
        n_ctx_override=args.n_ctx,
        max_new_tokens_override=args.max_new_tokens,
        temperature_override=args.temperature,
        top_p_override=args.top_p,
        disable_system_prompt=args.disable_system_prompt,
    )
    print("[SERVER] Model loaded.")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
