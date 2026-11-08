"""Frozen scalar-v3 adapter for the opt-in asynchronous delivery service.

Requires a verified reference-answer registry. Unmatched spontaneous queries are
recorded as skipped, never judged against a fabricated answer.
"""
from __future__ import annotations

import json
import os
import hashlib
import inspect
from pathlib import Path
from typing import Any, Mapping

from judge.scalar_evaluator_v3 import ScalarJudgeV3, load_frozen_rubric
from judge.runtime_contract_v1 import EvaluationRequest
from judge.runtime_service_v1 import ReferenceUnavailable
from llm.llmcli.providers.remote_http import RemoteHTTPProvider

GENERATION = {"n_ctx": 4096, "max_new_tokens": 256,
              "temperature": 0.0, "top_p": 1.0, "seed": 0, "batch_size": 1}


def exact_reference_registry(path: str | Path) -> dict[str, str]:
    """Explicit question-to-gold mapping; no fuzzy or implicit RAG reference."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Reference registry must be a JSON object mapping questions to answers")
    if not all(isinstance(q, str) and q.strip() and isinstance(a, str) and a.strip()
               for q, a in data.items()):
        raise ValueError("Reference registry entries must contain nonempty question/answer strings")
    return {q.strip(): a.strip() for q, a in data.items()}


class RuntimeScalarV3Adapter:
    def __init__(self, judge: ScalarJudgeV3, references: Mapping[str, str]):
        self.judge = judge
        self.references = dict(references)

    def __call__(self, request: EvaluationRequest) -> tuple[int, str]:
        reference = self.references.get(request.question_text.strip())
        if not reference:
            raise ReferenceUnavailable("no independently verified reference for exact question")
        if request.evaluation_dimension == "groundedness" and not request.evidence:
            raise ReferenceUnavailable("no retrieved evidence")
        context = "\n\n".join(item.text for item in request.evidence) or None
        result = self.judge.evaluate_dimension(
            dimension=request.evaluation_dimension,
            user_utterance=request.question_text,
            conversation_history=request.history,
            answer=request.delivered_answer_text,
            reference_answer=reference,
            context=context,
            teaching_style=request.teacher_style,
        )
        if result["status"] == "invalid":
            raise ValueError("scalar output invalid after protocol-v2.1 format retry")
        if result["status"] != "valid":
            raise RuntimeError("scalar model inference error")
        parsed = result["result"]
        return parsed["score"], parsed["feedback"]


def make_runtime_scalar_v3_adapter() -> RuntimeScalarV3Adapter:
    """Fail closed unless every required artifact is explicitly configured."""
    rubric = os.environ["LLMTUTOR_RUBRIC_V3_PATH"]
    reference_path = os.environ["LLMTUTOR_REFERENCE_REGISTRY_JSON"]
    server = os.environ["LLMTUTOR_JUDGE_URL"]
    spec = load_frozen_rubric(rubric)
    expected = os.environ["LLMTUTOR_JUDGE_RUBRIC_SHA256"]
    if spec.sha256 != expected:
        raise ValueError("runtime rubric SHA256 differs from frozen evaluation rubric")
    from judge.scalar_evaluator_v3 import build_scalar_prompt_v3
    prompt_digest = hashlib.sha256(inspect.getsource(build_scalar_prompt_v3).encode("utf-8")).hexdigest()
    if prompt_digest != os.environ["LLMTUTOR_JUDGE_PROMPT_SHA256"]:
        raise ValueError("runtime scalar-v3 prompt-builder SHA256 differs from configured provenance")
    references = exact_reference_registry(reference_path)
    provider = RemoteHTTPProvider(base_url=server, connect_timeout=5.0, read_timeout=180.0)
    if not provider.healthcheck():
        raise RuntimeError("FARE judge endpoint failed health check")
    judge = ScalarJudgeV3(
        provider, model_id="fare_20b_q4", quantization="Q4_K_M",
        hardware="NVIDIA A40 48 GB", output_path=Path("judge_outputs/runtime_scalar_v3_unused.jsonl"),
        dataset_id="runtime-voice-tutor", dataset_version="1.0",
        rubric_path=rubric, declared_generation_config=GENERATION,
    )
    return RuntimeScalarV3Adapter(judge, references)
