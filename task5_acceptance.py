
"""Task 5: live FARE smoke test, runtime startup, and SQLite reporting.

Run from the repository root:
    python task5_acceptance.py smoke
    python task5_acceptance.py serve
    python task5_acceptance.py report

The reference below is for engineering validation only.
It is NOT independently verified thesis ground truth.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent
RUBRIC = ROOT / "data/judge_evaluation/streamtutor_judge_dataset_v3/rubric_v3.json"
REGISTRY = ROOT / "judge_outputs/task5_engineering_reference.json"
SMOKE_DB = ROOT / "judge_outputs/task5_acceptance.sqlite"
RUNTIME_DB = ROOT / "judge_outputs/runtime_v1_spool.sqlite"

QUESTION = "What is entropy?"
REFERENCE = (
    "In information theory, entropy H(Y) = -sum_y p(y) log2 p(y) "
    "measures uncertainty in a discrete random variable. "
    "In classification, entropy measures class impurity and is zero "
    "when all examples belong to one class."
)


def configure():
    os.chdir(ROOT)

    from judge.scalar_evaluator_v3 import (
        load_frozen_rubric,
        build_scalar_prompt_v3,
    )

    rubric = load_frozen_rubric(RUBRIC)

    prompt_hash = hashlib.sha256(
        inspect.getsource(build_scalar_prompt_v3).encode("utf-8")
    ).hexdigest()

    REGISTRY.parent.mkdir(parents=True, exist_ok=True)

    if not REGISTRY.exists():
        REGISTRY.write_text(
            json.dumps(
                {
                    "What is entropy?": REFERENCE,
                    "What is entropy": REFERENCE,
                    "what is entropy?": REFERENCE,
                    "what is entropy": REFERENCE,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    os.environ.update({
        "LLMTUTOR_RUNTIME_JUDGE": "0",
        "LLMTUTOR_JUDGE_V1": "1",
        "LLMTUTOR_JUDGE_URL": "http://127.0.0.1:8004",
        "LLMTUTOR_JUDGE_V3_EVALUATOR":
            "judge.runtime_scalar_adapter_v3:make_runtime_scalar_v3_adapter",
        "LLMTUTOR_RUBRIC_V3_PATH": str(RUBRIC),
        "LLMTUTOR_REFERENCE_REGISTRY_JSON": str(REGISTRY),
        "LLMTUTOR_JUDGE_RUBRIC_SHA256": rubric.sha256,
        "LLMTUTOR_JUDGE_PROMPT_SHA256": prompt_hash,
    })

    print("[CONFIG] Rubric SHA256:", rubric.sha256)
    print("[CONFIG] Prompt SHA256:", prompt_hash)
    print("[CONFIG] Engineering reference:", REGISTRY)


def report_database(path):
    print("\n[DATABASE]", path)

    if not path.exists():
        print("Database not created")
        return

    with sqlite3.connect(path) as db:
        rows = db.execute("""
            SELECT
                status,
                reason,
                score,
                attempts,
                json_extract(payload, '$.evaluation_dimension'),
                json_extract(payload, '$.question_text'),
                json_extract(payload, '$.outcome')
            FROM evaluations
            ORDER BY created_at
        """).fetchall()

    from collections import Counter

    print("Total evaluations:", len(rows))
    print("Status counts:", dict(Counter(row[0] for row in rows)))

    for row in rows[-18:]:
        status, reason, score, attempts, dimension, question, outcome = row

        print(
            f"status={status} | dimension={dimension} | "
            f"score={score} | outcome={outcome} | "
            f"attempts={attempts} | reason={reason} | "
            f"question={question[:65]!r}"
        )


def smoke():
    configure()

    from judge.runtime_scalar_adapter_v3 import (
        make_runtime_scalar_v3_adapter,
    )
    from judge.runtime_service_v1 import RuntimeJudgeService
    from judge.runtime_bridge_v1 import (
        Candidate,
        DeliveryCoordinator,
    )

    evaluator = make_runtime_scalar_v3_adapter()
    print("[PASS] FARE adapter initialized")

    service = RuntimeJudgeService(
        SMOKE_DB,
        evaluator,
        max_attempts=1,
    )

    bridge = DeliveryCoordinator(
        service,
        rubric_sha256=os.environ["LLMTUTOR_JUDGE_RUBRIC_SHA256"],
        judge_prompt_sha256=os.environ["LLMTUTOR_JUDGE_PROMPT_SHA256"],
        dimensions=("correctness",),
    )

    def make_candidate(question):
        return Candidate(
            turn_id=str(uuid.uuid4()),
            conversation_id="task5-engineering-smoke",
            question=question,
            answer="Entropy measures uncertainty in class labels.",
            history=(),
            rag_context="",
            tutor_model="llama3_8b_q4",
            tutor_prompt_sha256="a" * 64,
            teaching_style="socratic",
            generation_completed=True,
        )

    try:
        # 1. Completed delivery
        completed = make_candidate(QUESTION)
        assert bridge.candidate_ready(completed)

        admitted = bridge.acknowledge(
            completed.turn_id,
            delivered=True,
        )

        assert len(admitted) == 1
        assert admitted[0][1] in ("queued", "running", "valid")

        # 2. Duplicate acknowledgment
        duplicate = bridge.acknowledge(
            completed.turn_id,
            delivered=True,
        )
        assert duplicate == []

        print("[PASS] Delivered turn admitted once")

        # 3. Interrupted delivery
        interrupted = make_candidate(QUESTION)
        assert bridge.candidate_ready(interrupted)

        skipped = bridge.acknowledge(
            interrupted.turn_id,
            delivered=False,
        )

        assert len(skipped) == 1
        assert skipped[0][1] == "skipped"

        print("[PASS] Interrupted turn skipped")

        # 4. Missing reference
        missing = make_candidate(
            "Question without a registered reference"
        )
        assert bridge.candidate_ready(missing)

        missing_id = bridge.acknowledge(
            missing.turn_id,
            delivered=True,
        )[0][0]

        completed_id = admitted[0][0]

        # Wait for actual asynchronous inference.
        deadline = time.monotonic() + 240

        while time.monotonic() < deadline:
            completed_result = service.read(completed_id)
            missing_result = service.read(missing_id)

            finished = (
                completed_result["status"] not in ("queued", "running")
                and missing_result["status"] not in ("queued", "running")
            )

            if finished:
                break

            time.sleep(1)
        else:
            raise RuntimeError("FARE inference timed out")

        print("[FARE RESULT]", completed_result)
        print("[MISSING REFERENCE]", missing_result)

        assert completed_result["status"] == "valid", completed_result
        assert isinstance(completed_result["score"], int)
        assert 1 <= completed_result["score"] <= 5

        assert missing_result["status"] == "skipped"
        assert "reference_unavailable" in (
            missing_result["reason"] or ""
        )

        print("[PASS] Real FARE scoring and SQLite persistence")
        print("[PASS] Missing-reference handling")

    finally:
        service.close()

    report_database(SMOKE_DB)


def serve():
    configure()

    from judge.runtime_scalar_adapter_v3 import (
        make_runtime_scalar_v3_adapter,
    )

    # Fail before starting FastAPI if FARE or hashes are invalid.
    make_runtime_scalar_v3_adapter()

    print("[PASS] FARE adapter ready")
    print("[STARTING] Delivery-gated FastAPI")

    import uvicorn

    uvicorn.run(
        "frontend_api:app",
        host="127.0.0.1",
        port=8001,
        reload=False,
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: task5_acceptance.py smoke|serve|report")

    command = sys.argv[1]

    if command == "smoke":
        smoke()
    elif command == "serve":
        serve()
    elif command == "report":
        report_database(SMOKE_DB)
        report_database(RUNTIME_DB)
    else:
        sys.exit(f"Unknown command: {command}")
