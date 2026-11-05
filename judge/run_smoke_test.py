from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from judge.judge import LocalLLMJudge
from llm.llmcli.providers.remote_http import RemoteHTTPProvider


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "judge" / "smoke_cases_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen StreamTutor judge smoke set."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8004")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--model-id", default="glm4_32b_0414_q4")
    parser.add_argument("--backend", default="llama.cpp via RemoteHTTPProvider")
    parser.add_argument("--quantization", default="Q4_K_M")
    parser.add_argument("--hardware", default="NVIDIA A30 24 GB")
    parser.add_argument("--read-timeout", type=float, default=180.0)
    return parser.parse_args()


def score(result: dict[str, Any], dimension: str) -> int | None:
    value = result["scores"][dimension]
    if isinstance(value, dict):
        return value["score"]
    return value


def check_expectations(
    record: dict[str, Any], expect: dict[str, Any]
) -> list[str]:
    failures: list[str] = []
    judge = record["judge"]
    if judge["status"] != "valid":
        return [f"judge_status={judge['status']}"]

    result = judge["result"]
    verdict = judge["derived_verdict"]
    if "verdict_in" in expect and verdict not in expect["verdict_in"]:
        failures.append(f"verdict={verdict}, expected one of {expect['verdict_in']}")

    dimensions = [
        "correctness",
        "completeness",
        "clarity",
        "pedagogy",
        "learner_appropriateness",
        "groundedness",
    ]
    for dimension in dimensions:
        value = score(result, dimension)
        minimum = expect.get(f"{dimension}_min")
        maximum = expect.get(f"{dimension}_max")
        if minimum is not None and (value is None or value < minimum):
            failures.append(f"{dimension}={value}, expected >= {minimum}")
        if maximum is not None and (value is None or value > maximum):
            failures.append(f"{dimension}={value}, expected <= {maximum}")

    retrieval = result["retrieval_relevance"]
    for suffix, operator in (("min", "minimum"), ("max", "maximum")):
        threshold = expect.get(f"retrieval_relevance_{suffix}")
        value = retrieval["score"]
        if threshold is None:
            continue
        failed = value is None or (
            value < threshold if operator == "minimum" else value > threshold
        )
        if failed:
            symbol = ">=" if operator == "minimum" else "<="
            failures.append(
                f"retrieval_relevance={value}, expected {symbol} {threshold}"
            )

    applicability_checks = {
        "groundedness_applicable": result["scores"]["groundedness"]["applicable"],
        "retrieval_relevance_applicable": retrieval["applicable"],
    }
    for key, actual in applicability_checks.items():
        if key in expect and actual is not expect[key]:
            failures.append(f"{key}={actual}, expected {expect[key]}")

    error_types = {item["type"] for item in result["critical_errors"]}
    required_any = set(expect.get("critical_error_any_of", []))
    if required_any and not error_types.intersection(required_any):
        failures.append(
            f"critical_errors={sorted(error_types)}, expected any of "
            f"{sorted(required_any)}"
        )
    forbidden = set(expect.get("critical_error_none_of", []))
    present_forbidden = error_types.intersection(forbidden)
    if present_forbidden:
        failures.append(f"forbidden critical errors: {sorted(present_forbidden)}")

    if expect.get("appropriateness_assumption_required"):
        if not result["appropriateness_assumption"]:
            failures.append("appropriateness_assumption is required")

    return failures


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * quantile)))
    return round(ordered[index], 4)


def main() -> int:
    args = parse_args()
    dataset = json.loads(args.cases.read_text(encoding="utf-8"))
    items = dataset.get("items", [])
    if len(items) != 20:
        raise ValueError(f"Expected exactly 20 smoke cases, found {len(items)}")
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Smoke case IDs must be unique")

    provider = RemoteHTTPProvider(
        base_url=args.base_url,
        connect_timeout=5.0,
        read_timeout=args.read_timeout,
    )
    if not provider.healthcheck():
        print(f"ERROR: judge server is not reachable at {args.base_url}", file=sys.stderr)
        return 2

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / "judge_outputs" / "smoke_v1" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    records_path = output_dir / "records.jsonl"
    report_path = output_dir / "report.json"

    judge = LocalLLMJudge(
        provider,
        output_path=records_path,
        model_id=args.model_id,
        backend=args.backend,
        quantization=args.quantization,
        hardware=args.hardware,
    )

    outcomes: list[dict[str, Any]] = []
    for index, item in enumerate(items, 1):
        print(f"[{index:02d}/20] {item['id']} ({item['category']})", flush=True)
        try:
            record = judge.judge_and_log(
                turn_id=f"smoke-v1-{item['id']}",
                question_id=item["id"],
                question=item["question"],
                answer=item["answer"],
                teaching_style=item.get("teaching_style", "direct"),
                context=item.get("context", ""),
                retrieved_ids=item.get("retrieved_ids", []),
                interruption_text=item.get("interruption_text"),
                interrupted_chunks=item.get("interrupted_chunks", []),
                learner_level=item.get("learner_level"),
                reference_answers=item.get("reference_answers", []),
                response_status=item.get("response_status", "complete"),
                retrieval_status=(
                    "provided" if item.get("context", "").strip() else "not_provided"
                ),
                metadata={
                    "evaluation_mode": "judge_smoke_test",
                    "dataset_id": dataset["dataset_id"],
                    "dataset_version": dataset["dataset_version"],
                    "category": item["category"],
                    "run_id": run_id,
                },
            )
            expectation_failures = check_expectations(record, item["expect"])
            outcomes.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "judge_status": record["judge"]["status"],
                    "derived_verdict": record["judge"]["derived_verdict"],
                    "retry_used": record["judge"]["retry_used"],
                    "expectation_pass": not expectation_failures,
                    "expectation_failures": expectation_failures,
                    "attempts": record["judge"]["attempts"],
                }
            )
        except Exception as exc:
            outcomes.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "judge_status": "runner_error",
                    "derived_verdict": None,
                    "retry_used": False,
                    "expectation_pass": False,
                    "expectation_failures": [f"{type(exc).__name__}: {exc}"],
                    "attempts": [],
                }
            )

    valid_count = sum(item["judge_status"] == "valid" for item in outcomes)
    expectation_count = sum(item["expectation_pass"] for item in outcomes)
    retry_count = sum(item["retry_used"] for item in outcomes)
    latencies = [
        float(attempt["metrics"]["wall_seconds"])
        for outcome in outcomes
        for attempt in outcome["attempts"]
        if attempt.get("metrics", {}).get("wall_seconds") is not None
    ]
    report = {
        "run_id": run_id,
        "dataset_id": dataset["dataset_id"],
        "dataset_version": dataset["dataset_version"],
        "judge_configuration": {
            "base_url": args.base_url,
            "model_id": args.model_id,
            "backend": args.backend,
            "quantization": args.quantization,
            "hardware": args.hardware,
        },
        "summary": {
            "total_cases": len(outcomes),
            "valid_json_cases": valid_count,
            "valid_json_rate": valid_count / len(outcomes),
            "day1_schema_gate_pass": valid_count >= 19,
            "semantic_expectation_cases": expectation_count,
            "semantic_expectation_rate": expectation_count / len(outcomes),
            "retry_cases": retry_count,
            "attempt_latency_seconds": {
                "mean": round(statistics.mean(latencies), 4) if latencies else None,
                "p50": percentile(latencies, 0.50),
                "p95": percentile(latencies, 0.95),
                "maximum": round(max(latencies), 4) if latencies else None,
            },
        },
        "outcomes": outcomes,
        "interpretation": (
            "The schema gate measures integration reliability. Semantic expectation "
            "agreement is diagnostic and must not be used as the final model-selection "
            "result without independent human labels."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report["summary"], indent=2))
    print(f"Records: {records_path}")
    print(f"Report:  {report_path}")
    return 0 if valid_count >= 19 else 1


if __name__ == "__main__":
    raise SystemExit(main())
