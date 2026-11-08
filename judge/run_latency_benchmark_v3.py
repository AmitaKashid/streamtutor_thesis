"""Run the frozen v3 scalar latency workload against one preconfigured server."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

from judge.run_judge_benchmark_v3 import (
    DEFAULT_DATASET_DIR,
    _canonical_hash,
    _git_commit,
    _percentile,
    _read_jsonl,
    _utc_now,
    _validate_run_id,
    _write_json_atomic,
    build_evaluation_tasks,
    verify_dataset,
)
from judge.scalar_contract_v2 import SCALAR_PROTOCOL_VERSION
from judge.scalar_evaluator_v3 import (
    ANSWER_DIMENSIONS,
    PROMPT_VERSION,
    RETRIEVAL_DIMENSION,
    ScalarJudgeV3,
    build_scalar_prompt_v3,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "judge_outputs" / "latency_v3"
MATRIX = ROOT / "docs" / "judge_evaluation" / "execution_matrix_v3.csv"
GENERATION = {
    "n_ctx": 4096,
    "max_new_tokens": 256,
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 0,
    "concurrent_requests": 1,
    "n_threads": 12,
    "n_gpu_layers": -1,
    "n_batch": 512,
    "n_ubatch": 512,
    "flash_attn": True,
    "offload_kqv": True,
    "system_prompt": "disabled",
}
WARMUPS = 5
REPETITIONS = 3
HARDWARE = "NVIDIA A30 24 GB"
TIMEOUT_TYPES = {"Timeout", "TimeoutError", "ReadTimeout", "ConnectTimeout"}


def build_latency_plan(tasks):
    """Select sorted IDs within each dimension; reserve five other warm-ups."""
    ordered = sorted(tasks, key=lambda task: task.evaluation_id)
    if len({task.evaluation_id for task in ordered}) != len(ordered):
        raise ValueError("Duplicate calibration evaluation ID")
    selected = []
    for dimension in (*ANSWER_DIMENSIONS, RETRIEVAL_DIMENSION):
        count = 2 if dimension == RETRIEVAL_DIMENSION else 4
        group = [task for task in ordered if task.dimension == dimension]
        if len(group) < count:
            raise ValueError(f"Not enough calibration tasks for {dimension}")
        selected.extend(group[:count])
    selected.sort(key=lambda task: task.evaluation_id)
    used = {task.evaluation_id for task in selected}
    warmups = [task for task in ordered if task.evaluation_id not in used][:WARMUPS]
    if len(warmups) != WARMUPS:
        raise ValueError("Not enough disjoint calibration warm-up tasks")
    return warmups, selected


def _positive(value):
    if type(value) in {int, float} and math.isfinite(value) and value > 0:
        return value
    return None


def _distribution(values):
    return {
        "count": len(values),
        "minimum": min(values, default=None),
        "p50": statistics.median(values) if values else None,
        "p95": _percentile(values, 0.95),
        "maximum": max(values, default=None),
    }


def _input(task):
    fields = task.evaluator_kwargs()
    return {
        key: value
        for key, value in fields.items()
        if key not in {"evaluation_id", "question_id", "response_id", "case_metadata"}
    }


def _plan_entries(warmups, tasks):
    warmup_entries = [
        {"phase": "warmup", "repetition": 0, "task_id": task.evaluation_id}
        for task in warmups
    ]
    measured_entries = [
        {"phase": "measured", "repetition": repeat, "task_id": task.evaluation_id}
        for repeat in range(1, REPETITIONS + 1)
        for task in tasks
    ]
    return warmup_entries + measured_entries


def _write_attempts_csv(path, records):
    columns = (
        "sequence", "phase", "repetition", "task_id", "dimension", "attempt",
        "evaluation_status", "attempt_valid", "evaluation_wall_seconds",
        "wall_seconds", "server_total_seconds", "prompt_tokens", "new_tokens",
        "tokens_per_server_second", "generation_error_type", "validation_error",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            for attempt in record["judge"]["attempts"]:
                metrics = attempt["metrics"]
                error = attempt["generation_error"] or {}
                writer.writerow({
                    **{key: record[key] for key in columns[:5]},
                    "attempt": attempt["attempt"],
                    "evaluation_status": record["judge"]["status"],
                    "attempt_valid": attempt["parsed_result"] is not None,
                    "evaluation_wall_seconds": record["evaluation_wall_seconds"],
                    **{key: metrics.get(key) for key in (
                        "wall_seconds", "server_total_seconds", "prompt_tokens",
                        "new_tokens",
                    )},
                    "tokens_per_server_second": metrics.get("tokens_per_second"),
                    "generation_error_type": error.get("type"),
                    "validation_error": attempt["validation_error"],
                })


def summarize(records, expected_count):
    measured = [record for record in records if record["phase"] == "measured"]
    judgments = [record["judge"] for record in measured]
    attempts = [a for j in judgments for a in j["attempts"]]
    successful_requests = [a for a in attempts if a["generation_error"] is None]
    first_valid = sum(j["attempts"][0]["parsed_result"] is not None for j in judgments)
    final_valid = sum(j["status"] == "valid" for j in judgments)
    rates = [
        value for attempt in successful_requests
        if (value := _positive(attempt["metrics"].get("tokens_per_second"))) is not None
    ]
    return {
        "planned_measured_evaluations": expected_count,
        "measured_evaluations": len(measured),
        "measurement_complete": len(measured) == expected_count,
        "warmups_recorded": len(records) - len(measured),
        "attempt_count": len(attempts),
        "retries": sum(j["retry_used"] for j in judgments),
        "first_attempt_valid_count": first_valid,
        "first_attempt_valid_rate": first_valid / len(measured) if measured else None,
        "final_valid_count": final_valid,
        "final_valid_rate": final_valid / len(measured) if measured else None,
        "invalid_or_error_evaluations": len(measured) - final_valid,
        "failed_requests": len(attempts) - len(successful_requests),
        "timeouts": sum(
            (a["generation_error"] or {}).get("type") in TIMEOUT_TYPES for a in attempts
        ),
        "attempt_wall_seconds": _distribution([
            a["metrics"]["wall_seconds"] for a in attempts
        ]),
        "evaluation_wall_seconds_including_retry": _distribution([
            r["evaluation_wall_seconds"] for r in measured
        ]),
        "server_completion_seconds": _distribution([
            value for a in successful_requests
            if (value := _positive(a["metrics"].get("server_total_seconds"))) is not None
        ]),
        "median_tokens_per_server_second": statistics.median(rates) if rates else None,
        "median_decode_tokens_per_second": None,
        "ttft_seconds": None,
        "deployment_gate_evaluated": False,
        "measurement_notes": (
            "The non-streaming API does not expose TTFT or decode-only timing. "
            "Tokens per server second includes prefill. GPU memory, offload and "
            "actual server settings must be verified from separate evidence."
        ),
    }


def _execute(evaluator, tasks_by_id, entries, path):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for sequence, entry in enumerate(entries, start=1):
            task = tasks_by_id[entry["task_id"]]
            inputs = _input(task)
            started_utc = _utc_now()
            started = time.perf_counter()
            if entry["phase"] == "warmup":
                prompt = build_scalar_prompt_v3(rubric=evaluator.rubric, **inputs)
                attempt, parsed = evaluator._run_attempt(prompt=prompt, attempt_number=1)
                status = (
                    "error" if attempt["generation_error"] is not None
                    else "valid" if parsed is not None else "invalid"
                )
                judgment = {"status": status, "retry_used": False, "attempts": [attempt]}
            else:
                judgment = evaluator.evaluate_dimension(**inputs)
            record = {
                **entry,
                "sequence": sequence,
                "started_utc": started_utc,
                "dimension": task.dimension,
                "evaluation_wall_seconds": time.perf_counter() - started,
                "judge": judgment,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if sequence == WARMUPS or (
                sequence > WARMUPS and (sequence - WARMUPS) % 30 == 0
            ):
                print(f"Recorded {sequence}/{len(entries)} evaluations (including warm-ups)", flush=True)
            if entry["phase"] == "warmup" and judgment["status"] == "error":
                raise RuntimeError(f"Warm-up request failed: {entry['task_id']}")


def _make_provider(base_url):
    from llm.llmcli.providers.remote_http import RemoteHTTPProvider

    return RemoteHTTPProvider(base_url=base_url, connect_timeout=5.0, read_timeout=180.0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-url")
    parser.add_argument("--slurm-job-id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.dry_run and not (args.base_url and args.slurm_job_id):
        parser.error("a live run requires --base-url and --slurm-job-id")
    run_id = _validate_run_id(args.run_id)
    with MATRIX.open(encoding="utf-8", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["stage"] == "comparative_latency" and row["model_id"] == args.model_id
        ]
    if len(rows) != 1:
        parser.error("model must have exactly one comparative_latency matrix row")
    expected = {
        "protocol": "scalar_v3", "split": "latency30", "hardware": HARDWARE,
        "warmups": str(WARMUPS), "repetitions": str(REPETITIONS), "task_count": "90",
        **{key: str(GENERATION[key]) for key in (
            "n_ctx", "max_new_tokens", "temperature", "top_p", "seed", "system_prompt",
        )},
    }
    if any(rows[0].get(key) != value for key, value in expected.items()):
        raise ValueError("Execution matrix does not match this runner's frozen workload")
    dataset = verify_dataset(args.dataset_dir)
    root = dataset["root"]
    warmups, tasks = build_latency_plan(build_evaluation_tasks(
        questions=_read_jsonl(root / "questions.jsonl"),
        responses=_read_jsonl(root / "responses.jsonl"),
        split="calibration",
    ))
    entries = _plan_entries(warmups, tasks)
    expected_count = len(tasks) * REPETITIONS
    if expected_count != int(rows[0]["task_count"]):
        raise ValueError("Unexpected measured evaluation count")
    plan = {
        "unique_measured_tasks": len(tasks),
        "measured_evaluations": expected_count,
        "warmup_requests": len(warmups),
        "entries": entries,
        "task_plan_sha256": _canonical_hash(entries),
    }
    run_dir = args.output_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json_atomic(run_dir / "plan.json", plan)
    manifest = {
        "protocol": "judge_execution_protocol_v3",
        "prompt_version": PROMPT_VERSION,
        "scalar_protocol_version": SCALAR_PROTOCOL_VERSION,
        "run_id": run_id,
        "git_commit": _git_commit(),
        "dataset_id": dataset["dataset_id"],
        "dataset_version": dataset["dataset_version"],
        "dataset_checksums_sha256": dataset["checksums_sha256"],
        "rubric_sha256": dataset["checked_files"]["rubric_v3.json"],
        "task_plan_sha256": plan["task_plan_sha256"],
        "model_id": args.model_id,
        "base_url": args.base_url,
        "slurm_job_id": args.slurm_job_id,
        "required_hardware": HARDWARE,
        "required_backend": "llama-cpp-python 0.3.20",
        "required_quantization": "Q4_K_M",
        "required_server_configuration": GENERATION,
        "runtime_configuration_verified": False,
        "created_utc": _utc_now(),
        "status": "dry_run_complete" if args.dry_run else "running",
    }
    manifest_path = run_dir / "manifest.json"
    _write_json_atomic(manifest_path, manifest)
    print(json.dumps({key: value for key, value in plan.items() if key != "entries"}))
    print(f"Run directory: {run_dir}", flush=True)
    if args.dry_run:
        return 0

    records_path = run_dir / "evaluations.jsonl"
    code = 0
    try:
        provider = _make_provider(args.base_url)
        if not provider.healthcheck():
            raise RuntimeError(f"Judge healthcheck failed: {args.base_url}")
        evaluator = ScalarJudgeV3(
            provider, model_id=args.model_id, quantization="Q4_K_M", hardware=HARDWARE,
            output_path=records_path, dataset_id=dataset["dataset_id"],
            dataset_version=dataset["dataset_version"], rubric_path=root / "rubric_v3.json",
            declared_generation_config=GENERATION,
        )
        _execute(evaluator, {t.evaluation_id: t for t in warmups + tasks}, entries, records_path)
        manifest["status"] = "completed"
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        code = 130
    except Exception as exc:
        manifest["status"] = "runner_error"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        print(manifest["error"])
        code = 2
    finally:
        manifest["finished_utc"] = _utc_now()
        _write_json_atomic(manifest_path, manifest)
    records = _read_jsonl(records_path) if records_path.exists() else []
    _write_attempts_csv(run_dir / "attempts.csv", records)
    _write_json_atomic(run_dir / "summary.json", summarize(records, expected_count))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
