from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import subprocess
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from judge.scalar_evaluator_v3 import (
    ANSWER_DIMENSIONS,
    RETRIEVAL_DIMENSION,
    ScalarJudgeV3,
    _validate_history,
    load_frozen_rubric,
)


RUNNER_VERSION = "3.0"
DEFAULT_DATASET_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "judge_evaluation"
    / "streamtutor_judge_dataset_v3"
)
DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "judge_outputs"
    / "benchmark_v3"
)
FORBIDDEN_LABEL_KEYS = frozenset(
    {
        "provisional_annotation",
        "controlled_quality_level",
        "degradation_profile",
    }
)
SAFE_METADATA_KEYS = (
    "conversation_id",
    "turn_index",
    "split",
    "course_unit",
    "question_type",
    "question_family",
    "context_dependency",
    "speech_features",
    "retrieval_condition",
)
SKIP_GROUNDEDNESS_CONDITIONS = frozenset({"missing", "irrelevant"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {path} line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected object in {path} line {line_number}"
                )
            rows.append(row)
    return rows


def verify_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    root = Path(dataset_dir).resolve()
    checksum_path = root / "checksums.sha256"
    if not checksum_path.is_file():
        raise ValueError(f"Missing checksum manifest: {checksum_path}")

    checked: dict[str, str] = {}
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(
                f"Malformed checksum line {line_number}: {line!r}"
            )
        expected, relative_name = parts
        candidate = (root / relative_name.strip()).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Checksum path escapes dataset directory: {relative_name}"
            ) from exc
        if not candidate.is_file():
            raise ValueError(f"Dataset file is missing: {relative_name}")
        actual = _sha256(candidate)
        if actual.lower() != expected.lower():
            raise ValueError(
                f"Checksum mismatch for {relative_name}: "
                f"expected {expected}, got {actual}"
            )
        checked[relative_name.strip()] = actual

    required = {
        "questions.jsonl",
        "responses.jsonl",
        "rubric_v3.json",
        "source_manifest.json",
        "validation_report.json",
    }
    missing_from_manifest = sorted(required.difference(checked))
    if missing_from_manifest:
        raise ValueError(
            "Checksum manifest omits required files: "
            + ", ".join(missing_from_manifest)
        )

    validation = _read_json(root / "validation_report.json")
    if validation.get("valid") is not True:
        raise ValueError("Frozen dataset validation report is not valid")
    source_manifest = _read_json(root / "source_manifest.json")
    if (
        validation.get("dataset_id") != source_manifest.get("dataset_id")
        or validation.get("dataset_version")
        != source_manifest.get("dataset_version")
    ):
        raise ValueError(
            "Dataset identity differs between validation and source manifests"
        )

    return {
        "root": root,
        "dataset_id": source_manifest["dataset_id"],
        "dataset_version": source_manifest["dataset_version"],
        "checksums_sha256": _sha256(checksum_path),
        "checked_files": checked,
    }


def _assert_no_label_keys(value: object, *, location: str) -> None:
    if isinstance(value, Mapping):
        present = FORBIDDEN_LABEL_KEYS.intersection(value)
        if present:
            raise ValueError(
                f"Label leakage at {location}: {sorted(present)}"
            )
        for key, child in value.items():
            _assert_no_label_keys(
                child,
                location=f"{location}.{key}",
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_label_keys(
                child,
                location=f"{location}[{index}]",
            )


def _required_string(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing non-empty string field: {key}")
    return value


def _task_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    retrieval = row.get("retrieval")
    condition = (
        retrieval.get("condition")
        if isinstance(retrieval, Mapping)
        else None
    )
    metadata = {
        "conversation_id": row.get("conversation_id"),
        "turn_index": row.get("turn_index"),
        "split": row.get("split"),
        "course_unit": row.get("course_unit"),
        "question_type": row.get("question_type"),
        "question_family": row.get("question_family"),
        "context_dependency": row.get("context_dependency"),
        "speech_features": row.get("speech_features"),
        "retrieval_condition": condition,
    }
    return {key: metadata[key] for key in SAFE_METADATA_KEYS}


def _common_inputs(row: Mapping[str, Any]) -> dict[str, Any]:
    retrieval = row.get("retrieval")
    if not isinstance(retrieval, Mapping):
        raise ValueError("retrieval must be an object")
    context = retrieval.get("context")
    if context is not None and not isinstance(context, str):
        raise ValueError("retrieval.context must be a string or null")
    history = row.get("conversation_history")
    if not isinstance(history, list):
        raise ValueError("conversation_history must be an array")
    return {
        "user_utterance": _required_string(row, "user_utterance"),
        "conversation_history": history,
        "reference_answer": _required_string(row, "reference_answer"),
        "context": context,
        "learner_level": row.get("learner_level"),
        "teaching_style": row.get("teaching_style"),
    }


@dataclass(frozen=True)
class EvaluationTask:
    evaluation_id: str
    question_id: str
    response_id: str | None
    dimension: str
    answer: str | None
    common_inputs: Mapping[str, Any]
    case_metadata: Mapping[str, Any]

    def evaluator_kwargs(self) -> dict[str, Any]:
        payload = {
            "evaluation_id": self.evaluation_id,
            "question_id": self.question_id,
            "response_id": self.response_id,
            "dimension": self.dimension,
            "answer": self.answer,
            "case_metadata": dict(self.case_metadata),
            **dict(self.common_inputs),
        }
        _assert_no_label_keys(payload, location=self.evaluation_id)
        return payload


def build_evaluation_tasks(
    *,
    questions: Sequence[Mapping[str, Any]],
    responses: Sequence[Mapping[str, Any]],
    split: str,
    max_responses: int | None = None,
) -> list[EvaluationTask]:
    if split == "selection" and max_responses is not None:
        raise ValueError(
            "--max-responses is forbidden for the selection split"
        )
    if max_responses is not None and max_responses <= 0:
        raise ValueError("max_responses must be positive")

    selected_questions = [
        row for row in questions if row.get("split") == split
    ]
    selected_responses = [
        row for row in responses if row.get("split") == split
    ]
    selected_questions.sort(key=lambda row: _required_string(row, "question_id"))
    selected_responses.sort(key=lambda row: _required_string(row, "response_id"))
    if max_responses is not None:
        selected_responses = selected_responses[:max_responses]

    question_by_id: dict[str, Mapping[str, Any]] = {}
    for row in selected_questions:
        question_id = _required_string(row, "question_id")
        if question_id in question_by_id:
            raise ValueError(f"Duplicate question_id: {question_id}")
        question_by_id[question_id] = row

    tasks: list[EvaluationTask] = []
    response_question_ids: set[str] = set()
    for row in selected_responses:
        _assert_no_label_keys(
            {
                key: value
                for key, value in row.items()
                if key not in FORBIDDEN_LABEL_KEYS
            },
            location="response_input",
        )
        response_id = _required_string(row, "response_id")
        question_id = _required_string(row, "question_id")
        if question_id not in question_by_id:
            raise ValueError(
                f"Response {response_id} has no question in split {split}"
            )
        response_question_ids.add(question_id)
        answer = row.get("candidate_response")
        if not isinstance(answer, str):
            raise ValueError(
                f"candidate_response must be a string: {response_id}"
            )
        common = _common_inputs(row)
        metadata = _task_metadata(row)
        condition = metadata["retrieval_condition"]
        for dimension in ANSWER_DIMENSIONS:
            if (
                dimension == "groundedness"
                and condition in SKIP_GROUNDEDNESS_CONDITIONS
            ):
                continue
            tasks.append(
                EvaluationTask(
                    evaluation_id=f"{response_id}::{dimension}",
                    question_id=question_id,
                    response_id=response_id,
                    dimension=dimension,
                    answer=answer,
                    common_inputs=common,
                    case_metadata=metadata,
                )
            )

    # Retrieval is a property of the question/context pair, not of the three
    # answer variants. Evaluate it once for each represented question.
    for question_id in sorted(response_question_ids):
        row = question_by_id[question_id]
        tasks.append(
            EvaluationTask(
                evaluation_id=(
                    f"{question_id}::{RETRIEVAL_DIMENSION}"
                ),
                question_id=question_id,
                response_id=None,
                dimension=RETRIEVAL_DIMENSION,
                answer=None,
                common_inputs=_common_inputs(row),
                case_metadata=_task_metadata(row),
            )
        )

    tasks.sort(key=lambda task: task.evaluation_id)
    identifiers = [task.evaluation_id for task in tasks]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Evaluation task identifiers are not unique")
    return tasks


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def summarize_predictions(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = list(records)
    status_counts: Counter[str] = Counter()
    dimension_counts: dict[str, Counter[str]] = defaultdict(Counter)
    attempt_latencies: list[float] = []
    throughputs: list[float] = []
    retries = 0
    attempts_total = 0
    for row in rows:
        judge = row.get("judge")
        if not isinstance(judge, Mapping):
            continue
        status = str(judge.get("status", "missing"))
        dimension = str(row.get("dimension", "unknown"))
        status_counts[status] += 1
        dimension_counts[dimension][status] += 1
        retries += int(bool(judge.get("retry_used")))
        attempts = judge.get("attempts")
        if not isinstance(attempts, list):
            continue
        attempts_total += len(attempts)
        for attempt in attempts:
            if not isinstance(attempt, Mapping):
                continue
            metrics = attempt.get("metrics")
            if not isinstance(metrics, Mapping):
                continue
            wall = metrics.get("wall_seconds")
            tps = metrics.get("tokens_per_second")
            if isinstance(wall, (int, float)):
                attempt_latencies.append(float(wall))
            if isinstance(tps, (int, float)) and float(tps) > 0:
                throughputs.append(float(tps))

    latency = {
        "p50": _percentile(attempt_latencies, 0.50),
        "p95": _percentile(attempt_latencies, 0.95),
        "maximum": max(attempt_latencies) if attempt_latencies else None,
    }
    return {
        "evaluation_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "retry_count": retries,
        "attempt_count": attempts_total,
        "dimensions": {
            dimension: dict(sorted(counts.items()))
            for dimension, counts in sorted(dimension_counts.items())
        },
        "attempt_latency_seconds": latency,
        "median_tokens_per_second": (
            statistics.median(throughputs) if throughputs else None
        ),
    }


def _load_prediction_records(path: Path) -> list[dict[str, Any]]:
    return _read_jsonl(path) if path.exists() else []


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _manifest_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "runner_version",
        "run_id",
        "git_commit",
        "dataset",
        "split",
        "judge_configuration",
        "task_plan_sha256",
        "planned_evaluation_count",
        "dry_run",
    )
    return {key: manifest.get(key) for key in keys}


def validate_record_provenance(
    record: Mapping[str, Any], manifest: Mapping[str, Any], input_hash: str
) -> None:
    dataset = manifest["dataset"]
    config = manifest["judge_configuration"]
    fields = {
        "dataset_id": dataset["id"],
        "dataset_version": dataset["version"],
        "rubric_sha256": dataset["rubric_sha256"],
        "input_sha256": input_hash,
        "judge_configuration": {
            **{key: config[key] for key in ("model_id", "quantization", "hardware")},
            "declared_generation_config": config["generation"],
        },
    }
    for key, expected in fields.items():
        if record.get(key) != expected:
            raise ValueError(f"{record.get('evaluation_id')}: mismatched {key}")
    judgment = record.get("judge")
    if not isinstance(judgment, Mapping) or judgment.get("status") not in ("valid", "invalid", "error"):
        raise ValueError(f"{record.get('evaluation_id')}: missing or invalid judge status")


def validate_scalar_records(
    records: Sequence[Mapping[str, Any]],
    tasks: Sequence[EvaluationTask],
    manifest: Mapping[str, Any],
) -> set[str]:
    expected = {task.evaluation_id: task for task in tasks}
    seen: set[str] = set()
    for row in records:
        evaluation_id = row.get("evaluation_id")
        if not isinstance(evaluation_id, str) or evaluation_id not in expected:
            raise ValueError(f"Evaluation ID is outside the frozen task plan: {evaluation_id!r}")
        if evaluation_id in seen:
            raise ValueError(f"Duplicate evaluation ID: {evaluation_id}")
        task = expected[evaluation_id]
        for key in ("question_id", "response_id", "dimension"):
            if key not in row or row[key] != getattr(task, key):
                raise ValueError(f"{evaluation_id}: mismatched {key}")
        inputs = {
            **task.common_inputs,
            "answer": task.answer,
            "conversation_history": _validate_history(task.common_inputs["conversation_history"]),
        }
        if row.get("inputs") != inputs or row.get("case_metadata") != task.case_metadata:
            raise ValueError(f"{evaluation_id}: inputs or case metadata differ from the frozen plan")
        validate_record_provenance(row, manifest, _canonical_hash(inputs))
        seen.add(evaluation_id)
    return seen


def _make_remote_provider(**kwargs):
    from llm.llmcli.providers.remote_http import RemoteHTTPProvider

    return RemoteHTTPProvider(**kwargs)


def prepare_run(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    resume: bool,
    tasks: Sequence[EvaluationTask] | None = None,
) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    predictions_path = run_dir / "predictions.jsonl"
    if run_dir.exists() and not resume:
        raise ValueError(
            f"Run directory already exists; use --resume: {run_dir}"
        )
    if resume:
        if not manifest_path.is_file():
            raise ValueError(
                f"Cannot resume without manifest: {manifest_path}"
            )
        existing = _read_json(manifest_path)
        if _manifest_identity(existing) != _manifest_identity(manifest):
            raise ValueError(
                "Resume configuration does not match existing manifest"
            )
        if tasks is not None:
            validate_scalar_records(_load_prediction_records(predictions_path), tasks, existing)
        manifest = existing
        for key in ("finished_utc", "completed_utc", "runner_error"):
            manifest.pop(key, None)
        manifest["status"] = "running"
        manifest["resumed_utc"] = _utc_now()
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest["created_utc"] = _utc_now()
        manifest["status"] = "running"
    if predictions_path.exists() and not resume:
        raise ValueError(f"Predictions already exist: {predictions_path}")
    _write_json_atomic(manifest_path, manifest)
    return manifest


def _write_summary_csv(path: Path, summary: Mapping[str, Any]) -> None:
    dimensions = summary.get("dimensions", {})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "dimension",
                "valid",
                "invalid",
                "error",
                "total",
            ),
        )
        writer.writeheader()
        for dimension, counts in sorted(dimensions.items()):
            total = sum(int(value) for value in counts.values())
            writer.writerow(
                {
                    "dimension": dimension,
                    "valid": counts.get("valid", 0),
                    "invalid": counts.get("invalid", 0),
                    "error": counts.get("error", 0),
                    "total": total,
                }
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen StreamTutor judge benchmark v3",
    )
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument(
        "--split",
        required=True,
        choices=("calibration", "selection", "robustness"),
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--quantization", required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    parser.add_argument("--read-timeout", type=float, default=180.0)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-responses", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
        raise ValueError(
            "run_id must contain only letters, digits, '.', '_' or '-' "
            "and must not exceed 128 characters"
        )
    return run_id


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dataset = verify_dataset(args.dataset_dir)
    root: Path = dataset["root"]
    questions = _read_jsonl(root / "questions.jsonl")
    responses = _read_jsonl(root / "responses.jsonl")
    tasks = build_evaluation_tasks(
        questions=questions,
        responses=responses,
        split=args.split,
        max_responses=args.max_responses,
    )
    if not tasks:
        raise ValueError(f"No evaluation tasks for split: {args.split}")

    rubric = load_frozen_rubric(root / "rubric_v3.json")
    generation_config = {
        "n_ctx": args.n_ctx,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "batch_size": 1,
    }
    run_id = _validate_run_id(args.run_id or (
        f"{args.model_id}-{args.split}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid.uuid4().hex[:8]}"
    ))
    run_dir = Path(args.output_root).resolve() / run_id
    task_ids = [task.evaluation_id for task in tasks]
    manifest: dict[str, Any] = {
        "runner_version": RUNNER_VERSION,
        "run_id": run_id,
        "dataset": {
            "id": dataset["dataset_id"],
            "version": dataset["dataset_version"],
            "checksums_sha256": dataset["checksums_sha256"],
            "responses_sha256": dataset["checked_files"]["responses.jsonl"],
            "questions_sha256": dataset["checked_files"]["questions.jsonl"],
            "rubric_sha256": rubric.sha256,
        },
        "split": args.split,
        "judge_configuration": {
            "base_url": args.base_url,
            "model_id": args.model_id,
            "quantization": args.quantization,
            "hardware": args.hardware,
            "generation": generation_config,
            "connect_timeout": args.connect_timeout,
            "read_timeout": args.read_timeout,
        },
        "git_commit": _git_commit(),
        "task_plan_sha256": _canonical_hash(task_ids),
        "planned_evaluation_count": len(tasks),
        "dry_run": bool(args.dry_run),
    }
    manifest = prepare_run(
        run_dir=run_dir,
        manifest=manifest,
        resume=args.resume,
        tasks=tasks,
    )
    manifest_path = run_dir / "manifest.json"

    if args.dry_run:
        plan = {
            "run_id": run_id,
            "split": args.split,
            "evaluation_count": len(tasks),
            "dimension_counts": dict(
                sorted(Counter(task.dimension for task in tasks).items())
            ),
            "task_plan_sha256": manifest["task_plan_sha256"],
        }
        _write_json_atomic(run_dir / "plan.json", plan)
        manifest["status"] = "dry_run_complete"
        manifest["completed_utc"] = _utc_now()
        _write_json_atomic(manifest_path, manifest)
        print(json.dumps(plan, indent=2))
        print(f"Run directory: {run_dir}")
        return 0

    provider = _make_remote_provider(
        base_url=args.base_url,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
    )
    if not provider.healthcheck():
        manifest["status"] = "failed_healthcheck"
        manifest["finished_utc"] = _utc_now()
        _write_json_atomic(manifest_path, manifest)
        print(
            f"Judge healthcheck failed: {args.base_url}",
            file=sys.stderr,
        )
        return 2

    predictions_path = run_dir / "predictions.jsonl"
    evaluator = ScalarJudgeV3(
        provider,
        model_id=args.model_id,
        quantization=args.quantization,
        hardware=args.hardware,
        output_path=predictions_path,
        dataset_id=dataset["dataset_id"],
        dataset_version=dataset["dataset_version"],
        rubric_path=root / "rubric_v3.json",
        declared_generation_config=generation_config,
    )
    completed = evaluator.logged_evaluation_ids
    remaining = [
        task for task in tasks if task.evaluation_id not in completed
    ]

    try:
        for index, task in enumerate(remaining, start=1):
            print(
                f"[{index}/{len(remaining)}] "
                f"{task.evaluation_id}",
                flush=True,
            )
            evaluator.evaluate_and_log(**task.evaluator_kwargs())
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["finished_utc"] = _utc_now()
        manifest["completed_evaluation_count"] = len(
            evaluator.logged_evaluation_ids
        )
        _write_json_atomic(manifest_path, manifest)
        return 130
    except Exception as exc:
        manifest["status"] = "runner_error"
        manifest["finished_utc"] = _utc_now()
        manifest["runner_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        manifest["completed_evaluation_count"] = len(
            evaluator.logged_evaluation_ids
        )
        _write_json_atomic(manifest_path, manifest)
        raise

    try:
        records = _load_prediction_records(predictions_path)
        recorded_ids = validate_scalar_records(records, tasks, manifest)
    except (OSError, ValueError) as exc:
        manifest.update(status="runner_error", finished_utc=_utc_now())
        manifest["runner_error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_json_atomic(manifest_path, manifest)
        raise
    summary = summarize_predictions(records)
    summary.update(
        {
            "run_id": run_id,
            "split": args.split,
            "planned_evaluation_count": len(tasks),
            "complete": recorded_ids == set(task_ids),
        }
    )
    _write_json_atomic(run_dir / "summary.json", summary)
    _write_summary_csv(run_dir / "summary_by_dimension.csv", summary)
    manifest["status"] = (
        "completed" if summary["complete"] else "incomplete"
    )
    manifest["finished_utc"] = _utc_now()
    manifest["completed_evaluation_count"] = len(records)
    _write_json_atomic(manifest_path, manifest)
    print(json.dumps(summary, indent=2))
    print(f"Run directory: {run_dir}")
    return 0 if summary["complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
