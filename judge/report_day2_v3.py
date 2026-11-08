"""Report the frozen v3 Day 2 deployment gate from verified latency archives."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from judge.archive_evidence_v3 import verify_archive
from judge.run_judge_benchmark_v3 import (
    DEFAULT_DATASET_DIR, _canonical_hash, _read_jsonl, build_evaluation_tasks, verify_dataset,
)
from judge.run_latency_benchmark_v3 import (
    GENERATION, HARDWARE, MATRIX, _plan_entries, build_latency_plan,
    summarize as summarize_latency,
)
from judge.sample_gpu_resources_v3 import LAYER_REPORT, summarize as summarize_gpu

PASS, FAIL, UNKNOWN = "pass", "fail", "not_verified"
GATES = {
    "frozen_a30_configuration": "Verified A30 hardware and frozen server configuration are required.",
    "gpu_layer_offload": "A matching startup log must demonstrate complete GPU-layer offload.",
    "no_unintended_cpu_tensor_offload": "GPU-layer placement alone does not establish tensor placement.",
    "minimum_free_vram": "GPU samples must cover the measured window and show at least 2048 MiB free.",
    "no_unrecovered_crash_or_hang": "The runtime log must be reviewed for unrecovered crashes and hangs.",
    "median_decode_throughput": "Decode-only throughput must be measured; server TPS includes prefill.",
    "p95_completion_latency": "P95 completion latency, including format retry, must be at most 25 s.",
    "valid_after_at_most_one_retry": "All measured evaluations must be valid after at most one format retry.",
}


def candidate_ids():
    with MATRIX.open(encoding="utf-8", newline="") as stream:
        rows = [row["model_id"] for row in csv.DictReader(stream) if row["stage"] == "comparative_latency"]
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("Invalid comparative-latency candidate matrix")
    return rows


def frozen_plan(dataset_dir):
    dataset = verify_dataset(dataset_dir)
    root = dataset["root"]
    tasks = build_evaluation_tasks(
        questions=_read_jsonl(root / "questions.jsonl"),
        responses=_read_jsonl(root / "responses.jsonl"),
        split="calibration",
    )
    warmups, selected = build_latency_plan(tasks)
    return dataset, _plan_entries(warmups, selected)


def _gate(status, reason, value=None):
    return {"status": status, "reason": reason, "value": value}


def _finish(row):
    states = {gate["status"] for gate in row["gates"].values()}
    row["decision"] = FAIL if FAIL in states else UNKNOWN if UNKNOWN in states else PASS
    return row


def _check(flag, reason, value=None):
    return _gate(PASS if flag else FAIL, reason, value)


def _jsonl(archive, name):
    with archive.open(name) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _valid_judgment(judgment):
    attempts = judgment["attempts"]
    if judgment["status"] != "valid" or len(attempts) not in (1, 2):
        return False
    if len(attempts) == 2 and (
        attempts[0].get("parsed_result") is not None
        or not attempts[0].get("validation_error")
        or attempts[0].get("generation_error") is not None
    ):
        return False
    last = attempts[-1]
    parsed = last.get("parsed_result")
    score = parsed.get("score") if isinstance(parsed, dict) else None
    return (
        last.get("generation_error") is None and not last.get("validation_error")
        and type(score) is int and 1 <= score <= 5
    )


def _reviewed_boolean(review, name, gate):
    item = review.get(name)
    if item is None:
        return _gate(UNKNOWN, GATES[gate])
    if not isinstance(item, dict) or type(item.get("value")) is not bool:
        raise ValueError(f"Invalid review field: {name}")
    return _check(item["value"], _reference(item, name), item["value"])


def _reference(item, name):
    source = item.get("evidence")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"Missing evidence reference for review field: {name}")
    return f"Operator-reviewed: {source}"


def _inspect_archive(archive, info, dataset, entries, review, row):
    manifest = json.loads(archive.read("run/manifest.json"))
    gates = row["gates"]
    if manifest["status"] != "completed":
        row["notes"].append(f"Run status is {manifest['status']}; no deployment decision can pass.")
        return
    plan_hash = _canonical_hash(entries)
    expected = {
        "protocol": "judge_execution_protocol_v3",
        "required_hardware": HARDWARE,
        "required_backend": "llama-cpp-python 0.3.20",
        "required_quantization": "Q4_K_M",
        "required_server_configuration": GENERATION,
        "dataset_id": dataset["dataset_id"],
        "dataset_version": dataset["dataset_version"],
        "dataset_checksums_sha256": dataset["checksums_sha256"],
        "rubric_sha256": dataset["checked_files"]["rubric_v3.json"],
        "task_plan_sha256": plan_hash,
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        row["notes"].append("Not the frozen A30 workload: " + ", ".join(mismatches))
        return
    plan = json.loads(archive.read("run/plan.json"))
    records = _jsonl(archive, "run/evaluations.jsonl")
    actual = [{key: rec.get(key) for key in entry} for entry, rec in zip(entries, records)]
    if (
        plan.get("entries") != entries or plan.get("task_plan_sha256") != plan_hash
        or len(records) != len(entries) or actual != entries
        or [rec.get("sequence") for rec in records] != list(range(1, len(entries) + 1))
        or not all(_positive(rec.get("evaluation_wall_seconds")) for rec in records)
    ):
        row["notes"].append("Plan or measured record sequence does not match the frozen workload.")
        return
    names = set(archive.namelist())
    if info["missing_evidence"]:
        row["notes"].append("Archive gaps: " + ", ".join(info["missing_evidence"]))
    provenance_files = {"attachments/configuration.txt", "attachments/slurm-job.txt", "attachments/server.log"}
    if info.get("run_git_commit") and info.get("slurm_job_id") and provenance_files <= names:
        gates["frozen_a30_configuration"] = _reviewed_boolean(
            review, "frozen_a30_configuration", "frozen_a30_configuration"
        )
    if gates["frozen_a30_configuration"]["status"] != PASS:
        row["notes"].append("Metrics do not enter the gate until the actual A30 configuration is confirmed.")
        return

    for gate in ("no_unintended_cpu_tensor_offload", "no_unrecovered_crash_or_hang"):
        gates[gate] = _reviewed_boolean(review, gate, gate)
    decode = review.get("median_decode_throughput")
    if decode is not None:
        if not isinstance(decode, dict) or not _positive(decode.get("value")):
            raise ValueError("median_decode_throughput requires a positive decode-only measurement")
        value = decode["value"]
        gates["median_decode_throughput"] = _check(
            value >= 10, _reference(decode, "median_decode_throughput") + "; minimum 10 tokens/s.", value
        )
    counts = [
        tuple(map(int, m)) for m in LAYER_REPORT.findall(
            archive.read("attachments/server.log").decode("utf-8", errors="replace")
        )
    ]
    if len(counts) == 1 and 0 <= counts[0][0] <= counts[0][1] and counts[0][1] > 0:
        loaded, total = counts[0]
        gates["gpu_layer_offload"] = _check(
            loaded == total, f"Startup log reports {loaded}/{total} layers on GPU.", loaded
        )
    if "attachments/gpu-samples.jsonl" in names:
        try:
            samples = _jsonl(archive, "attachments/gpu-samples.jsonl")
            gpu = summarize_gpu(manifest, records, samples, {})
            free = gpu["sampled_minimum_free_mib"]
            if gpu["observed_headroom"] == "below_threshold":
                gates["minimum_free_vram"] = _gate(FAIL, f"Observed {free} MiB free; minimum is 2048.", free)
            elif gpu["coverage_ok"]:
                gates["minimum_free_vram"] = _check(free >= 2048, "Covered window; minimum is 2048 MiB.", free)
        except (ValueError, KeyError, TypeError) as exc:
            row["notes"].append(f"GPU samples not accepted: {exc}")
    count = sum(entry["phase"] == "measured" for entry in entries)
    summary = summarize_latency(records, count)
    latency = summary["evaluation_wall_seconds_including_retry"]["p95"]
    gates["p95_completion_latency"] = _check(latency <= 25, "Evaluation wall time including retry; maximum 25 s.", latency)
    measured = [rec["judge"] for rec in records if rec["phase"] == "measured"]
    valid = sum(_valid_judgment(judgment) for judgment in measured)
    gates["valid_after_at_most_one_retry"] = _check(valid == count, f"{valid}/{count} valid within two attempts.", valid)


def build_report(paths=(), *, dataset_dir=DEFAULT_DATASET_DIR, reviews=None):
    candidates = {
        name: {"model_id": name, "archive_sha256": None, "run_id": None, "notes": [],
               "gates": {key: _gate(UNKNOWN, reason) for key, reason in GATES.items()}}
        for name in candidate_ids()
    }
    reviews = {} if reviews is None else reviews
    if not isinstance(reviews, dict):
        raise ValueError("Reviews must be keyed by evidence ZIP SHA-256")
    frozen = frozen_plan(dataset_dir) if paths else None
    used = set()
    for path in paths:
        info = verify_archive(path)
        if info["stage"] != "latency":
            raise ValueError("Day 2 deployment gates require latency evidence archives")
        digest = info["archive_sha256"]
        review = reviews.get(digest, {})
        if not isinstance(review, dict):
            raise ValueError("Each review must be an object")
        used.add(digest)
        with ZipFile(path) as archive:
            manifest = json.loads(archive.read("run/manifest.json"))
            model = manifest.get("model_id")
            if model not in candidates or candidates[model]["run_id"] is not None:
                raise ValueError(f"Unknown candidate or multiple archives for the same model: {model}")
            row = candidates[model]
            row.update(run_id=manifest["run_id"], archive_sha256=digest)
            dataset, entries = frozen
            _inspect_archive(archive, info, dataset, entries, review, row)
    if reviews.keys() - used:
        raise ValueError("A review refers to an archive not supplied to this report")
    for row in candidates.values():
        if row["run_id"] is None:
            row["notes"].append("No archived A30 latency run supplied.")
    return {
        "protocol": "judge_execution_protocol_v3",
        "scope": "Day 2 standalone A30 deployment gate; tutor interference and judge quality are separate.",
        "candidates": [_finish(row) for row in candidates.values()],
    }


def markdown(report):
    lines = ["# Day 2 A30 deployment gate", "", report["scope"], "",
             "| Candidate | Run | Decision |", "|---|---|---|"]
    for row in report["candidates"]:
        run_id = (row["run_id"] or "—").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {row['model_id']} | {run_id} | {row['decision']} |")
    for row in report["candidates"]:
        lines.extend(["", f"## {row['model_id']}", ""])
        lines.extend(f"- {note}" for note in row["notes"])
        lines.extend(f"- **{key} — {gate['status']}**: {gate['reason']}" for key, gate in row["gates"].items())
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, action="append", default=[])
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        reviews = json.loads(args.reviews.read_text(encoding="utf-8")) if args.reviews else None
        report = build_report(args.archive, dataset_dir=args.dataset_dir, reviews=reviews)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "day2_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / "day2_report.md").write_text(markdown(report), encoding="utf-8")
        print(f"Day 2 report: {args.output_dir}")
        return 0
    except (ValueError, KeyError, TypeError, OSError, BadZipFile) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
