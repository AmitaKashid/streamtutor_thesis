"""Collect single-GPU telemetry and associate it with a v3 latency run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import socket
import statistics
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

QUERY = "uuid,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu"
MIN_FREE_MIB = 2048
LAYER_REPORT = re.compile(r"\boffloaded\s+(\d+)\s*/\s*(\d+)\s+layers?\s+to\s+GPU\b", re.I)


def _now():
    return datetime.now(timezone.utc).isoformat()


def read_gpu(gpu_uuid):
    result = subprocess.run(
        ["nvidia-smi", "-i", gpu_uuid, f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=5,
    )
    timestamp = _now()
    rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 7 or rows[0][0].strip() != gpu_uuid:
        raise ValueError("Expected exactly one nvidia-smi row for the requested GPU UUID")
    _, name, driver, total, used, free, utilization = (value.strip() for value in rows[0])
    total, used, free = map(int, (total, used, free))
    utilization = None if utilization in {"N/A", "[N/A]", "[Not Supported]"} else int(utilization)
    if not (0 <= used <= total and 0 <= free <= total and total > 0):
        raise ValueError("Invalid GPU memory readings")
    if utilization is not None and not 0 <= utilization <= 100:
        raise ValueError("Invalid GPU utilization reading")
    return {
        "sampled_utc": timestamp, "gpu_uuid": gpu_uuid, "gpu_name": name,
        "driver_version": driver, "total_mib": total, "used_mib": used,
        "free_mib": free, "utilization_pct": utilization,
    }


def collect(args):
    job = os.environ.get("SLURM_JOB_ID")
    if not job or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Run the sampler inside the judge's GPU allocation")
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", args.gpu_uuid):
        raise ValueError("Use the physical GPU UUID confirmed for the judge allocation")
    if args.interval_ms < 100 or args.samples < 0:
        raise ValueError("interval-ms must be >= 100 and samples must be >= 0")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        index = 0
        try:
            while not args.samples or index < args.samples:
                started = time.monotonic()
                sample = read_gpu(args.gpu_uuid)
                handle.write(json.dumps({
                    **sample, "slurm_job_id": job, "hostname": socket.gethostname(),
                    "phase": args.phase, "interval_ms": args.interval_ms,
                }) + "\n")
                handle.flush()
                index += 1
                if index == 1:
                    print(f"Sampling {sample['gpu_name']} {args.gpu_uuid}, job {job}", flush=True)
                if not args.samples or index < args.samples:
                    time.sleep(max(0, args.interval_ms / 1000 - (time.monotonic() - started)))
        except KeyboardInterrupt:
            pass
    return 0


def _jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _timestamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() != timedelta(0):
        raise ValueError("Expected a UTC timestamp")
    return parsed


def offload_evidence(path):
    if path is None:
        return {"layer_offload": "unknown", "unintended_cpu_tensor_offload": "not_verified"}
    content = path.read_bytes()
    lines = content.decode("utf-8", errors="replace").splitlines()
    counts = [tuple(map(int, match.groups())) for line in lines if (match := LAYER_REPORT.search(line))]
    status = "unknown"
    if len(counts) == 1:
        on_gpu, total = counts[0]
        if 0 <= on_gpu <= total and total > 0:
            status = "all_layers_reported_on_gpu" if on_gpu == total else "partial_layer_offload"
    hints = [
        line for line in lines
        if re.search(r"CPU(?:_Mapped|_Host)?\s+(?:model|KV|compute)\s+buffer|offloaded.*layers", line, re.I)
    ]
    return {
        "layer_offload": status, "reported_layer_counts": counts,
        "unintended_cpu_tensor_offload": "not_verified",
        "server_log_sha256": hashlib.sha256(content).hexdigest(),
        "matching_log_lines": hints,
    }


def summarize(manifest, evaluations, samples, offload, idle_samples=()):
    if manifest.get("status") != "completed" or manifest.get("required_hardware") != "NVIDIA A30 24 GB":
        raise ValueError("Expected a completed A30 latency-run manifest")
    measured = [row for row in evaluations if row.get("phase") == "measured"]
    expected = {(repeat, task["task_id"]) for repeat in (1, 2, 3) for task in measured if task.get("repetition") == 1}
    actual = {(row["repetition"], row["task_id"]) for row in measured}
    if len(measured) != 90 or len(expected) != 90 or actual != expected:
        raise ValueError("Expected three complete repetitions of 30 distinct measured evaluations")
    starts = [_timestamp(row["started_utc"]) for row in measured]
    start = min(starts)
    end = max(
        at + timedelta(seconds=row["evaluation_wall_seconds"])
        for at, row in zip(starts, measured)
    )
    identity_fields = ("slurm_job_id", "hostname", "gpu_uuid", "gpu_name", "driver_version", "total_mib")
    identities = {tuple(row[key] for key in identity_fields) for row in [*samples, *idle_samples]}
    if len(identities) != 1:
        raise ValueError("Samples must all come from one GPU and Slurm allocation")
    identity = dict(zip(identity_fields, next(iter(identities))))
    if (not manifest.get("slurm_job_id") or str(identity["slurm_job_id"]) != str(manifest["slurm_job_id"])
            or "A30" not in identity["gpu_name"].upper()):
        raise ValueError("Sampled job and GPU must match the A30 latency run")
    if any(row["phase"] != "benchmark" for row in samples) or any(
        row["phase"] != "loaded-idle" for row in idle_samples
    ):
        raise ValueError("Use benchmark samples and, optionally, separate loaded-idle samples")
    times = [_timestamp(row["sampled_utc"]) for row in samples]
    if times != sorted(set(times)):
        raise ValueError("Benchmark timestamps must be strictly increasing")
    if any(_timestamp(row["sampled_utc"]) >= start for row in idle_samples):
        raise ValueError("Loaded-idle samples must precede the measured workload")
    before = [at for at in times if at <= start]
    after = [at for at in times if at >= end]
    within = [row for at, row in zip(times, samples) if start <= at <= end]
    bracketed = bool(before and after and within)
    max_gap = None
    if bracketed:
        window_times = [at for at in times if before[-1] <= at <= after[0]]
        max_gap = max((b - a).total_seconds() for a, b in zip(window_times, window_times[1:]))
    coverage = bracketed and max_gap <= 2.0
    min_free = min((row["free_mib"] for row in within), default=None)
    utilizations = [row["utilization_pct"] for row in within if row["utilization_pct"] is not None]
    observed_headroom = "not_verified"
    if min_free is not None and min_free < MIN_FREE_MIB:
        observed_headroom = "below_threshold"
    elif coverage:
        observed_headroom = "threshold_met_in_samples"
    return {
        "run_id": manifest["run_id"], **identity,
        "window_start_utc": start.isoformat(), "window_end_utc": end.isoformat(),
        "window_sample_count": len(within), "coverage_ok": coverage, "max_sample_gap_seconds": max_gap,
        "loaded_idle_used_mib_median": statistics.median(
            row["used_mib"] for row in idle_samples
        ) if idle_samples else None,
        "loaded_idle_sample_count": len(idle_samples),
        "sampled_peak_used_mib": max((row["used_mib"] for row in within), default=None),
        "sampled_minimum_free_mib": min_free,
        "sampled_utilization_pct_median": statistics.median(utilizations) if utilizations else None,
        "sampled_utilization_pct_max": max(utilizations, default=None),
        "minimum_free_threshold_mib": MIN_FREE_MIB,
        "observed_headroom": observed_headroom,
        "offload_evidence": offload,
        "deployment_gate_evaluated": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sampler = sub.add_parser("collect")
    sampler.add_argument("--gpu-uuid", required=True)
    sampler.add_argument("--output", required=True, type=Path)
    sampler.add_argument("--phase", choices=("loaded-idle", "benchmark"), required=True)
    sampler.add_argument("--interval-ms", type=int, default=500)
    sampler.add_argument("--samples", type=int, default=0, help="0 means run until Ctrl+C")
    report = sub.add_parser("summarize")
    report.add_argument("--run-dir", required=True, type=Path)
    report.add_argument("--samples", required=True, type=Path)
    report.add_argument("--loaded-idle", type=Path)
    report.add_argument("--server-log", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "collect":
            return collect(args)
        summary = summarize(
            json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8")),
            _jsonl(args.run_dir / "evaluations.jsonl"), _jsonl(args.samples),
            offload_evidence(args.server_log), _jsonl(args.loaded_idle) if args.loaded_idle else (),
        )
        with (args.run_dir / "gpu_summary.json").open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
        print(json.dumps(summary, indent=2))
        return 0 if summary["coverage_ok"] else 2
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
