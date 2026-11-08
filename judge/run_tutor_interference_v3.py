"""Replay fixed text turns through the tutor WebSocket and compare judge modes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

from judge.run_judge_benchmark_v3 import _git_commit, _percentile, _utc_now, _validate_run_id

ROOT = Path(__file__).resolve().parents[1]
MODES = ("disabled", "same-gpu", "isolated-gpu", "unavailable")
CONFIG = {
    "llm_model": "llama3_8b_q4", "tts_backend": "kokoro", "rag": True,
    "rag_top_k": 2, "teaching_style": "socratic", "conversation_context": False,
}
METRICS = ("ttft", "firstAudio", "llmTotal", "total")


def load_plan():
    rows = [
        json.loads(line) for line in
        (ROOT / "data" / "benchmark_questions.jsonl").read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    prompts = [{"id": row["id"], "text": row["question"]} for row in rows]
    if len(prompts) != 30 or len({row["id"] for row in prompts}) != 30:
        raise ValueError("Expected 30 distinct fixed tutor prompts")
    if any(not row["text"].strip() for row in prompts):
        raise ValueError("Empty tutor prompt")
    warmup = {"id": "warmup", "text": "What is data mining?"}
    return {"warmup": warmup, "measured": prompts, "config": CONFIG}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def duration(value):
    return value if type(value) in (float, int) and math.isfinite(value) and value > 0 else None


def play_turn(ws, case, timeout):
    started_utc = _utc_now()
    start = time.perf_counter()
    ws.send(json.dumps({"action": "text_query", "text": case["text"], "config": CONFIG}))
    result = {
        "case_id": case["id"], "started_utc": started_utc, "metrics": {},
        "judge_submission": None, "effective_config": None, "first_audio_received_s": None,
        "audio_chunk_count": 0, "audio_done": False, "assistant_done": False,
        "tts_worker_errors": [], "errors": [],
    }
    while not result["assistant_done"]:
        remaining = timeout - (time.perf_counter() - start)
        if remaining <= 0:
            raise TimeoutError("Tutor turn exceeded its deadline")
        ws.settimeout(remaining)
        event = json.loads(ws.recv())
        kind = event.get("type")
        data = event.get("data")
        if kind == "audio_chunk" and isinstance(data, str) and data:
            result["audio_chunk_count"] += 1
            if result["first_audio_received_s"] is None:
                result["first_audio_received_s"] = time.perf_counter() - start
        elif kind == "config_applied":
            result["effective_config"] = data
        elif kind == "metrics" and isinstance(data, dict):
            result["metrics"].update({key: data[key] for key in METRICS if key in data})
        elif kind == "judge_submission":
            if result["judge_submission"] is not None:
                raise ValueError("Duplicate judge submission event")
            result["judge_submission"] = data
        elif kind in {"error", "tts_worker_error", "audio_stop"}:
            target = "tts_worker_errors" if kind == "tts_worker_error" else "errors"
            result[target].append(event.get("message", kind))
        elif kind in {"audio_done", "assistant_done"}:
            result[kind] = True
    result["client_turn_s"] = time.perf_counter() - start
    return result


def usable(row, mode):
    submission = row.get("judge_submission")
    metrics = row.get("metrics", {})
    if (
        not isinstance(submission, dict) or not submission.get("turn_id")
        or submission.get("response_status") != "completed"
        or not row.get("assistant_done") or not row.get("audio_done")
        or not duration(row.get("first_audio_received_s"))
        or row.get("errors") or row.get("tts_worker_errors")
        or not isinstance(row.get("effective_config"), dict)
        or not all(duration(metrics.get(key)) for key in METRICS)
    ):
        return False
    enabled = mode in {"same-gpu", "isolated-gpu"}
    if enabled:
        return bool(
            submission.get("judge_available") is True and submission.get("task_id")
            and duration(submission.get("enqueue_seconds"))
        )
    return submission.get("judge_available") is False and submission.get("task_id") is None


def compare(runs):
    baseline = runs["disabled"]["median_total_s"]
    for mode in ("same-gpu", "isolated-gpu"):
        report = runs[mode]
        report["median_total_change_pct"] = 100 * (report["median_total_s"] / baseline - 1)
        report["median_total_within_5pct"] = report["median_total_s"] <= baseline * 1.05
        report["enqueue_p95_below_50ms"] = report["enqueue_p95_s"] < 0.05
    return {
        "conditions": runs,
        "deployment_gate_evaluated": False,
        "limitations": [
            "Typed input bypasses STT; first audio received is not audible playback.",
            "GPU topology, judge/tutor overlap and peak VRAM require separate evidence.",
            "An unavailable-at-startup run does not demonstrate recovery from a mid-run outage.",
        ],
    }


def compare_dirs(paths):
    reports = {}
    identity = None
    config = None
    for path in paths:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        mode = manifest["mode"]
        current = (manifest["plan_sha256"], manifest["git_commit"])
        if (
            mode in reports or mode not in MODES or manifest["status"] != "completed"
            or manifest.get("git_commit") is None or (identity is not None and identity != current)
        ):
            raise ValueError("Use one completed run per condition with the same plan and Git commit")
        identity = current
        plan = json.loads((path / "plan.json").read_text(encoding="utf-8"))
        if digest(plan) != manifest["plan_sha256"]:
            raise ValueError("Plan checksum mismatch")
        rows = [
            json.loads(line) for line in (path / "turns.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        measured = [row for row in rows if row["phase"] == "measured"]
        if (
            len(measured) != 30 or [r["case_id"] for r in measured] != [r["id"] for r in plan["measured"]]
            or not all(usable(row, mode) for row in measured)
            or len({row["judge_submission"]["turn_id"] for row in measured}) != 30
        ):
            raise ValueError(f"{mode}: missing, duplicate, or unusable measured turns; inspect turns.jsonl")
        for row in measured:
            if config is not None and row["effective_config"] != config:
                raise ValueError("Tutor effective configuration changed across measured turns")
            config = row["effective_config"]
        report = {"turns": len(measured)}
        for key in METRICS:
            values = [row["metrics"][key] for row in measured]
            report[f"median_{key}_s"] = statistics.median(values)
            report[f"p95_{key}_s"] = _percentile(values, 0.95)
        report["median_first_audio_received_s"] = statistics.median(
            row["first_audio_received_s"] for row in measured
        )
        enqueue = [
            row["judge_submission"]["enqueue_seconds"] for row in measured
            if duration(row["judge_submission"].get("enqueue_seconds"))
        ]
        report["enqueue_p95_s"] = _percentile(enqueue, 0.95)
        reports[mode] = report
    if not set(MODES[:3]) <= set(reports):
        raise ValueError("Comparison requires disabled, same-gpu, and isolated-gpu runs")
    return compare(reports)


def _write(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args):
    plan = load_plan()
    if not args.dry_run and not args.ws_url:
        raise ValueError("A live run requires --ws-url")
    if not duration(args.timeout):
        raise ValueError("Timeout must be finite and positive")
    root = args.output_root / _validate_run_id(args.run_id)
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "mode": args.mode, "run_id": args.run_id, "git_commit": _git_commit(),
        "plan_sha256": digest(plan), "ws_url": args.ws_url, "started_utc": _utc_now(),
        "status": "dry_run" if args.dry_run else "running",
    }
    _write(root / "plan.json", plan)
    _write(root / "manifest.json", manifest)
    if args.dry_run:
        print(json.dumps({"warmups": 1, "measured": 30, "plan_sha256": digest(plan)}))
        return 0
    try:
        from websocket import create_connection

        with (root / "turns.jsonl").open("x", encoding="utf-8") as handle:
            for case in [plan["warmup"], *plan["measured"]]:
                phase = "warmup" if case["id"] == "warmup" else "measured"
                try:
                    ws = create_connection(args.ws_url, timeout=args.timeout)
                    try:
                        row = play_turn(ws, case, args.timeout)
                    finally:
                        ws.close()
                except Exception as exc:
                    row = {"case_id": case["id"], "errors": [f"{type(exc).__name__}: {exc}"]}
                    handle.write(json.dumps({**row, "phase": phase}) + "\n")
                    handle.flush()
                    raise
                handle.write(json.dumps({**row, "phase": phase}) + "\n")
                handle.flush()
                if phase == "warmup" and not usable(row, args.mode):
                    raise RuntimeError("Warm-up did not produce valid audio, metrics, and expected judge state")
        manifest["status"] = "completed"
        return 0
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        return 130
    except Exception as exc:
        manifest.update(status="incomplete", error=f"{type(exc).__name__}: {exc}")
        print(manifest["error"])
        return 2
    finally:
        _write(root / "manifest.json", manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    replay = sub.add_parser("run")
    replay.add_argument("--mode", choices=MODES, required=True)
    replay.add_argument("--run-id", required=True)
    replay.add_argument("--ws-url")
    replay.add_argument("--timeout", type=float, default=180)
    replay.add_argument("--dry-run", action="store_true")
    replay.add_argument("--output-root", type=Path, default=ROOT / "judge_outputs" / "interference_v3")
    aggregate = sub.add_parser("compare")
    aggregate.add_argument("run_dirs", type=Path, nargs="+")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return run(args)
        print(json.dumps(compare_dirs(args.run_dirs), indent=2))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
