#!/usr/bin/env python3
"""Controlled latency benchmark for the project's STT backends.

This script deliberately keeps three measurements separate:

1. backend initialization (model/client construction),
2. first request after initialization (a client-side cold request), and
3. repeated warm ``backend.transcribe`` calls plus real-time factor (RTF).

It also has an ``endpoint`` command for measuring WebRTC-VAD endpoint delay in
audio time.  The model benchmark never includes disk loading, resampling, VAD,
microphone capture, playback, an LLM, or TTS in its timed interval.

The raw JSONL output is the source of truth.  Every cold, warm-up, measured, and
failed request is written immediately instead of retaining only averages.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import json
import math
import os
import platform
import random
import re
import socket
import sys
import time
import traceback
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np


TARGET_SAMPLE_RATE = 16_000
SUPPORTED_AUDIO_SUFFIXES = {".wav", ".flac"}
DEFAULT_SEED = 20_260_719


@dataclass
class AudioCase:
    audio_id: str
    path: Path
    audio: np.ndarray
    sample_rate: int
    duration_s: float
    load_latency_ms: float
    original_sample_rate: int
    original_channels: int
    canonical_audio_sha256: str
    reference_text: Optional[str] = None
    speech_end_s: Optional[float] = None
    speaker: Optional[str] = None
    condition: Optional[str] = None
    duration_bin: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return str(value)


def write_jsonl(handle, record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(jsonable(dict(record)), ensure_ascii=False) + "\n")
    handle.flush()


def normalize_for_wer(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"\w+(?:['’]\w+)?", text, flags=re.UNICODE)


def word_error_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    """Return total edit errors and reference-word count."""
    ref = normalize_for_wer(reference)
    hyp = normalize_for_wer(hypothesis)

    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            substitution = previous[j - 1] + int(ref_word != hyp_word)
            deletion = previous[j] + 1
            insertion = current[j - 1] + 1
            current.append(min(substitution, deletion, insertion))
        previous = current

    return previous[-1], len(ref)


def word_error_rate(reference: str, hypothesis: str) -> Optional[float]:
    """Calculate utterance-level word error rate."""
    errors, reference_words = word_error_counts(reference, hypothesis)

    if reference_words == 0:
        return 0.0 if errors == 0 else None

    return errors / reference_words


def load_audio_16k_mono(path: Path, require_canonical: bool) -> tuple[np.ndarray, dict[str, Any]]:
    """Load outside the inference timer and standardize to float32 mono/16 kHz."""
    try:
        import soundfile as sf
    except ImportError:
        sf = None

    t0 = time.perf_counter_ns()
    if sf is not None:
        data, original_sr = sf.read(str(path), dtype="float32", always_2d=True)
    elif path.suffix.lower() == ".wav":
        try:
            from scipy.io import wavfile
        except ImportError as exc:
            raise RuntimeError("Install scipy or soundfile to load WAV files") from exc
        original_sr, raw = wavfile.read(str(path))
        if np.issubdtype(raw.dtype, np.signedinteger):
            scale = float(max(abs(np.iinfo(raw.dtype).min), np.iinfo(raw.dtype).max))
            data = raw.astype(np.float32) / scale
        elif np.issubdtype(raw.dtype, np.unsignedinteger):
            info = np.iinfo(raw.dtype)
            midpoint = (info.max + 1) / 2.0
            data = (raw.astype(np.float32) - midpoint) / midpoint
        elif np.issubdtype(raw.dtype, np.floating):
            data = raw.astype(np.float32, copy=False)
        else:
            raise ValueError(f"Unsupported WAV dtype for {path}: {raw.dtype}")
        if data.ndim == 1:
            data = data[:, np.newaxis]
    else:
        raise RuntimeError("Install soundfile to load FLAC: pip install soundfile")

    original_channels = int(data.shape[1])

    if require_canonical and (original_sr != TARGET_SAMPLE_RATE or original_channels != 1):
        raise ValueError(
            f"{path} is {original_sr} Hz/{original_channels} channels; "
            "--require-canonical-audio requires 16 kHz mono files"
        )

    mono = data.mean(axis=1, dtype=np.float32)
    if original_sr != TARGET_SAMPLE_RATE:
        try:
            from scipy.signal import resample_poly
        except ImportError as exc:
            raise RuntimeError("Install scipy to resample non-16-kHz files") from exc
        divisor = math.gcd(int(original_sr), TARGET_SAMPLE_RATE)
        mono = resample_poly(
            mono,
            TARGET_SAMPLE_RATE // divisor,
            int(original_sr) // divisor,
        ).astype(np.float32, copy=False)

    mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
    mono = np.ascontiguousarray(np.clip(mono, -1.0, 1.0), dtype=np.float32)
    t1 = time.perf_counter_ns()
    return mono, {
        "load_latency_ms": (t1 - t0) / 1_000_000.0,
        "original_sample_rate": int(original_sr),
        "original_channels": original_channels,
    }


def read_manifest_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "path" not in reader.fieldnames:
            raise ValueError("Manifest must contain a 'path' column")
        return [dict(row) for row in reader]


def discover_case_rows(audio_dir: Optional[Path], manifest: Optional[Path]) -> list[dict[str, str]]:
    if manifest is not None:
        rows = read_manifest_rows(manifest)
        base = manifest.resolve().parent
        for row in rows:
            raw_path = Path((row.get("path") or "").strip())
            if not raw_path.is_absolute():
                raw_path = base / raw_path
            row["path"] = str(raw_path.resolve())
        return rows

    assert audio_dir is not None
    files = sorted(
        p.resolve()
        for p in audio_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
    )
    return [{"audio_id": p.stem, "path": str(p)} for p in files]


def optional_float(value: Optional[str]) -> Optional[float]:
    if value is None or not value.strip():
        return None
    return float(value)


def derived_duration_bin(duration_s: float) -> str:
    if duration_s < 5.0:
        return "short_lt_5s"
    if duration_s < 15.0:
        return "medium_5_to_15s"
    return "long_ge_15s"


def load_cases(args: argparse.Namespace) -> list[AudioCase]:
    audio_dir = Path(args.audio_dir).resolve() if getattr(args, "audio_dir", None) else None
    manifest = Path(args.manifest).resolve() if getattr(args, "manifest", None) else None
    rows = discover_case_rows(audio_dir, manifest)
    if not rows:
        raise RuntimeError("No WAV or FLAC inputs were found")

    cases: list[AudioCase] = []
    seen_ids: set[str] = set()
    for row in rows:
        path = Path(row["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
            raise ValueError(f"Unsupported audio type for {path}; use WAV or FLAC")

        audio_id = (row.get("audio_id") or path.stem).strip()
        if not audio_id:
            raise ValueError(f"Empty audio_id for {path}")
        if audio_id in seen_ids:
            raise ValueError(f"Duplicate audio_id in input set: {audio_id}")
        seen_ids.add(audio_id)

        audio, load_info = load_audio_16k_mono(path, args.require_canonical_audio)
        duration_s = len(audio) / TARGET_SAMPLE_RATE
        if duration_s < 0.15:
            raise ValueError(f"{path} is shorter than 150 ms")

        standard_fields = {
            "audio_id", "path", "reference_text", "speech_end_s",
            "speaker", "condition", "duration_bin",
        }
        extra = {k: v for k, v in row.items() if k not in standard_fields and v not in (None, "")}
        cases.append(
            AudioCase(
                audio_id=audio_id,
                path=path,
                audio=audio,
                sample_rate=TARGET_SAMPLE_RATE,
                duration_s=duration_s,
                load_latency_ms=float(load_info["load_latency_ms"]),
                original_sample_rate=int(load_info["original_sample_rate"]),
                original_channels=int(load_info["original_channels"]),
                canonical_audio_sha256=hashlib.sha256(audio.tobytes()).hexdigest(),
                reference_text=(row.get("reference_text") or "").strip() or None,
                speech_end_s=optional_float(row.get("speech_end_s")),
                speaker=(row.get("speaker") or "").strip() or None,
                condition=(row.get("condition") or "").strip() or None,
                duration_bin=(row.get("duration_bin") or "").strip() or derived_duration_bin(duration_s),
                metadata=extra,
            )
        )
    return sorted(cases, key=lambda case: case.audio_id)


def resolve_backend_factory(spec: str) -> tuple[Callable[..., Any], str]:
    candidates = (
        ["stt.stt_backends:create_backend", "stt_backends:create_backend"]
        if spec == "auto"
        else [spec]
    )
    failures: list[str] = []
    for candidate in candidates:
        try:
            module_name, function_name = candidate.split(":", 1)
            module = importlib.import_module(module_name)
            factory = getattr(module, function_name)
            return factory, candidate
        except Exception as exc:
            failures.append(f"{candidate}: {type(exc).__name__}: {exc}")
    raise ImportError("Could not import the STT backend factory:\n  " + "\n  ".join(failures))


def backend_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.backend == "faster-whisper":
        model_name = args.model or "large-v3"
        return {
            "model": model_name,
            "device": args.device,
            "device_index": args.device_index,
            "compute_type": args.compute_type,
            "beam_size": args.beam_size,
            "diag_dll": args.diag_dll,
        }
    if args.backend == "parakeet-http":
        return {
            "base_url": args.base_url or "http://127.0.0.1:8096",
            "timeout_s": args.timeout_s,
            "include_timestamps": False,
            "should_chunk": args.parakeet_should_chunk,
        }
    return {
        "base_url": args.base_url or "http://127.0.0.1:8095",
        "timeout_s": args.timeout_s,
    }


def model_label(args: argparse.Namespace) -> str:
    if args.model_label:
        return args.model_label
    if args.backend == "faster-whisper":
        return args.model or "large-v3"
    return "server-default"


def environment_record() -> dict[str, Any]:
    return {
        "timestamp_utc": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "logical_cpu_count": os.cpu_count(),
        "pid": os.getpid(),
    }


def case_record(case: AudioCase) -> dict[str, Any]:
    return {
        "audio_id": case.audio_id,
        "audio_path": str(case.path),
        "audio_duration_s": case.duration_s,
        "sample_rate": case.sample_rate,
        "sample_count": len(case.audio),
        "original_sample_rate": case.original_sample_rate,
        "original_channels": case.original_channels,
        "canonical_audio_sha256": case.canonical_audio_sha256,
        "audio_load_latency_ms": case.load_latency_ms,
        "speaker": case.speaker,
        "condition": case.condition,
        "duration_bin": case.duration_bin,
        "case_metadata": case.metadata,
    }


def build_schedule(cases: Sequence[AudioCase], repeats: int, seed: int) -> list[tuple[AudioCase, int]]:
    schedule = [(case, repeat_index) for case in cases for repeat_index in range(1, repeats + 1)]
    random.Random(seed).shuffle(schedule)
    return schedule


def schedule_digest(schedule: Sequence[tuple[AudioCase, int]]) -> str:
    payload = "\n".join(
        f"{case.audio_id}\t{case.canonical_audio_sha256}\t{repeat}"
        for case, repeat in schedule
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def measure_transcription(
    *,
    backend: Any,
    case: AudioCase,
    args: argparse.Namespace,
    run_id: str,
    phase: str,
    order_index: int,
    repeat_index: int,
    init_latency_ms: float,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "record_type": "trial",
        "run_id": run_id,
        "timestamp_utc": utc_now(),
        "phase": phase,
        "backend": args.backend,
        "model": model_label(args),
        "order_index": order_index,
        "repeat_index": repeat_index,
        **case_record(case),
    }

    start_ns = time.perf_counter_ns()
    try:
        result = backend.transcribe(
            audio=case.audio,
            sample_rate=TARGET_SAMPLE_RATE,
            language=args.language,
            context=None,
            return_words=False,
        )
        end_ns = time.perf_counter_ns()
        latency_ns = end_ns - start_ns
        latency_s = latency_ns / 1_000_000_000.0
        transcript = (getattr(result, "text", "") or "").strip()
        backend_internal = getattr(backend, "last_timing", None)

        record.update(
            {
                "ok": True,
                "latency_ns": latency_ns,
                "latency_ms": latency_s * 1000.0,
                "latency_s": latency_s,
                "rtf": latency_s / max(case.duration_s, 1e-12),
                "transcript": transcript if not args.omit_transcripts else None,
                "transcript_chars": len(transcript),
                "engine": getattr(result, "engine", getattr(backend, "engine", args.backend)),
                "wer": word_error_rate(case.reference_text, transcript)
                if case.reference_text is not None
                else None,
                "wer_errors": word_error_counts(case.reference_text, transcript)[0]
                if case.reference_text is not None
                else None,
                "wer_reference_words": word_error_counts(case.reference_text, transcript)[1]
                if case.reference_text is not None
                else None,
                "backend_internal_timing": jsonable(backend_internal)
                if isinstance(backend_internal, Mapping)
                else None,
            }
        )
        if phase == "cold":
            record["backend_init_latency_ms"] = init_latency_ms
            record["cold_total_latency_ms"] = init_latency_ms + (latency_s * 1000.0)
    except Exception as exc:
        end_ns = time.perf_counter_ns()
        record.update(
            {
                "ok": False,
                "latency_ns": end_ns - start_ns,
                "latency_ms": (end_ns - start_ns) / 1_000_000.0,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    return record


def default_run_output(backend: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("stt_results") / f"{backend}_{stamp}.jsonl"


def ensure_new_output(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}. Use --overwrite or choose another path.")


def run_model_benchmark(args: argparse.Namespace) -> int:
    cases = load_cases(args)
    noncanonical = [
        case for case in cases
        if case.original_sample_rate != TARGET_SAMPLE_RATE or case.original_channels != 1
    ]
    if noncanonical:
        print(
            f"[note] {len(noncanonical)} input(s) were standardized to 16-kHz mono before timing; "
            "audio loading/resampling is excluded from inference latency."
        )

    schedule = build_schedule(cases, args.repeats, args.seed)
    output = Path(args.output).resolve() if args.output else default_run_output(args.backend).resolve()
    ensure_new_output(output, args.overwrite)
    summary_output = (
        Path(args.summary_output).resolve()
        if args.summary_output
        else output.with_suffix(".summary.csv")
    )
    if summary_output == output:
        raise ValueError("Raw JSONL and summary CSV must use different paths")
    if summary_output.exists() and not args.overwrite:
        raise FileExistsError(f"Summary exists: {summary_output}. Use --overwrite or choose another path.")

    run_id = f"{args.backend}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
    config = backend_config(args)

    with output.open("w", encoding="utf-8", newline="\n") as handle:
        write_jsonl(
            handle,
            {
                "record_type": "run_metadata",
                "run_id": run_id,
                "backend": args.backend,
                "model": model_label(args),
                "backend_config": config,
                "language": args.language,
                "warmups": args.warmups,
                "repeats": args.repeats,
                "seed": args.seed,
                "schedule_sha256": schedule_digest(schedule),
                "case_count": len(cases),
                "measured_request_count": len(schedule),
                "timer": "time.perf_counter_ns",
                "timed_boundary": "backend.transcribe(audio already loaded and standardized)",
                "environment": environment_record(),
            },
        )

        init_start_ns = time.perf_counter_ns()
        try:
            factory, resolved_factory = resolve_backend_factory(args.factory)
            backend = factory(args.backend, config)
            init_end_ns = time.perf_counter_ns()
            init_latency_ms = (init_end_ns - init_start_ns) / 1_000_000.0
            write_jsonl(
                handle,
                {
                    "record_type": "initialization",
                    "run_id": run_id,
                    "timestamp_utc": utc_now(),
                    "backend": args.backend,
                    "model": model_label(args),
                    "factory": resolved_factory,
                    "ok": True,
                    "init_latency_ms": init_latency_ms,
                },
            )
        except Exception as exc:
            init_end_ns = time.perf_counter_ns()
            write_jsonl(
                handle,
                {
                    "record_type": "initialization",
                    "run_id": run_id,
                    "timestamp_utc": utc_now(),
                    "backend": args.backend,
                    "model": model_label(args),
                    "ok": False,
                    "init_latency_ms": (init_end_ns - init_start_ns) / 1_000_000.0,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            raise

        request_counter = 0
        if not args.skip_cold_request:
            cold_case = next((case for case in cases if case.audio_id == args.cold_audio_id), cases[0])
            if args.cold_audio_id and cold_case.audio_id != args.cold_audio_id:
                raise ValueError(f"Unknown --cold-audio-id: {args.cold_audio_id}")
            cold = measure_transcription(
                backend=backend,
                case=cold_case,
                args=args,
                run_id=run_id,
                phase="cold",
                order_index=request_counter,
                repeat_index=0,
                init_latency_ms=init_latency_ms,
            )
            write_jsonl(handle, cold)
            request_counter += 1
            print(
                f"[cold] init={init_latency_ms:.1f} ms, "
                f"first_request={cold.get('latency_ms', float('nan')):.1f} ms"
            )
            if not cold.get("ok") and args.fail_fast:
                raise RuntimeError(cold.get("error", "Cold request failed"))

        if args.cold_only:
            print(f"Raw results: {output}")
            return 0

        warm_rng = random.Random(args.seed + 1)
        warm_cases = list(cases)
        warm_rng.shuffle(warm_cases)
        for warmup_index in range(1, args.warmups + 1):
            case = warm_cases[(warmup_index - 1) % len(warm_cases)]
            trial = measure_transcription(
                backend=backend,
                case=case,
                args=args,
                run_id=run_id,
                phase="warmup",
                order_index=request_counter,
                repeat_index=warmup_index,
                init_latency_ms=init_latency_ms,
            )
            write_jsonl(handle, trial)
            request_counter += 1
            if not trial.get("ok") and args.fail_fast:
                raise RuntimeError(trial.get("error", "Warm-up request failed"))

        for measured_order, (case, repeat_index) in enumerate(schedule, start=1):
            trial = measure_transcription(
                backend=backend,
                case=case,
                args=args,
                run_id=run_id,
                phase="measured",
                order_index=measured_order,
                repeat_index=repeat_index,
                init_latency_ms=init_latency_ms,
            )
            write_jsonl(handle, trial)
            state = "ok" if trial.get("ok") else "ERROR"
            latency = trial.get("latency_ms", float("nan"))
            rtf = trial.get("rtf", float("nan"))
            print(
                f"[{measured_order:04d}/{len(schedule):04d}] {state:5s} "
                f"{case.audio_id} repeat={repeat_index} latency={latency:.1f} ms rtf={rtf:.4f}"
            )
            if not trial.get("ok") and args.fail_fast:
                raise RuntimeError(trial.get("error", "Measured request failed"))
            if args.settle_ms > 0:
                time.sleep(args.settle_ms / 1000.0)

    rows = summarize_jsonl_files(
        [output],
        include_per_audio=True,
        manifest=Path(args.manifest).resolve(),
    )
    write_summary_csv(summary_output, rows)
    print_summary(rows)
    print(f"Raw results: {output}")
    print(f"Summary:     {summary_output}")

    del backend
    gc.collect()
    return 0


SUMMARY_FIELDS = [
    "backend", "model", "phase", "scope", "scope_value", "n_success", "n_error",
    "latency_p50_ms", "latency_p95_ms", "latency_mean_ms", "latency_min_ms", "latency_max_ms",
    "rtf_p50", "rtf_p95", "wer_mean", "wer_corpus", 
    "cold_total_p50_ms", "warning",
]


def stats_row(
    records: Sequence[dict[str, Any]],
    *,
    backend: str,
    model: str,
    phase: str,
    scope: str,
    scope_value: str,
) -> dict[str, Any]:
    good = [record for record in records if record.get("ok") and record.get("latency_ms") is not None]
    latency = [float(record["latency_ms"]) for record in good]
    rtf = [float(record["rtf"]) for record in good if record.get("rtf") is not None]
    wer = [float(record["wer"]) for record in good if record.get("wer") is not None]

    wer_errors = sum(
        int(record["wer_errors"])
        for record in good
        if record.get("wer_errors") is not None
    )

    wer_reference_words = sum(
        int(record["wer_reference_words"])
        for record in good
        if record.get("wer_reference_words") is not None
    )

    wer_corpus = (
        wer_errors / wer_reference_words
        if wer_reference_words > 0
        else None
    )

    cold_total = [
        float(record["cold_total_latency_ms"])
        for record in good
        if record.get("cold_total_latency_ms") is not None
    ]
    warning = ""
    if phase == "measured" and len(good) < 20:
        warning = "p95 is unstable with fewer than 20 successful requests"
    return {
        "backend": backend,
        "model": model,
        "phase": phase,
        "scope": scope,
        "scope_value": scope_value,
        "n_success": len(good),
        "n_error": len(records) - len(good),
        "latency_p50_ms": percentile(latency, 50),
        "latency_p95_ms": percentile(latency, 95),
        "latency_mean_ms": mean_or_none(latency),
        "latency_min_ms": min(latency) if latency else None,
        "latency_max_ms": max(latency) if latency else None,
        "rtf_p50": percentile(rtf, 50),
        "rtf_p95": percentile(rtf, 95),
        "wer_mean": mean_or_none(wer),
        "wer_corpus": wer_corpus,
        "cold_total_p50_ms": percentile(cold_total, 50),
        "warning": warning,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path}:{line_number}: {exc}") from exc
    return records


def summarize_jsonl_files(
    paths: Sequence[Path],
    include_per_audio: bool,
    manifest: Path,
) -> list[dict[str, Any]]:
    records = [record for path in paths for record in read_jsonl(path)]

    manifest_rows = read_manifest_rows(manifest)
    references = {
        (row.get("audio_id") or Path(row["path"]).stem).strip():
        (row.get("reference_text") or "").strip()
        for row in manifest_rows
        if (row.get("reference_text") or "").strip()
    }

    for record in records:
        if record.get("record_type") != "trial" or not record.get("ok"):
            continue

        reference = references.get(str(record.get("audio_id") or ""))
        transcript = record.get("transcript")

        if reference is None or transcript is None:
            continue

        errors, reference_words = word_error_counts(reference, str(transcript))
        record["wer_errors"] = errors
        record["wer_reference_words"] = reference_words
        record["wer"] = (
            errors / reference_words
            if reference_words > 0
            else (0.0 if errors == 0 else None)
        )
    rows: list[dict[str, Any]] = []

    initialization = [r for r in records if r.get("record_type") == "initialization"]
    init_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in initialization:
        converted = dict(record)
        converted["latency_ms"] = record.get("init_latency_ms")
        init_groups.setdefault((record.get("backend", "unknown"), record.get("model", "unknown")), []).append(converted)
    for (backend, model), group in sorted(init_groups.items()):
        rows.append(stats_row(group, backend=backend, model=model, phase="initialization", scope="overall", scope_value="all"))

    trials = [r for r in records if r.get("record_type") == "trial"]
    model_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in trials:
        model_groups.setdefault((record.get("backend", "unknown"), record.get("model", "unknown")), []).append(record)

    for (backend, model), model_records in sorted(model_groups.items()):
        cold = [r for r in model_records if r.get("phase") == "cold"]
        if cold:
            rows.append(stats_row(cold, backend=backend, model=model, phase="cold", scope="overall", scope_value="all"))

        measured = [r for r in model_records if r.get("phase") == "measured"]
        if not measured:
            continue
        rows.append(stats_row(measured, backend=backend, model=model, phase="measured", scope="overall", scope_value="all"))

        bins = sorted({str(r.get("duration_bin") or "unspecified") for r in measured})
        for duration_bin in bins:
            group = [r for r in measured if str(r.get("duration_bin") or "unspecified") == duration_bin]
            rows.append(
                stats_row(
                    group,
                    backend=backend,
                    model=model,
                    phase="measured",
                    scope="duration_bin",
                    scope_value=duration_bin,
                )
            )

        if include_per_audio:
            audio_ids = sorted({str(r.get("audio_id")) for r in measured})
            for audio_id in audio_ids:
                group = [r for r in measured if str(r.get("audio_id")) == audio_id]
                rows.append(
                    stats_row(
                        group,
                        backend=backend,
                        model=model,
                        phase="measured",
                        scope="audio",
                        scope_value=audio_id,
                    )
                )
    return rows


def write_summary_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(rows: Sequence[dict[str, Any]]) -> None:
    print("\nPrimary warm-inference results (overall):")
    primary = [
        row for row in rows
        if row["phase"] == "measured"
        and row["scope"] == "overall"
        and int(row["n_success"]) > 0
    ]
    if not primary:
        print("  No successful measured requests.")
        return
    for row in primary:
        print(
            f"  {row['backend']} / {row['model']}: n={row['n_success']}, "
            f"p50={row['latency_p50_ms']:.1f} ms, p95={row['latency_p95_ms']:.1f} ms, "
            f"RTF p50={row['rtf_p50']:.4f}, RTF p95={row['rtf_p95']:.4f}"
        )


def run_summarize(args: argparse.Namespace) -> int:
    paths = [Path(path).resolve() for path in args.inputs]
    manifest = Path(args.manifest).resolve()
    rows = summarize_jsonl_files(
        paths,
        include_per_audio=not args.no_per_audio,
        manifest=manifest,
    )
    output = Path(args.output).resolve()
    if output in paths:
        raise ValueError("Combined summary output must not overwrite a raw JSONL input")
    ensure_new_output(output, args.overwrite)
    write_summary_csv(output, rows)
    print_summary(rows)
    print(f"Combined summary: {output}")
    return 0


def resolve_vad_module(spec: str):
    candidates = ["stt.vad_segmenter", "vad_segmenter"] if spec == "auto" else [spec]
    failures: list[str] = []
    for module_name in candidates:
        try:
            return importlib.import_module(module_name), module_name
        except Exception as exc:
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
    raise ImportError("Could not import VAD module:\n  " + "\n  ".join(failures))


def frames_pcm16(audio: np.ndarray, frame_samples: int) -> Iterable[bytes]:
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    for start in range(0, len(pcm16), frame_samples):
        frame = pcm16[start : start + frame_samples]
        if len(frame) < frame_samples:
            padded = np.zeros(frame_samples, dtype=np.int16)
            padded[: len(frame)] = frame
            frame = padded
        yield frame.tobytes()


def endpoint_one_case(case: AudioCase, vad_module: Any, aggressiveness: int) -> dict[str, Any]:
    if case.speech_end_s is None:
        raise ValueError(f"{case.audio_id} has no speech_end_s in the manifest")
    if not 0.0 <= case.speech_end_s <= case.duration_s:
        raise ValueError(
            f"{case.audio_id}: speech_end_s={case.speech_end_s} is outside "
            f"the {case.duration_s:.3f}s audio file"
        )

    frame_ms = int(getattr(vad_module, "FRAME_MS", 20))
    frame_samples = int(getattr(vad_module, "FRAME_SAMPLES", TARGET_SAMPLE_RATE * frame_ms // 1000))
    segmenter = vad_module.VADSegmenter(aggressiveness=aggressiveness)
    emissions: list[dict[str, Any]] = []

    processing_start = time.perf_counter_ns()
    for frame_index, frame in enumerate(frames_pcm16(case.audio, frame_samples)):
        frame_start_s = frame_index * frame_ms / 1000.0
        utterance = segmenter.process_frame(frame, frame_start_s)
        if utterance is not None:
            emissions.append(
                {
                    "utterance": utterance,
                    "emitted_at_audio_s": (frame_index + 1) * frame_ms / 1000.0,
                }
            )
    processing_end = time.perf_counter_ns()

    matching = [
        item for item in emissions
        if float(getattr(item["utterance"], "t_start", 0.0)) <= case.speech_end_s
        <= item["emitted_at_audio_s"]
    ]
    if not matching:
        return {
            "ok": False,
            "error": (
                "VAD did not emit the annotated utterance. Append more trailing silence "
                "than MAX_SILENCE_MS; do not use flush() for endpoint latency."
            ),
            "emission_count": len(emissions),
            "vad_processing_cpu_ms": (processing_end - processing_start) / 1_000_000.0,
        }

    selected = matching[0]
    utterance = selected["utterance"]
    emitted_at = float(selected["emitted_at_audio_s"])
    return {
        "ok": True,
        "speech_end_s": case.speech_end_s,
        "vad_emitted_at_audio_s": emitted_at,
        "endpoint_delay_ms": (emitted_at - case.speech_end_s) * 1000.0,
        "emitted_audio_duration_s": len(utterance.audio) / TARGET_SAMPLE_RATE,
        "vad_processing_cpu_ms": (processing_end - processing_start) / 1_000_000.0,
        "emission_count": len(emissions),
    }


def run_endpoint_benchmark(args: argparse.Namespace) -> int:
    cases = load_cases(args)
    vad_module, resolved_module = resolve_vad_module(args.vad_module)
    output = Path(args.output).resolve()
    ensure_new_output(output, args.overwrite)
    summary_output = (
        Path(args.summary_output).resolve()
        if args.summary_output
        else output.with_suffix(".summary.csv")
    )
    if summary_output == output:
        raise ValueError("Raw endpoint JSONL and summary CSV must use different paths")
    if summary_output.exists() and not args.overwrite:
        raise FileExistsError(f"Summary exists: {summary_output}. Use --overwrite.")

    records: list[dict[str, Any]] = []
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        write_jsonl(
            handle,
            {
                "record_type": "endpoint_metadata",
                "timestamp_utc": utc_now(),
                "vad_module": resolved_module,
                "aggressiveness": args.vad_aggressiveness,
                "frame_ms": int(getattr(vad_module, "FRAME_MS", 20)),
                "max_silence_ms": getattr(vad_module, "MAX_SILENCE_MS", None),
                "metric": "audio-time delay from annotated last speech to VAD utterance emission",
            },
        )
        for case in cases:
            base = {
                "record_type": "endpoint_trial",
                "timestamp_utc": utc_now(),
                **case_record(case),
            }
            try:
                record = {**base, **endpoint_one_case(case, vad_module, args.vad_aggressiveness)}
            except Exception as exc:
                record = {
                    **base,
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            records.append(record)
            write_jsonl(handle, record)
            if record.get("ok"):
                print(f"{case.audio_id}: endpoint_delay={record['endpoint_delay_ms']:.1f} ms")
            else:
                print(f"{case.audio_id}: ERROR {record.get('error')}")

    good = [record for record in records if record.get("ok")]
    values = [float(record["endpoint_delay_ms"]) for record in good]
    summary_row = {
        "n_success": len(good),
        "n_error": len(records) - len(good),
        "endpoint_delay_p50_ms": percentile(values, 50),
        "endpoint_delay_p95_ms": percentile(values, 95),
        "endpoint_delay_mean_ms": mean_or_none(values),
        "endpoint_delay_min_ms": min(values) if values else None,
        "endpoint_delay_max_ms": max(values) if values else None,
        "warning": "p95 is unstable with fewer than 20 files" if len(good) < 20 else "",
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    with summary_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)
    print(f"Raw endpoint results: {output}")
    print(f"Endpoint summary:     {summary_output}")
    return 0


def add_audio_source_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio-dir", help="Directory containing WAV/FLAC test files")
    source.add_argument("--manifest", help="CSV manifest; paths are relative to the CSV file")
    parser.add_argument(
        "--require-canonical-audio",
        action="store_true",
        help="Reject anything other than 16-kHz mono instead of standardizing before timing",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Controlled latency benchmark for faster-whisper, Parakeet HTTP, and Mega-ASR HTTP"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Benchmark one backend in one isolated process")
    add_audio_source_arguments(run)
    run.add_argument(
        "--backend",
        required=True,
        choices=["faster-whisper", "parakeet-http", "mega-asr-http"],
    )
    run.add_argument(
        "--factory",
        default="auto",
        help="Backend factory as module:function; auto tries stt.stt_backends then stt_backends",
    )
    run.add_argument("--model", default=None, help="Faster-Whisper model/path (default: large-v3)")
    run.add_argument("--model-label", default=None, help="Descriptive server model name stored in results")
    run.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    run.add_argument("--device-index", type=int, default=0)
    run.add_argument("--compute-type", default="int8_float16")
    run.add_argument("--beam-size", type=int, default=10)
    run.add_argument("--diag-dll", action="store_true")
    run.add_argument("--base-url", default=None)
    run.add_argument("--timeout-s", type=float, default=120.0)
    run.add_argument("--parakeet-should-chunk", action="store_true")
    run.add_argument("--language", default="en")
    run.add_argument("--warmups", type=int, default=5)
    run.add_argument("--repeats", type=int, default=5)
    run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run.add_argument("--settle-ms", type=float, default=0.0)
    run.add_argument("--cold-audio-id", default=None)
    run.add_argument("--skip-cold-request", action="store_true")
    run.add_argument("--cold-only", action="store_true")
    run.add_argument("--omit-transcripts", action="store_true")
    run.add_argument("--fail-fast", action="store_true")
    run.add_argument("--output", default=None)
    run.add_argument("--summary-output", default=None)
    run.add_argument("--overwrite", action="store_true")
    run.set_defaults(func=run_model_benchmark)

    summarize = subparsers.add_parser("summarize", help="Combine one or more raw JSONL runs")
    summarize.add_argument("inputs", nargs="+", help="Raw JSONL run files")
    summarize.add_argument(
        "--manifest",
        required=True,
        help="CSV manifest containing audio_id and reference_text",
    )
    summarize.add_argument("--output", required=True, help="Combined CSV output")
    summarize.add_argument("--no-per-audio", action="store_true")
    summarize.add_argument("--overwrite", action="store_true")
    summarize.set_defaults(func=run_summarize)

    endpoint = subparsers.add_parser("endpoint", help="Measure VAD/endpointer delay separately")
    add_audio_source_arguments(endpoint)
    endpoint.add_argument("--vad-module", default="auto")
    endpoint.add_argument("--vad-aggressiveness", type=int, choices=[0, 1, 2, 3], default=3)
    endpoint.add_argument("--output", required=True)
    endpoint.add_argument("--summary-output", default=None)
    endpoint.add_argument("--overwrite", action="store_true")
    endpoint.set_defaults(func=run_endpoint_benchmark)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.command == "run":
        if args.warmups < 0:
            raise ValueError("--warmups must be non-negative")
        if args.repeats < 1:
            raise ValueError("--repeats must be at least 1")
        if args.cold_only and args.skip_cold_request:
            raise ValueError("--cold-only and --skip-cold-request cannot be combined")
        if args.settle_ms < 0:
            raise ValueError("--settle-ms must be non-negative")


def main(argv: Optional[Sequence[str]] = None) -> int:
    # When this file is invoked by path from a subdirectory, Python otherwise
    # puts only the script's directory on sys.path.  The STT packages normally
    # live under the current project root.
    project_root = str(Path.cwd().resolve())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    args = build_parser().parse_args(argv)
    validate_args(args)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
