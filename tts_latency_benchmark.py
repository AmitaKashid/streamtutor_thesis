#!/usr/bin/env python3
"""
Independent TTS engine latency benchmark for llmtutor.

Measures synthesis only: no WAV saving and no audio playback.

Batch metrics:
  - total synthesis latency
  - TTFA-ready (equal to total latency because the API returns a full waveform)
  - generated-audio duration
  - synthesis RTF = synthesis latency / generated-audio duration

Streaming metrics:
  - TTFA-ready (first non-empty PCM chunk received by this client)
  - total stream latency (final chunk received)
  - generated-audio duration
  - synthesis RTF = total stream latency / generated-audio duration
  - chunk count and inter-arrival gaps

Examples (run from the llmtutor project root):
  python tts_latency_benchmark.py --engine kokoro --mode batch
  python tts_latency_benchmark.py --engine chatterbox --mode batch --voice-prompt sample.wav
  python tts_latency_benchmark.py --engine qwen-http --mode batch
  python tts_latency_benchmark.py --engine qwen-stream --mode stream
  python tts_latency_benchmark.py --engine chatterbox-http --mode stream

The local Kokoro and local Qwen stream_text implementations generate the full
waveform before slicing it. Their stream mode is therefore labelled
"chunked_delivery", not native streaming.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import numpy as np


ENGINE_CHOICES = (
    "kokoro",
    "chatterbox",
    "chatterbox-http",
    "qwen-local",
    "qwen-http",
    "qwen-stream",
)

NATIVE_STREAM_ENGINES = {"chatterbox-http", "qwen-stream"}
CHUNKED_DELIVERY_ENGINES = {"kokoro", "qwen-local"}
BATCH_ENGINES = {"kokoro", "chatterbox", "qwen-local", "qwen-http"}


@dataclass(frozen=True)
class ManifestItem:
    text_id: str
    category: str
    text: str
    word_count: int
    char_count: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def synchronize_cuda() -> None:
    """Synchronize CUDA when torch is installed and CUDA is active."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except (ImportError, RuntimeError):
        pass


def percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def load_manifest(path: Path, limit: Optional[int] = None) -> list[ManifestItem]:
    if not path.is_file():
        raise FileNotFoundError(f"TTS manifest not found: {path}")

    items: list[ManifestItem] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"text_id", "category", "text", "word_count", "char_count"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        for row in reader:
            text_id = (row["text_id"] or "").strip()
            text = (row["text"] or "").strip()
            if not text_id or not text:
                raise ValueError("Manifest contains a blank text_id or text.")
            if text_id in seen:
                raise ValueError(f"Duplicate text_id in manifest: {text_id}")
            seen.add(text_id)
            items.append(
                ManifestItem(
                    text_id=text_id,
                    category=(row["category"] or "").strip(),
                    text=text,
                    word_count=int(row["word_count"]),
                    char_count=int(row["char_count"]),
                )
            )

    if limit is not None:
        items = items[:limit]
    if not items:
        raise ValueError("Manifest contains no benchmark texts.")
    return items


def build_engine(name: str) -> Any:
    """Import only the selected backend so unrelated dependencies are optional."""
    if name == "kokoro":
        from tts.kokoro_engine import KokoroEngine

        return KokoroEngine()
    if name == "chatterbox":
        from tts.chatterbox_engine import ChatterboxEngine

        return ChatterboxEngine()
    if name == "chatterbox-http":
        from tts.elevenlabs_streaming_engine import ChatterboxHTTPStreamingEngine

        return ChatterboxHTTPStreamingEngine()
    if name == "qwen-local":
        from tts.qwen3tts_engine import Qwen3TTSEngine

        return Qwen3TTSEngine()
    if name == "qwen-http":
        from tts.qwen3_http_engine import Qwen3HTTPEngine

        return Qwen3HTTPEngine()
    if name == "qwen-stream":
        from tts.qwen3_streaming_engine import Qwen3StreamingEngine

        return Qwen3StreamingEngine()
    raise ValueError(f"Unknown engine: {name}")


def resolve_mode(engine_name: str, requested_mode: str) -> tuple[str, str]:
    if requested_mode == "auto":
        requested_mode = "stream" if engine_name in NATIVE_STREAM_ENGINES else "batch"

    if requested_mode == "batch" and engine_name not in BATCH_ENGINES:
        raise ValueError(f"{engine_name} does not expose a batch synthesis API.")
    if requested_mode == "stream" and engine_name == "chatterbox":
        raise ValueError("Local Chatterbox exposes batch synthesis only.")
    if requested_mode == "stream" and engine_name == "qwen-http":
        raise ValueError("Qwen HTTP returns a complete WAV and is batch-only.")

    if requested_mode == "batch":
        classification = "batch"
    elif engine_name in NATIVE_STREAM_ENGINES:
        classification = "native_streaming"
    elif engine_name in CHUNKED_DELIVERY_ENGINES:
        classification = "chunked_delivery"
    else:
        raise ValueError(f"Unsupported engine/mode combination: {engine_name}/{requested_mode}")
    return requested_mode, classification


def sample_count(wav: Any) -> int:
    try:
        import torch

        if torch.is_tensor(wav):
            return int(wav.numel())
    except ImportError:
        pass
    return int(np.asarray(wav).size)


def one_text_event(text: str) -> Iterable[Any]:
    from tts.base_engine import StreamingTextChunk

    yield StreamingTextChunk(text=text, flush=True, is_final=True)


def run_batch_trial(
    engine: Any,
    item: ManifestItem,
    voice_prompt: Optional[Path],
) -> dict[str, Any]:
    synchronize_cuda()
    start = time.perf_counter()
    wav, sample_rate = engine.synthesize(
        text=item.text,
        voice_prompt=voice_prompt,
        config=None,
        save_to=None,
    )
    synchronize_cuda()
    end = time.perf_counter()

    samples = sample_count(wav)
    if samples <= 0 or int(sample_rate) <= 0:
        raise RuntimeError("Engine returned empty audio or an invalid sample rate.")
    total_s = end - start
    audio_s = samples / int(sample_rate)
    return {
        "ttfa_ready_ms": total_s * 1000.0,
        "synthesis_total_ms": total_s * 1000.0,
        "audio_duration_s": audio_s,
        "rtf": total_s / audio_s,
        "sample_rate": int(sample_rate),
        "sample_count": samples,
        "audio_chunk_count": 1,
        "chunk_gap_p50_ms": None,
        "chunk_gap_p95_ms": None,
        "chunk_gap_max_ms": None,
    }


def run_stream_trial(
    engine: Any,
    item: ManifestItem,
    voice_prompt: Optional[Path],
) -> dict[str, Any]:
    arrival_times: list[float] = []
    samples = 0
    sample_rate: Optional[int] = None

    synchronize_cuda()
    start = time.perf_counter()
    iterator: Iterator[tuple[np.ndarray, int]] = engine.stream_text(
        one_text_event(item.text),
        voice_prompt=voice_prompt,
        config=None,
    )
    for audio, sr in iterator:
        now = time.perf_counter()
        count = int(np.asarray(audio).size)
        if count <= 0:
            continue
        if sample_rate is not None and int(sr) != sample_rate:
            raise RuntimeError(f"Sample rate changed within one stream: {sample_rate} -> {sr}")
        sample_rate = int(sr)
        samples += count
        arrival_times.append(now)
    synchronize_cuda()
    end = time.perf_counter()

    if not arrival_times or samples <= 0 or not sample_rate:
        raise RuntimeError("Streaming engine returned no playable PCM.")

    gaps_ms = [
        (current - previous) * 1000.0
        for previous, current in zip(arrival_times, arrival_times[1:])
    ]
    total_s = end - start
    audio_s = samples / sample_rate
    return {
        "ttfa_ready_ms": (arrival_times[0] - start) * 1000.0,
        "synthesis_total_ms": total_s * 1000.0,
        "audio_duration_s": audio_s,
        "rtf": total_s / audio_s,
        "sample_rate": sample_rate,
        "sample_count": samples,
        "audio_chunk_count": len(arrival_times),
        "chunk_gap_p50_ms": percentile(gaps_ms, 0.50),
        "chunk_gap_p95_ms": percentile(gaps_ms, 0.95),
        "chunk_gap_max_ms": max(gaps_ms) if gaps_ms else None,
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def measured_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_type") == "trial" and record.get("phase") == "measured":
                records.append(record)
    return records


def summarize_group(
    records: list[dict[str, Any]],
    engine: str,
    mode: str,
    stream_classification: str,
    category: str,
) -> dict[str, Any]:
    successful = [r for r in records if r.get("ok")]
    attempted = len(records)

    def values(key: str) -> list[float]:
        return [float(r[key]) for r in successful if r.get(key) is not None]

    ttfa = values("ttfa_ready_ms")
    total = values("synthesis_total_ms")
    rtf = values("rtf")
    audio = values("audio_duration_s")
    gaps = values("chunk_gap_p95_ms")
    return {
        "engine": engine,
        "mode": mode,
        "stream_classification": stream_classification,
        "category": category,
        "n_attempted": attempted,
        "n_success": len(successful),
        "n_failed": attempted - len(successful),
        "failure_rate_percent": ((attempted - len(successful)) / attempted * 100.0) if attempted else None,
        "ttfa_p50_ms": percentile(ttfa, 0.50),
        "ttfa_p90_ms": percentile(ttfa, 0.90),
        "ttfa_p95_ms": percentile(ttfa, 0.95),
        "synthesis_p50_ms": percentile(total, 0.50),
        "synthesis_p95_ms": percentile(total, 0.95),
        "rtf_p50": percentile(rtf, 0.50),
        "rtf_p95": percentile(rtf, 0.95),
        "audio_duration_p50_s": percentile(audio, 0.50),
        "chunk_gap_p95_trial_p50_ms": percentile(gaps, 0.50),
    }


def write_summary(
    jsonl_path: Path,
    summary_path: Path,
    engine: str,
    mode: str,
    stream_classification: str,
) -> None:
    records = measured_records(jsonl_path)
    rows = [summarize_group(records, engine, mode, stream_classification, "ALL")]
    categories = sorted({str(r["category"]) for r in records})
    for category in categories:
        subset = [r for r in records if r["category"] == category]
        rows.append(summarize_group(subset, engine, mode, stream_classification, category))

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark llmtutor TTS engine latency.")
    parser.add_argument("--engine", required=True, choices=ENGINE_CHOICES)
    parser.add_argument("--mode", choices=("auto", "batch", "stream"), default="auto")
    parser.add_argument("--manifest", type=Path, default=Path("tts_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/tts"))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N texts for a smoke test.")
    parser.add_argument("--voice-prompt", type=Path, default=None)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing engine result file. Default refuses to overwrite it.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue after a failed request. The failed trial is still recorded.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repetitions < 1 or args.warmups < 0:
        raise ValueError("--repetitions must be >= 1 and --warmups must be >= 0.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1.")
    if args.voice_prompt is not None and not args.voice_prompt.is_file():
        raise FileNotFoundError(f"Voice prompt not found: {args.voice_prompt}")

    mode, stream_classification = resolve_mode(args.engine, args.mode)
    items = load_manifest(args.manifest, args.limit)
    stem = f"{args.engine}_{stream_classification}"
    jsonl_path = args.output_dir / f"{stem}.jsonl"
    summary_path = args.output_dir / f"{stem}_summary.csv"

    if jsonl_path.exists() and not args.append:
        raise FileExistsError(
            f"Result file already exists: {jsonl_path}. "
            "Move it, choose another --output-dir, or explicitly pass --append."
        )

    print(f"[benchmark] engine={args.engine} mode={mode} classification={stream_classification}")
    print(f"[benchmark] texts={len(items)} repetitions={args.repetitions} warmups={args.warmups}")
    print(f"[benchmark] output={jsonl_path}")

    engine = build_engine(args.engine)
    synchronize_cuda()
    load_start = time.perf_counter()
    engine.load()
    synchronize_cuda()
    load_ms = (time.perf_counter() - load_start) * 1000.0

    append_jsonl(
        jsonl_path,
        {
            "record_type": "run_metadata",
            "created_at_utc": utc_now(),
            "engine": args.engine,
            "mode": mode,
            "stream_classification": stream_classification,
            "manifest": str(args.manifest.resolve()),
            "manifest_text_count": len(items),
            "repetitions": args.repetitions,
            "warmups": args.warmups,
            "seed": args.seed,
            "model_load_ms": load_ms,
            "voice_prompt": str(args.voice_prompt.resolve()) if args.voice_prompt else None,
            "python": sys.version,
            "rtf_definition": "synthesis_total_s / generated_audio_duration_s",
        },
    )
    print(f"[benchmark] engine.load()={load_ms:.1f} ms")

    trial_function = run_batch_trial if mode == "batch" else run_stream_trial

    for warmup_index in range(args.warmups):
        item = items[warmup_index % len(items)]
        print(f"[warmup {warmup_index + 1}/{args.warmups}] {item.text_id}")
        trial_function(engine, item, args.voice_prompt)

    schedule = [
        (repeat_index, item)
        for repeat_index in range(1, args.repetitions + 1)
        for item in items
    ]
    random.Random(args.seed).shuffle(schedule)

    failures = 0
    for trial_index, (repeat_index, item) in enumerate(schedule, start=1):
        base = {
            "record_type": "trial",
            "created_at_utc": utc_now(),
            "engine": args.engine,
            "mode": mode,
            "stream_classification": stream_classification,
            "phase": "measured",
            "trial_index": trial_index,
            "repeat_index": repeat_index,
            **asdict(item),
        }
        print(f"[{trial_index}/{len(schedule)}] {item.text_id} repeat={repeat_index}")
        try:
            metrics = trial_function(engine, item, args.voice_prompt)
            record = {**base, **metrics, "ok": True, "error_type": None, "error": None}
            print(
                f"  TTFA={metrics['ttfa_ready_ms']:.1f} ms | "
                f"total={metrics['synthesis_total_ms']:.1f} ms | "
                f"audio={metrics['audio_duration_s']:.3f} s | "
                f"RTF={metrics['rtf']:.4f} | chunks={metrics['audio_chunk_count']}"
            )
        except Exception as exc:
            failures += 1
            record = {
                **base,
                "ttfa_ready_ms": None,
                "synthesis_total_ms": None,
                "audio_duration_s": None,
                "rtf": None,
                "sample_rate": None,
                "sample_count": None,
                "audio_chunk_count": None,
                "chunk_gap_p50_ms": None,
                "chunk_gap_p95_ms": None,
                "chunk_gap_max_ms": None,
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            append_jsonl(jsonl_path, record)
            if not args.continue_on_error:
                write_summary(jsonl_path, summary_path, args.engine, mode, stream_classification)
                print(f"[benchmark] partial summary={summary_path}")
                return 1
            continue
        append_jsonl(jsonl_path, record)

    write_summary(jsonl_path, summary_path, args.engine, mode, stream_classification)
    print(f"[benchmark] complete successes={len(schedule) - failures} failures={failures}")
    print(f"[benchmark] raw={jsonl_path}")
    print(f"[benchmark] summary={summary_path}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
