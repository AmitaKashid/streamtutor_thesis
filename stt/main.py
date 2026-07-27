# main.py
from __future__ import annotations
import argparse
import queue
import threading
import time
from collections import deque
import numpy as np

from stt.vad_segmenter import VADSegmenter, TARGET_SR
from stt_backends import create_backend

from pathlib import Path


from audio_file import load_audio_mono_16k
from stt.audio_frames import float32_to_frames_int16, frame_index_to_ts

def transcribe_one_file(path: str, backend, args) -> str:
    audio, sr = load_audio_mono_16k(path)  # float32 mono, resampled to 16k inside
    if sr != TARGET_SR:
        raise RuntimeError(f"Expected {TARGET_SR} Hz after loading, got {sr} Hz")

    # Option A: transcribe whole file (no VAD)
    if args.no_vad:
        result = backend.transcribe(
            audio=audio,
            sample_rate=TARGET_SR,
            language=args.language,
            context=None,
            return_words=args.print_words,
        )
        return result.text

    # Option B: VAD split into utterances, then transcribe each
    segmenter = VADSegmenter(aggressiveness=args.vad_aggr)
    texts = []

    # Rolling context across utterances (optional)
    context_tail = deque(maxlen=80)


    for i, frame_bytes in enumerate(float32_to_frames_int16(audio)):
        ts = frame_index_to_ts(i)
        utt = segmenter.process_frame(frame_bytes, ts)
        if utt is None:
            continue

        # Utterance just completed (file-time equivalent of mic speech end)
        t_utt_done = time.perf_counter()

        utt_audio = utt.audio.astype(np.float32) / 32768.0
        audio_len_s = len(utt_audio) / TARGET_SR

        print(
            f"\n[utt {utt.index}] "
            f"{max(0.0, utt.t_start):.2f}s → {max(utt.t_end, 0.0):.2f}s "
            f"(len {audio_len_s:.2f}s)"
        )

        # Optional rolling context (E3)
        context_prompt = None
        if args.keep_context and len(context_tail) > 0:
            context_prompt = "Previous context: " + " ".join(list(context_tail)[-30:]) + "."

        # ---- model inference timing ----
        t_start = time.perf_counter()
        result = backend.transcribe(
            audio=utt_audio,
            sample_rate=TARGET_SR,
            language=args.language,
            context=context_prompt,
            return_words=args.print_words,
        )
        t_end = time.perf_counter()

        processing_ms = (t_end - t_start) * 1000.0
        latency_ms = (t_end - t_utt_done) * 1000.0

        # Real-Time Factor (offline-only metric)
        rtf = (t_end - t_start) / max(audio_len_s, 1e-6)

        # Print segments (same format as mic mode)
        for seg in result.segments:
            if args.print_words and seg.words:
                words_str = " ".join(
                    [f"[{w.start:.2f}-{w.end:.2f}] {w.word}" for w in seg.words]
                )
                print(f"  {seg.start:.2f}-{seg.end:.2f}: {words_str}")
            else:
                print(f"  {seg.start:.2f}-{seg.end:.2f}: {seg.text}")

        if result.text:
            texts.append(result.text)

            print(
                f"[text] [{result.engine}] "
                f"[proc={processing_ms:.0f} ms | lat={latency_ms:.0f} ms | rtf={rtf:.2f}] "
                f"{result.text}"
            )

            # update rolling context
            if args.keep_context:
                for w in result.text.split():
                    context_tail.append(w)
        else:
            print(
                f"[text] [{result.engine}] "
                f"[proc={processing_ms:.0f} ms | lat={latency_ms:.0f} ms | rtf={rtf:.2f}] "
                f"(empty)"
            )

    last = segmenter.flush()
    
    if last is not None and len(last.audio) > 0:
        utt_audio = last.audio.astype(np.float32) / 32768.0

        context_prompt = None
        if args.keep_context and len(context_tail) > 0:
            context_prompt = "Previous context: " + " ".join(list(context_tail)[-30:]) + "."

        result = backend.transcribe(
            audio=utt_audio,
            sample_rate=TARGET_SR,
            language=args.language,
            context=context_prompt,
            return_words=False,
        )
        if result.text:
            texts.append(result.text)
            if args.keep_context:
                for w in result.text.split():
                    context_tail.append(w)


    return "\n".join(texts).strip()

def main():
    parser = argparse.ArgumentParser("Realtime mic transcription (model-agnostic)")
    parser.add_argument("--backend", default="faster-whisper", help="Backend name (e.g. faster-whisper)")
    parser.add_argument("--model", default="large-v3", help="Model name or path (backend-specific)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--device_index", type=int, default=0)
    parser.add_argument("--compute_type", default="int8_float16")
    parser.add_argument("--beam_size", type=int, default=10)
    parser.add_argument("--language", default="en")
    parser.add_argument("--vad_aggr", type=int, default=3)
    parser.add_argument("--mic", type=int, default=None)
    parser.add_argument("--print_words", action="store_true")
    parser.add_argument("--keep_context", action="store_true")
    parser.add_argument("--diag", action="store_true", help="Backend diagnostics (e.g. DLL probing on Windows)")
    # ---- File-mode options (NEW) ----
    parser.add_argument("--input", type=str, default=None,
                        help="Path to a single WAV/FLAC file to transcribe (file mode).")

    parser.add_argument("--input_dir", type=str, default=None,
                        help="Path to a directory of audio files to transcribe (batch file mode).")

    parser.add_argument("--output", type=str, default=None,
                        help="Output text file path (single input). If omitted, prints to stdout.")

    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for transcripts (batch input).")

    parser.add_argument("--no_vad", action="store_true",
                        help="Disable VAD segmentation in file mode; transcribe the whole file as one chunk.")

    args = parser.parse_args()


    file_mode = (args.input is not None) or (args.input_dir is not None)

    if args.input and args.input_dir:
        parser.error("Use only one: --input OR --input_dir")

    if args.output and args.output_dir:
        parser.error("Use only one: --output OR --output_dir")

    # Create backend (only place where model-specific config is assembled)
    backend_config = {
        "model": args.model,
        "device": args.device,
        "device_index": args.device_index,
        "compute_type": args.compute_type,
        "beam_size": args.beam_size,
        "diag_dll": args.diag,
    }
    backend = create_backend(args.backend, backend_config)

    # ---- FILE MODE: run and exit early ----
    if file_mode:
        if args.input:
            text = transcribe_one_file(args.input, backend, args)
            if args.output:
                Path(args.output).write_text(text + "\n", encoding="utf-8")
                print(f"[INFO] Wrote transcript to: {args.output}")
            else:
                print(text)
            return

        # batch directory mode
        in_dir = Path(args.input_dir)
        if not in_dir.exists() or not in_dir.is_dir():
            raise RuntimeError(f"--input_dir is not a directory: {in_dir}")

        out_dir = Path(args.output_dir) if args.output_dir else in_dir / "transcripts"
        out_dir.mkdir(parents=True, exist_ok=True)

        exts = {".wav"}

        files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in exts])

        if not files:
            raise RuntimeError(f"No audio files found in {in_dir} with extensions {sorted(exts)}")

        for p in files:
            text = transcribe_one_file(str(p), backend, args)
            out_path = out_dir / (p.stem + ".txt")
            out_path.write_text(text + "\n", encoding="utf-8")
            print(f"[INFO] {p.name} -> {out_path.name}")

        print(f"[INFO] Done. Transcripts in: {out_dir}")
        return

# ---- MIC MODE continues below ----


   

    # Shared queues / objects
    q_frames: queue.Queue = queue.Queue()
    segmenter = VADSegmenter(aggressiveness=args.vad_aggr)

    # Optional rolling context (model-agnostic)
    context_tail = deque(maxlen=80)
    from audio_capture import start_input_stream

    def worker():
        while True:
            try:
                frame_bytes, ts = q_frames.get(timeout=0.1)
            except queue.Empty:
                continue

            utt = segmenter.process_frame(frame_bytes, ts)
            if utt is None:
                continue

            t_utt_done = time.perf_counter()

            # int16 -> float32 [-1, 1]
            audio = utt.audio.astype(np.float32) / 32768.0
            audio_len_s = len(audio) / TARGET_SR
            print(f"\n[utt {utt.index}] (len {audio_len_s:.2f}s)")

            # Best-effort context prompt
            context_prompt = None
            if args.keep_context and len(context_tail) > 0:
                context_prompt = "Previous context: " + " ".join(list(context_tail)[-30:]) + "."

            # Transcribe via backend (model-agnostic)
            t_start = time.perf_counter()
            result = backend.transcribe(
                audio=audio,
                sample_rate=TARGET_SR,
                language=args.language,
                context=context_prompt,
                return_words=args.print_words,
            )
            t_end = time.perf_counter()

            processing_ms = (t_end - t_start) * 1000.0
            post_speech_latency_ms = (t_end - t_utt_done) * 1000.0

            # Print segments
            for seg in result.segments:
                if args.print_words and seg.words:
                    words_str = " ".join([f"[{w.start:.2f}-{w.end:.2f}] {w.word}" for w in seg.words])
                    print(f"  {seg.start:.2f}-{seg.end:.2f}: {words_str}")
                else:
                    print(f"  {seg.start:.2f}-{seg.end:.2f}: {seg.text}")

            print(f"[text] [{result.engine}] [proc={processing_ms:.0f} ms | lat={post_speech_latency_ms:.0f} ms] {result.text}")

            # Update rolling context
            if args.keep_context and result.text:
                for w in result.text.split():
                    context_tail.append(w)

    # Start worker thread
    t = threading.Thread(target=worker, daemon=True)
    t.start()

    # Start microphone stream
    stream = start_input_stream(q_frames, mic=args.mic)
    with stream:
        print("[INFO] Listening… press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.25)
        except KeyboardInterrupt:
            print("[INFO] Stopping…")
            last = segmenter.flush()
            if last is not None and len(last.audio) > 0:
                audio = last.audio.astype(np.float32) / 32768.0
                t0 = time.perf_counter()
                result = backend.transcribe(
                    audio=audio,
                    sample_rate=TARGET_SR,
                    language=args.language,
                    context=None,
                    return_words=False,
                )
                t1 = time.perf_counter()
                print(f"[final] [proc={(t1 - t0) * 1000:.0f} ms] {result.text if result.text else '(no speech)'}")


if __name__ == "__main__":
    main()
