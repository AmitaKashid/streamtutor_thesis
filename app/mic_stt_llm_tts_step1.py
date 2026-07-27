from __future__ import annotations

import sys
import argparse
import queue
import threading
import time
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---- STT ----
from stt.vad_segmenter import VADSegmenter, TARGET_SR
from stt.stt_backends import create_backend
#from stt.audio_capture import start_input_stream
from stt.audio_file import load_audio_mono_16k

# ---- LLM ----
from llm.llmcli.models.registry import get_model_spec
from llm.llmcli.providers.llamacpp import LlamaCppProvider

# ---- TTS ----
from tts.chatterbox_engine import ChatterboxEngine
from tts.tts_queue import TTSRequest  # reuse dataclass for consistency

# ---- Streaming helpers ----
from llm.llmcli.common.text_chunker import TextChunker, ChunkerConfig
from llm.llmcli.common.audio_smoother import AudioCrossfader, CrossfadeConfig
from llm.llmcli.common.audio_output import AudioOutput, AudioOutputConfig

# ---- Prompt / RAG ----
from prompts.registry import compose_prompt
from rag.rag_retrieve import RagRetriever


# ---------------- Timing ----------------
class Timer:
    def __init__(self):
        self.t0 = perf_counter()

    def elapsed(self) -> float:
        return perf_counter() - self.t0


# ---------------- LLM ----------------
def build_llm(model_key: str) -> LlamaCppProvider:
    spec = get_model_spec(model_key)
    return LlamaCppProvider(
        model_path=spec.gguf_path,
        chat_format=spec.chat_format,
        system_prompt="You are a voice assistant. Answer briefly in 1–2 short sentences.",
        n_ctx=2048,
        n_threads=8,
        n_gpu_layers=-1,
        max_new_tokens=128,
        temperature=0.2,
        top_p=0.8,
        seed=0,
    )


# ---------------- Helpers ----------------
def drain_queue(q: "queue.Queue[tuple[bytes, float]]") -> int:
    n = 0
    try:
        while True:
            q.get_nowait()
            n += 1
    except queue.Empty:
        return n


HALLUCINATION_PHRASES = {
    "thanks for watching",
    "thank you for watching",
    "like and subscribe",
    "subscribe to my channel",
    "see you next time",
    "thank you",
}


def torch_to_np_tc(wav_torch) -> np.ndarray:
    """
    Convert torch waveform [C, T] or [T] to numpy float32 [T, C].
    """
    w = wav_torch.detach().cpu()
    if w.dim() == 1:
        w = w.unsqueeze(0)
    w = w.transpose(0, 1).contiguous()
    return w.numpy().astype(np.float32, copy=False)


def frame_rms(frame_bytes: bytes) -> float:
    if not frame_bytes:
        return 0.0
    a = np.frombuffer(frame_bytes, dtype=np.int16)
    if a.size == 0:
        return 0.0
    a = a.astype(np.float32)
    return float(np.sqrt(np.mean(a * a) + 1e-12))


class PlaybackControl:
    """
    Shared playback state so the mic worker can stop current audio immediately.
    """
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.audio_out = None
        self.interrupted = threading.Event()

    def attach_audio_out(self, audio_out) -> None:
        with self._lock:
            self.audio_out = audio_out

    def clear_audio_out(self) -> None:
        with self._lock:
            self.audio_out = None

    def reset_interrupt(self) -> None:
        self.interrupted.clear()

    def stop_now(self) -> None:
        """
        Stop both batch playback (sounddevice) and stream playback (AudioOutput).
        """
        self.interrupted.set()

        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

        with self._lock:
            ao = self.audio_out

        if ao is not None:
            try:
                if hasattr(ao, "interrupt"):
                    ao.interrupt()
                else:
                    ao.stop()
            except Exception:
                pass


def set_tts_state(
    active: bool,
    tts_busy: threading.Event | None,
    drain_after_tts: threading.Event | None,
) -> None:
    """
    Mic mode uses half-duplex gating. File mode passes None and becomes a no-op.
    """
    if active:
        if tts_busy is not None:
            tts_busy.set()
    else:
        if tts_busy is not None:
            tts_busy.clear()
        if drain_after_tts is not None:
            drain_after_tts.set()


def run_llm_tts_pipeline(
    *,
    transcript: str,
    stt_time: float,
    total_timer: Timer,
    llm: LlamaCppProvider,
    tts_engine: ChatterboxEngine,
    retriever: RagRetriever | None,
    args,
    playback: PlaybackControl,
    tts_busy: threading.Event | None = None,
    drain_after_tts: threading.Event | None = None,
) -> None:
    """
    Shared post-STT pipeline for both mic mode and file mode:
      transcript -> (RAG) -> prompt -> LLM -> TTS
    """
    norm = transcript.lower().strip(" .!?")
    playback.reset_interrupt()

    print(f"[STT] {transcript}")
    print(f"[STT Latency] {stt_time:.3f}s")

    if not transcript:
        print("[INFO] Empty transcript.")
        return

    if len(transcript.split()) < args.min_words:
        print("[STT] Ignored short utterance")
        return

    if norm in HALLUCINATION_PHRASES:
        print(f"[STT] Ignored hallucination: '{transcript}'")
        return

    # ---------------- RAG ----------------
    rag_time = 0.0
    context = ""

    if retriever is not None:
        rag_timer = Timer()
        hits = retriever.retrieve(transcript, top_k=args.rag_top_k)
        context = retriever.format_context(hits)
        rag_time = rag_timer.elapsed()

        print(f"\n[RAG]")
        print(f"  latency : {rag_time:.3f}s")
        print(f"  top_k   : {len(hits)}")
        print(f"  retrieved:")

        for i, h in enumerate(hits, 1):
            chunk_id = getattr(h, "chunk_id", None)
            slide_id = getattr(h, "slide_id", None)
            score = getattr(h, "score", None)
            id_value = chunk_id or slide_id or "unknown"

            if score is not None:
                print(f"    {i}. id={id_value} | score={score:.4f}")
            else:
                print(f"    {i}. id={id_value}")

        if args.rag_print_context:
            if args.rag_context_chars == 0:
                preview = context
            else:
                preview = context[: args.rag_context_chars]
                if len(context) > args.rag_context_chars:
                    preview += "\n\n... [truncated]"
            print("\n[RAG Context -> LLM]\n" + preview + "\n")

    llm_prompt = compose_prompt(
        style=args.teaching_style,
        user_query=transcript,
        context=context if retriever is not None else None,
    )

    # ---------------- Mode A: batch ----------------
    if args.mode == "batch":
        llm_timer = Timer()
        llm_res = llm.generate(llm_prompt)
        answer = (llm_res.text or "").strip()
        llm_time = llm_timer.elapsed()

        print("\n[LLM Answer]")
        print(answer)
        print(f"[LLM Latency] {llm_time:.3f}s")

        if not answer:
            print("[INFO] Empty LLM output; nothing to synthesize.")
            return

        out_path = None
        if args.save:
            Path("tts_output").mkdir(exist_ok=True)
            out_path = Path("tts_output") / f"answer_{int(time.time())}.wav"

        req = TTSRequest(
            text=answer,
            request_id=f"tts_{int(time.time())}",
            voice_prompt=Path(args.voice_prompt) if args.voice_prompt else None,
            config=None,
            save_to=out_path,
            play_audio=args.speak,
        )

        set_tts_state(True, tts_busy, drain_after_tts)
        tts_proc_timer = Timer()
        try:
            wav, sr = tts_engine.synthesize(
                text=req.text,
                voice_prompt=req.voice_prompt,
                config=req.config,
                save_to=req.save_to,
            )
            tts_proc_time = tts_proc_timer.elapsed()

            if req.play_audio:
                try:
                    import sounddevice as sd
                except Exception:
                    sd = None
                if sd is None:
                    print("[TTS] sounddevice not available, skipping playback.")
                else:
                    wav_np = torch_to_np_tc(wav)
                    sd.play(wav_np, sr)
                    sd.wait()

                    if playback.interrupted.is_set():
                        print("[INTERRUPT] Batch playback stopped by user speech.")
                        return
        finally:
            set_tts_state(False, tts_busy, drain_after_tts)

        total_time = total_timer.elapsed()
        print(f"[TTS Proc Latency] {tts_proc_time:.3f}s  (playback excluded)")
        print(
            "\n[TIMING SUMMARY]\n"
            f"  STT        : {stt_time:.3f} s\n"
            f"  RAG        : {rag_time:.3f} s\n"
            f"  LLM        : {llm_time:.3f} s\n"
            f"  TTS(proc)  : {tts_proc_time:.3f} s\n"
            f"  -----------------------------\n"
            f"  TOTAL      : {total_time:.3f} s\n"
        )
        return

    # ---------------- Mode B: stream-chunked ----------------
    print("\n[MODE] stream-chunked")
    llm_first_text_t = None
    first_audio_t = None

    chunk_cfg = ChunkerConfig(
        first_max_chars=args.chunk_first,
        mid_max_chars=args.chunk_mid,
        later_max_chars=args.chunk_later,
        max_wait_s=args.chunk_wait,
        min_chars_timer=80,
        min_chars_sentence=35,
    )
    chunker = TextChunker(chunk_cfg)

    audio_out = None
    crossfader = None

    def llm_piece_iter():
        nonlocal llm_first_text_t
        for piece in llm.stream(llm_prompt):
            if llm_first_text_t is None and piece.strip():
                llm_first_text_t = perf_counter()
            yield piece

    llm_timer = Timer()
    tts_proc_total = 0.0
    chunks_count = 0

    set_tts_state(True, tts_busy, drain_after_tts)
    try:
        for chunk_text in chunker.feed(llm_piece_iter()):
            if playback.interrupted.is_set():
                print("[INTERRUPT] Stream playback stopped by user speech.")
                break

            if not chunk_text:
                continue

            chunks_count += 1
            print(f"[CHUNK {chunks_count}] {chunk_text}")

            tts_t = Timer()
            wav, sr = tts_engine.synthesize(
                text=chunk_text,
                voice_prompt=Path(args.voice_prompt) if args.voice_prompt else None,
                config=None,
                save_to=None,
            )
            tts_proc_total += tts_t.elapsed()

            wav_np = torch_to_np_tc(wav)

            if audio_out is None:
                channels = wav_np.shape[1]
                audio_out = AudioOutput(sr=sr, channels=channels, cfg=AudioOutputConfig())
                audio_out.start()
                playback.attach_audio_out(audio_out)
                crossfader = AudioCrossfader(
                    sr=sr,
                    channels=channels,
                    cfg=CrossfadeConfig(crossfade_ms=args.crossfade_ms, enabled=True),
                )

            assert crossfader is not None
            smoothed = crossfader.process(wav_np)

            if args.speak:
                audio_out.enqueue(smoothed)
                if first_audio_t is None and len(smoothed) > 0:
                    first_audio_t = perf_counter()

        if playback.interrupted.is_set():
            last = None
        else:
            last = chunker.flush()
        if last:
            chunks_count += 1
            print(f"[CHUNK {chunks_count}] {last}")

            tts_t = Timer()
            wav, sr = tts_engine.synthesize(
                text=last,
                voice_prompt=Path(args.voice_prompt) if args.voice_prompt else None,
                config=None,
                save_to=None,
            )
            tts_proc_total += tts_t.elapsed()

            wav_np = torch_to_np_tc(wav)
            if audio_out is None:
                channels = wav_np.shape[1]
                audio_out = AudioOutput(sr=sr, channels=channels, cfg=AudioOutputConfig())
                audio_out.start()
                crossfader = AudioCrossfader(
                    sr=sr,
                    channels=channels,
                    cfg=CrossfadeConfig(crossfade_ms=args.crossfade_ms, enabled=True),
                )

            assert crossfader is not None
            smoothed = crossfader.process(wav_np)

            if args.speak:
                audio_out.enqueue(smoothed)
                if first_audio_t is None and len(smoothed) > 0:
                    first_audio_t = perf_counter()

        if crossfader is not None and audio_out is not None and args.speak:
            tail = crossfader.flush()
            if len(tail) > 0:
                audio_out.enqueue(tail)

        llm_time = llm_timer.elapsed()

        if audio_out is not None and args.speak:
            if playback.interrupted.is_set():
                try:
                    audio_out.stop()
                except Exception:
                    pass
            else:
                audio_out.wait_done()
                audio_out.stop()

    finally:
        playback.clear_audio_out()
        set_tts_state(False, tts_busy, drain_after_tts)

    total_time = total_timer.elapsed()

    ttft = (llm_first_text_t - total_timer.t0) if llm_first_text_t is not None else None
    ttfa = (first_audio_t - total_timer.t0) if first_audio_t is not None else None

    print("\n[STREAM METRICS]")
    print(f"  chunks              : {chunks_count}")
    print(f"  RAG latency         : {rag_time:.3f} s")
    print(f"  LLM total           : {llm_time:.3f} s")
    print(f"  TTS(proc total)     : {tts_proc_total:.3f} s (sum of chunk synth times)")
    print(f"  TTFT (first text)   : {ttft:.3f} s" if ttft is not None else "  TTFT (first text)   : N/A")
    print(f"  TTFA (first audio)  : {ttfa:.3f} s" if ttfa is not None else "  TTFA (first audio)  : N/A")
    print(f"  TOTAL (end-to-end)  : {total_time:.3f} s")


# ---------------- Main ----------------
def main():
    p = argparse.ArgumentParser("Mic/File → STT → LLM → TTS (batch or stream-chunked)")

    # Input mode
    p.add_argument("--input", default=None, help="Optional audio file path. If set, runs file mode; otherwise mic mode.")

    # STT
    p.add_argument("--stt_backend", default="faster-whisper")
    p.add_argument("--stt_model", default="large-v3")
    p.add_argument("--stt_device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--stt_device_index", type=int, default=0)
    p.add_argument("--stt_compute_type", default="int8_float16")
    p.add_argument("--stt_beam_size", type=int, default=10)
    p.add_argument("--language", default="en")
    p.add_argument("--vad_aggr", type=int, default=3)
    p.add_argument("--mic", type=int, default=None)

    # LLM
    p.add_argument("--llm_model", required=True)

    # Modes
    p.add_argument("--mode", choices=["batch", "stream-chunked"], default="batch")

    # TTS
    p.add_argument("--speak", action="store_true")
    p.add_argument("--save", action="store_true")
    p.add_argument("--voice_prompt", default=None)

    # Safety / tuning
    p.add_argument("--tail_settle", type=float, default=0.3)
    p.add_argument("--min_words", type=int, default=2)

    # RAG
    p.add_argument("--rag", action="store_true", help="Enable retrieval-augmented generation (RAG)")
    p.add_argument("--rag_config", default="configs/retrieval.yaml")
    p.add_argument("--rag_top_k", type=int, default=None)
    p.add_argument("--rag_print_context", action="store_true", help="Print retrieved RAG context passed to LLM")
    p.add_argument("--rag_context_chars", type=int, default=2500, help="Max chars of context to print preview. Use 0 for full.")

    # Streaming tuning
    p.add_argument("--chunk_first", type=int, default=80)
    p.add_argument("--chunk_mid", type=int, default=140)
    p.add_argument("--chunk_later", type=int, default=220)
    p.add_argument("--chunk_wait", type=float, default=0.9)
    p.add_argument("--crossfade_ms", type=float, default=50.0)

    # Teaching style prompt
    p.add_argument(
        "--teaching_style",
        type=str,
        default="socratic",
        choices=["socratic", "adaptive", "inquiry", "collaborative"],
        help="Teaching style prompt injected into the LLM prompt",
    )

    args = p.parse_args()

    # ---- STT backend ----
    stt_backend = create_backend(
        args.stt_backend,
        {
            "model": args.stt_model,
            "device": args.stt_device,
            "device_index": args.stt_device_index,
            "compute_type": args.stt_compute_type,
            "beam_size": args.stt_beam_size,
            "diag_dll": False,
        },
    )

    # ---- LLM ----
    llm = build_llm(args.llm_model)

    # ---- TTS engine ----
    tts_engine = ChatterboxEngine()
    tts_engine.load()

    # ---- RAG ----
    retriever = RagRetriever(args.rag_config) if args.rag else None

    playback = PlaybackControl()
    pipeline_active = threading.Event()

    # =========================
    # FILE MODE
    # =========================
    if args.input:
        wav_path = Path(args.input)
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")

        print(f"[INFO] Running FILE mode on: {wav_path}")

        total_timer = Timer()

        stt_timer = Timer()
        audio, sr = load_audio_mono_16k(str(wav_path))
        stt_res = stt_backend.transcribe(
            audio=audio,
            sample_rate=sr,
            language=args.language,
            return_words=False,
        )
        stt_time = stt_timer.elapsed()

        transcript = (stt_res.text or "").strip()
        print("\n[STT Transcript]")
        print(transcript)

        run_llm_tts_pipeline(
            transcript=transcript,
            stt_time=stt_time,
            total_timer=total_timer,
            llm=llm,
            tts_engine=tts_engine,
            retriever=retriever,
            args=args,
            playback=playback,
            tts_busy=None,
            drain_after_tts=None,
        )
        return

    # =========================
    # MIC MODE
    # =========================
    q_frames: "queue.Queue[tuple[bytes, float]]" = queue.Queue()
    segmenter = VADSegmenter(aggressiveness=args.vad_aggr)

    # Half-duplex control for mic mode only
    tts_busy = threading.Event()
    drain_after_tts = threading.Event()

    def worker():
        while True:
            try:
                frame_bytes, ts = q_frames.get(timeout=0.1)
            except queue.Empty:
                continue

            if tts_busy.is_set():
                # Step 1 only: if user starts speaking while TTS is active,
                # stop audio immediately.
                if frame_rms(frame_bytes) >= 180.0:
                    print("[INTERRUPT] Speech detected during TTS -> stopping audio now.")
                    playback.stop_now()
                    segmenter.reset()
                    drain_after_tts.set()
                continue

            if pipeline_active.is_set():
                segmenter.reset()
                continue

            if drain_after_tts.is_set():
                dropped = drain_queue(q_frames)
                segmenter.reset()
                drain_after_tts.clear()
                time.sleep(args.tail_settle)
                if dropped:
                    print(f"[MIC] Drained {dropped} frames after TTS.")

            utt = segmenter.process_frame(frame_bytes, ts)
            if utt is None:
                continue

            total_timer = Timer()

            audio = utt.audio.astype(np.float32) / 32768.0
            print(f"\n[utt {utt.index}] len={len(audio)/TARGET_SR:.2f}s")

            stt_timer = Timer()
            stt_res = stt_backend.transcribe(
                audio=audio,
                sample_rate=TARGET_SR,
                language=args.language,
                return_words=False,
            )
            stt_time = stt_timer.elapsed()

            transcript = (stt_res.text or "").strip()

            def pipeline_job():
                pipeline_active.set()
                try:
                    run_llm_tts_pipeline(
                        transcript=transcript,
                        stt_time=stt_time,
                        total_timer=total_timer,
                        llm=llm,
                        tts_engine=tts_engine,
                        retriever=retriever,
                        args=args,
                        playback=playback,
                        tts_busy=tts_busy,
                        drain_after_tts=drain_after_tts,
                    )
                finally:
                    pipeline_active.clear()

            threading.Thread(target=pipeline_job, daemon=True).start()

    threading.Thread(target=worker, daemon=True).start()
    from stt.audio_capture import start_input_stream
    stream = start_input_stream(q_frames, mic=args.mic)

    try:
        with stream:
            print("[INFO] Using MIC mode")
            print("[INFO] Listening… Ctrl+C to stop.")
            while True:
                time.sleep(0.25)
    except KeyboardInterrupt:
        print("[INFO] Stopping…")


if __name__ == "__main__":
    main()