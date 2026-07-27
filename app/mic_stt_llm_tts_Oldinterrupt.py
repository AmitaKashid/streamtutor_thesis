from __future__ import annotations

import sys
import argparse
import queue
import threading
import time
import re
from pathlib import Path
from time import perf_counter
from typing import Optional
from collections import deque

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---- STT ----
from stt.vad_segmenter import VADSegmenter, TARGET_SR
from stt.stt_backends import create_backend
from stt.audio_file import load_audio_mono_16k

# ---- LLM ----
from llm.llmcli.models.registry import get_model_spec
from llm.llmcli.providers.llamacpp import LlamaCppProvider

# ---- TTS ----
from tts.chatterbox_engine import ChatterboxEngine

# ---- Streaming helpers ----
from llm.llmcli.common.text_chunker import TextChunker, ChunkerConfig

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


def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def normalize_for_similarity(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def looks_like_false_barge_in(
    transcript: str,
    last_spoken: str,
    *,
    min_words: int,
    min_chars: int,
    max_overlap: float,
) -> bool:
    """
    Reject:
    - empty / very short interruption transcripts
    - interruption transcripts too similar to assistant's last spoken sentence
    """
    t = normalize_for_similarity(transcript)
    last = normalize_for_similarity(last_spoken)

    if not t:
        return True

    if len(t.split()) < min_words:
        return True

    if len(t) < min_chars:
        return True

    if last:
        if t in last or last in t:
            return True

        t_words = set(t.split())
        l_words = set(last.split())
        if t_words and l_words:
            overlap = len(t_words & l_words) / max(1, len(t_words))
            if overlap >= max_overlap:
                return True

    return False


def build_interruption_transcript(last_spoken: str, new_user_utterance: str) -> str:
    last_spoken = (last_spoken or "").strip()
    new_user_utterance = (new_user_utterance or "").strip()

    if not last_spoken:
        return new_user_utterance

    return (
        "The user interrupted you while you were saying:\n"
        f"\"{last_spoken}\"\n\n"
        "The user then interrupted with:\n"
        f"\"{new_user_utterance}\"\n\n"
        "Respond directly to the interruption, continue helpfully, and adapt your explanation."
    )


def frame_rms(frame_bytes: bytes) -> float:
    if not frame_bytes:
        return 0.0
    a = np.frombuffer(frame_bytes, dtype=np.int16)
    if a.size == 0:
        return 0.0
    a = a.astype(np.float32)
    return float(np.sqrt(np.mean(a * a) + 1e-12))


def flush_frame_queue(q: "queue.Queue[tuple[bytes, float]]") -> int:
    n = 0
    try:
        while True:
            q.get_nowait()
            n += 1
    except queue.Empty:
        return n


def frames_to_audio_float32(frames: list[bytes]) -> np.ndarray:
    if not frames:
        return np.zeros(0, dtype=np.float32)
    arr = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32)
    return arr / 32768.0


# ---------------- Turn / state controller ----------------
class TurnController:
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    INTERRUPTING = "INTERRUPTING"

    def __init__(self, llm, tts_engine, retriever, args):
        self.llm = llm
        self.tts_engine = tts_engine
        self.retriever = retriever
        self.args = args

        self._lock = threading.Lock()
        self._state = self.LISTENING
        self._response_thread: Optional[threading.Thread] = None

        self._cancel_requested = False
        self._pending_interrupt_transcript: Optional[str] = None
        self._pending_interrupt_last_spoken: str = ""
        self._last_spoken_sentence = ""

        self._llm_lock = threading.Lock()

        self._save_chunks: list[np.ndarray] = []
        self._save_sr: Optional[int] = None

    # ---------- state ----------
    def get_state(self) -> str:
        with self._lock:
            return self._state

    def _set_state(self, state: str) -> None:
        with self._lock:
            self._state = state

    def wait_until_idle(self) -> None:
        while self.get_state() != self.LISTENING:
            time.sleep(0.05)

    # ---------- cancellation ----------
    def request_cancel(self) -> None:
        with self._lock:
            self._cancel_requested = True

    def clear_cancel(self) -> None:
        with self._lock:
            self._cancel_requested = False

    def should_cancel(self) -> bool:
        with self._lock:
            return self._cancel_requested

    # ---------- last spoken ----------
    def set_last_spoken_sentence(self, sentence: str) -> None:
        with self._lock:
            self._last_spoken_sentence = (sentence or "").strip()

    def get_last_spoken_sentence(self) -> str:
        with self._lock:
            return self._last_spoken_sentence

    # ---------- audio ----------
    def _play_audio_blocking(self, wav_np: np.ndarray, sr: int) -> None:
        if not self.args.speak:
            return

        import sounddevice as sd

        self._set_state(self.SPEAKING)
        try:
            sd.play(wav_np, sr)
            sd.wait()
        except Exception as e:
            print(f"[TTS] Playback unavailable or failed: {e}")
        finally:
            # If we were interrupted, remain INTERRUPTING; otherwise go back to THINKING
            if self.get_state() != self.INTERRUPTING:
                self._set_state(self.THINKING)

    def interrupt_playback(self) -> None:
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

    def _reset_save_buffer(self) -> None:
        self._save_chunks = []
        self._save_sr = None

    def _append_save_audio(self, wav_np: np.ndarray, sr: int) -> None:
        if not self.args.save:
            return
        self._save_chunks.append(wav_np)
        if self._save_sr is None:
            self._save_sr = sr

    def _flush_save_audio(self) -> None:
        if not self.args.save or not self._save_chunks or self._save_sr is None:
            return

        try:
            import soundfile as sf
            Path("tts_output").mkdir(exist_ok=True)
            out_path = Path("tts_output") / f"answer_{int(time.time())}.wav"
            full = np.concatenate(self._save_chunks, axis=0)
            sf.write(str(out_path), full, self._save_sr)
            print(f"[TTS] Saved audio to {out_path}")
        except Exception as e:
            print(f"[TTS] Failed to save audio: {e}")
        finally:
            self._reset_save_buffer()

    # ---------- turn lifecycle ----------
    def start_user_turn(self, transcript: str) -> None:
        transcript = (transcript or "").strip()
        if not transcript:
            return

        norm = transcript.lower().strip(" .!?")
        if len(transcript.split()) < self.args.min_words:
            print("[STT] Ignored short utterance")
            return

        if norm in HALLUCINATION_PHRASES:
            print(f"[STT] Ignored hallucination: '{transcript}'")
            return

        if self.get_state() != self.LISTENING:
            return

        self.clear_cancel()
        self._set_state(self.THINKING)
        self._reset_save_buffer()

        self._response_thread = threading.Thread(
            target=self._run_response,
            args=(transcript,),
            daemon=True,
        )
        self._response_thread.start()

    def begin_interruption(self) -> bool:
        """
        Only real speaking can be interrupted.
        """
        if self.get_state() != self.SPEAKING:
            return False

        self.request_cancel()
        self._set_state(self.INTERRUPTING)
        self.interrupt_playback()
        return True

    def finish_interruption_capture(self, interruption_text: str, last_spoken: str) -> None:
        interruption_text = (interruption_text or "").strip()

        if not interruption_text:
            print("[INTERRUPTION] Empty interruption transcript, waiting for current turn to exit.")
            return

        norm = interruption_text.lower().strip(" .!?")
        if norm in HALLUCINATION_PHRASES:
            print(f"[BARGE-IN] Rejected hallucination interruption: {interruption_text!r}")
            return

        if looks_like_false_barge_in(
            interruption_text,
            last_spoken,
            min_words=self.args.barge_in_min_words,
            min_chars=self.args.barge_in_min_chars,
            max_overlap=self.args.barge_in_max_overlap,
        ):
            print(f"[BARGE-IN] Rejected likely false interruption: {interruption_text!r}")
            return

        print(f"\n[INTERRUPTION]")
        if last_spoken:
            print(f"  while_speaking : {last_spoken}")
        print(f"  user_said      : {interruption_text}")

        # Store pending interruption. New response starts ONLY after old response exits.
        with self._lock:
            self._pending_interrupt_transcript = interruption_text
            self._pending_interrupt_last_spoken = last_spoken

    # ---------- prompt / rag ----------
    def _build_llm_prompt(self, transcript: str) -> tuple[str, float]:
        rag_time = 0.0
        context = ""

        if self.retriever is not None:
            rag_timer = Timer()
            hits = self.retriever.retrieve(transcript, top_k=self.args.rag_top_k)
            context = self.retriever.format_context(hits)
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

            if self.args.rag_print_context:
                if self.args.rag_context_chars == 0:
                    preview = context
                else:
                    preview = context[: self.args.rag_context_chars]
                    if len(context) > self.args.rag_context_chars:
                        preview += "\n\n... [truncated]"
                print("\n[RAG Context -> LLM]\n" + preview + "\n")

        llm_prompt = compose_prompt(
            style=self.args.teaching_style,
            user_query=transcript,
            context=context if self.retriever is not None else None,
        )
        return llm_prompt, rag_time

    # ---------- speaking ----------
    def _speak_sentence(self, sentence: str) -> float:
        if self.should_cancel():
            return 0.0

        self.set_last_spoken_sentence(sentence)

        tts_timer = Timer()
        wav, sr = self.tts_engine.synthesize(
            text=sentence,
            voice_prompt=Path(self.args.voice_prompt) if self.args.voice_prompt else None,
            config=None,
            save_to=None,
        )
        tts_time = tts_timer.elapsed()

        if self.should_cancel():
            return tts_time

        wav_np = torch_to_np_tc(wav)
        self._append_save_audio(wav_np, sr)
        self._play_audio_blocking(wav_np, sr)
        return tts_time

    # ---------- response execution ----------
    def _run_response(self, transcript: str) -> None:
        total_timer = Timer()
        try:
            llm_prompt, rag_time = self._build_llm_prompt(transcript)
            if self.should_cancel():
                return

            # ---------------- Mode A: batch ----------------
            if self.args.mode == "batch":
                llm_timer = Timer()
                with self._llm_lock:
                    llm_res = self.llm.generate(llm_prompt)
                llm_time = llm_timer.elapsed()

                if self.should_cancel():
                    return

                answer = (llm_res.text or "").strip()

                print("\n[LLM Answer]")
                print(answer)
                print(f"[LLM Latency] {llm_time:.3f}s")

                if not answer:
                    print("[INFO] Empty LLM output; nothing to synthesize.")
                    return

                tts_proc_total = 0.0
                for sent in split_sentences(answer):
                    if self.should_cancel():
                        return
                    tts_proc_total += self._speak_sentence(sent)

                if self.should_cancel():
                    return

                self._flush_save_audio()
                total_time = total_timer.elapsed()
                print(
                    "\n[TIMING SUMMARY]\n"
                    f"  RAG        : {rag_time:.3f} s\n"
                    f"  LLM        : {llm_time:.3f} s\n"
                    f"  TTS(proc)  : {tts_proc_total:.3f} s\n"
                    f"  -----------------------------\n"
                    f"  TOTAL      : {total_time:.3f} s\n"
                )
                return

            # ---------------- Mode B: stream-chunked ----------------
            print("\n[MODE] stream-chunked")
            llm_first_text_t = None
            first_audio_t = None

            chunk_cfg = ChunkerConfig(
                first_max_chars=self.args.chunk_first,
                mid_max_chars=self.args.chunk_mid,
                later_max_chars=self.args.chunk_later,
                max_wait_s=self.args.chunk_wait,
                min_chars_timer=80,
                min_chars_sentence=35,
            )
            chunker = TextChunker(chunk_cfg)

            def llm_piece_iter():
                nonlocal llm_first_text_t
                with self._llm_lock:
                    for piece in self.llm.stream(llm_prompt):
                        if self.should_cancel():
                            return
                        if llm_first_text_t is None and piece.strip():
                            llm_first_text_t = perf_counter()
                        yield piece

            llm_timer = Timer()
            tts_proc_total = 0.0
            chunks_count = 0

            for chunk_text in chunker.feed(llm_piece_iter()):
                if self.should_cancel():
                    return
                if not chunk_text:
                    continue

                chunks_count += 1
                print(f"[CHUNK {chunks_count}] {chunk_text}")
                tts_proc_total += self._speak_sentence(chunk_text)
                if first_audio_t is None:
                    first_audio_t = perf_counter()

            last = chunker.flush()
            if last and not self.should_cancel():
                chunks_count += 1
                print(f"[CHUNK {chunks_count}] {last}")
                tts_proc_total += self._speak_sentence(last)
                if first_audio_t is None:
                    first_audio_t = perf_counter()

            if self.should_cancel():
                return

            self._flush_save_audio()
            llm_time = llm_timer.elapsed()
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

        finally:
            pending_text = None
            pending_last = ""

            with self._lock:
                if self._pending_interrupt_transcript:
                    pending_text = self._pending_interrupt_transcript
                    pending_last = self._pending_interrupt_last_spoken
                    self._pending_interrupt_transcript = None
                    self._pending_interrupt_last_spoken = ""

                self._cancel_requested = False

            if pending_text:
                final_user_query = build_interruption_transcript(pending_last, pending_text)

                self._set_state(self.THINKING)
                self._reset_save_buffer()

                self._response_thread = threading.Thread(
                    target=self._run_response,
                    args=(final_user_query,),
                    daemon=True,
                )
                self._response_thread.start()
            else:
                self._set_state(self.LISTENING)


def main():
    p = argparse.ArgumentParser("Mic/File → STT → LLM → TTS (state-machine interruption handling)")

    p.add_argument("--input", default=None, help="Optional audio file path. If set, runs file mode; otherwise mic mode.")

    p.add_argument("--stt_backend", default="faster-whisper")
    p.add_argument("--stt_model", default="large-v3")
    p.add_argument("--stt_device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--stt_device_index", type=int, default=0)
    p.add_argument("--stt_compute_type", default="int8_float16")
    p.add_argument("--stt_beam_size", type=int, default=10)
    p.add_argument("--language", default="en")
    p.add_argument("--vad_aggr", type=int, default=3)
    p.add_argument("--mic", type=int, default=None)

    p.add_argument("--llm_model", required=True)
    p.add_argument("--mode", choices=["batch", "stream-chunked"], default="batch")

    p.add_argument("--speak", action="store_true")
    p.add_argument("--save", action="store_true")
    p.add_argument("--voice_prompt", default=None)

    p.add_argument("--min_words", type=int, default=2)

    p.add_argument("--rag", action="store_true")
    p.add_argument("--rag_config", default="configs/retrieval.yaml")
    p.add_argument("--rag_top_k", type=int, default=None)
    p.add_argument("--rag_print_context", action="store_true")
    p.add_argument("--rag_context_chars", type=int, default=2500)

    p.add_argument("--chunk_first", type=int, default=80)
    p.add_argument("--chunk_mid", type=int, default=140)
    p.add_argument("--chunk_later", type=int, default=220)
    p.add_argument("--chunk_wait", type=float, default=0.9)

    p.add_argument(
        "--teaching_style",
        type=str,
        default="socratic",
        choices=["socratic", "adaptive", "inquiry", "collaborative"],
    )

    p.add_argument("--barge_in", action="store_true")
    p.add_argument("--interrupt_rms_threshold", type=float, default=220.0)
    p.add_argument("--interrupt_onset_frames", type=int, default=2)
    p.add_argument("--interrupt_settle", type=float, default=0.08)
    p.add_argument("--interrupt_preroll_ms", type=int, default=600)
    p.add_argument("--barge_in_min_words", type=int, default=3)
    p.add_argument("--barge_in_min_chars", type=int, default=8)
    p.add_argument("--barge_in_max_overlap", type=float, default=0.75)

    args = p.parse_args()

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

    llm = build_llm(args.llm_model)

    tts_engine = ChatterboxEngine()
    tts_engine.load()

    retriever = RagRetriever(args.rag_config) if args.rag else None

    controller = TurnController(
        llm=llm,
        tts_engine=tts_engine,
        retriever=retriever,
        args=args,
    )

    if args.input:
        wav_path = Path(args.input)
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")

        print(f"[INFO] Running FILE mode on: {wav_path}")

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
        print(f"[STT Latency] {stt_time:.3f}s")

        controller.start_user_turn(transcript)
        controller.wait_until_idle()
        return

    q_frames: "queue.Queue[tuple[bytes, float]]" = queue.Queue()
    main_segmenter = VADSegmenter(aggressiveness=args.vad_aggr)
    interrupt_segmenter = VADSegmenter(aggressiveness=args.vad_aggr)

    # 20 ms frames => ~50 frames/sec
    frames_per_sec = 50
    preroll_frames = max(1, int((args.interrupt_preroll_ms / 1000.0) * frames_per_sec))
    recent_frames: deque[bytes] = deque(maxlen=preroll_frames)

    def worker():
        interrupt_collecting = False
        interrupt_skip_until = 0.0
        onset_count = 0
        captured_interrupt_frames: list[bytes] = []
        interrupted_last_spoken = ""

        while True:
            try:
                frame_bytes, ts = q_frames.get(timeout=0.1)
            except queue.Empty:
                continue

            recent_frames.append(frame_bytes)

            now = perf_counter()
            state = controller.get_state()

            # ---------------- INTERRUPTION CAPTURE MODE ----------------
            if interrupt_collecting:
                if now < interrupt_skip_until:
                    continue

                captured_interrupt_frames.append(frame_bytes)

                utt = interrupt_segmenter.process_frame(frame_bytes, ts)
                if utt is None:
                    continue

                fresh_audio = utt.audio.astype(np.float32) / 32768.0
                preroll_audio = frames_to_audio_float32(captured_interrupt_frames)

                if preroll_audio.size > 0:
                    audio = np.concatenate([preroll_audio, fresh_audio], axis=0)
                else:
                    audio = fresh_audio

                stt_timer = Timer()
                stt_res = stt_backend.transcribe(
                    audio=audio,
                    sample_rate=TARGET_SR,
                    language=args.language,
                    return_words=False,
                )
                stt_time = stt_timer.elapsed()

                transcript = (stt_res.text or "").strip()
                print(f"\n[BARGE-IN STT] {transcript}")
                print(f"[BARGE-IN STT Latency] {stt_time:.3f}s")

                interrupt_collecting = False
                interrupt_segmenter.reset()
                captured_interrupt_frames = []
                controller.finish_interruption_capture(transcript, interrupted_last_spoken)
                continue

            # ---------------- INTERRUPTION ONSET DETECTION ----------------
            if state == TurnController.SPEAKING:
                if not args.barge_in:
                    continue

                rms = frame_rms(frame_bytes)
                if rms >= args.interrupt_rms_threshold:
                    onset_count += 1
                else:
                    onset_count = 0

                if onset_count >= args.interrupt_onset_frames:
                    interrupted_last_spoken = controller.get_last_spoken_sentence()
                    started = controller.begin_interruption()
                    onset_count = 0

                    if started:
                        flushed = flush_frame_queue(q_frames)
                        if flushed:
                            print(f"[BARGE-IN] Flushed {flushed} buffered mic frames after cutover.")

                        main_segmenter.reset()
                        interrupt_segmenter.reset()
                        interrupt_collecting = True
                        interrupt_skip_until = perf_counter() + args.interrupt_settle

                        captured_interrupt_frames = list(recent_frames)
                        captured_interrupt_frames.append(frame_bytes)
                    continue

                continue

            # While THINKING or INTERRUPTING, do not treat speech as normal turn yet
            if state in (TurnController.THINKING, TurnController.INTERRUPTING):
                main_segmenter.reset()
                onset_count = 0
                continue

            onset_count = 0

            # ---------------- LISTENING MODE ----------------
            utt = main_segmenter.process_frame(frame_bytes, ts)
            if utt is None:
                continue

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
            print(f"[STT] {transcript}")
            print(f"[STT Latency] {stt_time:.3f}s")

            controller.start_user_turn(transcript)

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