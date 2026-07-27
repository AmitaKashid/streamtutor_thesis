from __future__ import annotations

import sys
import argparse
import queue
import threading
import time
import copy
import re
from pathlib import Path
from time import perf_counter
from collections import deque
import itertools
import base64
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
from llm.llmcli.providers.remote_http import RemoteHTTPProvider

# ---- TTS ----
from tts.base_engine import BaseTTSEngine, StreamingTextChunk

try:
    from tts.chatterbox_engine import ChatterboxEngine
except Exception:
    ChatterboxEngine = None

try:
    from tts.elevenlabs_streaming_engine import ChatterboxHTTPStreamingEngine
except Exception:
    ChatterboxHTTPStreamingEngine = None
    
from tts.tts_queue import TTSRequest  # reuse dataclass for consistency
from tts.voxtral_server_engine import VoxtralServerEngine, VoxtralServerConfig
#from tts.config import TTS_BACKEND
from tts.config import (
    TTS_BACKEND,
    KOKORO_VOICE,
    KOKORO_LANG_CODE,
    KOKORO_SPEED,
    KOKORO_SAMPLE_RATE,
    KOKORO_STREAM_AUDIO_CHUNK_MS,
    VOXTRAL_BASE_URL,
    VOXTRAL_ENDPOINT,
    VOXTRAL_MODEL,
    VOXTRAL_VOICE,
    VOXTRAL_RESPONSE_FORMAT,
    VOXTRAL_CONNECT_TIMEOUT_S,
    VOXTRAL_READ_TIMEOUT_S,
    VOXTRAL_SAMPLE_RATE_HINT,
)
# ---- Streaming helpers ----
#from llm.llmcli.common.text_chunker import TextChunker, ChunkerConfig
from llm.llmcli.common.prosody_chunker import ProsodyChunker, ProsodyChunkerConfig
from llm.llmcli.common.audio_smoother import AudioCrossfader, CrossfadeConfig
from llm.llmcli.common.audio_output import AudioOutput, AudioOutputConfig

# ---- Prompt / RAG ----
from prompts.registry import compose_prompt
from rag.rag_retrieve import RagRetriever

from judge.judge import LocalLLMJudge

from tts.kokoro_engine import KokoroEngine, KokoroConfig

from dataclasses import dataclass

@dataclass
class TTSAudioChunk:
    text: str
    audio: np.ndarray
    sr: int
    duration_s: float
# ---------------- Timing ----------------
class Timer:
    def __init__(self):
        self.t0 = perf_counter()

    def elapsed(self) -> float:
        return perf_counter() - self.t0

# ---------------- Conversation Context Memory ----------------
class ConversationMemory:
    """
    Stores the full User/Assistant conversation history.

    Purpose:
      - Every new user question is sent together with previous turns.
      - Context length grows over time.
      - This allows latency experiments with increasing prompt size.

    Notes:
      - max_turns=None means unlimited turns.
      - max_chars=None means unlimited characters.
      - For thesis latency experiments, use unlimited first.
    """

    def __init__(self, max_turns: int | None = None, max_chars: int | None = None):
        self._lock = threading.Lock()
        self.turns: list[dict[str, str]] = []
        self.max_turns = max_turns
        self.max_chars = max_chars

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        user_text = (user_text or "").strip()
        assistant_text = (assistant_text or "").strip()

        if not user_text or not assistant_text:
            return

        with self._lock:
            self.turns.append(
                {
                    "user": user_text,
                    "assistant": assistant_text,
                }
            )

            if self.max_turns is not None and self.max_turns > 0:
                self.turns = self.turns[-self.max_turns:]

    def format_history(self) -> str:
        with self._lock:
            turns_copy = list(self.turns)

        if not turns_copy:
            return ""

        lines = []
        for i, turn in enumerate(turns_copy, 1):
            lines.append(f"Turn {i}")
            lines.append(f"User: {turn['user']}")
            lines.append(f"Assistant: {turn['assistant']}")
            lines.append("")

        history = "\n".join(lines).strip()

        if self.max_chars is not None and self.max_chars > 0 and len(history) > self.max_chars:
            history = history[-self.max_chars:]
            history = "[Conversation history truncated to most recent characters]\n" + history

        return history

    def build_contextual_user_query(self, current_user_query: str) -> str:
        current_user_query = (current_user_query or "").strip()
        history = self.format_history()

        if not history:
            return current_user_query

        return (
            "Previous conversation between the user and assistant:\n"
            f"{history}\n\n"
            "Current user question:\n"
            f"{current_user_query}\n\n"
            "Use the previous conversation as context. Answer the current user question directly."
        )

    def stats(self) -> dict:
        history = self.format_history()
        with self._lock:
            turns_count = len(self.turns)

        return {
            "turns": turns_count,
            "chars": len(history),
            "approx_words": len(history.split()),
        }

def start_terminal_dots(message: str = "Preparing your explanation"):
    """
    Show animated dots in terminal until stop_event is set.
    Returns (stop_event, thread).
    """
    stop_event = threading.Event()

    def _worker():
        for dots in itertools.cycle(["", ".", "..", "..."]):
            if stop_event.is_set():
                break
            sys.stdout.write("\r" + message + dots + "   ")
            sys.stdout.flush()
            time.sleep(0.4)

        # clear the line when stopping
        sys.stdout.write("\r" + " " * (len(message) + 6) + "\r")
        sys.stdout.flush()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return stop_event, t
# ---------------- LLM ----------------
def build_llm(args):
    if args.llm_backend == "remote":
        if not args.llm_server_url:
            raise ValueError("--llm_server_url is required when --llm_backend remote")

        provider = RemoteHTTPProvider(
            base_url=args.llm_server_url,
            connect_timeout=args.llm_connect_timeout,
            read_timeout=args.llm_read_timeout,
        )
        if not provider.healthcheck():
            raise RuntimeError(f"Remote LLM server not reachable at {args.llm_server_url}")
        return provider

    spec = get_model_spec(args.llm_model)

    model_key = args.llm_model.lower()

    temp = 0.2
    top_p = 0.9
    n_ctx = 2048

    if "qwen" in model_key:
        temp = 0.6
        top_p = 0.95

    elif "ministral" in model_key:
        temp = 0.3
        top_p = 0.9
        n_ctx = 4096

    elif "gemma" in model_key:
        temp = 1.0
        top_p = 0.95
        n_ctx = 4096

    return LlamaCppProvider(
    model_path=spec.gguf_path,
    chat_format=spec.chat_format,
    system_prompt=spec.system_prompt,
    n_ctx=n_ctx,
    n_threads=12,
    n_gpu_layers=-1,
    n_batch=512,
    n_ubatch=512,
    flash_attn=True,
    offload_kqv=True,
    max_new_tokens=360,
    temperature=temp,
    top_p=top_p,
    seed=0,
)
# ---------------- LLMJudge ----------------
def build_judge_llm(model_key: str) -> LlamaCppProvider:
    spec = get_model_spec(model_key)
    return LlamaCppProvider(
        model_path=spec.gguf_path,
        chat_format=spec.chat_format,
        system_prompt="You are a strict educational judge that outputs JSON only.",
        n_ctx=2048,
        n_threads=8,
        n_gpu_layers=-1,
        max_new_tokens=180,
        temperature=0.0,
        top_p=0.9,
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
    "Bye-bye.",
    
}

# ---------------- Strict Built-in Barge-in Policy ----------------
# This is NOT user-configurable.
# The system must always allow real user interruption,
# but must not interrupt itself from its own TTS echo.

STRICT_BARGE_IN_MIN_WORDS = 3
STRICT_BARGE_IN_ECHO_OVERLAP = 0.55

# For backend microphone mode, raw RMS is not reliable enough to stop TTS.
# The professional path is frontend/browser AEC or a real backend AEC pipeline.
ALLOW_RAW_BACKEND_RMS_BARGE_IN = False
# ---------------- Runtime Configuration ----------------

ALLOWED_LLM_MODELS = {
    "llama3_8b_q4",
    "llama3_8b_q5",
    "llama3_8b_q6",
    "qwen3_5_9b_q4",
    "gemma_4_26b",
}

ALLOWED_TTS_BACKENDS = {
    "kokoro",
    "chatterbox",
    "chatterbox_stream",
    "voxtral",
}

ALLOWED_TEACHING_STYLES = {
    "socratic",
    "adaptive",
    "inquiry",
    "collaborative",
}


def _safe_int(value, default: int, min_value: int = 0, max_value: int | None = None) -> int:
    try:
        out = int(value)
    except Exception:
        out = default

    out = max(min_value, out)

    if max_value is not None:
        out = min(out, max_value)

    return out


def apply_runtime_config(base_args, config: dict | None):
    """
    Create a per-request copy of argparse args and apply frontend overrides.

    This does not mutate the global startup args.
    It is safe to call for every websocket request.
    """
    runtime_args = copy.copy(base_args)

    if not isinstance(config, dict):
        return runtime_args

    llm_model = config.get("llm_model")
    if llm_model in ALLOWED_LLM_MODELS:
        runtime_args.llm_model = llm_model

    tts_backend = config.get("tts_backend")
    if tts_backend in ALLOWED_TTS_BACKENDS:
        runtime_args.tts_backend = tts_backend

    teaching_style = config.get("teaching_style")
    if teaching_style in ALLOWED_TEACHING_STYLES:
        runtime_args.teaching_style = teaching_style

    if "rag" in config:
        runtime_args.rag = bool(config.get("rag"))

    runtime_args.rag_top_k = _safe_int(
        config.get("rag_top_k", getattr(runtime_args, "rag_top_k", 2)),
        default=getattr(runtime_args, "rag_top_k", 2) or 2,
        min_value=1,
        max_value=10,
    )

    if "conversation_context" in config:
        runtime_args.conversation_context = bool(config.get("conversation_context"))

    runtime_args.conversation_max_turns = _safe_int(
        config.get("conversation_max_turns", getattr(runtime_args, "conversation_max_turns", 0)),
        default=getattr(runtime_args, "conversation_max_turns", 0),
        min_value=0,
        max_value=50,
    )

    return runtime_args


def effective_config_payload(args) -> dict:
    """
    Configuration confirmation sent back to frontend.
    This is useful for proving that frontend config was actually applied.
    """
    return {
        "llm_model": getattr(args, "llm_model", None),
        "llm_backend": getattr(args, "llm_backend", None),
        "llm_server_url": getattr(args, "llm_server_url", None),
        "tts_backend": getattr(args, "tts_backend", None),
        "rag": bool(getattr(args, "rag", False)),
        "rag_top_k": getattr(args, "rag_top_k", None),
        "teaching_style": getattr(args, "teaching_style", None),
        "conversation_context": bool(getattr(args, "conversation_context", False)),
        "conversation_max_turns": getattr(args, "conversation_max_turns", 0),
    }


def print_effective_config(args) -> None:
    print("\n[EFFECTIVE RUNTIME CONFIG]")
    print(f"  llm_backend          : {getattr(args, 'llm_backend', None)}")
    print(f"  llm_server_url       : {getattr(args, 'llm_server_url', None)}")
    print(f"  llm_model            : {getattr(args, 'llm_model', None)}")
    print(f"  tts_backend          : {getattr(args, 'tts_backend', None)}")
    print(f"  rag                  : {getattr(args, 'rag', None)}")
    print(f"  rag_top_k            : {getattr(args, 'rag_top_k', None)}")
    print(f"  teaching_style       : {getattr(args, 'teaching_style', None)}")
    print(f"  conversation_context : {getattr(args, 'conversation_context', None)}")
    print(f"  conversation_max_turns: {getattr(args, 'conversation_max_turns', None)}")


def torch_to_np_tc(wav_torch) -> np.ndarray:
    """
    Convert torch waveform [C, T] or [T] to numpy float32 [T, C].
    """
    w = wav_torch.detach().cpu()
    if w.dim() == 1:
        w = w.unsqueeze(0)
    w = w.transpose(0, 1).contiguous()
    return w.numpy().astype(np.float32, copy=False)
def np_audio_to_pcm16_base64(audio_np: np.ndarray) -> tuple[str, int]:
    """
    Convert float audio [-1, 1] to PCM16 base64 for browser playback.
    Returns base64 string and number of channels.
    """
    if audio_np.ndim == 1:
        audio_np = audio_np[:, None]

    audio_np = np.clip(audio_np, -1.0, 1.0)
    pcm16 = (audio_np * 32767.0).astype(np.int16)

    return base64.b64encode(pcm16.tobytes()).decode("ascii"), audio_np.shape[1]

def frame_rms(frame_bytes: bytes) -> float:
    if not frame_bytes:
        return 0.0
    a = np.frombuffer(frame_bytes, dtype=np.int16)
    if a.size == 0:
        return 0.0
    a = a.astype(np.float32)
    return float(np.sqrt(np.mean(a * a) + 1e-12))

def normalize_words_for_echo_check(text: str) -> list[str]:
    """
    Normalize text for comparing a candidate interruption
    against recently spoken assistant chunks.
    """
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [w for w in text.split() if len(w) > 1]


def is_likely_assistant_echo(
    candidate_text: str,
    spoken_chunks: list[str],
    *,
    min_words: int = STRICT_BARGE_IN_MIN_WORDS,
    overlap_threshold: float = STRICT_BARGE_IN_ECHO_OVERLAP,
) -> bool:
    """
    Reject barge-in if the recognized text mostly matches what the assistant
    was just speaking.

    Example:
      assistant said: "Decision trees split data based on attributes..."
      microphone hears: "decision trees split data..."
      -> reject as assistant echo.
    """
    candidate_words = normalize_words_for_echo_check(candidate_text)
    spoken_words = set(normalize_words_for_echo_check(" ".join(spoken_chunks)))

    if len(candidate_words) < min_words:
        return True

    if not spoken_words:
        return False

    overlap = sum(1 for w in candidate_words if w in spoken_words) / max(1, len(candidate_words))
    return overlap >= overlap_threshold


def is_valid_barge_in_text(candidate_text: str, spoken_chunks: list[str]) -> bool:
    """
    Single strict validation gate for all interruption text.
    This should be used for frontend barge-in and backend interruption STT.
    """
    candidate_text = (candidate_text or "").strip()
    if not candidate_text:
        return False

    norm = candidate_text.lower().strip(" .!?")

    if norm in HALLUCINATION_PHRASES:
        print(f"[BARGE-IN FILTER] Rejected known hallucination/echo phrase: {candidate_text}")
        return False

    if is_likely_assistant_echo(candidate_text, spoken_chunks):
        print(f"[BARGE-IN FILTER] Rejected assistant echo: {candidate_text}")
        print("[BARGE-IN FILTER] Recently spoken chunks:")
        for i, ch in enumerate(spoken_chunks, 1):
            print(f"  {i}. {ch}")
        return False

    return True



class PlaybackControl:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.audio_out = None
        self.interrupted = threading.Event()
        self.finished_chunks = deque(maxlen=3)
        self.current_chunk = None
        self.current_chunk_started_at = 0.0
        self.interrupt_snapshot = []

        # Speaker-safe mode state
        self.tts_started_at = 0.0
        self.tts_active = threading.Event()

    def attach_audio_out(self, audio_out) -> None:
        with self._lock:
            self.audio_out = audio_out

    def clear_audio_out(self) -> None:
        with self._lock:
            self.audio_out = None

    def reset_interrupt(self) -> None:
        self.interrupted.clear()

    def mark_tts_started(self) -> None:
        self.tts_started_at = perf_counter()
        self.tts_active.set()

    def mark_tts_stopped(self) -> None:
        self.tts_active.clear()

    def seconds_since_tts_start(self) -> float:
        if self.tts_started_at <= 0:
            return 999.0
        return perf_counter() - self.tts_started_at

    def stop_now(self) -> None:
        """
        Stop both batch playback (sounddevice) and stream playback (AudioOutput).
        Capture interruption context before clearing playback state.
        """
        self.interrupted.set()

        # Freeze the interruption context immediately:
        self.capture_interrupt_snapshot()

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

        self.clear_chunk_tracking()
        self.mark_tts_stopped()

    def mark_chunk_started(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            self.current_chunk = text
            self.current_chunk_started_at = perf_counter()

    def mark_chunk_finished(self) -> None:
        with self._lock:
            if self.current_chunk:
                self.finished_chunks.append(self.current_chunk)
            self.current_chunk = None
            self.current_chunk_started_at = 0.0

    def clear_chunk_tracking(self) -> None:
        with self._lock:
            self.current_chunk = None
            self.current_chunk_started_at = 0.0

    def capture_interrupt_snapshot(self) -> list[str]:
        with self._lock:
            prev_chunk = self.finished_chunks[-1] if self.finished_chunks else None
            curr_chunk = self.current_chunk

            out = []
            if prev_chunk:
                out.append(prev_chunk)
            if curr_chunk:
                out.append(curr_chunk)

            self.interrupt_snapshot = out[:]
            return out[:]

    def get_interrupt_snapshot(self) -> list[str]:
        with self._lock:
            return list(self.interrupt_snapshot)

    def clear_interrupt_snapshot(self) -> None:
        with self._lock:
            self.interrupt_snapshot = []
    def get_interrupt_context(self, n_finished: int = 2) -> list[str]:
        with self._lock:
            out = list(self.finished_chunks)[-n_finished:]
            if self.current_chunk:
                out.append(self.current_chunk)
            return out


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


def build_interrupt_user_query(interruption_text: str, interrupted_chunks: list[str]) -> str:
    interrupted_chunks = [c.strip() for c in interrupted_chunks if c and c.strip()]
    if interrupted_chunks:
        chunks_text = "\n".join(f"- {c}" for c in interrupted_chunks)
    else:
        chunks_text = "- (no previous spoken chunks captured)"

    return (
        "The user interrupted the assistant.\n\n"
        "Last spoken chunks before interruption:\n"
        f"{chunks_text}\n\n"
        "User interruption text:\n"
        f"{interruption_text.strip()}\n\n"
        "Respond directly to the interruption while continuing the explanation naturally."
    )



def run_llm_tts_pipeline(
    *,
    transcript: str,
    stt_time: float,
    total_timer: Timer,
    llm: LlamaCppProvider,
    tts_engine: BaseTTSEngine,
    retriever: RagRetriever | None,
    args,
    playback: PlaybackControl,
    pipeline_active: threading.Event,
    judge: LocalLLMJudge | None,
    conversation_memory: ConversationMemory | None = None,
    event_cb=None,
    tts_busy: threading.Event | None = None,
    drain_after_tts: threading.Event | None = None,
    interruption_text: str | None = None,
    interrupted_chunks: list[str] | None = None,
) -> None:
    """
    Shared post-STT pipeline for both mic mode and file mode:
      transcript -> (RAG) -> prompt -> LLM -> TTS

    For interruption mode:
      interruption_text + last spoken chunks -> (RAG) -> prompt -> LLM -> TTS
    """
    norm = transcript.lower().strip(" .!?")
    playback.reset_interrupt()
    playback.clear_interrupt_snapshot()

    # Give streaming TTS engines access to the interruption flag.
    # Qwen uses this to close its WebSocket stream immediately during barge-in.
    if hasattr(tts_engine, "set_cancel_event"):
        tts_engine.set_cancel_event(playback.interrupted)

    def emit(payload: dict) -> None:
        if event_cb is None:
            return
        try:
            event_cb(payload)
        except Exception:
            pass

    print_effective_config(args)
    emit({
        "type": "config_applied",
        "data": effective_config_payload(args),
    })
    if stt_time > 0:
        print(f"[STT] {transcript}")
        print(f"[STT Full Transcription Latency] {stt_time:.3f}s")
    else:
        print(f"[TEXT] {transcript}")

    if not transcript:
        print("[INFO] Empty transcript.")
        return

    if len(transcript.split()) < args.min_words:
        print("[STT] Ignored short utterance")
        return

    if norm in HALLUCINATION_PHRASES:
        print(f"[STT] Ignored hallucination: '{transcript}'")
        return

    is_interrupt = interruption_text is not None
    retrieval_query = interruption_text.strip() if is_interrupt else transcript.strip()
    emit({"type": "stage", "value": "retrieving" if retriever is not None else "thinking"})
    # ---------------- RAG ----------------
    rag_time = 0.0
    context = ""
    retrieved_ids = []
    if retriever is not None:
        rag_timer = Timer()
        hits = retriever.retrieve(retrieval_query, top_k=args.rag_top_k)
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
            retrieved_ids.append(id_value)
            if score is not None:
                print(f"    {i}. id={id_value} | score={score:.4f}")
            else:
                print(f"    {i}. id={id_value}")
            emit({
                "type": "sources",
                "data": [
                    {
                        "id": str(chunk_id or slide_id or f"source_{i}"),
                        "title": str(chunk_id or slide_id or "Retrieved source"),
                        "subtitle": f"Top-{i}",
                        "snippet": getattr(h, "text", "")[:220] if hasattr(h, "text") else "",
                        "score": float(score) if score is not None else None,
                    }
                    for i, h in enumerate(hits, 1)
                    for chunk_id, slide_id, score in [(
                        getattr(h, "chunk_id", None),
                        getattr(h, "slide_id", None),
                        getattr(h, "score", None),
                    )]
                ],
            })
            emit({"type": "metrics", "data": {"ragLatency": rag_time}})

        if args.rag_print_context:
            if args.rag_context_chars == 0:
                preview = context
            else:
                preview = context[: args.rag_context_chars]
                if len(context) > args.rag_context_chars:
                    preview += "\n\n... [truncated]"
            print("\n[RAG Context -> LLM]\n" + preview + "\n")

    # Build final user query for LLM
    if is_interrupt:
        final_user_query = build_interrupt_user_query(
            interruption_text=interruption_text,
            interrupted_chunks=interrupted_chunks or [],
        )
    else:
        final_user_query = transcript

    # ---------------- Conversation Context ----------------
    # Retrieval still uses only the current question/interruption text.
    # The LLM receives the growing conversation history + current question.
    raw_current_user_query = final_user_query

    if getattr(args, "conversation_context", False) and conversation_memory is not None:
        memory_stats_before = conversation_memory.stats()
        emit({
            "type": "metrics",
            "data": {
                "conversationTurns": memory_stats_before["turns"],
                "conversationChars": memory_stats_before["chars"],
            },
        })

        final_user_query = conversation_memory.build_contextual_user_query(
            current_user_query=final_user_query
        )

        print("\n[CONVERSATION CONTEXT]")
        print(f"  enabled           : True")
        print(f"  previous turns    : {memory_stats_before['turns']}")
        print(f"  history chars     : {memory_stats_before['chars']}")
        print(f"  history words     : {memory_stats_before['approx_words']}")
        print(f"  final query chars : {len(final_user_query)}")

        if getattr(args, "conversation_print_context", False):
            print("\n[CONVERSATION HISTORY -> LLM]")
            print(conversation_memory.format_history())
            print("")
    else:
        print("\n[CONVERSATION CONTEXT]")
        print("  enabled           : False")

    llm_prompt = compose_prompt(
        style=args.teaching_style,
        user_query=final_user_query,
        context=context if retriever is not None else None,
    )
    emit({
        "type": "metrics",
        "data": {
            "promptChars": len(llm_prompt),
            "promptWords": len(llm_prompt.split()),
        },
    })
    print("\n[PROMPT SIZE]")
    print(f"  prompt chars      : {len(llm_prompt)}")
    print(f"  prompt words      : {len(llm_prompt.split())}")
    print("\n[LLM PROMPT STATS]")
    print(f"  prompt_chars : {len(llm_prompt)}")
    preview = llm_prompt[:1200]
    if len(llm_prompt) > 1200:
        preview += "\n... [truncated]"
    print("[LLM PROMPT PREVIEW]")
    print(preview)
    if is_interrupt:
        print("\n[INTERRUPTION -> LLM]")
        print("Interruption text:")
        print(interruption_text)
        print("\nLast spoken chunks:")
        if interrupted_chunks:
            for i, ch in enumerate(interrupted_chunks, 1):
                print(f"  {i}. {ch}")
        else:
            print("  (none)")
        print("\n[FINAL PROMPT SENT TO LLM]")
        print(llm_prompt)
        print("")

    # ---------------- Mode A: batch ----------------
    if args.mode == "batch":
        llm_timer = Timer()
        llm_res = llm.generate(llm_prompt)
        answer = (llm_res.text or "").strip()

        # ---- LLM AS JUDGE (ASYNC) ----
        if judge is not None:
            judge.judge_and_log_async(
                question=transcript,
                answer=answer,
                teaching_style=args.teaching_style,
                context=context,
                retrieved_ids=retrieved_ids,
                interruption_text=interruption_text,
                interrupted_chunks=interrupted_chunks or [],
                metadata={
                    "mode": args.mode,
                    "rag_enabled": retriever is not None,
                    "is_interrupt": interruption_text is not None,
                },
            )
        llm_time = llm_timer.elapsed()

        print("\n[LLM Answer]")
        print(answer)
        print(f"[LLM Latency] {llm_time:.3f}s")

        if not answer:
            print("[INFO] Empty LLM output; nothing to synthesize.")
            return

        # Store completed turn after successful LLM answer
        if getattr(args, "conversation_context", False) and conversation_memory is not None:
            conversation_memory.add_turn(
                user_text=raw_current_user_query,
                assistant_text=answer,
            )
            memory_stats_after = conversation_memory.stats()
            print("\n[CONVERSATION MEMORY UPDATED]")
            print(f"  stored turns      : {memory_stats_after['turns']}")
            print(f"  history chars     : {memory_stats_after['chars']}")
            print(f"  history words     : {memory_stats_after['approx_words']}")

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

        if tts_engine.supports_streaming_input():
            raise RuntimeError("Batch mode is not supported with the streaming TTS backend. Use --mode stream-input or switch to --tts_backend chatterbox.")

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

            # record full answer as one spoken unit in batch mode
            #playback.record_spoken_chunk(answer)
            playback.mark_chunk_started(answer)
            playback.mark_chunk_finished()


            if req.play_audio:
                try:
                    import sounddevice as sd
                except Exception:
                    sd = None
                if sd is None:
                    print("[TTS] sounddevice not available, skipping playback.")
                else:
                    wav_np = torch_to_np_tc(wav)
                    playback.mark_tts_started()
                    try:
                        sd.play(wav_np, sr)
                        sd.wait()
                    finally:
                        playback.mark_tts_stopped()

                    if playback.interrupted.is_set():
                        print("[INTERRUPT] Batch playback stopped by user speech.")
                        return
        finally:
            pipeline_active.clear()
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



    # ---------------- Mode B: streaming ----------------
    print(f"\n[MODE] {args.mode}")
    emit({"type": "stage", "value": "thinking"})
    dots_stop_event, dots_thread = start_terminal_dots("Preparing your explanation")
    dots_stopped = False

    audio_prebuffer = []
    audio_prebuffer_seconds = 0.0
    playback_started = False
    #prebuffer_target_seconds = 0.35 if args.mode == "stream-input" else 0.2
    prebuffer_target_seconds = 0.35 
    full_answer_parts = []
    llm_first_text_t = None
    first_audio_t = None
    llm_stream_done_t = None
    
    last_enqueued_chunk = ""
    first_chunk_sent_to_tts_t = None
    first_tts_synth_start_t = None
    tts_play_start_t = None
    first_tts_actual_playback_t = None

    #chunk_cfg = ChunkerConfig(
    #    first_max_chars=args.chunk_first,
    #    mid_max_chars=args.chunk_mid,
    #    later_max_chars=args.chunk_later,
    #    max_wait_s=args.chunk_wait,
    #    min_chars_timer=120,
    #    min_chars_sentence=80,
    #)

    #chunk_cfg = ChunkerConfig(
    #first_max_chars=args.chunk_first,
    #mid_max_chars=args.chunk_mid,
    #later_max_chars=args.chunk_later,
    #max_wait_s=args.chunk_wait,
    #min_chars_timer=140,
    #min_chars_sentence=100,
    #)
    #chunker = TextChunker(chunk_cfg)
    
    chunk_cfg = ProsodyChunkerConfig(
    first_target_chars=args.chunk_first,
    later_target_chars=args.chunk_mid,
    min_sentence_chars=45,
    min_comma_chars=60,
    max_chars=args.chunk_later,
    max_wait_s=args.chunk_wait,
    )
    chunker = ProsodyChunker(chunk_cfg)
    
    audio_out = None
    crossfader = None
    llm_request_start_t = None
    llm_first_text_t = None
    llm_stream_done_t = None

    tts_proc_total = 0.0
    chunks_count = 0
    audio_chunk_count = 0
    first_audio_t = None

    def text_event_iter():
        nonlocal llm_first_text_t, llm_stream_done_t, chunks_count, last_enqueued_chunk
        nonlocal first_chunk_sent_to_tts_t
        nonlocal dots_stopped
        nonlocal llm_request_start_t

        # Prefer provider-side timing if available
        if hasattr(llm, "stream_with_timing"):
            for piece, meta in llm.stream_with_timing(llm_prompt):
                if llm_request_start_t is None:
                    llm_request_start_t = meta.get("request_start_s", perf_counter())

                if llm_first_text_t is None and piece.strip():
                    llm_first_text_t = meta.get("first_token_s", perf_counter())

                    ttft_s = llm_first_text_t - llm_request_start_t

                    print("\n[LLM FIRST TOKEN]")
                    print(f"  ttft_s : {ttft_s:.3f}")

                    emit({"type": "metrics", "data": {"ttft": ttft_s}})
                    emit({"type": "stage", "value": "streaming"})

                    if not dots_stopped:
                        dots_stop_event.set()
                        dots_thread.join(timeout=0.2)
                        dots_stopped = True

                full_answer_parts.append(piece)

                for chunk_text in chunker.feed([piece]):
                    if playback.interrupted.is_set():
                        return

                    if not chunk_text:
                        continue

                    clean_chunk = chunk_text.replace("**", "").replace("*", "").strip()
                    if not clean_chunk:
                        continue

                    if clean_chunk == last_enqueued_chunk:
                        continue

                    chunks_count += 1
                    print(f"\n[TEXT CHUNK {chunks_count}] {clean_chunk}", flush=True)
                    emit({"type": "llm_chunk", "text": clean_chunk + " "})

                    last_enqueued_chunk = clean_chunk
                    yield StreamingTextChunk(text=clean_chunk + " ", flush=True, is_final=False)

            llm_stream_done_t = perf_counter()

        else:
            # Fallback if provider-side timing is unavailable.
            # Start timing immediately before the actual LLM stream request.
            llm_request_start_t = perf_counter()

            for piece in llm.stream(llm_prompt):
                if llm_first_text_t is None and piece.strip():
                    llm_first_text_t = perf_counter()

                    ttft_s = llm_first_text_t - llm_request_start_t

                    print("\n[LLM FIRST TOKEN]")
                    print(f"  ttft_s : {ttft_s:.3f}")

                    emit({"type": "metrics", "data": {"ttft": ttft_s}})
                    emit({"type": "stage", "value": "streaming"})

                    if not dots_stopped:
                        dots_stop_event.set()
                        dots_thread.join(timeout=0.2)
                        dots_stopped = True
                full_answer_parts.append(piece)

                for chunk_text in chunker.feed([piece]):
                    if playback.interrupted.is_set():
                        return

                    if not chunk_text:
                        continue

                    clean_chunk = chunk_text.replace("**", "").replace("*", "").strip()
                    if not clean_chunk:
                        continue

                    if clean_chunk == last_enqueued_chunk:
                        continue

                    chunks_count += 1
                    print(f"\n[TEXT CHUNK {chunks_count}] {clean_chunk}", flush=True)
                    emit({"type": "llm_chunk", "text": clean_chunk + " "})

                    last_enqueued_chunk = clean_chunk
                    yield StreamingTextChunk(text=clean_chunk + " ", flush=True, is_final=False)

            llm_stream_done_t = perf_counter()

        if not playback.interrupted.is_set():
            last = chunker.flush()
            if last:
                clean_last = last.replace("**", "").replace("*", "").strip()
                if clean_last and clean_last != last_enqueued_chunk:
                    chunks_count += 1
                    print(f"\n[TEXT CHUNK {chunks_count}] {clean_last}", flush=True)

                    last_enqueued_chunk = clean_last
                    yield StreamingTextChunk(text=clean_last + " ", flush=True, is_final=True)
                    return

        yield StreamingTextChunk(text="", flush=True, is_final=True)
    def make_chunk_callbacks(chunk_text: str | None):
        if not chunk_text:
            return None, None

        def _on_start():
            nonlocal first_tts_actual_playback_t, first_chunk_sent_to_tts_t

            if first_tts_actual_playback_t is None:
                first_tts_actual_playback_t = perf_counter()

                if first_chunk_sent_to_tts_t is not None:
                    tts_actual_start_latency_now = (
                        first_tts_actual_playback_t - first_chunk_sent_to_tts_t
                    )

                    print(f"\n[TTS ACTUAL PLAYBACK START]")
                    print(f"  tts_actual_start_latency : {tts_actual_start_latency_now:.3f} s")

                    emit({
                        "type": "metrics",
                        "data": {
                            "ttsFirstChunk": tts_actual_start_latency_now,
                        },
                    })

            playback.mark_chunk_started(chunk_text)

        def _on_end():
            playback.mark_chunk_finished()

        return _on_start, _on_end
    audio_chunk_count = 0

    def enqueue_audio_chunk(chunk_np: np.ndarray, sr: int, chunk_text: str | None = None) -> None:
        nonlocal audio_out, crossfader, playback_started, audio_prebuffer_seconds, first_audio_t, audio_chunk_count
        nonlocal tts_play_start_t, first_tts_actual_playback_t
        if playback.interrupted.is_set():
            return
        if getattr(args, "browser_audio", False):
            audio_chunk_count += 1

            if chunk_text:
                playback.mark_chunk_started(chunk_text)
                playback.mark_chunk_finished()

            if not playback_started:
                playback.mark_tts_started()
                playback_started = True

                if first_audio_t is None:
                    first_audio_t = perf_counter()
                    emit({"type": "stage", "value": "speaking"})
                    emit({
                        "type": "metrics",
                        "data": {"firstAudio": first_audio_t - total_timer.t0},
                    })

                # In browser-audio mode, Python cannot know exact speaker playback start.
                # This records first audio chunk emitted to the browser as the practical
                # "TTS first chunk" latency.
                if first_tts_actual_playback_t is None:
                    first_tts_actual_playback_t = perf_counter()
                    tts_play_start_t = first_tts_actual_playback_t

                    if first_chunk_sent_to_tts_t is not None:
                        emit({
                            "type": "metrics",
                            "data": {
                                "ttsFirstChunk": first_tts_actual_playback_t - first_chunk_sent_to_tts_t,
                            },
                        })

            audio_b64, channels = np_audio_to_pcm16_base64(chunk_np)

            emit({
                "type": "audio_chunk",
                "data": audio_b64,
                "sample_rate": sr,
                "channels": channels,
            })

            return
        audio_chunk_count += 1

        if audio_out is None:
            channels = chunk_np.shape[1] if chunk_np.ndim == 2 else 1
            audio_out = AudioOutput(sr=sr, channels=channels, cfg=AudioOutputConfig())
            audio_out.start()
            playback.attach_audio_out(audio_out)

        if not args.speak:
            return

        chunk_duration_s = len(chunk_np) / sr

        if not playback_started:
            audio_prebuffer.append((chunk_np, chunk_text, chunk_duration_s))
            audio_prebuffer_seconds += chunk_duration_s

            if audio_prebuffer_seconds >= prebuffer_target_seconds:
                playback.mark_tts_started()
                if tts_play_start_t is None:
                    tts_play_start_t = perf_counter()
                for buffered_np, buffered_text, _dur in audio_prebuffer:
                    on_start, on_end = make_chunk_callbacks(buffered_text)
                    audio_out.enqueue(buffered_np, on_start=on_start, on_end=on_end)

                audio_prebuffer.clear()
                audio_prebuffer_seconds = 0.0
                playback_started = True

                if first_audio_t is None:
                    first_audio_t = perf_counter()
                    emit({"type": "stage", "value": "speaking"})
                    emit({
                        "type": "metrics",
                        "data": {"firstAudio": first_audio_t - total_timer.t0},
                    })
        else:
            playback.mark_tts_started()
            on_start, on_end = make_chunk_callbacks(chunk_text)
            audio_out.enqueue(chunk_np, on_start=on_start, on_end=on_end)

            

    set_tts_state(True, tts_busy, drain_after_tts)
    try:
        if args.mode == "stream-input":
            if not tts_engine.supports_streaming_input():
                raise RuntimeError(
                    "True stream-input mode requires a streaming TTS backend. "
                    "Set --tts_backend chatterbox_stream and make sure the local Chatterbox streaming server is running."
                )

            tts_timer = Timer()
            for chunk_np, sr in tts_engine.stream_text(
                text_event_iter(),
                voice_prompt=Path(args.voice_prompt) if args.voice_prompt else None,
                config=None,
            ):
                if playback.interrupted.is_set():
                    break
                enqueue_audio_chunk(chunk_np, sr)

            # Real TTS processing time if engine exposes it; otherwise fallback
            tts_proc_total = getattr(tts_engine, "last_stream_tts_proc_total", tts_timer.elapsed())


        else:
            text_chunk_queue = queue.Queue()

            def tts_worker():
                nonlocal audio_out, crossfader, tts_proc_total, first_audio_t
                nonlocal playback_started, audio_prebuffer_seconds
                nonlocal first_tts_synth_start_t, first_chunk_sent_to_tts_t

                while True:
                    chunk_text = text_chunk_queue.get()
                    if chunk_text is None:
                        text_chunk_queue.task_done()
                        break

                    if playback.interrupted.is_set():
                        text_chunk_queue.task_done()
                        break

                    if first_tts_synth_start_t is None:
                        first_tts_synth_start_t = perf_counter()

                        if first_chunk_sent_to_tts_t is not None:
                            chunk_to_tts_start_now = first_tts_synth_start_t - first_chunk_sent_to_tts_t

                            print(f"\n[TTS FIRST CHUNK]")
                            print(f"  chunk_to_tts_start : {chunk_to_tts_start_now:.3f} s")

                            

                    tts_t = Timer()
                    wav, sr = tts_engine.synthesize(
                        text=chunk_text,
                        voice_prompt=Path(args.voice_prompt) if args.voice_prompt else None,
                        config=None,
                        save_to=None,
                    )
                    tts_proc_total += tts_t.elapsed()
                    wav_np = torch_to_np_tc(wav)

                    if getattr(args, "browser_audio", False):
                        # Web UI mode:
                        # Do not open Python sounddevice/AudioOutput.
                        # Send synthesized audio directly to the browser.
                        enqueue_audio_chunk(wav_np, sr, chunk_text=chunk_text)
                        text_chunk_queue.task_done()
                        continue

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

                    smoothed = crossfader.process(wav_np)
                    enqueue_audio_chunk(smoothed, sr, chunk_text=chunk_text)
                    text_chunk_queue.task_done()
            tts_thread = threading.Thread(target=tts_worker, daemon=True)
            tts_thread.start()
            for ev in text_event_iter():
                if playback.interrupted.is_set():
                    break
                if ev.text:
                    if first_chunk_sent_to_tts_t is None:
                        first_chunk_sent_to_tts_t = perf_counter()
                    text_chunk_queue.put(ev.text.strip())
            text_chunk_queue.put(None)
            tts_thread.join()

            if audio_out is not None and args.speak and (not playback_started) and audio_prebuffer:
                playback.mark_tts_started()
                for buffered_np, buffered_text, _dur in audio_prebuffer:
                    on_start, on_end = make_chunk_callbacks(buffered_text)
                    audio_out.enqueue(buffered_np, on_start=on_start, on_end=on_end)
                audio_prebuffer.clear()
                if first_audio_t is None:
                    first_audio_t = perf_counter()
                playback_started = True
                
            if crossfader is not None and audio_out is not None and args.speak:
                tail = crossfader.flush()
                if len(tail) > 0:
                    audio_out.enqueue(tail)

        if llm_request_start_t is not None and llm_stream_done_t is not None:
            llm_time = llm_stream_done_t - llm_request_start_t
        else:
            llm_time = 0.0

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
        if not dots_stopped:
            dots_stop_event.set()
            dots_thread.join(timeout=0.2)

        # Tell the browser what happened to the audio stream.
        # audio_stop  = interruption/manual stop; browser should clear queued audio.
        # audio_done  = normal completion; browser can finish/clear playback state.
        if getattr(args, "browser_audio", False):
            if playback.interrupted.is_set():
                emit({"type": "audio_stop"})
            else:
                emit({"type": "audio_done"})

        playback.mark_tts_stopped()
        playback.clear_audio_out()
        playback.clear_chunk_tracking()
        pipeline_active.clear()
        set_tts_state(False, tts_busy, drain_after_tts)

    total_time = total_timer.elapsed()

    llm_ttft_real = None
    if llm_request_start_t is not None and llm_first_text_t is not None:
        llm_ttft_real = llm_first_text_t - llm_request_start_t

    ttfa = (first_audio_t - total_timer.t0) if first_audio_t is not None else None

    chunk_to_tts_start = None
    if first_chunk_sent_to_tts_t is not None and first_tts_synth_start_t is not None:
        chunk_to_tts_start = first_tts_synth_start_t - first_chunk_sent_to_tts_t

    tts_start_latency = None
    if first_chunk_sent_to_tts_t is not None and tts_play_start_t is not None:
        tts_start_latency = tts_play_start_t - first_chunk_sent_to_tts_t

    tts_actual_start_latency = None
    if first_chunk_sent_to_tts_t is not None and first_tts_actual_playback_t is not None:
        tts_actual_start_latency = first_tts_actual_playback_t - first_chunk_sent_to_tts_t

    emit({
        "type": "metrics",
        "data": {
            "stt": stt_time,
            "ttft": llm_ttft_real,
            "ttsFirstChunk": tts_actual_start_latency,

            "llmTotal": llm_time,
            "ttsTotal": tts_proc_total,
            "total": total_time,
            "firstAudio": ttfa,
        },
    })
    final_answer = "".join(full_answer_parts).strip()

    # Store completed turn after full streaming answer is generated
    if final_answer and getattr(args, "conversation_context", False) and conversation_memory is not None:
        conversation_memory.add_turn(
            user_text=raw_current_user_query,
            assistant_text=final_answer,
        )
        memory_stats_after = conversation_memory.stats()
        print("\n[CONVERSATION MEMORY UPDATED]")
        print(f"  stored turns      : {memory_stats_after['turns']}")
        print(f"  history chars     : {memory_stats_after['chars']}")
        print(f"  history words     : {memory_stats_after['approx_words']}")

    if final_answer and judge is not None:
        judge.judge_and_log_async(
            question=transcript,
            answer=final_answer,
            teaching_style=args.teaching_style,
            context=context,
            retrieved_ids=retrieved_ids,
            interruption_text=interruption_text,
            interrupted_chunks=interrupted_chunks or [],
            metadata={
                "mode": args.mode,
                "rag_enabled": retriever is not None,
                "is_interrupt": interruption_text is not None,
                "tts_backend": getattr(args, "tts_backend", None) or TTS_BACKEND,
            },
        )
    


    print("\n[STREAM COUNTS]")
    print(f"  text chunks                    : {chunks_count}")
    print(f"  audio chunks                   : {audio_chunk_count}")

    print("\n[LLM METRICS]")
    print(
        f"  LLM TTFT (prompt -> first token) : {llm_ttft_real:.3f} s"
        if llm_ttft_real is not None else
        "  LLM TTFT (prompt -> first token) : N/A"
    )
    print(f"  LLM full completion time         : {llm_time:.3f} s")

    print("\n[TTS / PLAYBACK METRICS]")
    print(
        f"  Chunk -> TTS synth start        : {chunk_to_tts_start:.3f} s"
        if chunk_to_tts_start is not None else
        "  Chunk -> TTS synth start        : N/A"
    )
    print(
        f"  ttsFirstChunk / actual audio start : {tts_actual_start_latency:.3f} s"
        if tts_actual_start_latency is not None else
        "  ttsFirstChunk / actual audio start : N/A"
    )
    print(
        f"  Chunk -> playback start         : {tts_start_latency:.3f} s"
        if tts_start_latency is not None else
        "  Chunk -> playback start         : N/A"
    )
    print(
        f"  First audio from pipeline       : {ttfa:.3f} s"
        if ttfa is not None else
        "  First audio from pipeline       : N/A"
    )
    
    print("\n[PIPELINE TOTALS]")
    print(f"  RAG latency                     : {rag_time:.3f} s")
    print(f"  TTS processing total            : {tts_proc_total:.3f} s")
    print(f"  End-to-end total                : {total_time:.3f} s")

# ---------------- Main ----------------
def main():
    p = argparse.ArgumentParser("Mic/File → STT → LLM → TTS (batch, chunked, or local HTTP streaming)")

    # Input mode
    #p.add_argument("--input", default=None, help="Optional audio file path. If set, runs file mode; otherwise mic mode.")
        # Input mode
    p.add_argument(
        "--input",
        default=None,
        help="Optional audio file path. If set, runs file mode; otherwise mic mode."
    )
    p.add_argument(
        "--text",
        default=None,
        help="Optional typed text query. If set, skips STT and runs text -> (RAG) -> LLM -> TTS."
    )
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
    p.add_argument("--llm_backend", choices=["local", "remote"], default="local")
    p.add_argument("--llm_server_url", default=None)
    p.add_argument("--llm_connect_timeout", type=float, default=3.0)
    p.add_argument("--llm_read_timeout", type=float, default=120.0)

    # Modes
    p.add_argument("--mode", choices=["batch", "stream-chunked", "stream-input"], default="batch")
    #p.add_argument("--tts_backend", choices=["chatterbox", "chatterbox_stream"], default=None)
    p.add_argument(
        "--tts_backend",
        type=str,
        default="chatterbox",
        choices=["chatterbox", "chatterbox_stream", "kokoro", "voxtral"],
        help="TTS backend to use",
    )
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
    #p.add_argument("--chunk_first", type=int, default=110)
    #p.add_argument("--chunk_mid", type=int, default=180)
    #p.add_argument("--chunk_later", type=int, default=220)
    #p.add_argument("--chunk_wait", type=float, default=0.35)
    #p.add_argument("--crossfade_ms", type=float, default=80.0)
    #p.add_argument("--chunk_first", type=int, default=120)
    #p.add_argument("--chunk_mid", type=int, default=320)
    #p.add_argument("--chunk_later", type=int, default=480)
    #p.add_argument("--chunk_wait", type=float, default=0.8)
    #p.add_argument("--crossfade_ms", type=float, default=120.0)
    p.add_argument("--chunk_first", type=int, default=200)
    p.add_argument("--chunk_mid", type=int, default=250)
    p.add_argument("--chunk_later", type=int, default=300)
    p.add_argument("--chunk_wait", type=float, default=0.5)
    p.add_argument("--crossfade_ms", type=float, default=80.0)
    # Teaching style prompt
    p.add_argument(
        "--teaching_style",
        type=str,
        default="socratic",
        choices=["socratic", "adaptive", "inquiry", "collaborative"],
        help="Teaching style prompt injected into the LLM prompt",
    )



    p.add_argument("--judge_model", default=None, help="Optional separate model for LLM-as-judge")
    # Conversation context experiment
    p.add_argument(
        "--conversation_context",
        action="store_true",
        help="Enable growing conversation history. Every new LLM prompt includes previous user/assistant turns.",
    )
    p.add_argument(
        "--conversation_max_turns",
        type=int,
        default=0,
        help="Maximum previous turns to keep. Use 0 for unlimited.",
    )
    p.add_argument(
        "--conversation_max_chars",
        type=int,
        default=0,
        help="Maximum history characters to keep. Use 0 for unlimited.",
    )
    p.add_argument(
        "--conversation_print_context",
        action="store_true",
        help="Print the full conversation history passed to the LLM.",
    )
    args = p.parse_args()


    conversation_memory = ConversationMemory(
        max_turns=None if args.conversation_max_turns == 0 else args.conversation_max_turns,
        max_chars=None if args.conversation_max_chars == 0 else args.conversation_max_chars,
    )


    input_modes_used = sum([
        1 if args.input else 0,
        1 if args.text else 0,
    ])
    if input_modes_used > 1:
        p.error("Use only one input source at a time: --text OR --input OR mic mode.")
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
    #llm = build_llm(args.llm_model)
    #judge_model_key = args.judge_model if args.judge_model else args.llm_model
    #judge_llm = build_judge_llm(judge_model_key)
    #judge = LocalLLMJudge(judge_llm)
        # ---- LLM ----
    llm = build_llm(args)

    judge = None
    if args.judge_model:
        judge_llm = build_judge_llm(args.judge_model)
        judge = LocalLLMJudge(judge_llm)

    # ---- TTS engine ----
    
    def build_tts_engine() -> BaseTTSEngine:
        backend = (getattr(args, "tts_backend", None) or TTS_BACKEND or "chatterbox").strip().lower()

        if backend == "chatterbox":
            return ChatterboxEngine()

        if backend in {"chatterbox_stream", "stream", "http_stream"}:
            return ChatterboxHTTPStreamingEngine()

        if backend == "kokoro":
            return KokoroEngine(
                KokoroConfig(
                    voice=KOKORO_VOICE,
                    lang_code=KOKORO_LANG_CODE,
                    speed=KOKORO_SPEED,
                    sample_rate=KOKORO_SAMPLE_RATE,
                    stream_audio_chunk_ms=KOKORO_STREAM_AUDIO_CHUNK_MS,
                )
            )

        if backend == "voxtral":
            return VoxtralServerEngine(
                VoxtralServerConfig(
                    base_url=VOXTRAL_BASE_URL,
                    endpoint=VOXTRAL_ENDPOINT,
                    model=VOXTRAL_MODEL,
                    voice=VOXTRAL_VOICE,
                    response_format=VOXTRAL_RESPONSE_FORMAT,
                    connect_timeout_s=VOXTRAL_CONNECT_TIMEOUT_S,
                    read_timeout_s=VOXTRAL_READ_TIMEOUT_S,
                    sample_rate_hint=VOXTRAL_SAMPLE_RATE_HINT,
                )
            )

        raise ValueError(f"Unsupported TTS backend: {backend}")

    tts_engine = build_tts_engine()
    tts_engine.load()

    # ---- RAG ----
    retriever = RagRetriever(args.rag_config) if args.rag else None

    playback = PlaybackControl()
    pipeline_active = threading.Event()
    # =========================
    # TEXT MODE
    # =========================
    if args.text:
        print("[INFO] Running TEXT mode")
        print(f"[TEXT INPUT] {args.text}")

        total_timer = Timer()
        stt_time = 0.0
        transcript = (args.text or "").strip()

        if not transcript:
            print("[INFO] Empty text input.")
            return

        pipeline_active.set()
        run_llm_tts_pipeline(
            transcript=transcript,
            stt_time=stt_time,
            total_timer=total_timer,
            llm=llm,
            tts_engine=tts_engine,
            retriever=retriever,
            args=args,
            playback=playback,
            pipeline_active=pipeline_active,
            tts_busy=None,
            drain_after_tts=None,
            judge=judge,
            conversation_memory=conversation_memory,
        )
        return
    # =========================
    # FILE MODE
    # =========================
    if args.input:
        wav_path = Path(args.input)
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")

        print(f"[INFO] Running FILE mode on: {wav_path}")

        total_timer = Timer()

        load_timer = Timer()
        audio, sr = load_audio_mono_16k(str(wav_path))
        audio_load_time = load_timer.elapsed()

        stt_timer = Timer()
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
        print(f"[Audio Load Latency] {audio_load_time:.3f}s")
        print(f"[STT Full Transcription Latency] {stt_time:.3f}s")
        print("[STT TTFT] N/A with current blocking transcribe() backend")
        pipeline_active.set()
        run_llm_tts_pipeline(
            transcript=transcript,
            stt_time=stt_time,
            total_timer=total_timer,
            llm=llm,
            tts_engine=tts_engine,
            retriever=retriever,
            args=args,
            playback=playback,
            pipeline_active=pipeline_active,
            tts_busy=None,
            drain_after_tts=None,
            judge=judge,
            conversation_memory=conversation_memory,
        )
        return

    # =========================
    # MIC MODE
    # =========================
    q_frames: "queue.Queue[tuple[bytes, float]]" = queue.Queue()
    segmenter = VADSegmenter(aggressiveness=args.vad_aggr)

    # separate segmenter only for interruption capture
    interrupt_segmenter = VADSegmenter(aggressiveness=args.vad_aggr)

    # Half-duplex control for mic mode only
    tts_busy = threading.Event()
    drain_after_tts = threading.Event()

    def worker():
        interrupt_capture_mode = False

        while True:
            try:
                frame_bytes, ts = q_frames.get(timeout=0.1)
            except queue.Empty:
                continue

            # -------- interruption capture mode --------
            if interrupt_capture_mode:
                utt_int = interrupt_segmenter.process_frame(frame_bytes, ts)
                if utt_int is None:
                    continue

                audio_np = utt_int.audio.astype(np.float32) / 32768.0

                interrupt_ready_t = perf_counter()

                stt_timer = Timer()
                stt_res = stt_backend.transcribe(
                    audio=audio_np,
                    sample_rate=TARGET_SR,
                    language=args.language,
                    return_words=False,
                )
                stt_time = stt_timer.elapsed()
                stt_done_t = perf_counter()
                interruption_stt_pipeline_latency = stt_done_t - interrupt_ready_t

                interrupt_text = (stt_res.text or "").strip()
                last_chunks = playback.get_interrupt_snapshot()

                print(f"\n[INTERRUPTION STT] {interrupt_text}")
                print(f"[INTERRUPTION STT Full Transcription Latency] {stt_time:.3f}s")
                print("[INTERRUPTION STT TTFT] N/A with current blocking transcribe() backend")

                if not is_valid_barge_in_text(interrupt_text, last_chunks):
                    interrupt_segmenter.reset()
                    interrupt_capture_mode = False
                    continue

                interrupt_segmenter.reset()
                interrupt_capture_mode = False

                # wait for previous pipeline to fully stop
                for _ in range(120):
                    if not pipeline_active.is_set():
                        break
                    time.sleep(0.05)
                total_timer = Timer()
                pipeline_active.set()

                def interrupted_pipeline_job():
                    try:
                        run_llm_tts_pipeline(
                        transcript=interrupt_text,
                        stt_time=stt_time,
                        total_timer=total_timer,
                        llm=llm,
                        tts_engine=tts_engine,
                        retriever=retriever,
                        args=args,
                        playback=playback,
                        pipeline_active=pipeline_active,
                        tts_busy=tts_busy,
                        drain_after_tts=drain_after_tts,
                        interruption_text=interrupt_text,
                        interrupted_chunks=last_chunks,
                        judge=judge,
                        conversation_memory=conversation_memory,
                    )
                    finally:
                        pipeline_active.clear()

                threading.Thread(target=interrupted_pipeline_job, daemon=True).start()
                continue

            # -------- while TTS is active: interrupt and then capture --------
            # -------- while TTS is active: strict built-in barge-in policy --------
            if tts_busy.is_set():
                # IMPORTANT:
                # Raw microphone RMS is not allowed to stop TTS.
                # Otherwise the assistant's own speaker audio can interrupt itself.
                #
                # In the UI path, validated user interruption comes from the frontend
                # as final recognized text and is filtered against recently spoken chunks.
                #
                # For true backend-native full duplex, this block must receive
                # echo-cancelled frames from a real AEC frontend, not raw mic frames.
                if not ALLOW_RAW_BACKEND_RMS_BARGE_IN:
                    continue

                # This fallback stays disabled by default.
                # Only use it for internal experiments if you later add real AEC.
                rms = frame_rms(frame_bytes)
                threshold = 650.0
                min_frames = max(1, int(600 / 20))

                if playback.seconds_since_tts_start() < 0.8:
                    continue

                if not hasattr(worker, "_interrupt_frames"):
                    worker._interrupt_frames = 0

                if rms >= threshold:
                    worker._interrupt_frames += 1
                else:
                    worker._interrupt_frames = 0

                if worker._interrupt_frames >= min_frames:
                    print("[BARGE-IN] Candidate detected from backend mic.")
                    playback.stop_now()
                    segmenter.reset()
                    interrupt_segmenter.reset()
                    drain_after_tts.set()
                    interrupt_capture_mode = True
                    worker._interrupt_frames = 0

                continue

            # block new user turns while old pipeline is still wrapping up
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
            utterance_ready_t = perf_counter()

            stt_timer = Timer()
            stt_res = stt_backend.transcribe(
                audio=audio,
                sample_rate=TARGET_SR,
                language=args.language,
                return_words=False,
            )
            stt_time = stt_timer.elapsed()
            stt_done_t = perf_counter()
            stt_pipeline_latency = stt_done_t - utterance_ready_t

            transcript = (stt_res.text or "").strip()

            print(f"[STT Full Transcription Latency] {stt_time:.3f}s")
            print("[STT TTFT] N/A with current blocking transcribe() backend")

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
                        pipeline_active=pipeline_active,
                        tts_busy=tts_busy,
                        drain_after_tts=drain_after_tts,
                        judge=judge,
                        conversation_memory=conversation_memory,
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