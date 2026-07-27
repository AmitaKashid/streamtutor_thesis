


from stt import vad_segmenter
from stt import stt_backends
import asyncio
import json
import queue
import threading
import time
import re
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware


from pathlib import Path

from judge.judge import LocalLLMJudge
from llm.llmcli.providers.remote_http import RemoteHTTPProvider


from app.mic_stt_llm_tts import (
    Timer,
    PlaybackControl,
    ConversationMemory,
    build_llm,
    run_llm_tts_pipeline,
    frame_rms,
    drain_queue,
    effective_config_payload,
    is_valid_barge_in_text,
)
from rag.rag_retrieve import RagRetriever
from stt.stt_backends import create_backend
from stt.vad_segmenter import VADSegmenter, TARGET_SR
from tts.base_engine import BaseTTSEngine
from tts.chatterbox_engine import ChatterboxEngine
from tts.config import (
    TTS_BACKEND,
    KOKORO_LANG_CODE,
    KOKORO_SAMPLE_RATE,
    KOKORO_SPEED,
    KOKORO_STREAM_AUDIO_CHUNK_MS,
    KOKORO_VOICE,
    QWEN3_TTS_MODEL,
    QWEN3_TTS_SPEAKER,
    QWEN3_TTS_LANGUAGE,
    QWEN3_TTS_SPEED,
    QWEN3_TTS_SAMPLE_RATE,
    QWEN3_TTS_STREAM_AUDIO_CHUNK_MS,
)
from tts.elevenlabs_streaming_engine import ChatterboxHTTPStreamingEngine
from tts.kokoro_engine import KokoroConfig, KokoroEngine
#from tts.qwen3tts_engine import Qwen3TTSConfig, Qwen3TTSEngine, QWEN3_TTS_SPEAKERS, QWEN3_TTS_LANGUAGES
from tts.qwen3tts_engine import QWEN3_TTS_SPEAKERS, QWEN3_TTS_LANGUAGES
from tts.qwen3_streaming_engine import Qwen3StreamingConfig, Qwen3StreamingEngine

app = FastAPI(title="LLMTutor Frontend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# CONFIG
# -------------------------
DEFAULT_LLM_MODEL = "llama3_8b_q4"

REMOTE_MODEL_URLS = {
    "llama3_8b_q4": "http://127.0.0.1:8000",

    # temporary alias if frontend still sends q5
    # This still routes to the q4 server unless you actually start a q5 model.
    "llama3_8b_q5": "http://127.0.0.1:8000",

    # match your cluster script model key
    "gemma4_26b_q3": "http://127.0.0.1:8002",

    "qwen3_5_9b_q6": "http://127.0.0.1:8003",
}


PROJECT_ROOT = Path(__file__).resolve().parent

ENABLE_RUNTIME_EVALUATION = True

JUDGE_SERVER_URL = "http://127.0.0.1:8004"

JUDGE_REFERENCE_PATH = (
    PROJECT_ROOT / "judge" / "test_samples.json"
)

JUDGE_OUTPUT_PATH = (
    PROJECT_ROOT / "judge_outputs" / "runtime_results.jsonl"
)

TTS_BACKEND_NAME = "kokoro"  # or "chatterbox"
ENABLE_RAG = True
RAG_CONFIG_PATH = "configs/retrieval.yaml"
RAG_TOP_K = 2
TEACHING_STYLE = "socratic"

STT_BACKEND_NAME = "faster-whisper"

STT_MODEL = "medium"
STT_DEVICE = "cuda"
STT_COMPUTE_TYPE = "float16"
STT_BEAM_SIZE = 5
STT_LANGUAGE = "en"

MEGA_ASR_BASE_URL = "http://127.0.0.1:8095"
MEGA_ASR_TIMEOUT_S = 120.0

PARAKEET_BASE_URL = "http://127.0.0.1:8096"
PARAKEET_TIMEOUT_S = 120.0

# Live partial transcription
PARTIAL_TRANSCRIBE_EVERY_S = 0.45
PARTIAL_MIN_AUDIO_S = 0.35

# Clears a preview accidentally started by a short noise spike.
PARTIAL_SILENCE_RESET_S = 1.0

# Normal listening threshold
INTERRUPT_RMS_THRESHOLD = 180.0
INTERRUPT_MIN_FRAMES = 1

# Barge-in detector while assistant/browser audio is active.
# The rolling probe is only used to decide whether to stop assistant audio.
# It must NOT be treated as the final user message.
BARGE_IN_GRACE_S = 0.35
BARGE_IN_PROBE_EVERY_S = 0.45
BARGE_IN_PROBE_MIN_AUDIO_S = 0.60
BARGE_IN_PROBE_MAX_FRAMES = 80  # 80 x 20 ms = 1.6 s rolling window
BARGE_IN_PROBE_MIN_RMS_FLOAT = 0.008

# After barge-in is accepted, capture the real user utterance until silence.
INTERRUPT_CAPTURE_MIN_S = 1.0
INTERRUPT_CAPTURE_SILENCE_S = 0.85
INTERRUPT_CAPTURE_MAX_S = 8.0
INTERRUPT_CAPTURE_SILENCE_RMS = 140.0

# -------------------------
# Shared cached components
# -------------------------
_llm = None
_tts_engine: BaseTTSEngine | None = None
_retriever = None
_stt_backend = None
_stt_backend_cache = {}
_init_lock = threading.RLock()

_llm_cache = {}
_tts_engine_cache: dict[str, BaseTTSEngine] = {}


def build_runtime_evaluator() -> LocalLLMJudge | None:
    if not ENABLE_RUNTIME_EVALUATION:
        print("[JUDGE] Runtime evaluation disabled")
        return None

    judge_provider = RemoteHTTPProvider(
        base_url=JUDGE_SERVER_URL,
        connect_timeout=5.0,
        read_timeout=180.0,
    )

    if not judge_provider.healthcheck():
        raise RuntimeError(
            f"Runtime judge is enabled but is not reachable at {JUDGE_SERVER_URL}. "
            "Start the judge server and its SSH tunnel before frontend_api."
        )

    evaluator = LocalLLMJudge(
        judge_provider,
        output_path=JUDGE_OUTPUT_PATH,
    )

    print("[JUDGE] Runtime evaluator initialized")
    print(f"[JUDGE] Server: {JUDGE_SERVER_URL}")
    print(f"[JUDGE] Output: {JUDGE_OUTPUT_PATH}")

    return evaluator


_runtime_evaluator = build_runtime_evaluator()

@app.on_event("shutdown")
def shutdown_runtime_evaluator():
    if _runtime_evaluator is not None:
        print("[JUDGE] Flushing pending evaluations...")
        _runtime_evaluator.shutdown(
            wait=True,
            timeout=30.0,
        )

def build_tts_engine(backend: str) -> BaseTTSEngine:
    backend = (backend or TTS_BACKEND or "chatterbox").strip().lower()

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

    if backend == "qwen3":
        return Qwen3StreamingEngine(
            Qwen3StreamingConfig(
                health_url="http://127.0.0.1:8094/health",
                stream_url="ws://127.0.0.1:8094/stream",
                speaker=QWEN3_TTS_SPEAKER,
                language=QWEN3_TTS_LANGUAGE,
                timeout_s=180.0,
                chunk_size=4,
            )
        )

    raise ValueError(f"Unsupported TTS backend: {backend}")

def build_stt_backend(backend: str):
    backend = (backend or STT_BACKEND_NAME).strip().lower()

    if backend in {"faster-whisper", "faster_whisper", "whisper"}:
        return create_backend(
            "faster-whisper",
            {
                "model": STT_MODEL,
                "device": STT_DEVICE,
                "device_index": 0,
                "compute_type": STT_COMPUTE_TYPE,
                "beam_size": STT_BEAM_SIZE,
                "diag_dll": False,
            },
        )

    if backend in {"mega-asr", "mega_asr", "mega-asr-http"}:
        return create_backend(
            "mega-asr-http",
            {
                "base_url": MEGA_ASR_BASE_URL,
                "timeout_s": MEGA_ASR_TIMEOUT_S,
            },
        )

    if backend in {"parakeet", "parakeet-http", "parakeet_http"}:
        return create_backend(
            "parakeet-http",
            {
                "base_url": PARAKEET_BASE_URL,
                "timeout_s": PARAKEET_TIMEOUT_S,
                "include_timestamps": False,
                "should_chunk": False,
            },
        )

    raise ValueError(f"Unsupported STT backend: {backend}")



def get_stt_backend_for_backend(backend: str):
    backend = (backend or STT_BACKEND_NAME).strip().lower()

    with _init_lock:
        if backend not in _stt_backend_cache:
            print(f"\n[STT CONFIG] Loading STT backend: {backend}")
            _stt_backend_cache[backend] = build_stt_backend(backend)
        else:
            print(f"\n[STT CONFIG] Reusing STT backend: {backend}")

        engine = _stt_backend_cache[backend]
        print(f"[STT CONFIG] Active backend class: {engine.__class__.__name__}")
        return engine


def build_llm_args(model_name: str | None = None):
    model_name = model_name or DEFAULT_LLM_MODEL

    if model_name not in REMOTE_MODEL_URLS:
        print(
            f"[LLM CONFIG WARNING] Unknown model '{model_name}'. "
            f"Falling back to {DEFAULT_LLM_MODEL}."
        )
        model_name = DEFAULT_LLM_MODEL

    return SimpleNamespace(
        llm_backend="remote",
        llm_model=model_name,
        llm_server_url=REMOTE_MODEL_URLS[model_name],
        llm_connect_timeout=5.0,
        llm_read_timeout=120.0,
    )

def get_llm_for_model(model_name: str):
    """
    Return a cached RemoteHTTPProvider for the selected model.

    This does not load model weights locally.
    It only selects the correct remote server endpoint.
    """
    llm_args = build_llm_args(model_name)

    cache_key = (
        llm_args.llm_backend,
        llm_args.llm_model,
        llm_args.llm_server_url,
    )

    with _init_lock:
        if cache_key not in _llm_cache:
            print("\n[LLM CONFIG] Creating remote LLM provider")
            print(f"  model      : {llm_args.llm_model}")
            print(f"  server_url : {llm_args.llm_server_url}")

            provider = build_llm(llm_args)
            _llm_cache[cache_key] = provider
        else:
            print("\n[LLM CONFIG] Reusing remote LLM provider")
            print(f"  model      : {llm_args.llm_model}")
            print(f"  server_url : {llm_args.llm_server_url}")

        return _llm_cache[cache_key]

def safe_int(value, default: int, min_value: int = 0, max_value: int | None = None) -> int:
    try:
        out = int(value)
    except Exception:
        out = default

    out = max(min_value, out)

    if max_value is not None:
        out = min(out, max_value)

    return out





def build_pipeline_args(config: dict | None = None):
    config = config or {}

    selected_model = config.get("llm_model", DEFAULT_LLM_MODEL)
    if selected_model not in REMOTE_MODEL_URLS:
        print(
            f"[LLM CONFIG WARNING] Unknown selected model '{selected_model}'. "
            f"Using default model {DEFAULT_LLM_MODEL}."
        )
        selected_model = DEFAULT_LLM_MODEL

    rag_top_k = safe_int(
        config.get("rag_top_k", RAG_TOP_K),
        default=RAG_TOP_K,
        min_value=1,
        max_value=10,
    )

    conversation_max_turns = safe_int(
        config.get("conversation_max_turns", 0),
        default=0,
        min_value=0,
        max_value=50,
    )

    teaching_style = config.get("teaching_style", TEACHING_STYLE)
    if teaching_style not in {"socratic", "adaptive", "inquiry", "collaborative"}:
        teaching_style = TEACHING_STYLE

    tts_backend = config.get("tts_backend", TTS_BACKEND_NAME)
    if tts_backend not in {"kokoro", "chatterbox", "chatterbox_stream", "qwen3"}:
        tts_backend = TTS_BACKEND_NAME

    stt_backend = config.get("stt_backend", STT_BACKEND_NAME)
    stt_backend = (stt_backend or STT_BACKEND_NAME).strip().lower()

    if stt_backend not in {"faster-whisper", "mega-asr", "parakeet"}:
        stt_backend = STT_BACKEND_NAME

    # Qwen3-TTS speaker and language (only used when backend=="qwen3")
    qwen3_speaker = config.get("qwen3_speaker", QWEN3_TTS_SPEAKER)
    if qwen3_speaker not in QWEN3_TTS_SPEAKERS:
        qwen3_speaker = QWEN3_TTS_SPEAKER

    qwen3_language = config.get("qwen3_language", QWEN3_TTS_LANGUAGE)
    if qwen3_language not in QWEN3_TTS_LANGUAGES:
        qwen3_language = QWEN3_TTS_LANGUAGE

    if tts_backend == "qwen3":
        pipeline_mode = "stream-input"
        chunk_first = 120
        chunk_mid = 180
        chunk_later = 220
        chunk_wait = 0.30
    else:
        pipeline_mode = "stream-chunked"
        chunk_first = 200
        chunk_mid = 250
        chunk_later = 300
        chunk_wait = 0.5

    return SimpleNamespace(
        mode=pipeline_mode,
        min_words=1,
        # STT
        stt_backend=stt_backend,
        # LLM routing
        llm_backend="remote",
        llm_model=selected_model,
        llm_server_url=REMOTE_MODEL_URLS[selected_model],

        # RAG
        rag=bool(config.get("rag", ENABLE_RAG)),
        rag_top_k=rag_top_k,
        rag_print_context=False,
        rag_context_chars=0,

        # Prompt
        teaching_style=teaching_style,

        # TTS
        voice_prompt=None,
        save=False,

        # IMPORTANT:
        # In the web UI, Python must not play TTS through sounddevice.
        # The backend should synthesize audio and send it to the browser.
        speak=False,
        browser_audio=True,
        tts_backend=tts_backend,
        qwen3_speaker=qwen3_speaker,
        qwen3_language=qwen3_language,

        # Streaming
        tail_settle=0.3,
        chunk_first=chunk_first,
        chunk_mid=chunk_mid,
        chunk_later=chunk_later,
        chunk_wait=chunk_wait,
        crossfade_ms=80.0,

        # Interruption
        speaker_mode=False,
        speaker_ignore_after_tts_start=0.6,
        speaker_min_interrupt_ms=350,

        # Conversation context
        conversation_context=bool(config.get("conversation_context", True)),
        conversation_max_turns=conversation_max_turns,
        conversation_max_chars=0,
        conversation_print_context=False,
    )


def ensure_components():
    global _llm, _tts_engine, _retriever, _stt_backend

    with _init_lock:
        if _llm is None:
            #default_llm_args = build_llm_args(DEFAULT_LLM_MODEL)
            #_llm = build_llm(default_llm_args)
            #_llm_cache[
            #    (
            #        default_llm_args.llm_backend,
            #        default_llm_args.llm_model,
            #        default_llm_args.llm_server_url,
            #    )
            #] = _llm
            _llm = None

        if _tts_engine is None:
            backend = TTS_BACKEND_NAME.strip().lower()
            if backend not in _tts_engine_cache:
                engine = build_tts_engine(backend)
                engine.load()
                _tts_engine_cache[backend] = engine
            _tts_engine = _tts_engine_cache[backend]

        if _retriever is None and ENABLE_RAG:
            _retriever = RagRetriever(RAG_CONFIG_PATH)

        if _stt_backend is None:
            _stt_backend = get_stt_backend_for_backend(STT_BACKEND_NAME)

    return _llm, _tts_engine, _retriever, _stt_backend

def get_tts_engine_for_backend(backend: str) -> BaseTTSEngine:
    backend = (backend or TTS_BACKEND_NAME).strip().lower()

    with _init_lock:
        if backend not in _tts_engine_cache:
            print(f"\n[TTS CONFIG] Loading TTS backend: {backend}")
            engine = build_tts_engine(backend)
            engine.load()
            _tts_engine_cache[backend] = engine
        else:
            print(f"\n[TTS CONFIG] Reusing TTS backend: {backend}")

        engine = _tts_engine_cache[backend]
        print(f"[TTS CONFIG] Active engine class: {engine.__class__.__name__}")
        return engine
    

    

def pcm16_bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
    if not audio_bytes:
        return np.zeros(0, dtype=np.float32)
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)

def audio_rms_float(audio_np: np.ndarray) -> float:
    if audio_np.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio_np * audio_np) + 1e-12))

STT_HALLUCINATION_EXACT = {
    "thank you",
    "thank you very much",
    "thank you so much",
    "thanks",
    "thanks a lot",
    "thanks very much",
    "thanks for watching",
    "thank you for watching",
    "thanks for watching.",
    "thank you for watching.",
    "bye",
    "bye bye",
    "bye-bye",
    "goodbye",
    "see you next time",
    "that's all",
}

STT_HALLUCINATION_SUBSTRINGS = {
    "thanks for watching",
    "thank you for watching",
    "like and subscribe",
    "subscribe to my channel",
    "don't forget to subscribe",
    "please subscribe",
    "hit the bell",
    "turn on notifications",
    "see you in the next video",
    "subtitles by",
    "captioned by",
    "transcribed by",
    "music playing",
    "applause",
}


def normalize_stt_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .!?,")


def is_stt_hallucination_text(text: str) -> bool:
    """
    Filters common Whisper/faster-whisper hallucinations from silence,
    echo tails, and non-speech audio. This must be used before showing
    partial text and before launching the LLM.
    """
    norm = normalize_stt_text(text)

    if not norm:
        return True

    if norm in STT_HALLUCINATION_EXACT:
        return True

    for phrase in STT_HALLUCINATION_SUBSTRINGS:
        if phrase in norm:
            return True

    words = norm.split()

    # Repeated hallucination loops, e.g. "thank you thank you"
    # and short generic politeness/closing phrases from browser/Whisper noise.
    thank_words = {"thank", "you", "thanks", "very", "much", "so"}
    if words and all(w in thank_words for w in words) and any(w in {"thank", "thanks"} for w in words):
        return True

    if len(words) >= 4:
        unique_ratio = len(set(words)) / max(1, len(words))
        if unique_ratio <= 0.50:
            return True

    # Very short generic closings are usually not real tutor questions.
    if len(words) <= 5 and norm in {
        "thank you",
        "thank you very much",
        "thank you so much",
        "thanks",
        "thanks very much",
        "bye",
        "bye bye",
        "goodbye",
        "okay bye",
    }:
        return True

    return False



def is_valid_normal_user_text(text: str, audio_np: np.ndarray) -> bool:
    text = (text or "").strip()

    if not text:
        return False

    if is_stt_hallucination_text(text):
        print(f"[STT FILTER] Rejected hallucination text: {text}")
        return False

    words = text.split()
    if len(words) < 2:
        return False

    if len(audio_np) < int(0.45 * TARGET_SR):
        return False

    if audio_rms_float(audio_np) < 0.008:
        return False

    return True

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/models")
def list_models():
    """
    Returns known model endpoints.
    'available' is checked through the remote provider health check indirectly
    when a model is selected, but this endpoint helps the frontend show model options.
    """
    return {
        "default": DEFAULT_LLM_MODEL,
        "models": [
            {
                "model": model_name,
                "url": url,
            }
            for model_name, url in REMOTE_MODEL_URLS.items()
        ],
    }
@app.get("/stt/models")
def list_stt_models():
    return {
        "default": STT_BACKEND_NAME,
        "models": [
            {
                "id": "faster-whisper",
                "label": "Faster-Whisper",
                "description": "Fast local Whisper backend",
            },
            {
                "id": "mega-asr",
                "label": "Mega-ASR",
                "description": "Robust Mega-ASR backend through local HTTP server",
            },
            {
                "id": "parakeet",
                "label": "Parakeet-TDT 0.6B v2",
                "description": "NVIDIA Parakeet-TDT 0.6B v2 backend through local HTTP server",
            },
        ],
    }
@app.get("/tts/speakers")
def list_tts_speakers():
    """
    Returns available Qwen3-TTS preset speakers and supported languages.
    Used by the frontend to populate speaker / language dropdowns.
    """
    return {
        "backend": "qwen3",
        "speakers": QWEN3_TTS_SPEAKERS,
        "languages": QWEN3_TTS_LANGUAGES,
        "default_speaker": QWEN3_TTS_SPEAKER,
        "default_language": QWEN3_TTS_LANGUAGE,
    }
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    llm, tts_engine, retriever, stt_backend = ensure_components()
    loop = asyncio.get_running_loop()

    q_frames: "queue.Queue[tuple[bytes, float]]" = queue.Queue()
    session_closed = threading.Event()
    mic_enabled = threading.Event()

    segmenter = VADSegmenter(aggressiveness=3)
    interrupt_segmenter = VADSegmenter(aggressiveness=3)

    playback = PlaybackControl()
    pipeline_active = threading.Event()
    pipeline_lock = threading.Lock()
    tts_busy = threading.Event()
    drain_after_tts = threading.Event()

    # Browser audio can still be playing after backend finished sending chunks.
    # Keep this active until frontend confirms playback has actually finished.
    browser_audio_active = threading.Event()

    active_config = {}
    args = build_pipeline_args(active_config)
    last_applied_tts_backend = None
    last_applied_llm_model = None
    last_final_user_text = ""
    last_final_user_at = 0.0
    conversation_memory = ConversationMemory(
        max_turns=None,
        max_chars=None,
    )

    interrupt_capture_mode = False
    interrupt_counter = 0
    ignore_audio_until = 0.0

    # Rolling audio buffer used only while assistant/browser audio is active.
    # We use this to run short STT probes and validate text before stopping audio.
    # Rolling audio buffer used only while assistant/browser audio is active.
    # We use this to run short STT probes and validate text before stopping audio.
    tts_probe_frames: list[bytes] = []
    tts_last_probe_t = 0.0

    # After a valid rolling probe stops browser audio, VAD may not always produce
    # a clean final utterance. Keep the validated probe text as a safe fallback.
    
    # After a valid rolling probe stops browser audio, keep capturing until the
    # user actually finishes speaking. The probe text is only a fallback.
    pending_barge_in_text = ""
    pending_barge_in_stt_time = 0.0
    interrupt_capture_started_at = 0.0
    interrupt_last_voice_at = 0.0

    partial_frames: list[bytes] = []
    partial_last_emit_t = 0.0
    partial_last_text = ""
    partial_last_voice_t = 0.0
    partial_generation = 0

    # Prevent multiple partial STT calls from accumulating.
    partial_inflight = threading.Event()

    # Protect partial transcript state shared by the audio and STT threads.
    partial_state_lock = threading.Lock()

    # The same backend is used for partial and final STT.
    stt_inference_lock = threading.Lock()

    # Partial STT must not block the microphone/VAD worker.
    partial_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="partial-stt",
    )

    async def ws_send(payload: dict):
        await websocket.send_text(json.dumps(payload))

    def emit_from_thread(payload: dict):
        msg_type = payload.get("type")

        if msg_type == "audio_chunk":
            browser_audio_active.set()

        if msg_type == "audio_stop":
            browser_audio_active.clear()

        asyncio.run_coroutine_threadsafe(ws_send(payload), loop)

    def clear_frontend_transcript():
        emit_from_thread({
            "type": "transcript_partial",
            "text": "",
        })
        emit_from_thread({
            "type": "captured_text",
            "text": "",
        })

    def reset_partial_state():
        nonlocal partial_frames, partial_last_emit_t, partial_last_text
        nonlocal partial_last_voice_t, partial_generation

        with partial_state_lock:
            partial_frames = []
            partial_last_emit_t = 0.0
            partial_last_text = ""
            partial_last_voice_t = 0.0

            # Any partial STT request already running becomes stale.
            # Its result will therefore not overwrite a completed turn.
            partial_generation += 1
    def reset_tts_probe_state():
        nonlocal tts_probe_frames, tts_last_probe_t
        tts_probe_frames = []
        tts_last_probe_t = 0.0
    def apply_config(config: dict):
        nonlocal args, conversation_memory, retriever, tts_engine, active_config, llm, stt_backend
        nonlocal last_applied_tts_backend, last_applied_llm_model

        active_config = config or {}
        args = build_pipeline_args(active_config)

        config_changed = (
            last_applied_tts_backend is not None
            and (
                args.tts_backend != last_applied_tts_backend
                or args.llm_model != last_applied_llm_model
            )
        )

        if config_changed:
            print(
                f"[CONFIG CHANGE] Clearing stale playback/filter state: "
                f"tts {last_applied_tts_backend} -> {args.tts_backend}, "
                f"llm {last_applied_llm_model} -> {args.llm_model}"
            )

            playback.stop_now()
            browser_audio_active.clear()
            tts_busy.clear()
            drain_after_tts.set()

            segmenter.reset()
            interrupt_segmenter.reset()
            reset_partial_state()
            reset_tts_probe_state()
            clear_frontend_transcript()

            # This is important: old assistant chunks from Kokoro should not be used
            # to reject the next user question after switching to Qwen3.
            try:
                playback.clear_chunk_tracking()
            except Exception as e:
                print(f"[CONFIG CHANGE] playback.clear_chunk_tracking failed: {e}")

            last_applied_tts_backend = args.tts_backend
            last_applied_llm_model = args.llm_model

        # LLM model from frontend.
        # This selects the remote server endpoint mapped to args.llm_model.
        #llm = get_llm_for_model(args.llm_model)

        try:
            llm = get_llm_for_model(args.llm_model)
        except Exception as e:
            print(f"[LLM CONFIG ERROR] {type(e).__name__}: {e}")
            llm = None
            emit_from_thread({
                "type": "error",
                "message": f"LLM server not reachable: {args.llm_server_url}",
            })

        
        # RAG on/off from frontend
        if args.rag:
            if retriever is None:
                print("\n[RAG CONFIG] Loading retriever because frontend enabled RAG")
                retriever = RagRetriever(RAG_CONFIG_PATH)
        else:
            print("\n[RAG CONFIG] Frontend disabled RAG")
            retriever = None

        # Conversation memory max turns from frontend
        conversation_memory.max_turns = (
            None if args.conversation_max_turns == 0 else args.conversation_max_turns
        )

        # TTS backend from frontend
        tts_engine = get_tts_engine_for_backend(args.tts_backend)
        stt_engine_selected = args.stt_backend
        stt_backend = get_stt_backend_for_backend(stt_engine_selected)
        # For Qwen3-TTS, update the engine's speaker/language on every config apply
        # so per-request speaker changes take effect without reloading the model.
        if args.tts_backend == "qwen3" and hasattr(tts_engine, "cfg"):
            tts_engine.cfg.speaker = args.qwen3_speaker
            tts_engine.cfg.language = args.qwen3_language

        print("\n[FRONTEND CONFIG APPLIED]")
        print(f"  stt_backend            : {args.stt_backend}")
        print(f"  llm_model              : {args.llm_model}")
        print(f"  llm_backend            : {args.llm_backend}")
        print(f"  llm_server_url         : {args.llm_server_url}")
        print(f"  tts_backend            : {args.tts_backend}")
        print(f"  rag                    : {args.rag}")
        print(f"  rag_top_k              : {args.rag_top_k}")
        print(f"  teaching_style         : {args.teaching_style}")
        print(f"  conversation_context   : {args.conversation_context}")
        print(f"  conversation_max_turns : {args.conversation_max_turns}")
        if args.tts_backend == "qwen3":
            print(f"  qwen3_speaker          : {args.qwen3_speaker}")
            print(f"  qwen3_language         : {args.qwen3_language}")

        payload = effective_config_payload(args)
        payload["stt_backend"] = args.stt_backend

        emit_from_thread({
            "type": "config_applied",
            "data": payload,
        })
    def run_partial_transcription(
        audio_np: np.ndarray,
        engine,
        generation: int,
    ):
        nonlocal partial_last_text

        try:
            # Some local and HTTP STT backends are not safe for simultaneous calls.
            with stt_inference_lock:
                res = engine.transcribe(
                    audio=audio_np,
                    sample_rate=TARGET_SR,
                    language=STT_LANGUAGE,
                    return_words=False,
                )

            text = (res.text or "").strip()

            with partial_state_lock:
                # Ignore results belonging to an earlier or completed utterance.
                if session_closed.is_set() or generation != partial_generation:
                    return

                if is_stt_hallucination_text(text):
                    print(f"[PARTIAL FILTER] Rejected hallucination partial: {text}")
                    return

                # Partial STT returns the complete current hypothesis.
                # Do not append it to the previous hypothesis.
                if not text or text == partial_last_text:
                    return

                partial_last_text = text

            emit_from_thread({
                "type": "transcript_partial",
                "text": text,
            })

        except Exception as e:
            print(f"[PARTIAL STT ERROR] {type(e).__name__}: {e}")

        finally:
            partial_inflight.clear()


    def maybe_emit_partial():
        nonlocal partial_last_emit_t

        # One preview request at a time. New audio continues entering partial_frames.
        if partial_inflight.is_set():
            return

        now = time.perf_counter()

        with partial_state_lock:
            if not partial_frames:
                return

            if now - partial_last_emit_t < PARTIAL_TRANSCRIBE_EVERY_S:
                return

            audio_bytes = b"".join(partial_frames)
            generation = partial_generation
            partial_last_emit_t = now

        audio_np = pcm16_bytes_to_float32(audio_bytes)

        if len(audio_np) < int(TARGET_SR * PARTIAL_MIN_AUDIO_S):
            return

        # Preserve the engine selected when this request was scheduled.
        engine = stt_backend

        partial_inflight.set()

        partial_executor.submit(
            run_partial_transcription,
            audio_np,
            engine,
            generation,
        )

    def transcribe_final(audio_np: np.ndarray) -> str:
        with stt_inference_lock:
            res = stt_backend.transcribe(
                audio=audio_np,
                sample_rate=TARGET_SR,
                language=STT_LANGUAGE,
                return_words=False,
            )

        return (res.text or "").strip()


    def transcribe_final_timed(audio_np: np.ndarray) -> tuple[str, float]:
        stt_timer = Timer()

        with stt_inference_lock:
            res = stt_backend.transcribe(
                audio=audio_np,
                sample_rate=TARGET_SR,
                language=STT_LANGUAGE,
                return_words=False,
            )

        stt_time = stt_timer.elapsed()
        return (res.text or "").strip(), stt_time

    def launch_pipeline(
        transcript: str,
        interruption_text: str | None = None,
        interrupted_chunks: list[str] | None = None,
        stt_time: float = 0.0,
    ):

        if llm is None:
            print("[PIPELINE BLOCKED] LLM is not available. STT worked, but answer generation is skipped.")
            emit_from_thread({
                "type": "error",
                "message": "Cannot answer yet: selected LLM server is not reachable.",
            })
            emit_from_thread({
                "type": "stage",
                "value": "listening" if mic_enabled.is_set() else "idle",
            })
            return
        
        # Freeze selected config/components for this specific request.
        request_args = args
        request_llm = llm
        request_tts_engine = tts_engine
        request_retriever = retriever
        request_judge = _runtime_evaluator

        def _job():
            # Only one pipeline may use the shared playback state at a time.
            with pipeline_lock:
                total_timer = Timer()
                pipeline_active.set()

                try:
                    run_llm_tts_pipeline(
                        transcript=transcript,
                        stt_time=stt_time,
                        total_timer=total_timer,
                        llm=request_llm,
                        tts_engine=request_tts_engine,
                        retriever=request_retriever,
                        args=request_args,
                        playback=playback,
                        pipeline_active=pipeline_active,
                        judge=request_judge,
                        conversation_memory=conversation_memory,
                        tts_busy=tts_busy,
                        drain_after_tts=drain_after_tts,
                        interruption_text=interruption_text,
                        interrupted_chunks=interrupted_chunks,
                        event_cb=emit_from_thread,
                    )

                except Exception as e:
                    emit_from_thread({
                        "type": "error",
                        "message": str(e),
                    })

                finally:
                    emit_from_thread({
                        "type": "metrics",
                        "data": {
                            "conversationTurns": conversation_memory.stats()["turns"],
                            "conversationChars": conversation_memory.stats()["chars"],
                        },
                    })

                    emit_from_thread({
                        "type": "assistant_done",
                    })

                    emit_from_thread({
                        "type": "stage",
                        "value": (
                            "listening"
                            if mic_enabled.is_set()
                            else "idle"
                        ),
                    })

                    pipeline_active.clear()

        threading.Thread(target=_job, daemon=True).start()
    
    def worker():
        nonlocal interrupt_capture_mode, interrupt_counter, ignore_audio_until
        nonlocal tts_probe_frames, tts_last_probe_t
        nonlocal pending_barge_in_text, pending_barge_in_stt_time
        nonlocal interrupt_capture_started_at, interrupt_last_voice_at
        nonlocal last_final_user_text, last_final_user_at
        nonlocal partial_last_voice_t

        while not session_closed.is_set():
            try:
                frame_bytes, ts = q_frames.get(timeout=0.1)
            except queue.Empty:
                continue
            
            try:
                if not mic_enabled.is_set():
                    continue

                rms = frame_rms(frame_bytes)

                if interrupt_capture_mode:
                    # Browser audio has already been stopped because a rolling STT probe
                    # detected valid user speech. Now capture the complete user utterance.
                    #
                    # Important:
                    # - Do not finalize after a fixed 1-second timeout.
                    # - Keep showing live partial transcription.
                    # - Finalize only after enough silence or a hard max duration.

                    now = time.monotonic()

                    partial_frames.append(frame_bytes)
                    emit_from_thread({"type": "stage", "value": "transcribing"})

                    # Live transcription while the user is still speaking.
                    maybe_emit_partial()

                    if rms >= INTERRUPT_CAPTURE_SILENCE_RMS:
                        interrupt_last_voice_at = now

                    capture_started = interrupt_capture_started_at or now
                    captured_s = now - capture_started
                    silence_s = now - interrupt_last_voice_at if interrupt_last_voice_at > 0 else 0.0

                    min_capture_reached = captured_s >= INTERRUPT_CAPTURE_MIN_S
                    silence_endpoint_reached = (
                        min_capture_reached
                        and interrupt_last_voice_at > 0
                        and silence_s >= INTERRUPT_CAPTURE_SILENCE_S
                    )
                    max_capture_reached = captured_s >= INTERRUPT_CAPTURE_MAX_S

                    if not silence_endpoint_reached and not max_capture_reached:
                        continue

                    audio_bytes = b"".join(partial_frames)
                    audio_np = pcm16_bytes_to_float32(audio_bytes)

                    transcript, stt_time = transcribe_final_timed(audio_np)
                    transcript = transcript.strip()
                    if is_stt_hallucination_text(transcript):
                        print(f"[INTERRUPTION FILTER] Rejected hallucination final: {transcript}")
                        transcript = ""
                        clear_frontend_transcript()
                    last_chunks = playback.get_interrupt_snapshot() or playback.get_interrupt_context(n_finished=3)

                    # Prefer the full captured utterance. Only fall back to the probe if
                    # final STT is empty or clearly shorter than the already validated probe.
                    pending_words = pending_barge_in_text.split()
                    transcript_words = transcript.split()

                    if (
                        pending_barge_in_text
                        and not is_stt_hallucination_text(pending_barge_in_text)
                        and (
                            not transcript
                            or len(transcript_words) < max(2, len(pending_words) - 1)
                        )
                    ):
                        print(f"[BARGE-IN] Final STT weak; using validated probe fallback: {pending_barge_in_text}")
                        transcript = pending_barge_in_text
                        stt_time = pending_barge_in_stt_time

                    reset_partial_state()
                    reset_tts_probe_state()
                    interrupt_segmenter.reset()

                    interrupt_capture_mode = False
                    pending_barge_in_text = ""
                    pending_barge_in_stt_time = 0.0
                    interrupt_capture_started_at = 0.0
                    interrupt_last_voice_at = 0.0

                    if (
                        transcript
                        and not is_stt_hallucination_text(transcript)
                        and is_valid_barge_in_text(transcript, last_chunks)
                    ):
                        emit_from_thread({
                            "type": "barge_in_accepted",
                            "text": transcript,
                        })
                        emit_from_thread({
                            "type": "stage",
                            "value": "interrupted",
                        })
                        emit_from_thread({
                            "type": "metrics",
                            "data": {
                                "stt": stt_time,
                                "sttBackend": args.stt_backend,
                            },
                        })
                        launch_pipeline(
                            transcript=transcript,
                            interruption_text=transcript,
                            interrupted_chunks=last_chunks,
                            stt_time=stt_time,
                        )
                    else:
                        print(f"[BARGE-IN FILTER] Rejected final interruption candidate: {transcript}")
                        emit_from_thread({
                            "type": "stage",
                            "value": "listening" if mic_enabled.is_set() else "idle",
                        })

                    continue
                if tts_busy.is_set() or browser_audio_active.is_set():
                    # Assistant/browser audio is active.
                    # Do not launch normal STT turns here.
                    # Do not stop from RMS alone.
                    # Instead, run short rolling STT probes on recent mic frames.
                    #
                    # If the probe text passes assistant-echo validation, stop browser audio
                    # immediately and then capture the rest of the user's utterance.

                    if tts_busy.is_set() and playback.seconds_since_tts_start() < BARGE_IN_GRACE_S:
                        reset_tts_probe_state()
                        continue

                    tts_probe_frames.append(frame_bytes)

                    if len(tts_probe_frames) > BARGE_IN_PROBE_MAX_FRAMES:
                        tts_probe_frames = tts_probe_frames[-BARGE_IN_PROBE_MAX_FRAMES:]

                    now = time.monotonic()
                    if now - tts_last_probe_t < BARGE_IN_PROBE_EVERY_S:
                        continue

                    tts_last_probe_t = now

                    probe_audio_bytes = b"".join(tts_probe_frames)
                    probe_audio_np = pcm16_bytes_to_float32(probe_audio_bytes)

                    if len(probe_audio_np) < int(BARGE_IN_PROBE_MIN_AUDIO_S * TARGET_SR):
                        continue

                    if audio_rms_float(probe_audio_np) < BARGE_IN_PROBE_MIN_RMS_FLOAT:
                        continue

                    probe_text, probe_stt_time = transcribe_final_timed(probe_audio_np)
                    probe_text = probe_text.strip()

                    if not probe_text:
                        continue

                    if is_stt_hallucination_text(probe_text):
                        print(f"[BARGE-IN PROBE FILTER] Rejected hallucination probe: {probe_text}")
                        clear_frontend_transcript()

                        # Do not reset the entire rolling window here.
                        # Whisper may hallucinate on one probe, but the next probe can
                        # still contain real user speech. Keep recent frames and let the
                        # next scheduled probe retry.
                        continue

                    last_chunks = playback.get_interrupt_context(n_finished=3)

                    if not is_valid_barge_in_text(probe_text, last_chunks):
                        print(f"[BARGE-IN PROBE] Rejected assistant echo/noise: {probe_text}")
                        continue

                    print("[BARGE-IN] Valid rolling user-speech probe detected.")
                    print(f"[BARGE-IN] Probe text: {probe_text}")

                    # Stop browser audio immediately after text-level validation.
                                    # Stop browser audio immediately after text-level validation.
                    emit_from_thread({"type": "audio_stop"})
                    emit_from_thread({"type": "stage", "value": "interrupted"})

                    playback.stop_now()
                    browser_audio_active.clear()
                    segmenter.reset()
                    interrupt_segmenter.reset()
                    drain_after_tts.set()

                    # Seed final interruption capture with the already validated probe audio.
                    # Also store validated text as fallback in case VAD does not finalize.
                    reset_partial_state()
                    partial_frames.extend(tts_probe_frames)

                    pending_barge_in_text = probe_text
                    pending_barge_in_stt_time = probe_stt_time

                    interrupt_capture_started_at = time.monotonic()
                    interrupt_last_voice_at = interrupt_capture_started_at

                    # Show the first accepted probe immediately as live text.
                    # The final message will still be based on the complete captured audio.
                    emit_from_thread({
                        "type": "transcript_partial",
                        "text": probe_text,
                    })

                    reset_tts_probe_state()
                    interrupt_capture_mode = True
                    interrupt_counter = 0

                    continue
                if drain_after_tts.is_set():
                    _ = drain_queue(q_frames)
                    segmenter.reset()
                    interrupt_segmenter.reset()
                    reset_partial_state()
                    reset_tts_probe_state()
                    interrupt_counter = 0
                    ignore_audio_until = time.monotonic() + 0.8
                    drain_after_tts.clear()
                    continue

                if time.monotonic() < ignore_audio_until:
                    segmenter.reset()
                    interrupt_segmenter.reset()
                    reset_partial_state()
                    reset_tts_probe_state()
                    continue

                # Start the live preview after voice-level energy is detected.
                # Once started, retain quieter frames so word endings are not cut.
                normal_now = time.monotonic()

                if rms >= INTERRUPT_RMS_THRESHOLD:
                    partial_last_voice_t = normal_now

                if rms >= INTERRUPT_RMS_THRESHOLD or partial_frames:
                    partial_frames.append(frame_bytes)

                    emit_from_thread({
                        "type": "stage",
                        "value": "transcribing",
                    })

                    maybe_emit_partial()

                # A noise spike can start a preview without creating a valid VAD utterance.
                # Clear that abandoned preview after sustained silence.
                if (
                    partial_frames
                    and partial_last_voice_t > 0.0
                    and normal_now - partial_last_voice_t >= PARTIAL_SILENCE_RESET_S
                ):
                    reset_partial_state()
                    clear_frontend_transcript()

                utt = segmenter.process_frame(frame_bytes, ts)
                if utt is None:
                    continue

                audio_np = utt.audio.astype(np.float32) / 32768.0
                transcript, stt_time = transcribe_final_timed(audio_np)

                reset_partial_state()
                segmenter.reset()

                
                if not is_valid_normal_user_text(transcript, audio_np):
                    clear_frontend_transcript()
                    continue

                # Only apply assistant-echo rejection while assistant/browser audio is actually active.
                # Otherwise repeated topic questions like "What is decision tree?" get falsely rejected
                # because old assistant chunks remain in playback history.
                if tts_busy.is_set() or browser_audio_active.is_set():
                    recent_assistant_chunks = playback.get_interrupt_context(n_finished=3)

                    if not is_valid_barge_in_text(transcript, recent_assistant_chunks):
                        print(f"[NORMAL FILTER] Rejected assistant echo as active-audio leakage: {transcript}")
                        clear_frontend_transcript()
                        continue

                norm_transcript = normalize_stt_text(transcript)
                now = time.monotonic()

                if (
                    norm_transcript
                    and norm_transcript == normalize_stt_text(last_final_user_text)
                    and now - last_final_user_at < 3.0
                ):
                    print(f"[DUPLICATE TURN FILTER] Ignored repeated final transcript: {transcript}")
                    reset_partial_state()
                    segmenter.reset()
                    continue

                if pipeline_active.is_set():
                    print(f"[NORMAL FILTER] Ignored normal user turn while pipeline is already active: {transcript}")
                    reset_partial_state()
                    segmenter.reset()
                    continue

                last_final_user_text = transcript
                last_final_user_at = now

                emit_from_thread({
                    "type": "user_transcript_final",
                    "text": transcript,
                })

                # Send STT latency immediately so the UI/pie chart updates before TTS finishes.
                emit_from_thread({
                    "type": "metrics",
                    "data": {
                        "stt": stt_time,
                        "sttBackend": args.stt_backend,
                    },
                })

                launch_pipeline(transcript=transcript, stt_time=stt_time)

                
            except Exception as e:
                print(f"[WORKER ERROR] {type(e).__name__}: {e}")
                reset_partial_state()
                reset_tts_probe_state()
                segmenter.reset()
                interrupt_segmenter.reset()
                interrupt_capture_mode = False
                interrupt_counter = 0
                pending_barge_in_text = ""
                pending_barge_in_stt_time = 0.0
                interrupt_capture_started_at = 0.0
                interrupt_last_voice_at = 0.0
                emit_from_thread({
                    "type": "stage",
                    "value": "listening" if mic_enabled.is_set() else "idle",
                })
                continue
    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    try:
        await ws_send({"type": "stage", "value": "idle"})

        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            action = data.get("action")
            config = data.get("config") or {}

            if action == "barge_in":
                apply_config(config)

                text = (data.get("text") or "").strip()
                if not text:
                    await ws_send({"type": "error", "message": "Empty barge-in text."})
                    continue

                # Browser speech recognition can emit polite hallucinations such as
                # "Thank you very much" from silence or TTS echo. Do not stop/restart
                # the tutor for these.
                if is_stt_hallucination_text(text):
                    print(f"[FRONTEND BARGE-IN FILTER] Rejected hallucination text: {text}")
                    clear_frontend_transcript()
                    await ws_send({
                        "type": "stage",
                        "value": "speaking" if (tts_busy.is_set() or browser_audio_active.is_set()) else "listening",
                    })
                    continue

                last_chunks = playback.get_interrupt_context(n_finished=2)

                if not is_valid_barge_in_text(text, last_chunks):
                    await ws_send({
                        "type": "stage",
                        "value": "speaking" if tts_busy.is_set() else "listening",
                    })
                    continue

                print("\n[BARGE-IN] Valid frontend user interruption")
                print(f"  text: {text}")

                reset_partial_state()

                await ws_send({
                    "type": "barge_in_accepted",
                    "text": text,
                })

                await ws_send({
                    "type": "stage",
                    "value": "interrupted",
                })

                playback.stop_now()
                segmenter.reset()
                interrupt_segmenter.reset()
                drain_after_tts.set()

                launch_pipeline(
                    transcript=text,
                    interruption_text=text,
                    interrupted_chunks=last_chunks,
                )
                continue


            if action in {"start_audio", "start_stream"}:
                apply_config(config)
                mic_enabled.set()
                reset_partial_state()
                await ws_send({"type": "stage", "value": "listening"})
                continue

            if action == "stop_audio":
                mic_enabled.clear()
                reset_partial_state()
                segmenter.reset()
                interrupt_segmenter.reset()
                await ws_send({"type": "stage", "value": "idle"})
                continue

            if action == "audio_frame":
                samples = data.get("data") or []
                if samples:
                    frame_np = np.array(samples, dtype=np.int16)
                    q_frames.put((frame_np.tobytes(), time.monotonic()))
                continue
            
            if action == "browser_audio_finished":
                browser_audio_active.clear()
                reset_tts_probe_state()

                if not interrupt_capture_mode:
                    drain_after_tts.set()
                    await ws_send({
                        "type": "stage",
                        "value": "listening" if mic_enabled.is_set() else "idle",
                    })

                continue


            if action == "interrupt":
                playback.stop_now()
                browser_audio_active.clear()
                drain_after_tts.set()

                interrupt_capture_mode = False
                interrupt_counter = 0
                pending_barge_in_text = ""
                pending_barge_in_stt_time = 0.0
                interrupt_capture_started_at = 0.0
                interrupt_last_voice_at = 0.0

                reset_partial_state()
                reset_tts_probe_state()
                segmenter.reset()
                interrupt_segmenter.reset()

                await ws_send({"type": "audio_stop"})
                await ws_send({"type": "stage", "value": "interrupted"})
                continue

            if action == "text_query":
                apply_config(config)
                text = (data.get("text") or "").strip()
                if not text:
                    await ws_send({"type": "error", "message": "Empty text query."})
                    continue

                reset_partial_state()
                launch_pipeline(transcript=text)
                continue

            await ws_send({"type": "error", "message": f"Unknown action: {action}"})

    except WebSocketDisconnect:
        pass

    finally:
        session_closed.set()
        mic_enabled.clear()
        playback.stop_now()
        reset_partial_state()
        partial_executor.shutdown(
            wait=False,
            cancel_futures=True,
        )
