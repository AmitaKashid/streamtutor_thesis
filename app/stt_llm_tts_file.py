import sys
import time
from pathlib import Path
out_path = None



ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ----- STT -----
from stt.stt_backends.faster_whisper_backend import FasterWhisperBackend
from stt.audio_file import load_audio_mono_16k

# ----- LLM -----
from llm.llmcli.models.registry import get_model_spec
from llm.llmcli.providers.llamacpp import LlamaCppProvider

# ----- TTS -----
from tts.chatterbox_engine import ChatterboxEngine
from tts.tts_queue import TTSQueue, TTSRequest


def main():
    import argparse

    p = argparse.ArgumentParser(description="WAV -> STT -> LLM -> TTS (non-streaming)")
    p.add_argument("--input", required=True, help="Path to WAV file")
    p.add_argument("--model", required=True, help="LLM model key from registry (e.g. llama3_8b_q5)")

    # STT options
    p.add_argument("--stt_model", default="small.en")
    p.add_argument("--stt_device", default="cuda", choices=["cpu", "cuda", "auto"])
    p.add_argument("--stt_compute_type", default="float16")

    # TTS options
    p.add_argument("--speak", action="store_true", help="Play audio via sounddevice")
    p.add_argument("--save", action="store_true", help="Always save output wav to tts/output/")
    p.add_argument("--voice_prompt", default=None, help="Optional voice prompt wav path")

    args = p.parse_args()

    
    out_path = None
    if args.save:
        out_path = Path("tts_output") / f"answer_{int(time.time())}.wav"

    wav_path = Path(args.input)
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV not found: {wav_path}")

    # ---------------- TTS init ----------------
    # If you are running on a cluster with no audio device, DO NOT use --speak.
    engine = ChatterboxEngine()
    tts_queue = TTSQueue(engine=engine, playback_fn=None if args.speak else (lambda wav, sr: None))
    tts_queue.start()

    try:
        # ---------------- STT ----------------
        stt_backend = FasterWhisperBackend(
            model=args.stt_model,
            device=args.stt_device,
            compute_type=args.stt_compute_type,
            beam_size=5,
        )

        stt_t0 = time.perf_counter()
        audio, sr = load_audio_mono_16k(str(wav_path))
        stt_res = stt_backend.transcribe(audio=audio, sample_rate=sr)
        stt_t1 = time.perf_counter()

        transcript = (stt_res.text or "").strip()
        print("\n[STT Transcript]")
        print(transcript)
        print(f"[STT Latency] total={(stt_t1 - stt_t0):.3f}s")

        if not transcript:
            print("\n[INFO] Empty transcript. Exiting.")
            return

        # ---------------- LLM ----------------
        spec = get_model_spec(args.model)
        llm = LlamaCppProvider(
            model_path=spec.gguf_path,
            chat_format=spec.chat_format,
            system_prompt=(spec.system_prompt or "You are a helpful assistant."),
            n_ctx=2048,
            n_threads=8,
            n_gpu_layers=-1,
            max_new_tokens=256,
            temperature=0.2,
            top_p=0.9,
            seed=0,
        )

        llm_res = llm.generate(transcript)
        answer = (llm_res.text or "").strip()
        print("\n[LLM Answer]")
        print(answer)

        print("\n[LLM Answer]")
        print(llm_res.text)
        print(
            f"[LLM Latency] total={llm_res.total_seconds:.3f}s | "
            f"prompt_tokens={llm_res.prompt_tokens} | new_tokens={llm_res.new_tokens} | "
            f"{llm_res.tokens_per_second:.2f} tok/s"
        )

        # ---------------- TTS ----------------
        # Non-streaming: enqueue ONE request with the full answer.
        text_for_tts = (llm_res.text or "").strip()
        if not text_for_tts:
            print("\n[INFO] Empty LLM output; nothing to speak.")
            return

        voice_prompt = Path(args.voice_prompt) if args.voice_prompt else None
        
        if not answer:
            print("[INFO] Empty LLM output. Nothing to synthesize.")
            return

        out_path = None
        if args.save:
            out_path = Path("tts_output") / f"answer_{int(time.time())}.wav"

        req = TTSRequest(
            text=answer,
            request_id=f"tts_{int(time.time())}",
            voice_prompt=Path(args.voice_prompt) if args.voice_prompt else None,
            config=None,
            save_to=out_path,
            play_audio=args.speak,
        )

        tts_queue.enqueue(req)
        tts_queue.join()
        print("\n[TTS] Done.")
        req = TTSRequest(
            text=answer,
            request_id=f"tts_{int(time.time())}",
            voice_prompt=Path(args.voice_prompt) if args.voice_prompt else None,
            config=None,
            save_to=out_path,
            play_audio=args.speak,
        )
        tts_queue.enqueue(req)
        tts_queue.join()
        print("\n[TTS] Done.")
        

        # Wait for queue to finish (non-streaming)
        # This prevents the program exiting before synthesis completes.
        
    finally:
        tts_queue.stop()


if __name__ == "__main__":
    main()
