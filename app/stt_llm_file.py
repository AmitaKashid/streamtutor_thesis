import sys
import time
from pathlib import Path

# -------------------------------------------------------
# Make local folders importable:
#   llmtutor/stt   (STT code)
#   llmtutor/llm   (contains llmcli package)
# -------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ----------------- STT imports -------------------------
from stt.stt_backends.faster_whisper_backend import FasterWhisperBackend
from stt.audio_file import load_audio_mono_16k

from llm.llmcli.models.registry import get_model_spec
from llm.llmcli.providers.llamacpp import LlamaCppProvider



def main():
    import argparse

    p = argparse.ArgumentParser(description="WAV -> STT -> LLM (non-streaming)")
    p.add_argument("--input", required=True, help="Path to WAV file")
    p.add_argument("--model", required=True, help="LLM model key from registry (e.g. llama3_8b_q5)")

    # STT options
    p.add_argument("--stt_model", default="small.en")
    p.add_argument("--stt_device", default="cuda", choices=["cpu", "cuda", "auto"])
    p.add_argument("--stt_compute_type", default="float16")

    args = p.parse_args()

    wav_path = Path(args.input)
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV not found: {wav_path}")

    # ---------- STT ----------
    stt_backend = FasterWhisperBackend(
        model=args.stt_model,
        device=args.stt_device,
        compute_type=args.stt_compute_type,
        beam_size=5,
    )

    t0 = time.perf_counter()
    audio, sr = load_audio_mono_16k(str(wav_path))
    stt_res = stt_backend.transcribe(audio=audio, sample_rate=sr)
    t1 = time.perf_counter()

    transcript = (stt_res.text or "").strip()

    print("\n[STT Transcript]")
    print(transcript)
    print(f"[STT Latency] total={(t1 - t0):.3f}s")

    if not transcript:
        print("\n[INFO] Empty transcript. Nothing to send to LLM.")
        return

    # ---------- LLM ----------
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

    print("\n[LLM Answer]")
    print(llm_res.text)
    print(
        f"[LLM Latency] total={llm_res.total_seconds:.3f}s | "
        f"prompt_tokens={llm_res.prompt_tokens} | new_tokens={llm_res.new_tokens} | "
        f"{llm_res.tokens_per_second:.2f} tok/s"
    )


if __name__ == "__main__":
    main()
