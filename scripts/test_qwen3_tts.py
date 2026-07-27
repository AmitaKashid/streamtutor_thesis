from pathlib import Path
from time import perf_counter

import soundfile as sf

from tts.qwen3tts_engine import Qwen3TTSEngine, Qwen3TTSConfig


def main():
    engine = Qwen3TTSEngine(
        Qwen3TTSConfig(
            speaker="Aiden",
            language="en",
        )
    )

    print("[TEST] Loading Qwen3-TTS...")
    engine.load()

    text = "Hello. This is a short Qwen three TTS test."
    print("[TEST] Synthesizing...")
    t0 = perf_counter()

    wav, sr = engine.synthesize(text)

    dt = perf_counter() - t0
    print(f"[TEST] Done in {dt:.3f}s | shape={tuple(wav.shape)} | sr={sr}")

    out = Path("tmp/qwen3_test.wav")
    out.parent.mkdir(exist_ok=True)
    sf.write(out, wav.squeeze(0).detach().cpu().numpy(), sr)

    print(f"[TEST] Saved: {out}")


if __name__ == "__main__":
    main()