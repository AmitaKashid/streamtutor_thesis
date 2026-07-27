# stt_backends/faster_whisper_backend.py
from __future__ import annotations
from typing import Optional
import os
import sys
import glob
from pathlib import Path

import numpy as np

from stt.stt_types import TranscriptResult, Segment, Word
from .base import STTBackend


def setup_cuda_windows(diag: bool = False) -> None:
    """Make pip-installed NVIDIA CUDA DLLs visible on Windows. Safe no-op elsewhere."""
    if os.name != "nt":
        return
    # Always add CUDA Toolkit bin if present (covers Toolkit installs like v12.0)
    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin"
    if os.path.isdir(cuda_bin):
        try:
            os.add_dll_directory(cuda_bin)
            if diag:
                print(f"[INFO] [dll] added CUDA Toolkit bin: {cuda_bin}")
        except Exception as e:
            if diag:
                print(f"[WARN] [dll] could not add CUDA Toolkit bin: {e}")
    if os.name == "nt":
        os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin")

    import ctypes
    import site

    roots: list[Path] = []
    try:
        roots += [Path(p) for p in site.getsitepackages() if p]
    except Exception:
        pass
    try:
        usp = site.getusersitepackages()
        if usp:
            roots.append(Path(usp))
    except Exception:
        pass
    roots.append(Path(sys.prefix) / "Lib" / "site-packages")

    uniq_roots: list[Path] = []
    seen = set()
    for r in roots:
        try:
            rr = r.resolve()
        except Exception:
            continue
        if rr.exists() and str(rr) not in seen:
            uniq_roots.append(rr)
            seen.add(str(rr))
    roots = uniq_roots

    added: list[str] = []
    for r in roots:
        nvidia_root = r / "nvidia"
        if not nvidia_root.exists():
            continue
        for bin_dir in nvidia_root.glob("**/bin"):
            if bin_dir.is_dir():
                try:
                    os.add_dll_directory(str(bin_dir))
                    added.append(str(bin_dir))
                except Exception:
                    pass

    cuda_major = None
    for r in roots:
        hits = glob.glob(str(r / "nvidia" / "**" / "cublas64_*.dll"), recursive=True)
        if hits:
            dll = Path(hits[0]).name  # cublas64_12.dll
            try:
                cuda_major = int(dll.split("_")[-1].split(".")[0])
            except Exception:
                cuda_major = None
            break

    if diag:
        print("[INFO] [dll] added NVIDIA bin dirs:")
        if added:
            for a in added:
                print("   ", a)
        else:
            print("    (none found)")

        if cuda_major:
            print(f"[INFO] [dll] detected CUDA major: {cuda_major}")
        else:
            print("[WARN] [dll] could not detect CUDA major")

        def _try(name: str):
            try:
                ctypes.WinDLL(name)
                print(f"[INFO] [dll] OK  {name}")
            except OSError as e:
                print(f"[WARN] [dll] ERR {name}: {e}")

        if cuda_major:
            m = cuda_major
            for n in (f"cudart64_{m}.dll", f"cublas64_{m}.dll", f"cublasLt64_{m}.dll", f"nvrtc64_{m}.dll"):
                _try(n)
        for n in ("cudnn64_9.dll", "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll"):
            _try(n)

    os.environ.setdefault("CUDNN_PRECOMPILED_ENGINE_ENABLE", "1")
    os.environ.setdefault("CUDNN_RUNTIME_COMPILE", "0")


class FasterWhisperBackend(STTBackend):
    def __init__(
        self,
        model: str = "small.en",
        device: str = "auto",
        device_index: int = 0,
        compute_type: str = "int8_float16",
        beam_size: int = 10,
        diag_dll: bool = False,
    ):
        self.engine = "faster-whisper"
        self.beam_size = beam_size

        # Windows CUDA DLL path fix (only relevant for CUDA usage on Windows)
        if device in ("auto", "cuda") and os.name == "nt":
            setup_cuda_windows(diag=diag_dll)
            # ---- STEP 3: Preflight DLL checks (fail early, clear error) ----
            # This prevents crashing later inside the worker thread when inference starts.
            if device == "cuda":
                import ctypes
                try:
                    ctypes.WinDLL("cudart64_12.dll")
                    ctypes.WinDLL("cublas64_12.dll")
                    ctypes.WinDLL("cublasLt64_12.dll")
                    # nvrtc is sometimes required depending on build/features
                    try:
                        ctypes.WinDLL("nvrtc64_12.dll")
                    except OSError:
                        # not always required; keep as warning if diag enabled
                        if diag_dll:
                            print("[WARN] [dll] nvrtc64_12.dll not loadable (often OK).")
                except OSError as e:
                    raise RuntimeError(
                        "CUDA requested but required CUDA 12 DLLs are not loadable. "
                        "Make sure CUDA Toolkit bin is added via setup_cuda_windows() "
                        "or added to PATH. Original error: "
                        f"{e}"
                    )

        from faster_whisper import WhisperModel

        # Friendly fallback: if CUDA init fails and user forced cuda, fallback to CPU
        # ---- Make compute_type safe across machines ----
        ct = (compute_type or "").lower()

        if device == "cpu":
            # CPU: int8 is broadly supported; int8_float16 is not always supported on Windows CPU backend
            if ct == "int8_float16":
                print("[WARN] compute_type int8_float16 not supported on this CPU backend; using int8 instead.")
                compute_type = "int8"
            elif ct not in ("int8", "float32", "int16"):
                print(f"[WARN] Unknown/unsupported CPU compute_type '{compute_type}'; using int8.")
                compute_type = "int8"

        if device == "cuda":
            # CUDA: float16 or int8_float16 are typical
            if ct not in ("float16", "int8_float16", "int8"):
                print(f"[WARN] Unknown/unsupported CUDA compute_type '{compute_type}'; using float16.")
                compute_type = "float16"

        if device == "auto":
            # Let CTranslate2 decide, but avoid int8_float16 causing CPU failures
            if ct == "int8_float16":
                # safest "auto" default is int8
                compute_type = "int8"

        try:
            self.model = WhisperModel(
                model,
                device=device,
                device_index=device_index,
                compute_type=compute_type,
            )
        except Exception as e:
            if device == "cuda":
                print(f"[WARN] CUDA init failed ({e}). Falling back to CPU (int8_float16).")
                #self.model = WhisperModel(model, device="cpu", compute_type="int8_float16")
                self.model = WhisperModel(model, device="cpu", compute_type="int8")
            else:
                raise

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: Optional[str] = None,
        context: Optional[str] = None,
        return_words: bool = False,
    ) -> TranscriptResult:
        # Expect float32 mono; sample_rate should be 16000 in your standardized pipeline
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        segments, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=self.beam_size,
            vad_filter=False,
            word_timestamps=return_words,
            condition_on_previous_text=True,
            initial_prompt=context or None,
            temperature=0.0,
            patience=1.1,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            task="transcribe",
        )

        segments = list(segments)

        out_segments = []
        text_parts = []
        for s in segments:
            text_parts.append(s.text)
            words_out = None
            if return_words and getattr(s, "words", None):
                words_out = [Word(start=w.start, end=w.end, word=w.word) for w in s.words]
            out_segments.append(Segment(start=s.start, end=s.end, text=s.text, words=words_out))

        return TranscriptResult(
            text="".join(text_parts).strip(),
            segments=out_segments,
            language=getattr(info, "language", None),
            engine=self.engine,
        )
