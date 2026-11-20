"""Download pinned CPU caption assets. Run once, outside the live request path."""
from pathlib import Path
import hashlib
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from app.streaming_ctc import MODEL_DIR, MODEL_REVISION, MODEL_SHA256, CpuAcousticModel
    files = {
        'model_quantized.onnx': ('onnx/model_quantized.onnx', MODEL_SHA256),
        'vocab.json': ('vocab.json', '4178db26b3c7570f6a47f14ac6a1c7b32950b8c2800fb097287e53776934f1c5'),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for filename, (remote, digest) in files.items():
        target = MODEL_DIR / filename
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
            print(f'Verified {filename}')
            continue
        temporary = target.with_suffix(target.suffix + '.download')
        url = f'https://huggingface.co/Xenova/wav2vec2-base-960h/resolve/{MODEL_REVISION}/{remote}'
        try:
            hasher = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    hasher.update(chunk)
            if hasher.hexdigest() != digest:
                raise RuntimeError(f'Checksum mismatch for {filename}')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'Installed {filename}')
    CpuAcousticModel()  # verify CPU inference and dependencies before restart
    print('CPU captions ready. Restart frontend_api. Qwen runtime needs no extra packages.')


if __name__ == '__main__':
    main()
