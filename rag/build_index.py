# rag/build_index.py
import os, json, argparse
from typing import Dict, Any, List

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def pick_device(device: str) -> str:
    device = (device or "auto").lower()
    if device in ("cpu", "cuda"):
        return device
    return "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1") else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrieval.yaml")
    ap.add_argument("--device", default=None, help="auto|cpu|cuda (overrides config)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    model_name = cfg["embedding_model"]
    chunks_path = cfg["kb_chunks_path"]
    index_dir = cfg["index_dir"]
    index_name = cfg["index_name"]
    batch_size = int(cfg.get("batch_size", 64))

    device = pick_device(args.device or cfg.get("device", "auto"))

    if not os.path.exists(chunks_path):
        raise SystemExit(f"Chunks not found: {chunks_path} (run rag/build_kb_clean.py first)")

    chunks = read_jsonl(chunks_path)
    texts = [c["text"] for c in chunks]

    chunk_ids = [c["chunk_id"] for c in chunks]
    slide_ids = [c.get("slide_id") for c in chunks]
    slide_numbers = [c.get("slide_number") for c in chunks]
    lecture_ids = [c.get("lecture_id") for c in chunks]
    titles = [c.get("title") for c in chunks]
    source_pdfs = [c.get("source_pdf") for c in chunks]

    model = SentenceTransformer(model_name, device=device)

    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    ensure_dir(index_dir)
    out_path = os.path.join(index_dir, f"{index_name}.npz")

    np.savez_compressed(
        out_path,
        vecs=vecs,
        chunk_ids=np.array(chunk_ids, dtype=object),
        slide_ids=np.array(slide_ids, dtype=object),
        slide_numbers=np.array(slide_numbers, dtype=object),
        lecture_ids=np.array(lecture_ids, dtype=object),
        titles=np.array(titles, dtype=object),
        source_pdfs=np.array(source_pdfs, dtype=object),
    )

    print("✅ Index built")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Chunks: {len(chunks)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()