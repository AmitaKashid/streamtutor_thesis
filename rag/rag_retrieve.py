# rag/rag_retrieve.py
import os, json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

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


def pick_device(device: str) -> str:
    device = (device or "auto").lower()
    if device in ("cpu", "cuda"):
        return device
    return "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1") else "cpu"


@dataclass
class RagHit:
    score: float
    slide_id: str
    lecture_id: str
    slide_number: Any
    title: str
    source_pdf: str
    text: str


class RagRetriever:
    """
    Minimal retriever:
    - loads embeddings index (.npz)
    - loads slide texts (slides_clean.jsonl)
    - retrieve top_k slides for a query
    """

    def __init__(self, config_path: str = "configs/retrieval.yaml"):
        cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))

        self.model_name = cfg["embedding_model"]
        self.kb_slides_path = cfg.get("kb_slides_path", "data/kb/slides_clean.jsonl")
        self.kb_chunks_path = cfg.get("kb_chunks_path", "data/kb/chunks_slide_level.jsonl")
        self.index_dir = cfg["index_dir"]
        self.index_name = cfg["index_name"]
        self.top_k = int(cfg.get("top_k", 5))
        self.max_context_chars = int(cfg.get("max_context_chars", 6000))
        self.device = pick_device(cfg.get("device", "auto"))

        index_path = os.path.join(self.index_dir, f"{self.index_name}.npz")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found: {index_path} (run python rag/build_index.py)")

        z = np.load(index_path, allow_pickle=True)
        self.vecs = z["vecs"]
        self.slide_ids = z["slide_ids"]
        self.titles = z["titles"]
        self.slide_numbers = z["slide_numbers"]
        self.lecture_ids = z["lecture_ids"]
        self.source_pdfs = z["source_pdfs"]

        slides = read_jsonl(self.kb_slides_path)
        self.slide_text_map = {s["slide_id"]: s["text_for_embedding"] for s in slides}

        self.embedder = SentenceTransformer(self.model_name, device=self.device)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RagHit]:
        k = top_k or self.top_k

        qv = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]
        scores = self.vecs @ qv

        k = min(k, len(scores))
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]

        hits: List[RagHit] = []
        for j in idx:
            sid = str(self.slide_ids[j])
            hits.append(
                RagHit(
                    score=float(scores[j]),
                    slide_id=sid,
                    lecture_id=str(self.lecture_ids[j]),
                    slide_number=self.slide_numbers[j].item() if hasattr(self.slide_numbers[j], "item") else self.slide_numbers[j],
                    title=str(self.titles[j]),
                    source_pdf=str(self.source_pdfs[j]),
                    text=self.slide_text_map.get(sid, ""),
                )
            )
        return hits

    def format_context(self, hits: List[RagHit]) -> str:
        blocks = []
        used = 0
        for h in hits:
            block = f"[{h.slide_id} | {h.source_pdf} | slide {h.slide_number} | {h.title} | score={h.score:.3f}]\n{h.text}"
            if used + len(block) > self.max_context_chars:
                break
            blocks.append(block)
            used += len(block) + 5
        return "\n\n---\n\n".join(blocks)

    def build_prompt(self, query: str, context: str) -> str:
        # Compact instruction to keep latency low
        return (
            "Use the provided context to answer the question. "
            "If the context is insufficient, answer from general knowledge but say so briefly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "Answer (brief):"
        )