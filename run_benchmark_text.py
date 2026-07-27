from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

# ✅ Fix imports for your project structure
sys.path.append(str(Path(__file__).resolve().parent))

from app.mic_stt_llm_tts import build_llm
from rag.rag_retrieve import RagRetriever
from prompts.registry import compose_prompt


# ---------- helpers ----------
def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------- main ----------
def main():
    QUESTIONS_PATH = "data/benchmark_questions.jsonl"
    OUTPUT_PATH = "data/runs/benchmark_output.jsonl"

    print("Loading questions...")
    questions = read_jsonl(QUESTIONS_PATH)

    print("Loading LLM...")
    llm = build_llm("llama3_8b_q5")   # change if needed

    print("Loading RAG...")
    retriever = RagRetriever("configs/retrieval.yaml")

    results = []

    for i, q in enumerate(questions):
        print(f"\n===== [{i+1}/{len(questions)}] {q['id']} =====")

        question = q["question"]
        print("Q:", question)

        # ---------- RAG ----------
        t0 = perf_counter()
        hits = retriever.retrieve(question, top_k=2)
        rag_latency = perf_counter() - t0

        context = retriever.format_context(hits)

        # ---------- PROMPT ----------
        prompt = compose_prompt(
            style="socratic",
            user_query=question,
            context=context
        )

        # ---------- LLM ----------
        t1 = perf_counter()
        response = llm.generate(prompt)
        llm_latency = perf_counter() - t1

        answer = response.text.strip()

        print("Answer:\n", answer)

        # ---------- SAVE ----------
        results.append({
            "id": q["id"],
            "question": question,
            "retrieved_chunks": [h.text for h in hits],
            "answer": answer,
            "latency": {
                "rag_s": rag_latency,
                "llm_s": llm_latency
            }
        })

    write_jsonl(OUTPUT_PATH, results)

    print("\n✅ DONE")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()