from __future__ import annotations

import json
import os
from pathlib import Path
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

from judge.judge import LocalLLMJudge
from llm.llmcli.providers.remote_http import RemoteHTTPProvider


def build_judge_llm() -> RemoteHTTPProvider:
    provider = RemoteHTTPProvider(
        base_url="http://127.0.0.1:8004",
        connect_timeout=5.0,
        read_timeout=120.0,
    )
    if not provider.healthcheck():
        raise RuntimeError("Judge server is not reachable on port 8004.")
    return provider


def run() -> None:
    root = Path(__file__).resolve().parents[1]
    samples_path = root / "judge" / "test_samples.json"
    output_path = root / "judge_outputs" / "offline_evaluation.jsonl"

    dataset = json.loads(samples_path.read_text(encoding="utf-8"))
    items = dataset.get("items", [])
    if not items:
        raise ValueError("test_samples.json contains no evaluation items.")

    judge = LocalLLMJudge(
        build_judge_llm(),
        output_path=output_path,
        reference_path=samples_path,
        enable_bleu=True,
        enable_bertscore=True,
        bertscore_model="microsoft/deberta-large-mnli",
        bertscore_device="cpu",
    )

    for index, sample in enumerate(items, 1):
        candidate_answer = str(sample.get("candidate_answer", "")).strip()
        if not candidate_answer:
            print(f"\n--- Sample {index}: SKIPPED (no candidate_answer) ---")
            continue

        record = judge.judge_and_log(
            question_id=sample["id"],
            question=sample["question"],
            answer=candidate_answer,
            teaching_style=sample.get("teaching_style", "socratic"),
            context=sample.get("context", ""),
            retrieved_ids=sample.get("retrieved_ids", []),
            metadata={
                "evaluation_mode": "offline_gold_set",
                "dataset_version": dataset.get("dataset_version"),
            },
        )

        print(f"\n--- Sample {index}: {sample['id']} ---")
        print("Question:", sample["question"])
        print("Judge:", json.dumps(record["judge"], indent=2))
        print(
            "Reference metrics:",
            json.dumps(record["reference_metrics"], indent=2),
        )

    print(f"\nResults appended to: {output_path}")


if __name__ == "__main__":
    run()
