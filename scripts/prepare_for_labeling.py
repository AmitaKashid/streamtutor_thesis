import json

INPUT = "data/runs/benchmark_output.jsonl"
OUTPUT = "data/evaluation_dataset.jsonl"

out = []

with open(INPUT, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)

        item["labels"] = {
            "retrieval_relevance": None,
            "groundedness": None,
            "correctness": None,
            "clarity": None,
            "pedagogy": None,
            "response_appropriateness": None
        }

        item["notes"] = ""

        out.append(item)

with open(OUTPUT, "w", encoding="utf-8") as f:
    for item in out:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("✅ evaluation_dataset.jsonl created")