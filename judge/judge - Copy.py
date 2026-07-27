from __future__ import annotations



import json
import re
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]


def load_judge_prompt() -> str:
    prompt_path = ROOT / "judge" / "judge_prompt.txt"
    return prompt_path.read_text(encoding="utf-8")


def safe_json_extract(text: str) -> Optional[dict[str, Any]]:
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except Exception:
            return None

    return None


def normalize_judge_result(result: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "correctness",
        "clarity",
        "completeness",
        "pedagogical_value",
        "faithfulness_to_context",
    ]

    out: dict[str, Any] = {}

    for field in fields:
        try:
            value = int(result.get(field, 0))
        except Exception:
            value = 0
        out[field] = max(0, min(5, value))

    out["reason"] = str(result.get("reason", "")).strip()

    if (
        out["correctness"] <= 2
        or out["clarity"] <= 2
        or out["completeness"] <= 2
    ):
        out["decision"] = "revise"
    elif out["correctness"] >= 4 and out["clarity"] >= 3:
        out["decision"] = "accept"
    else:
        out["decision"] = "review"

    return out


def normalize_question(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", normalized, flags=re.UNICODE))


def normalize_metric_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", normalized).strip()


class ReferenceRegistry:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.dataset_version: str | None = None
        self.items_by_id: dict[str, dict[str, Any]] = {}
        self.item_id_by_question: dict[str, str] = {}

        if self.path is not None:
            self._load()

    def _load(self) -> None:
        if self.path is None:
            return
        if not self.path.exists():
            raise FileNotFoundError(f"Reference registry not found: {self.path}")

        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            data = {"dataset_version": "legacy", "items": raw}
        elif isinstance(raw, dict):
            data = raw
        else:
            raise ValueError("Reference registry must be a JSON object or list.")

        items = data.get("items")
        if not isinstance(items, list):
            raise ValueError("Reference registry must contain an 'items' list.")

        self.dataset_version = str(data.get("dataset_version", "unversioned"))

        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"Registry item {index} must be a JSON object.")

            item_id = str(item.get("id", "")).strip()
            question = str(item.get("question", "")).strip()
            aliases = item.get("aliases", [])
            references = item.get("reference_answers", [])

            if not item_id:
                raise ValueError(f"Registry item {index} is missing a stable 'id'.")
            if item_id in self.items_by_id:
                raise ValueError(f"Duplicate reference item id: {item_id}")
            if not question:
                raise ValueError(f"Registry item {item_id} is missing 'question'.")
            if not isinstance(aliases, list) or not all(
                isinstance(alias, str) for alias in aliases
            ):
                raise ValueError(f"Item {item_id} has invalid 'aliases'.")
            if not isinstance(references, list) or not all(
                isinstance(reference, str) and reference.strip()
                for reference in references
            ):
                raise ValueError(
                    f"Item {item_id} has invalid 'reference_answers'."
                )

            clean_item = dict(item)
            clean_item["id"] = item_id
            clean_item["question"] = question
            clean_item["aliases"] = [alias.strip() for alias in aliases if alias.strip()]
            clean_item["reference_answers"] = [
                reference.strip() for reference in references
            ]
            self.items_by_id[item_id] = clean_item

            for wording in [question, *clean_item["aliases"]]:
                normalized = normalize_question(wording)
                existing_id = self.item_id_by_question.get(normalized)
                if existing_id is not None and existing_id != item_id:
                    raise ValueError(
                        "Question/alias collision between "
                        f"{existing_id!r} and {item_id!r}: {wording!r}"
                    )
                self.item_id_by_question[normalized] = item_id

    def match(
        self,
        *,
        question: str,
        question_id: str | None = None,
    ) -> tuple[dict[str, Any] | None, str | None, str | None]:
        if question_id:
            item = self.items_by_id.get(question_id)
            if item is None:
                return None, None, "unknown_question_id"
            return item, "question_id", None

        normalized = normalize_question(question)
        item_id = self.item_id_by_question.get(normalized)
        if item_id is None:
            return None, None, "no_exact_reference_match"

        item = self.items_by_id[item_id]
        match_type = (
            "exact_question"
            if normalized == normalize_question(item["question"])
            else "exact_alias"
        )
        return item, match_type, None


class LocalLLMJudge:
    def __init__(
        self,
        llm,
        output_path: str | Path = "judge_outputs/judge_results.jsonl",
        *,
        reference_path: str | Path | None = None,
        enable_bleu: bool = False,
        enable_bertscore: bool = False,
        bertscore_model: str = "microsoft/deberta-large-mnli",
        bertscore_device: str = "cpu",
        bertscore_rescale_with_baseline: bool = True,
    ):
        self.llm = llm
        self.prompt = load_judge_prompt()
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.reference_registry = ReferenceRegistry(reference_path)
        self.enable_bleu = enable_bleu
        self.enable_bertscore = enable_bertscore
        self.bertscore_model = bertscore_model
        self.bertscore_device = bertscore_device
        self.bertscore_rescale_with_baseline = bertscore_rescale_with_baseline

        self._write_lock = threading.Lock()
        self._llm_lock = threading.Lock()
        self._metric_lock = threading.Lock()
        self._bleu_metric = None
        self._bertscorer = None

    def build_user_input(
        self,
        *,
        question: str,
        answer: str,
        teaching_style: str,
        context: str = "",
        retrieved_ids: list[str] | None = None,
        interruption_text: str | None = None,
        interrupted_chunks: list[str] | None = None,
    ) -> str:
        retrieved_ids = retrieved_ids or []
        interrupted_chunks = interrupted_chunks or []

        return f"""
Student Question:
{question}

Assistant Answer:
{answer}

Teaching Style:
{teaching_style}

Retrieved Chunk IDs:
{retrieved_ids}

Retrieved Context:
{context}

Interruption Text:
{interruption_text or ""}

Last Spoken Chunks Before Interruption:
{interrupted_chunks}
""".strip()

    def judge_answer(
        self,
        *,
        question: str,
        answer: str,
        teaching_style: str,
        context: str = "",
        retrieved_ids: list[str] | None = None,
        interruption_text: str | None = None,
        interrupted_chunks: list[str] | None = None,
    ) -> dict[str, Any]:
        user_input = self.build_user_input(
            question=question,
            answer=answer,
            teaching_style=teaching_style,
            context=context,
            retrieved_ids=retrieved_ids,
            interruption_text=interruption_text,
            interrupted_chunks=interrupted_chunks,
        )

        with self._llm_lock:
            combined_prompt = f"{self.prompt}\n\n{user_input}"
            result = self.llm.generate(combined_prompt)

        if isinstance(result, str):
            raw_text = result.strip()
        else:
            raw_text = (getattr(result, "text", "") or "").strip()

        parsed = safe_json_extract(raw_text)
        if parsed is None:
            parsed = {
                "correctness": 0,
                "clarity": 0,
                "completeness": 0,
                "pedagogical_value": 0,
                "faithfulness_to_context": 0,
                "reason": f"JSON parsing failed. Raw output: {raw_text[:500]}",
                "decision": "review",
            }

        return normalize_judge_result(parsed)

    def _get_bleu_metric(self):
        if self._bleu_metric is None:
            try:
                from sacrebleu.metrics import BLEU
            except ImportError as exc:
                raise RuntimeError(
                    "Sentence BLEU requires sacrebleu. "
                    "Install it with: pip install sacrebleu==2.6.0"
                ) from exc

            self._bleu_metric = BLEU(
                lowercase=True,
                tokenize="13a",
                smooth_method="exp",
                effective_order=True,
            )
        return self._bleu_metric

    def _get_bertscorer(self):
        if self._bertscorer is None:
            try:
                from bert_score import BERTScorer
            except ImportError as exc:
                raise RuntimeError(
                    "BERTScore requires bert-score. "
                    "Install it with: pip install bert-score==0.3.13"
                ) from exc

            scorer = BERTScorer(
                model_type=self.bertscore_model,
                lang="en",
                device=self.bertscore_device,
                batch_size=1,
                rescale_with_baseline=self.bertscore_rescale_with_baseline,
            )

            # Some DeBERTa tokenizer configs expose an enormous placeholder
            # instead of the model's real maximum input length.
            model_limit = int(
                getattr(scorer._model.config, "max_position_embeddings", 512)
            )
            scorer._tokenizer.model_max_length = model_limit

            self._bertscorer = scorer

        return self._bertscorer

    def score_reference_metrics(
        self,
        *,
        question: str,
        answer: str,
        question_id: str | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": "not_scored",
            "reason": None,
            "sample_id": None,
            "match_type": None,
            "dataset_version": self.reference_registry.dataset_version,
            "reference_count": 0,
            "bleu": None,
            "bertscore": None,
            "errors": [],
        }

        if not self.enable_bleu and not self.enable_bertscore:
            result["reason"] = "reference_metrics_disabled"
            return result

        if not self.reference_registry.items_by_id:
            result["reason"] = "reference_registry_empty"
            return result

        item, match_type, match_error = self.reference_registry.match(
            question=question,
            question_id=question_id,
        )
        if item is None:
            result["reason"] = match_error
            return result

        references = [
            normalize_metric_text(reference)
            for reference in item["reference_answers"]
            if normalize_metric_text(reference)
        ]
        if not references:
            result.update(
                {
                    "reason": "matched_item_has_no_reference_answers",
                    "sample_id": item["id"],
                    "match_type": match_type,
                }
            )
            return result

        candidate = normalize_metric_text(answer)
        result.update(
            {
                "sample_id": item["id"],
                "match_type": match_type,
                "reference_count": len(references),
            }
        )

        if not candidate:
            result["reason"] = "empty_candidate_answer"
            return result

        with self._metric_lock:
            if self.enable_bleu:
                try:
                    bleu_metric = self._get_bleu_metric()
                    bleu_result = bleu_metric.sentence_score(candidate, references)
                    result["bleu"] = {
                        "score_0_100": round(float(bleu_result.score), 6),
                        "score_0_1": round(float(bleu_result.score) / 100.0, 6),
                        "signature": str(bleu_metric.get_signature()),
                    }
                except Exception as exc:
                    result["errors"].append(
                        {
                            "metric": "bleu",
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )

            if self.enable_bertscore:
                try:
                    scorer = self._get_bertscorer()
                    precision, recall, f1 = scorer.score(
                        [candidate],
                        [references],
                    )
                    result["bertscore"] = {
                        "precision": round(float(precision[0].item()), 6),
                        "recall": round(float(recall[0].item()), 6),
                        "f1": round(float(f1[0].item()), 6),
                        "model": self.bertscore_model,
                        "device": self.bertscore_device,
                        "rescaled_with_baseline": (
                            self.bertscore_rescale_with_baseline
                        ),
                        "hash": str(getattr(scorer, "hash", "")),
                    }
                except Exception as exc:
                    result["errors"].append(
                        {
                            "metric": "bertscore",
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )

        requested_count = int(self.enable_bleu) + int(self.enable_bertscore)
        completed_count = int(result["bleu"] is not None) + int(
            result["bertscore"] is not None
        )

        if completed_count == requested_count:
            result["status"] = "scored"
            result["reason"] = None
        elif completed_count > 0:
            result["status"] = "partial"
            result["reason"] = "one_or_more_metrics_failed"
        else:
            result["status"] = "failed"
            result["reason"] = "all_requested_metrics_failed"

        return result

    def append_result(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False)
        with self._write_lock:
            with self.output_path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")

    def judge_and_log(
        self,
        *,
        question: str,
        answer: str,
        teaching_style: str,
        context: str = "",
        retrieved_ids: list[str] | None = None,
        interruption_text: str | None = None,
        interrupted_chunks: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        question_id: str | None = None,
    ) -> dict[str, Any]:
        metadata = metadata or {}

        judge_scores = self.judge_answer(
            question=question,
            answer=answer,
            teaching_style=teaching_style,
            context=context,
            retrieved_ids=retrieved_ids,
            interruption_text=interruption_text,
            interrupted_chunks=interrupted_chunks,
        )
        reference_metrics = self.score_reference_metrics(
            question=question,
            answer=answer,
            question_id=question_id,
        )

        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "question_id": question_id,
            "question": question,
            "answer": answer,
            "teaching_style": teaching_style,
            "retrieved_ids": retrieved_ids or [],
            "context": context,
            "interruption_text": interruption_text,
            "interrupted_chunks": interrupted_chunks or [],
            "judge": judge_scores,
            "reference_metrics": reference_metrics,
            "metadata": metadata,
        }

        self.append_result(record)
        return record

    def judge_and_log_async(self, **kwargs) -> threading.Thread:
        thread = threading.Thread(
            target=self.judge_and_log,
            kwargs=kwargs,
            daemon=True,
            name="runtime-evaluation",
        )
        thread.start()
        return thread
