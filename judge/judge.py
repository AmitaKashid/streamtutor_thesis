from __future__ import annotations



import json
import re
import threading
import time
import unicodedata
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, wait as wait_for_futures
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
PROMPT_VERSION = "1.2"
RUBRIC_VERSION = "1.0"
SCHEMA_VERSION = "1.0"
DEFAULT_PROMPT_PATH = ROOT / "judge" / "judge_prompt_v1_2.txt"
DEFAULT_SCHEMA_PATH = ROOT / "docs" / "judge_evaluation" / "schema_v1.json"


def load_judge_prompt(path: str | Path = DEFAULT_PROMPT_PATH) -> str:
    return Path(path).read_text(encoding="utf-8")


def parse_exact_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse exactly one JSON object without extracting or repairing text."""
    stripped = text.strip()
    if not stripped:
        return None, "empty_output"
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return None, f"json_decode_error: {exc.msg} at line {exc.lineno} column {exc.colno}"
    if not isinstance(data, dict):
        return None, "top_level_value_must_be_an_object"
    return data, None


def derive_verdict(result: dict[str, Any]) -> str:
    """Apply rubric v1 thresholds after strict schema validation."""
    scores = result["scores"]
    groundedness = scores["groundedness"]
    has_critical_error = bool(result["critical_errors"])

    if (
        has_critical_error
        or scores["correctness"] <= 2
        or scores["completeness"] <= 2
        or (
            groundedness["applicable"]
            and groundedness["score"] <= 2
        )
    ):
        return "revise"

    if (
        scores["correctness"] >= 4
        and (
            not groundedness["applicable"]
            or groundedness["score"] >= 4
        )
        and scores["completeness"] >= 3
        and scores["clarity"] >= 3
        and scores["pedagogy"] >= 3
        and scores["learner_appropriateness"] >= 3
    ):
        return "accept"

    return "review"


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
        prompt_path: str | Path = DEFAULT_PROMPT_PATH,
        schema_path: str | Path = DEFAULT_SCHEMA_PATH,
        model_id: str = "unknown",
        backend: str = "unknown",
        quantization: str | None = None,
        hardware: str | None = None,
        max_pending: int = 64,
    ):
        self.llm = llm
        self.prompt_path = Path(prompt_path)
        self.schema_path = Path(schema_path)
        self.prompt = load_judge_prompt(self.prompt_path)
        self.schema = json.loads(self.schema_path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(self.schema)
        self.schema_validator = Draft202012Validator(self.schema)
        self.model_id = model_id
        self.backend = backend
        self.quantization = quantization
        self.hardware = hardware
        self.max_pending = max(1, int(max_pending))
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

        # Runtime evaluations are serialized because the remote judge model
        # handles one generation most efficiently at a time. Submission returns
        # immediately so evaluation does not delay the tutor response or TTS.
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="runtime-evaluation",
        )
        self._pending_lock = threading.Lock()
        self._pending: dict[str, Future] = {}
        self._submitted_turn_ids: set[str] = set()
        self._closed = False
        self.last_submit_error: str | None = None

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
        learner_level: str | None = None,
        reference_answers: list[str] | None = None,
        response_status: str | None = None,
        retrieval_status: str | None = None,
    ) -> str:
        retrieved_ids = retrieved_ids or []
        interrupted_chunks = interrupted_chunks or []
        reference_answers = reference_answers or []
        if response_status is None:
            response_status = (
                "interrupted"
                if interruption_text or interrupted_chunks
                else "complete"
            )
        if retrieval_status is None:
            retrieval_status = "provided" if context.strip() else "not_provided"

        evaluation_input = {
            "learner_question": question,
            "tutor_answer": answer,
            "teaching_style": teaching_style,
            "learner_level": learner_level,
            "response_status": response_status,
            "retrieval_status": retrieval_status,
            "retrieved_chunk_ids": retrieved_ids,
            "retrieved_context": context,
            "reference_answers": reference_answers,
            "interruption_text": interruption_text,
            "last_spoken_chunks_before_interruption": interrupted_chunks,
        }
        serialized = json.dumps(evaluation_input, ensure_ascii=False, indent=2)
        return (
            "EVALUATION_INPUT_START\n"
            f"{serialized}\n"
            "EVALUATION_INPUT_END"
        )

    def _run_generation_attempt(
        self,
        *,
        prompt: str,
        attempt_number: int,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        started = time.perf_counter()
        try:
            with self._llm_lock:
                generation = self.llm.generate(prompt)
        except Exception as exc:
            wall_seconds = time.perf_counter() - started
            return (
                {
                    "attempt": attempt_number,
                    "raw_output": None,
                    "json_valid": False,
                    "schema_valid": False,
                    "validation_error": None,
                    "generation_error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                    "metrics": {"wall_seconds": wall_seconds},
                },
                None,
            )

        wall_seconds = time.perf_counter() - started
        if isinstance(generation, str):
            raw_text = generation.strip()
        else:
            raw_text = (getattr(generation, "text", "") or "").strip()

        metrics = {
            "wall_seconds": wall_seconds,
            "server_total_seconds": getattr(generation, "total_seconds", None),
            "prompt_tokens": getattr(generation, "prompt_tokens", None),
            "new_tokens": getattr(generation, "new_tokens", None),
            "tokens_per_second": getattr(
                generation,
                "tokens_per_second",
                None,
            ),
        }
        parsed, parse_error = parse_exact_json_object(raw_text)
        validation_error = parse_error
        schema_valid = False

        if parsed is not None:
            errors = sorted(
                self.schema_validator.iter_errors(parsed),
                key=lambda error: tuple(str(part) for part in error.absolute_path),
            )
            if errors:
                first_error = errors[0]
                location = ".".join(str(part) for part in first_error.absolute_path)
                prefix = f"{location}: " if location else ""
                validation_error = prefix + first_error.message
            else:
                schema_valid = True
                validation_error = None

        attempt = {
            "attempt": attempt_number,
            "raw_output": raw_text,
            "json_valid": parsed is not None,
            "schema_valid": schema_valid,
            "validation_error": validation_error,
            "generation_error": None,
            "metrics": metrics,
        }
        return attempt, parsed if schema_valid else None

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
        learner_level: str | None = None,
        reference_answers: list[str] | None = None,
        response_status: str | None = None,
        retrieval_status: str | None = None,
    ) -> dict[str, Any]:
        user_input = self.build_user_input(
            question=question,
            answer=answer,
            teaching_style=teaching_style,
            context=context,
            retrieved_ids=retrieved_ids,
            interruption_text=interruption_text,
            interrupted_chunks=interrupted_chunks,
            learner_level=learner_level,
            reference_answers=reference_answers,
            response_status=response_status,
            retrieval_status=retrieval_status,
        )
        combined_prompt = f"{self.prompt}\n\n{user_input}"
        attempts: list[dict[str, Any]] = []

        first_attempt, validated = self._run_generation_attempt(
            prompt=combined_prompt,
            attempt_number=1,
        )
        attempts.append(first_attempt)

        if first_attempt["generation_error"] is not None:
            return {
                "status": "error",
                "result": None,
                "derived_verdict": None,
                "attempt_count": 1,
                "retry_used": False,
                "attempts": attempts,
            }

        if validated is None:
            retry_prompt = (
                f"{combined_prompt}\n\n"
                "RETRY_NOTICE: Your previous response was invalid. "
                "Return exactly one JSON object matching output schema v1.0. "
                "Do not include Markdown or explanatory text. "
                "Every critical_errors item must contain exactly type, description, "
                "and evidence. Do not use quotation or any additional property. "
                "Evidence is mandatory for every critical error. Return an empty "
                "critical_errors array when no critical error exists. "
                f"Validation error: {first_attempt['validation_error']}"
            )
            second_attempt, validated = self._run_generation_attempt(
                prompt=retry_prompt,
                attempt_number=2,
            )
            attempts.append(second_attempt)

        if validated is None:
            final_status = (
                "error"
                if attempts[-1]["generation_error"] is not None
                else "invalid"
            )
            return {
                "status": final_status,
                "result": None,
                "derived_verdict": None,
                "attempt_count": len(attempts),
                "retry_used": len(attempts) > 1,
                "attempts": attempts,
            }

        return {
            "status": "valid",
            "result": validated,
            "derived_verdict": derive_verdict(validated),
            "attempt_count": len(attempts),
            "retry_used": len(attempts) > 1,
            "attempts": attempts,
        }

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
        turn_id: str | None = None,
        learner_level: str | None = None,
        reference_answers: list[str] | None = None,
        response_status: str | None = None,
        retrieval_status: str | None = None,
    ) -> dict[str, Any]:
        metadata = metadata or {}
        turn_id = str(turn_id or metadata.get("turn_id") or f"turn-{uuid.uuid4().hex}")

        if reference_answers is None:
            reference_item, _match_type, _match_error = self.reference_registry.match(
                question=question,
                question_id=question_id,
            )
            if reference_item is not None:
                reference_answers = list(reference_item["reference_answers"])
            else:
                reference_answers = []

        if response_status is None:
            response_status = (
                "interrupted"
                if interruption_text or interrupted_chunks
                else "complete"
            )
        if retrieval_status is None:
            retrieval_status = "provided" if context.strip() else "not_provided"

        judge_result = self.judge_answer(
            question=question,
            answer=answer,
            teaching_style=teaching_style,
            context=context,
            retrieved_ids=retrieved_ids,
            interruption_text=interruption_text,
            interrupted_chunks=interrupted_chunks,
            learner_level=learner_level,
            reference_answers=reference_answers,
            response_status=response_status,
            retrieval_status=retrieval_status,
        )
        reference_metrics = self.score_reference_metrics(
            question=question,
            answer=answer,
            question_id=question_id,
        )

        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "turn_id": turn_id,
            "question_id": question_id,
            "question": question,
            "answer": answer,
            "teaching_style": teaching_style,
            "learner_level": learner_level,
            "response_status": response_status,
            "retrieval_status": retrieval_status,
            "retrieved_ids": retrieved_ids or [],
            "context": context,
            "reference_answers": reference_answers,
            "interruption_text": interruption_text,
            "interrupted_chunks": interrupted_chunks or [],
            "judge": judge_result,
            "judge_configuration": {
                "model_id": self.model_id,
                "backend": self.backend,
                "quantization": self.quantization,
                "hardware": self.hardware,
                "prompt_version": PROMPT_VERSION,
                "rubric_version": RUBRIC_VERSION,
                "schema_version": SCHEMA_VERSION,
                "prompt_path": str(self.prompt_path),
                "schema_path": str(self.schema_path),
            },
            "reference_metrics": reference_metrics,
            "metadata": metadata,
        }

        self.append_result(record)
        return record

    def _evaluation_finished(self, task_id: str, future: Future) -> None:
        try:
            record = future.result()
            decision = record.get("judge", {}).get(
                "derived_verdict",
                "unknown",
            )
            print(
                f"[JUDGE] Evaluation saved: {task_id} "
                f"(decision={decision}) -> {self.output_path}"
            )
        except Exception as exc:
            self.last_submit_error = f"{type(exc).__name__}: {exc}"
            print(
                f"[JUDGE] Evaluation failed: {task_id}: "
                f"{self.last_submit_error}"
            )
        finally:
            with self._pending_lock:
                self._pending.pop(task_id, None)

    def judge_and_log_async(self, **kwargs) -> str | None:
        kwargs = dict(kwargs)
        metadata = kwargs.get("metadata") or {}
        turn_id = str(
            kwargs.get("turn_id")
            or metadata.get("turn_id")
            or f"turn-{uuid.uuid4().hex}"
        )
        kwargs["turn_id"] = turn_id

        with self._pending_lock:
            if self._closed:
                self.last_submit_error = "The runtime evaluator is shutting down."
                return None

            if len(self._pending) >= self.max_pending:
                self.last_submit_error = (
                    f"Judge queue is full ({self.max_pending} pending evaluations)."
                )
                return None

            if turn_id in self._submitted_turn_ids:
                self.last_submit_error = f"Duplicate judge turn_id rejected: {turn_id}"
                return None

            task_id = f"judge-{uuid.uuid4().hex[:12]}"

            try:
                self._submitted_turn_ids.add(turn_id)
                future = self._executor.submit(self.judge_and_log, **kwargs)
            except Exception as exc:
                self._submitted_turn_ids.discard(turn_id)
                self.last_submit_error = f"{type(exc).__name__}: {exc}"
                return None

            self._pending[task_id] = future
            self.last_submit_error = None

        future.add_done_callback(
            lambda completed, current_task_id=task_id: self._evaluation_finished(
                current_task_id,
                completed,
            )
        )
        return task_id

    def shutdown(self, *, wait: bool = True, timeout: float = 30.0) -> None:
        with self._pending_lock:
            self._closed = True
            pending = list(self._pending.values())

        unfinished = set()
        if wait and pending:
            _finished, unfinished = wait_for_futures(
                pending,
                timeout=max(0.0, timeout),
            )

        if unfinished:
            print(
                f"[JUDGE] Shutdown timeout: {len(unfinished)} "
                "evaluation(s) are still running."
            )

        # Do not cancel submitted evaluations; completed work must still be
        # appended to the runtime JSONL file.
        self._executor.shutdown(wait=False, cancel_futures=False)
