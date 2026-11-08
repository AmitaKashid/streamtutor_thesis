"""Versioned *boundary* for asynchronous tutor evaluation; no network or runtime side-effects.

Python 3.11+. Strict payload validation and pure eligibility decision, not a job queue.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from typing import Any, Mapping
import json
import uuid

CONTRACT_VERSION = "1.0.0"
RUBRIC_DIMENSIONS = frozenset({
    "correctness", "groundedness", "clarity", "completeness",
    "pedagogy", "learner_appropriateness", "conversational_context",
    "retrieval_relevance",
})


class TurnOutcome(str, Enum):
    DELIVERED = "delivered"                 # Playback completed successfully, or text-only response shown in full
    INTERRUPTED = "interrupted"             # Playback stopped before completion
    GENERATION_FAILED = "generation_failed"
    DELIVERY_FAILED = "delivery_failed"
    EMPTY = "empty"


class EvaluationStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    VALID = "valid"
    INVALID = "invalid"   # Parser/rubric failure; never silently score zero
    ERROR = "error"       # Service failure/timeouts; never silently score zero
    SKIPPED = "skipped"   # Ineligible or backpressure; observable


def digest_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def stable_json_hash(value: Any) -> str:
    return digest_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def new_turn_id() -> str:
    return str(uuid.uuid4())


def _require_string(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty")


@dataclass(frozen=True)
class Evidence:
    source_id: str
    content_sha256: str
    text: str
    retrieval_rank: int
    retrieval_score: float | None = None
    def __post_init__(self):
        _require_string(self.source_id, "source_id")
        if digest_text(self.text) != self.content_sha256:
            raise ValueError("evidence content hash mismatch")
        if not isinstance(self.retrieval_rank, int) or isinstance(self.retrieval_rank, bool) or self.retrieval_rank < 1:
            raise ValueError("retrieval_rank must be positive integer")


@dataclass(frozen=True)
class EvaluationRequest:
    contract_version: str
    turn_id: str
    conversation_id: str
    outcome: TurnOutcome
    question_text: str
    generated_answer_text: str
    delivered_answer_text: str
    history: tuple[Mapping[str, str], ...]
    evidence: tuple[Evidence, ...]
    tutor_model_id: str
    tutor_prompt_sha256: str
    rubric_sha256: str
    judge_model_id: str
    judge_prompt_sha256: str
    evaluation_dimension: str
    teacher_style: str
    generation_completed: bool
    delivery_completed: bool
    interruption_parent_turn_id: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError("unsupported contract version")
        for name in ("turn_id", "conversation_id", "question_text", "tutor_model_id", "tutor_prompt_sha256", "rubric_sha256", "judge_model_id", "judge_prompt_sha256", "evaluation_dimension", "teacher_style"):
            _require_string(getattr(self, name), name)
        if not isinstance(self.outcome, TurnOutcome):
            raise ValueError("invalid turn outcome")
        if self.evaluation_dimension not in RUBRIC_DIMENSIONS:
            raise ValueError("unknown evaluation dimension")
        if len({e.source_id for e in self.evidence}) != len(self.evidence):
            raise ValueError("duplicate evidence source id")
        if any(x.get("role") not in ("user", "assistant") or not isinstance(x.get("content"), str) for x in self.history):
            raise ValueError("invalid history entry")
        if self.outcome == TurnOutcome.DELIVERED:
            if not (self.generation_completed and self.delivery_completed):
                raise ValueError("delivered turn must complete generation and delivery")
            if not self.generated_answer_text.strip() or self.generated_answer_text != self.delivered_answer_text:
                raise ValueError("delivered output must be nonempty and match generated output")
        if self.outcome == TurnOutcome.INTERRUPTED and self.delivery_completed:
            raise ValueError("interrupted turn cannot be marked delivered")
        if self.interruption_parent_turn_id == self.turn_id:
            raise ValueError("turn cannot interrupt itself")

    @property
    def eligible(self) -> bool:
        return self.outcome is TurnOutcome.DELIVERED

    @property
    def idempotency_key(self) -> str:
        # Unique per turn AND judge configuration, dimension, rubric/prompt version.
        # Re-evaluations under a new model or prompt must not overwrite prior results.
        return stable_json_hash({
            "contract_version": self.contract_version,
            "turn_id": self.turn_id,
            "judge_model_id": self.judge_model_id,
            "judge_prompt_sha256": self.judge_prompt_sha256,
            "rubric_sha256": self.rubric_sha256,
            "dimension": self.evaluation_dimension,
        })

    @property
    def payload_hash(self) -> str:
        return stable_json_hash({
            "question": self.question_text,
            "answer": self.delivered_answer_text,
            "history": list(self.history),
            "evidence": [e.__dict__ for e in self.evidence],
            "tutor_model_id": self.tutor_model_id,
            "tutor_prompt_sha256": self.tutor_prompt_sha256,
            "teacher_style": self.teacher_style,
        })


def submission_decision(request: EvaluationRequest) -> tuple[bool, str]:
    if not request.eligible:
        return False, request.outcome.value
    if request.evaluation_dimension == "groundedness" and not request.evidence:
        return False, "no_retrieved_evidence"
    if request.evaluation_dimension == "retrieval_relevance" and not request.evidence:
        return False, "no_retrieved_evidence"
    return True, "eligible"


def check_result(score: int | None, status: EvaluationStatus, feedback: str | None) -> None:
    if status is EvaluationStatus.VALID:
        if type(score) is not int or not (1 <= score <= 5):
            raise ValueError("valid evaluation must contain integer score 1..5")
        _require_string(feedback, "feedback")
    elif score is not None:
        raise ValueError("non-valid evaluation must not assign score")
