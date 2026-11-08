"""Turn-delivery coordinator. Called from normal pipeline/WS event layer, never audio callback.

Evaluator injection deliberately prevents an unverified prompt from being presented
as the frozen scalar-v3 protocol. Supply a tested ScalarJudgeV3 adapter in Task 6.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import threading
from typing import Callable

from judge.runtime_contract_v1 import (
    CONTRACT_VERSION, EvaluationRequest, Evidence, TurnOutcome, digest_text,
    RUBRIC_DIMENSIONS,
)
from judge.runtime_service_v1 import RuntimeJudgeService

# Retrieval relevance requires its own question-context scoring protocol; excluded here.
ANSWER_DIMENSIONS = ('correctness','clarity','completeness','pedagogy',
                     'learner_appropriateness','conversational_context','groundedness')

@dataclass(frozen=True)
class Candidate:
    turn_id: str
    conversation_id: str
    question: str
    answer: str
    history: tuple[dict[str, str], ...]
    rag_context: str
    tutor_model: str
    tutor_prompt_sha256: str
    teaching_style: str
    generation_completed: bool

class DeliveryCoordinator:
    def __init__(self, service: RuntimeJudgeService, *, rubric_sha256: str,
                 judge_prompt_sha256: str, judge_model_id: str = 'fare_20b_q4',
                 dimensions=ANSWER_DIMENSIONS):
        self.service=service
        self.rubric_sha256=rubric_sha256
        self.judge_prompt_sha256=judge_prompt_sha256
        self.judge_model_id=judge_model_id
        self.dimensions=dimensions
        self._pending:dict[str,Candidate]={}
        self._resolved:set[str]=set()
        self._lock=threading.RLock()

    def candidate_ready(self, candidate: Candidate) -> bool:
        with self._lock:
            if candidate.turn_id in self._pending or candidate.turn_id in self._resolved:
                return False
            self._pending[candidate.turn_id]=candidate
            return True

    def acknowledge(self, turn_id: str, *, delivered: bool) -> list[tuple[str,str]]:
        with self._lock:
            if turn_id in self._resolved or turn_id not in self._pending:
                return []
            c=self._pending.pop(turn_id)
            self._resolved.add(turn_id)
        outcome=TurnOutcome.DELIVERED if delivered and c.generation_completed else (
            TurnOutcome.INTERRUPTED if not delivered else TurnOutcome.GENERATION_FAILED)
        evidence=() if not c.rag_context.strip() else (
            Evidence('rag-context-snapshot',digest_text(c.rag_context),c.rag_context,1),)
        results=[]
        for dimension in self.dimensions:
            request=EvaluationRequest(
                contract_version=CONTRACT_VERSION, turn_id=c.turn_id,conversation_id=c.conversation_id,
                outcome=outcome,question_text=c.question,generated_answer_text=c.answer,
                delivered_answer_text=c.answer if outcome==TurnOutcome.DELIVERED else '',
                history=c.history,evidence=evidence,tutor_model_id=c.tutor_model,
                tutor_prompt_sha256=c.tutor_prompt_sha256,rubric_sha256=self.rubric_sha256,
                judge_model_id=self.judge_model_id,judge_prompt_sha256=self.judge_prompt_sha256,
                evaluation_dimension=dimension,teacher_style=c.teaching_style,
                generation_completed=c.generation_completed,
                delivery_completed=outcome==TurnOutcome.DELIVERED)
            results.append(self.service.submit(request))
        return results

    def cancel_all(self) -> None:
        with self._lock:
            ids=list(self._pending)
        for key in ids: self.acknowledge(key, delivered=False)

    def only_pending_id(self):
        with self._lock:
            return next(iter(self._pending)) if len(self._pending)==1 else None
