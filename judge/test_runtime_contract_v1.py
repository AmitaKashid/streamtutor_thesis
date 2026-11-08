import unittest
from dataclasses import replace
from judge.runtime_contract_v1 import (
    CONTRACT_VERSION, Evidence, EvaluationRequest, EvaluationStatus,
    TurnOutcome, check_result, digest_text, submission_decision,
)


def request(**overrides):
    ev = Evidence("slides:decision-tree:11", digest_text("Information gain."), "Information gain.", 1, .88)
    attrs = dict(contract_version=CONTRACT_VERSION, turn_id="turn-001", conversation_id="conv-001",
                 outcome=TurnOutcome.DELIVERED, question_text="What is information gain?",
                 generated_answer_text="Reduction in uncertainty.", delivered_answer_text="Reduction in uncertainty.",
                 history=({"role":"user", "content":"Explain trees."},), evidence=(ev,), tutor_model_id="tutor-9b",
                 tutor_prompt_sha256="tutor-prompt-hash", rubric_sha256="rubric-hash", judge_model_id="fare_20b_q4",
                 judge_prompt_sha256="judge-prompt-hash", evaluation_dimension="correctness", teacher_style="socratic",
                 generation_completed=True, delivery_completed=True)
    attrs.update(overrides)
    return EvaluationRequest(**attrs)


class RuntimeContractTests(unittest.TestCase):
    def test_valid_delivered_turn_is_eligible(self):
        self.assertEqual(submission_decision(request()), (True, "eligible"))

    def test_interrupted_turn_is_skipped_even_if_generated(self):
        r = request(outcome=TurnOutcome.INTERRUPTED, delivery_completed=False,
                    delivered_answer_text="Reduction")
        self.assertEqual(submission_decision(r), (False, "interrupted"))

    def test_incomplete_delivered_turn_is_rejected(self):
        with self.assertRaises(ValueError):
            request(delivery_completed=False)
        with self.assertRaises(ValueError):
            request(delivered_answer_text="different")

    def test_groundedness_without_context_is_skipped(self):
        self.assertEqual(submission_decision(request(evidence=(), evaluation_dimension="groundedness")),
                         (False, "no_retrieved_evidence"))

    def test_immutable_idempotency_key(self):
        r = request()
        self.assertEqual(r.idempotency_key, request().idempotency_key)
        self.assertNotEqual(r.idempotency_key, replace(r, judge_model_id="compassjudger_2_7b_q4").idempotency_key)
        self.assertNotEqual(r.idempotency_key, replace(r, rubric_sha256="other-rubric").idempotency_key)
        self.assertNotEqual(r.idempotency_key, replace(r, evaluation_dimension="pedagogy").idempotency_key)

    def test_payload_conflict_detectable(self):
        a = request()
        b = replace(a, question_text="different question")
        self.assertEqual(a.idempotency_key, b.idempotency_key)
        self.assertNotEqual(a.payload_hash, b.payload_hash)

    def test_evidence_tamper_rejected(self):
        with self.assertRaises(ValueError):
            Evidence("slide1", "incorrect-sha", "actual", 1)

    def test_duplicate_sources_rejected(self):
        r = request()
        with self.assertRaises(ValueError):
            request(evidence=(r.evidence[0], r.evidence[0]))

    def test_result_requires_integer_score_and_feedback(self):
        check_result(5, EvaluationStatus.VALID, "Factually correct")
        for score in (0, 6, 2.5, True, None):
            with self.assertRaises(ValueError):
                check_result(score, EvaluationStatus.VALID, "feedback")
        with self.assertRaises(ValueError):
            check_result(1, EvaluationStatus.ERROR, "server offline")

    def test_cancelled_generation_never_eligible(self):
        r = request(outcome=TurnOutcome.GENERATION_FAILED, generation_completed=False,
                    delivery_completed=False, generated_answer_text="", delivered_answer_text="")
        self.assertFalse(r.eligible)

    def test_no_self_interrupt(self):
        with self.assertRaises(ValueError):
            request(interruption_parent_turn_id="turn-001")

if __name__ == "__main__":
    unittest.main()
