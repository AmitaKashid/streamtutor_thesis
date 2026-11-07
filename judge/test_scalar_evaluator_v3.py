from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from judge.scalar_evaluator_v3 import (
    DEFAULT_RUBRIC_PATH,
    ScalarJudgeV3,
    build_scalar_prompt_v3,
    load_frozen_rubric,
)
from llm.llmcli.providers.base import GenerationResult


def _generation(
    text: str,
    *,
    raw_text: str | None = None,
    protocol_error: str | None = None,
) -> GenerationResult:
    return GenerationResult(
        text=text,
        raw_text=raw_text,
        response_protocol="harmony" if raw_text else None,
        protocol_error=protocol_error,
        total_seconds=0.25,
        prompt_tokens=100,
        new_tokens=20,
        tokens_per_second=80.0,
    )


class FakeProvider:
    def __init__(self, *outputs: GenerationResult | Exception) -> None:
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> GenerationResult:
        self.prompts.append(prompt)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


class ScalarEvaluatorV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.rubric = load_frozen_rubric(DEFAULT_RUBRIC_PATH)

    def _prompt(self, **overrides: object) -> str:
        values = {
            "rubric": self.rubric,
            "dimension": "correctness",
            "user_utterance": "And why does that happen?",
            "conversation_history": [
                {"role": "user", "content": "What is entropy?"},
                {"role": "assistant", "content": "It measures impurity."},
            ],
            "answer": "Because the class distribution becomes purer.",
            "reference_answer": "A correct explanation.",
            "context": "Course evidence.",
            "learner_level": "masters",
            "teaching_style": "socratic",
        }
        values.update(overrides)
        return build_scalar_prompt_v3(**values)  # type: ignore[arg-type]

    def _evaluator(
        self,
        output_path: Path,
        provider: FakeProvider,
    ) -> ScalarJudgeV3:
        return ScalarJudgeV3(
            provider,
            model_id="candidate",
            quantization="Q4_K_M",
            hardware="test-gpu",
            output_path=output_path,
            dataset_id="dataset-v3",
            dataset_version="3.0.0",
            rubric_path=DEFAULT_RUBRIC_PATH,
            declared_generation_config={"temperature": 0.0},
        )

    def _log_kwargs(self) -> dict[str, object]:
        return {
            "evaluation_id": "response-1::correctness",
            "question_id": "question-1",
            "response_id": "response-1",
            "dimension": "correctness",
            "user_utterance": "And why?",
            "conversation_history": [
                {"role": "user", "content": "What is entropy?"}
            ],
            "answer": "Because impurity falls.",
            "reference_answer": "Explain why impurity falls.",
            "context": "Course evidence.",
            "learner_level": "masters",
            "teaching_style": "socratic",
            "case_metadata": {"split": "calibration"},
        }

    def test_prompt_uses_frozen_anchors_and_history(self) -> None:
        prompt = self._prompt()
        anchor = self.rubric.anchors_for("correctness")[5]
        self.assertIn(anchor, prompt)
        self.assertIn('"role": "assistant"', prompt)
        self.assertIn("And why does that happen?", prompt)

    def test_retrieval_prompt_omits_tutor_answer(self) -> None:
        prompt = self._prompt(
            dimension="retrieval_relevance",
            answer=None,
            context=None,
        )
        self.assertNotIn("<tutor_answer>", prompt)
        self.assertIn("<MISSING>", prompt)
        self.assertIn("Do not evaluate any tutor answer", prompt)

    def test_groundedness_requires_usable_context(self) -> None:
        with self.assertRaisesRegex(ValueError, "usable context"):
            self._prompt(dimension="groundedness", context=None)

    def test_retry_preserves_raw_and_decoded_provenance(self) -> None:
        provider = FakeProvider(
            _generation("not valid"),
            _generation(
                "Feedback: Correct.\n[RESULT] 5",
                raw_text=(
                    "<|channel|>final<|message|>"
                    "Feedback: Correct.\n[RESULT] 5"
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            record = self._evaluator(path, provider).evaluate_and_log(
                **self._log_kwargs()
            )
            judge = record["judge"]
            self.assertEqual(judge["status"], "valid")
            self.assertTrue(judge["retry_used"])
            self.assertEqual(judge["attempt_count"], 2)
            self.assertTrue(
                judge["attempts"][1]["raw_output"].startswith(
                    "<|channel|>"
                )
            )

    def test_log_records_frozen_provenance_and_rejects_duplicate(self) -> None:
        provider = FakeProvider(
            _generation("Feedback: Correct.\n[RESULT] 5")
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            evaluator = self._evaluator(path, provider)
            record = evaluator.evaluate_and_log(**self._log_kwargs())
            self.assertEqual(record["prompt_version"], "3.0")
            self.assertEqual(record["rubric_sha256"], self.rubric.sha256)
            self.assertEqual(
                record["judge_configuration"]["declared_generation_config"],
                {"temperature": 0.0},
            )
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["evaluation_id"], record["evaluation_id"])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                evaluator.evaluate_and_log(**self._log_kwargs())
            self.assertEqual(len(provider.prompts), 1)

    def test_generation_failure_is_logged_as_error(self) -> None:
        provider = FakeProvider(ConnectionError("offline"))
        with tempfile.TemporaryDirectory() as directory:
            record = self._evaluator(
                Path(directory) / "predictions.jsonl",
                provider,
            ).evaluate_and_log(**self._log_kwargs())
            self.assertEqual(record["judge"]["status"], "error")
            self.assertEqual(
                record["judge"]["attempts"][0]["generation_error"]["type"],
                "ConnectionError",
            )


if __name__ == "__main__":
    unittest.main()
