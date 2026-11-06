from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from judge.scalar_evaluator_v2 import (
    ScalarJudgeV2,
    build_scalar_prompt,
)


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def generate(self, prompt: str):
        self.prompts.append(prompt)
        output = self.outputs.pop(0)

        if isinstance(output, Exception):
            raise output

        return SimpleNamespace(
            text=output,
            raw_text=output,
            response_protocol=None,
            protocol_error=None,
            total_seconds=0.25,
            prompt_tokens=100,
            new_tokens=20,
            tokens_per_second=80.0,
        )


class ScalarEvaluatorV2Tests(unittest.TestCase):
    def make_evaluator(self, outputs, directory):
        return ScalarJudgeV2(
            FakeLLM(outputs),
            model_id="test-model",
            quantization="test-quantization",
            hardware="test-gpu",
            output_path=Path(directory) / "results.jsonl",
        )

    def test_prompt_contains_dimension_and_data(self):
        prompt = build_scalar_prompt(
            dimension="correctness",
            question="What is clustering?",
            answer="Clustering groups similar objects.",
            reference_answer="Clustering groups similar data objects.",
            context="Course context.",
            learner_level="beginner",
            teaching_style="direct",
        )

        self.assertIn("correctness", prompt)
        self.assertIn("Score 1:", prompt)
        self.assertIn("Score 5:", prompt)
        self.assertIn("What is clustering?", prompt)
        self.assertIn("[RESULT]", prompt)

    def test_unknown_dimension_is_rejected(self):
        with self.assertRaises(ValueError):
            build_scalar_prompt(
                dimension="unknown",
                question="Question",
                answer="Answer",
                reference_answer="Reference",
            )

    def test_groundedness_requires_context(self):
        with self.assertRaises(ValueError):
            build_scalar_prompt(
                dimension="groundedness",
                question="Question",
                answer="Answer",
                reference_answer="Reference",
                context=None,
            )

    def test_valid_first_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = self.make_evaluator(
                ["Feedback: Fully correct. [RESULT] 5"],
                directory,
            )

            result = evaluator.evaluate_dimension(
                dimension="correctness",
                question="Question",
                answer="Answer",
                reference_answer="Reference",
            )

            self.assertEqual(result["status"], "valid")
            self.assertEqual(result["result"]["score"], 5)
            self.assertEqual(result["attempt_count"], 1)
            self.assertFalse(result["retry_used"])

    def test_invalid_first_attempt_is_retried_once(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = self.make_evaluator(
                [
                    "The score is four.",
                    "Feedback: Mostly correct. [RESULT] 4",
                ],
                directory,
            )

            result = evaluator.evaluate_dimension(
                dimension="correctness",
                question="Question",
                answer="Answer",
                reference_answer="Reference",
            )

            self.assertEqual(result["status"], "valid")
            self.assertEqual(result["result"]["score"], 4)
            self.assertEqual(result["attempt_count"], 2)
            self.assertTrue(result["retry_used"])
            self.assertEqual(
                result["attempts"][0]["raw_output"],
                "The score is four.",
            )

    def test_two_invalid_attempts_remain_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = self.make_evaluator(
                ["Invalid output", "Still invalid"],
                directory,
            )

            result = evaluator.evaluate_dimension(
                dimension="correctness",
                question="Question",
                answer="Answer",
                reference_answer="Reference",
            )

            self.assertEqual(result["status"], "invalid")
            self.assertIsNone(result["result"])
            self.assertEqual(result["attempt_count"], 2)

    def test_generation_failure_becomes_error(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = self.make_evaluator(
                [RuntimeError("server unavailable")],
                directory,
            )

            result = evaluator.evaluate_dimension(
                dimension="correctness",
                question="Question",
                answer="Answer",
                reference_answer="Reference",
            )

            self.assertEqual(result["status"], "error")
            self.assertEqual(result["attempt_count"], 1)
            self.assertEqual(
                result["attempts"][0]["generation_error"]["type"],
                "RuntimeError",
            )

    def test_evaluate_and_log_preserves_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = self.make_evaluator(
                ["Feedback: Correct. [RESULT] 5"],
                directory,
            )

            record = evaluator.evaluate_and_log(
                case_id="quality-001",
                dimension="correctness",
                question="Question",
                answer="Answer",
                reference_answer="Reference",
                learner_level="master",
            )

            path = Path(directory) / "results.jsonl"
            saved = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(record["case_id"], "quality-001")
            self.assertEqual(saved["protocol_version"], "2.1")
            self.assertEqual(
                saved["judge_configuration"]["model_id"],
                "test-model",
            )
            self.assertEqual(
                saved["judge"]["result"]["score"],
                5,
            )
            self.assertEqual(
                saved["judge"]["attempts"][0]["raw_output"],
                "Feedback: Correct. [RESULT] 5",
            )


if __name__ == "__main__":
    unittest.main()
