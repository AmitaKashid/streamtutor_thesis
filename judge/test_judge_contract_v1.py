from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from judge.judge import (
    DEFAULT_PROMPT_PATH,
    PROMPT_VERSION,
    LocalLLMJudge,
    derive_verdict,
    load_judge_prompt,
    parse_exact_json_object,
)


def valid_result(*, context_applicable: bool = True) -> dict:
    return {
        "schema_version": "1.0",
        "rubric_version": "1.0",
        "scores": {
            "correctness": 4,
            "groundedness": {
                "applicable": context_applicable,
                "score": 4 if context_applicable else None,
            },
            "completeness": 4,
            "clarity": 4,
            "pedagogy": 4,
            "learner_appropriateness": 4,
        },
        "retrieval_relevance": {
            "applicable": context_applicable,
            "score": 4 if context_applicable else None,
        },
        "critical_errors": [],
        "uncertain": False,
        "uncertainty_reason": None,
        "appropriateness_assumption": None,
        "rationale": "The central claims are correct, clear, and supported.",
    }


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
            total_seconds=0.25,
            prompt_tokens=100,
            new_tokens=40,
            tokens_per_second=160.0,
        )


class JudgeContractV1Tests(unittest.TestCase):
    def make_judge(self, outputs, directory: str) -> LocalLLMJudge:
        return LocalLLMJudge(
            FakeLLM(outputs),
            output_path=Path(directory) / "results.jsonl",
            model_id="test-model",
            backend="fake",
            quantization="none",
            hardware="test-device",
        )

    def test_exact_parser_rejects_surrounding_text(self):
        parsed, error = parse_exact_json_object('prefix {"x": 1} suffix')
        self.assertIsNone(parsed)
        self.assertIn("json_decode_error", error)

    def test_prompt_v1_2_defines_critical_error_serialization(self):
        self.assertEqual(PROMPT_VERSION, "1.2")
        self.assertEqual(DEFAULT_PROMPT_PATH.name, "judge_prompt_v1_2.txt")
        prompt = load_judge_prompt()
        self.assertIn('"critical_errors": [', prompt)
        self.assertIn('"description":', prompt)
        self.assertIn('"evidence":', prompt)
        self.assertIn("Do not use quotation", prompt)
        self.assertIn('Use "critical_errors": []', prompt)

    def test_valid_result_is_accepted_and_verdict_is_derived(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = valid_result()
            judge = self.make_judge([json.dumps(payload)], directory)
            result = judge.judge_answer(
                question="What is clustering?",
                answer="Clustering groups similar observations.",
                teaching_style="direct",
                context="Clustering groups similar observations.",
            )
            self.assertEqual(result["status"], "valid")
            self.assertEqual(result["derived_verdict"], "accept")
            self.assertEqual(result["attempt_count"], 1)
            self.assertFalse(result["retry_used"])
            self.assertTrue(result["attempts"][0]["schema_valid"])

    def test_invalid_first_output_is_retained_and_retried_once(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = valid_result()
            judge = self.make_judge(
                ["Here is the result: " + json.dumps(payload), json.dumps(payload)],
                directory,
            )
            result = judge.judge_answer(
                question="What is clustering?",
                answer="Clustering groups similar observations.",
                teaching_style="direct",
                context="Clustering groups similar observations.",
            )
            self.assertEqual(result["status"], "valid")
            self.assertTrue(result["retry_used"])
            self.assertEqual(result["attempt_count"], 2)
            self.assertIn("Here is the result", result["attempts"][0]["raw_output"])
            self.assertFalse(result["attempts"][0]["json_valid"])
            self.assertTrue(result["attempts"][1]["schema_valid"])

    def test_out_of_range_score_is_not_clamped_or_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = valid_result()
            payload["scores"]["correctness"] = 6
            raw = json.dumps(payload)
            judge = self.make_judge([raw, raw], directory)
            result = judge.judge_answer(
                question="Question",
                answer="Answer",
                teaching_style="direct",
            )
            self.assertEqual(result["status"], "invalid")
            self.assertIsNone(result["result"])
            self.assertEqual(result["attempt_count"], 2)
            self.assertEqual(result["attempts"][0]["raw_output"], raw)
            self.assertIn("greater than the maximum", result["attempts"][0]["validation_error"])

    def test_generation_failure_becomes_error_result(self):
        with tempfile.TemporaryDirectory() as directory:
            judge = self.make_judge([RuntimeError("server unavailable")], directory)
            result = judge.judge_answer(
                question="Question",
                answer="Answer",
                teaching_style="direct",
            )
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["attempt_count"], 1)
            self.assertEqual(
                result["attempts"][0]["generation_error"]["type"],
                "RuntimeError",
            )

    def test_logging_keeps_provenance_raw_output_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = valid_result(context_applicable=False)
            judge = self.make_judge([json.dumps(payload)], directory)
            record = judge.judge_and_log(
                turn_id="turn-001",
                question_id="q-001",
                question="Define support.",
                answer="Support is the fraction of transactions containing an itemset.",
                teaching_style="direct",
                learner_level="master",
                response_status="complete",
                retrieval_status="not_provided",
            )
            self.assertEqual(record["turn_id"], "turn-001")
            self.assertEqual(record["judge"]["status"], "valid")
            self.assertEqual(record["judge_configuration"]["prompt_version"], "1.2")
            self.assertEqual(record["judge_configuration"]["model_id"], "test-model")
            self.assertEqual(record["judge"]["attempts"][0]["metrics"]["new_tokens"], 40)
            saved = json.loads((Path(directory) / "results.jsonl").read_text())
            self.assertEqual(saved["turn_id"], "turn-001")

    def test_duplicate_async_turn_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = json.dumps(valid_result(context_applicable=False))
            judge = self.make_judge([payload], directory)
            first = judge.judge_and_log_async(
                turn_id="turn-duplicate",
                question="Question",
                answer="Answer",
                teaching_style="direct",
            )
            second = judge.judge_and_log_async(
                turn_id="turn-duplicate",
                question="Question",
                answer="Answer",
                teaching_style="direct",
            )
            self.assertIsNotNone(first)
            self.assertIsNone(second)
            self.assertIn("Duplicate", judge.last_submit_error)
            judge.shutdown(wait=True)

    def test_critical_error_forces_revision(self):
        payload = valid_result()
        payload["critical_errors"] = [
            {
                "type": "major_factual_error",
                "description": "Central definition is incorrect.",
                "evidence": "Quoted incorrect statement.",
            }
        ]
        self.assertEqual(derive_verdict(payload), "revise")


if __name__ == "__main__":
    unittest.main()
