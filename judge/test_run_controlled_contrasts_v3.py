from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from judge.run_controlled_contrasts_v3 import (
    PairwiseJudgeV3,
    _read_jsonl,
    build_contrast_tasks,
    build_pairwise_prompt,
    main,
    summarize_contrasts,
    validate_pairwise_records,
)
from judge.run_judge_benchmark_v3 import DEFAULT_DATASET_DIR
from judge.scalar_evaluator_v3 import load_frozen_rubric
from llm.llmcli.providers.base import GenerationResult


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


def _generation(text: str, *, raw_text: str | None = None) -> GenerationResult:
    return GenerationResult(
        text=text,
        raw_text=raw_text,
        response_protocol="harmony" if raw_text else None,
        total_seconds=0.2,
        prompt_tokens=100,
        new_tokens=20,
        tokens_per_second=100.0,
    )


class ControlledContrastsV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset_dir = Path(DEFAULT_DATASET_DIR)
        cls.contrasts = _read_jsonl(
            cls.dataset_dir / "controlled_contrasts.jsonl"
        )
        cls.tasks = build_contrast_tasks(cls.contrasts)
        cls.rubric = load_frozen_rubric(
            cls.dataset_dir / "rubric_v3.json"
        )

    def _evaluator(
        self,
        path: Path,
        provider: FakeProvider,
    ) -> PairwiseJudgeV3:
        return PairwiseJudgeV3(
            provider,
            model_id="candidate",
            quantization="Q4_K_M",
            hardware="test-gpu",
            output_path=path,
            dataset_id="streamtutor-llm-judge-evaluation-v3",
            dataset_version="3.0.0",
            rubric_path=self.dataset_dir / "rubric_v3.json",
            declared_generation_config={"temperature": 0.0},
        )

    @staticmethod
    def _record(task, selected_source: str) -> dict:
        choice = 1 if task.candidate_1_source == selected_source else 2
        return {
            "evaluation_id": task.evaluation_id,
            "contrast_id": task.contrast_id,
            "question_id": task.question_id,
            "target_dimension": task.target_dimension,
            "order": task.order,
            "candidate_1_source": task.candidate_1_source,
            "candidate_2_source": task.candidate_2_source,
            "judge": {
                "status": "valid",
                "result": {"choice": choice},
                "retry_used": False,
                "attempts": [],
            },
        }

    def test_frozen_dataset_creates_both_orders_for_every_contrast(self) -> None:
        self.assertEqual(len(self.contrasts), 14)
        self.assertEqual(len(self.tasks), 28)
        by_contrast: dict[str, set[str]] = {}
        for task in self.tasks:
            by_contrast.setdefault(task.contrast_id, set()).add(task.order)
        self.assertTrue(
            all(orders == {"ab", "ba"} for orders in by_contrast.values())
        )

    def test_task_order_is_deterministic_and_contains_no_labels(self) -> None:
        reversed_tasks = build_contrast_tasks(list(reversed(self.contrasts)))
        self.assertEqual(
            [task.evaluation_id for task in self.tasks],
            [task.evaluation_id for task in reversed_tasks],
        )
        serialized = json.dumps(
            [task.__dict__ for task in self.tasks], ensure_ascii=False
        )
        for forbidden in (
            "preferred_answer",
            "expected_reason",
            "control_statement",
            "annotation_status",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_prompt_uses_rubric_and_never_leaks_expected_preference(self) -> None:
        row = self.contrasts[0]
        task = self.tasks[0]
        prompt = build_pairwise_prompt(
            rubric=self.rubric,
            target_dimension=task.target_dimension,
            conversation_history=task.conversation_history,
            user_utterance=task.user_utterance,
            reference_answer=task.reference_answer,
            context=task.context,
            candidate_1=task.candidate_1,
            candidate_2=task.candidate_2,
        )
        self.assertIn(
            self.rubric.anchors_for(task.target_dimension)[5], prompt
        )
        self.assertIn(task.user_utterance, prompt)
        self.assertNotIn(row["expected_reason"], prompt)
        self.assertNotIn(row["control_statement"], prompt)
        self.assertNotIn("preferred_answer", prompt)

    def test_summary_requires_position_consistency(self) -> None:
        first_id = self.tasks[0].contrast_id
        first_tasks = [t for t in self.tasks if t.contrast_id == first_id]
        expected = next(
            row["preferred_answer"]
            for row in self.contrasts
            if row["contrast_id"] == first_id
        )
        records = [self._record(task, expected) for task in first_tasks]

        second_id = next(
            task.contrast_id
            for task in self.tasks
            if task.contrast_id != first_id
        )
        second_tasks = [t for t in self.tasks if t.contrast_id == second_id]
        records.extend(
            [
                self._record(second_tasks[0], "a"),
                self._record(second_tasks[1], "b"),
            ]
        )
        summary = summarize_contrasts(records, self.contrasts)
        self.assertEqual(summary["complete_valid_pairs"], 2)
        self.assertEqual(summary["position_consistent_pairs"], 1)
        self.assertEqual(
            summary["provisional_expected_preference_accuracy"], 1.0
        )
        self.assertIn("not independent human", summary["reference_warning"])

    def test_record_validation_rejects_unknown_or_tampered_tasks(self) -> None:
        task = self.tasks[0]
        record = self._record(task, "a")
        result = validate_pairwise_records([record], self.tasks)
        self.assertFalse(result["complete"])
        tampered = dict(record)
        tampered["order"] = "ba"
        with self.assertRaisesRegex(ValueError, "mismatched order"):
            validate_pairwise_records([tampered], self.tasks)
        unknown = dict(record)
        unknown["evaluation_id"] = "not-in-plan"
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_pairwise_records([unknown], self.tasks)

    def test_evaluator_retries_and_preserves_raw_transport(self) -> None:
        provider = FakeProvider(
            _generation("invalid"),
            _generation(
                "Feedback: Candidate 1 is better.\n[RESULT] 1",
                raw_text=(
                    "<|channel|>final<|message|>Feedback: Candidate 1 "
                    "is better.\n[RESULT] 1"
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            record = self._evaluator(path, provider).evaluate_and_log(
                self.tasks[0]
            )
            self.assertEqual(record["judge"]["status"], "valid")
            self.assertTrue(record["judge"]["retry_used"])
            self.assertEqual(record["judge"]["attempt_count"], 2)
            self.assertTrue(
                record["judge"]["attempts"][1]["raw_output"].startswith(
                    "<|channel|>"
                )
            )
            self.assertEqual(record["dataset_version"], "3.0.0")
            self.assertEqual(len(record["input_sha256"]), 64)

    def test_dry_run_writes_exact_frozen_plan_without_server(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            code = main(
                [
                    "--dataset-dir",
                    str(self.dataset_dir),
                    "--base-url",
                    "http://127.0.0.1:9",
                    "--model-id",
                    "candidate",
                    "--quantization",
                    "Q4_K_M",
                    "--hardware",
                    "test-gpu",
                    "--output-root",
                    directory,
                    "--run-id",
                    "contrast-dry-run",
                    "--dry-run",
                ]
            )
            self.assertEqual(code, 0)
            plan = json.loads(
                (Path(directory) / "contrast-dry-run" / "plan.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(plan["contrast_count"], 14)
            self.assertEqual(plan["ordering_count"], 28)


if __name__ == "__main__":
    unittest.main()
