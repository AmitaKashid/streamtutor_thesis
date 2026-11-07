from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from judge.run_judge_benchmark_v3 import (
    DEFAULT_DATASET_DIR,
    FORBIDDEN_LABEL_KEYS,
    _read_jsonl,
    build_evaluation_tasks,
    prepare_run,
    summarize_predictions,
    verify_dataset,
)


class JudgeBenchmarkV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset_dir = Path(DEFAULT_DATASET_DIR)
        cls.questions = _read_jsonl(cls.dataset_dir / "questions.jsonl")
        cls.responses = _read_jsonl(cls.dataset_dir / "responses.jsonl")

    def test_frozen_dataset_and_task_counts(self) -> None:
        identity = verify_dataset(self.dataset_dir)
        self.assertEqual(
            identity["dataset_id"],
            "streamtutor-llm-judge-evaluation-v3",
        )
        tasks = build_evaluation_tasks(
            questions=self.questions,
            responses=self.responses,
            split="selection",
        )
        counts = Counter(task.dimension for task in tasks)
        self.assertEqual(len(tasks), 792)
        self.assertEqual(counts["correctness"], 108)
        self.assertEqual(counts["retrieval_relevance"], 36)

    def test_retrieval_is_once_per_question_and_has_no_answer(self) -> None:
        tasks = build_evaluation_tasks(
            questions=self.questions,
            responses=self.responses,
            split="calibration",
        )
        retrieval = [
            task
            for task in tasks
            if task.dimension == "retrieval_relevance"
        ]
        self.assertEqual(len(retrieval), 6)
        self.assertEqual(
            len({task.question_id for task in retrieval}),
            len(retrieval),
        )
        self.assertTrue(all(task.answer is None for task in retrieval))
        self.assertTrue(all(task.response_id is None for task in retrieval))

    def test_unusable_retrieval_skips_only_groundedness(self) -> None:
        tasks = build_evaluation_tasks(
            questions=self.questions,
            responses=self.responses,
            split="robustness",
        )
        counts = Counter(task.dimension for task in tasks)
        self.assertEqual(counts["correctness"], 18)
        self.assertEqual(counts["groundedness"], 9)
        self.assertEqual(counts["retrieval_relevance"], 6)

    def test_tasks_are_deterministic_and_exclude_labels(self) -> None:
        first = build_evaluation_tasks(
            questions=self.questions,
            responses=self.responses,
            split="calibration",
        )
        second = build_evaluation_tasks(
            questions=list(reversed(self.questions)),
            responses=list(reversed(self.responses)),
            split="calibration",
        )
        self.assertEqual(
            [task.evaluation_id for task in first],
            [task.evaluation_id for task in second],
        )
        serialized = json.dumps(
            [task.evaluator_kwargs() for task in first]
        )
        for forbidden in FORBIDDEN_LABEL_KEYS:
            self.assertNotIn(forbidden, serialized)

    def test_selection_split_cannot_be_truncated(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden"):
            build_evaluation_tasks(
                questions=self.questions,
                responses=self.responses,
                split="selection",
                max_responses=1,
            )

    def test_resume_requires_identical_immutable_configuration(self) -> None:
        manifest = {
            "runner_version": "3.0",
            "dataset": {"id": "d", "version": "3"},
            "split": "calibration",
            "judge_configuration": {"model_id": "m"},
            "task_plan_sha256": "abc",
            "planned_evaluation_count": 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            prepare_run(
                run_dir=run_dir,
                manifest=dict(manifest),
                resume=False,
            )
            resumed = prepare_run(
                run_dir=run_dir,
                manifest=dict(manifest),
                resume=True,
            )
            self.assertEqual(resumed["status"], "running")
            changed = dict(manifest)
            changed["split"] = "robustness"
            with self.assertRaisesRegex(ValueError, "does not match"):
                prepare_run(
                    run_dir=run_dir,
                    manifest=changed,
                    resume=True,
                )

    def test_checksum_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "dataset"
            shutil.copytree(self.dataset_dir, copied)
            with (copied / "questions.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                verify_dataset(copied)

    def test_summary_reports_retries_latency_and_status(self) -> None:
        records = [
            {
                "dimension": "correctness",
                "judge": {
                    "status": "valid",
                    "retry_used": True,
                    "attempts": [
                        {
                            "metrics": {
                                "wall_seconds": 1.0,
                                "tokens_per_second": 20.0,
                            }
                        },
                        {
                            "metrics": {
                                "wall_seconds": 3.0,
                                "tokens_per_second": 40.0,
                            }
                        },
                    ],
                },
            },
            {
                "dimension": "clarity",
                "judge": {
                    "status": "invalid",
                    "retry_used": False,
                    "attempts": [],
                },
            },
        ]
        summary = summarize_predictions(records)
        self.assertEqual(summary["status_counts"], {"invalid": 1, "valid": 1})
        self.assertEqual(summary["retry_count"], 1)
        self.assertEqual(summary["attempt_latency_seconds"]["p50"], 2.0)
        self.assertEqual(summary["median_tokens_per_second"], 30.0)


if __name__ == "__main__":
    unittest.main()
