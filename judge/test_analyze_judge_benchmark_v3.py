from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from judge.analyze_judge_benchmark_v3 import (
    PROVISIONAL_REFERENCE,
    _provisional_answer_score,
    _provisional_retrieval_score,
    agreement_metrics,
    analyze_run,
    build_scored_rows,
    load_human_annotations,
    quadratic_weighted_kappa,
    quality_level_discrimination,
)
from judge.run_judge_benchmark_v3 import (
    DEFAULT_DATASET_DIR,
    _read_jsonl,
    build_evaluation_tasks,
    verify_dataset,
)


class JudgeBenchmarkAnalysisV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset_dir = Path(DEFAULT_DATASET_DIR)
        cls.responses = _read_jsonl(cls.dataset_dir / "responses.jsonl")
        cls.questions = _read_jsonl(cls.dataset_dir / "questions.jsonl")
        cls.response_by_id = {
            row["response_id"]: row for row in cls.responses
        }
        cls.responses_by_question: dict[str, list[dict]] = {}
        for row in cls.responses:
            cls.responses_by_question.setdefault(
                row["question_id"], []
            ).append(row)

    def _prediction(
        self,
        *,
        evaluation_id: str,
        question_id: str,
        response_id: str | None,
        dimension: str,
        score: int | None,
        status: str = "valid",
        retry_used: bool = False,
    ) -> dict:
        return {
            "evaluation_id": evaluation_id,
            "question_id": question_id,
            "response_id": response_id,
            "dimension": dimension,
            "judge": {
                "status": status,
                "result": (
                    {
                        "score": score,
                        "feedback": "Evidence-based feedback.",
                    }
                    if score is not None
                    else None
                ),
                "attempt_count": 2 if retry_used else 1,
                "retry_used": retry_used,
                "attempts": [],
            },
        }

    def _perfect_calibration_predictions(self) -> list[dict]:
        tasks = build_evaluation_tasks(
            questions=self.questions,
            responses=self.responses,
            split="calibration",
        )
        predictions = []
        for task in tasks:
            if task.dimension == "retrieval_relevance":
                score = _provisional_retrieval_score(
                    task.question_id,
                    self.responses_by_question[task.question_id],
                )
            else:
                score = _provisional_answer_score(
                    self.response_by_id[str(task.response_id)],
                    task.dimension,
                )
            predictions.append(
                self._prediction(
                    evaluation_id=task.evaluation_id,
                    question_id=task.question_id,
                    response_id=task.response_id,
                    dimension=task.dimension,
                    score=score,
                )
            )
        return predictions

    def _write_complete_run(
        self,
        root: Path,
        predictions: list[dict],
    ) -> Path:
        identity = verify_dataset(self.dataset_dir)
        run_dir = root / "run"
        run_dir.mkdir()
        manifest = {
            "run_id": "test-calibration-run",
            "status": "completed",
            "split": "calibration",
            "planned_evaluation_count": len(predictions),
            "judge_configuration": {"model_id": "test-model"},
            "dataset": {
                "id": identity["dataset_id"],
                "version": identity["dataset_version"],
                "checksums_sha256": identity["checksums_sha256"],
                "responses_sha256": identity["checked_files"][
                    "responses.jsonl"
                ],
                "questions_sha256": identity["checked_files"][
                    "questions.jsonl"
                ],
                "rubric_sha256": identity["checked_files"][
                    "rubric_v3.json"
                ],
            },
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        with (run_dir / "predictions.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for prediction in predictions:
                handle.write(json.dumps(prediction) + "\n")
        return run_dir

    def test_quadratic_weighted_kappa_boundaries(self) -> None:
        self.assertEqual(
            quadratic_weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
            1.0,
        )
        self.assertAlmostEqual(
            quadratic_weighted_kappa([1, 2, 4, 5], [5, 4, 2, 1]),
            -1.0,
        )
        self.assertIsNone(quadratic_weighted_kappa([], []))

    def test_build_scored_rows_keeps_na_groundedness_unpaired(self) -> None:
        response = next(
            row
            for row in self.responses
            if row["retrieval"]["condition"] in {"missing", "irrelevant"}
        )
        prediction = self._prediction(
            evaluation_id=response["response_id"] + "::groundedness",
            question_id=response["question_id"],
            response_id=response["response_id"],
            dimension="groundedness",
            score=1,
        )
        rows = build_scored_rows(
            predictions=[prediction],
            responses=self.responses,
            reference_source=PROVISIONAL_REFERENCE,
        )
        self.assertIsNone(rows[0]["reference_score"])
        metrics = agreement_metrics(
            rows,
            bootstrap_iterations=10,
            bootstrap_seed=1,
        )
        self.assertEqual(metrics["reference_applicable_count"], 0)
        self.assertIsNone(metrics["quadratic_weighted_kappa"])

    def test_invalid_prediction_reduces_coverage_without_score_repair(self) -> None:
        response = self.responses[0]
        prediction = self._prediction(
            evaluation_id=response["response_id"] + "::correctness",
            question_id=response["question_id"],
            response_id=response["response_id"],
            dimension="correctness",
            score=None,
            status="invalid",
        )
        rows = build_scored_rows(
            predictions=[prediction],
            responses=self.responses,
            reference_source=PROVISIONAL_REFERENCE,
        )
        metrics = agreement_metrics(
            rows,
            bootstrap_iterations=10,
            bootstrap_seed=1,
        )
        self.assertEqual(metrics["reference_applicable_count"], 1)
        self.assertEqual(metrics["paired_valid_count"], 0)
        self.assertEqual(metrics["coverage_of_applicable_reference"], 0.0)

    def test_duplicate_predictions_are_rejected(self) -> None:
        response = self.responses[0]
        prediction = self._prediction(
            evaluation_id="duplicate",
            question_id=response["question_id"],
            response_id=response["response_id"],
            dimension="correctness",
            score=5,
        )
        with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
            build_scored_rows(
                predictions=[prediction, prediction],
                responses=self.responses,
                reference_source=PROVISIONAL_REFERENCE,
            )

    def test_human_annotation_loader_requires_selected_rater(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "human.csv"
            fields = [
                "response_id",
                "rater_id",
                "correctness",
                "groundedness_or_na",
                "completeness",
                "clarity",
                "pedagogy",
                "learner_appropriateness",
                "conversational_context",
                "retrieval_relevance",
            ]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "response_id": "response-1",
                        "rater_id": "rater-a",
                        "correctness": "5",
                        "groundedness_or_na": "N/A",
                        "completeness": "4",
                        "clarity": "4",
                        "pedagogy": "3",
                        "learner_appropriateness": "5",
                        "conversational_context": "5",
                        "retrieval_relevance": "2",
                    }
                )
            loaded = load_human_annotations(path, rater_id="rater-a")
            self.assertIsNone(loaded["response-1"]["groundedness"])
            self.assertEqual(loaded["response-1"]["correctness"], 5)
            with self.assertRaisesRegex(ValueError, "No annotations"):
                load_human_annotations(path, rater_id="missing")

    def test_quality_discrimination_requires_strict_order(self) -> None:
        rows = []
        for level, score in (
            ("excellent", 5),
            ("borderline", 3),
            ("critical", 1),
        ):
            rows.append(
                {
                    "question_id": "q1",
                    "response_id": f"q1-{level}",
                    "dimension": "correctness",
                    "prediction_score": score,
                    "reference_score": score,
                    "controlled_quality_level": level,
                }
            )
        result = quality_level_discrimination(rows)
        self.assertEqual(result["complete_question_triplets"], 1)
        self.assertEqual(result["strict_order_accuracy"], 1.0)

    def test_provisional_analysis_requires_explicit_acknowledgement(self) -> None:
        predictions = self._perfect_calibration_predictions()
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._write_complete_run(Path(directory), predictions)
            with self.assertRaisesRegex(ValueError, "not human ground truth"):
                analyze_run(
                    run_dir=run_dir,
                    dataset_dir=self.dataset_dir,
                    reference_source=PROVISIONAL_REFERENCE,
                )

    def test_complete_perfect_run_reports_perfect_agreement(self) -> None:
        predictions = self._perfect_calibration_predictions()
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._write_complete_run(Path(directory), predictions)
            report, scored = analyze_run(
                run_dir=run_dir,
                dataset_dir=self.dataset_dir,
                reference_source=PROVISIONAL_REFERENCE,
                acknowledge_provisional_not_human=True,
                bootstrap_iterations=50,
                bootstrap_seed=7,
            )
        self.assertEqual(len(scored), 132)
        self.assertEqual(report["macro_answer_dimension_qwk"]["value"], 1.0)
        self.assertEqual(
            report["macro_answer_dimension_qwk"]["qwk_95_ci"]["lower_95"],
            1.0,
        )
        self.assertEqual(
            report["agreement_by_dimension"]["correctness"][
                "quadratic_weighted_kappa"
            ],
            1.0,
        )
        self.assertFalse(report["reference"]["confirmatory_human_ground_truth"])
        self.assertEqual(
            report["critical_error_evaluation"]["status"],
            "not_measured",
        )

    def test_incomplete_run_is_rejected_by_default(self) -> None:
        predictions = self._perfect_calibration_predictions()[:1]
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._write_complete_run(Path(directory), predictions)
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["status"] = "interrupted"
            manifest["planned_evaluation_count"] = 132
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                analyze_run(
                    run_dir=run_dir,
                    dataset_dir=self.dataset_dir,
                    reference_source=PROVISIONAL_REFERENCE,
                    acknowledge_provisional_not_human=True,
                )


if __name__ == "__main__":
    unittest.main()
