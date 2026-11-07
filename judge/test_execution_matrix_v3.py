from __future__ import annotations

import csv
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX = (
    ROOT
    / "docs"
    / "judge_evaluation"
    / "execution_matrix_v3.csv"
)


class ExecutionMatrixV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with MATRIX.open(encoding="utf-8", newline="") as handle:
            cls.rows = list(csv.DictReader(handle))

    def test_every_stage_contains_exact_frozen_candidate_set(self) -> None:
        expected = {
            "glm4_32b_0414_q4",
            "fare_20b_q4",
            "compassjudger_2_7b_q4",
            "prometheus_7b_v2_q4",
        }
        by_stage: dict[str, set[str]] = defaultdict(set)
        for row in self.rows:
            by_stage[row["stage"]].add(row["model_id"])
        self.assertEqual(
            set(by_stage),
            {
                "quality_calibration",
                "quality_selection",
                "robustness",
                "controlled_contrasts",
                "comparative_latency",
            },
        )
        self.assertTrue(
            all(candidates == expected for candidates in by_stage.values())
        )

    def test_v3_generation_configuration_is_identical(self) -> None:
        configurations = {
            (
                row["n_ctx"],
                row["max_new_tokens"],
                row["temperature"],
                row["top_p"],
                row["seed"],
                row["system_prompt"],
            )
            for row in self.rows
        }
        self.assertEqual(
            configurations,
            {("4096", "256", "0.0", "1.0", "0", "disabled")},
        )

    def test_task_counts_and_repetitions_are_frozen(self) -> None:
        expected = {
            "quality_calibration": (132, 1),
            "quality_selection": (792, 1),
            "robustness": (123, 1),
            "controlled_contrasts": (28, 1),
            "comparative_latency": (90, 3),
        }
        for stage, (task_count, repetitions) in expected.items():
            rows = [row for row in self.rows if row["stage"] == stage]
            self.assertEqual(
                Counter(int(row["task_count"]) for row in rows),
                Counter({task_count: 4}),
            )
            self.assertTrue(
                all(int(row["repetitions"]) == repetitions for row in rows)
            )

    def test_latency_stage_uses_a30_and_five_warmups(self) -> None:
        rows = [
            row
            for row in self.rows
            if row["stage"] == "comparative_latency"
        ]
        self.assertTrue(
            all(row["hardware"] == "NVIDIA A30 24 GB" for row in rows)
        )
        self.assertTrue(all(int(row["warmups"]) == 5 for row in rows))


if __name__ == "__main__":
    unittest.main()
