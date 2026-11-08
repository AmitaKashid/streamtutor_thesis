from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from judge.run_judge_benchmark_v3 import (
    DEFAULT_DATASET_DIR,
    _read_jsonl,
    build_evaluation_tasks,
)
from judge.run_latency_benchmark_v3 import build_latency_plan, main


class FakeProvider:
    def __init__(self, *, fail_warmup=False):
        self.fail_warmup = fail_warmup
        self.prompts = []

    def healthcheck(self):
        return True

    def generate(self, prompt):
        self.prompts.append(prompt)
        index = len(self.prompts)
        if self.fail_warmup or index == 8:
            raise TimeoutError("simulated network timeout")
        answer = (
            "not in contract format" if index <= 6
            else "Feedback: The explanation is sound. [RESULT] 4"
        )
        return SimpleNamespace(
            text=answer, total_seconds=2.0, prompt_tokens=400,
            new_tokens=20, tokens_per_second=10.0,
        )


class LatencyBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = build_evaluation_tasks(
            questions=_read_jsonl(DEFAULT_DATASET_DIR / "questions.jsonl"),
            responses=_read_jsonl(DEFAULT_DATASET_DIR / "responses.jsonl"),
            split="calibration",
        )

    def test_frozen_selection_is_deterministic_stratified_and_disjoint(self):
        warmups, selected = build_latency_plan(self.tasks)
        self.assertEqual((warmups, selected), build_latency_plan(reversed(self.tasks)))
        self.assertEqual(len(warmups), 5)
        self.assertEqual(len(selected), 30)
        self.assertEqual(Counter(t.dimension for t in selected)["retrieval_relevance"], 2)
        self.assertEqual(sorted(Counter(t.dimension for t in selected).values()), [2] + [4] * 7)
        self.assertFalse({t.evaluation_id for t in warmups} & {t.evaluation_id for t in selected})

    @staticmethod
    def _args(folder):
        return [
            "--run-id", "sample", "--model-id", "prometheus_7b_v2_q4",
            "--output-root", str(folder),
        ]

    @staticmethod
    def _json(folder, file):
        return json.loads((Path(folder) / "sample" / file).read_text(encoding="utf-8"))

    def test_dry_run_writes_plan_without_service_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            args = self._args(folder) + ["--dry-run"]
            self.assertEqual(main(args), 0)
            plan = self._json(folder, "plan.json")
            self.assertEqual(len(plan["entries"]), 95)
            self.assertEqual(plan["measured_evaluations"], 90)
            self.assertEqual(self._json(folder, "manifest.json")["status"], "dry_run_complete")
            with self.assertRaises(FileExistsError):
                main(args)

    def test_measured_requests_keep_retry_timeout_and_raw_records(self):
        provider = FakeProvider()
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()), patch(
            "judge.run_latency_benchmark_v3._make_provider", return_value=provider
        ):
            self.assertEqual(main(self._args(folder) + [
                "--base-url", "http://localhost:9999", "--slurm-job-id", "1234",
            ]), 0)
            root = Path(folder) / "sample"
            rows = _read_jsonl(root / "evaluations.jsonl")
            self.assertEqual(len(rows), 95)
            self.assertTrue(all(len(r["judge"]["attempts"]) == 1 for r in rows[:5]))
            self.assertEqual(rows[5]["judge"]["attempts"][0]["raw_output"], "not in contract format")
            self.assertEqual(
                [r["task_id"] for r in rows[5:35]],
                [r["task_id"] for r in rows[35:65]],
            )
            self.assertEqual(
                [r["task_id"] for r in rows[5:35]],
                [r["task_id"] for r in rows[65:95]],
            )
            self.assertEqual(len(provider.prompts), 96)
            self.assertEqual(provider.prompts[5], provider.prompts[36])
            self.assertIn("RETRY", provider.prompts[6].upper())
            report = self._json(folder, "summary.json")
            self.assertEqual(report["retries"], 1)
            self.assertEqual(report["first_attempt_valid_count"], 88)
            self.assertEqual(report["final_valid_count"], 89)
            self.assertEqual(report["timeouts"], 1)
            self.assertEqual(report["failed_requests"], 1)
            self.assertEqual(report["attempt_wall_seconds"]["count"], 91)
            self.assertEqual(report["evaluation_wall_seconds_including_retry"]["count"], 90)
            self.assertIsNone(report["median_decode_tokens_per_second"])
            self.assertIsNone(report["ttft_seconds"])
            self.assertFalse(report["deployment_gate_evaluated"])
            with (root / "attempts.csv").open(encoding="utf-8", newline="") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 96)

    def test_failed_warmup_preserves_incomplete_evidence(self):
        provider = FakeProvider(fail_warmup=True)
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()), patch(
            "judge.run_latency_benchmark_v3._make_provider", return_value=provider
        ):
            self.assertEqual(main(self._args(folder) + [
                "--base-url", "http://localhost:9999", "--slurm-job-id", "1234",
            ]), 2)
            report = self._json(folder, "summary.json")
            self.assertEqual(report["warmups_recorded"], 1)
            self.assertEqual(report["measured_evaluations"], 0)
            self.assertFalse(report["measurement_complete"])
            self.assertEqual(self._json(folder, "manifest.json")["status"], "runner_error")


if __name__ == "__main__":
    unittest.main()
