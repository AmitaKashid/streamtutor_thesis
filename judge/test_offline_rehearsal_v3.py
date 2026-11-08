"""Cross-module smoke tests: no model, network, GPU, or Slurm required."""

import csv
import io
import json
import tempfile
import unittest
from collections import defaultdict
from contextlib import ExitStack, redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from judge import analyze_judge_benchmark_v3 as analyzer
from judge import archive_evidence_v3 as archive
from judge import report_day2_v3 as report
from judge import run_controlled_contrasts_v3 as pairwise
from judge import run_judge_benchmark_v3 as scalar
from judge import run_latency_benchmark_v3 as latency
from judge import run_tutor_interference_v3 as interference
from judge import sample_gpu_resources_v3 as gpu

OFFLINE_URL = "http://127.0.0.1:1"
MODEL = "prometheus_7b_v2_q4"
COMMON = ["--base-url", OFFLINE_URL, "--model-id", MODEL]
A40 = ["--hardware", "NVIDIA A40 48 GB", "--quantization", "Q4_K_M"]


class MockProvider:
    def healthcheck(self):
        return True

    def generate(self, _prompt):
        return SimpleNamespace(
            text="Feedback: Synthetic test response. [RESULT] 4",
            total_seconds=2.0,
            prompt_tokens=400,
            new_tokens=20,
            tokens_per_second=10.0,
        )


class OfflineRehearsalTests(unittest.TestCase):
    def setUp(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.root = Path(workspace.name)
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        contexts.enter_context(redirect_stdout(io.StringIO()))
        contexts.enter_context(
            patch("socket.socket.connect", side_effect=AssertionError("Network forbidden"))
        )

    @staticmethod
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    def test_frozen_candidate_and_tutor_plans(self):
        hashes = defaultdict(set)
        seen = defaultdict(set)
        with latency.MATRIX.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            split, model = row["split"], row["model_id"]
            run_id = f"{model}-{split}"
            args = ["--model-id", model, "--run-id", run_id, "--dry-run",
                    "--output-root", str(self.root)]
            if split == "latency30":
                command = latency.main
                count_field = "measured_evaluations"
            else:
                args += ["--base-url", OFFLINE_URL, "--hardware", row["hardware"],
                         "--quantization", "Q4_K_M"]
                if split == "controlled_contrasts":
                    command, count_field = pairwise.main, "ordering_count"
                else:
                    command, count_field = scalar.main, "evaluation_count"
                    args += ["--split", split]
            with self.subTest(split=split, model=model):
                self.assertEqual(command(args), 0)
                plan = self.read(self.root / run_id / "plan.json")
                self.assertEqual(plan[count_field], int(row["task_count"]))
                hashes[split].add(plan["task_plan_sha256"])
                seen[split].add(model)
        self.assertEqual(set(seen), {
            "calibration", "selection", "robustness", "controlled_contrasts", "latency30",
        })
        self.assertTrue(all(models == set(report.candidate_ids()) for models in seen.values()))
        self.assertTrue(all(len(values) == 1 for values in hashes.values()))
        tutor_hashes = set()
        for mode in interference.MODES[:3]:
            self.assertEqual(interference.main([
                "run", "--mode", mode, "--run-id", mode, "--output-root", str(self.root), "--dry-run",
            ]), 0)
            plan = self.read(self.root / mode / "plan.json")
            self.assertEqual(len(plan["measured"]), 30)
            self.assertIn("warmup", plan)
            tutor_hashes.add(self.read(self.root / mode / "manifest.json")["plan_sha256"])
        self.assertEqual(len(tutor_hashes), 1)

    def test_scalar_calibration_to_analysis(self):
        with patch.object(scalar, "_make_remote_provider", return_value=MockProvider()):
            self.assertEqual(scalar.main([
                *COMMON, *A40, "--split", "calibration", "--run-id", "mock-calibration",
                "--output-root", str(self.root),
            ]), 0)
        run = self.root / "mock-calibration"
        self.assertEqual(analyzer.main([
            "--run-dir", str(run), "--reference-source", "provisional_ai",
            "--acknowledge-provisional-not-human", "--bootstrap-iterations", "0",
        ]), 0)
        result = self.read(run / "analysis_provisional_ai" / "analysis_report.json")
        self.assertTrue(result["run_complete"])
        self.assertFalse(result["reference"]["confirmatory_human_ground_truth"])

    def test_latency_to_gpu_summary_to_archive_to_day2_report(self):
        with patch.object(latency, "_make_provider", return_value=MockProvider()):
            self.assertEqual(latency.main([
                *COMMON, "--slurm-job-id", "offline-fixture", "--run-id", "mock-latency",
                "--output-root", str(self.root),
            ]), 0)
        run = self.root / "mock-latency"
        rows = scalar._read_jsonl(run / "evaluations.jsonl")
        measured = [row for row in rows if row["phase"] == "measured"]
        self.assertEqual(len(rows), 95)
        start = datetime.fromisoformat(measured[0]["started_utc"])
        end = max(
            datetime.fromisoformat(row["started_utc"]) + timedelta(seconds=row["evaluation_wall_seconds"])
            for row in measured
        )
        samples = []
        cursor = start - timedelta(milliseconds=100)
        while cursor <= end + timedelta(milliseconds=600):
            samples.append({
                "sampled_utc": cursor.isoformat(), "slurm_job_id": "offline-fixture",
                "hostname": "mock-host", "gpu_uuid": "GPU-fixture", "gpu_name": "NVIDIA A30",
                "driver_version": "mock", "total_mib": 24576, "used_mib": 21000, "free_mib": 3000,
                "phase": "benchmark", "utilization_pct": 80,
            })
            cursor += timedelta(milliseconds=500)
        # Include an in-window sample even when the mocked run takes less than one interval.
        samples.append({**samples[0], "sampled_utc": start.isoformat()})
        samples.sort(key=lambda row: row["sampled_utc"])
        source = self.root / "gpu.jsonl"
        source.write_text("".join(json.dumps(row) + "\n" for row in samples), encoding="utf-8")
        self.assertEqual(gpu.main([
            "summarize", "--run-dir", str(run), "--samples", str(source),
        ]), 0)
        evidence = self.root / "offline-fixture.zip"
        self.assertEqual(archive.main([
            "create", "--stage", "latency", "--run-dir", str(run),
            "--output", str(evidence), "--attach", "gpu-samples", str(source),
        ]), 0)
        self.assertEqual(archive.main(["verify", str(evidence)]), 0)
        out = self.root / "day2"
        self.assertEqual(report.main(["--archive", str(evidence), "--output-dir", str(out)]), 0)
        result = self.read(out / "day2_report.json")
        self.assertTrue(all(row["decision"] == "not_verified" for row in result["candidates"]))
        matching = next(row for row in result["candidates"] if row["model_id"] == MODEL)
        self.assertEqual(matching["run_id"], "mock-latency")


if __name__ == "__main__":
    unittest.main()
