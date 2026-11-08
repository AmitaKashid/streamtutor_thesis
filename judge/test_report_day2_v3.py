import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from judge.archive_evidence_v3 import create_archive, verify_archive
from judge.report_day2_v3 import (
    DEFAULT_DATASET_DIR, GENERATION, HARDWARE, _canonical_hash, build_report, frozen_plan, main,
)


class Day2ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset, cls.entries = frozen_plan(DEFAULT_DATASET_DIR)

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.run = self.root / "latency"
        self.run.mkdir()
        self.zip = self.root / "evidence.zip"

    def fixture(self, *, total_seconds=2, hardware=HARDWARE, free=3000, plan_tampered=False):
        plan_hash = _canonical_hash(self.entries)
        manifest = {
            "run_id": "test-run", "model_id": "prometheus_7b_v2_q4", "status": "completed",
            "git_commit": "commit", "slurm_job_id": "123", "protocol": "judge_execution_protocol_v3",
            "required_hardware": hardware, "required_backend": "llama-cpp-python 0.3.20",
            "required_quantization": "Q4_K_M", "required_server_configuration": GENERATION,
            "dataset_id": self.dataset["dataset_id"], "dataset_version": self.dataset["dataset_version"],
            "dataset_checksums_sha256": self.dataset["checksums_sha256"],
            "rubric_sha256": self.dataset["checked_files"]["rubric_v3.json"],
            "task_plan_sha256": plan_hash,
        }
        plan = {"entries": self.entries, "task_plan_sha256": plan_hash}
        origin = datetime(2026, 11, 7, tzinfo=timezone.utc)
        interval = total_seconds + 1
        result = {
            "status": "valid", "retry_used": False,
            "attempts": [{"metrics": {"wall_seconds": total_seconds},
                          "generation_error": None, "parsed_result": {"score": 4}}],
        }
        records = [
            {**entry, "sequence": seq, "started_utc": (origin + timedelta(seconds=seq * interval)).isoformat(),
             "evaluation_wall_seconds": total_seconds, "judge": result}
            for seq, entry in enumerate(self.entries, 1)
        ]
        if plan_tampered:
            records[-1]["task_id"] = records[-2]["task_id"]
        for name, value in (("manifest.json", manifest), ("plan.json", plan), ("summary.json", {})):
            (self.run / name).write_text(json.dumps(value), encoding="utf-8")
        (self.run / "attempts.csv").write_text("attempt\n", encoding="utf-8")
        (self.run / "evaluations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        paths = {}
        for role, value in (
            ("configuration", "frozen flags"), ("slurm-job", "123"),
            ("server-log", "load_tensors: offloaded 33/33 layers to GPU\n"),
        ):
            paths[role] = self.root / role
            paths[role].write_text(value, encoding="utf-8")
        paths["gpu-samples"] = self.root / "gpu.jsonl"
        with paths["gpu-samples"].open("w", encoding="utf-8") as handle:
            for second in range(0, len(records) * interval + total_seconds + 2):
                sample = {
                    "sampled_utc": (origin + timedelta(seconds=second)).isoformat(),
                    "slurm_job_id": "123", "hostname": "gpu1", "gpu_uuid": "GPU-123",
                    "gpu_name": "NVIDIA A30", "driver_version": "test", "total_mib": 24576,
                    "phase": "benchmark", "free_mib": free, "used_mib": 24576 - free, "utilization_pct": 80,
                }
                handle.write(json.dumps(sample) + "\n")
        create_archive(self.run, "latency", self.zip, paths.items())
        digest = verify_archive(self.zip)["archive_sha256"]
        return {digest: {
            "frozen_a30_configuration": {"value": True, "evidence": "allocation and configuration"},
            "no_unintended_cpu_tensor_offload": {"value": True, "evidence": "runtime trace"},
            "no_unrecovered_crash_or_hang": {"value": True, "evidence": "server log"},
            "median_decode_throughput": {"value": 10, "evidence": "per-request decode-only timings"},
        }}

    def reported(self, reviews=None):
        return next(row for row in build_report([self.zip], reviews=reviews)["candidates"] if row["run_id"])

    def test_no_evidence_outputs_only_not_verified(self):
        out = self.root / "report"
        self.assertEqual(main(["--output-dir", str(out)]), 0)
        result = json.loads((out / "day2_report.json").read_text(encoding="utf-8"))
        self.assertTrue(result["candidates"])
        self.assertTrue(all(row["decision"] == "not_verified" for row in result["candidates"]))
        self.assertIn("not_verified", (out / "day2_report.md").read_text(encoding="utf-8"))

    def test_unreviewed_and_missing_decode_do_not_pass(self):
        reviews = self.fixture()
        self.assertEqual(self.reported()["decision"], "not_verified")
        next(iter(reviews.values())).pop("median_decode_throughput")
        self.assertEqual(self.reported(reviews)["decision"], "not_verified")

    def test_complete_archive_with_all_checks_passes(self):
        report = self.reported(self.fixture(total_seconds=25, free=2048))
        self.assertEqual(report["decision"], "pass")
        self.assertEqual(report["gates"]["p95_completion_latency"]["value"], 25)

    def test_threshold_failure_is_not_treated_as_missing(self):
        report = self.reported(self.fixture(total_seconds=26, free=2047))
        self.assertEqual(report["decision"], "fail")
        self.assertEqual(report["gates"]["p95_completion_latency"]["status"], "fail")
        self.assertEqual(report["gates"]["minimum_free_vram"]["status"], "fail")

    def test_non_a30_and_modified_record_order_are_not_verified(self):
        for kwargs in ({"hardware": "NVIDIA A40 48 GB"}, {"plan_tampered": True}):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(self.reported(self.fixture(**kwargs))["decision"], "not_verified")
                self.zip.unlink()

    def test_duplicate_candidate_and_unbound_review_are_rejected(self):
        self.fixture()
        with self.assertRaisesRegex(ValueError, "multiple archives"):
            build_report([self.zip, self.zip])
        with self.assertRaisesRegex(ValueError, "not supplied"):
            build_report([self.zip], reviews={"0" * 64: {}})


if __name__ == "__main__":
    unittest.main()
