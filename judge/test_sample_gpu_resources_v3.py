import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from judge.sample_gpu_resources_v3 import collect, offload_evidence, read_gpu, summarize


def stamp(seconds):
    return (datetime(2026, 11, 7, tzinfo=timezone.utc) + timedelta(seconds=seconds)).isoformat()


def sample(seconds, *, free=3000, phase="benchmark", job="42"):
    return {
        "sampled_utc": stamp(seconds), "slurm_job_id": job, "hostname": "gpu-node",
        "gpu_uuid": "GPU-abc", "gpu_name": "NVIDIA A30", "driver_version": "test",
        "total_mib": 24576, "used_mib": 24576 - free, "free_mib": free,
        "utilization_pct": 70, "phase": phase, "interval_ms": 500,
    }


MANIFEST = {
    "status": "completed", "required_hardware": "NVIDIA A30 24 GB",
    "slurm_job_id": "42", "run_id": "test",
}
EVALUATIONS = [
    {
        "phase": "measured", "task_id": str(index % 30), "repetition": index // 30 + 1,
        "started_utc": stamp(1 + index / 100), "evaluation_wall_seconds": 0.1,
    }
    for index in range(90)
]


class GpuResourceTests(unittest.TestCase):
    @patch("judge.sample_gpu_resources_v3.subprocess.run")
    def test_explicit_uuid_query_and_memory_fields(self, run):
        run.return_value = SimpleNamespace(stdout="GPU-abc, NVIDIA A30, driver, 24576, 20000, 4000, 77\n")
        row = read_gpu("GPU-abc")
        self.assertEqual(row["free_mib"], 4000)
        self.assertEqual(row["utilization_pct"], 77)
        self.assertEqual(run.call_args.args[0][1:3], ["-i", "GPU-abc"])
        with self.assertRaisesRegex(ValueError, "requested GPU UUID"):
            read_gpu("GPU-def")

    def test_collect_requires_allocation_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "gpu.jsonl"
            args = SimpleNamespace(gpu_uuid="GPU-abc", output=path, phase="loaded-idle", samples=1, interval_ms=500)
            with patch.dict("os.environ", {}, clear=True):
                with self.assertRaisesRegex(ValueError, "GPU allocation"):
                    collect(args)
            with patch.dict("os.environ", {"SLURM_JOB_ID": "42", "CUDA_VISIBLE_DEVICES": "0"}):
                with patch("judge.sample_gpu_resources_v3.read_gpu", return_value=sample(0)):
                    self.assertEqual(collect(args), 0)
                    self.assertIn('"phase": "loaded-idle"', path.read_text(encoding="utf-8"))
                    with self.assertRaises(FileExistsError):
                        collect(args)

    def test_scope_peak_headroom_and_loaded_idle_separately(self):
        rows = [sample(0.5, free=500), sample(1.2, free=2400), sample(1.8, free=1900), sample(2.0, free=500)]
        result = summarize(MANIFEST, EVALUATIONS, rows, {}, [sample(0, free=8000, phase="loaded-idle")])
        self.assertTrue(result["coverage_ok"])
        self.assertEqual(result["window_sample_count"], 2)
        self.assertEqual(result["sampled_minimum_free_mib"], 1900)
        self.assertEqual(result["sampled_peak_used_mib"], 22676)
        self.assertEqual(result["loaded_idle_used_mib_median"], 16576)
        self.assertEqual(result["observed_headroom"], "below_threshold")
        self.assertFalse(result["deployment_gate_evaluated"])

    def test_missing_bracket_or_large_gap_never_verifies_headroom(self):
        for rows in ([sample(1.5), sample(2.0)], [sample(0), sample(1.2), sample(4.0)]):
            with self.subTest(rows=rows):
                result = summarize(MANIFEST, EVALUATIONS, rows, {})
                self.assertFalse(result["coverage_ok"])
                self.assertEqual(result["observed_headroom"], "not_verified")

    def test_mixed_jobs_or_duplicate_workload_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "one GPU and Slurm allocation"):
            summarize(MANIFEST, EVALUATIONS, [sample(0), sample(1, job="43")], {})
        with self.assertRaisesRegex(ValueError, "three complete repetitions"):
            summarize(MANIFEST, EVALUATIONS[:-1] + EVALUATIONS[:1], [sample(0)], {})

    def test_layer_offload_does_not_attest_cpu_tensor_placement(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "server.log"
            path.write_text("load_tensors: offloaded 33/33 layers to GPU\nCPU_Mapped model buffer size = 70 MiB\n")
            result = offload_evidence(path)
            self.assertEqual(result["layer_offload"], "all_layers_reported_on_gpu")
            self.assertEqual(result["unintended_cpu_tensor_offload"], "not_verified")
            self.assertEqual(len(result["matching_log_lines"]), 2)
            path.write_text("load_tensors: offloaded 31/33 layers to GPU\n")
            self.assertEqual(offload_evidence(path)["layer_offload"], "partial_layer_offload")
            path.write_text("offloaded 31/33 layers to GPU\noffloaded 33/33 layers to GPU\n")
            self.assertEqual(offload_evidence(path)["layer_offload"], "unknown")


if __name__ == "__main__":
    unittest.main()
