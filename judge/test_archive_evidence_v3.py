import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from judge.archive_evidence_v3 import STAGES, create_archive, verify_archive


class EvidenceArchiveTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.run = self.root / "latency-001"
        self.run.mkdir()
        self.output = self.root / "evidence.zip"
        self.manifest = {
            "run_id": "latency-001", "status": "completed",
            "git_commit": "recorded-commit", "slurm_job_id": "123",
        }
        self.set_status("completed")
        for name in STAGES["latency"]:
            (self.run / name).write_bytes(b"recorded evidence\r\n")

    def set_status(self, status):
        self.manifest["status"] = status
        (self.run / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_round_trip_preserves_bytes_and_run_provenance(self):
        (self.run / ".env").write_text("NOT_AN_EVIDENCE_FILE=ignored")
        config = self.root / "configuration.txt"
        config.write_bytes(b"frozen configuration\r\n")
        create_archive(self.run, "latency", self.output, [("configuration", config)])
        report = verify_archive(self.output)
        self.assertTrue(report["verified"])
        self.assertEqual(report["run_git_commit"], "recorded-commit")
        self.assertEqual(report["slurm_job_id"], "123")
        self.assertEqual(report["slurm_job_id_source"], "run_manifest")
        self.assertIn("gpu-samples", report["missing_evidence"])
        self.assertNotIn("configuration", report["missing_evidence"])
        with ZipFile(self.output) as archive:
            self.assertEqual(archive.read("run/evaluations.jsonl"), b"recorded evidence\r\n")
            self.assertEqual(archive.read("attachments/configuration.txt"), config.read_bytes())
            self.assertNotIn("run/.env", archive.namelist())
        self.assertTrue(verify_archive(self.output, report["archive_sha256"])["verified"])
        with self.assertRaisesRegex(ValueError, "ZIP SHA-256"):
            verify_archive(self.output, "0" * 64)

    def test_changed_member_is_detected_even_with_valid_zip_crc(self):
        create_archive(self.run, "latency", self.output)
        tampered = self.root / "tampered.zip"
        with ZipFile(self.output) as source, ZipFile(tampered, "w") as dest:
            for name in source.namelist():
                dest.writestr(name, b"modified" if name == "run/attempts.csv" else source.read(name))
        with self.assertRaisesRegex(ValueError, "Checksum mismatch: run/attempts.csv"):
            verify_archive(tampered)

    def test_unexpected_member_is_rejected(self):
        create_archive(self.run, "latency", self.output)
        with ZipFile(self.output, "a") as archive:
            archive.writestr("../unexpected", b"data")
        with self.assertRaisesRegex(ValueError, "unexpected or duplicate"):
            verify_archive(self.output)

    def test_incomplete_requires_opt_in_and_running_is_always_rejected(self):
        self.set_status("running")
        with self.assertRaisesRegex(ValueError, "Stop the run"):
            create_archive(self.run, "latency", self.output, allow_incomplete=True)
        self.set_status("interrupted")
        with self.assertRaisesRegex(ValueError, "allow-incomplete"):
            create_archive(self.run, "latency", self.output)
        (self.run / "summary.json").unlink()
        create_archive(self.run, "latency", self.output, allow_incomplete=True)
        self.assertEqual(verify_archive(self.output)["run_status"], "interrupted")

    def test_completed_missing_artifact_job_mismatch_and_no_overwrite(self):
        with self.assertRaisesRegex(ValueError, "Slurm job ID conflicts"):
            create_archive(self.run, "latency", self.output, slurm_job_id="999")
        summary = self.run / "summary.json"
        summary.unlink()
        with self.assertRaisesRegex(ValueError, "summary.json"):
            create_archive(self.run, "latency", self.output)
        self.assertFalse(self.output.exists())
        summary.write_text("{}")
        create_archive(self.run, "latency", self.output)
        before = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            create_archive(self.run, "latency", self.output)
        self.assertEqual(self.output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
