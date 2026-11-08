"""Exercise resume and rejection paths without a model, GPU, or network."""

import copy
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from judge import run_judge_benchmark_v3 as runner
from judge import run_controlled_contrasts_v3 as pairwise


class FakeProvider:
    def __init__(self, stop_after=None, score=4):
        self.calls = 0
        self.stop_after = stop_after
        self.score = score

    def healthcheck(self):
        return True

    def generate(self, _prompt):
        if self.calls == self.stop_after:
            raise KeyboardInterrupt
        self.calls += 1
        return SimpleNamespace(text=f"Feedback: Clear explanation.\n[RESULT] {self.score}")


class ResumeSafetyTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name) / "resume-test"
        self.predictions = self.directory / "predictions.jsonl"
        self.manifest = self.directory / "manifest.json"
        self.args = [
            "--split", "calibration", "--max-responses", "1",
            "--base-url", "http://127.0.0.1:12345",
            "--model-id", "mock-model", "--quantization", "test",
            "--hardware", "test", "--output-root", temp.name,
            "--run-id", self.directory.name,
        ]

    def invoke(self, provider, *extra, commit="test-commit"):
        with (
            redirect_stdout(StringIO()),
            patch.object(runner, "_make_remote_provider", return_value=provider) as factory,
            patch.object(runner, "_git_commit", return_value=commit),
        ):
            code = runner.main([*self.args, *extra])
        return code, factory

    def interrupt(self):
        code, _ = self.invoke(FakeProvider(stop_after=1))
        self.assertEqual(code, 130)
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "interrupted")
        self.assertEqual(manifest["completed_evaluation_count"], 1)
        self.assertFalse((self.directory / "summary.json").exists())
        return manifest, runner._read_jsonl(self.predictions)[0]

    def test_interrupt_and_resume_does_not_repeat_saved_work(self):
        manifest, first = self.interrupt()
        provider = FakeProvider()
        self.assertEqual(self.invoke(provider, "--resume")[0], 0)
        rows = runner._read_jsonl(self.predictions)
        self.assertEqual(rows[0], first)
        self.assertEqual(provider.calls, manifest["planned_evaluation_count"] - 1)
        self.assertEqual(len({row["evaluation_id"] for row in rows}), len(rows))
        self.assertTrue(json.loads((self.directory / "summary.json").read_text())["complete"])

    def test_resume_refuses_changed_sampling_or_source_revision(self):
        self.interrupt()
        before = self.manifest.read_bytes()
        for extra, commit in ((("--temperature", "0.7"), "test-commit"), ((), "different")):
            with self.subTest(extra=extra, commit=commit):
                provider = FakeProvider()
                with self.assertRaisesRegex(ValueError, "does not match"):
                    self.invoke(provider, "--resume", *extra, commit=commit)
                self.assertEqual(provider.calls, 0)
                self.assertEqual(self.manifest.read_bytes(), before)

    def test_foreign_duplicate_modified_and_partial_rows_are_rejected_before_http(self):
        _, first = self.interrupt()
        before = self.manifest.read_bytes()
        foreign = {**first, "evaluation_id": "outside-plan"}
        modified = copy.deepcopy(first)
        modified["inputs"]["answer"] = "replaced answer"
        wrong_config = copy.deepcopy(first)
        wrong_config["judge_configuration"]["model_id"] = "different"
        for content, error in (
            (json.dumps(foreign), "outside the frozen task plan"),
            (json.dumps(first) + "\n" + json.dumps(first), "Duplicate evaluation ID"),
            (json.dumps(modified), "inputs or case metadata differ"),
            (json.dumps(wrong_config), "mismatched judge_configuration"),
            ('{"evaluation_id":', "Invalid JSONL"),
        ):
            with self.subTest(error=error):
                self.predictions.write_text(content + "\n", encoding="utf-8")
                with (
                    patch.object(runner, "_make_remote_provider") as factory,
                    patch.object(runner, "_git_commit", return_value="test-commit"),
                    self.assertRaisesRegex(ValueError, error),
                ):
                    runner.main([*self.args, "--resume"])
                factory.assert_not_called()
                self.assertEqual(self.manifest.read_bytes(), before)
                self.assertFalse((self.directory / "summary.json").exists())

    def test_pairwise_resume_rejects_modified_input_hash(self):
        args = self.args[4:]
        provider = FakeProvider(stop_after=1, score=1)
        with (
            redirect_stdout(StringIO()),
            patch.object(pairwise, "_make_remote_provider", return_value=provider) as factory,
            patch.object(pairwise, "_git_commit", return_value="test-commit"),
        ):
            self.assertEqual(pairwise.main(args), 130)
            before = self.manifest.read_bytes()
            path = self.directory / "pairwise_predictions.jsonl"
            first, = runner._read_jsonl(path)
            path.write_text(json.dumps({**first, "input_sha256": "bad"}) + "\n")
            factory.reset_mock()
            with self.assertRaisesRegex(ValueError, "mismatched input_sha256"):
                pairwise.main([*args, "--resume"])
            factory.assert_not_called()
            self.assertEqual(self.manifest.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
