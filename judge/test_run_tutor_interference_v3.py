import json
import tempfile
import unittest
from pathlib import Path

from judge.run_tutor_interference_v3 import compare_dirs, digest, load_plan, main, play_turn, usable


def events():
    return [
        {"type": "config_applied", "data": {"llm_model": "llama3_8b_q4"}},
        {"type": "metrics", "data": {"ttft": 0.4, "firstAudio": 1, "llmTotal": 2, "total": 3}},
        {"type": "audio_chunk", "data": "AAA="},
        {"type": "audio_done"},
        {"type": "judge_submission", "data": {
            "turn_id": "tutor-test", "response_status": "completed", "judge_available": False,
            "task_id": None, "enqueue_seconds": None,
        }},
        {"type": "assistant_done"},
    ]


class FakeSocket:
    def __init__(self, messages):
        self.messages = iter(messages)

    def send(self, text):
        self.request = json.loads(text)

    def recv(self):
        return json.dumps(next(self.messages))

    def settimeout(self, _):
        pass


class InterferenceTests(unittest.TestCase):
    def test_turn_uses_real_event_names_and_requires_audio_and_judge_state(self):
        case = {"id": "q", "text": "What is a decision tree?"}
        ws = FakeSocket(events())
        row = play_turn(ws, case, 2)
        self.assertEqual(ws.request["action"], "text_query")
        self.assertTrue(usable(row, "disabled"))
        self.assertFalse(usable(row, "same-gpu"))
        self.assertEqual(row["metrics"]["firstAudio"], 1)
        self.assertIsNotNone(row["first_audio_received_s"])
        row["judge_submission"].update(judge_available=True, enqueue_seconds=0.001, task_id="judge-test")
        self.assertTrue(usable(row, "same-gpu"))
        row["judge_submission"]["task_id"] = None
        self.assertFalse(usable(row, "same-gpu"))

    def test_tts_worker_error_invalidates_even_partial_audio(self):
        messages = events()
        messages.insert(-1, {"type": "tts_worker_error", "message": "RuntimeError"})
        row = play_turn(FakeSocket(messages), {"id": "q", "text": "test"}, 2)
        self.assertFalse(usable(row, "disabled"))
        self.assertEqual(row["tts_worker_errors"], ["RuntimeError"])

    def test_dry_plan_is_independent_of_condition_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            for mode in ("disabled", "same-gpu", "isolated-gpu"):
                args = ["run", "--mode", mode, "--run-id", mode, "--output-root", folder, "--dry-run"]
                self.assertEqual(main(args), 0)
                manifest = json.loads((Path(folder) / mode / "manifest.json").read_text())
                self.assertEqual(manifest["plan_sha256"], digest(load_plan()))
                self.assertFalse((Path(folder) / mode / "turns.jsonl").exists())
            with self.assertRaises(SystemExit):
                main(args)

    def test_comparison_checks_thresholds_and_rejects_wrong_effective_config(self):
        plan = load_plan()
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for mode, total in (("disabled", 3), ("same-gpu", 4), ("isolated-gpu", 3.1)):
                path = Path(folder) / mode
                path.mkdir()
                paths.append(path)
                (path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
                (path / "manifest.json").write_text(json.dumps({
                    "mode": mode, "status": "completed", "git_commit": "test",
                    "plan_sha256": digest(plan),
                }), encoding="utf-8")
                rows = []
                for case in plan["measured"]:
                    row = play_turn(FakeSocket(events()), case, 2)
                    row["phase"] = "measured"
                    row["metrics"]["total"] = total
                    row["judge_submission"]["turn_id"] = case["id"]
                    if mode != "disabled":
                        row["judge_submission"].update(judge_available=True, enqueue_seconds=0.01, task_id="judge-test")
                    rows.append(row)
                (path / "turns.jsonl").write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            report = compare_dirs(paths)
            self.assertTrue(report["conditions"]["isolated-gpu"]["median_total_within_5pct"])
            self.assertFalse(report["conditions"]["same-gpu"]["median_total_within_5pct"])
            self.assertFalse(report["deployment_gate_evaluated"])
            rows[-1]["effective_config"]["llm_model"] = "wrong"
            (path / "turns.jsonl").write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "effective configuration changed"):
                compare_dirs(paths)


if __name__ == "__main__":
    unittest.main()
