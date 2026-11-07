from __future__ import annotations

import unittest

from judge.pairwise_contract_v3 import (
    build_pairwise_retry_prompt,
    parse_pairwise_output,
)


class PairwiseContractV3Tests(unittest.TestCase):
    def test_accepts_both_choices_and_feedback_variants(self) -> None:
        cases = (
            ("Feedback: Candidate 1 is more accurate.\n[RESULT] 1", 1),
            ("Candidate 2 is better grounded. [RESULT] 2", 2),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                parsed, error = parse_pairwise_output(text)
                self.assertIsNone(error)
                self.assertEqual(parsed["choice"], expected)

    def test_rejects_non_contract_outputs(self) -> None:
        cases = (
            "",
            "Feedback: tied\n[RESULT] tie",
            "Feedback: first\n[RESULT] A",
            "Feedback: first\n[RESULT] 1.0",
            "Feedback: first\n[RESULT] 1 trailing",
            "Feedback: first\n[RESULT] 1\n[RESULT] 2",
            "```\nFeedback: first\n[RESULT] 1\n```",
            "[RESULT] 1",
        )
        for text in cases:
            with self.subTest(text=text):
                parsed, error = parse_pairwise_output(text)
                self.assertIsNone(parsed)
                self.assertIsNotNone(error)

    def test_retry_is_format_only_and_contains_no_answer_hint(self) -> None:
        retry = build_pairwise_retry_prompt(
            "Compare candidates without revealing labels.",
            "missing marker",
        )
        self.assertIn("[RESULT] <1 or 2>", retry)
        self.assertNotIn("correct answer", retry.lower())
        self.assertNotIn("expected choice", retry.lower())


if __name__ == "__main__":
    unittest.main()
