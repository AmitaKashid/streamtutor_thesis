from __future__ import annotations

import unittest

from judge.scalar_contract_v2 import (
    build_scalar_retry_prompt,
    parse_scalar_output,
)


class ScalarContractV2Tests(unittest.TestCase):
    def test_accepts_valid_single_line_output(self):
        parsed, error = parse_scalar_output(
            "Feedback: The answer is correct and clear. [RESULT] 5"
        )

        self.assertIsNone(error)
        self.assertEqual(parsed["protocol_version"], "2.1")
        self.assertEqual(parsed["score"], 5)
        self.assertEqual(parsed["format_variant"], "prefixed_feedback")
        self.assertEqual(
            parsed["feedback"],
            "The answer is correct and clear.",
        )

    def test_accepts_valid_multiline_feedback(self):
        parsed, error = parse_scalar_output(
            "Feedback: The definition is correct.\n"
            "The example is also relevant.\n"
            "[RESULT] 4"
        )

        self.assertIsNone(error)
        self.assertEqual(parsed["score"], 4)
        self.assertIn("example is also relevant", parsed["feedback"])

    def test_accepts_boundary_scores(self):
        low, low_error = parse_scalar_output(
            "Feedback: Fundamentally incorrect. [RESULT] 1"
        )
        high, high_error = parse_scalar_output(
            "Feedback: Fully correct and precise. [RESULT] 5"
        )

        self.assertIsNone(low_error)
        self.assertIsNone(high_error)
        self.assertEqual(low["score"], 1)
        self.assertEqual(high["score"], 5)

    def test_rejects_empty_output(self):
        parsed, error = parse_scalar_output("   ")

        self.assertIsNone(parsed)
        self.assertEqual(error, "empty_output")

    def test_accepts_unprefixed_feedback(self):
        parsed, error = parse_scalar_output(
            "The answer is correct. [RESULT] 4"
        )

        self.assertIsNone(error)
        self.assertEqual(parsed["feedback"], "The answer is correct.")
        self.assertEqual(parsed["score"], 4)
        self.assertEqual(
            parsed["format_variant"],
            "unprefixed_feedback",
        )

    def test_accepts_observed_prometheus_serialization(self):
        parsed, error = parse_scalar_output(
            "The explanation is evidence based. Therefore, the score is 3. "
            "[RESULT] 3"
        )

        self.assertIsNone(error)
        self.assertEqual(parsed["score"], 3)
        self.assertEqual(
            parsed["format_variant"],
            "unprefixed_feedback",
        )

    def test_rejects_empty_feedback(self):
        parsed, error = parse_scalar_output(
            "Feedback:   [RESULT] 4"
        )

        self.assertIsNone(parsed)
        self.assertEqual(error, "empty_feedback")

    def test_rejects_missing_result_marker(self):
        parsed, error = parse_scalar_output(
            "Feedback: The answer is correct."
        )

        self.assertIsNone(parsed)
        self.assertIn("found 0", error)

    def test_rejects_duplicate_result_markers(self):
        parsed, error = parse_scalar_output(
            "Feedback: First [RESULT] 3 second [RESULT] 4"
        )

        self.assertIsNone(parsed)
        self.assertIn("found 2", error)

    def test_rejects_out_of_range_score(self):
        parsed, error = parse_scalar_output(
            "Feedback: Incorrect rating. [RESULT] 6"
        )

        self.assertIsNone(parsed)
        self.assertIn("expected non-empty feedback", error)

    def test_rejects_zero_score(self):
        parsed, error = parse_scalar_output(
            "Feedback: Incorrect rating. [RESULT] 0"
        )

        self.assertIsNone(parsed)
        self.assertIn("expected non-empty feedback", error)

    def test_rejects_decimal_score(self):
        parsed, error = parse_scalar_output(
            "Feedback: Mostly correct. [RESULT] 4.0"
        )

        self.assertIsNone(parsed)
        self.assertIn("expected non-empty feedback", error)

    def test_rejects_text_after_result(self):
        parsed, error = parse_scalar_output(
            "Feedback: Correct. [RESULT] 4 Additional explanation."
        )

        self.assertIsNone(parsed)
        self.assertIn("expected non-empty feedback", error)

    def test_rejects_markdown_fence(self):
        parsed, error = parse_scalar_output(
            "```\nFeedback: Correct. [RESULT] 4\n```"
        )

        self.assertIsNone(parsed)
        self.assertIn("Markdown fences", error)

    def test_retry_prompt_retains_error_without_expected_score(self):
        prompt = build_scalar_retry_prompt(
            "Evaluate this answer.",
            "result_marker_count_error: expected 1 marker, found 0",
        )

        self.assertIn("Evaluate this answer.", prompt)
        self.assertIn("RETRY_NOTICE", prompt)
        self.assertIn("found 0", prompt)
        self.assertNotIn("correct score is", prompt.lower())
        self.assertTrue(
            prompt.rstrip().endswith(
                "result_marker_count_error: "
                "expected 1 marker, found 0"
            )
        )

    def test_retry_prompt_rejects_empty_inputs(self):
        with self.assertRaises(ValueError):
            build_scalar_retry_prompt(
                "",
                "format error",
            )

        with self.assertRaises(ValueError):
            build_scalar_retry_prompt(
                "Evaluate.",
                "",
            )


if __name__ == "__main__":
    unittest.main()