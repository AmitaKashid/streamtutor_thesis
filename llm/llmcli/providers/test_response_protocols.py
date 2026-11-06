from __future__ import annotations

import unittest

from llm.llmcli.providers.response_protocols import (
    ResponseProtocolError,
    decode_harmony_final,
    decode_response_protocol,
)


class ResponseProtocolTests(unittest.TestCase):
    def test_no_protocol_preserves_plain_output(self):
        self.assertEqual(
            decode_response_protocol('  {"ok": true}  ', None),
            '{"ok": true}',
        )

    def test_observed_fare_output_is_decoded(self):
        raw = '<|channel|>final<|message|>{"judge_ready": true}'
        self.assertEqual(
            decode_harmony_final(raw),
            '{"judge_ready": true}',
        )

    def test_official_analysis_to_final_transition_is_decoded(self):
        raw = (
            '<|channel|>analysis<|message|>Internal evaluation.'
            '<|end|><|start|>assistant'
            '<|channel|>final<|message|>{"judge_ready": true}'
        )
        self.assertEqual(
            decode_harmony_final(raw),
            '{"judge_ready": true}',
        )

    def test_terminal_return_token_is_removed(self):
        raw = (
            '<|channel|>final<|message|>{"judge_ready": true}'
            '<|return|>'
        )
        self.assertEqual(
            decode_harmony_final(raw),
            '{"judge_ready": true}',
        )

    def test_missing_final_marker_is_rejected(self):
        with self.assertRaises(ResponseProtocolError):
            decode_harmony_final('{"judge_ready": true}')

    def test_duplicate_final_markers_are_rejected(self):
        marker = '<|channel|>final<|message|>'
        with self.assertRaises(ResponseProtocolError):
            decode_harmony_final(marker + "one" + marker + "two")

    def test_arbitrary_prefix_is_rejected(self):
        raw = 'explanation<|channel|>final<|message|>{"ok": true}'
        with self.assertRaises(ResponseProtocolError):
            decode_harmony_final(raw)

    def test_residual_control_token_is_rejected(self):
        raw = '<|channel|>final<|message|>value<|channel|>analysis'
        with self.assertRaises(ResponseProtocolError):
            decode_harmony_final(raw)


if __name__ == "__main__":
    unittest.main()
