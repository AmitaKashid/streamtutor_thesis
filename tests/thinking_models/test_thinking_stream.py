import unittest
from app.thinking_stream import ThinkingStream, ThinkingStreamError


def event(delta=None, finish=None):
    return {"choices": [{"delta": delta or {}, "finish_reason": finish}]}


class ThinkingStreamTests(unittest.TestCase):
    def test_reasoning_is_suppressed_and_final_answer_preserved(self):
        s = ThinkingStream()
        self.assertEqual(s.feed(event({"reasoning_content": "private"})), "")
        text = s.feed(event({"content": "Clustering groups similar examples."}))
        text += s.feed(event(finish="stop"))
        s.end()
        self.assertEqual(text, "Clustering groups similar examples.")

    def test_unverified_content_is_rejected(self):
        with self.assertRaises(ThinkingStreamError):
            ThinkingStream().feed(event({"content": "unverified"}))

    def test_split_reasoning_marker_is_rejected(self):
        s = ThinkingStream()
        s.feed(event({"reasoning_content": "reasoning"}))
        s.feed(event({"content": "<thi"}))
        with self.assertRaises(ThinkingStreamError):
            s.feed(event({"content": "nk>"}))

    def test_truncation_and_missing_completion_are_rejected(self):
        s = ThinkingStream()
        s.feed(event({"reasoning_content": "reasoning"}))
        with self.assertRaises(ThinkingStreamError):
            s.feed(event(finish="length"))
        with self.assertRaises(ThinkingStreamError):
            ThinkingStream().end()

if __name__ == "__main__":
    unittest.main()

class ModelRetryTests(unittest.TestCase):
    def test_retry_once_if_reasoning_not_observed(self):
        from unittest.mock import patch
        from app.thinking_stream import ReasoningNotObserved
        from llm.llmcli.providers.thinking_http import ThinkingHTTPProvider
        provider = ThinkingHTTPProvider('http://example.invalid')
        with patch.object(provider, '_stream_once', side_effect=[ReasoningNotObserved('missing'), iter(['answer'])]) as call:
            self.assertEqual(''.join(provider.stream('question')), 'answer')
            self.assertEqual(call.call_count, 2)

    def test_retry_does_not_swallow_other_errors(self):
        from unittest.mock import patch
        from llm.llmcli.providers.thinking_http import ThinkingHTTPProvider
        provider = ThinkingHTTPProvider('http://example.invalid')
        with patch.object(provider, '_stream_once', side_effect=ThinkingStreamError('truncated')) as call:
            with self.assertRaises(ThinkingStreamError):
                list(provider.stream('question'))
            self.assertEqual(call.call_count, 1)
