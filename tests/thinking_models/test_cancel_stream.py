import json
import threading
import unittest
from unittest.mock import patch
from llm.llmcli.providers.thinking_http import ThinkingHTTPProvider


class CancelResponse:
    def __init__(self, cancel): self.cancel = cancel; self.closed = False
    def __enter__(self): return self
    def __exit__(self, *args): self.closed = True
    def raise_for_status(self): pass
    def iter_lines(self, decode_unicode=False):
        yield b'data: {"choices":[{"delta":{"reasoning_content":"hidden"}}]}'
        self.cancel.set()
        yield b'data: {"choices":[{"delta":{"content":"must not be sent"}}]}'


class ThinkingCancellationTest(unittest.TestCase):
    def test_cancel_during_reasoning_stops_before_answer_and_closes_response(self):
        cancel = threading.Event(); response = CancelResponse(cancel)
        with patch('llm.llmcli.providers.thinking_http.requests.post', return_value=response):
            result = list(ThinkingHTTPProvider('http://test').stream_with_cancel('question', cancel))
        self.assertEqual(result, [])
        self.assertTrue(response.closed)

    def test_already_cancelled_skips_request(self):
        cancel = threading.Event(); cancel.set()
        with patch('llm.llmcli.providers.thinking_http.requests.post') as post:
            self.assertEqual(list(ThinkingHTTPProvider('http://test').stream_with_cancel('q', cancel)), [])
            post.assert_not_called()
