import json
import threading
import unittest
from unittest.mock import patch
from app.thinking_stream import ThinkingStreamError
from llm.llmcli.providers.thinking_http import ThinkingHTTPProvider


class Response:
    def __init__(self, events): self.events = events
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def raise_for_status(self): pass
    def iter_lines(self, decode_unicode=False):
        for delta, finish in self.events:
            yield ('data: ' + json.dumps({'choices': [{'delta': delta, 'finish_reason': finish}]})).encode()
        yield b'data: [DONE]'


class ThinkingToggleTests(unittest.TestCase):
    def test_on_streams_reasoning_separately_from_answer(self):
        response = Response([({'reasoning_content': 'reasoning text'}, None), ({'content': 'final answer'}, None), ({}, 'stop')])
        shown = []
        with patch('llm.llmcli.providers.thinking_http.requests.post', return_value=response) as post:
            answer = ''.join(ThinkingHTTPProvider('http://test').stream_with_cancel('q', threading.Event(), thinking_enabled=True, on_reasoning=shown.append))
        self.assertEqual(answer, 'final answer')
        self.assertEqual(shown, ['reasoning text'])
        self.assertTrue(post.call_args.kwargs['json']['chat_template_kwargs']['enable_thinking'])

    def test_off_requests_direct_answer_without_reasoning(self):
        response = Response([({'content': 'final answer'}, None), ({}, 'stop')])
        shown = []
        with patch('llm.llmcli.providers.thinking_http.requests.post', return_value=response) as post:
            answer = ''.join(ThinkingHTTPProvider('http://test').stream_with_cancel('q', None, thinking_enabled=False, on_reasoning=shown.append))
        self.assertEqual(answer, 'final answer')
        self.assertEqual(shown, [])
        self.assertFalse(post.call_args.kwargs['json']['chat_template_kwargs']['enable_thinking'])

    def test_off_rejects_unexpected_reasoning(self):
        response = Response([({'reasoning_content': 'unexpected'}, None)])
        shown = []
        with patch('llm.llmcli.providers.thinking_http.requests.post', return_value=response):
            with self.assertRaises(ThinkingStreamError):
                list(ThinkingHTTPProvider('http://test').stream_with_cancel('q', None, thinking_enabled=False, on_reasoning=shown.append))
        self.assertEqual(shown, [])

if __name__ == '__main__': unittest.main()
