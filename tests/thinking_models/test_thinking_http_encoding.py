import json
import unittest
from unittest.mock import patch
from llm.llmcli.providers.thinking_http import ThinkingHTTPProvider


class Response:
    def __init__(self): self.encoding = 'ISO-8859-1'
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def raise_for_status(self): pass
    def iter_lines(self, decode_unicode=False):
        assert decode_unicode is False
        events = [
            {'choices': [{'delta': {'reasoning_content': 'hidden'}, 'finish_reason': None}]},
            {'choices': [{'delta': {'content': 'Example 😊'}, 'finish_reason': None}]},
            {'choices': [{'delta': {}, 'finish_reason': 'stop'}]},
        ]
        for event in events:
            yield ('data: ' + json.dumps(event, ensure_ascii=False)).encode('utf-8')
        yield b'data: [DONE]'


class Utf8StreamTest(unittest.TestCase):
    def test_unicode_is_decoded_as_utf8(self):
        with patch('llm.llmcli.providers.thinking_http.requests.post', return_value=Response()):
            answer = ''.join(ThinkingHTTPProvider('http://example.invalid').stream('question'))
        self.assertEqual(answer, 'Example 😊')
