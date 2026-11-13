"""Thinking time uses observed reasoning and answer events from one request."""
import json
import threading
import unittest
from unittest.mock import patch
from llm.llmcli.providers.thinking_http import ThinkingHTTPProvider
from app.thinking_stream import ThinkingStreamError


class Response:
    def __init__(self, events, done=True, cancel=None):
        self.events, self.done, self.cancel = events, done, cancel
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def raise_for_status(self): pass
    def iter_lines(self, decode_unicode=False):
        for i, delta in enumerate(self.events):
            if self.cancel and i == 1:
                self.cancel.set()
            yield ('data: ' + json.dumps({'choices': [{'delta': delta, 'finish_reason': None}]})).encode()
        if self.done:
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'
            yield b'data: [DONE]'


class ThinkingTimingTests(unittest.TestCase):
    def test_timing_is_first_reasoning_to_first_answer(self):
        response = Response([{'reasoning_content': 'first'}, {'reasoning_content': 'more'}, {'content': 'Final answer'}])
        samples = []
        # Request starts at 100; first reasoning at 102; first answer at 109.
        with patch('llm.llmcli.providers.thinking_http.requests.post', return_value=response), \
             patch('llm.llmcli.providers.thinking_http.time.perf_counter', side_effect=[100, 102, 109, 110]):
            answer = ''.join(ThinkingHTTPProvider('http://test').stream_with_cancel('q', None, on_thinking_time=samples.append))
        self.assertEqual(answer, 'Final answer')
        self.assertEqual(samples, [7])

    def test_disabled_thinking_has_no_timing(self):
        samples = []
        with patch('llm.llmcli.providers.thinking_http.requests.post', return_value=Response([{'content': 'answer'}])):
            list(ThinkingHTTPProvider('http://test').stream_with_cancel('q', None, thinking_enabled=False, on_thinking_time=samples.append))
        self.assertEqual(samples, [])

    def test_incomplete_stream_has_no_timing(self):
        samples = []
        response = Response([{'reasoning_content': 'reasoning'}, {'content': 'answer'}], done=False)
        with patch('llm.llmcli.providers.thinking_http.requests.post', return_value=response):
            with self.assertRaises(ThinkingStreamError):
                list(ThinkingHTTPProvider('http://test').stream_with_cancel('q', None, on_thinking_time=samples.append))
        self.assertEqual(samples, [])

    def test_cancelled_stream_has_no_timing(self):
        samples, cancel = [], threading.Event()
        response = Response([{'reasoning_content': 'reasoning'}, {'content': 'answer'}], cancel=cancel)
        with patch('llm.llmcli.providers.thinking_http.requests.post', return_value=response):
            list(ThinkingHTTPProvider('http://test').stream_with_cancel('q', cancel, on_thinking_time=samples.append))
        self.assertEqual(samples, [])

class PipelineThinkingTimingTests(unittest.TestCase):
    def test_completed_turn_sends_model_and_thinking_time(self):
        import time
        import types
        from unittest.mock import Mock
        from test_pipeline_failure import load_pipeline
        class Provider:
            def stream_with_cancel(self, prompt, cancel_event, **kwargs):
                kwargs['on_reasoning']('reasoning')
                kwargs['on_thinking_time'](2.5)
                yield 'A complete answer.'
        playback = Mock(); playback.interrupted = threading.Event()
        args = types.SimpleNamespace(min_words=1, tts_enabled=False, mode='stream',
            teaching_style='default', chunk_first=80, chunk_mid=100, chunk_later=200,
            chunk_wait=1, browser_audio=True, voice_prompt=None,
            llm_model='qwen3_5_9b_q6', thinking_enabled=True)
        events = []; active = threading.Event(); active.set()
        timer = types.SimpleNamespace(elapsed=lambda: 3.0, t0=time.perf_counter())
        load_pipeline()(transcript='test question', stt_time=0, total_timer=timer,
            llm=Provider(), tts_engine=None, retriever=None, args=args,
            playback=playback, pipeline_active=active, judge=None, event_cb=events.append)
        started = next(e for e in events if e['type'] == 'tutor_turn_started')
        self.assertEqual(started['model'], 'qwen3_5_9b_q6')
        final = next(e['data'] for e in events if e['type'] == 'metrics' and 'thinkingTime' in e['data'])
        self.assertEqual(final['thinkingTime'], 2.5)
        reasoning = next(e for e in events if e['type'] == 'thinking_chunk')
        self.assertEqual(reasoning['text'], 'reasoning')
        answer = ''.join(e.get('text', '') for e in events if e['type'] == 'llm_chunk')
        self.assertEqual(answer, 'A complete answer.')

if __name__ == '__main__': unittest.main()
