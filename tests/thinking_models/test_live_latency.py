"""Verify live timing arrives before stream completion and stays provisional."""
import json
import threading
import unittest
from unittest.mock import patch
from app.thinking_stream import ThinkingStreamError
from llm.llmcli.providers.thinking_http import ThinkingHTTPProvider

class Response:
    def __init__(self, done=True): self.done = done
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def raise_for_status(self): pass
    def iter_lines(self, decode_unicode=False):
        for delta in ({'reasoning_content':'thought'}, {'content':'A sufficiently long final answer to stream.'}):
            yield ('data: ' + json.dumps({'choices':[{'delta':delta}]})).encode()
        if self.done:
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'
            yield b'data: [DONE]'

class LiveTimingTests(unittest.TestCase):
    def test_preview_arrives_before_completion_and_final_is_confirmed(self):
        previews=[]; confirmed=[]
        with patch('llm.llmcli.providers.thinking_http.requests.post', return_value=Response()), \
             patch('llm.llmcli.providers.thinking_http.time.perf_counter', side_effect=[100,102,109,110]):
            stream=ThinkingHTTPProvider('http://test').stream_with_cancel('question', threading.Event(), on_thinking_preview=previews.append, on_thinking_time=confirmed.append)
            first=next(stream)
            self.assertTrue(first)
            self.assertEqual(previews,[7])
            self.assertEqual(confirmed,[])
            list(stream)
        self.assertEqual(confirmed,[7])

    def test_incomplete_stream_does_not_confirm_preview(self):
        previews=[]; confirmed=[]
        with patch('llm.llmcli.providers.thinking_http.requests.post', return_value=Response(done=False)):
            stream=ThinkingHTTPProvider('http://test').stream_with_cancel('question', None, on_thinking_preview=previews.append, on_thinking_time=confirmed.append)
            with self.assertRaises(ThinkingStreamError): list(stream)
        self.assertEqual(len(previews),1)
        self.assertEqual(confirmed,[])

    def test_thinking_disabled_never_emits_preview(self):
        previews=[]
        class Direct(Response):
            def iter_lines(self, decode_unicode=False):
                yield b'data: {"choices":[{"delta":{"content":"answer"}}]}'
                yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'
                yield b'data: [DONE]'
        with patch('llm.llmcli.providers.thinking_http.requests.post', return_value=Direct()):
            list(ThinkingHTTPProvider('http://test').stream_with_cancel('question', None, thinking_enabled=False, on_thinking_preview=previews.append))
        self.assertEqual(previews,[])

class PipelineLiveTimingTests(unittest.TestCase):
    def test_preview_and_answer_progress_reach_frontend_before_completion(self):
        import time
        import types
        from unittest.mock import Mock
        from test_pipeline_failure import load_pipeline
        class Provider:
            def stream_with_cancel(self, prompt, cancel_event, **kwargs):
                kwargs['on_thinking_preview'](2.5)
                yield 'A complete streamed answer.'
                kwargs['on_thinking_time'](2.5)
        playback = Mock(); playback.interrupted = threading.Event()
        args = types.SimpleNamespace(min_words=1, tts_enabled=False, mode='stream',
            teaching_style='default', chunk_first=80, chunk_mid=100, chunk_later=200,
            chunk_wait=1, browser_audio=True, voice_prompt=None,
            llm_model='qwen3_5_9b_q6', thinking_enabled=True)
        events=[]; active=threading.Event(); active.set()
        timer=types.SimpleNamespace(elapsed=lambda: 3.0, t0=time.perf_counter())
        load_pipeline()(transcript='test question', stt_time=0, total_timer=timer,
            llm=Provider(), tts_engine=None, retriever=None, args=args,
            playback=playback, pipeline_active=active, judge=None, event_cb=events.append)
        types_seen=[e['type'] for e in events]
        self.assertLess(types_seen.index('thinking_preview'), types_seen.index('llm_chunk'))
        self.assertLess(types_seen.index('llm_answer_progress'), types_seen.index('text_complete'))
        final=next(e['data'] for e in events if e['type']=='metrics' and 'llmAnswerTime' in e['data'])
        self.assertEqual(final['thinkingTime'],2.5)
        self.assertGreaterEqual(final['llmAnswerTime'],0)

if __name__ == '__main__': unittest.main()
