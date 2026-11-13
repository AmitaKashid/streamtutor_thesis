"""Verify reported server timing and current-turn history selection."""
import ast
import json
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from app.inference_metrics import server_timing_metrics
from app.thinking_stream import ThinkingStreamError
from llm.llmcli.providers.thinking_http import ThinkingHTTPProvider


class Response:
    def __init__(self, finish='stop', timings=None, done=True):
        self.finish, self.timings, self.done = finish, timings, done

    def __enter__(self): return self
    def __exit__(self, *args): pass
    def raise_for_status(self): pass

    def iter_lines(self, decode_unicode=False):
        yield b'data: {"choices":[{"delta":{"content":"A valid answer."},"finish_reason":null}]}'
        event = {'choices': [{'delta': {}, 'finish_reason': self.finish}]}
        if self.timings is not None:
            event['timings'] = self.timings
        yield ('data: ' + json.dumps(event)).encode('utf-8')
        if self.done:
            yield b'data: [DONE]'


class ServerTimingTests(unittest.TestCase):
    def test_final_counters_include_cached_tokens_and_convert_ms(self):
        stats = {'cache_n': 100, 'prompt_n': 300, 'prompt_ms': 250,
                 'predicted_n': 50, 'predicted_ms': 2000, 'predicted_per_second': 24.5}
        recorded = []
        with patch('llm.llmcli.providers.thinking_http.requests.post', return_value=Response(timings=stats)):
            result = ''.join(ThinkingHTTPProvider('http://test').stream_with_cancel(
                'question', None, thinking_enabled=False, on_server_metrics=recorded.append))
        self.assertEqual(result, 'A valid answer.')
        self.assertEqual(recorded, [{'serverPromptTokens': 400, 'serverCachedTokens': 100,
            'serverProcessedPromptTokens': 300, 'serverPrefillSeconds': 0.25,
            'serverOutputTokens': 50, 'serverDecodeSeconds': 2.0,
            'serverDecodeTokensPerSecond': 24.5}])

    def test_missing_or_invalid_data_is_not_reported_as_zero(self):
        self.assertEqual(server_timing_metrics({'cache_n': 2, 'prompt_n': -1,
            'prompt_ms': float('nan'), 'predicted_ms': False}), {})
        self.assertEqual(server_timing_metrics(None), {})

    def test_failed_or_incomplete_generation_does_not_confirm_metrics(self):
        timings = {'cache_n': 0, 'prompt_n': 20, 'prompt_ms': 10}
        for response in [Response(finish='length', timings=timings), Response(timings=timings, done=False)]:
            recorded = []
            with patch('llm.llmcli.providers.thinking_http.requests.post', return_value=response):
                with self.assertRaises(ThinkingStreamError):
                    list(ThinkingHTTPProvider('http://test').stream_with_cancel(
                        'q', None, thinking_enabled=False, on_server_metrics=recorded.append))
            self.assertEqual(recorded, [])


def load_memory_class():
    path = Path(__file__).resolve().parents[2] / 'app/mic_stt_llm_tts.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ConversationMemory')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
    env = {'threading': threading}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), env)
    return env['ConversationMemory']


class ConversationLimitTests(unittest.TestCase):
    def test_lower_limit_is_applied_before_another_turn_is_added(self):
        memory = load_memory_class()()
        for i in range(4): memory.add_turn(f'question {i}', f'answer {i}')
        memory.max_turns = 2
        self.assertEqual(memory.stats()['turns'], 2)
        history = memory.format_history()
        self.assertNotIn('question 1', history)
        self.assertIn('question 2', history)
        self.assertIn('question 3', history)
        self.assertEqual(len(memory.snapshot()), 2)

    def test_new_turn_obeys_limit_and_unlimited_is_supported(self):
        memory = load_memory_class()(max_turns=2)
        for i in range(3): memory.add_turn(f'q{i}', f'a{i}')
        self.assertEqual([t['user'] for t in memory.snapshot()], ['q1', 'q2'])
        memory.max_turns = None
        memory.add_turn('q3', 'a3')
        self.assertEqual([t['user'] for t in memory.snapshot()], ['q1', 'q2', 'q3'])

class ProbeTests(unittest.TestCase):
    def test_synthetic_probe_uses_the_same_history_format_as_the_app(self):
        from scripts.context_diagnostics.probe import make_prompt, CONTEXT, TURNS, QUESTION
        from prompts.registry import compose_prompt
        memory = load_memory_class()(max_turns=3)
        for q, a in TURNS[:3]: memory.add_turn(q, a)
        expected = compose_prompt(style='socratic', user_query=memory.build_contextual_user_query(QUESTION), context=CONTEXT)
        self.assertEqual(make_prompt(TURNS[:3]), expected)

    def test_probe_reports_incomplete_results_instead_of_hiding_them(self):
        from scripts.context_diagnostics.probe import request
        stats = {'cache_n': 20, 'prompt_n': 80, 'prompt_ms': 50, 'predicted_n': 10, 'predicted_ms': 100}
        with patch('scripts.context_diagnostics.probe.requests.post', return_value=Response(finish='length', timings=stats)) as post:
            result = request('http://test', 'question', cache_prompt=False)
        self.assertEqual(result['finish_reason'], 'length')
        self.assertTrue(result['stream_done'])
        self.assertEqual(result['serverPromptTokens'], 100)
        self.assertFalse(post.call_args.kwargs['json']['cache_prompt'])

class PipelineMetricFlowTests(unittest.TestCase):
    def test_server_metrics_reach_the_frontend_event_without_guessing(self):
        import time
        import types
        from unittest.mock import Mock
        from test_pipeline_failure import load_pipeline

        class Provider:
            def stream_with_cancel(self, prompt, cancel_event, **kwargs):
                yield 'A complete answer.'
                kwargs['on_server_metrics']({'serverPromptTokens': 120, 'serverPrefillSeconds': 0.3})

        playback = Mock(); playback.interrupted = threading.Event()
        args = types.SimpleNamespace(min_words=1, tts_enabled=False, mode='stream',
            teaching_style='default', chunk_first=80, chunk_mid=100, chunk_later=200,
            chunk_wait=1, browser_audio=True, voice_prompt=None,
            llm_model='qwen3_5_9b_q6', thinking_enabled=False)
        events = []; active = threading.Event(); active.set()
        timer = types.SimpleNamespace(elapsed=lambda: 3.0, t0=time.perf_counter())
        load_pipeline()(transcript='test question', stt_time=0, total_timer=timer,
            llm=Provider(), tts_engine=None, retriever=None, args=args, playback=playback,
            pipeline_active=active, judge=None, event_cb=events.append)
        self.assertTrue(any(e.get('data', {}).get('serverPromptTokens') == 120
                            for e in events if e.get('type') == 'metrics'))


if __name__ == '__main__': unittest.main()
